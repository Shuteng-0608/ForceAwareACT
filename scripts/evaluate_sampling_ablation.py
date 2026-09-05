#!/usr/bin/env python3
"""Evaluate all four sampling-ablation checkpoints on one paired data split."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from force_aware_act.act_aligned_training import (  # noqa: E402
    ACTAlignedHighRateHDF5Dataset,
    NormalizationStats,
    collate_high_rate_samples,
)
from force_aware_act.experiments import (  # noqa: E402
    load_paired_episode_subset_manifest,
    validate_checkpoint_experiment_provenance,
)
from force_aware_act.inference import (  # noqa: E402
    ACT_ALIGNED_HIGH_RATE_ROLLOUT_KIND,
    OFFICIAL_ACT_ROLLOUT_KIND,
    OFFICIAL_TEMPORAL_AGGREGATION_DECAY,
    RolloutPolicyAdapter,
)
from scripts.evaluate_paired50_holdout import (  # noqa: E402
    HORIZON_INDICES,
    MetricSums,
    _denormalize,
    _load_checkpoint,
    _normalize,
    temporal_ensemble_l1,
)


EVALUATION_VERSION = "scripted200_sampling_ablation_all_timesteps_v1"
PRIMARY_METRIC = "action_chunk_l1_physical"
POLICY_ORDER = (
    "official_episodic",
    "official_dense",
    "contact_dense",
    "contact_episodic",
)
EXPECTED_CONTRACTS = {
    "official_episodic": (OFFICIAL_ACT_ROLLOUT_KIND, "episodic_uniform"),
    "official_dense": (OFFICIAL_ACT_ROLLOUT_KIND, "dense_windows"),
    "contact_dense": (ACT_ALIGNED_HIGH_RATE_ROLLOUT_KIND, "dense_windows"),
    "contact_episodic": (
        ACT_ALIGNED_HIGH_RATE_ROLLOUT_KIND,
        "episodic_uniform",
    ),
}


@dataclass
class EpisodePredictions:
    chunks: dict[str, list[np.ndarray]] = field(
        default_factory=lambda: defaultdict(list)
    )
    current_targets: list[np.ndarray] = field(default_factory=list)
    chunk_error_totals: dict[str, float] = field(
        default_factory=lambda: defaultdict(float)
    )
    chunk_error_counts: dict[str, int] = field(
        default_factory=lambda: defaultdict(int)
    )


def checkpoint_sha256(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def paired_bootstrap_summary(
    differences: Sequence[float],
    *,
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    """Summarize paired episode-level differences with a percentile CI."""

    values = np.asarray(differences, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("paired differences must be a finite non-empty vector")
    if resamples <= 0:
        raise ValueError("bootstrap resamples must be positive")
    rng = np.random.default_rng(seed)
    draw_indices = rng.integers(0, values.size, size=(resamples, values.size))
    bootstrap_means = values[draw_indices].mean(axis=1)
    return {
        "episode_count": int(values.size),
        "mean_difference": float(values.mean()),
        "median_difference": float(np.median(values)),
        "second_wins": int(np.count_nonzero(values < 0.0)),
        "first_wins": int(np.count_nonzero(values > 0.0)),
        "ties": int(np.count_nonzero(values == 0.0)),
        "bootstrap_resamples": int(resamples),
        "bootstrap_seed": int(seed),
        "mean_difference_ci95": [
            float(np.quantile(bootstrap_means, 0.025)),
            float(np.quantile(bootstrap_means, 0.975)),
        ],
    }


def interaction_difference(row: Mapping[str, float]) -> float:
    """Return (Contact-Official)_dense - (Contact-Official)_episodic."""

    return float(
        (row["contact_dense"] - row["official_dense"])
        - (row["contact_episodic"] - row["official_episodic"])
    )


def _category(episode_id: str) -> str:
    category = episode_id.split("/", 1)[0]
    if category not in {"clean_100", "collision_100"}:
        raise ValueError(f"unexpected scripted200 category: {category!r}")
    return category


def _sibling_sampling_mode(checkpoint_path: Path) -> str:
    metadata_path = checkpoint_path.parent / "run_metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"checkpoint companion run_metadata.json is missing: {metadata_path}"
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    mode = metadata.get("train_sampling_mode")
    if mode not in {"episodic_uniform", "dense_windows"}:
        raise ValueError(f"invalid training sampling mode in {metadata_path}: {mode!r}")
    return str(mode)


def _same_numeric_contract(
    left: Mapping[str, Any], right: Mapping[str, Any], names: Sequence[str]
) -> bool:
    for name in names:
        if name not in left or name not in right:
            return False
        if not np.array_equal(
            np.asarray(left[name], dtype=np.float64),
            np.asarray(right[name], dtype=np.float64),
        ):
            return False
    return True


def _validate_policy_contracts(
    adapters: Mapping[str, RolloutPolicyAdapter],
    checkpoint_paths: Mapping[str, Path],
) -> None:
    reference = adapters[POLICY_ORDER[0]]
    shared_config_names = (
        "chunk_len",
        "num_cameras",
        "image_height",
        "image_width",
        "q_dim",
        "action_dim",
        "imagenet_normalize",
    )
    for policy_id in POLICY_ORDER:
        adapter = adapters[policy_id]
        expected_kind, expected_sampling = EXPECTED_CONTRACTS[policy_id]
        if adapter.kind != expected_kind:
            raise ValueError(
                f"{policy_id} has rollout kind {adapter.kind!r}, expected "
                f"{expected_kind!r}"
            )
        actual_sampling = _sibling_sampling_mode(checkpoint_paths[policy_id])
        if actual_sampling != expected_sampling:
            raise ValueError(
                f"{policy_id} sampling mode is {actual_sampling!r}, expected "
                f"{expected_sampling!r}"
            )
        for name in shared_config_names:
            if getattr(adapter.config, name) != getattr(reference.config, name):
                raise ValueError(f"checkpoint model contracts differ for {name}")
    contact_reference = adapters["contact_dense"]
    contact_other = adapters["contact_episodic"]
    if contact_reference.config != contact_other.config:
        raise ValueError("the two Contact-CVAE model contracts differ")
    if not _same_numeric_contract(
        contact_reference.normalization,
        contact_other.normalization,
        (
            "qpos_mean",
            "qpos_std",
            "action_mean",
            "action_std",
            "force_mean",
            "force_std",
        ),
    ):
        raise ValueError("the two Contact-CVAE normalization contracts differ")


def _add_action_metrics(
    metrics: MetricSums,
    policy_id: str,
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
) -> None:
    metrics.add_absolute_error(
        f"{policy_id}_{PRIMARY_METRIC}", prediction, target, valid
    )
    for horizon in HORIZON_INDICES:
        if horizon < valid.shape[1]:
            metrics.add_absolute_error(
                f"{policy_id}_action_h{horizon + 1}_l1_physical",
                prediction[:, horizon],
                target[:, horizon],
                valid[:, horizon],
            )


def _contact_output(adapter: RolloutPolicyAdapter, batch) -> Mapping[str, Any]:
    output = adapter.model(
        batch.images,
        batch.qpos,
        batch.online_force_intervals,
        batch.online_force_relative_time,
        batch.online_force_sample_padding_mask,
        batch.online_force_interval_padding_mask,
        contact_latent_mode="zero",
        deterministic_prior=True,
    )
    if output.get("contact_latent_source") != "zero":
        raise RuntimeError("Contact-CVAE evaluation did not use zero latent")
    if float(output["z_contact"].abs().max().item()) != 0.0:
        raise RuntimeError("Contact-CVAE deployment latent is not exactly zero")
    return output


def _rows_for_scope(
    rows: Sequence[Mapping[str, Any]], scope: str
) -> list[Mapping[str, Any]]:
    if scope == "all":
        return list(rows)
    return [row for row in rows if row["category"] == scope]


def _comparison_report(
    rows: Sequence[Mapping[str, Any]],
    global_means: Mapping[str, float],
    *,
    bootstrap_resamples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    reports: dict[str, Any] = {}
    for scope_index, scope in enumerate(("all", "clean_100", "collision_100")):
        selected = _rows_for_scope(rows, scope)
        if not selected:
            reports[scope] = {
                "episode_count": 0,
                "pairwise": {},
                "interaction": None,
            }
            continue
        pairwise = {}
        for pair_index, (first, second) in enumerate(
            itertools.combinations(POLICY_ORDER, 2)
        ):
            differences = [
                float(row[f"{second}_{PRIMARY_METRIC}"])
                - float(row[f"{first}_{PRIMARY_METRIC}"])
                for row in selected
            ]
            item = paired_bootstrap_summary(
                differences,
                resamples=bootstrap_resamples,
                seed=bootstrap_seed + scope_index * 100 + pair_index,
            )
            item.update(
                {
                    "first": first,
                    "second": second,
                    "difference_semantics": "second_minus_first; lower_is_better",
                }
            )
            if scope == "all":
                item["global_scalar_weighted_difference"] = float(
                    global_means[f"{second}_{PRIMARY_METRIC}"]
                    - global_means[f"{first}_{PRIMARY_METRIC}"]
                )
            pairwise[f"{second}_minus_{first}"] = item
        interaction_values = [
            interaction_difference(
                {
                    policy_id: float(row[f"{policy_id}_{PRIMARY_METRIC}"])
                    for policy_id in POLICY_ORDER
                }
            )
            for row in selected
        ]
        interaction = paired_bootstrap_summary(
            interaction_values,
            resamples=bootstrap_resamples,
            seed=bootstrap_seed + scope_index * 100 + 99,
        )
        interaction["definition"] = (
            "(contact_dense-official_dense)-"
            "(contact_episodic-official_episodic); lower favors Contact more "
            "under dense sampling"
        )
        reports[scope] = {"pairwise": pairwise, "interaction": interaction}
    return reports


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested but CUDA is unavailable")
    device = torch.device(args.device)
    manifest = load_paired_episode_subset_manifest(
        args.experiment_manifest, data_root=args.data_root
    )
    records = (
        manifest.validation_episodes
        if args.split == "validation"
        else manifest.holdout_episodes
    )
    checkpoint_paths = {
        policy_id: getattr(args, f"{policy_id}_checkpoint")
        for policy_id in POLICY_ORDER
    }
    checkpoints = {
        policy_id: _load_checkpoint(path)
        for policy_id, path in checkpoint_paths.items()
    }
    for checkpoint in checkpoints.values():
        validate_checkpoint_experiment_provenance(
            checkpoint.get("experiment_manifest"), manifest
        )
    adapters = {
        policy_id: RolloutPolicyAdapter.from_checkpoint(checkpoint, device=device)
        for policy_id, checkpoint in checkpoints.items()
    }
    _validate_policy_contracts(adapters, checkpoint_paths)

    contact = adapters["contact_dense"]
    dataset = ACTAlignedHighRateHDF5Dataset(
        args.data_root,
        records,
        NormalizationStats.from_dict(dict(contact.normalization)),
        contact.config,
    )
    sample_count = len(dataset)
    if args.max_samples is not None:
        sample_count = min(sample_count, args.max_samples)
    evaluation_dataset = (
        dataset if sample_count == len(dataset) else Subset(dataset, range(sample_count))
    )
    loader = DataLoader(
        evaluation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_high_rate_samples,
    )
    metrics = MetricSums()
    episodes: dict[str, EpisodePredictions] = defaultdict(EpisodePredictions)
    cursor = 0
    start_time = time.monotonic()
    try:
        for batch_index, cpu_batch in enumerate(loader, start=1):
            batch_size = cpu_batch.batch_size
            batch = cpu_batch.to(device, non_blocking=device.type == "cuda")
            batch.validate(contact.config)
            raw_qpos = _denormalize(
                batch.qpos,
                contact.normalization["qpos_mean"],
                contact.normalization["qpos_std"],
            )
            raw_action = _denormalize(
                batch.action_chunk,
                contact.normalization["action_mean"],
                contact.normalization["action_std"],
            )
            action_valid = ~batch.action_padding_mask
            physical_predictions: dict[str, torch.Tensor] = {}
            contact_outputs: dict[str, Mapping[str, Any]] = {}
            with torch.inference_mode():
                for policy_id in POLICY_ORDER:
                    adapter = adapters[policy_id]
                    if adapter.kind == OFFICIAL_ACT_ROLLOUT_KIND:
                        normalized_qpos = _normalize(
                            raw_qpos,
                            adapter.normalization["qpos_mean"],
                            adapter.normalization["qpos_std"],
                        )
                        output = adapter.model(batch.images, normalized_qpos)
                    else:
                        output = _contact_output(adapter, batch)
                        contact_outputs[policy_id] = output
                    physical = _denormalize(
                        output["pred_action"],
                        adapter.normalization["action_mean"],
                        adapter.normalization["action_std"],
                    )
                    physical_predictions[policy_id] = physical
                    _add_action_metrics(
                        metrics, policy_id, physical, raw_action, action_valid
                    )

            endpoint_target = _denormalize(
                batch.future_force_target,
                contact.normalization["force_mean"],
                contact.normalization["force_std"],
            )
            highrate_target = _denormalize(
                batch.future_force_intervals,
                contact.normalization["force_mean"],
                contact.normalization["force_std"],
            )
            for policy_id, output in contact_outputs.items():
                endpoint_prediction = _denormalize(
                    output["pred_force"],
                    contact.normalization["force_mean"],
                    contact.normalization["force_std"],
                )
                highrate_prediction = _denormalize(
                    output["pred_force_highrate"],
                    contact.normalization["force_mean"],
                    contact.normalization["force_std"],
                )
                metrics.add_absolute_error(
                    f"{policy_id}_endpoint_wrench_component_l1_mixed_units",
                    endpoint_prediction,
                    endpoint_target,
                    ~batch.future_force_interval_padding_mask,
                )
                metrics.add_absolute_error(
                    f"{policy_id}_highrate_wrench_component_l1_mixed_units",
                    highrate_prediction,
                    highrate_target,
                    ~batch.future_force_sample_padding_mask,
                )

            for offset in range(batch_size):
                episode_index, timestep = dataset.index[cursor + offset]
                record = records[episode_index]
                item = episodes[record.episode_id]
                if timestep != len(item.current_targets):
                    raise RuntimeError("evaluation samples are not in episode-time order")
                valid = action_valid[offset]
                target = raw_action[offset]
                item.current_targets.append(target[0].detach().cpu().numpy())
                for policy_id, prediction in physical_predictions.items():
                    item.chunks[policy_id].append(
                        prediction[offset].detach().cpu().numpy()
                    )
                    selected = (prediction[offset] - target).abs()[valid]
                    item.chunk_error_totals[policy_id] += float(
                        selected.double().sum().item()
                    )
                    item.chunk_error_counts[policy_id] += int(selected.numel())
            cursor += batch_size
            if batch_index % args.log_interval == 0 or cursor == sample_count:
                elapsed = time.monotonic() - start_time
                rate = cursor / max(elapsed, 1.0e-9)
                eta = (sample_count - cursor) / max(rate, 1.0e-9)
                print(
                    f"split={args.split} samples={cursor}/{sample_count} "
                    f"rate={rate:.2f}/s eta_seconds={eta:.1f}",
                    flush=True,
                )
    finally:
        dataset.close()
    if cursor != sample_count:
        raise RuntimeError(f"evaluated {cursor} samples, expected {sample_count}")

    episode_rows: list[dict[str, Any]] = []
    for record in records:
        item = episodes.get(record.episode_id)
        if item is None:
            if args.max_samples is not None:
                continue
            raise RuntimeError(f"missing {args.split} episode {record.episode_id}")
        row: dict[str, Any] = {
            "episode_id": record.episode_id,
            "category": _category(record.episode_id),
            "num_steps": len(item.current_targets),
        }
        for policy_id in POLICY_ORDER:
            error_count = item.chunk_error_counts[policy_id]
            if error_count <= 0:
                raise RuntimeError(f"no valid action targets for {record.episode_id}")
            row[f"{policy_id}_{PRIMARY_METRIC}"] = (
                item.chunk_error_totals[policy_id] / error_count
            )
            temporal_total, temporal_count = temporal_ensemble_l1(
                item.chunks[policy_id],
                item.current_targets,
                decay=args.temporal_decay,
            )
            temporal_name = f"{policy_id}_temporal_action_l1_physical"
            row[temporal_name] = temporal_total / temporal_count
            metrics.totals[temporal_name] += temporal_total
            metrics.counts[temporal_name] += temporal_count
        row["sampling_model_interaction"] = interaction_difference(
            {
                policy_id: float(row[f"{policy_id}_{PRIMARY_METRIC}"])
                for policy_id in POLICY_ORDER
            }
        )
        episode_rows.append(row)

    global_means = metrics.means()
    stratified_primary = {}
    for category in ("clean_100", "collision_100"):
        category_rows = _rows_for_scope(episode_rows, category)
        stratified_primary[category] = (
            {
                policy_id: float(
                    np.mean(
                        [
                            row[f"{policy_id}_{PRIMARY_METRIC}"]
                            for row in category_rows
                        ]
                    )
                )
                for policy_id in POLICY_ORDER
            }
            if category_rows
            else None
        )
    comparisons = _comparison_report(
        episode_rows,
        global_means,
        bootstrap_resamples=args.bootstrap_resamples,
        bootstrap_seed=args.bootstrap_seed,
    )
    checkpoint_report = {
        policy_id: {
            "path": str(path),
            "sha256": checkpoint_sha256(path),
            "rollout_kind": adapters[policy_id].kind,
            "architecture": adapters[policy_id].config.architecture_version,
            "train_sampling_mode": EXPECTED_CONTRACTS[policy_id][1],
        }
        for policy_id, path in checkpoint_paths.items()
    }
    summary = {
        "evaluation_version": EVALUATION_VERSION,
        "evaluation_split": args.split,
        "evaluation_scope": f"scripted200_manifest_{args.split}_all_timesteps",
        "partial_evaluation": args.max_samples is not None,
        "teacher_forced": True,
        "closed_loop_rollout": False,
        "primary_metric": PRIMARY_METRIC,
        "primary_metric_weighting": "all_valid_action_chunk_scalars",
        "paired_inference_unit": "episode_equal_weight",
        "deployment_latent": "zero",
        "data_root": str(args.data_root),
        "experiment_manifest": str(args.experiment_manifest),
        "dataset_fingerprint": manifest.dataset_fingerprint,
        "episode_count": len(episode_rows),
        "timestep_count": cursor,
        "batch_size": args.batch_size,
        "device": str(device),
        "temporal_decay": args.temporal_decay,
        "checkpoints": checkpoint_report,
        "global_weighted_metrics": global_means,
        "stratified_episode_mean_primary": stratified_primary,
        "paired_comparisons": comparisons,
        "elapsed_seconds": time.monotonic() - start_time,
        "passed": True,
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    rows_path = args.output_dir / "per_episode.csv"
    fieldnames = sorted({name for row in episode_rows for name in row})
    with rows_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(episode_rows)
    print(f"summary={summary_path}")
    print(f"per_episode={rows_path}")
    return summary


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_root", type=Path)
    parser.add_argument("--experiment-manifest", type=Path, required=True)
    for policy_id in POLICY_ORDER:
        parser.add_argument(
            f"--{policy_id.replace('_', '-')}-checkpoint",
            dest=f"{policy_id}_checkpoint",
            type=Path,
            required=True,
        )
    parser.add_argument("--split", choices=("validation", "holdout"), required=True)
    parser.add_argument(
        "--confirm-holdout-evaluation",
        action="store_true",
        help="Required acknowledgement when --split=holdout.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--temporal-decay",
        type=float,
        default=OFFICIAL_TEMPORAL_AGGREGATION_DECAY,
    )
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--max-samples", type=int)
    args = parser.parse_args(argv)
    for name in (
        "data_root",
        "experiment_manifest",
        *(f"{policy_id}_checkpoint" for policy_id in POLICY_ORDER),
    ):
        path = getattr(args, name).expanduser().resolve()
        setattr(args, name, path)
        if not path.exists():
            parser.error(f"{name.replace('_', ' ')} does not exist: {path}")
    args.output_dir = args.output_dir.expanduser().resolve()
    if args.output_dir.exists():
        parser.error(f"output directory already exists: {args.output_dir}")
    if args.split == "holdout" and not args.confirm_holdout_evaluation:
        parser.error("--split=holdout requires --confirm-holdout-evaluation")
    if args.split != "holdout" and args.confirm_holdout_evaluation:
        parser.error("--confirm-holdout-evaluation is valid only for holdout")
    if (
        args.batch_size <= 0
        or args.num_workers < 0
        or args.log_interval <= 0
        or args.bootstrap_resamples <= 0
    ):
        parser.error("batch/log/bootstrap values must be positive and workers non-negative")
    if args.max_samples is not None and args.max_samples <= 0:
        parser.error("max samples must be positive")
    if not math.isfinite(args.temporal_decay) or args.temporal_decay < 0.0:
        parser.error("temporal decay must be finite and non-negative")
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        evaluate(args)
    except Exception as error:
        print(f"error: sampling-ablation evaluation failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
