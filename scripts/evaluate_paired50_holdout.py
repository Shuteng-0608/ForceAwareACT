#!/usr/bin/env python3
"""Evaluate paired Official ACT and high-rate Contact-CVAE on holdout episodes."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean, median
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
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
    OfficialTemporalActionChunkExecutor,
    RolloutPolicyAdapter,
)


EVALUATION_VERSION = "paired50_holdout_all_timesteps_v1"
HORIZON_INDICES = (0, 9, 24, 49, 99)


@dataclass
class MetricSums:
    totals: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    counts: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    def add_absolute_error(
        self,
        name: str,
        prediction: torch.Tensor,
        target: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> None:
        if prediction.shape != target.shape:
            raise ValueError(f"{name} prediction and target shapes differ")
        if valid_mask.shape != prediction.shape[:-1]:
            raise ValueError(f"{name} mask must match all non-feature dimensions")
        expanded = valid_mask.unsqueeze(-1).expand_as(prediction)
        selected = (prediction - target).abs()[expanded]
        if selected.numel() == 0:
            return
        if not torch.isfinite(selected).all():
            raise FloatingPointError(f"{name} contains non-finite errors")
        self.totals[name] += float(selected.double().sum().item())
        self.counts[name] += int(selected.numel())

    def add_numpy_error(
        self,
        name: str,
        prediction: np.ndarray,
        target: np.ndarray,
    ) -> None:
        difference = np.abs(
            np.asarray(prediction, dtype=np.float64)
            - np.asarray(target, dtype=np.float64)
        )
        if difference.size == 0 or not np.isfinite(difference).all():
            raise FloatingPointError(f"{name} contains invalid errors")
        self.totals[name] += float(difference.sum())
        self.counts[name] += int(difference.size)

    def merge(self, other: "MetricSums") -> None:
        for name, value in other.totals.items():
            self.totals[name] += value
        for name, value in other.counts.items():
            self.counts[name] += value

    def means(self) -> dict[str, float]:
        return {
            name: self.totals[name] / self.counts[name]
            for name in sorted(self.totals)
            if self.counts[name] > 0
        }


@dataclass
class EpisodePredictions:
    official_chunks: list[np.ndarray] = field(default_factory=list)
    contact_chunks: list[np.ndarray] = field(default_factory=list)
    current_targets: list[np.ndarray] = field(default_factory=list)


def temporal_ensemble_l1(
    chunks: Sequence[np.ndarray],
    targets: Sequence[np.ndarray],
    *,
    decay: float,
) -> tuple[float, int]:
    if len(chunks) != len(targets) or not chunks:
        raise ValueError("temporal evaluation requires matching non-empty sequences")
    executor = OfficialTemporalActionChunkExecutor(decay=decay)
    total = 0.0
    count = 0
    for step, (chunk, target) in enumerate(zip(chunks, targets)):
        result = executor.update(step, chunk)
        difference = np.abs(
            result.action - np.asarray(target, dtype=np.float64)
        )
        if not np.isfinite(difference).all():
            raise FloatingPointError("temporal ensemble error is not finite")
        total += float(difference.sum())
        count += int(difference.size)
    return total, count


def _tensor_values(
    values: Sequence[float],
    reference: torch.Tensor,
) -> torch.Tensor:
    return reference.new_tensor(values)


def _denormalize(
    values: torch.Tensor,
    mean_values: Sequence[float],
    std_values: Sequence[float],
) -> torch.Tensor:
    return values * _tensor_values(std_values, values) + _tensor_values(
        mean_values, values
    )


def _normalize(
    values: torch.Tensor,
    mean_values: Sequence[float],
    std_values: Sequence[float],
) -> torch.Tensor:
    return (values - _tensor_values(mean_values, values)) / _tensor_values(
        std_values, values
    )


def _load_checkpoint(path: Path) -> Mapping[str, Any]:
    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    if not isinstance(checkpoint, Mapping):
        raise ValueError(f"checkpoint must contain a mapping: {path}")
    return checkpoint


def _validate_pair(
    official: RolloutPolicyAdapter,
    contact: RolloutPolicyAdapter,
) -> None:
    if official.kind != OFFICIAL_ACT_ROLLOUT_KIND:
        raise ValueError("official checkpoint is not an Official ACT policy")
    if contact.kind != ACT_ALIGNED_HIGH_RATE_ROLLOUT_KIND:
        raise ValueError("contact checkpoint is not a high-rate Contact-CVAE")
    shared_names = (
        "chunk_len",
        "num_cameras",
        "image_height",
        "image_width",
        "q_dim",
        "action_dim",
        "imagenet_normalize",
    )
    for name in shared_names:
        if getattr(official.config, name) != getattr(contact.config, name):
            raise ValueError(f"checkpoint model contracts differ for {name}")


def _add_batch_metrics(
    accumulator: MetricSums,
    *,
    official_prediction_normalized: torch.Tensor,
    contact_prediction_normalized: torch.Tensor,
    official_target_normalized: torch.Tensor,
    contact_target_normalized: torch.Tensor,
    official_prediction_physical: torch.Tensor,
    contact_prediction_physical: torch.Tensor,
    target_physical: torch.Tensor,
    action_valid: torch.Tensor,
    contact_force_prediction_physical: torch.Tensor,
    endpoint_force_target_physical: torch.Tensor,
    endpoint_force_valid: torch.Tensor,
    contact_highrate_prediction_physical: torch.Tensor,
    highrate_force_target_physical: torch.Tensor,
    highrate_force_valid: torch.Tensor,
) -> None:
    accumulator.add_absolute_error(
        "official_action_chunk_l1_normalized",
        official_prediction_normalized,
        official_target_normalized,
        action_valid,
    )
    accumulator.add_absolute_error(
        "contact_action_chunk_l1_normalized",
        contact_prediction_normalized,
        contact_target_normalized,
        action_valid,
    )
    accumulator.add_absolute_error(
        "official_action_chunk_l1_physical",
        official_prediction_physical,
        target_physical,
        action_valid,
    )
    accumulator.add_absolute_error(
        "contact_action_chunk_l1_physical",
        contact_prediction_physical,
        target_physical,
        action_valid,
    )
    for horizon in HORIZON_INDICES:
        if horizon >= action_valid.shape[1]:
            continue
        valid = action_valid[:, horizon]
        accumulator.add_absolute_error(
            f"official_action_h{horizon + 1}_l1_physical",
            official_prediction_physical[:, horizon],
            target_physical[:, horizon],
            valid,
        )
        accumulator.add_absolute_error(
            f"contact_action_h{horizon + 1}_l1_physical",
            contact_prediction_physical[:, horizon],
            target_physical[:, horizon],
            valid,
        )
    accumulator.add_absolute_error(
        "contact_endpoint_wrench_component_l1_mixed_units",
        contact_force_prediction_physical,
        endpoint_force_target_physical,
        endpoint_force_valid,
    )
    accumulator.add_absolute_error(
        "contact_endpoint_linear_force_l1_n",
        contact_force_prediction_physical[..., :3],
        endpoint_force_target_physical[..., :3],
        endpoint_force_valid,
    )
    accumulator.add_absolute_error(
        "contact_endpoint_torque_l1_nm",
        contact_force_prediction_physical[..., 3:],
        endpoint_force_target_physical[..., 3:],
        endpoint_force_valid,
    )
    accumulator.add_absolute_error(
        "contact_highrate_wrench_component_l1_mixed_units",
        contact_highrate_prediction_physical,
        highrate_force_target_physical,
        highrate_force_valid,
    )
    accumulator.add_absolute_error(
        "contact_highrate_linear_force_l1_n",
        contact_highrate_prediction_physical[..., :3],
        highrate_force_target_physical[..., :3],
        highrate_force_valid,
    )
    accumulator.add_absolute_error(
        "contact_highrate_torque_l1_nm",
        contact_highrate_prediction_physical[..., 3:],
        highrate_force_target_physical[..., 3:],
        highrate_force_valid,
    )


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested but CUDA is unavailable")
    device = torch.device(args.device)
    subset = load_paired_episode_subset_manifest(
        args.experiment_manifest,
        data_root=args.data_root,
    )
    official_checkpoint = _load_checkpoint(args.official_checkpoint)
    contact_checkpoint = _load_checkpoint(args.contact_checkpoint)
    validate_checkpoint_experiment_provenance(
        official_checkpoint.get("experiment_manifest"), subset
    )
    validate_checkpoint_experiment_provenance(
        contact_checkpoint.get("experiment_manifest"), subset
    )
    official = RolloutPolicyAdapter.from_checkpoint(
        official_checkpoint, device=device
    )
    contact = RolloutPolicyAdapter.from_checkpoint(
        contact_checkpoint, device=device
    )
    _validate_pair(official, contact)
    contact_stats = NormalizationStats.from_dict(dict(contact.normalization))
    dataset = ACTAlignedHighRateHDF5Dataset(
        args.data_root,
        subset.holdout_episodes,
        contact_stats,
        contact.config,
    )
    sample_count = len(dataset)
    if args.max_samples is not None:
        sample_count = min(sample_count, args.max_samples)
    evaluation_dataset = (
        dataset
        if sample_count == len(dataset)
        else Subset(dataset, range(sample_count))
    )
    loader = DataLoader(
        evaluation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_high_rate_samples,
    )
    global_metrics = MetricSums()
    episodes: dict[str, EpisodePredictions] = defaultdict(EpisodePredictions)
    cursor = 0
    start_time = time.monotonic()
    official_norm = official.normalization
    contact_norm = contact.normalization
    try:
        for batch_index, cpu_batch in enumerate(loader, start=1):
            batch_size = cpu_batch.batch_size
            batch = cpu_batch.to(device, non_blocking=device.type == "cuda")
            batch.validate(contact.config)
            raw_qpos = _denormalize(
                batch.qpos,
                contact_norm["qpos_mean"],
                contact_norm["qpos_std"],
            )
            raw_action = _denormalize(
                batch.action_chunk,
                contact_norm["action_mean"],
                contact_norm["action_std"],
            )
            official_qpos = _normalize(
                raw_qpos,
                official_norm["qpos_mean"],
                official_norm["qpos_std"],
            )
            official_action_target = _normalize(
                raw_action,
                official_norm["action_mean"],
                official_norm["action_std"],
            )
            with torch.inference_mode():
                official_output = official.model(batch.images, official_qpos)
                contact_output = contact.model(
                    batch.images,
                    batch.qpos,
                    batch.online_force_intervals,
                    batch.online_force_relative_time,
                    batch.online_force_sample_padding_mask,
                    batch.online_force_interval_padding_mask,
                    contact_latent_mode="zero",
                    deterministic_prior=True,
                )
            if contact_output.get("contact_latent_source") != "zero":
                raise RuntimeError("contact holdout evaluation did not use zero latent")
            if float(contact_output["z_contact"].abs().max().item()) != 0.0:
                raise RuntimeError("contact deployment latent is not exactly zero")
            official_action_physical = _denormalize(
                official_output["pred_action"],
                official_norm["action_mean"],
                official_norm["action_std"],
            )
            contact_action_physical = _denormalize(
                contact_output["pred_action"],
                contact_norm["action_mean"],
                contact_norm["action_std"],
            )
            endpoint_force_target_physical = _denormalize(
                batch.future_force_target,
                contact_norm["force_mean"],
                contact_norm["force_std"],
            )
            highrate_force_target_physical = _denormalize(
                batch.future_force_intervals,
                contact_norm["force_mean"],
                contact_norm["force_std"],
            )
            contact_force_physical = _denormalize(
                contact_output["pred_force"],
                contact_norm["force_mean"],
                contact_norm["force_std"],
            )
            contact_highrate_force_physical = _denormalize(
                contact_output["pred_force_highrate"],
                contact_norm["force_mean"],
                contact_norm["force_std"],
            )
            action_valid = ~batch.action_padding_mask
            endpoint_force_valid = ~batch.future_force_interval_padding_mask
            highrate_force_valid = ~batch.future_force_sample_padding_mask
            batch_metrics = MetricSums()
            _add_batch_metrics(
                batch_metrics,
                official_prediction_normalized=official_output["pred_action"],
                contact_prediction_normalized=contact_output["pred_action"],
                official_target_normalized=official_action_target,
                contact_target_normalized=batch.action_chunk,
                official_prediction_physical=official_action_physical,
                contact_prediction_physical=contact_action_physical,
                target_physical=raw_action,
                action_valid=action_valid,
                contact_force_prediction_physical=contact_force_physical,
                endpoint_force_target_physical=endpoint_force_target_physical,
                endpoint_force_valid=endpoint_force_valid,
                contact_highrate_prediction_physical=contact_highrate_force_physical,
                highrate_force_target_physical=highrate_force_target_physical,
                highrate_force_valid=highrate_force_valid,
            )
            global_metrics.merge(batch_metrics)
            for offset in range(batch_size):
                episode_index, timestep = dataset.index[cursor + offset]
                record = subset.holdout_episodes[episode_index]
                item = episodes[record.episode_id]
                if timestep != len(item.current_targets):
                    raise RuntimeError("holdout samples are not in episode-time order")
                item.official_chunks.append(
                    official_action_physical[offset].detach().cpu().numpy()
                )
                item.contact_chunks.append(
                    contact_action_physical[offset].detach().cpu().numpy()
                )
                item.current_targets.append(
                    raw_action[offset, 0].detach().cpu().numpy()
                )
            cursor += batch_size
            if batch_index % args.log_interval == 0 or cursor == sample_count:
                elapsed = time.monotonic() - start_time
                rate = cursor / max(elapsed, 1.0e-9)
                remaining = (sample_count - cursor) / max(rate, 1.0e-9)
                print(
                    f"samples={cursor}/{sample_count} "
                    f"rate={rate:.2f}/s eta_seconds={remaining:.1f}",
                    flush=True,
                )
    finally:
        dataset.close()
    if cursor != sample_count:
        raise RuntimeError(f"evaluated {cursor} samples, expected {sample_count}")

    episode_rows = []
    for record in subset.holdout_episodes:
        item = episodes.get(record.episode_id)
        if item is None:
            if args.max_samples is not None:
                continue
            raise RuntimeError(f"missing holdout episode {record.episode_id}")
        official_sum, official_count = temporal_ensemble_l1(
            item.official_chunks,
            item.current_targets,
            decay=args.temporal_decay,
        )
        contact_sum, contact_count = temporal_ensemble_l1(
            item.contact_chunks,
            item.current_targets,
            decay=args.temporal_decay,
        )
        global_metrics.totals["official_temporal_action_l1_physical"] += official_sum
        global_metrics.counts["official_temporal_action_l1_physical"] += official_count
        global_metrics.totals["contact_temporal_action_l1_physical"] += contact_sum
        global_metrics.counts["contact_temporal_action_l1_physical"] += contact_count
        targets = np.asarray(item.current_targets, dtype=np.float64)
        official_current = np.asarray(item.official_chunks, dtype=np.float64)[:, 0]
        contact_current = np.asarray(item.contact_chunks, dtype=np.float64)[:, 0]
        official_current_l1 = float(np.abs(official_current - targets).mean())
        contact_current_l1 = float(np.abs(contact_current - targets).mean())
        official_temporal_l1 = official_sum / official_count
        contact_temporal_l1 = contact_sum / contact_count
        episode_rows.append(
            {
                "episode_id": record.episode_id,
                "num_steps": len(item.current_targets),
                "official_current_action_l1_physical": official_current_l1,
                "contact_current_action_l1_physical": contact_current_l1,
                "current_contact_minus_official": (
                    contact_current_l1 - official_current_l1
                ),
                "official_temporal_action_l1_physical": official_temporal_l1,
                "contact_temporal_action_l1_physical": contact_temporal_l1,
                "temporal_contact_minus_official": (
                    contact_temporal_l1 - official_temporal_l1
                ),
            }
        )

    global_means = global_metrics.means()
    comparison_names = (
        "action_chunk_l1_physical",
        "action_h1_l1_physical",
        "action_h10_l1_physical",
        "action_h25_l1_physical",
        "action_h50_l1_physical",
        "action_h100_l1_physical",
        "temporal_action_l1_physical",
    )
    comparisons = {}
    for suffix in comparison_names:
        official_name = f"official_{suffix}"
        contact_name = f"contact_{suffix}"
        if official_name not in global_means or contact_name not in global_means:
            continue
        official_value = global_means[official_name]
        contact_value = global_means[contact_name]
        comparisons[suffix] = {
            "official": official_value,
            "contact": contact_value,
            "contact_minus_official": contact_value - official_value,
            "contact_relative_improvement": (
                (official_value - contact_value) / official_value
                if official_value > 0.0
                else None
            ),
        }
    temporal_differences = [
        float(row["temporal_contact_minus_official"])
        for row in episode_rows
    ]
    summary = {
        "evaluation_version": EVALUATION_VERSION,
        "evaluation_scope": (
            "paired50_manifest_holdout_all_timesteps"
            if args.max_samples is None
            else "paired50_manifest_holdout_prefix_smoke"
        ),
        "partial_evaluation": args.max_samples is not None,
        "teacher_forced": True,
        "closed_loop_rollout": False,
        "deployment_latents": {"official": "zero", "contact": "zero"},
        "data_root": str(args.data_root.resolve()),
        "experiment_manifest": str(args.experiment_manifest.resolve()),
        "dataset_fingerprint": subset.dataset_fingerprint,
        "holdout_episode_count": len(episode_rows),
        "holdout_timestep_count": cursor,
        "batch_size": args.batch_size,
        "device": str(device),
        "temporal_decay": args.temporal_decay,
        "official_checkpoint": str(args.official_checkpoint.resolve()),
        "contact_checkpoint": str(args.contact_checkpoint.resolve()),
        "official_architecture": official.config.architecture_version,
        "contact_architecture": contact.config.architecture_version,
        "global_weighted_metrics": global_means,
        "comparisons": comparisons,
        "per_episode_temporal_comparison": {
            "contact_wins": sum(value < 0.0 for value in temporal_differences),
            "official_wins": sum(value > 0.0 for value in temporal_differences),
            "ties": sum(value == 0.0 for value in temporal_differences),
            "mean_contact_minus_official": mean(temporal_differences),
            "median_contact_minus_official": median(temporal_differences),
        },
        "elapsed_seconds": time.monotonic() - start_time,
        "passed": True,
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    rows_path = args.output_dir / "per_episode.csv"
    fieldnames = sorted({key for row in episode_rows for key in row})
    with rows_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(episode_rows)
    print(f"summary={summary_path.resolve()}")
    print(f"per_episode={rows_path.resolve()}")
    return summary


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_root", type=Path)
    parser.add_argument("--experiment-manifest", type=Path, required=True)
    parser.add_argument("--official-checkpoint", type=Path, required=True)
    parser.add_argument("--contact-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--temporal-decay",
        type=float,
        default=OFFICIAL_TEMPORAL_AGGREGATION_DECAY,
    )
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--max-samples", type=int)
    args = parser.parse_args(argv)
    for name in (
        "data_root",
        "experiment_manifest",
        "official_checkpoint",
        "contact_checkpoint",
    ):
        path = getattr(args, name).expanduser().resolve()
        setattr(args, name, path)
        if not path.exists():
            parser.error(f"{name.replace('_', ' ')} does not exist: {path}")
    args.output_dir = args.output_dir.expanduser().resolve()
    if args.output_dir.exists():
        parser.error(f"output directory already exists: {args.output_dir}")
    if args.batch_size <= 0 or args.num_workers < 0 or args.log_interval <= 0:
        parser.error("batch size/log interval must be positive and workers non-negative")
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
        print(f"error: paired holdout evaluation failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
