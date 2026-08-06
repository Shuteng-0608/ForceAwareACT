#!/usr/bin/env python3
"""Sweep action execution semantics on the paired50 validation episodes."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
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
    OfficialTemporalActionChunkExecutor,
    RecedingChunkActionExecutor,
    RecencyTemporalActionChunkExecutor,
    RolloutPolicyAdapter,
    SignedAgeTemporalActionChunkExecutor,
    TemporalEndpointActionChunkExecutor,
)


SWEEP_VERSION = "paired50_validation_action_executor_sweep_v2"
CONTACT_STAGE_NAMES = ("free_lt5n", "contact_5_to_20n", "contact_ge20n")
OFFICIAL_BASELINE_ID = "official_temporal_k0p01"
SIGNED_OFFICIAL_BASELINE_ID = "signed_temporal_km0p01"
DEFAULT_SIGNED_DECAYS = (
    -1.0,
    -0.5,
    -0.3,
    -0.2,
    -0.1,
    -0.05,
    -0.03,
    -0.02,
    -0.01,
    -0.005,
    0.0,
    0.005,
    0.01,
    0.02,
    0.03,
    0.05,
    0.1,
    0.2,
    0.3,
    0.5,
    1.0,
)
TEMPORAL_FAMILIES = {
    "official_temporal",
    "recency_temporal",
    "signed_temporal",
    "temporal_endpoint",
}


@dataclass(frozen=True)
class ExecutorSpec:
    executor_id: str
    family: str
    decay: float | None = None
    query_interval: int | None = None
    endpoint: str | None = None

    def build(self):
        if self.family == "official_temporal":
            return OfficialTemporalActionChunkExecutor(decay=float(self.decay))
        if self.family == "recency_temporal":
            return RecencyTemporalActionChunkExecutor(decay=float(self.decay))
        if self.family == "signed_temporal":
            return SignedAgeTemporalActionChunkExecutor(
                signed_decay=float(self.decay)
            )
        if self.family == "temporal_endpoint":
            return TemporalEndpointActionChunkExecutor(
                preference=str(self.endpoint)
            )
        if self.family == "receding_chunk":
            return RecedingChunkActionExecutor(
                query_interval=int(self.query_interval)
            )
        raise ValueError(f"unsupported executor family: {self.family}")


@dataclass
class EpisodeCache:
    official_chunks: list[np.ndarray] = field(default_factory=list)
    contact_chunks: list[np.ndarray] = field(default_factory=list)
    current_targets: list[np.ndarray] = field(default_factory=list)
    current_force_norms: list[float] = field(default_factory=list)


@dataclass(frozen=True)
class ReplayResult:
    actions: np.ndarray
    prediction_ages: np.ndarray
    policy_queried: np.ndarray
    chunk_indices: np.ndarray


@dataclass
class ErrorAccumulator:
    absolute_error_sum: float = 0.0
    component_count: int = 0
    step_l2_sum: float = 0.0
    step_count: int = 0

    def add(self, prediction: np.ndarray, target: np.ndarray) -> None:
        difference = np.asarray(prediction, dtype=np.float64) - np.asarray(
            target, dtype=np.float64
        )
        if difference.ndim != 2 or difference.shape != target.shape:
            raise ValueError("prediction and target must match [T, action_dim]")
        if not np.isfinite(difference).all():
            raise FloatingPointError("action replay produced non-finite error")
        self.absolute_error_sum += float(np.abs(difference).sum())
        self.component_count += int(difference.size)
        self.step_l2_sum += float(np.linalg.norm(difference, axis=1).sum())
        self.step_count += int(difference.shape[0])

    @property
    def l1(self) -> float | None:
        if self.component_count == 0:
            return None
        return self.absolute_error_sum / self.component_count

    @property
    def step_l2(self) -> float | None:
        if self.step_count == 0:
            return None
        return self.step_l2_sum / self.step_count


def _float_token(value: float) -> str:
    text = format(float(value), ".9g")
    return text.replace("-", "m").replace(".", "p")


def build_executor_specs(
    official_decays: Sequence[float],
    recency_decays: Sequence[float],
    receding_intervals: Sequence[int],
) -> tuple[ExecutorSpec, ...]:
    specs = [
        ExecutorSpec(
            executor_id=f"official_temporal_k{_float_token(value)}",
            family="official_temporal",
            decay=float(value),
        )
        for value in official_decays
    ]
    specs.extend(
        ExecutorSpec(
            executor_id=f"recency_temporal_k{_float_token(value)}",
            family="recency_temporal",
            decay=float(value),
        )
        for value in recency_decays
    )
    specs.extend(
        ExecutorSpec(
            executor_id=f"receding_chunk_q{value}",
            family="receding_chunk",
            query_interval=int(value),
        )
        for value in receding_intervals
    )
    identifiers = [item.executor_id for item in specs]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("executor sweep contains duplicate configurations")
    return tuple(specs)


def build_signed_temporal_specs(
    signed_decays: Sequence[float],
    *,
    include_endpoints: bool = True,
) -> tuple[ExecutorSpec, ...]:
    """Build a Q=1 grid with one signed old/new preference convention."""

    specs = [
        ExecutorSpec(
            executor_id=f"signed_temporal_k{_float_token(value)}",
            family="signed_temporal",
            decay=float(value),
            query_interval=1,
        )
        for value in signed_decays
    ]
    if include_endpoints:
        specs.extend(
            (
                ExecutorSpec(
                    executor_id="latest_only",
                    family="temporal_endpoint",
                    query_interval=1,
                    endpoint="newest",
                ),
                ExecutorSpec(
                    executor_id="oldest_only",
                    family="temporal_endpoint",
                    query_interval=1,
                    endpoint="oldest",
                ),
            )
        )
    identifiers = [item.executor_id for item in specs]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("signed temporal sweep contains duplicate configurations")
    if SIGNED_OFFICIAL_BASELINE_ID not in identifiers:
        raise ValueError(
            "signed temporal sweep must contain k=-0.01, which is equivalent "
            "to the official ACT k=0.01 baseline"
        )
    return tuple(specs)


def classify_contact_stages(force_norms: np.ndarray) -> np.ndarray:
    values = np.asarray(force_norms, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("force norms must be a finite one-dimensional array")
    stages = np.empty(values.shape, dtype=np.int64)
    stages[values < 5.0] = 0
    stages[(values >= 5.0) & (values < 20.0)] = 1
    stages[values >= 20.0] = 2
    return stages


def replay_action_chunks(
    chunks: Sequence[np.ndarray],
    spec: ExecutorSpec,
) -> ReplayResult:
    if not chunks:
        raise ValueError("action replay requires at least one predicted chunk")
    executor = spec.build()
    actions: list[np.ndarray] = []
    ages: list[float] = []
    queried: list[bool] = []
    chunk_indices: list[int] = []
    for step, chunk_value in enumerate(chunks):
        chunk = np.asarray(chunk_value, dtype=np.float64)
        if spec.family in TEMPORAL_FAMILIES:
            result = executor.update(step, chunk)
            actions.append(result.action)
            ages.append(result.weighted_mean_age)
            queried.append(True)
            chunk_indices.append(-1)
        else:
            query_now = executor.should_query(step)
            result = executor.update(step, chunk if query_now else None)
            actions.append(result.action)
            ages.append(float(result.prediction_age))
            queried.append(query_now)
            chunk_indices.append(result.chunk_index)
    action_array = np.asarray(actions, dtype=np.float64)
    if action_array.ndim != 2 or not np.isfinite(action_array).all():
        raise FloatingPointError("action replay returned invalid actions")
    return ReplayResult(
        actions=action_array,
        prediction_ages=np.asarray(ages, dtype=np.float64),
        policy_queried=np.asarray(queried, dtype=np.bool_),
        chunk_indices=np.asarray(chunk_indices, dtype=np.int64),
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


def _tensor_values(values: Sequence[float], reference: torch.Tensor) -> torch.Tensor:
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


def _latest_linear_force_norms(
    normalized_intervals: torch.Tensor,
    sample_padding_mask: torch.Tensor,
    force_mean: Sequence[float],
    force_std: Sequence[float],
) -> torch.Tensor:
    if normalized_intervals.shape[:-1] != sample_padding_mask.shape:
        raise ValueError("force interval values and padding mask shapes disagree")
    physical = _denormalize(normalized_intervals, force_mean, force_std)
    batch_size = physical.shape[0]
    flattened = physical.flatten(1, 2)
    valid = (~sample_padding_mask).flatten(1, 2)
    positions = torch.arange(valid.shape[1], device=valid.device).expand(
        batch_size, -1
    )
    latest = positions.masked_fill(~valid, -1).max(dim=1).values
    if (latest < 0).any():
        raise ValueError("each validation sample requires causal force history")
    selected = flattened[
        torch.arange(batch_size, device=flattened.device), latest
    ]
    return torch.linalg.vector_norm(selected[:, :3], dim=-1)


def _validate_pair(
    official: RolloutPolicyAdapter,
    contact: RolloutPolicyAdapter,
) -> None:
    if official.kind != OFFICIAL_ACT_ROLLOUT_KIND:
        raise ValueError("official checkpoint is not an Official ACT policy")
    if contact.kind != ACT_ALIGNED_HIGH_RATE_ROLLOUT_KIND:
        raise ValueError("contact checkpoint is not a high-rate Contact-CVAE")
    for name in (
        "chunk_len",
        "num_cameras",
        "image_height",
        "image_width",
        "q_dim",
        "action_dim",
        "imagenet_normalize",
    ):
        if getattr(official.config, name) != getattr(contact.config, name):
            raise ValueError(f"checkpoint model contracts differ for {name}")


def _phase_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    stage_indices: np.ndarray,
) -> dict[str, float | int | None]:
    result: dict[str, float | int | None] = {}
    for stage_index, stage_name in enumerate(CONTACT_STAGE_NAMES):
        selected = stage_indices == stage_index
        result[f"{stage_name}_steps"] = int(selected.sum())
        result[f"{stage_name}_l1_physical"] = (
            float(np.abs(prediction[selected] - target[selected]).mean())
            if selected.any()
            else None
        )
    return result


def _episode_replay_row(
    *,
    episode_id: str,
    model_name: str,
    spec: ExecutorSpec,
    replay: ReplayResult,
    targets: np.ndarray,
    force_norms: np.ndarray,
) -> dict[str, Any]:
    if replay.actions.shape != targets.shape:
        raise ValueError("replayed actions and current targets differ in shape")
    stages = classify_contact_stages(force_norms)
    differences = replay.actions - targets
    command_deltas = np.diff(replay.actions, axis=0)
    return {
        "episode_id": episode_id,
        "model": model_name,
        "executor_id": spec.executor_id,
        "executor_family": spec.family,
        "decay": spec.decay,
        "signed_decay": spec.decay if spec.family == "signed_temporal" else None,
        "temporal_endpoint": spec.endpoint,
        "query_interval": spec.query_interval,
        "num_steps": int(targets.shape[0]),
        "action_l1_physical": float(np.abs(differences).mean()),
        "action_step_l2_physical": float(
            np.linalg.norm(differences, axis=1).mean()
        ),
        "command_step_delta_l2_mean": (
            float(np.linalg.norm(command_deltas, axis=1).mean())
            if command_deltas.size
            else 0.0
        ),
        "command_total_variation_l2": (
            float(np.linalg.norm(command_deltas, axis=1).sum())
            if command_deltas.size
            else 0.0
        ),
        "prediction_age_mean_steps": float(replay.prediction_ages.mean()),
        "prediction_age_p95_steps": float(
            np.quantile(replay.prediction_ages, 0.95)
        ),
        "prediction_age_max_steps": float(replay.prediction_ages.max()),
        "policy_query_count": int(replay.policy_queried.sum()),
        "policy_query_fraction": float(replay.policy_queried.mean()),
        **_phase_metrics(replay.actions, targets, stages),
    }


def aggregate_replay_rows(
    episode_caches: Mapping[str, EpisodeCache],
    specs: Sequence[ExecutorSpec],
    *,
    reference_baseline_id: str = OFFICIAL_BASELINE_ID,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    per_episode: list[dict[str, Any]] = []
    aggregate: list[dict[str, Any]] = []
    for model_name, chunk_getter in (
        ("official", lambda item: item.official_chunks),
        ("contact", lambda item: item.contact_chunks),
    ):
        for spec in specs:
            overall = ErrorAccumulator()
            stages = {name: ErrorAccumulator() for name in CONTACT_STAGE_NAMES}
            episode_l1_values: list[float] = []
            prediction_ages: list[np.ndarray] = []
            command_delta_sum = 0.0
            command_delta_count = 0
            query_count = 0
            step_count = 0
            for episode_id, cache in episode_caches.items():
                targets = np.asarray(cache.current_targets, dtype=np.float64)
                force_norms = np.asarray(cache.current_force_norms, dtype=np.float64)
                replay = replay_action_chunks(chunk_getter(cache), spec)
                row = _episode_replay_row(
                    episode_id=episode_id,
                    model_name=model_name,
                    spec=spec,
                    replay=replay,
                    targets=targets,
                    force_norms=force_norms,
                )
                per_episode.append(row)
                episode_l1_values.append(float(row["action_l1_physical"]))
                overall.add(replay.actions, targets)
                stage_indices = classify_contact_stages(force_norms)
                for index, name in enumerate(CONTACT_STAGE_NAMES):
                    selected = stage_indices == index
                    if selected.any():
                        stages[name].add(replay.actions[selected], targets[selected])
                deltas = np.diff(replay.actions, axis=0)
                if deltas.size:
                    command_delta_sum += float(
                        np.linalg.norm(deltas, axis=1).sum()
                    )
                    command_delta_count += int(deltas.shape[0])
                prediction_ages.append(replay.prediction_ages)
                query_count += int(replay.policy_queried.sum())
                step_count += int(replay.actions.shape[0])
            all_ages = np.concatenate(prediction_ages)
            aggregate.append(
                {
                    "model": model_name,
                    "executor_id": spec.executor_id,
                    "executor_family": spec.family,
                    "decay": spec.decay,
                    "signed_decay": (
                        spec.decay if spec.family == "signed_temporal" else None
                    ),
                    "temporal_endpoint": spec.endpoint,
                    "query_interval": spec.query_interval,
                    "episode_count": len(episode_l1_values),
                    "timestep_count": step_count,
                    "action_l1_physical_global": overall.l1,
                    "action_step_l2_physical_global": overall.step_l2,
                    "action_l1_physical_episode_mean": mean(episode_l1_values),
                    "action_l1_physical_episode_median": median(episode_l1_values),
                    "command_step_delta_l2_mean": (
                        command_delta_sum / command_delta_count
                        if command_delta_count
                        else 0.0
                    ),
                    "prediction_age_mean_steps": float(all_ages.mean()),
                    "prediction_age_p95_steps": float(
                        np.quantile(all_ages, 0.95)
                    ),
                    "prediction_age_max_steps": float(all_ages.max()),
                    "policy_query_count": query_count,
                    "policy_query_fraction": query_count / step_count,
                    **{
                        f"{name}_l1_physical_global": stages[name].l1
                        for name in CONTACT_STAGE_NAMES
                    },
                    **{
                        f"{name}_steps": stages[name].step_count
                        for name in CONTACT_STAGE_NAMES
                    },
                }
            )
    baseline_by_model = {
        row["model"]: float(row["action_l1_physical_global"])
        for row in aggregate
        if row["executor_id"] == reference_baseline_id
    }
    if set(baseline_by_model) != {"official", "contact"}:
        raise ValueError(
            f"sweep must include the baseline {reference_baseline_id!r}"
        )
    for row in aggregate:
        baseline = baseline_by_model[row["model"]]
        current = float(row["action_l1_physical_global"])
        row["l1_delta_from_official_k0p01"] = current - baseline
        row["l1_relative_improvement_from_official_k0p01"] = (
            (baseline - current) / baseline
            if baseline > 0.0
            else (0.0 if current == 0.0 else None)
        )
        row["l1_delta_from_reference_baseline"] = current - baseline
        row["l1_relative_improvement_from_reference_baseline"] = row[
            "l1_relative_improvement_from_official_k0p01"
        ]
    return aggregate, per_episode


def build_rankings(
    aggregate_rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    rankings: dict[str, list[dict[str, Any]]] = {}
    for model_name in ("official", "contact"):
        ranked = sorted(
            (row for row in aggregate_rows if row["model"] == model_name),
            key=lambda row: float(row["action_l1_physical_global"]),
        )
        rankings[model_name] = [
            {
                "rank": index,
                "executor_id": row["executor_id"],
                "action_l1_physical_global": row["action_l1_physical_global"],
                "l1_relative_improvement_from_official_k0p01": row[
                    "l1_relative_improvement_from_official_k0p01"
                ],
                "prediction_age_mean_steps": row["prediction_age_mean_steps"],
                "policy_query_fraction": row["policy_query_fraction"],
            }
            for index, row in enumerate(ranked, start=1)
        ]
    rows_by_executor: dict[str, dict[str, Mapping[str, Any]]] = {}
    for row in aggregate_rows:
        rows_by_executor.setdefault(str(row["executor_id"]), {})[
            str(row["model"])
        ] = row
    paired = []
    for executor_id, model_rows in rows_by_executor.items():
        if set(model_rows) != {"official", "contact"}:
            raise ValueError(f"executor {executor_id!r} lacks a paired model result")
        improvements = []
        for name in ("official", "contact"):
            value = model_rows[name][
                "l1_relative_improvement_from_official_k0p01"
            ]
            if value is None:
                raise ValueError(
                    "paired ranking is undefined when a zero-error baseline "
                    "is compared with a nonzero-error executor"
                )
            improvements.append(float(value))
        paired.append(
            {
                "executor_id": executor_id,
                "official_action_l1_physical_global": model_rows["official"][
                    "action_l1_physical_global"
                ],
                "contact_action_l1_physical_global": model_rows["contact"][
                    "action_l1_physical_global"
                ],
                "mean_relative_improvement": mean(improvements),
                "worst_model_relative_improvement": min(improvements),
                "prediction_age_mean_steps": model_rows["official"][
                    "prediction_age_mean_steps"
                ],
                "policy_query_fraction": model_rows["official"][
                    "policy_query_fraction"
                ],
            }
        )
    paired.sort(
        key=lambda row: (
            float(row["mean_relative_improvement"]),
            float(row["worst_model_relative_improvement"]),
        ),
        reverse=True,
    )
    for index, row in enumerate(paired, start=1):
        row["rank"] = index
    return rankings, paired


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    fieldnames = sorted({name for row in rows for name in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_sweep(args: argparse.Namespace) -> dict[str, Any]:
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
    if args.sweep_profile == "legacy":
        specs = build_executor_specs(
            args.official_decays,
            args.recency_decays,
            args.receding_intervals,
        )
    else:
        specs = build_signed_temporal_specs(args.signed_decays)
    reference_baseline_id = (
        OFFICIAL_BASELINE_ID
        if args.sweep_profile == "legacy"
        else SIGNED_OFFICIAL_BASELINE_ID
    )
    for spec in specs:
        if (
            spec.query_interval is not None
            and spec.query_interval > contact.config.chunk_len
        ):
            raise ValueError(
                f"query interval {spec.query_interval} exceeds chunk length "
                f"{contact.config.chunk_len}"
            )
    contact_stats = NormalizationStats.from_dict(dict(contact.normalization))
    dataset = ACTAlignedHighRateHDF5Dataset(
        args.data_root,
        subset.validation_episodes,
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
    episodes: dict[str, EpisodeCache] = {}
    cursor = 0
    start_time = time.monotonic()
    official_norm = official.normalization
    contact_norm = contact.normalization
    try:
        for batch_index, cpu_batch in enumerate(loader, start=1):
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
                raise RuntimeError("validation sweep did not use zero contact latent")
            if float(contact_output["z_contact"].abs().max().item()) != 0.0:
                raise RuntimeError("contact deployment latent is not exactly zero")
            official_actions = _denormalize(
                official_output["pred_action"],
                official_norm["action_mean"],
                official_norm["action_std"],
            )
            contact_actions = _denormalize(
                contact_output["pred_action"],
                contact_norm["action_mean"],
                contact_norm["action_std"],
            )
            current_force_norms = _latest_linear_force_norms(
                batch.online_force_intervals,
                batch.online_force_sample_padding_mask,
                contact_norm["force_mean"],
                contact_norm["force_std"],
            )
            for offset in range(batch.batch_size):
                episode_index, timestep = dataset.index[cursor + offset]
                record = subset.validation_episodes[episode_index]
                cache = episodes.setdefault(record.episode_id, EpisodeCache())
                if timestep != len(cache.current_targets):
                    raise RuntimeError(
                        "validation samples are not in episode-time order"
                    )
                cache.official_chunks.append(
                    official_actions[offset].detach().cpu().numpy()
                )
                cache.contact_chunks.append(
                    contact_actions[offset].detach().cpu().numpy()
                )
                cache.current_targets.append(
                    raw_action[offset, 0].detach().cpu().numpy()
                )
                cache.current_force_norms.append(
                    float(current_force_norms[offset].item())
                )
            cursor += batch.batch_size
            if batch_index % args.log_interval == 0 or cursor == sample_count:
                elapsed = time.monotonic() - start_time
                rate = cursor / max(elapsed, 1.0e-9)
                eta = (sample_count - cursor) / max(rate, 1.0e-9)
                print(
                    f"cached_samples={cursor}/{sample_count} "
                    f"rate={rate:.2f}/s eta_seconds={eta:.1f}",
                    flush=True,
                )
    finally:
        dataset.close()
    if cursor != sample_count:
        raise RuntimeError(f"cached {cursor} validation samples, expected {sample_count}")
    aggregate_rows, episode_rows = aggregate_replay_rows(
        episodes,
        specs,
        reference_baseline_id=reference_baseline_id,
    )
    rankings, paired_ranking = build_rankings(aggregate_rows)
    all_force_norms = np.concatenate(
        [
            np.asarray(cache.current_force_norms, dtype=np.float64)
            for cache in episodes.values()
        ]
    )
    all_stage_indices = classify_contact_stages(all_force_norms)
    stage_counts = {
        name: int((all_stage_indices == index).sum())
        for index, name in enumerate(CONTACT_STAGE_NAMES)
    }
    summary = {
        "sweep_version": SWEEP_VERSION,
        "evaluation_scope": (
            "paired50_manifest_validation_all_timesteps"
            if args.max_samples is None
            else "paired50_manifest_validation_prefix_smoke"
        ),
        "partial_evaluation": args.max_samples is not None,
        "teacher_forced": True,
        "closed_loop_rollout": False,
        "hyperparameter_selection_split": "validation",
        "holdout_used": False,
        "deployment_latents": {"official": "zero", "contact": "zero"},
        "prediction_cache": "in_memory_once_per_model_timestep",
        "sweep_profile": args.sweep_profile,
        "fixed_policy_query_interval": (
            1 if args.sweep_profile == "expanded_signed_temporal" else None
        ),
        "signed_decay_convention": (
            "weight=softmax(-signed_k*prediction_age); negative=old, "
            "zero=uniform, positive=new"
            if args.sweep_profile == "expanded_signed_temporal"
            else None
        ),
        "reference_baseline_executor_id": reference_baseline_id,
        "official_act_decay_mapping": (
            {"official_candidate_index_k": 0.01, "signed_age_k": -0.01}
            if args.sweep_profile == "expanded_signed_temporal"
            else None
        ),
        "model_forward_samples_per_model": cursor,
        "data_root": str(args.data_root.resolve()),
        "experiment_manifest": str(args.experiment_manifest.resolve()),
        "dataset_fingerprint": subset.dataset_fingerprint,
        "validation_episode_count": len(episodes),
        "validation_timestep_count": cursor,
        "contact_stage_counts": stage_counts,
        "batch_size": args.batch_size,
        "device": str(device),
        "official_checkpoint": str(args.official_checkpoint.resolve()),
        "contact_checkpoint": str(args.contact_checkpoint.resolve()),
        "executor_specs": [item.__dict__ for item in specs],
        "ranking_metric": "action_l1_physical_global",
        "rankings": rankings,
        "paired_executor_ranking": paired_ranking,
        "elapsed_seconds": time.monotonic() - start_time,
        "passed": True,
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    summary_path = args.output_dir / "summary.json"
    aggregate_path = args.output_dir / "aggregate.csv"
    episode_path = args.output_dir / "per_episode.csv"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(aggregate_path, aggregate_rows)
    _write_csv(episode_path, episode_rows)
    print(f"summary={summary_path.resolve()}")
    print(f"aggregate={aggregate_path.resolve()}")
    print(f"per_episode={episode_path.resolve()}")
    return summary


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_root", type=Path)
    parser.add_argument("--experiment-manifest", type=Path, required=True)
    parser.add_argument("--official-checkpoint", type=Path, required=True)
    parser.add_argument("--contact-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument(
        "--sweep-profile",
        choices=("legacy", "expanded_signed_temporal"),
        default="legacy",
        help=(
            "legacy reproduces the original mixed executor grid; "
            "expanded_signed_temporal fixes policy query interval to one and "
            "sweeps a signed prediction-age decay"
        ),
    )
    parser.add_argument(
        "--signed-decays",
        type=float,
        nargs="+",
        default=DEFAULT_SIGNED_DECAYS,
    )
    parser.add_argument(
        "--official-decays",
        type=float,
        nargs="+",
        default=(0.0, 0.01, 0.03),
    )
    parser.add_argument(
        "--recency-decays",
        type=float,
        nargs="+",
        default=(0.01, 0.03, 0.1),
    )
    parser.add_argument(
        "--receding-intervals",
        type=int,
        nargs="+",
        default=(1, 5, 10, 25, 100),
    )
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
    for name in ("official_decays", "recency_decays"):
        values = getattr(args, name)
        if not values or any(not math.isfinite(value) or value < 0 for value in values):
            parser.error(f"{name.replace('_', ' ')} must be finite and non-negative")
    if not args.signed_decays or any(
        not math.isfinite(value) for value in args.signed_decays
    ):
        parser.error("signed decays must be finite")
    if len(set(args.signed_decays)) != len(args.signed_decays):
        parser.error("signed decays must not contain duplicates")
    if (
        args.sweep_profile == "expanded_signed_temporal"
        and -0.01 not in args.signed_decays
    ):
        parser.error(
            "expanded signed temporal profile requires -0.01 as the official "
            "ACT reference baseline"
        )
    if not args.receding_intervals or any(
        value <= 0 for value in args.receding_intervals
    ):
        parser.error("receding intervals must be positive")
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        run_sweep(args)
    except Exception as error:
        print(f"error: validation action sweep failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
