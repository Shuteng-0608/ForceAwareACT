"""Epoch accounting, deployment validation, and early-stopping utilities."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import torch
from torch.utils.data import DataLoader

from force_aware_act.data import normalize_tensor


EARLY_STOP_METRICS = ("deploy_loss", "action_l1", "force_l1")
VALIDATION_DEPLOYMENT_MODES = ("auto", "zero", "prior")


def compute_steps_per_epoch(dataset_length: int, batch_size: int) -> int:
    """Return optimizer steps in one full pass with ``drop_last=False``."""

    if dataset_length <= 0:
        raise ValueError("dataset_length must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    return math.ceil(dataset_length / batch_size)


def canonical_episode_paths(paths: Sequence[Path | str]) -> set[Path]:
    """Return normalized absolute episode paths without requiring existence."""

    return {Path(path).expanduser().resolve(strict=False) for path in paths}


def validate_disjoint_episode_splits(
    train_paths: Sequence[Path | str],
    val_paths: Sequence[Path | str],
) -> None:
    """Reject episode leakage between training and validation splits."""

    overlap = canonical_episode_paths(train_paths) & canonical_episode_paths(val_paths)
    if overlap:
        preview = ", ".join(str(path) for path in sorted(overlap)[:5])
        raise ValueError(
            "training and validation episode lists overlap; "
            f"overlap_count={len(overlap)} examples=[{preview}]"
        )


def validate_normalization_training_episodes(
    stats: Optional[Mapping[str, Any]],
    train_paths: Sequence[Path | str],
) -> None:
    """Ensure recorded normalization inputs exactly match the training split.

    Legacy statistics without ``episode_paths`` remain supported because their
    provenance cannot be checked. Newly generated statistics always include it.
    """

    if stats is None or "episode_paths" not in stats:
        return
    recorded = stats["episode_paths"]
    if not isinstance(recorded, (list, tuple)):
        raise ValueError("normalization stats episode_paths must be a list or tuple")
    stats_paths = canonical_episode_paths(recorded)
    training_paths = canonical_episode_paths(train_paths)
    if stats_paths == training_paths:
        return
    outside_training = sorted(stats_paths - training_paths)
    missing_from_stats = sorted(training_paths - stats_paths)
    raise ValueError(
        "normalization stats episode_paths do not match the training split: "
        f"outside_training={len(outside_training)} "
        f"missing_from_stats={len(missing_from_stats)}"
    )


def resolve_validation_deployment_mode(
    *,
    policy_variant: str,
    requested_mode: str,
    train_latent_mode: str = "posterior",
    lambda_prior: float = 0.0,
) -> str:
    """Resolve the deterministic online latent path used for validation."""

    if requested_mode not in VALIDATION_DEPLOYMENT_MODES:
        raise ValueError(
            "validation deployment mode must be one of: "
            + ", ".join(VALIDATION_DEPLOYMENT_MODES)
        )
    if requested_mode != "auto":
        if requested_mode == "prior" and policy_variant in {
            "act_baseline",
            "force_aware_motion_cvae",
        }:
            raise ValueError(f"policy_variant={policy_variant!r} has no deployable contact prior")
        return requested_mode
    if policy_variant in {"act_baseline", "force_aware_motion_cvae"}:
        return "zero"
    if train_latent_mode == "posterior" and lambda_prior > 0:
        return "prior"
    return "zero"


@dataclass
class EarlyStoppingState:
    """State machine for relative-improvement early stopping."""

    patience: int = 8
    min_epochs: int = 10
    min_delta: float = 0.005
    best_metric: Optional[float] = None
    best_epoch: Optional[int] = None
    best_step: Optional[int] = None
    epochs_without_improvement: int = 0

    def __post_init__(self) -> None:
        if self.patience <= 0:
            raise ValueError("early-stop patience must be positive")
        if self.min_epochs < 0:
            raise ValueError("early-stop minimum epochs must be non-negative")
        if not 0.0 <= self.min_delta < 1.0:
            raise ValueError("early-stop min_delta must be in [0, 1)")

    def update(self, metric: float, *, epoch: int, step: int) -> tuple[bool, bool]:
        """Update state and return ``(improved, should_stop)``."""

        if not math.isfinite(metric):
            raise ValueError("early-stop metric must be finite")
        if epoch <= 0 or step <= 0:
            raise ValueError("epoch and step must be positive")

        improved = self.best_metric is None
        if self.best_metric is not None:
            required_change = abs(self.best_metric) * self.min_delta
            improved = metric < self.best_metric - required_change

        if improved:
            self.best_metric = metric
            self.best_epoch = epoch
            self.best_step = step
            self.epochs_without_improvement = 0
        elif epoch >= self.min_epochs:
            self.epochs_without_improvement += 1

        should_stop = (
            epoch >= self.min_epochs
            and not improved
            and self.epochs_without_improvement >= self.patience
        )
        return improved, should_stop

    def checkpoint_metadata(self) -> dict[str, object]:
        return {
            "best_metric": self.best_metric,
            "best_epoch": self.best_epoch,
            "best_step": self.best_step,
            "epochs_without_improvement": self.epochs_without_improvement,
        }


def _move_batch_to_device(
    batch: Mapping[str, object],
    device: torch.device,
) -> dict[str, object]:
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def _normalize_validation_batch(
    batch: Mapping[str, object],
    stats: Optional[Mapping[str, Any]],
    *,
    include_force: bool,
) -> dict[str, object]:
    normalized = dict(batch)
    if stats is None:
        return normalized
    normalized["qpos"] = normalize_tensor(
        normalized["qpos"], stats["qpos_mean"], stats["qpos_std"]
    )
    normalized["action_chunk"] = normalize_tensor(
        normalized["action_chunk"], stats["action_mean"], stats["action_std"]
    )
    if include_force:
        normalized["force_window"] = normalize_tensor(
            normalized["force_window"], stats["force_mean"], stats["force_std"]
        )
        normalized["future_force_chunk"] = normalize_tensor(
            normalized["future_force_chunk"], stats["force_mean"], stats["force_std"]
        )
    return normalized


def _run_deployment_forward(
    model: torch.nn.Module,
    batch: Mapping[str, object],
    *,
    policy_variant: str,
    deployment_mode: str,
) -> Mapping[str, torch.Tensor]:
    if policy_variant == "act_baseline":
        return model(
            images=batch["images"],
            qpos=batch["qpos"],
            action_chunk=None,
            is_training=False,
        )
    common_kwargs = {
        "images": batch["images"],
        "qpos": batch["qpos"],
        "force_window": batch["force_window"],
        "action_chunk": None,
        "future_force_chunk": None,
        "is_training": False,
    }
    if policy_variant == "force_aware_motion_cvae":
        return model(**common_kwargs)
    return model(
        **common_kwargs,
        contact_latent_mode=deployment_mode,
        deterministic_prior=True,
    )


def evaluate_deployment_metrics(
    *,
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    policy_variant: str,
    deployment_mode: str,
    normalization_stats: Optional[Mapping[str, Any]],
    lambda_force: float,
) -> dict[str, float]:
    """Evaluate deterministic online behavior over a complete validation loader."""

    include_force = policy_variant != "act_baseline"
    if lambda_force < 0:
        raise ValueError("lambda_force must be non-negative")
    module_training_states = [(module, module.training) for module in model.modules()]
    sample_count = 0
    action_sum = 0.0
    force_sum = 0.0
    model.eval()
    try:
        with torch.inference_mode():
            for raw_batch in dataloader:
                batch = _normalize_validation_batch(
                    raw_batch,
                    normalization_stats,
                    include_force=include_force,
                )
                batch = _move_batch_to_device(batch, device)
                outputs = _run_deployment_forward(
                    model,
                    batch,
                    policy_variant=policy_variant,
                    deployment_mode=deployment_mode,
                )
                action_target = batch["action_chunk"]
                action_per_sample = (outputs["pred_action"] - action_target).abs().flatten(1).mean(1)
                batch_size = int(action_per_sample.shape[0])
                sample_count += batch_size
                action_sum += float(action_per_sample.sum().cpu().item())
                if include_force:
                    force_target = batch["future_force_chunk"]
                    force_per_sample = (
                        (outputs["pred_force"] - force_target).abs().flatten(1).mean(1)
                    )
                    force_sum += float(force_per_sample.sum().cpu().item())
    finally:
        for module, training in module_training_states:
            module.training = training

    if sample_count == 0:
        raise ValueError("validation dataloader produced zero samples")
    action_l1 = action_sum / sample_count
    metrics = {
        "action_l1": action_l1,
        "num_samples": float(sample_count),
    }
    if include_force:
        force_l1 = force_sum / sample_count
        metrics["force_l1"] = force_l1
        metrics["deploy_loss"] = action_l1 + lambda_force * force_l1
    else:
        metrics["deploy_loss"] = action_l1
    for name, value in metrics.items():
        if not math.isfinite(value):
            raise ValueError(f"validation metric {name} is not finite")
    return metrics
