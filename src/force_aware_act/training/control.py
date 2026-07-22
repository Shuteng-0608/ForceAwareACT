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
VALIDATION_AGGREGATIONS = ("sample", "episode_uniform")


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
    aggregation: str = "sample",
) -> dict[str, float]:
    """Evaluate deterministic online behavior over a complete validation loader."""

    include_force = policy_variant != "act_baseline"
    if lambda_force < 0:
        raise ValueError("lambda_force must be non-negative")
    if aggregation not in VALIDATION_AGGREGATIONS:
        raise ValueError(
            "validation aggregation must be one of: "
            + ", ".join(VALIDATION_AGGREGATIONS)
        )
    module_training_states = [(module, module.training) for module in model.modules()]
    sample_count = 0
    action_sum = 0.0
    force_sum = 0.0
    episode_sums: dict[str, dict[str, float]] = {}
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
                action_values = action_per_sample.detach().cpu().tolist()
                action_sum += float(sum(action_values))
                episode_paths = None
                if aggregation == "episode_uniform":
                    episode_paths = batch.get("episode_path")
                    if not isinstance(episode_paths, (list, tuple)) or len(
                        episode_paths
                    ) != batch_size or any(
                        not isinstance(path, str) or not path for path in episode_paths
                    ):
                        raise ValueError(
                            "episode-uniform validation requires one non-empty "
                            "episode_path string per sample"
                        )
                    for path, action_value in zip(episode_paths, action_values):
                        accum = episode_sums.setdefault(
                            path, {"count": 0.0, "action": 0.0, "force": 0.0}
                        )
                        accum["count"] += 1.0
                        accum["action"] += float(action_value)
                if include_force:
                    force_target = batch["future_force_chunk"]
                    force_per_sample = (
                        (outputs["pred_force"] - force_target).abs().flatten(1).mean(1)
                    )
                    force_values = force_per_sample.detach().cpu().tolist()
                    force_sum += float(sum(force_values))
                    if aggregation == "episode_uniform":
                        assert episode_paths is not None
                        for path, force_value in zip(episode_paths, force_values):
                            episode_sums[path]["force"] += float(force_value)
    finally:
        for module, training in module_training_states:
            module.training = training

    if sample_count == 0:
        raise ValueError("validation dataloader produced zero samples")
    if aggregation == "episode_uniform":
        if not episode_sums:
            raise ValueError("episode-uniform validation found no episodes")
        action_l1 = sum(
            values["action"] / values["count"]
            for values in episode_sums.values()
        ) / len(episode_sums)
    else:
        action_l1 = action_sum / sample_count
    metrics = {
        "action_l1": action_l1,
        "num_samples": float(sample_count),
    }
    if aggregation == "episode_uniform":
        metrics["num_episodes"] = float(len(episode_sums))
    if include_force:
        if aggregation == "episode_uniform":
            force_l1 = sum(
                values["force"] / values["count"]
                for values in episode_sums.values()
            ) / len(episode_sums)
        else:
            force_l1 = force_sum / sample_count
        metrics["force_l1"] = force_l1
        metrics["deploy_loss"] = action_l1 + lambda_force * force_l1
    else:
        metrics["deploy_loss"] = action_l1
    for name, value in metrics.items():
        if not math.isfinite(value):
            raise ValueError(f"validation metric {name} is not finite")
    return metrics


def evaluate_named_deployment_metrics(
    *,
    model: torch.nn.Module,
    dataloaders: Mapping[str, DataLoader],
    device: torch.device,
    policy_variant: str,
    deployment_mode: str,
    normalization_stats: Optional[Mapping[str, Any]],
    lambda_force: float,
    aggregation: str = "sample",
) -> dict[str, dict[str, float]]:
    """Evaluate independent validation domains without merging their metrics.

    Domain names are preserved in the returned mapping so a staged-training
    controller can optimize a contact-rich validation split while separately
    enforcing retention on a broad-spatial split.
    """

    if not isinstance(dataloaders, Mapping) or not dataloaders:
        raise ValueError("named validation dataloaders must be a non-empty mapping")

    results: dict[str, dict[str, float]] = {}
    for domain, dataloader in dataloaders.items():
        if not isinstance(domain, str) or not domain or domain.strip() != domain:
            raise ValueError("validation domain names must be non-empty trimmed strings")
        if not isinstance(dataloader, DataLoader):
            raise ValueError(f"validation dataloader for domain {domain!r} is not a DataLoader")
        results[domain] = evaluate_deployment_metrics(
            model=model,
            dataloader=dataloader,
            device=device,
            policy_variant=policy_variant,
            deployment_mode=deployment_mode,
            normalization_stats=normalization_stats,
            lambda_force=lambda_force,
            aggregation=aggregation,
        )
    return results


def flatten_named_deployment_metrics(
    metrics_by_domain: Mapping[str, Mapping[str, float]],
    *,
    separator: str = "/",
) -> dict[str, float]:
    """Flatten named-domain metrics for CSV/TensorBoard-style logging."""

    if not separator:
        raise ValueError("metric separator must not be empty")
    flattened: dict[str, float] = {}
    for domain, metrics in metrics_by_domain.items():
        if not isinstance(domain, str) or not domain or domain.strip() != domain:
            raise ValueError("validation domain names must be non-empty trimmed strings")
        if not isinstance(metrics, Mapping) or not metrics:
            raise ValueError(f"metrics for validation domain {domain!r} must be non-empty")
        for metric_name, raw_value in metrics.items():
            if not isinstance(metric_name, str) or not metric_name:
                raise ValueError("validation metric names must be non-empty strings")
            value = float(raw_value)
            if not math.isfinite(value):
                raise ValueError(
                    f"validation metric {domain!r}/{metric_name!r} is not finite"
                )
            flattened[f"{domain}{separator}{metric_name}"] = value
    return flattened


@dataclass(frozen=True)
class RetentionSelectionDecision:
    """Result of considering one named-domain validation checkpoint."""

    selected: bool
    retention_passed: bool
    objective_improved: bool
    objective_value: float
    retention_value: float
    retention_limit: float
    epoch: int
    step: int
    reason: str


@dataclass
class RetentionGatedCheckpointSelector:
    """Select lower-is-better objective metrics subject to a retention gate.

    ``retention_baseline`` should come from the pre-finetuning checkpoint on
    the retention domain.  A candidate is eligible only when its retention
    metric is no greater than the baseline plus the configured relative and
    absolute allowances.
    """

    objective_domain: str
    retention_domain: str
    retention_baseline: float
    metric: str = "deploy_loss"
    max_relative_degradation: float = 0.0
    max_absolute_degradation: float = 0.0
    min_relative_improvement: float = 0.0
    best_objective_value: Optional[float] = None
    best_retention_value: Optional[float] = None
    best_epoch: Optional[int] = None
    best_step: Optional[int] = None

    def __post_init__(self) -> None:
        for label, value in (
            ("objective_domain", self.objective_domain),
            ("retention_domain", self.retention_domain),
            ("metric", self.metric),
        ):
            if not isinstance(value, str) or not value or value.strip() != value:
                raise ValueError(f"{label} must be a non-empty trimmed string")
        if self.objective_domain == self.retention_domain:
            raise ValueError("objective_domain and retention_domain must be different")
        if not math.isfinite(self.retention_baseline):
            raise ValueError("retention_baseline must be finite")
        if not math.isfinite(self.max_relative_degradation) or self.max_relative_degradation < 0:
            raise ValueError("max_relative_degradation must be finite and non-negative")
        if not math.isfinite(self.max_absolute_degradation) or self.max_absolute_degradation < 0:
            raise ValueError("max_absolute_degradation must be finite and non-negative")
        if not 0.0 <= self.min_relative_improvement < 1.0:
            raise ValueError("min_relative_improvement must be in [0, 1)")
        for label, value in (
            ("best_objective_value", self.best_objective_value),
            ("best_retention_value", self.best_retention_value),
        ):
            if value is not None and not math.isfinite(value):
                raise ValueError(f"{label} must be finite when provided")
        if (self.best_epoch is None) != (self.best_step is None):
            raise ValueError("best_epoch and best_step must either both be set or both be None")
        if self.best_epoch is not None and (self.best_epoch <= 0 or self.best_step <= 0):
            raise ValueError("best_epoch and best_step must be positive")

    @property
    def retention_limit(self) -> float:
        return (
            self.retention_baseline
            + abs(self.retention_baseline) * self.max_relative_degradation
            + self.max_absolute_degradation
        )

    def _metric_value(
        self,
        metrics_by_domain: Mapping[str, Mapping[str, float]],
        domain: str,
    ) -> float:
        if domain not in metrics_by_domain:
            raise ValueError(f"missing validation domain: {domain!r}")
        domain_metrics = metrics_by_domain[domain]
        if not isinstance(domain_metrics, Mapping):
            raise ValueError(f"metrics for validation domain {domain!r} must be a mapping")
        if self.metric not in domain_metrics:
            raise ValueError(
                f"validation domain {domain!r} is missing metric {self.metric!r}"
            )
        value = float(domain_metrics[self.metric])
        if not math.isfinite(value):
            raise ValueError(
                f"validation metric {domain!r}/{self.metric!r} must be finite"
            )
        return value

    def update(
        self,
        metrics_by_domain: Mapping[str, Mapping[str, float]],
        *,
        epoch: int,
        step: int,
    ) -> RetentionSelectionDecision:
        """Consider one checkpoint and return a fully explained decision."""

        if epoch <= 0 or step <= 0:
            raise ValueError("epoch and step must be positive")
        objective_value = self._metric_value(metrics_by_domain, self.objective_domain)
        retention_value = self._metric_value(metrics_by_domain, self.retention_domain)
        retention_passed = retention_value <= self.retention_limit

        objective_improved = self.best_objective_value is None
        if self.best_objective_value is not None:
            required_change = abs(self.best_objective_value) * self.min_relative_improvement
            objective_improved = objective_value < self.best_objective_value - required_change

        selected = retention_passed and objective_improved
        if selected:
            self.best_objective_value = objective_value
            self.best_retention_value = retention_value
            self.best_epoch = epoch
            self.best_step = step
            reason = "selected"
        elif not retention_passed:
            reason = "retention_gate_failed"
        else:
            reason = "objective_not_improved"

        return RetentionSelectionDecision(
            selected=selected,
            retention_passed=retention_passed,
            objective_improved=objective_improved,
            objective_value=objective_value,
            retention_value=retention_value,
            retention_limit=self.retention_limit,
            epoch=epoch,
            step=step,
            reason=reason,
        )

    def checkpoint_metadata(self) -> dict[str, object]:
        """Return selector configuration and current best checkpoint metadata."""

        return {
            "objective_domain": self.objective_domain,
            "retention_domain": self.retention_domain,
            "metric": self.metric,
            "retention_baseline": self.retention_baseline,
            "retention_limit": self.retention_limit,
            "max_relative_degradation": self.max_relative_degradation,
            "max_absolute_degradation": self.max_absolute_degradation,
            "min_relative_improvement": self.min_relative_improvement,
            "best_objective_value": self.best_objective_value,
            "best_retention_value": self.best_retention_value,
            "best_epoch": self.best_epoch,
            "best_step": self.best_step,
        }

    def state_dict(self) -> dict[str, object]:
        """Return a serializable selector state."""

        return {"version": 1, **self.checkpoint_metadata()}

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        """Restore progress while rejecting a mismatched gate configuration."""

        if not isinstance(state, Mapping) or state.get("version") != 1:
            raise ValueError("unsupported retention selector state")
        expected = {
            "objective_domain": self.objective_domain,
            "retention_domain": self.retention_domain,
            "metric": self.metric,
            "retention_baseline": self.retention_baseline,
            "max_relative_degradation": self.max_relative_degradation,
            "max_absolute_degradation": self.max_absolute_degradation,
            "min_relative_improvement": self.min_relative_improvement,
        }
        for key, current in expected.items():
            if state.get(key) != current:
                raise ValueError(
                    f"retention selector state {key} mismatch: "
                    f"checkpoint={state.get(key)!r} current={current!r}"
                )

        restored_values = {
            "best_objective_value": state.get("best_objective_value"),
            "best_retention_value": state.get("best_retention_value"),
            "best_epoch": state.get("best_epoch"),
            "best_step": state.get("best_step"),
        }
        candidate = RetentionGatedCheckpointSelector(
            objective_domain=self.objective_domain,
            retention_domain=self.retention_domain,
            retention_baseline=self.retention_baseline,
            metric=self.metric,
            max_relative_degradation=self.max_relative_degradation,
            max_absolute_degradation=self.max_absolute_degradation,
            min_relative_improvement=self.min_relative_improvement,
            **restored_values,
        )
        self.best_objective_value = candidate.best_objective_value
        self.best_retention_value = candidate.best_retention_value
        self.best_epoch = candidate.best_epoch
        self.best_step = candidate.best_step
