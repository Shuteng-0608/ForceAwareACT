"""Epoch loops for the independent motion-latent control experiment."""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, Optional

import torch

from force_aware_act.act_aligned_training.batch import ACTAlignedBatch
from force_aware_act.act_aligned_training.motion_config import (
    ACTAlignedMotionTrainingConfig,
)
from force_aware_act.act_aligned_training.motion_losses import (
    ACTAlignedMotionCriterion,
)
from force_aware_act.act_aligned_training.motion_trainer import (
    evaluate_motion_one_batch,
    train_motion_one_step,
)
from force_aware_act.models.act_aligned.motion_policy import (
    ACTAlignedMotionCVAEControlPolicy,
)


def run_motion_training_epoch(
    model: ACTAlignedMotionCVAEControlPolicy,
    criterion: ACTAlignedMotionCriterion,
    optimizer: torch.optim.Optimizer,
    batches: Iterable[ACTAlignedBatch],
    training_config: ACTAlignedMotionTrainingConfig,
    *,
    device: torch.device,
    max_optimizer_steps: Optional[int] = None,
    skip_batches: int = 0,
    step_callback: Optional[Callable[[int, Dict[str, float]], None]] = None,
) -> Dict[str, float]:
    """Train one epoch segment with exact mid-epoch resume support."""

    if max_optimizer_steps is not None:
        if (
            not isinstance(max_optimizer_steps, int)
            or isinstance(max_optimizer_steps, bool)
            or max_optimizer_steps <= 0
        ):
            raise ValueError("max_optimizer_steps must be positive or None")
    if (
        not isinstance(skip_batches, int)
        or isinstance(skip_batches, bool)
        or skip_batches < 0
    ):
        raise ValueError("skip_batches must be a non-negative integer")

    accumulator = _MotionEpochAccumulator(
        model.config,
        training_config,
    )
    batch_count = 0
    skipped_count = 0
    for batch_index, batch in enumerate(batches):
        if batch_index < skip_batches:
            skipped_count += 1
            continue
        if (
            max_optimizer_steps is not None
            and batch_count >= max_optimizer_steps
        ):
            break
        batch = batch.to(device)
        metrics = train_motion_one_step(
            model,
            criterion,
            optimizer,
            batch,
            training_config,
        )
        accumulator.add_training(batch, metrics)
        batch_count += 1
        if step_callback is not None:
            step_callback(batch_count, dict(metrics))
    if batch_count == 0:
        raise ValueError("training epoch requires at least one batch")
    result = accumulator.training_result()
    result["optimizer_steps"] = float(batch_count)
    result["batches_skipped"] = float(skipped_count)
    return result


def run_motion_validation_epoch(
    model: ACTAlignedMotionCVAEControlPolicy,
    criterion: ACTAlignedMotionCriterion,
    batches: Iterable[ACTAlignedBatch],
    training_config: ACTAlignedMotionTrainingConfig,
    *,
    device: torch.device,
) -> Dict[str, float]:
    """Aggregate posterior-mean and zero-deployment diagnostics."""

    accumulator = _MotionEpochAccumulator(
        model.config,
        training_config,
    )
    batch_count = 0
    for batch in batches:
        batch = batch.to(device)
        metrics = evaluate_motion_one_batch(model, criterion, batch)
        accumulator.add_validation(batch, metrics)
        batch_count += 1
    if batch_count == 0:
        raise ValueError("validation epoch requires at least one batch")
    return accumulator.validation_result()


class _MotionEpochAccumulator:
    def __init__(self, model_config, training_config) -> None:
        self.model_config = model_config
        self.training_config = training_config
        self.weighted_sums: Dict[str, float] = {}
        self.weights: Dict[str, float] = {}
        self.latent_moments: Dict[str, torch.Tensor] = {}
        self.latent_sample_count = 0

    def add_training(
        self,
        batch: ACTAlignedBatch,
        metrics: Dict[str, Any],
    ) -> None:
        valid_steps = float((~batch.future_padding_mask).sum().item())
        batch_size = float(batch.batch_size)
        self._add(
            "loss_action",
            metrics["loss_action"],
            valid_steps * self.model_config.action_dim,
        )
        self._add(
            "loss_force",
            metrics["loss_force"],
            valid_steps * self.model_config.force_dim,
        )
        self._add(
            "loss_posterior_kl",
            metrics["loss_posterior_kl"],
            batch_size,
        )
        self._add("gradient_norm", metrics["gradient_norm"], 1.0)
        self._add(
            "posterior_mean_abs",
            metrics["posterior_mean_abs"],
            batch_size,
        )
        self._add(
            "posterior_std_mean",
            metrics["posterior_std_mean"],
            batch_size,
        )

    def add_validation(
        self,
        batch: ACTAlignedBatch,
        metrics: Dict[str, Any],
    ) -> None:
        valid_steps = float((~batch.future_padding_mask).sum().item())
        batch_size = float(batch.batch_size)
        action_weight = valid_steps * self.model_config.action_dim
        force_weight = valid_steps * self.model_config.force_dim
        for name in (
            "posterior_action_l1",
            "deployment_zero_action_l1",
            "posterior_zero_action_delta",
        ):
            self._add(name, metrics[name], action_weight)
        for name in (
            "posterior_force_l1",
            "deployment_zero_force_l1",
            "posterior_zero_force_delta",
        ):
            self._add(name, metrics[name], force_weight)
        for name in (
            "posterior_kl_standard",
            "posterior_mean_abs",
            "posterior_std_mean",
        ):
            self._add(name, metrics[name], batch_size)
        for name in (
            "_posterior_mean_sum",
            "_posterior_mean_square_sum",
        ):
            value = metrics[name]
            if not isinstance(value, torch.Tensor) or value.ndim != 1:
                raise RuntimeError(f"{name} must be a latent vector")
            if name not in self.latent_moments:
                self.latent_moments[name] = torch.zeros_like(
                    value,
                    dtype=torch.float64,
                    device="cpu",
                )
            self.latent_moments[name] += value.to(
                dtype=torch.float64,
                device="cpu",
            )
        self.latent_sample_count += batch.batch_size

    def training_result(self) -> Dict[str, float]:
        action = self._mean("loss_action")
        force = self._mean("loss_force")
        posterior_kl = self._mean("loss_posterior_kl")
        total = (
            self.training_config.action_loss_weight * action
            + self.training_config.force_loss_weight * force
            + self.training_config.posterior_kl_weight * posterior_kl
        )
        return {
            "loss_total": total,
            "loss_action": action,
            "loss_force": force,
            "loss_posterior_kl": posterior_kl,
            "gradient_norm": self._mean("gradient_norm"),
            "posterior_mean_abs": self._mean("posterior_mean_abs"),
            "posterior_std_mean": self._mean("posterior_std_mean"),
        }

    def validation_result(self) -> Dict[str, float]:
        names = (
            "posterior_action_l1",
            "posterior_force_l1",
            "posterior_kl_standard",
            "posterior_mean_abs",
            "posterior_std_mean",
            "deployment_zero_action_l1",
            "deployment_zero_force_l1",
            "posterior_zero_action_delta",
            "posterior_zero_force_delta",
        )
        result = {name: self._mean(name) for name in names}
        result["posterior_mean_across_sample_variance"] = (
            self._latent_variance()
        )
        result["posterior_total"] = (
            self.training_config.action_loss_weight
            * result["posterior_action_l1"]
            + self.training_config.force_loss_weight
            * result["posterior_force_l1"]
            + self.training_config.posterior_kl_weight
            * result["posterior_kl_standard"]
        )
        return result

    def _add(self, name: str, value: float, weight: float) -> None:
        self.weighted_sums[name] = (
            self.weighted_sums.get(name, 0.0) + value * weight
        )
        self.weights[name] = self.weights.get(name, 0.0) + weight

    def _mean(self, name: str) -> float:
        if self.weights.get(name, 0.0) <= 0:
            raise RuntimeError(f"metric {name!r} has no observations")
        return self.weighted_sums[name] / self.weights[name]

    def _latent_variance(self) -> float:
        if self.latent_sample_count <= 0:
            raise RuntimeError("latent variance requires validation samples")
        mean = (
            self.latent_moments["_posterior_mean_sum"]
            / self.latent_sample_count
        )
        mean_square = (
            self.latent_moments["_posterior_mean_square_sum"]
            / self.latent_sample_count
        )
        variance = (mean_square - mean.square()).clamp_min(0.0)
        return float(variance.mean().item())
