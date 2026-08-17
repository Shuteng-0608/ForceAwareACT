"""Epoch loops for native-rate force training."""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, Optional

import torch

from force_aware_act.act_aligned_training.high_rate_batch import ACTAlignedHighRateBatch
from force_aware_act.act_aligned_training.high_rate_config import ACTAlignedHighRateTrainingConfig
from force_aware_act.act_aligned_training.high_rate_losses import ACTAlignedHighRateCriterion
from force_aware_act.act_aligned_training.high_rate_trainer import (
    evaluate_high_rate_one_batch,
    train_high_rate_one_step,
)
from force_aware_act.models.act_aligned.high_rate_policy import ACTAlignedHighRateContactCVAEPolicy


def run_high_rate_training_epoch(
    model: ACTAlignedHighRateContactCVAEPolicy,
    criterion: ACTAlignedHighRateCriterion,
    optimizer: torch.optim.Optimizer,
    batches: Iterable[ACTAlignedHighRateBatch],
    training_config: ACTAlignedHighRateTrainingConfig,
    *,
    device: torch.device,
    max_optimizer_steps: Optional[int] = None,
    skip_batches: int = 0,
    step_callback: Optional[Callable[[int, Dict[str, float]], None]] = None,
) -> Dict[str, float]:
    if max_optimizer_steps is not None and max_optimizer_steps <= 0:
        raise ValueError("max_optimizer_steps must be positive or None")
    if skip_batches < 0:
        raise ValueError("skip_batches must be non-negative")
    accumulator = _HighRateEpochAccumulator(model.config, training_config)
    batch_count = 0
    skipped = 0
    for batch_index, batch in enumerate(batches):
        if batch_index < skip_batches:
            skipped += 1
            continue
        if max_optimizer_steps is not None and batch_count >= max_optimizer_steps:
            break
        batch = batch.to(device)
        metrics = train_high_rate_one_step(model, criterion, optimizer, batch, training_config)
        accumulator.add_training(batch, metrics)
        batch_count += 1
        if step_callback is not None:
            step_callback(batch_count, dict(metrics))
    if batch_count == 0:
        raise ValueError("training epoch requires at least one batch")
    result = accumulator.training_result()
    result["optimizer_steps"] = float(batch_count)
    result["batches_skipped"] = float(skipped)
    return result


def run_high_rate_validation_epoch(
    model: ACTAlignedHighRateContactCVAEPolicy,
    criterion: ACTAlignedHighRateCriterion,
    batches: Iterable[ACTAlignedHighRateBatch],
    training_config: ACTAlignedHighRateTrainingConfig,
    *,
    device: torch.device,
) -> Dict[str, float]:
    accumulator = _HighRateEpochAccumulator(model.config, training_config)
    count = 0
    for batch in batches:
        batch = batch.to(device)
        accumulator.add_validation(batch, evaluate_high_rate_one_batch(model, criterion, batch))
        count += 1
    if count == 0:
        raise ValueError("validation epoch requires at least one batch")
    return accumulator.validation_result()


def run_high_rate_physical_action_validation_epoch(
    model: ACTAlignedHighRateContactCVAEPolicy,
    batches: Iterable[ACTAlignedHighRateBatch],
    action_std: tuple[float, ...],
    *,
    device: torch.device,
) -> Dict[str, float]:
    """Measure zero-latent action L1 in physical joint-command units."""

    if len(action_std) != model.config.action_dim:
        raise ValueError("action_std dimension does not match the model")
    previous_mode = model.training
    model.eval()
    error_sum = 0.0
    valid_scalar_count = 0
    window_count = 0
    try:
        with torch.no_grad():
            for batch in batches:
                batch = batch.to(device)
                batch.validate(model.config)
                outputs = model(
                    batch.images,
                    batch.qpos,
                    batch.online_force_intervals,
                    batch.online_force_relative_time,
                    batch.online_force_sample_padding_mask,
                    batch.online_force_interval_padding_mask,
                    contact_latent_mode="zero",
                    deterministic_prior=True,
                )
                difference = (
                    outputs["pred_action"] - batch.action_chunk
                ).abs()
                difference = difference * difference.new_tensor(action_std)
                valid = (
                    (~batch.action_padding_mask)
                    .unsqueeze(-1)
                    .expand_as(difference)
                )
                error_sum += float(
                    difference.masked_select(valid).sum().item()
                )
                valid_scalar_count += int(valid.sum().item())
                window_count += batch.batch_size
    finally:
        model.train(previous_mode)
    if valid_scalar_count <= 0 or window_count <= 0:
        raise ValueError("physical action validation epoch is empty")
    return {
        "deployment_zero_action_l1_physical": (
            error_sum / valid_scalar_count
        ),
        "full_validation_windows": float(window_count),
        "full_validation_action_scalars": float(valid_scalar_count),
    }


class _HighRateEpochAccumulator:
    def __init__(self, model_config, training_config) -> None:
        self.model_config = model_config
        self.training_config = training_config
        self.sums: Dict[str, float] = {}
        self.weights: Dict[str, float] = {}
        self.moments: Dict[str, torch.Tensor] = {}
        self.latent_count = 0

    def _add(self, name: str, value: float, weight: float) -> None:
        if weight <= 0:
            return
        self.sums[name] = self.sums.get(name, 0.0) + value * weight
        self.weights[name] = self.weights.get(name, 0.0) + weight

    def _mean(self, name: str) -> float:
        if self.weights.get(name, 0.0) <= 0:
            raise RuntimeError(f"metric {name!r} has no observations")
        return self.sums[name] / self.weights[name]

    def add_training(self, batch: ACTAlignedHighRateBatch, metrics: Dict[str, Any]) -> None:
        action_intervals = float((~batch.action_padding_mask).sum().item())
        force_intervals = float((~batch.future_force_interval_padding_mask).sum().item())
        batch_size = float(batch.batch_size)
        self._add("loss_action", metrics["loss_action"], action_intervals)
        self._add("loss_force", metrics["loss_force"], force_intervals)
        self._add("loss_force_highrate", metrics["loss_force_highrate"], force_intervals)
        for name in ("loss_posterior_kl", "loss_prior_match", "posterior_mean_abs", "prior_mean_abs", "posterior_std_mean", "prior_std_mean", "posterior_prior_mean_l1"):
            self._add(name, metrics[name], batch_size)
        self._add("gradient_norm", metrics["gradient_norm"], 1.0)

    def add_validation(self, batch: ACTAlignedHighRateBatch, metrics: Dict[str, Any]) -> None:
        action_weight = float((~batch.action_padding_mask).sum().item())
        force_weight = float((~batch.future_force_interval_padding_mask).sum().item())
        batch_weight = float(batch.batch_size)
        for name in ("posterior_action_l1", "deployment_zero_action_l1", "deployment_prior_action_l1", "posterior_zero_action_delta", "prior_zero_action_delta"):
            self._add(name, metrics[name], action_weight)
        for name in ("posterior_force_l1", "posterior_force_highrate_l1", "deployment_zero_force_l1", "deployment_zero_force_highrate_l1", "deployment_prior_force_l1", "deployment_prior_force_highrate_l1", "posterior_zero_force_delta", "posterior_zero_force_highrate_delta", "prior_zero_force_delta", "prior_zero_force_highrate_delta"):
            self._add(name, metrics[name], force_weight)
        for name in ("posterior_kl_standard", "posterior_prior_match_kl", "posterior_prior_mean_l1", "posterior_std_mean", "prior_std_mean"):
            self._add(name, metrics[name], batch_weight)
        for name in ("_posterior_mean_sum", "_posterior_mean_square_sum", "_prior_mean_sum", "_prior_mean_square_sum"):
            value = metrics[name].to(dtype=torch.float64, device="cpu")
            self.moments[name] = self.moments.get(name, torch.zeros_like(value)) + value
        self.latent_count += batch.batch_size

    def training_result(self) -> Dict[str, float]:
        result = {name: self._mean(name) for name in ("loss_action", "loss_force", "loss_force_highrate", "loss_posterior_kl", "loss_prior_match", "gradient_norm", "posterior_mean_abs", "prior_mean_abs", "posterior_std_mean", "prior_std_mean", "posterior_prior_mean_l1")}
        result["loss_total"] = (
            self.training_config.action_loss_weight * result["loss_action"]
            + self.training_config.force_loss_weight * result["loss_force"]
            + self.training_config.high_rate_force_loss_weight * result["loss_force_highrate"]
            + self.training_config.posterior_kl_weight * result["loss_posterior_kl"]
            + self.training_config.prior_match_weight * result["loss_prior_match"]
        )
        return result

    def validation_result(self) -> Dict[str, float]:
        names = tuple(self.weights)
        result = {name: self._mean(name) for name in names}
        result["posterior_mean_across_sample_variance"] = self._variance("posterior")
        result["prior_mean_across_sample_variance"] = self._variance("prior")
        result["posterior_total"] = (
            self.training_config.action_loss_weight * result["posterior_action_l1"]
            + self.training_config.force_loss_weight * result["posterior_force_l1"]
            + self.training_config.high_rate_force_loss_weight * result["posterior_force_highrate_l1"]
            + self.training_config.posterior_kl_weight * result["posterior_kl_standard"]
            + self.training_config.prior_match_weight * result["posterior_prior_match_kl"]
        )
        return result

    def _variance(self, prefix: str) -> float:
        mean = self.moments[f"_{prefix}_mean_sum"] / self.latent_count
        square = self.moments[f"_{prefix}_mean_square_sum"] / self.latent_count
        return float((square - mean.square()).clamp_min(0.0).mean().item())
