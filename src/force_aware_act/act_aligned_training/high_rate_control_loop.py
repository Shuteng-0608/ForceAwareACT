"""Epoch loops for native-rate motion and latent-free controls."""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, Optional

import torch

from force_aware_act.act_aligned_training.high_rate_batch import (
    ACTAlignedHighRateBatch,
)
from force_aware_act.act_aligned_training.high_rate_control_trainer import (
    evaluate_high_rate_dual_zero_one_batch,
    evaluate_high_rate_motion_one_batch,
    train_high_rate_dual_zero_one_step,
    train_high_rate_motion_one_step,
)


def _run_training_epoch(
    model,
    criterion,
    optimizer,
    batches: Iterable[ACTAlignedHighRateBatch],
    training_config,
    *,
    device: torch.device,
    train_step,
    accumulator,
    max_optimizer_steps: Optional[int] = None,
    skip_batches: int = 0,
    step_callback: Optional[Callable[[int, Dict[str, float]], None]] = None,
) -> Dict[str, float]:
    if max_optimizer_steps is not None and (
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
    count = 0
    skipped = 0
    for batch_index, batch in enumerate(batches):
        if batch_index < skip_batches:
            skipped += 1
            continue
        if max_optimizer_steps is not None and count >= max_optimizer_steps:
            break
        batch = batch.to(device)
        metrics = train_step(
            model,
            criterion,
            optimizer,
            batch,
            training_config,
        )
        accumulator.add_training(batch, metrics)
        count += 1
        if step_callback is not None:
            step_callback(count, dict(metrics))
    if count == 0:
        raise ValueError("training epoch requires at least one batch")
    result = accumulator.training_result()
    result["optimizer_steps"] = float(count)
    result["batches_skipped"] = float(skipped)
    return result


def run_high_rate_motion_training_epoch(
    model,
    criterion,
    optimizer,
    batches,
    training_config,
    *,
    device,
    max_optimizer_steps=None,
    skip_batches=0,
    step_callback=None,
):
    return _run_training_epoch(
        model,
        criterion,
        optimizer,
        batches,
        training_config,
        device=device,
        train_step=train_high_rate_motion_one_step,
        accumulator=_HighRateMotionAccumulator(training_config),
        max_optimizer_steps=max_optimizer_steps,
        skip_batches=skip_batches,
        step_callback=step_callback,
    )


def run_high_rate_dual_zero_training_epoch(
    model,
    criterion,
    optimizer,
    batches,
    training_config,
    *,
    device,
    max_optimizer_steps=None,
    skip_batches=0,
    step_callback=None,
):
    return _run_training_epoch(
        model,
        criterion,
        optimizer,
        batches,
        training_config,
        device=device,
        train_step=train_high_rate_dual_zero_one_step,
        accumulator=_HighRateDualZeroAccumulator(training_config),
        max_optimizer_steps=max_optimizer_steps,
        skip_batches=skip_batches,
        step_callback=step_callback,
    )


def _run_validation_epoch(
    model,
    criterion,
    batches,
    *,
    device,
    evaluate_batch,
    accumulator,
) -> Dict[str, float]:
    count = 0
    for batch in batches:
        batch = batch.to(device)
        accumulator.add_validation(
            batch,
            evaluate_batch(model, criterion, batch),
        )
        count += 1
    if count == 0:
        raise ValueError("validation epoch requires at least one batch")
    return accumulator.validation_result()


def run_high_rate_motion_validation_epoch(
    model,
    criterion,
    batches,
    training_config,
    *,
    device,
):
    return _run_validation_epoch(
        model,
        criterion,
        batches,
        device=device,
        evaluate_batch=evaluate_high_rate_motion_one_batch,
        accumulator=_HighRateMotionAccumulator(training_config),
    )


def run_high_rate_dual_zero_validation_epoch(
    model,
    criterion,
    batches,
    training_config,
    *,
    device,
):
    return _run_validation_epoch(
        model,
        criterion,
        batches,
        device=device,
        evaluate_batch=evaluate_high_rate_dual_zero_one_batch,
        accumulator=_HighRateDualZeroAccumulator(training_config),
    )


class _WeightedAccumulator:
    def __init__(self, training_config) -> None:
        self.training_config = training_config
        self.sums: Dict[str, float] = {}
        self.weights: Dict[str, float] = {}

    def _add(self, name: str, value: float, weight: float) -> None:
        if weight <= 0:
            return
        self.sums[name] = self.sums.get(name, 0.0) + value * weight
        self.weights[name] = self.weights.get(name, 0.0) + weight

    def _mean(self, name: str) -> float:
        if self.weights.get(name, 0.0) <= 0:
            raise RuntimeError(f"metric {name!r} has no observations")
        return self.sums[name] / self.weights[name]

    @staticmethod
    def _weights(batch: ACTAlignedHighRateBatch) -> tuple[float, float, float]:
        return (
            float((~batch.action_padding_mask).sum().item()),
            float((~batch.future_force_interval_padding_mask).sum().item()),
            float(batch.batch_size),
        )


class _HighRateMotionAccumulator(_WeightedAccumulator):
    def __init__(self, training_config) -> None:
        super().__init__(training_config)
        self.moment_sum: Optional[torch.Tensor] = None
        self.moment_square_sum: Optional[torch.Tensor] = None
        self.latent_count = 0

    def add_training(self, batch, metrics: Dict[str, Any]) -> None:
        action_weight, force_weight, batch_weight = self._weights(batch)
        self._add("loss_action", metrics["loss_action"], action_weight)
        self._add("loss_force", metrics["loss_force"], force_weight)
        self._add(
            "loss_force_highrate",
            metrics["loss_force_highrate"],
            force_weight,
        )
        for name in (
            "loss_posterior_kl",
            "posterior_mean_abs",
            "posterior_std_mean",
        ):
            self._add(name, metrics[name], batch_weight)
        self._add("gradient_norm", metrics["gradient_norm"], 1.0)

    def training_result(self) -> Dict[str, float]:
        result = {
            name: self._mean(name)
            for name in (
                "loss_action",
                "loss_force",
                "loss_force_highrate",
                "loss_posterior_kl",
                "posterior_mean_abs",
                "posterior_std_mean",
                "gradient_norm",
            )
        }
        result["loss_total"] = (
            self.training_config.action_loss_weight * result["loss_action"]
            + self.training_config.force_loss_weight * result["loss_force"]
            + self.training_config.high_rate_force_loss_weight
            * result["loss_force_highrate"]
            + self.training_config.posterior_kl_weight
            * result["loss_posterior_kl"]
        )
        return result

    def add_validation(self, batch, metrics: Dict[str, Any]) -> None:
        action_weight, force_weight, batch_weight = self._weights(batch)
        for name in (
            "posterior_action_l1",
            "deployment_zero_action_l1",
            "posterior_zero_action_delta",
        ):
            self._add(name, metrics[name], action_weight)
        for name in (
            "posterior_force_l1",
            "posterior_force_highrate_l1",
            "deployment_zero_force_l1",
            "deployment_zero_force_highrate_l1",
            "posterior_zero_force_delta",
            "posterior_zero_force_highrate_delta",
        ):
            self._add(name, metrics[name], force_weight)
        for name in (
            "posterior_kl_standard",
            "posterior_mean_abs",
            "posterior_std_mean",
        ):
            self._add(name, metrics[name], batch_weight)
        mean_sum = metrics["_posterior_mean_sum"].to(torch.float64)
        square_sum = metrics["_posterior_mean_square_sum"].to(torch.float64)
        if self.moment_sum is None:
            self.moment_sum = torch.zeros_like(mean_sum)
            self.moment_square_sum = torch.zeros_like(square_sum)
        self.moment_sum += mean_sum
        self.moment_square_sum += square_sum
        self.latent_count += batch.batch_size

    def validation_result(self) -> Dict[str, float]:
        result = {name: self._mean(name) for name in self.weights}
        if self.moment_sum is None or self.moment_square_sum is None:
            raise RuntimeError("motion validation requires posterior moments")
        mean = self.moment_sum / self.latent_count
        mean_square = self.moment_square_sum / self.latent_count
        result["posterior_mean_across_sample_variance"] = float(
            (mean_square - mean.square()).clamp_min(0.0).mean().item()
        )
        result["posterior_total"] = (
            self.training_config.action_loss_weight
            * result["posterior_action_l1"]
            + self.training_config.force_loss_weight
            * result["posterior_force_l1"]
            + self.training_config.high_rate_force_loss_weight
            * result["posterior_force_highrate_l1"]
            + self.training_config.posterior_kl_weight
            * result["posterior_kl_standard"]
        )
        return result


class _HighRateDualZeroAccumulator(_WeightedAccumulator):
    def add_training(self, batch, metrics: Dict[str, Any]) -> None:
        action_weight, force_weight, _batch_weight = self._weights(batch)
        self._add("loss_action", metrics["loss_action"], action_weight)
        self._add("loss_force", metrics["loss_force"], force_weight)
        self._add(
            "loss_force_highrate",
            metrics["loss_force_highrate"],
            force_weight,
        )
        self._add("gradient_norm", metrics["gradient_norm"], 1.0)

    def training_result(self) -> Dict[str, float]:
        result = {
            name: self._mean(name)
            for name in (
                "loss_action",
                "loss_force",
                "loss_force_highrate",
                "gradient_norm",
            )
        }
        result["loss_total"] = (
            self.training_config.action_loss_weight * result["loss_action"]
            + self.training_config.force_loss_weight * result["loss_force"]
            + self.training_config.high_rate_force_loss_weight
            * result["loss_force_highrate"]
        )
        return result

    def add_validation(self, batch, metrics: Dict[str, Any]) -> None:
        action_weight, force_weight, _batch_weight = self._weights(batch)
        self._add(
            "deployment_action_l1",
            metrics["deployment_action_l1"],
            action_weight,
        )
        for name in (
            "deployment_force_l1",
            "deployment_force_highrate_l1",
        ):
            self._add(name, metrics[name], force_weight)

    def validation_result(self) -> Dict[str, float]:
        result = {name: self._mean(name) for name in self.weights}
        result["deployment_total"] = (
            self.training_config.action_loss_weight
            * result["deployment_action_l1"]
            + self.training_config.force_loss_weight
            * result["deployment_force_l1"]
            + self.training_config.high_rate_force_loss_weight
            * result["deployment_force_highrate_l1"]
        )
        return result
