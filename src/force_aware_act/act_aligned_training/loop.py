"""Multi-batch epoch loops for the independent ACT-aligned trainer."""

from __future__ import annotations

from typing import Callable, Dict, Iterable, Optional

import torch

from force_aware_act.act_aligned_training.batch import ACTAlignedBatch
from force_aware_act.act_aligned_training.config import ACTAlignedTrainingConfig
from force_aware_act.act_aligned_training.losses import ACTAlignedCriterion
from force_aware_act.act_aligned_training.trainer import (
    evaluate_one_batch,
    train_one_step,
)
from force_aware_act.models.act_aligned.policy import ACTAlignedContactCVAEPolicy


def run_training_epoch(
    model: ACTAlignedContactCVAEPolicy,
    criterion: ACTAlignedCriterion,
    optimizer: torch.optim.Optimizer,
    batches: Iterable[ACTAlignedBatch],
    training_config: ACTAlignedTrainingConfig,
    *,
    device: torch.device,
    max_optimizer_steps: Optional[int] = None,
    skip_batches: int = 0,
    step_callback: Optional[Callable[[int, Dict[str, float]], None]] = None,
) -> Dict[str, float]:
    """Train over one iterable and aggregate with correct denominators.

    ``skip_batches`` supports exact mid-epoch resume when the iterable's
    shuffle generator is restored to its epoch-start state. The callback
    receives the number of newly completed optimizer steps and their metrics.
    """

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

    accumulator = _EpochAccumulator(model.config, training_config)
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
        metrics = train_one_step(
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


def run_validation_epoch(
    model: ACTAlignedContactCVAEPolicy,
    criterion: ACTAlignedCriterion,
    batches: Iterable[ACTAlignedBatch],
    training_config: ACTAlignedTrainingConfig,
    *,
    device: torch.device,
) -> Dict[str, float]:
    """Aggregate posterior, zero, and prior validation over one iterable."""

    accumulator = _EpochAccumulator(model.config, training_config)
    batch_count = 0
    for batch in batches:
        batch = batch.to(device)
        metrics = evaluate_one_batch(model, criterion, batch)
        accumulator.add_validation(batch, metrics)
        batch_count += 1
    if batch_count == 0:
        raise ValueError("validation epoch requires at least one batch")
    return accumulator.validation_result()


class _EpochAccumulator:
    def __init__(self, model_config, training_config) -> None:
        self.model_config = model_config
        self.training_config = training_config
        self.weighted_sums: Dict[str, float] = {}
        self.weights: Dict[str, float] = {}

    def add_training(
        self,
        batch: ACTAlignedBatch,
        metrics: Dict[str, float],
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
        self._add(
            "loss_prior_match",
            metrics["loss_prior_match"],
            batch_size,
        )
        self._add("gradient_norm", metrics["gradient_norm"], 1.0)
        for name in (
            "posterior_mean_abs",
            "prior_mean_abs",
            "posterior_std_mean",
            "prior_std_mean",
            "posterior_prior_mean_l1",
        ):
            self._add(name, metrics[name], batch_size)

    def add_validation(
        self,
        batch: ACTAlignedBatch,
        metrics: Dict[str, float],
    ) -> None:
        valid_steps = float((~batch.future_padding_mask).sum().item())
        batch_size = float(batch.batch_size)
        action_weight = valid_steps * self.model_config.action_dim
        force_weight = valid_steps * self.model_config.force_dim
        for name in (
            "posterior_action_l1",
            "deployment_zero_action_l1",
            "deployment_prior_action_l1",
        ):
            self._add(name, metrics[name], action_weight)
        for name in (
            "posterior_force_l1",
            "deployment_zero_force_l1",
            "deployment_prior_force_l1",
        ):
            self._add(name, metrics[name], force_weight)
        self._add(
            "posterior_kl_standard",
            metrics["posterior_kl_standard"],
            batch_size,
        )
        self._add(
            "posterior_prior_match_kl",
            metrics["posterior_prior_match_kl"],
            batch_size,
        )
        for name in (
            "posterior_prior_mean_l1",
            "posterior_std_mean",
            "prior_std_mean",
        ):
            self._add(name, metrics[name], batch_size)

    def training_result(self) -> Dict[str, float]:
        action = self._mean("loss_action")
        force = self._mean("loss_force")
        posterior_kl = self._mean("loss_posterior_kl")
        prior_match = self._mean("loss_prior_match")
        total = (
            self.training_config.action_loss_weight * action
            + self.training_config.force_loss_weight * force
            + self.training_config.posterior_kl_weight * posterior_kl
            + self.training_config.prior_match_weight * prior_match
        )
        return {
            "loss_total": total,
            "loss_action": action,
            "loss_force": force,
            "loss_posterior_kl": posterior_kl,
            "loss_prior_match": prior_match,
            "gradient_norm": self._mean("gradient_norm"),
            "posterior_mean_abs": self._mean("posterior_mean_abs"),
            "prior_mean_abs": self._mean("prior_mean_abs"),
            "posterior_std_mean": self._mean("posterior_std_mean"),
            "prior_std_mean": self._mean("prior_std_mean"),
            "posterior_prior_mean_l1": self._mean(
                "posterior_prior_mean_l1"
            ),
        }

    def validation_result(self) -> Dict[str, float]:
        names = (
            "posterior_action_l1",
            "posterior_force_l1",
            "posterior_kl_standard",
            "posterior_prior_match_kl",
            "posterior_prior_mean_l1",
            "posterior_std_mean",
            "prior_std_mean",
            "deployment_zero_action_l1",
            "deployment_zero_force_l1",
            "deployment_prior_action_l1",
            "deployment_prior_force_l1",
        )
        result = {name: self._mean(name) for name in names}
        result["posterior_total"] = (
            self.training_config.action_loss_weight
            * result["posterior_action_l1"]
            + self.training_config.force_loss_weight
            * result["posterior_force_l1"]
            + self.training_config.posterior_kl_weight
            * result["posterior_kl_standard"]
            + self.training_config.prior_match_weight
            * result["posterior_prior_match_kl"]
        )
        return result

    def _add(self, name: str, value: float, weight: float) -> None:
        self.weighted_sums[name] = self.weighted_sums.get(name, 0.0) + value * weight
        self.weights[name] = self.weights.get(name, 0.0) + weight

    def _mean(self, name: str) -> float:
        if self.weights.get(name, 0.0) <= 0:
            raise RuntimeError(f"metric {name!r} has no observations")
        return self.weighted_sums[name] / self.weights[name]
