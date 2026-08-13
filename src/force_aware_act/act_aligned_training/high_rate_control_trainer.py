"""Training steps for native-rate motion and latent-free controls."""

from __future__ import annotations

import math
from typing import Any, Dict

import torch

from force_aware_act.act_aligned_training.high_rate_batch import (
    ACTAlignedHighRateBatch,
)
from force_aware_act.act_aligned_training.high_rate_control_losses import (
    ACTAlignedHighRateDualZeroCriterion,
    ACTAlignedHighRateMotionCriterion,
)
from force_aware_act.act_aligned_training.high_rate_dual_zero_config import (
    ACTAlignedHighRateDualZeroTrainingConfig,
)
from force_aware_act.act_aligned_training.high_rate_losses import (
    masked_interval_balanced_high_rate_l1_loss,
    masked_optional_l1_loss,
)
from force_aware_act.act_aligned_training.high_rate_motion_config import (
    ACTAlignedHighRateMotionTrainingConfig,
)
from force_aware_act.act_aligned_training.losses import masked_l1_loss
from force_aware_act.models.act_aligned.high_rate_dual_zero_policy import (
    ACTAlignedHighRateDualZeroPolicy,
)
from force_aware_act.models.act_aligned.high_rate_motion_policy import (
    ACTAlignedHighRateMotionCVAEPolicy,
)


def _online_inputs(batch: ACTAlignedHighRateBatch) -> tuple[torch.Tensor, ...]:
    return (
        batch.images,
        batch.qpos,
        batch.online_force_intervals,
        batch.online_force_relative_time,
        batch.online_force_sample_padding_mask,
        batch.online_force_interval_padding_mask,
    )


def _criterion_inputs(batch: ACTAlignedHighRateBatch) -> tuple[torch.Tensor, ...]:
    return (
        batch.action_chunk,
        batch.future_force_target,
        batch.future_force_intervals,
        batch.action_padding_mask,
        batch.future_force_interval_padding_mask,
        batch.future_force_sample_padding_mask,
    )


def train_high_rate_motion_one_step(
    model: ACTAlignedHighRateMotionCVAEPolicy,
    criterion: ACTAlignedHighRateMotionCriterion,
    optimizer: torch.optim.Optimizer,
    batch: ACTAlignedHighRateBatch,
    training_config: ACTAlignedHighRateMotionTrainingConfig,
) -> Dict[str, float]:
    _validate_motion_objects(model, criterion, training_config)
    batch.validate(model.config)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    outputs = model.forward_train(
        *_online_inputs(batch),
        batch.action_chunk,
        action_padding_mask=batch.action_padding_mask,
        sample_posterior=True,
    )
    losses = criterion(outputs, *_criterion_inputs(batch))
    metrics = _backward_and_step(model, optimizer, losses, training_config)
    metrics.update(
        {
            "posterior_mean_abs": float(
                outputs["mu_motion"].detach().abs().mean().item()
            ),
            "posterior_std_mean": float(
                torch.exp(0.5 * outputs["logvar_motion"].detach())
                .mean()
                .item()
            ),
        }
    )
    return metrics


def evaluate_high_rate_motion_one_batch(
    model: ACTAlignedHighRateMotionCVAEPolicy,
    criterion: ACTAlignedHighRateMotionCriterion,
    batch: ACTAlignedHighRateBatch,
) -> Dict[str, Any]:
    if not isinstance(model, ACTAlignedHighRateMotionCVAEPolicy):
        raise TypeError("model must be ACTAlignedHighRateMotionCVAEPolicy")
    if not isinstance(criterion, ACTAlignedHighRateMotionCriterion):
        raise TypeError("criterion must be ACTAlignedHighRateMotionCriterion")
    batch.validate(model.config)
    previous_mode = model.training
    model.eval()
    try:
        with torch.no_grad():
            posterior = model.forward_train(
                *_online_inputs(batch),
                batch.action_chunk,
                action_padding_mask=batch.action_padding_mask,
                sample_posterior=False,
            )
            posterior_losses = criterion(
                posterior,
                *_criterion_inputs(batch),
            )
            zero = model(*_online_inputs(batch))
            result = {
                "posterior_total": float(
                    posterior_losses["loss_total"].item()
                ),
                "posterior_kl_standard": float(
                    posterior_losses["loss_posterior_kl"].item()
                ),
                "posterior_mean_abs": float(
                    posterior["mu_motion"].abs().mean().item()
                ),
                "posterior_std_mean": float(
                    torch.exp(0.5 * posterior["logvar_motion"])
                    .mean()
                    .item()
                ),
            }
            result.update(_prediction_metrics("posterior", posterior, batch))
            result.update(_prediction_metrics("deployment_zero", zero, batch))
            result.update(
                _prediction_deltas("posterior_zero", posterior, zero, batch)
            )
            mean = posterior["mu_motion"].detach()
            result["_posterior_mean_sum"] = mean.double().sum(dim=0).cpu()
            result["_posterior_mean_square_sum"] = (
                mean.double().square().sum(dim=0).cpu()
            )
    finally:
        model.train(previous_mode)
    return result


def train_high_rate_dual_zero_one_step(
    model: ACTAlignedHighRateDualZeroPolicy,
    criterion: ACTAlignedHighRateDualZeroCriterion,
    optimizer: torch.optim.Optimizer,
    batch: ACTAlignedHighRateBatch,
    training_config: ACTAlignedHighRateDualZeroTrainingConfig,
) -> Dict[str, float]:
    _validate_dual_zero_objects(model, criterion, training_config)
    batch.validate(model.config)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    outputs = model(*_online_inputs(batch))
    losses = criterion(outputs, *_criterion_inputs(batch))
    return _backward_and_step(model, optimizer, losses, training_config)


def evaluate_high_rate_dual_zero_one_batch(
    model: ACTAlignedHighRateDualZeroPolicy,
    criterion: ACTAlignedHighRateDualZeroCriterion,
    batch: ACTAlignedHighRateBatch,
) -> Dict[str, float]:
    if not isinstance(model, ACTAlignedHighRateDualZeroPolicy):
        raise TypeError("model must be ACTAlignedHighRateDualZeroPolicy")
    if not isinstance(criterion, ACTAlignedHighRateDualZeroCriterion):
        raise TypeError("criterion must be ACTAlignedHighRateDualZeroCriterion")
    batch.validate(model.config)
    previous_mode = model.training
    model.eval()
    try:
        with torch.no_grad():
            outputs = model(*_online_inputs(batch))
            losses = criterion(outputs, *_criterion_inputs(batch))
            result = {
                "deployment_total": float(losses["loss_total"].item()),
                **_prediction_metrics("deployment", outputs, batch),
            }
    finally:
        model.train(previous_mode)
    return result


def _prediction_metrics(
    prefix: str,
    outputs: Dict[str, Any],
    batch: ACTAlignedHighRateBatch,
) -> Dict[str, float]:
    return {
        f"{prefix}_action_l1": float(
            masked_l1_loss(
                outputs["pred_action"],
                batch.action_chunk,
                batch.action_padding_mask,
                name=f"{prefix}_action",
            ).item()
        ),
        f"{prefix}_force_l1": float(
            masked_optional_l1_loss(
                outputs["pred_force"],
                batch.future_force_target,
                batch.future_force_interval_padding_mask,
                name=f"{prefix}_force",
            ).item()
        ),
        f"{prefix}_force_highrate_l1": float(
            masked_interval_balanced_high_rate_l1_loss(
                outputs["pred_force_highrate"],
                batch.future_force_intervals,
                batch.future_force_sample_padding_mask,
                batch.future_force_interval_padding_mask,
                name=f"{prefix}_force_highrate",
            ).item()
        ),
    }


def _prediction_deltas(
    prefix: str,
    outputs: Dict[str, Any],
    reference: Dict[str, Any],
    batch: ACTAlignedHighRateBatch,
) -> Dict[str, float]:
    return {
        f"{prefix}_action_delta": float(
            masked_l1_loss(
                outputs["pred_action"],
                reference["pred_action"],
                batch.action_padding_mask,
                name=f"{prefix}_action_delta",
            ).item()
        ),
        f"{prefix}_force_delta": float(
            masked_optional_l1_loss(
                outputs["pred_force"],
                reference["pred_force"],
                batch.future_force_interval_padding_mask,
                name=f"{prefix}_force_delta",
            ).item()
        ),
        f"{prefix}_force_highrate_delta": float(
            masked_interval_balanced_high_rate_l1_loss(
                outputs["pred_force_highrate"],
                reference["pred_force_highrate"],
                batch.future_force_sample_padding_mask,
                batch.future_force_interval_padding_mask,
                name=f"{prefix}_force_highrate_delta",
            ).item()
        ),
    }


def _backward_and_step(model, optimizer, losses, training_config) -> Dict[str, float]:
    total = losses["loss_total"]
    if not torch.isfinite(total):
        raise FloatingPointError("training loss is not finite")
    total.backward()
    parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    if not parameters:
        raise RuntimeError("training step produced no parameter gradients")
    if training_config.gradient_clip_norm is None:
        gradient_norm = math.sqrt(
            sum(
                float(parameter.grad.detach().float().square().sum().item())
                for parameter in parameters
            )
        )
    else:
        gradient_norm = float(
            torch.nn.utils.clip_grad_norm_(
                parameters,
                training_config.gradient_clip_norm,
            ).item()
        )
    if not math.isfinite(gradient_norm):
        raise FloatingPointError("gradient norm is not finite")
    optimizer.step()
    metrics = {
        name: float(value.detach().item()) for name, value in losses.items()
    }
    metrics["gradient_norm"] = gradient_norm
    metrics["main_learning_rate"] = _group_learning_rate(optimizer, "main")
    metrics["backbone_learning_rate"] = _group_learning_rate(
        optimizer,
        "backbone",
    )
    return metrics


def _validate_motion_objects(model, criterion, config) -> None:
    if not isinstance(model, ACTAlignedHighRateMotionCVAEPolicy):
        raise TypeError("model must be ACTAlignedHighRateMotionCVAEPolicy")
    if not isinstance(criterion, ACTAlignedHighRateMotionCriterion):
        raise TypeError("criterion must be ACTAlignedHighRateMotionCriterion")
    if not isinstance(config, ACTAlignedHighRateMotionTrainingConfig):
        raise TypeError("config must be ACTAlignedHighRateMotionTrainingConfig")
    if criterion.config != config:
        raise ValueError("criterion and training step must use the same config")


def _validate_dual_zero_objects(model, criterion, config) -> None:
    if not isinstance(model, ACTAlignedHighRateDualZeroPolicy):
        raise TypeError("model must be ACTAlignedHighRateDualZeroPolicy")
    if not isinstance(criterion, ACTAlignedHighRateDualZeroCriterion):
        raise TypeError("criterion must be ACTAlignedHighRateDualZeroCriterion")
    if not isinstance(config, ACTAlignedHighRateDualZeroTrainingConfig):
        raise TypeError(
            "config must be ACTAlignedHighRateDualZeroTrainingConfig"
        )
    if criterion.config != config:
        raise ValueError("criterion and training step must use the same config")


def _group_learning_rate(optimizer, name: str) -> float:
    for group in optimizer.param_groups:
        if group.get("name") == name:
            return float(group["lr"])
    raise RuntimeError(f"optimizer is missing parameter group {name!r}")
