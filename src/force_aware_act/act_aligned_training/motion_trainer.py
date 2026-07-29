"""One-step training and zero-latent validation for the motion control."""

from __future__ import annotations

import math
from typing import Any, Dict

import torch

from force_aware_act.act_aligned_training.batch import ACTAlignedBatch
from force_aware_act.act_aligned_training.losses import masked_l1_loss
from force_aware_act.act_aligned_training.motion_config import (
    ACTAlignedMotionTrainingConfig,
)
from force_aware_act.act_aligned_training.motion_losses import (
    ACTAlignedMotionCriterion,
)
from force_aware_act.models.act_aligned.motion_policy import (
    ACTAlignedMotionCVAEControlPolicy,
)


def train_motion_one_step(
    model: ACTAlignedMotionCVAEControlPolicy,
    criterion: ACTAlignedMotionCriterion,
    optimizer: torch.optim.Optimizer,
    batch: ACTAlignedBatch,
    training_config: ACTAlignedMotionTrainingConfig,
) -> Dict[str, float]:
    """Run one optimizer step with the sampled action-only posterior."""

    _validate_training_objects(model, criterion, training_config)
    batch.validate(model.config)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    outputs = model.forward_train(
        batch.images,
        batch.qpos,
        batch.force_history,
        batch.action_chunk,
        force_padding_mask=batch.force_padding_mask,
        future_padding_mask=batch.future_padding_mask,
        sample_posterior=True,
    )
    losses = criterion(
        outputs,
        batch.action_chunk,
        batch.future_force_chunk,
        batch.future_padding_mask,
    )
    total = losses["loss_total"]
    if not torch.isfinite(total):
        raise FloatingPointError("training loss is not finite")
    total.backward()

    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    if not trainable_parameters:
        raise RuntimeError("training step produced no parameter gradients")
    if training_config.gradient_clip_norm is None:
        gradient_norm = _gradient_norm(trainable_parameters)
    else:
        gradient_norm_tensor = torch.nn.utils.clip_grad_norm_(
            trainable_parameters,
            max_norm=training_config.gradient_clip_norm,
        )
        gradient_norm = float(gradient_norm_tensor.item())
    if not math.isfinite(gradient_norm):
        raise FloatingPointError("gradient norm is not finite")
    optimizer.step()

    metrics = {
        name: float(value.detach().item())
        for name, value in losses.items()
    }
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
            "gradient_norm": gradient_norm,
            "main_learning_rate": _group_learning_rate(
                optimizer,
                "main",
            ),
            "backbone_learning_rate": _group_learning_rate(
                optimizer,
                "backbone",
            ),
        }
    )
    return metrics


def evaluate_motion_one_batch(
    model: ACTAlignedMotionCVAEControlPolicy,
    criterion: ACTAlignedMotionCriterion,
    batch: ACTAlignedBatch,
) -> Dict[str, Any]:
    """Compare posterior-mean reconstruction against true zero deployment."""

    if not isinstance(model, ACTAlignedMotionCVAEControlPolicy):
        raise TypeError(
            "model must be an ACTAlignedMotionCVAEControlPolicy"
        )
    if not isinstance(criterion, ACTAlignedMotionCriterion):
        raise TypeError("criterion must be an ACTAlignedMotionCriterion")
    batch.validate(model.config)
    previous_training_mode = model.training
    model.eval()
    try:
        with torch.no_grad():
            posterior_outputs = model.forward_train(
                batch.images,
                batch.qpos,
                batch.force_history,
                batch.action_chunk,
                force_padding_mask=batch.force_padding_mask,
                future_padding_mask=batch.future_padding_mask,
                sample_posterior=False,
            )
            posterior_losses = criterion(
                posterior_outputs,
                batch.action_chunk,
                batch.future_force_chunk,
                batch.future_padding_mask,
            )
            zero_outputs = model(
                batch.images,
                batch.qpos,
                batch.force_history,
                force_padding_mask=batch.force_padding_mask,
            )
            zero_action = masked_l1_loss(
                zero_outputs["pred_action"],
                batch.action_chunk,
                batch.future_padding_mask,
                name="deployment_zero_action",
            )
            zero_force = masked_l1_loss(
                zero_outputs["pred_force"],
                batch.future_force_chunk,
                batch.future_padding_mask,
                name="deployment_zero_force",
            )
            posterior_zero_action_delta = masked_l1_loss(
                posterior_outputs["pred_action"],
                zero_outputs["pred_action"],
                batch.future_padding_mask,
                name="posterior_zero_action_delta",
            )
            posterior_zero_force_delta = masked_l1_loss(
                posterior_outputs["pred_force"],
                zero_outputs["pred_force"],
                batch.future_padding_mask,
                name="posterior_zero_force_delta",
            )
            posterior_std_mean = torch.exp(
                0.5 * posterior_outputs["logvar_motion"]
            ).mean()
    finally:
        model.train(previous_training_mode)

    return {
        "posterior_total": float(posterior_losses["loss_total"].item()),
        "posterior_action_l1": float(
            posterior_losses["loss_action"].item()
        ),
        "posterior_force_l1": float(
            posterior_losses["loss_force"].item()
        ),
        "posterior_kl_standard": float(
            posterior_losses["loss_posterior_kl"].item()
        ),
        "posterior_mean_abs": float(
            posterior_outputs["mu_motion"].abs().mean().item()
        ),
        "posterior_std_mean": float(posterior_std_mean.item()),
        "deployment_zero_action_l1": float(zero_action.item()),
        "deployment_zero_force_l1": float(zero_force.item()),
        "posterior_zero_action_delta": float(
            posterior_zero_action_delta.item()
        ),
        "posterior_zero_force_delta": float(
            posterior_zero_force_delta.item()
        ),
        "_posterior_mean_sum": (
            posterior_outputs["mu_motion"]
            .detach()
            .double()
            .sum(dim=0)
            .cpu()
        ),
        "_posterior_mean_square_sum": (
            posterior_outputs["mu_motion"]
            .detach()
            .double()
            .square()
            .sum(dim=0)
            .cpu()
        ),
    }


def _validate_training_objects(
    model: ACTAlignedMotionCVAEControlPolicy,
    criterion: ACTAlignedMotionCriterion,
    training_config: ACTAlignedMotionTrainingConfig,
) -> None:
    if not isinstance(model, ACTAlignedMotionCVAEControlPolicy):
        raise TypeError(
            "model must be an ACTAlignedMotionCVAEControlPolicy"
        )
    if not isinstance(criterion, ACTAlignedMotionCriterion):
        raise TypeError("criterion must be an ACTAlignedMotionCriterion")
    if not isinstance(training_config, ACTAlignedMotionTrainingConfig):
        raise TypeError(
            "training_config must be an ACTAlignedMotionTrainingConfig"
        )
    if criterion.config != training_config:
        raise ValueError("criterion and training step must use the same config")


def _gradient_norm(parameters: list[torch.nn.Parameter]) -> float:
    squared_norm = 0.0
    for parameter in parameters:
        squared_norm += float(
            parameter.grad.detach().float().square().sum().item()
        )
    return math.sqrt(squared_norm)


def _group_learning_rate(
    optimizer: torch.optim.Optimizer,
    group_name: str,
) -> float:
    for group in optimizer.param_groups:
        if group.get("name") == group_name:
            return float(group["lr"])
    raise RuntimeError(f"optimizer is missing parameter group {group_name!r}")
