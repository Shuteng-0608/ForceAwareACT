"""One-step training and leakage-safe two-path validation."""

from __future__ import annotations

import math
from typing import Dict

import torch

from force_aware_act.act_aligned_training.batch import ACTAlignedBatch
from force_aware_act.act_aligned_training.config import ACTAlignedTrainingConfig
from force_aware_act.act_aligned_training.losses import (
    ACTAlignedCriterion,
    masked_l1_loss,
)
from force_aware_act.models.act_aligned.policy import (
    ACTAlignedContactCVAEPolicy,
)


def train_one_step(
    model: ACTAlignedContactCVAEPolicy,
    criterion: ACTAlignedCriterion,
    optimizer: torch.optim.Optimizer,
    batch: ACTAlignedBatch,
    training_config: ACTAlignedTrainingConfig,
) -> Dict[str, float]:
    """Run one complete ACT-style optimization step."""

    _validate_training_objects(model, criterion, training_config)
    batch.validate(model.config)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    outputs = model.forward_train(
        batch.images,
        batch.qpos,
        batch.force_history,
        batch.action_chunk,
        batch.future_force_chunk,
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
                outputs["mu_contact"].detach().abs().mean().item()
            ),
            "prior_mean_abs": float(
                outputs["mu_contact_prior"].detach().abs().mean().item()
            ),
            "posterior_std_mean": float(
                torch.exp(0.5 * outputs["logvar_contact"].detach())
                .mean()
                .item()
            ),
            "prior_std_mean": float(
                torch.exp(0.5 * outputs["logvar_contact_prior"].detach())
                .mean()
                .item()
            ),
            "posterior_prior_mean_l1": float(
                (
                    outputs["mu_contact"].detach()
                    - outputs["mu_contact_prior"].detach()
                )
                .abs()
                .mean()
                .item()
            ),
        }
    )
    metrics["gradient_norm"] = gradient_norm
    metrics["main_learning_rate"] = _group_learning_rate(optimizer, "main")
    metrics["backbone_learning_rate"] = _group_learning_rate(
        optimizer,
        "backbone",
    )
    return metrics


def evaluate_one_batch(
    model: ACTAlignedContactCVAEPolicy,
    criterion: ACTAlignedCriterion,
    batch: ACTAlignedBatch,
) -> Dict[str, float]:
    """Evaluate posterior reconstruction and true deployment in one batch.

    Future targets are passed to ``forward_train`` only. The deployment
    ``forward`` call receives online observations and masks exclusively.
    """

    if not isinstance(model, ACTAlignedContactCVAEPolicy):
        raise TypeError("model must be an ACTAlignedContactCVAEPolicy")
    if not isinstance(criterion, ACTAlignedCriterion):
        raise TypeError("criterion must be an ACTAlignedCriterion")
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
                batch.future_force_chunk,
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
                contact_latent_mode="zero",
                deterministic_prior=True,
            )
            prior_outputs = model(
                batch.images,
                batch.qpos,
                batch.force_history,
                force_padding_mask=batch.force_padding_mask,
                contact_latent_mode="prior",
                deterministic_prior=True,
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
            prior_action = masked_l1_loss(
                prior_outputs["pred_action"],
                batch.action_chunk,
                batch.future_padding_mask,
                name="deployment_prior_action",
            )
            prior_force = masked_l1_loss(
                prior_outputs["pred_force"],
                batch.future_force_chunk,
                batch.future_padding_mask,
                name="deployment_prior_force",
            )
            posterior_prior_mean_l1 = (
                posterior_outputs["mu_contact"]
                - posterior_outputs["mu_contact_prior"]
            ).abs().mean()
            posterior_std_mean = torch.exp(
                0.5 * posterior_outputs["logvar_contact"]
            ).mean()
            prior_std_mean = torch.exp(
                0.5 * posterior_outputs["logvar_contact_prior"]
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
        "posterior_prior_match_kl": float(
            posterior_losses["loss_prior_match"].item()
        ),
        "posterior_prior_mean_l1": float(posterior_prior_mean_l1.item()),
        "posterior_std_mean": float(posterior_std_mean.item()),
        "prior_std_mean": float(prior_std_mean.item()),
        "deployment_zero_action_l1": float(zero_action.item()),
        "deployment_zero_force_l1": float(zero_force.item()),
        "deployment_prior_action_l1": float(prior_action.item()),
        "deployment_prior_force_l1": float(prior_force.item()),
    }


def _validate_training_objects(
    model: ACTAlignedContactCVAEPolicy,
    criterion: ACTAlignedCriterion,
    training_config: ACTAlignedTrainingConfig,
) -> None:
    if not isinstance(model, ACTAlignedContactCVAEPolicy):
        raise TypeError("model must be an ACTAlignedContactCVAEPolicy")
    if not isinstance(criterion, ACTAlignedCriterion):
        raise TypeError("criterion must be an ACTAlignedCriterion")
    if not isinstance(training_config, ACTAlignedTrainingConfig):
        raise TypeError("training_config must be an ACTAlignedTrainingConfig")
    if criterion.config != training_config:
        raise ValueError("criterion and training step must use the same config")


def _gradient_norm(parameters: list[torch.nn.Parameter]) -> float:
    squared_norm = 0.0
    for parameter in parameters:
        squared_norm += float(parameter.grad.detach().float().square().sum().item())
    return math.sqrt(squared_norm)


def _group_learning_rate(
    optimizer: torch.optim.Optimizer,
    group_name: str,
) -> float:
    for group in optimizer.param_groups:
        if group.get("name") == group_name:
            return float(group["lr"])
    raise RuntimeError(f"optimizer is missing parameter group {group_name!r}")
