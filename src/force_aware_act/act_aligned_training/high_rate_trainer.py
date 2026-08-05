"""One-step training and leakage-safe validation for the 500 Hz policy."""

from __future__ import annotations

import math
from typing import Any, Dict

import torch

from force_aware_act.act_aligned_training.high_rate_batch import ACTAlignedHighRateBatch
from force_aware_act.act_aligned_training.high_rate_config import ACTAlignedHighRateTrainingConfig
from force_aware_act.act_aligned_training.high_rate_losses import (
    ACTAlignedHighRateCriterion,
    masked_interval_balanced_high_rate_l1_loss,
    masked_optional_l1_loss,
)
from force_aware_act.act_aligned_training.losses import masked_l1_loss
from force_aware_act.models.act_aligned.high_rate_policy import (
    ACTAlignedHighRateContactCVAEPolicy,
)


def _training_forward(
    model: ACTAlignedHighRateContactCVAEPolicy,
    batch: ACTAlignedHighRateBatch,
    *,
    sample_posterior: bool,
) -> Dict[str, Any]:
    return model.forward_train(
        batch.images,
        batch.qpos,
        batch.online_force_intervals,
        batch.online_force_relative_time,
        batch.online_force_sample_padding_mask,
        batch.online_force_interval_padding_mask,
        batch.action_chunk,
        batch.future_force_intervals,
        batch.future_force_relative_time,
        batch.future_force_sample_padding_mask,
        batch.future_force_interval_padding_mask,
        action_padding_mask=batch.action_padding_mask,
        sample_posterior=sample_posterior,
    )


def _deployment_forward(
    model: ACTAlignedHighRateContactCVAEPolicy,
    batch: ACTAlignedHighRateBatch,
    *,
    mode: str,
) -> Dict[str, Any]:
    return model(
        batch.images,
        batch.qpos,
        batch.online_force_intervals,
        batch.online_force_relative_time,
        batch.online_force_sample_padding_mask,
        batch.online_force_interval_padding_mask,
        contact_latent_mode=mode,
        deterministic_prior=True,
    )


def _losses(
    criterion: ACTAlignedHighRateCriterion,
    outputs: Dict[str, Any],
    batch: ACTAlignedHighRateBatch,
) -> Dict[str, torch.Tensor]:
    return criterion(
        outputs,
        batch.action_chunk,
        batch.future_force_target,
        batch.future_force_intervals,
        batch.action_padding_mask,
        batch.future_force_interval_padding_mask,
        batch.future_force_sample_padding_mask,
    )


def train_high_rate_one_step(
    model: ACTAlignedHighRateContactCVAEPolicy,
    criterion: ACTAlignedHighRateCriterion,
    optimizer: torch.optim.Optimizer,
    batch: ACTAlignedHighRateBatch,
    training_config: ACTAlignedHighRateTrainingConfig,
) -> Dict[str, float]:
    _validate_training_objects(model, criterion, training_config)
    batch.validate(model.config)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    outputs = _training_forward(model, batch, sample_posterior=True)
    losses = _losses(criterion, outputs, batch)
    total = losses["loss_total"]
    if not torch.isfinite(total):
        raise FloatingPointError("training loss is not finite")
    total.backward()
    parameters = [
        parameter for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    if not parameters:
        raise RuntimeError("training step produced no parameter gradients")
    if training_config.gradient_clip_norm is None:
        gradient_norm = math.sqrt(sum(
            float(parameter.grad.detach().float().square().sum().item())
            for parameter in parameters
        ))
    else:
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(
            parameters, training_config.gradient_clip_norm
        ).item())
    if not math.isfinite(gradient_norm):
        raise FloatingPointError("gradient norm is not finite")
    optimizer.step()
    metrics = {name: float(value.detach().item()) for name, value in losses.items()}
    metrics.update(_latent_metrics(outputs))
    metrics["gradient_norm"] = gradient_norm
    metrics["main_learning_rate"] = _group_learning_rate(optimizer, "main")
    metrics["backbone_learning_rate"] = _group_learning_rate(optimizer, "backbone")
    return metrics


def evaluate_high_rate_one_batch(
    model: ACTAlignedHighRateContactCVAEPolicy,
    criterion: ACTAlignedHighRateCriterion,
    batch: ACTAlignedHighRateBatch,
) -> Dict[str, Any]:
    if not isinstance(model, ACTAlignedHighRateContactCVAEPolicy):
        raise TypeError("model must be ACTAlignedHighRateContactCVAEPolicy")
    if not isinstance(criterion, ACTAlignedHighRateCriterion):
        raise TypeError("criterion must be ACTAlignedHighRateCriterion")
    batch.validate(model.config)
    previous_mode = model.training
    model.eval()
    try:
        with torch.no_grad():
            posterior = _training_forward(model, batch, sample_posterior=False)
            posterior_losses = _losses(criterion, posterior, batch)
            zero = _deployment_forward(model, batch, mode="zero")
            prior = _deployment_forward(model, batch, mode="prior")
            result = {
                "posterior_total": float(posterior_losses["loss_total"].item()),
                "posterior_action_l1": _masked_value(
                    posterior["pred_action"], batch.action_chunk,
                    batch.action_padding_mask, "posterior_action"
                ),
                "posterior_force_l1": _masked_value(
                    posterior["pred_force"], batch.future_force_target,
                    batch.future_force_interval_padding_mask, "posterior_force"
                ),
                "posterior_force_highrate_l1": _high_rate_value(
                    posterior["pred_force_highrate"], batch, "posterior_highrate"
                ),
                "posterior_kl_standard": float(posterior_losses["loss_posterior_kl"].item()),
                "posterior_prior_match_kl": float(posterior_losses["loss_prior_match"].item()),
            }
            for prefix, outputs in (("deployment_zero", zero), ("deployment_prior", prior)):
                result[f"{prefix}_action_l1"] = _masked_value(
                    outputs["pred_action"], batch.action_chunk,
                    batch.action_padding_mask, f"{prefix}_action"
                )
                result[f"{prefix}_force_l1"] = _masked_value(
                    outputs["pred_force"], batch.future_force_target,
                    batch.future_force_interval_padding_mask, f"{prefix}_force"
                )
                result[f"{prefix}_force_highrate_l1"] = _high_rate_value(
                    outputs["pred_force_highrate"], batch, f"{prefix}_highrate"
                )
            for prefix, outputs in (("posterior_zero", posterior), ("prior_zero", prior)):
                reference = zero
                result[f"{prefix}_action_delta"] = _masked_value(
                    outputs["pred_action"], reference["pred_action"],
                    batch.action_padding_mask, f"{prefix}_action_delta"
                )
                result[f"{prefix}_force_delta"] = _masked_value(
                    outputs["pred_force"], reference["pred_force"],
                    batch.future_force_interval_padding_mask, f"{prefix}_force_delta"
                )
                result[f"{prefix}_force_highrate_delta"] = _high_rate_value(
                    outputs["pred_force_highrate"], batch,
                    f"{prefix}_highrate_delta", target_override=reference["pred_force_highrate"]
                )
            result.update(_latent_validation_metrics(posterior))
    finally:
        model.train(previous_mode)
    return result


def _masked_value(prediction, target, mask, name: str) -> float:
    if prediction.shape[-1] == 6:
        loss = masked_optional_l1_loss(prediction, target, mask, name=name)
    else:
        loss = masked_l1_loss(prediction, target, mask, name=name)
    return float(loss.item())


def _high_rate_value(prediction, batch, name: str, *, target_override=None) -> float:
    target = batch.future_force_intervals if target_override is None else target_override
    return float(masked_interval_balanced_high_rate_l1_loss(
        prediction, target, batch.future_force_sample_padding_mask,
        batch.future_force_interval_padding_mask, name=name,
    ).item())


def _latent_metrics(outputs: Dict[str, Any]) -> Dict[str, float]:
    return {
        "posterior_mean_abs": float(outputs["mu_contact"].detach().abs().mean().item()),
        "prior_mean_abs": float(outputs["mu_contact_prior"].detach().abs().mean().item()),
        "posterior_std_mean": float(torch.exp(0.5 * outputs["logvar_contact"].detach()).mean().item()),
        "prior_std_mean": float(torch.exp(0.5 * outputs["logvar_contact_prior"].detach()).mean().item()),
        "posterior_prior_mean_l1": float((outputs["mu_contact"].detach() - outputs["mu_contact_prior"].detach()).abs().mean().item()),
    }


def _latent_validation_metrics(outputs: Dict[str, Any]) -> Dict[str, Any]:
    mu = outputs["mu_contact"].detach()
    prior_mu = outputs["mu_contact_prior"].detach()
    return {
        "posterior_prior_mean_l1": float((mu - prior_mu).abs().mean().item()),
        "posterior_std_mean": float(torch.exp(0.5 * outputs["logvar_contact"]).mean().item()),
        "prior_std_mean": float(torch.exp(0.5 * outputs["logvar_contact_prior"]).mean().item()),
        "_posterior_mean_sum": mu.double().sum(dim=0).cpu(),
        "_posterior_mean_square_sum": mu.double().square().sum(dim=0).cpu(),
        "_prior_mean_sum": prior_mu.double().sum(dim=0).cpu(),
        "_prior_mean_square_sum": prior_mu.double().square().sum(dim=0).cpu(),
    }


def _validate_training_objects(model, criterion, config) -> None:
    if not isinstance(model, ACTAlignedHighRateContactCVAEPolicy):
        raise TypeError("model must be ACTAlignedHighRateContactCVAEPolicy")
    if not isinstance(criterion, ACTAlignedHighRateCriterion):
        raise TypeError("criterion must be ACTAlignedHighRateCriterion")
    if not isinstance(config, ACTAlignedHighRateTrainingConfig):
        raise TypeError("config must be ACTAlignedHighRateTrainingConfig")
    if criterion.config != config:
        raise ValueError("criterion and training step must use the same config")


def _group_learning_rate(optimizer, name: str) -> float:
    for group in optimizer.param_groups:
        if group.get("name") == name:
            return float(group["lr"])
    raise RuntimeError(f"optimizer is missing parameter group {name!r}")
