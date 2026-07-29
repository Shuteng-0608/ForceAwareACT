"""Conditional-CVAE objective for the ACT-aligned policy."""

from __future__ import annotations

from typing import Dict, Mapping

import torch
from torch import nn

from force_aware_act.act_aligned_training.config import ACTAlignedTrainingConfig


def masked_l1_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    padding_mask: torch.Tensor,
    *,
    name: str,
) -> torch.Tensor:
    """Average absolute error over valid scalar elements only."""

    if not isinstance(prediction, torch.Tensor) or not isinstance(
        target,
        torch.Tensor,
    ):
        raise TypeError(f"{name} prediction and target must be torch.Tensor instances")
    if prediction.shape != target.shape:
        raise ValueError(
            f"{name} prediction and target must have the same shape"
        )
    if prediction.ndim != 3:
        raise ValueError(f"{name} prediction and target must have shape [B, K, F]")
    if not prediction.is_floating_point() or not target.is_floating_point():
        raise ValueError(f"{name} prediction and target must be floating point")
    if prediction.device != target.device or prediction.dtype != target.dtype:
        raise ValueError(
            f"{name} prediction and target must share device and dtype"
        )
    if not isinstance(padding_mask, torch.Tensor):
        raise TypeError("padding_mask must be a torch.Tensor")
    if padding_mask.shape != prediction.shape[:2]:
        raise ValueError(
            f"padding_mask must have shape {tuple(prediction.shape[:2])}"
        )
    if padding_mask.dtype is not torch.bool:
        raise ValueError("padding_mask must have dtype torch.bool")
    if padding_mask.device != prediction.device:
        raise ValueError("padding_mask must be on the prediction device")

    valid_steps = ~padding_mask
    valid_scalar_count = valid_steps.sum() * prediction.shape[-1]
    if valid_scalar_count.item() == 0:
        raise ValueError(f"{name} loss requires at least one valid target step")
    absolute_error = (prediction - target).abs()
    masked_error = absolute_error * valid_steps.unsqueeze(-1)
    return masked_error.sum() / valid_scalar_count


def diagonal_gaussian_kl(
    posterior_mean: torch.Tensor,
    posterior_log_variance: torch.Tensor,
    prior_mean: torch.Tensor,
    prior_log_variance: torch.Tensor,
) -> torch.Tensor:
    """Return mean KL(q || p) for two diagonal Gaussian distributions."""

    tensors = (
        ("posterior_mean", posterior_mean),
        ("posterior_log_variance", posterior_log_variance),
        ("prior_mean", prior_mean),
        ("prior_log_variance", prior_log_variance),
    )
    reference = posterior_mean
    for name, tensor in tensors:
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if tensor.ndim != 2:
            raise ValueError(f"{name} must have shape [B, latent_dim]")
        if tensor.shape != reference.shape:
            raise ValueError("all Gaussian tensors must have the same shape")
        if not tensor.is_floating_point():
            raise ValueError(f"{name} must be floating point")
        if tensor.device != reference.device or tensor.dtype != reference.dtype:
            raise ValueError("all Gaussian tensors must share device and dtype")
    if reference.shape[0] <= 0 or reference.shape[1] <= 0:
        raise ValueError("Gaussian tensors must have non-empty dimensions")

    variance_ratio = torch.exp(
        posterior_log_variance - prior_log_variance
    )
    squared_mean_distance = (posterior_mean - prior_mean).square()
    scaled_mean_distance = squared_mean_distance * torch.exp(
        -prior_log_variance
    )
    per_dimension = 0.5 * (
        prior_log_variance
        - posterior_log_variance
        + variance_ratio
        + scaled_mean_distance
        - 1.0
    )
    return per_dimension.sum(dim=-1).mean()


def standard_normal_kl(
    mean: torch.Tensor,
    log_variance: torch.Tensor,
) -> torch.Tensor:
    """Return mean KL(q || N(0, I)) for a diagonal Gaussian posterior."""

    return diagonal_gaussian_kl(
        mean,
        log_variance,
        torch.zeros_like(mean),
        torch.zeros_like(log_variance),
    )


def detached_posterior_prior_kl(
    posterior_mean: torch.Tensor,
    posterior_log_variance: torch.Tensor,
    prior_mean: torch.Tensor,
    prior_log_variance: torch.Tensor,
) -> torch.Tensor:
    """Train the conditional prior against a stop-gradient posterior."""

    return diagonal_gaussian_kl(
        posterior_mean.detach(),
        posterior_log_variance.detach(),
        prior_mean,
        prior_log_variance,
    )


class ACTAlignedCriterion(nn.Module):
    """Compute reconstruction, posterior KL, and asymmetric prior matching."""

    required_output_names = (
        "pred_action",
        "pred_force",
        "mu_contact",
        "logvar_contact",
        "mu_contact_prior",
        "logvar_contact_prior",
    )

    def __init__(self, config: ACTAlignedTrainingConfig) -> None:
        super().__init__()
        self.config = config

    def forward(
        self,
        outputs: Mapping[str, torch.Tensor],
        action_target: torch.Tensor,
        force_target: torch.Tensor,
        future_padding_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        for name in self.required_output_names:
            if name not in outputs:
                raise KeyError(f"training outputs are missing {name!r}")

        action_loss = masked_l1_loss(
            outputs["pred_action"],
            action_target,
            future_padding_mask,
            name="action",
        )
        force_loss = masked_l1_loss(
            outputs["pred_force"],
            force_target,
            future_padding_mask,
            name="force",
        )
        posterior_kl = standard_normal_kl(
            outputs["mu_contact"],
            outputs["logvar_contact"],
        )
        prior_match = detached_posterior_prior_kl(
            outputs["mu_contact"],
            outputs["logvar_contact"],
            outputs["mu_contact_prior"],
            outputs["logvar_contact_prior"],
        )
        total = (
            self.config.action_loss_weight * action_loss
            + self.config.force_loss_weight * force_loss
            + self.config.posterior_kl_weight * posterior_kl
            + self.config.prior_match_weight * prior_match
        )
        return {
            "loss_total": total,
            "loss_action": action_loss,
            "loss_force": force_loss,
            "loss_posterior_kl": posterior_kl,
            "loss_prior_match": prior_match,
        }
