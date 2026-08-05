"""Losses for native 500 Hz ACT-aligned contact-CVAE training."""

from __future__ import annotations

from typing import Dict, Mapping

import torch
from torch import nn

from force_aware_act.act_aligned_training.high_rate_config import (
    ACTAlignedHighRateTrainingConfig,
)
from force_aware_act.act_aligned_training.losses import (
    detached_posterior_prior_kl,
    masked_l1_loss,
    standard_normal_kl,
)


def masked_optional_l1_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    padding_mask: torch.Tensor,
    *,
    name: str,
) -> torch.Tensor:
    """Masked L1 that is a differentiable zero when no future interval exists."""

    if padding_mask.numel() > 0 and bool((~padding_mask).any().item()):
        return masked_l1_loss(prediction, target, padding_mask, name=name)
    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError(f"{name} prediction/target must share shape [B, K, F]")
    if padding_mask.shape != prediction.shape[:2] or padding_mask.dtype is not torch.bool:
        raise ValueError(f"{name} padding mask must have shape [B, K] and bool dtype")
    return prediction.sum() * 0.0


def masked_interval_balanced_high_rate_l1_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    sample_padding_mask: torch.Tensor,
    interval_padding_mask: torch.Tensor,
    *,
    name: str,
) -> torch.Tensor:
    """Give every valid state interval equal weight, independent of 16/17 samples."""

    if prediction.shape != target.shape or prediction.ndim != 4:
        raise ValueError(f"{name} prediction/target must share shape [B, K, S, F]")
    if prediction.device != target.device or prediction.dtype != target.dtype:
        raise ValueError(f"{name} prediction/target must share device and dtype")
    if not prediction.is_floating_point():
        raise ValueError(f"{name} prediction/target must be floating point")
    if sample_padding_mask.shape != prediction.shape[:3]:
        raise ValueError(f"{name} sample mask must have shape [B, K, S]")
    if interval_padding_mask.shape != prediction.shape[:2]:
        raise ValueError(f"{name} interval mask must have shape [B, K]")
    if sample_padding_mask.dtype is not torch.bool or interval_padding_mask.dtype is not torch.bool:
        raise ValueError(f"{name} masks must have dtype torch.bool")
    if sample_padding_mask.device != prediction.device or interval_padding_mask.device != prediction.device:
        raise ValueError(f"{name} masks must share prediction device")
    derived_interval_mask = sample_padding_mask.all(dim=-1)
    if not torch.equal(derived_interval_mask, interval_padding_mask):
        raise ValueError(f"{name} sample and interval masks disagree")

    valid_samples = ~sample_padding_mask
    per_sample = (prediction - target).abs().mean(dim=-1)
    sample_counts = valid_samples.sum(dim=-1)
    per_interval = (
        (per_sample * valid_samples).sum(dim=-1)
        / sample_counts.clamp_min(1)
    )
    valid_intervals = ~interval_padding_mask
    interval_count = valid_intervals.sum()
    if interval_count.item() == 0:
        return prediction.sum() * 0.0
    return (per_interval * valid_intervals).sum() / interval_count


class ACTAlignedHighRateCriterion(nn.Module):
    required_output_names = (
        "pred_action",
        "pred_force",
        "pred_force_highrate",
        "mu_contact",
        "logvar_contact",
        "mu_contact_prior",
        "logvar_contact_prior",
    )

    def __init__(self, config: ACTAlignedHighRateTrainingConfig) -> None:
        super().__init__()
        if not isinstance(config, ACTAlignedHighRateTrainingConfig):
            raise TypeError("config must be ACTAlignedHighRateTrainingConfig")
        self.config = config

    def forward(
        self,
        outputs: Mapping[str, torch.Tensor],
        action_target: torch.Tensor,
        force_target: torch.Tensor,
        high_rate_force_target: torch.Tensor,
        action_padding_mask: torch.Tensor,
        force_interval_padding_mask: torch.Tensor,
        force_sample_padding_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        for name in self.required_output_names:
            if name not in outputs:
                raise KeyError(f"training outputs are missing {name!r}")
        action_loss = masked_l1_loss(
            outputs["pred_action"], action_target, action_padding_mask, name="action"
        )
        force_loss = masked_optional_l1_loss(
            outputs["pred_force"], force_target, force_interval_padding_mask,
            name="force_interval_endpoint",
        )
        high_rate_force_loss = masked_interval_balanced_high_rate_l1_loss(
            outputs["pred_force_highrate"],
            high_rate_force_target,
            force_sample_padding_mask,
            force_interval_padding_mask,
            name="force_highrate",
        )
        posterior_kl = standard_normal_kl(
            outputs["mu_contact"], outputs["logvar_contact"]
        )
        prior_match = detached_posterior_prior_kl(
            outputs["mu_contact"], outputs["logvar_contact"],
            outputs["mu_contact_prior"], outputs["logvar_contact_prior"],
        )
        total = (
            self.config.action_loss_weight * action_loss
            + self.config.force_loss_weight * force_loss
            + self.config.high_rate_force_loss_weight * high_rate_force_loss
            + self.config.posterior_kl_weight * posterior_kl
            + self.config.prior_match_weight * prior_match
        )
        return {
            "loss_total": total,
            "loss_action": action_loss,
            "loss_force": force_loss,
            "loss_force_highrate": high_rate_force_loss,
            "loss_posterior_kl": posterior_kl,
            "loss_prior_match": prior_match,
        }
