"""Objectives for native-rate motion and latent-free controls."""

from __future__ import annotations

from typing import Dict, Mapping

import torch
from torch import nn

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
from force_aware_act.act_aligned_training.losses import (
    masked_l1_loss,
    standard_normal_kl,
)


def _reconstruction_losses(
    outputs: Mapping[str, torch.Tensor],
    action_target: torch.Tensor,
    force_target: torch.Tensor,
    high_rate_force_target: torch.Tensor,
    action_padding_mask: torch.Tensor,
    force_interval_padding_mask: torch.Tensor,
    force_sample_padding_mask: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    required = ("pred_action", "pred_force", "pred_force_highrate")
    for name in required:
        if name not in outputs:
            raise KeyError(f"training outputs are missing {name!r}")
    return {
        "loss_action": masked_l1_loss(
            outputs["pred_action"],
            action_target,
            action_padding_mask,
            name="action",
        ),
        "loss_force": masked_optional_l1_loss(
            outputs["pred_force"],
            force_target,
            force_interval_padding_mask,
            name="force_interval_endpoint",
        ),
        "loss_force_highrate": masked_interval_balanced_high_rate_l1_loss(
            outputs["pred_force_highrate"],
            high_rate_force_target,
            force_sample_padding_mask,
            force_interval_padding_mask,
            name="force_highrate",
        ),
    }


class ACTAlignedHighRateMotionCriterion(nn.Module):
    """Reconstruction plus KL for the action-only motion posterior."""

    def __init__(self, config: ACTAlignedHighRateMotionTrainingConfig) -> None:
        super().__init__()
        if not isinstance(config, ACTAlignedHighRateMotionTrainingConfig):
            raise TypeError("config must be ACTAlignedHighRateMotionTrainingConfig")
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
        for name in ("mu_motion", "logvar_motion"):
            if name not in outputs:
                raise KeyError(f"training outputs are missing {name!r}")
        losses = _reconstruction_losses(
            outputs,
            action_target,
            force_target,
            high_rate_force_target,
            action_padding_mask,
            force_interval_padding_mask,
            force_sample_padding_mask,
        )
        losses["loss_posterior_kl"] = standard_normal_kl(
            outputs["mu_motion"],
            outputs["logvar_motion"],
        )
        losses["loss_total"] = (
            self.config.action_loss_weight * losses["loss_action"]
            + self.config.force_loss_weight * losses["loss_force"]
            + self.config.high_rate_force_loss_weight
            * losses["loss_force_highrate"]
            + self.config.posterior_kl_weight
            * losses["loss_posterior_kl"]
        )
        return losses


class ACTAlignedHighRateDualZeroCriterion(nn.Module):
    """Pure reconstruction objective with no latent loss terms."""

    def __init__(self, config: ACTAlignedHighRateDualZeroTrainingConfig) -> None:
        super().__init__()
        if not isinstance(config, ACTAlignedHighRateDualZeroTrainingConfig):
            raise TypeError(
                "config must be ACTAlignedHighRateDualZeroTrainingConfig"
            )
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
        losses = _reconstruction_losses(
            outputs,
            action_target,
            force_target,
            high_rate_force_target,
            action_padding_mask,
            force_interval_padding_mask,
            force_sample_padding_mask,
        )
        losses["loss_total"] = (
            self.config.action_loss_weight * losses["loss_action"]
            + self.config.force_loss_weight * losses["loss_force"]
            + self.config.high_rate_force_loss_weight
            * losses["loss_force_highrate"]
        )
        return losses
