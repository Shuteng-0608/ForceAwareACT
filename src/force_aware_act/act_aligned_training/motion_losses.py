"""Official-style motion-CVAE objective for the control experiment."""

from __future__ import annotations

from typing import Dict, Mapping

import torch
from torch import nn

from force_aware_act.act_aligned_training.losses import (
    masked_l1_loss,
    standard_normal_kl,
)
from force_aware_act.act_aligned_training.motion_config import (
    ACTAlignedMotionTrainingConfig,
)


class ACTAlignedMotionCriterion(nn.Module):
    """Compute action/force reconstruction plus ``10 * KL(q_motion || N)``."""

    required_output_names = (
        "pred_action",
        "pred_force",
        "mu_motion",
        "logvar_motion",
    )

    def __init__(self, config: ACTAlignedMotionTrainingConfig) -> None:
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
            outputs["mu_motion"],
            outputs["logvar_motion"],
        )
        total = (
            self.config.action_loss_weight * action_loss
            + self.config.force_loss_weight * force_loss
            + self.config.posterior_kl_weight * posterior_kl
        )
        return {
            "loss_total": total,
            "loss_action": action_loss,
            "loss_force": force_loss,
            "loss_posterior_kl": posterior_kl,
        }
