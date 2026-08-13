"""Action-only reconstruction objective for latent-free official ACT."""

from __future__ import annotations

from typing import Dict, Mapping

import torch
from torch import nn

from force_aware_act.official_act_training.losses import official_masked_l1
from force_aware_act.official_act_training.no_latent_config import (
    OfficialACTNoLatentTrainingConfig,
)


class OfficialACTNoLatentCriterion(nn.Module):
    def __init__(self, config: OfficialACTNoLatentTrainingConfig) -> None:
        super().__init__()
        self.config = config

    def forward(
        self,
        outputs: Mapping[str, torch.Tensor],
        action_target: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if "pred_action" not in outputs:
            raise KeyError("outputs are missing 'pred_action'")
        l1 = official_masked_l1(
            outputs["pred_action"],
            action_target,
            padding_mask,
        )
        total = self.config.action_loss_weight * l1
        return {
            "loss_total": total,
            "loss_l1": l1,
        }
