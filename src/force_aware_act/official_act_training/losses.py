"""Exact official ACT L1 plus KL objective."""

from __future__ import annotations

from typing import Dict, Mapping

import torch
from torch import nn

from force_aware_act.act_aligned_training.losses import standard_normal_kl
from force_aware_act.official_act_training.config import (
    OfficialACTTrainingConfig,
)


def official_masked_l1(
    prediction: torch.Tensor,
    target: torch.Tensor,
    padding_mask: torch.Tensor,
) -> torch.Tensor:
    """Mask padding, then mean over the complete B x K x D tensor."""

    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError("prediction and target must share [B, K, D]")
    if padding_mask.shape != prediction.shape[:2]:
        raise ValueError("padding_mask must have shape [B, K]")
    if padding_mask.dtype is not torch.bool:
        raise ValueError("padding_mask must have dtype torch.bool")
    return (
        (prediction - target).abs()
        * (~padding_mask).unsqueeze(-1)
    ).mean()


class OfficialACTCriterion(nn.Module):
    def __init__(self, config: OfficialACTTrainingConfig) -> None:
        super().__init__()
        self.config = config

    def forward(
        self,
        outputs: Mapping[str, torch.Tensor],
        action_target: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        for name in ("pred_action", "mu_motion", "logvar_motion"):
            if name not in outputs:
                raise KeyError(f"outputs are missing {name!r}")
        l1 = official_masked_l1(
            outputs["pred_action"],
            action_target,
            padding_mask,
        )
        kl = standard_normal_kl(
            outputs["mu_motion"],
            outputs["logvar_motion"],
        )
        total = self.config.action_loss_weight * l1 + self.config.kl_weight * kl
        return {
            "loss_total": total,
            "loss_l1": l1,
            "loss_kl": kl,
        }
