"""Force-conditioned visual cross-attention."""

from __future__ import annotations

from typing import Optional, Union

import torch
from torch import nn

from force_aware_act.models.act_aligned.config import ACTAlignedConfig
from force_aware_act.models.act_aligned.contracts import require_padding_mask


class ACTAlignedForceVisionFusion(nn.Module):
    """Attend from one online force feature into spatial visual tokens."""

    def __init__(self, config: ACTAlignedConfig) -> None:
        super().__init__()
        self.config = config
        self.cross_attention = nn.MultiheadAttention(
            config.d_model,
            config.nhead,
            dropout=config.dropout,
            batch_first=True,
        )

    def forward(
        self,
        online_force_feature: torch.Tensor,
        visual_tokens: torch.Tensor,
        *,
        visual_position: Optional[torch.Tensor] = None,
        visual_padding_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        if not isinstance(return_attention, bool):
            raise ValueError("return_attention must be a bool")
        self._validate_inputs(
            online_force_feature,
            visual_tokens,
            visual_position,
            visual_padding_mask,
        )
        query = online_force_feature.unsqueeze(1)
        key = (
            visual_tokens
            if visual_position is None
            else visual_tokens + visual_position
        )
        attended, attention = self.cross_attention(
            query,
            key,
            visual_tokens,
            key_padding_mask=visual_padding_mask,
            need_weights=return_attention,
        )
        fused = attended[:, 0]
        if return_attention:
            return fused, attention
        return fused

    def _validate_inputs(
        self,
        online_force_feature: torch.Tensor,
        visual_tokens: torch.Tensor,
        visual_position: Optional[torch.Tensor],
        visual_padding_mask: Optional[torch.Tensor],
    ) -> None:
        if not isinstance(online_force_feature, torch.Tensor):
            raise TypeError("online_force_feature must be a torch.Tensor")
        if (
            online_force_feature.ndim != 2
            or online_force_feature.shape[-1] != self.config.d_model
        ):
            raise ValueError(
                "online_force_feature must have shape "
                f"[B, {self.config.d_model}]"
            )
        if not online_force_feature.is_floating_point():
            raise ValueError("online_force_feature must be floating point")
        if not isinstance(visual_tokens, torch.Tensor):
            raise TypeError("visual_tokens must be a torch.Tensor")
        expected_visual_shape = (
            online_force_feature.shape[0],
            self.config.visual_token_count,
            self.config.d_model,
        )
        if visual_tokens.shape != expected_visual_shape:
            raise ValueError(
                f"visual_tokens must have shape {expected_visual_shape}, "
                f"got {tuple(visual_tokens.shape)}"
            )
        if not visual_tokens.is_floating_point():
            raise ValueError("visual_tokens must be floating point")
        if visual_tokens.device != online_force_feature.device:
            raise ValueError(
                "visual_tokens must be on the same device as online_force_feature"
            )
        if visual_tokens.dtype != online_force_feature.dtype:
            raise ValueError(
                "visual_tokens must have the same dtype as online_force_feature"
            )
        if visual_position is not None:
            if not isinstance(visual_position, torch.Tensor):
                raise TypeError("visual_position must be a torch.Tensor")
            if visual_position.shape != visual_tokens.shape:
                raise ValueError(
                    "visual_position must have the same shape as visual_tokens"
                )
            if not visual_position.is_floating_point():
                raise ValueError("visual_position must be floating point")
            if (
                visual_position.device != visual_tokens.device
                or visual_position.dtype != visual_tokens.dtype
            ):
                raise ValueError(
                    "visual_position must share visual_tokens device and dtype"
                )
        if visual_padding_mask is not None:
            require_padding_mask(
                visual_padding_mask,
                name="visual_padding_mask",
                batch_size=visual_tokens.shape[0],
                sequence_length=visual_tokens.shape[1],
            )
