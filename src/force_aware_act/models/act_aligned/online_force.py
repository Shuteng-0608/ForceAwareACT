"""Causal historical-force encoding with an ACT-depth Transformer."""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from force_aware_act.models.act_aligned.config import ACTAlignedConfig
from force_aware_act.models.act_aligned.contracts import require_padding_mask
from force_aware_act.models.act_aligned.position_encoding import (
    SinusoidalSequencePositionEncoding,
    TokenTypeEmbedding,
)
from force_aware_act.models.act_aligned.token_adapters import ForceTokenAdapter
from force_aware_act.models.act_aligned.transformer import ACTTransformerEncoder


FORCE_CLS_TYPE = 0
FORCE_SAMPLE_TYPE = 1


class ACTAlignedOnlineForceEncoder(nn.Module):
    """Map a fixed causal force window to one online force feature.

    Input: ``[B, force_window_len, force_dim]``
    Output: ``[B, d_model]``
    """

    def __init__(self, config: ACTAlignedConfig) -> None:
        super().__init__()
        self.config = config
        self.force_adapter = ForceTokenAdapter(config.force_dim, config.d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.d_model))
        self.sequence_position = SinusoidalSequencePositionEncoding(
            config.force_encoder_token_count,
            config.d_model,
        )
        self.token_type = TokenTypeEmbedding(2, config.d_model)
        self.encoder = ACTTransformerEncoder(config)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(
        self,
        force_history: torch.Tensor,
        *,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        self._validate_inputs(force_history, padding_mask)
        batch_size = force_history.shape[0]
        force_tokens = self.force_adapter(force_history)
        cls_token = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat((cls_token, force_tokens), dim=1)

        type_ids = torch.full(
            (self.config.force_encoder_token_count,),
            FORCE_SAMPLE_TYPE,
            dtype=torch.long,
            device=tokens.device,
        )
        type_ids[0] = FORCE_CLS_TYPE
        position = self.sequence_position(tokens) + self.token_type(type_ids).unsqueeze(0)
        encoder_padding_mask = None
        if padding_mask is not None:
            cls_mask = torch.zeros(
                batch_size,
                1,
                dtype=torch.bool,
                device=padding_mask.device,
            )
            encoder_padding_mask = torch.cat((cls_mask, padding_mask), dim=1)

        encoded = self.encoder(
            tokens,
            position=position,
            padding_mask=encoder_padding_mask,
        )
        return encoded[:, 0]

    def _validate_inputs(
        self,
        force_history: torch.Tensor,
        padding_mask: Optional[torch.Tensor],
    ) -> None:
        if not isinstance(force_history, torch.Tensor):
            raise TypeError("force_history must be a torch.Tensor")
        expected_shape = (
            self.config.force_window_len,
            self.config.force_dim,
        )
        if force_history.ndim != 3 or tuple(force_history.shape[1:]) != expected_shape:
            raise ValueError(
                "force_history must have shape "
                f"[B, {expected_shape[0]}, {expected_shape[1]}], "
                f"got {tuple(force_history.shape)}"
            )
        if not force_history.is_floating_point():
            raise ValueError("force_history must be floating point")
        if padding_mask is not None:
            require_padding_mask(
                padding_mask,
                name="padding_mask",
                batch_size=force_history.shape[0],
                sequence_length=self.config.force_window_len,
            )
