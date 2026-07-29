"""Official ACT action-only CVAE posterior."""

from __future__ import annotations

import copy
from typing import Optional

import torch
from torch import nn

from force_aware_act.models.act_aligned.contact_latent import (
    reparameterize_gaussian,
)
from force_aware_act.models.act_aligned.contracts import require_padding_mask
from force_aware_act.models.act_aligned.position_encoding import (
    SinusoidalSequencePositionEncoding,
)
from force_aware_act.models.act_aligned.transformer import ACTEncoderLayer
from force_aware_act.models.official_act.config import OfficialACTConfig


class _OfficialPosteriorEncoder(nn.Module):
    """Four official post-norm layers without DETR's global Xavier reset."""

    def __init__(self, config: OfficialACTConfig) -> None:
        super().__init__()
        layer = ACTEncoderLayer(config)
        self.layers = nn.ModuleList(
            copy.deepcopy(layer) for _ in range(config.encoder_layers)
        )

    def forward(
        self,
        tokens: torch.Tensor,
        *,
        position: torch.Tensor,
        padding_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        output = tokens
        for layer in self.layers:
            output = layer(
                output,
                position=position,
                padding_mask=padding_mask,
            )
        return output


class OfficialACTPosterior(nn.Module):
    """Map ``[CLS, qpos, action sequence]`` to a 32-D Gaussian."""

    def __init__(self, config: OfficialACTConfig) -> None:
        super().__init__()
        self.config = config
        self.action_projection = nn.Linear(
            config.action_dim,
            config.d_model,
        )
        self.qpos_projection = nn.Linear(config.q_dim, config.d_model)
        self.cls_embedding = nn.Embedding(1, config.d_model)
        self.position = SinusoidalSequencePositionEncoding(
            config.posterior_token_count,
            config.d_model,
        )
        self.encoder = _OfficialPosteriorEncoder(config)
        self.latent_projection = nn.Linear(
            config.d_model,
            2 * config.latent_dim,
        )

    def forward(
        self,
        qpos: torch.Tensor,
        action_chunk: torch.Tensor,
        padding_mask: torch.Tensor,
        *,
        sample: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not isinstance(sample, bool):
            raise ValueError("sample must be a bool")
        self._validate(qpos, action_chunk, padding_mask)
        batch_size = qpos.shape[0]
        tokens = torch.cat(
            (
                self.cls_embedding.weight.unsqueeze(0).expand(
                    batch_size,
                    -1,
                    -1,
                ),
                self.qpos_projection(qpos).unsqueeze(1),
                self.action_projection(action_chunk),
            ),
            dim=1,
        )
        prefix = torch.zeros(
            batch_size,
            2,
            dtype=torch.bool,
            device=padding_mask.device,
        )
        encoded = self.encoder(
            tokens,
            position=self.position(tokens),
            padding_mask=torch.cat((prefix, padding_mask), dim=1),
        )
        latent_info = self.latent_projection(encoded[:, 0])
        mean, log_variance = latent_info.chunk(2, dim=-1)
        latent = (
            reparameterize_gaussian(mean, log_variance)
            if sample
            else mean
        )
        return mean, log_variance, latent

    def _validate(
        self,
        qpos: torch.Tensor,
        action_chunk: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> None:
        if not isinstance(qpos, torch.Tensor) or not isinstance(
            action_chunk,
            torch.Tensor,
        ):
            raise TypeError(
                "qpos and action_chunk must be torch.Tensor instances"
            )
        if qpos.ndim != 2:
            raise ValueError(f"qpos must have shape [B, {self.config.q_dim}]")
        expected_qpos = (qpos.shape[0], self.config.q_dim)
        expected_action = (
            qpos.shape[0],
            self.config.chunk_len,
            self.config.action_dim,
        )
        if tuple(qpos.shape) != expected_qpos:
            raise ValueError(f"qpos must have shape [B, {self.config.q_dim}]")
        if tuple(action_chunk.shape) != expected_action:
            raise ValueError(
                f"action_chunk must have shape {expected_action}"
            )
        if not qpos.is_floating_point() or not action_chunk.is_floating_point():
            raise ValueError("qpos and action_chunk must be floating point")
        if (
            action_chunk.device != qpos.device
            or action_chunk.dtype != qpos.dtype
        ):
            raise ValueError("action_chunk must share qpos context")
        require_padding_mask(
            padding_mask,
            name="padding_mask",
            batch_size=qpos.shape[0],
            sequence_length=self.config.chunk_len,
        )
