"""Official-style action-only ACT motion posterior."""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from force_aware_act.models.act_aligned.config import (
    ACT_ALIGNED_MOTION_CONTROL_ARCHITECTURE_VERSION,
    ACTAlignedConfig,
)
from force_aware_act.models.act_aligned.contact_latent import (
    reparameterize_gaussian,
)
from force_aware_act.models.act_aligned.contracts import require_padding_mask
from force_aware_act.models.act_aligned.position_encoding import (
    SinusoidalSequencePositionEncoding,
    TokenTypeEmbedding,
)
from force_aware_act.models.act_aligned.token_adapters import (
    ActionTokenAdapter,
    QposTokenAdapter,
)
from force_aware_act.models.act_aligned.transformer import (
    ACTTransformerEncoder,
)


MOTION_CLS_TYPE = 0
MOTION_QPOS_TYPE = 1
MOTION_ACTION_TYPE = 2


class ACTAlignedMotionPosterior(nn.Module):
    """Encode ``[CLS, qpos, action_0, ..., action_K-1]`` into ``z_motion``.

    This is the action-only posterior used by the ACT control experiment.
    Future force labels are intentionally absent from both the constructor and
    the forward signature, making accidental contact-label leakage impossible.
    """

    def __init__(self, config: ACTAlignedConfig) -> None:
        super().__init__()
        if (
            config.architecture_version
            != ACT_ALIGNED_MOTION_CONTROL_ARCHITECTURE_VERSION
        ):
            raise ValueError(
                "ACTAlignedMotionPosterior requires the motion-control "
                "architecture version"
            )
        self.config = config
        self.qpos_adapter = QposTokenAdapter(config.q_dim, config.d_model)
        self.action_adapter = ActionTokenAdapter(
            config.action_dim,
            config.d_model,
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.d_model))
        self.sequence_position = SinusoidalSequencePositionEncoding(
            config.motion_posterior_token_count,
            config.d_model,
        )
        self.token_type = TokenTypeEmbedding(3, config.d_model)
        self.encoder = ACTTransformerEncoder(config)
        self.mean_head = nn.Linear(config.d_model, config.latent_dim)
        self.log_variance_head = nn.Linear(
            config.d_model,
            config.latent_dim,
        )
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(
        self,
        qpos: torch.Tensor,
        action_chunk: torch.Tensor,
        *,
        padding_mask: Optional[torch.Tensor] = None,
        sample: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not isinstance(sample, bool):
            raise ValueError("sample must be a bool")
        self._validate_inputs(qpos, action_chunk, padding_mask)

        batch_size = qpos.shape[0]
        cls_token = self.cls_token.expand(batch_size, -1, -1)
        qpos_token = self.qpos_adapter(qpos)
        action_tokens = self.action_adapter(action_chunk)
        tokens = torch.cat((cls_token, qpos_token, action_tokens), dim=1)

        type_ids = torch.full(
            (self.config.motion_posterior_token_count,),
            MOTION_ACTION_TYPE,
            dtype=torch.long,
            device=tokens.device,
        )
        type_ids[0] = MOTION_CLS_TYPE
        type_ids[1] = MOTION_QPOS_TYPE
        position = (
            self.sequence_position(tokens)
            + self.token_type(type_ids).unsqueeze(0)
        )
        encoder_padding_mask = _prefix_unmasked_tokens(
            padding_mask,
            prefix_length=2,
        )
        encoded = self.encoder(
            tokens,
            position=position,
            padding_mask=encoder_padding_mask,
        )

        cls_output = encoded[:, 0]
        mean = self.mean_head(cls_output)
        log_variance = self.log_variance_head(cls_output)
        latent = (
            reparameterize_gaussian(mean, log_variance)
            if sample
            else mean
        )
        return mean, log_variance, latent

    def _validate_inputs(
        self,
        qpos: torch.Tensor,
        action_chunk: torch.Tensor,
        padding_mask: Optional[torch.Tensor],
    ) -> None:
        values = (("qpos", qpos), ("action_chunk", action_chunk))
        for name, value in values:
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"{name} must be a torch.Tensor")
            if not value.is_floating_point():
                raise ValueError(f"{name} must be floating point")

        if qpos.ndim != 2:
            raise ValueError(
                f"qpos must have shape [B, {self.config.q_dim}]"
            )
        expected_qpos = (qpos.shape[0], self.config.q_dim)
        expected_action = (
            qpos.shape[0],
            self.config.chunk_len,
            self.config.action_dim,
        )
        if tuple(qpos.shape) != expected_qpos:
            raise ValueError(
                f"qpos must have shape [B, {self.config.q_dim}]"
            )
        if tuple(action_chunk.shape) != expected_action:
            raise ValueError(
                f"action_chunk must have shape {expected_action}, "
                f"got {tuple(action_chunk.shape)}"
            )
        if qpos.shape[0] <= 0:
            raise ValueError("batch size must be positive")
        if (
            action_chunk.device != qpos.device
            or action_chunk.dtype != qpos.dtype
        ):
            raise ValueError(
                "action_chunk must share qpos device and dtype"
            )
        if padding_mask is not None:
            require_padding_mask(
                padding_mask,
                name="padding_mask",
                batch_size=qpos.shape[0],
                sequence_length=self.config.chunk_len,
            )


def _prefix_unmasked_tokens(
    padding_mask: Optional[torch.Tensor],
    *,
    prefix_length: int,
) -> Optional[torch.Tensor]:
    if padding_mask is None:
        return None
    prefix = torch.zeros(
        padding_mask.shape[0],
        prefix_length,
        dtype=torch.bool,
        device=padding_mask.device,
    )
    return torch.cat((prefix, padding_mask), dim=1)
