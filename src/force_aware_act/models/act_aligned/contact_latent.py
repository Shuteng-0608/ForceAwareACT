"""Contact posterior and causal conditional-prior modules."""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from force_aware_act.models.act_aligned.config import (
    ACTAlignedConfig,
    ACTAlignedHighRateConfig,
)
from force_aware_act.models.act_aligned.contracts import require_padding_mask
from force_aware_act.models.act_aligned.position_encoding import (
    SinusoidalSequencePositionEncoding,
    TokenTypeEmbedding,
)
from force_aware_act.models.act_aligned.token_adapters import (
    ActionTokenAdapter,
    ForceTokenAdapter,
    QposTokenAdapter,
)
from force_aware_act.models.act_aligned.transformer import ACTTransformerEncoder


CONTACT_CLS_TYPE = 0
CONTACT_QPOS_TYPE = 1
CONTACT_STEP_TYPE = 2


def reparameterize_gaussian(
    mean: torch.Tensor,
    log_variance: torch.Tensor,
) -> torch.Tensor:
    """Sample a diagonal Gaussian while preserving the gradient path."""

    if not isinstance(mean, torch.Tensor) or not isinstance(
        log_variance,
        torch.Tensor,
    ):
        raise TypeError("mean and log_variance must be torch.Tensor instances")
    if mean.shape != log_variance.shape:
        raise ValueError("mean and log_variance must have the same shape")
    if not mean.is_floating_point() or not log_variance.is_floating_point():
        raise ValueError("mean and log_variance must be floating point")
    standard_deviation = torch.exp(0.5 * log_variance)
    return mean + torch.randn_like(standard_deviation) * standard_deviation


class ACTAlignedContactPosterior(nn.Module):
    """Encode time-aligned future action/force labels during training only.

    Sequence layout:
    ``[CLS, qpos, action_0 + force_0, ..., action_K-1 + force_K-1]``.
    """

    def __init__(self, config: ACTAlignedConfig) -> None:
        super().__init__()
        self.config = config
        self.qpos_adapter = QposTokenAdapter(config.q_dim, config.d_model)
        self.action_adapter = ActionTokenAdapter(config.action_dim, config.d_model)
        self.force_adapter = ForceTokenAdapter(config.force_dim, config.d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.d_model))
        self.sequence_position = SinusoidalSequencePositionEncoding(
            config.contact_posterior_token_count,
            config.d_model,
        )
        self.token_type = TokenTypeEmbedding(3, config.d_model)
        self.encoder = ACTTransformerEncoder(config)
        self.mean_head = nn.Linear(config.d_model, config.latent_dim)
        self.log_variance_head = nn.Linear(config.d_model, config.latent_dim)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(
        self,
        qpos: torch.Tensor,
        action_chunk: torch.Tensor,
        future_force_chunk: torch.Tensor,
        *,
        padding_mask: Optional[torch.Tensor] = None,
        sample: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not isinstance(sample, bool):
            raise ValueError("sample must be a bool")
        self._validate_inputs(
            qpos,
            action_chunk,
            future_force_chunk,
            padding_mask,
        )
        batch_size = qpos.shape[0]
        cls_token = self.cls_token.expand(batch_size, -1, -1)
        qpos_token = self.qpos_adapter(qpos)
        contact_steps = self.action_adapter(action_chunk) + self.force_adapter(
            future_force_chunk
        )
        tokens = torch.cat((cls_token, qpos_token, contact_steps), dim=1)

        type_ids = torch.full(
            (self.config.contact_posterior_token_count,),
            CONTACT_STEP_TYPE,
            dtype=torch.long,
            device=tokens.device,
        )
        type_ids[0] = CONTACT_CLS_TYPE
        type_ids[1] = CONTACT_QPOS_TYPE
        position = self.sequence_position(tokens) + self.token_type(type_ids).unsqueeze(0)
        encoder_padding_mask = _prefix_unmasked_tokens(padding_mask, prefix_length=2)
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
        future_force_chunk: torch.Tensor,
        padding_mask: Optional[torch.Tensor],
    ) -> None:
        batch_size = _require_floating_shape(
            qpos,
            "qpos",
            (None, self.config.q_dim),
        )
        _require_floating_shape(
            action_chunk,
            "action_chunk",
            (batch_size, self.config.chunk_len, self.config.action_dim),
        )
        _require_floating_shape(
            future_force_chunk,
            "future_force_chunk",
            (batch_size, self.config.chunk_len, self.config.force_dim),
        )
        _require_same_context(
            qpos,
            action_chunk,
            future_force_chunk,
            names=("qpos", "action_chunk", "future_force_chunk"),
        )
        if padding_mask is not None:
            require_padding_mask(
                padding_mask,
                name="padding_mask",
                batch_size=batch_size,
                sequence_length=self.config.chunk_len,
            )


class ACTAlignedHighRateContactPosterior(nn.Module):
    """Encode actions fused with pre-encoded native-force interval tokens."""

    def __init__(self, config: ACTAlignedHighRateConfig) -> None:
        super().__init__()
        self.config = config
        self.qpos_adapter = QposTokenAdapter(config.q_dim, config.d_model)
        self.action_adapter = ActionTokenAdapter(config.action_dim, config.d_model)
        self.contact_step_norm = nn.LayerNorm(config.d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.d_model))
        self.sequence_position = SinusoidalSequencePositionEncoding(
            config.contact_posterior_token_count,
            config.d_model,
        )
        self.token_type = TokenTypeEmbedding(3, config.d_model)
        self.encoder = ACTTransformerEncoder(config)
        self.mean_head = nn.Linear(config.d_model, config.latent_dim)
        self.log_variance_head = nn.Linear(config.d_model, config.latent_dim)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(
        self,
        qpos: torch.Tensor,
        action_chunk: torch.Tensor,
        future_force_tokens: torch.Tensor,
        *,
        action_padding_mask: Optional[torch.Tensor] = None,
        future_force_interval_padding_mask: Optional[torch.Tensor] = None,
        sample: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not isinstance(sample, bool):
            raise ValueError("sample must be a bool")
        self._validate_high_rate_inputs(
            qpos,
            action_chunk,
            future_force_tokens,
            action_padding_mask,
            future_force_interval_padding_mask,
        )
        batch_size = qpos.shape[0]
        action_tokens = self.action_adapter(action_chunk)
        if future_force_interval_padding_mask is not None:
            future_force_tokens = future_force_tokens.masked_fill(
                future_force_interval_padding_mask.unsqueeze(-1),
                0.0,
            )
        contact_steps = self.contact_step_norm(action_tokens + future_force_tokens)
        cls_token = self.cls_token.expand(batch_size, -1, -1)
        qpos_token = self.qpos_adapter(qpos)
        tokens = torch.cat((cls_token, qpos_token, contact_steps), dim=1)

        type_ids = torch.full(
            (self.config.contact_posterior_token_count,),
            CONTACT_STEP_TYPE,
            dtype=torch.long,
            device=tokens.device,
        )
        type_ids[0] = CONTACT_CLS_TYPE
        type_ids[1] = CONTACT_QPOS_TYPE
        position = self.sequence_position(tokens) + self.token_type(type_ids).unsqueeze(0)
        encoder_padding_mask = _prefix_unmasked_tokens(
            action_padding_mask,
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
        latent = reparameterize_gaussian(mean, log_variance) if sample else mean
        return mean, log_variance, latent

    def _validate_high_rate_inputs(
        self,
        qpos: torch.Tensor,
        action_chunk: torch.Tensor,
        future_force_tokens: torch.Tensor,
        action_padding_mask: Optional[torch.Tensor],
        future_force_interval_padding_mask: Optional[torch.Tensor],
    ) -> None:
        batch_size = _require_floating_shape(
            qpos,
            "qpos",
            (None, self.config.q_dim),
        )
        _require_floating_shape(
            action_chunk,
            "action_chunk",
            (batch_size, self.config.chunk_len, self.config.action_dim),
        )
        _require_floating_shape(
            future_force_tokens,
            "future_force_tokens",
            (batch_size, self.config.chunk_len, self.config.d_model),
        )
        _require_same_context(
            qpos,
            action_chunk,
            future_force_tokens,
            names=("qpos", "action_chunk", "future_force_tokens"),
        )
        for name, mask in (
            ("action_padding_mask", action_padding_mask),
            (
                "future_force_interval_padding_mask",
                future_force_interval_padding_mask,
            ),
        ):
            if mask is not None:
                require_padding_mask(
                    mask,
                    name=name,
                    batch_size=batch_size,
                    sequence_length=self.config.chunk_len,
                )


class ACTAlignedContactPrior(nn.Module):
    """Predict a contact latent using only deployment-available features."""

    def __init__(self, config: ACTAlignedConfig) -> None:
        super().__init__()
        self.config = config
        self.trunk = nn.Sequential(
            nn.Linear(4 * config.d_model, config.d_model),
            nn.ReLU(),
            nn.Dropout(config.dropout),
        )
        self.mean_head = nn.Linear(config.d_model, config.latent_dim)
        self.log_variance_head = nn.Linear(config.d_model, config.latent_dim)

    def forward(
        self,
        qpos_feature: torch.Tensor,
        online_force_feature: torch.Tensor,
        force_vision_feature: torch.Tensor,
        visual_summary: torch.Tensor,
        *,
        deterministic: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not isinstance(deterministic, bool):
            raise ValueError("deterministic must be a bool")
        features = (
            ("qpos_feature", qpos_feature),
            ("online_force_feature", online_force_feature),
            ("force_vision_feature", force_vision_feature),
            ("visual_summary", visual_summary),
        )
        batch_size = None
        for name, feature in features:
            current_batch_size = _require_floating_shape(
                feature,
                name,
                (batch_size, self.config.d_model),
            )
            if batch_size is None:
                batch_size = current_batch_size

        _require_same_context(
            *(feature for _name, feature in features),
            names=tuple(name for name, _feature in features),
        )
        concatenated = torch.cat([feature for _name, feature in features], dim=-1)
        hidden = self.trunk(concatenated)
        mean = self.mean_head(hidden)
        log_variance = self.log_variance_head(hidden)
        latent = (
            mean
            if deterministic
            else reparameterize_gaussian(mean, log_variance)
        )
        return mean, log_variance, latent


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


def _require_floating_shape(
    tensor: torch.Tensor,
    name: str,
    expected_shape: tuple[Optional[int], ...],
) -> int:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.ndim != len(expected_shape):
        raise ValueError(f"{name} must have shape {expected_shape}")
    for actual, expected in zip(tensor.shape, expected_shape):
        if expected is not None and actual != expected:
            raise ValueError(
                f"{name} must have shape {expected_shape}, got {tuple(tensor.shape)}"
            )
    if not tensor.is_floating_point():
        raise ValueError(f"{name} must be floating point")
    if tensor.shape[0] <= 0:
        raise ValueError(f"{name} batch dimension must be positive")
    return tensor.shape[0]


def _require_same_context(
    *tensors: torch.Tensor,
    names: tuple[str, ...],
) -> None:
    reference = tensors[0]
    for tensor, name in zip(tensors[1:], names[1:]):
        if tensor.device != reference.device:
            raise ValueError(f"{name} must be on the same device as {names[0]}")
        if tensor.dtype != reference.dtype:
            raise ValueError(f"{name} must have the same dtype as {names[0]}")
