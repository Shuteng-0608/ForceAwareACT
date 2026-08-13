"""Latent-free force-aware policy using native 500 Hz force intervals."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from torch import nn

from force_aware_act.models.act_aligned.backbone import (
    ACTAlignedResNet18Backbone,
)
from force_aware_act.models.act_aligned.config import (
    ACT_ALIGNED_HIGH_RATE_DUAL_ZERO_ARCHITECTURE_VERSION,
    ACTAlignedHighRateConfig,
)
from force_aware_act.models.act_aligned.contracts import (
    DUAL_ZERO_POLICY_SPECIAL_TOKEN_NAMES,
)
from force_aware_act.models.act_aligned.fusion import (
    ACTAlignedForceVisionFusion,
)
from force_aware_act.models.act_aligned.high_rate_force import (
    ACTAlignedHighRateForceEncoder,
)
from force_aware_act.models.act_aligned.online_force import (
    ACTAlignedOnlineForceIntervalEncoder,
)
from force_aware_act.models.act_aligned.position_encoding import (
    TokenTypeEmbedding,
)
from force_aware_act.models.act_aligned.token_adapters import QposTokenAdapter
from force_aware_act.models.act_aligned.transformer import (
    ACTQueryDecoder,
    ACTTransformerEncoder,
)


class ACTAlignedHighRateDualZeroPolicy(nn.Module):
    """Force-aware action/force predictor with no latent mechanism at all."""

    policy_special_token_names = DUAL_ZERO_POLICY_SPECIAL_TOKEN_NAMES

    def __init__(self, config: Optional[ACTAlignedHighRateConfig] = None) -> None:
        super().__init__()
        self.config = config or ACTAlignedHighRateConfig.dual_zero()
        config = self.config
        if (
            config.architecture_version
            != ACT_ALIGNED_HIGH_RATE_DUAL_ZERO_ARCHITECTURE_VERSION
        ):
            raise ValueError(
                "ACTAlignedHighRateDualZeroPolicy requires the native-rate "
                "dual-zero architecture version"
            )
        self.vision_backbone = ACTAlignedResNet18Backbone(config)
        self.qpos_adapter = QposTokenAdapter(config.q_dim, config.d_model)
        self.high_rate_force_encoder = ACTAlignedHighRateForceEncoder(config)
        self.online_force_encoder = ACTAlignedOnlineForceIntervalEncoder(config)
        self.force_vision_fusion = ACTAlignedForceVisionFusion(config)
        self.policy_special_position = TokenTypeEmbedding(
            len(DUAL_ZERO_POLICY_SPECIAL_TOKEN_NAMES),
            config.d_model,
        )
        self.policy_encoder = ACTTransformerEncoder(config)
        self.query_decoder = ACTQueryDecoder(config)
        self.action_head = nn.Linear(config.d_model, config.action_dim)
        self.force_head = nn.Linear(config.d_model, config.force_dim)
        self.high_rate_force_head = nn.Linear(
            config.d_model,
            config.max_force_samples_per_interval * config.force_dim,
        )

    def forward(
        self,
        images: torch.Tensor,
        qpos: torch.Tensor,
        online_force_intervals: torch.Tensor,
        online_force_relative_time: torch.Tensor,
        online_force_sample_padding_mask: torch.Tensor,
        online_force_interval_padding_mask: torch.Tensor,
        *,
        return_intermediate_decoder: bool = False,
    ) -> Dict[str, Any]:
        """Predict from online observations; no future or latent API exists."""

        online = self._encode_online_high_rate(
            images,
            qpos,
            online_force_intervals,
            online_force_relative_time,
            online_force_sample_padding_mask,
            online_force_interval_padding_mask,
        )
        policy_tokens, policy_position = self._assemble_policy_input(online)
        policy_memory = self.policy_encoder(
            policy_tokens,
            position=policy_position,
        )
        decoder_output = self.query_decoder(
            policy_memory,
            memory_position=policy_position,
            return_intermediate=return_intermediate_decoder,
        )
        if return_intermediate_decoder:
            decoder_hidden = decoder_output[-1]
            decoder_intermediate = decoder_output
        else:
            decoder_hidden = decoder_output
            decoder_intermediate = None
        batch_size = decoder_hidden.shape[0]
        outputs: Dict[str, Any] = {
            **online,
            "latent_mechanism": "none",
            "policy_tokens": policy_tokens,
            "policy_position": policy_position,
            "policy_memory": policy_memory,
            "decoder_hidden": decoder_hidden,
            "pred_action": self.action_head(decoder_hidden),
            "pred_force": self.force_head(decoder_hidden),
            "pred_force_highrate": self.high_rate_force_head(
                decoder_hidden
            ).reshape(
                batch_size,
                self.config.chunk_len,
                self.config.max_force_samples_per_interval,
                self.config.force_dim,
            ),
        }
        if decoder_intermediate is not None:
            outputs["decoder_intermediate"] = decoder_intermediate
        return outputs

    def _encode_online_high_rate(
        self,
        images: torch.Tensor,
        qpos: torch.Tensor,
        force_intervals: torch.Tensor,
        relative_time: torch.Tensor,
        sample_padding_mask: torch.Tensor,
        interval_padding_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        self._validate_online_context(images, qpos, force_intervals)
        if (
            force_intervals.ndim != 4
            or force_intervals.shape[1] != self.config.max_online_force_intervals
        ):
            raise ValueError(
                "online_force_intervals must have shape "
                f"[B, {self.config.max_online_force_intervals}, S, F]"
            )
        visual_tokens, visual_position = self.vision_backbone(images)
        qpos_token = self.qpos_adapter(qpos)
        interval_tokens = self.high_rate_force_encoder(
            force_intervals,
            relative_time,
            sample_padding_mask,
            interval_padding_mask,
        )
        online_force = self.online_force_encoder(
            interval_tokens,
            interval_padding_mask=interval_padding_mask,
        )
        force_vision = self.force_vision_fusion(
            online_force,
            visual_tokens,
            visual_position=visual_position,
        )
        return {
            "visual_tokens": visual_tokens,
            "visual_position": visual_position,
            "visual_summary": visual_tokens.mean(dim=1),
            "qpos_token": qpos_token,
            "qpos_feature": qpos_token[:, 0],
            "online_force_interval_tokens": interval_tokens,
            "z_F_online": online_force,
            "z_VF": force_vision,
        }

    def _assemble_policy_input(
        self,
        online: Dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        special_tokens = torch.cat(
            (
                online["qpos_token"],
                online["z_F_online"].unsqueeze(1),
                online["z_VF"].unsqueeze(1),
            ),
            dim=1,
        )
        policy_tokens = torch.cat(
            (special_tokens, online["visual_tokens"]),
            dim=1,
        )
        special_ids = torch.arange(
            len(DUAL_ZERO_POLICY_SPECIAL_TOKEN_NAMES),
            dtype=torch.long,
            device=policy_tokens.device,
        )
        special_position = self.policy_special_position(special_ids)
        special_position = special_position.unsqueeze(0).expand(
            policy_tokens.shape[0],
            -1,
            -1,
        )
        policy_position = torch.cat(
            (special_position, online["visual_position"]),
            dim=1,
        )
        expected = (
            policy_tokens.shape[0],
            self.config.policy_memory_token_count,
            self.config.d_model,
        )
        if tuple(policy_tokens.shape) != expected:
            raise RuntimeError(
                f"policy token assembly expected {expected}, "
                f"got {tuple(policy_tokens.shape)}"
            )
        if tuple(policy_position.shape) != expected:
            raise RuntimeError(
                f"policy position assembly expected {expected}, "
                f"got {tuple(policy_position.shape)}"
            )
        return policy_tokens, policy_position

    @staticmethod
    def _validate_online_context(
        images: torch.Tensor,
        qpos: torch.Tensor,
        force_intervals: torch.Tensor,
    ) -> None:
        values = (
            ("images", images),
            ("qpos", qpos),
            ("online_force_intervals", force_intervals),
        )
        for name, value in values:
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"{name} must be a torch.Tensor")
            if not value.is_floating_point():
                raise ValueError(f"{name} must be floating point")
        batch_size = images.shape[0]
        if batch_size <= 0:
            raise ValueError("batch size must be positive")
        if qpos.shape[0] != batch_size or force_intervals.shape[0] != batch_size:
            raise ValueError("all online inputs must have the same batch size")
        for name, value in values[1:]:
            if value.device != images.device or value.dtype != images.dtype:
                raise ValueError(
                    f"{name} must share images device and dtype"
                )
