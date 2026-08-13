"""Motion-CVAE control using native 500 Hz force intervals."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from torch import nn

from force_aware_act.models.act_aligned.config import (
    ACT_ALIGNED_HIGH_RATE_MOTION_ARCHITECTURE_VERSION,
    ACTAlignedHighRateConfig,
)
from force_aware_act.models.act_aligned.high_rate_force import (
    ACTAlignedHighRateForceEncoder,
)
from force_aware_act.models.act_aligned.motion_policy import (
    ACTAlignedMotionCVAEControlPolicy,
)
from force_aware_act.models.act_aligned.online_force import (
    ACTAlignedOnlineForceIntervalEncoder,
)


class ACTAlignedHighRateMotionCVAEPolicy(ACTAlignedMotionCVAEControlPolicy):
    """Replace the contact latent with ACT's action-only motion latent.

    Online force always follows the native-rate interval contract. Future
    force is a reconstruction target only and never enters the motion
    posterior, whose inputs remain exactly ``(qpos, action_chunk)``.
    """

    def __init__(self, config: Optional[ACTAlignedHighRateConfig] = None) -> None:
        nn.Module.__init__(self)
        self.config = config or ACTAlignedHighRateConfig.motion_control()
        config = self.config
        if (
            config.architecture_version
            != ACT_ALIGNED_HIGH_RATE_MOTION_ARCHITECTURE_VERSION
        ):
            raise ValueError(
                "ACTAlignedHighRateMotionCVAEPolicy requires the native-rate "
                "motion architecture version"
            )
        self._initialize_common_modules(config)
        self.high_rate_force_encoder = ACTAlignedHighRateForceEncoder(config)
        self.online_force_encoder = ACTAlignedOnlineForceIntervalEncoder(config)
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
        motion_latent_override: Optional[torch.Tensor] = None,
        return_intermediate_decoder: bool = False,
    ) -> Dict[str, Any]:
        """Deploy with an exact-zero motion latent unless explicitly overridden."""

        online = self._encode_online_high_rate(
            images,
            qpos,
            online_force_intervals,
            online_force_relative_time,
            online_force_sample_padding_mask,
            online_force_interval_padding_mask,
        )
        if motion_latent_override is None:
            motion_latent = qpos.new_zeros(qpos.shape[0], self.config.latent_dim)
            latent_source = "zero"
        else:
            self._validate_motion_latent_override(
                motion_latent_override,
                batch_size=qpos.shape[0],
                reference=qpos,
            )
            motion_latent = motion_latent_override
            latent_source = "override"
        decoded = self._decode(
            motion_latent,
            online,
            return_intermediate_decoder=return_intermediate_decoder,
        )
        return {
            **online,
            "z_motion": motion_latent,
            "motion_latent_source": latent_source,
            **decoded,
        }

    def forward_train(
        self,
        images: torch.Tensor,
        qpos: torch.Tensor,
        online_force_intervals: torch.Tensor,
        online_force_relative_time: torch.Tensor,
        online_force_sample_padding_mask: torch.Tensor,
        online_force_interval_padding_mask: torch.Tensor,
        action_chunk: torch.Tensor,
        *,
        action_padding_mask: Optional[torch.Tensor] = None,
        sample_posterior: bool = True,
        return_intermediate_decoder: bool = False,
    ) -> Dict[str, Any]:
        """Train with the official action-only posterior and native online force."""

        if not isinstance(sample_posterior, bool):
            raise ValueError("sample_posterior must be a bool")
        online = self._encode_online_high_rate(
            images,
            qpos,
            online_force_intervals,
            online_force_relative_time,
            online_force_sample_padding_mask,
            online_force_interval_padding_mask,
        )
        mean, log_variance, latent = self.encode_motion_posterior(
            qpos,
            action_chunk,
            padding_mask=action_padding_mask,
            sample=sample_posterior,
        )
        decoded = self._decode(
            latent,
            online,
            return_intermediate_decoder=return_intermediate_decoder,
        )
        return {
            **online,
            "mu_motion": mean,
            "logvar_motion": log_variance,
            "z_motion": latent,
            "motion_latent_source": "posterior",
            **decoded,
        }

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

    def _decode(
        self,
        motion_latent: torch.Tensor,
        online: Dict[str, torch.Tensor],
        *,
        return_intermediate_decoder: bool,
    ) -> Dict[str, torch.Tensor]:
        outputs = super()._decode(
            motion_latent,
            online,
            return_intermediate_decoder=return_intermediate_decoder,
        )
        batch_size = outputs["decoder_hidden"].shape[0]
        outputs["pred_force_highrate"] = self.high_rate_force_head(
            outputs["decoder_hidden"]
        ).reshape(
            batch_size,
            self.config.chunk_len,
            self.config.max_force_samples_per_interval,
            self.config.force_dim,
        )
        return outputs
