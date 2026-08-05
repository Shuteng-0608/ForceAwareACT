"""ACT-aligned contact-CVAE with native 500 Hz force intervals."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from torch import nn

from force_aware_act.models.act_aligned.config import ACTAlignedHighRateConfig
from force_aware_act.models.act_aligned.contact_latent import (
    ACTAlignedHighRateContactPosterior,
)
from force_aware_act.models.act_aligned.high_rate_force import (
    ACTAlignedHighRateForceEncoder,
)
from force_aware_act.models.act_aligned.online_force import (
    ACTAlignedOnlineForceIntervalEncoder,
)
from force_aware_act.models.act_aligned.policy import ACTAlignedContactCVAEPolicy


class ACTAlignedHighRateContactCVAEPolicy(ACTAlignedContactCVAEPolicy):
    """V2 policy with one shared local encoder for online and future force."""

    def __init__(self, config: Optional[ACTAlignedHighRateConfig] = None) -> None:
        nn.Module.__init__(self)
        self.config = config or ACTAlignedHighRateConfig.canonical_act()
        config = self.config
        self._initialize_common_modules(config)
        self.high_rate_force_encoder = ACTAlignedHighRateForceEncoder(config)
        self.online_force_encoder = ACTAlignedOnlineForceIntervalEncoder(config)
        self.contact_posterior = ACTAlignedHighRateContactPosterior(config)
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
        contact_latent_mode: str = "zero",
        deterministic_prior: bool = True,
        contact_latent_override: Optional[torch.Tensor] = None,
        return_intermediate_decoder: bool = False,
    ) -> Dict[str, Any]:
        if contact_latent_mode not in {"zero", "prior"}:
            raise ValueError("contact_latent_mode must be 'zero' or 'prior'")
        if not isinstance(deterministic_prior, bool):
            raise ValueError("deterministic_prior must be a bool")
        online = self._encode_online_high_rate(
            images,
            qpos,
            online_force_intervals,
            online_force_relative_time,
            online_force_sample_padding_mask,
            online_force_interval_padding_mask,
        )
        prior_mean, prior_log_variance, prior_latent = self.contact_prior(
            online["qpos_feature"],
            online["z_F_online"],
            online["z_VF"],
            online["visual_summary"],
            deterministic=(
                True
                if contact_latent_override is not None
                or contact_latent_mode == "zero"
                else deterministic_prior
            ),
        )
        if contact_latent_override is not None:
            self._validate_contact_latent_override(
                contact_latent_override,
                batch_size=qpos.shape[0],
                reference=qpos,
            )
            contact_latent = contact_latent_override
            latent_source = "override"
        elif contact_latent_mode == "zero":
            contact_latent = qpos.new_zeros(qpos.shape[0], self.config.latent_dim)
            latent_source = "zero"
        else:
            contact_latent = prior_latent
            latent_source = "prior_mean" if deterministic_prior else "prior_sample"
        decoded = self._decode(
            contact_latent,
            online,
            return_intermediate_decoder=return_intermediate_decoder,
        )
        return {
            **online,
            "mu_contact_prior": prior_mean,
            "logvar_contact_prior": prior_log_variance,
            "z_contact_prior": prior_latent,
            "z_contact": contact_latent,
            "contact_latent_source": latent_source,
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
        future_force_intervals: torch.Tensor,
        future_force_relative_time: torch.Tensor,
        future_force_sample_padding_mask: torch.Tensor,
        future_force_interval_padding_mask: torch.Tensor,
        *,
        action_padding_mask: Optional[torch.Tensor] = None,
        sample_posterior: bool = True,
        return_intermediate_decoder: bool = False,
    ) -> Dict[str, Any]:
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
        future_force_tokens = self.high_rate_force_encoder(
            future_force_intervals,
            future_force_relative_time,
            future_force_sample_padding_mask,
            future_force_interval_padding_mask,
        )
        posterior_mean, posterior_log_variance, posterior_latent = (
            self.contact_posterior(
                qpos,
                action_chunk,
                future_force_tokens,
                action_padding_mask=action_padding_mask,
                future_force_interval_padding_mask=(
                    future_force_interval_padding_mask
                ),
                sample=sample_posterior,
            )
        )
        # Prior matching is intentionally isolated from all shared encoders.
        prior_mean, prior_log_variance, prior_latent = self.contact_prior(
            online["qpos_feature"].detach(),
            online["z_F_online"].detach(),
            online["z_VF"].detach(),
            online["visual_summary"].detach(),
            deterministic=not sample_posterior,
        )
        decoded = self._decode(
            posterior_latent,
            online,
            return_intermediate_decoder=return_intermediate_decoder,
        )
        return {
            **online,
            "future_force_interval_tokens": future_force_tokens,
            "mu_contact": posterior_mean,
            "logvar_contact": posterior_log_variance,
            "z_contact_posterior": posterior_latent,
            "mu_contact_prior": prior_mean,
            "logvar_contact_prior": prior_log_variance,
            "z_contact_prior": prior_latent,
            "z_contact": posterior_latent,
            "contact_latent_source": "posterior",
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
        expected_intervals = self.config.max_online_force_intervals
        if force_intervals.ndim != 4 or force_intervals.shape[1] != expected_intervals:
            raise ValueError(
                "online_force_intervals must have shape "
                f"[B, {expected_intervals}, S, F]"
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
        contact_latent: torch.Tensor,
        online: Dict[str, torch.Tensor],
        *,
        return_intermediate_decoder: bool,
    ) -> Dict[str, torch.Tensor]:
        outputs = super()._decode(
            contact_latent,
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
