"""Independent action-only motion-latent control policy."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from torch import nn

from force_aware_act.models.act_aligned.backbone import (
    ACTAlignedResNet18Backbone,
)
from force_aware_act.models.act_aligned.config import (
    ACT_ALIGNED_MOTION_CONTROL_ARCHITECTURE_VERSION,
    ACTAlignedConfig,
)
from force_aware_act.models.act_aligned.contracts import (
    MOTION_POLICY_SPECIAL_TOKEN_NAMES,
)
from force_aware_act.models.act_aligned.fusion import (
    ACTAlignedForceVisionFusion,
)
from force_aware_act.models.act_aligned.motion_latent import (
    ACTAlignedMotionPosterior,
)
from force_aware_act.models.act_aligned.online_force import (
    ACTAlignedOnlineForceEncoder,
)
from force_aware_act.models.act_aligned.position_encoding import (
    TokenTypeEmbedding,
)
from force_aware_act.models.act_aligned.token_adapters import (
    LatentTokenAdapter,
    QposTokenAdapter,
)
from force_aware_act.models.act_aligned.transformer import (
    ACTQueryDecoder,
    ACTTransformerEncoder,
)


class ACTAlignedMotionCVAEControlPolicy(nn.Module):
    """Force-aware policy with the official ACT action-only latent mechanism.

    The online observation path and action/force prediction heads deliberately
    match the contact-CVAE model. The isolated experimental variable is the
    latent mechanism:

    - training posterior: ``q(z_motion | qpos, action_chunk)``;
    - deployment latent: fixed zero;
    - no learned conditional prior;
    - no future-force input to the posterior.
    """

    policy_special_token_names = MOTION_POLICY_SPECIAL_TOKEN_NAMES

    def __init__(self, config: Optional[ACTAlignedConfig] = None) -> None:
        super().__init__()
        self.config = config or ACTAlignedConfig.motion_control()
        config = self.config
        if (
            config.architecture_version
            != ACT_ALIGNED_MOTION_CONTROL_ARCHITECTURE_VERSION
        ):
            raise ValueError(
                "ACTAlignedMotionCVAEControlPolicy requires the "
                "motion-control architecture version"
            )

        self._initialize_common_modules(config)

    def _initialize_common_modules(self, config: ACTAlignedConfig) -> None:
        """Build modules shared by state-rate and native-rate motion controls."""

        self.vision_backbone = ACTAlignedResNet18Backbone(config)
        self.qpos_adapter = QposTokenAdapter(config.q_dim, config.d_model)
        self.online_force_encoder = ACTAlignedOnlineForceEncoder(config)
        self.force_vision_fusion = ACTAlignedForceVisionFusion(config)
        self.motion_posterior = ACTAlignedMotionPosterior(config)
        self.motion_latent_adapter = LatentTokenAdapter(
            config.latent_dim,
            config.d_model,
        )

        self.policy_special_position = TokenTypeEmbedding(
            len(MOTION_POLICY_SPECIAL_TOKEN_NAMES),
            config.d_model,
        )
        self.policy_encoder = ACTTransformerEncoder(config)
        self.query_decoder = ACTQueryDecoder(config)
        self.action_head = nn.Linear(config.d_model, config.action_dim)
        self.force_head = nn.Linear(config.d_model, config.force_dim)

    def forward(
        self,
        images: torch.Tensor,
        qpos: torch.Tensor,
        force_history: torch.Tensor,
        *,
        force_padding_mask: Optional[torch.Tensor] = None,
        motion_latent_override: Optional[torch.Tensor] = None,
        return_intermediate_decoder: bool = False,
    ) -> Dict[str, Any]:
        """Run deployment inference with zero ``z_motion`` by default."""

        if not isinstance(return_intermediate_decoder, bool):
            raise ValueError("return_intermediate_decoder must be a bool")
        online = self._encode_online(
            images,
            qpos,
            force_history,
            force_padding_mask=force_padding_mask,
        )
        if motion_latent_override is None:
            motion_latent = qpos.new_zeros(
                qpos.shape[0],
                self.config.latent_dim,
            )
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
        force_history: torch.Tensor,
        action_chunk: torch.Tensor,
        *,
        force_padding_mask: Optional[torch.Tensor] = None,
        future_padding_mask: Optional[torch.Tensor] = None,
        sample_posterior: bool = True,
        return_intermediate_decoder: bool = False,
    ) -> Dict[str, Any]:
        """Run training with the action-only motion posterior."""

        if not isinstance(sample_posterior, bool):
            raise ValueError("sample_posterior must be a bool")
        if not isinstance(return_intermediate_decoder, bool):
            raise ValueError("return_intermediate_decoder must be a bool")
        online = self._encode_online(
            images,
            qpos,
            force_history,
            force_padding_mask=force_padding_mask,
        )
        posterior_mean, posterior_log_variance, posterior_latent = (
            self.encode_motion_posterior(
                qpos,
                action_chunk,
                padding_mask=future_padding_mask,
                sample=sample_posterior,
            )
        )
        decoded = self._decode(
            posterior_latent,
            online,
            return_intermediate_decoder=return_intermediate_decoder,
        )
        return {
            **online,
            "mu_motion": posterior_mean,
            "logvar_motion": posterior_log_variance,
            "z_motion": posterior_latent,
            "motion_latent_source": "posterior",
            **decoded,
        }

    def encode_motion_posterior(
        self,
        qpos: torch.Tensor,
        action_chunk: torch.Tensor,
        *,
        padding_mask: Optional[torch.Tensor] = None,
        sample: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Expose the action-only posterior for offline diagnostics."""

        return self.motion_posterior(
            qpos,
            action_chunk,
            padding_mask=padding_mask,
            sample=sample,
        )

    def _encode_online(
        self,
        images: torch.Tensor,
        qpos: torch.Tensor,
        force_history: torch.Tensor,
        *,
        force_padding_mask: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        self._validate_online_context(images, qpos, force_history)
        visual_tokens, visual_position = self.vision_backbone(images)
        qpos_token = self.qpos_adapter(qpos)
        online_force = self.online_force_encoder(
            force_history,
            padding_mask=force_padding_mask,
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
        policy_tokens, policy_position = self._assemble_policy_input(
            motion_latent=motion_latent,
            qpos_token=online["qpos_token"],
            online_force=online["z_F_online"],
            force_vision=online["z_VF"],
            visual_tokens=online["visual_tokens"],
            visual_position=online["visual_position"],
        )
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

        outputs = {
            "policy_tokens": policy_tokens,
            "policy_position": policy_position,
            "policy_memory": policy_memory,
            "decoder_hidden": decoder_hidden,
            "pred_action": self.action_head(decoder_hidden),
            "pred_force": self.force_head(decoder_hidden),
        }
        if decoder_intermediate is not None:
            outputs["decoder_intermediate"] = decoder_intermediate
        return outputs

    def _assemble_policy_input(
        self,
        *,
        motion_latent: torch.Tensor,
        qpos_token: torch.Tensor,
        online_force: torch.Tensor,
        force_vision: torch.Tensor,
        visual_tokens: torch.Tensor,
        visual_position: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = visual_tokens.shape[0]
        motion_token = self.motion_latent_adapter(motion_latent)
        special_tokens = torch.cat(
            (
                motion_token,
                qpos_token,
                online_force.unsqueeze(1),
                force_vision.unsqueeze(1),
            ),
            dim=1,
        )
        policy_tokens = torch.cat((special_tokens, visual_tokens), dim=1)

        special_ids = torch.arange(
            len(MOTION_POLICY_SPECIAL_TOKEN_NAMES),
            dtype=torch.long,
            device=policy_tokens.device,
        )
        special_position = self.policy_special_position(
            special_ids
        ).unsqueeze(0).expand(batch_size, -1, -1)
        policy_position = torch.cat(
            (special_position, visual_position),
            dim=1,
        )

        expected_shape = (
            batch_size,
            self.config.policy_memory_token_count,
            self.config.d_model,
        )
        if policy_tokens.shape != expected_shape:
            raise RuntimeError(
                f"policy token assembly expected {expected_shape}, "
                f"got {tuple(policy_tokens.shape)}"
            )
        if policy_position.shape != expected_shape:
            raise RuntimeError(
                f"policy position assembly expected {expected_shape}, "
                f"got {tuple(policy_position.shape)}"
            )
        return policy_tokens, policy_position

    def _validate_online_context(
        self,
        images: torch.Tensor,
        qpos: torch.Tensor,
        force_history: torch.Tensor,
    ) -> None:
        values = (
            ("images", images),
            ("qpos", qpos),
            ("force_history", force_history),
        )
        for name, value in values:
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"{name} must be a torch.Tensor")
            if not value.is_floating_point():
                raise ValueError(f"{name} must be floating point")
        if images.ndim == 0 or qpos.ndim == 0 or force_history.ndim == 0:
            raise ValueError("online inputs must include a batch dimension")
        batch_size = images.shape[0]
        if batch_size <= 0:
            raise ValueError("batch size must be positive")
        if qpos.shape[0] != batch_size or force_history.shape[0] != batch_size:
            raise ValueError(
                "images, qpos, and force_history must have the same batch size"
            )
        for name, value in values[1:]:
            if value.device != images.device:
                raise ValueError(f"{name} must be on the same device as images")
            if value.dtype != images.dtype:
                raise ValueError(f"{name} must have the same dtype as images")

    def _validate_motion_latent_override(
        self,
        latent: torch.Tensor,
        *,
        batch_size: int,
        reference: torch.Tensor,
    ) -> None:
        if not isinstance(latent, torch.Tensor):
            raise TypeError("motion_latent_override must be a torch.Tensor")
        expected_shape = (batch_size, self.config.latent_dim)
        if latent.shape != expected_shape:
            raise ValueError(
                f"motion_latent_override must have shape {expected_shape}"
            )
        if not latent.is_floating_point():
            raise ValueError("motion_latent_override must be floating point")
        if latent.device != reference.device or latent.dtype != reference.dtype:
            raise ValueError(
                "motion_latent_override must share qpos device and dtype"
            )
