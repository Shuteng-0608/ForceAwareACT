"""Faithful single-arm adaptation of the official ACT DETR-VAE."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from torch import nn

from force_aware_act.models.act_aligned.transformer import (
    ACTTransformerDecoder,
    ACTTransformerEncoder,
)
from force_aware_act.models.official_act.backbone import OfficialACTBackbone
from force_aware_act.models.official_act.config import OfficialACTConfig
from force_aware_act.models.official_act.posterior import OfficialACTPosterior


class OfficialACTPolicy(nn.Module):
    """Official image/qpos/action CVAE with deployment-safe inference."""

    def __init__(self, config: Optional[OfficialACTConfig] = None) -> None:
        super().__init__()
        self.config = config or OfficialACTConfig.canonical()
        config = self.config
        self.backbone = OfficialACTBackbone(config)
        self.posterior = OfficialACTPosterior(config)
        self.latent_output_projection = nn.Linear(
            config.latent_dim,
            config.d_model,
        )
        self.robot_state_projection = nn.Linear(
            config.q_dim,
            config.d_model,
        )
        self.additional_position_embedding = nn.Embedding(2, config.d_model)
        self.policy_encoder = ACTTransformerEncoder(config)
        self.query_embedding = nn.Embedding(config.chunk_len, config.d_model)
        self.policy_decoder = ACTTransformerDecoder(config)
        self.action_head = nn.Linear(config.d_model, config.action_dim)
        self.is_pad_head = nn.Linear(config.d_model, 1)

    def forward(
        self,
        images: torch.Tensor,
        qpos: torch.Tensor,
        *,
        latent_override: Optional[torch.Tensor] = None,
        return_intermediate_decoder: bool = False,
    ) -> Dict[str, Any]:
        """Run official deployment inference with a zero latent."""

        self._validate_online(images, qpos)
        if latent_override is None:
            latent = qpos.new_zeros(qpos.shape[0], self.config.latent_dim)
            source = "zero"
        else:
            self._validate_latent(latent_override, qpos)
            latent = latent_override
            source = "override"
        decoded = self._decode(
            images,
            qpos,
            latent,
            return_intermediate_decoder=return_intermediate_decoder,
        )
        return {
            "z_motion": latent,
            "motion_latent_source": source,
            **decoded,
        }

    def forward_train(
        self,
        images: torch.Tensor,
        qpos: torch.Tensor,
        action_chunk: torch.Tensor,
        padding_mask: torch.Tensor,
        *,
        sample_posterior: bool = True,
        return_intermediate_decoder: bool = False,
    ) -> Dict[str, Any]:
        """Run the official action-conditioned posterior path."""

        self._validate_online(images, qpos)
        mean, log_variance, latent = self.posterior(
            qpos,
            action_chunk,
            padding_mask,
            sample=sample_posterior,
        )
        decoded = self._decode(
            images,
            qpos,
            latent,
            return_intermediate_decoder=return_intermediate_decoder,
        )
        return {
            "mu_motion": mean,
            "logvar_motion": log_variance,
            "z_motion": latent,
            "motion_latent_source": "posterior",
            **decoded,
        }

    def _decode(
        self,
        images: torch.Tensor,
        qpos: torch.Tensor,
        latent: torch.Tensor,
        *,
        return_intermediate_decoder: bool,
    ) -> Dict[str, torch.Tensor]:
        normalized_images = self._normalize_images(images)
        visual_tokens, visual_position = self.backbone(normalized_images)
        extra_tokens = torch.stack(
            (
                self.latent_output_projection(latent),
                self.robot_state_projection(qpos),
            ),
            dim=1,
        )
        memory_tokens = torch.cat((extra_tokens, visual_tokens), dim=1)
        extra_position = self.additional_position_embedding.weight
        extra_position = extra_position.unsqueeze(0).expand(
            qpos.shape[0],
            -1,
            -1,
        )
        memory_position = torch.cat(
            (extra_position, visual_position),
            dim=1,
        )
        memory = self.policy_encoder(
            memory_tokens,
            position=memory_position,
        )
        query_position = self.query_embedding.weight.unsqueeze(0).expand(
            qpos.shape[0],
            -1,
            -1,
        )
        target = torch.zeros_like(query_position)
        decoder = self.policy_decoder(
            target,
            memory,
            query_position=query_position,
            memory_position=memory_position,
            return_intermediate=return_intermediate_decoder,
        )
        hidden = decoder[-1] if return_intermediate_decoder else decoder
        outputs = {
            "visual_tokens": visual_tokens,
            "memory_tokens": memory_tokens,
            "policy_memory": memory,
            "decoder_hidden": hidden,
            "pred_action": self.action_head(hidden),
            "pred_is_pad": self.is_pad_head(hidden),
        }
        if return_intermediate_decoder:
            outputs["decoder_intermediate"] = decoder
        return outputs

    def _normalize_images(self, images: torch.Tensor) -> torch.Tensor:
        if not self.config.imagenet_normalize:
            return images
        mean = images.new_tensor((0.485, 0.456, 0.406)).view(
            1,
            1,
            3,
            1,
            1,
        )
        std = images.new_tensor((0.229, 0.224, 0.225)).view(
            1,
            1,
            3,
            1,
            1,
        )
        return (images - mean) / std

    def _validate_online(
        self,
        images: torch.Tensor,
        qpos: torch.Tensor,
    ) -> None:
        if not isinstance(images, torch.Tensor) or not isinstance(
            qpos,
            torch.Tensor,
        ):
            raise TypeError("images and qpos must be torch.Tensor instances")
        expected_images = (
            self.config.num_cameras,
            3,
            self.config.image_height,
            self.config.image_width,
        )
        if images.ndim != 5 or tuple(images.shape[1:]) != expected_images:
            raise ValueError("images do not match the official ACT config")
        if qpos.shape != (images.shape[0], self.config.q_dim):
            raise ValueError(f"qpos must have shape [B, {self.config.q_dim}]")
        if not images.is_floating_point() or not qpos.is_floating_point():
            raise ValueError("images and qpos must be floating point")
        if images.device != qpos.device or images.dtype != qpos.dtype:
            raise ValueError("images and qpos must share device and dtype")

    def _validate_latent(
        self,
        latent: torch.Tensor,
        qpos: torch.Tensor,
    ) -> None:
        if latent.shape != (qpos.shape[0], self.config.latent_dim):
            raise ValueError(
                f"latent_override must have shape "
                f"[B, {self.config.latent_dim}]"
            )
        if (
            not latent.is_floating_point()
            or latent.device != qpos.device
            or latent.dtype != qpos.dtype
        ):
            raise ValueError("latent_override must share qpos context")
