"""Structurally latent-free variant of the official ACT baseline."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from torch import nn

from force_aware_act.models.act_aligned.transformer import (
    ACTTransformerDecoder,
    ACTTransformerEncoder,
)
from force_aware_act.models.official_act.backbone import OfficialACTBackbone
from force_aware_act.models.official_act.config import (
    OFFICIAL_ACT_NO_LATENT_ARCHITECTURE_VERSION,
    OfficialACTNoLatentConfig,
)


class OfficialACTNoLatentPolicy(nn.Module):
    """Official ACT image/qpos policy with no latent modules or tokens."""

    def __init__(
        self,
        config: Optional[OfficialACTNoLatentConfig] = None,
    ) -> None:
        super().__init__()
        self.config = config or OfficialACTNoLatentConfig.canonical()
        config = self.config
        if (
            config.architecture_version
            != OFFICIAL_ACT_NO_LATENT_ARCHITECTURE_VERSION
        ):
            raise ValueError(
                "OfficialACTNoLatentPolicy requires the no-latent "
                "architecture version"
            )
        self.backbone = OfficialACTBackbone(config)
        self.robot_state_projection = nn.Linear(
            config.q_dim,
            config.d_model,
        )
        self.additional_position_embedding = nn.Embedding(1, config.d_model)
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
        return_intermediate_decoder: bool = False,
    ) -> Dict[str, Any]:
        """Predict an action chunk from online image and qpos observations."""

        self._validate_online(images, qpos)
        normalized_images = self._normalize_images(images)
        visual_tokens, visual_position = self.backbone(normalized_images)
        qpos_token = self.robot_state_projection(qpos).unsqueeze(1)
        memory_tokens = torch.cat((qpos_token, visual_tokens), dim=1)
        qpos_position = self.additional_position_embedding.weight
        qpos_position = qpos_position.unsqueeze(0).expand(
            qpos.shape[0],
            -1,
            -1,
        )
        memory_position = torch.cat(
            (qpos_position, visual_position),
            dim=1,
        )
        expected = (
            qpos.shape[0],
            self.config.policy_memory_token_count,
            self.config.d_model,
        )
        if tuple(memory_tokens.shape) != expected:
            raise RuntimeError(
                f"policy memory tokens expected {expected}, "
                f"got {tuple(memory_tokens.shape)}"
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
        outputs: Dict[str, Any] = {
            "latent_mechanism": "none",
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
            raise ValueError("images do not match the no-latent ACT config")
        if qpos.shape != (images.shape[0], self.config.q_dim):
            raise ValueError(f"qpos must have shape [B, {self.config.q_dim}]")
        if not images.is_floating_point() or not qpos.is_floating_point():
            raise ValueError("images and qpos must be floating point")
        if images.device != qpos.device or images.dtype != qpos.dtype:
            raise ValueError("images and qpos must share device and dtype")
