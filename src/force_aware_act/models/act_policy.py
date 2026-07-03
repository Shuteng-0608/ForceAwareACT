"""Structurally force-free ACT-style baseline policy."""

from __future__ import annotations

from typing import Any, Dict

import torch
from torch import nn

from force_aware_act.models.heads import ActionHead
from force_aware_act.models.policy import JointMLP
from force_aware_act.models.vision import ResNet18VisionEncoder


class ACTPolicyBaseline(nn.Module):
    """ACT-style zero-latent policy without force or contact modules.

    Online inputs:
        images: [B, N_cam, 3, H, W]
        qpos: [B, q_dim]

    Outputs:
        pred_action: [B, chunk_len, action_dim]
    """

    def __init__(
        self,
        d_model: int = 512,
        z_dim: int = 32,
        q_dim: int = 7,
        action_dim: int = 7,
        chunk_len: int = 50,
        nhead: int = 8,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 2,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        pretrained_resnet18: bool = True,
        freeze_resnet18: bool = False,
    ) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError("d_model must be positive")
        if z_dim <= 0:
            raise ValueError("z_dim must be positive")
        if q_dim <= 0:
            raise ValueError("q_dim must be positive")
        if action_dim <= 0:
            raise ValueError("action_dim must be positive")
        if chunk_len <= 0:
            raise ValueError("chunk_len must be positive")
        if nhead <= 0:
            raise ValueError("nhead must be positive")
        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead")

        self.d_model = d_model
        self.z_dim = z_dim
        self.q_dim = q_dim
        self.action_dim = action_dim
        self.chunk_len = chunk_len

        self.vision_encoder = ResNet18VisionEncoder(
            d_model=d_model,
            pretrained=pretrained_resnet18,
            freeze_backbone=freeze_resnet18,
        )
        self.joint_encoder = JointMLP(q_dim=q_dim, d_model=d_model)
        self.motion_latent_proj = nn.Linear(z_dim, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.policy_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.policy_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        self.future_queries = nn.Parameter(torch.zeros(1, chunk_len, d_model))
        self.action_head = ActionHead(d_model=d_model, action_dim=action_dim)
        self._reset_parameters()

    def forward(self, images: torch.Tensor, qpos: torch.Tensor) -> Dict[str, Any]:
        self._validate_inputs(images, qpos)
        batch_size = qpos.shape[0]

        visual_tokens = self.vision_encoder(images)
        z_q = self.joint_encoder(qpos)
        z_motion = qpos.new_zeros(batch_size, self.z_dim)
        motion_token = self.motion_latent_proj(z_motion)
        policy_tokens = torch.cat(
            [
                visual_tokens,
                z_q[:, None, :],
                motion_token[:, None, :],
            ],
            dim=1,
        )
        memory = self.policy_encoder(policy_tokens)
        query_tokens = self.future_queries.expand(batch_size, -1, -1)
        decoder_hidden = self.policy_decoder(query_tokens, memory)
        return {
            "visual_tokens": visual_tokens,
            "z_q": z_q,
            "z_motion": z_motion,
            "decoder_hidden": decoder_hidden,
            "pred_action": self.action_head(decoder_hidden),
        }

    def _validate_inputs(self, images: torch.Tensor, qpos: torch.Tensor) -> None:
        if images.ndim != 5:
            raise ValueError("images must have shape [B, N_cam, 3, H, W]")
        if images.shape[2] != 3:
            raise ValueError(f"images channel dimension must be 3, got {images.shape[2]}")
        if qpos.ndim != 2 or qpos.shape[1] != self.q_dim:
            raise ValueError(f"qpos must have shape [B, {self.q_dim}]")
        if images.shape[0] != qpos.shape[0]:
            raise ValueError("images and qpos must have the same batch size")

    def _reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.future_queries, std=0.02)
