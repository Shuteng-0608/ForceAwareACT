"""Structurally force-free ACT-style Motion-CVAE baseline policy."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from torch import nn

from force_aware_act.models.heads import ActionHead
from force_aware_act.models.policy import JointMLP
from force_aware_act.models.posterior import MotionPosteriorEncoder
from force_aware_act.models.vision import ResNet18VisionEncoder


class ACTPolicyBaseline(nn.Module):
    """ACT-style Motion-CVAE policy without force or contact modules.

    Training inputs:
        images: [B, N_cam, 3, H, W]
        qpos: [B, q_dim]
        action_chunk: [B, chunk_len, action_dim]

    Deployment inputs:
        images: [B, N_cam, 3, H, W]
        qpos: [B, q_dim]

    Outputs:
        pred_action: [B, chunk_len, action_dim]

    Training samples z_motion from q(z | q_t, a_{t:t+K}). Deployment uses an
    exact zero motion latent and never reads future action labels.
    """

    act_baseline_version = "motion_cvae_v1"

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
        self.motion_posterior = MotionPosteriorEncoder(
            q_dim=q_dim,
            action_dim=action_dim,
            d_model=d_model,
            z_dim=z_dim,
            nhead=nhead,
            num_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_chunk_len=chunk_len,
        )
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

    def forward(
        self,
        images: torch.Tensor,
        qpos: torch.Tensor,
        action_chunk: Optional[torch.Tensor] = None,
        is_training: bool = True,
        motion_latent_override: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        self._validate_inputs(images, qpos)
        self._validate_forward_state(action_chunk, is_training)
        batch_size = qpos.shape[0]

        visual_tokens = self.vision_encoder(images)
        z_q = self.joint_encoder(qpos)

        outputs: Dict[str, Any] = {
            "visual_tokens": visual_tokens,
            "z_q": z_q,
        }
        if is_training:
            if motion_latent_override is not None:
                raise ValueError("motion_latent_override is only supported when is_training=False")
            mu_motion, logvar_motion, z_motion = self.encode_motion_posterior(qpos, action_chunk)
            outputs.update(
                {
                    "mu_motion": mu_motion,
                    "logvar_motion": logvar_motion,
                }
            )
        elif motion_latent_override is not None:
            self._validate_motion_latent_override(motion_latent_override, batch_size)
            z_motion = motion_latent_override
        else:
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
        outputs.update(
            {
                "z_motion": z_motion,
                "decoder_hidden": decoder_hidden,
                "pred_action": self.action_head(decoder_hidden),
            }
        )
        return outputs

    def encode_motion_posterior(
        self,
        qpos: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode qpos and future actions into the motion posterior."""

        return self.motion_posterior(qpos, action_chunk)

    def _validate_inputs(self, images: torch.Tensor, qpos: torch.Tensor) -> None:
        if images.ndim != 5:
            raise ValueError("images must have shape [B, N_cam, 3, H, W]")
        if images.shape[2] != 3:
            raise ValueError(f"images channel dimension must be 3, got {images.shape[2]}")
        if qpos.ndim != 2 or qpos.shape[1] != self.q_dim:
            raise ValueError(f"qpos must have shape [B, {self.q_dim}]")
        if images.shape[0] != qpos.shape[0]:
            raise ValueError("images and qpos must have the same batch size")

    def _validate_forward_state(
        self,
        action_chunk: Optional[torch.Tensor],
        is_training: bool,
    ) -> None:
        if is_training:
            if action_chunk is None:
                raise ValueError("action_chunk is required when is_training=True")
            if action_chunk.ndim != 3 or action_chunk.shape[1:] != (
                self.chunk_len,
                self.action_dim,
            ):
                raise ValueError(
                    "action_chunk must have shape "
                    f"[B, {self.chunk_len}, {self.action_dim}] when is_training=True"
                )
            return
        if action_chunk is not None:
            raise ValueError("action_chunk must be None when is_training=False")

    def _validate_motion_latent_override(
        self,
        motion_latent_override: torch.Tensor,
        batch_size: int,
    ) -> None:
        if motion_latent_override.ndim != 2:
            raise ValueError("motion_latent_override must have shape [B, z_dim]")
        if motion_latent_override.shape != (batch_size, self.z_dim):
            raise ValueError(
                "motion_latent_override must have shape "
                f"[{batch_size}, {self.z_dim}], got {tuple(motion_latent_override.shape)}"
            )

    def _reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.future_queries, std=0.02)


class LegacyZeroLatentACTPolicyBaseline(nn.Module):
    """Strict loader for legacy pre-CVAE ACT baseline checkpoints."""

    act_baseline_version = "zero_latent_legacy"

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

    def forward(self, images: torch.Tensor, qpos: torch.Tensor, **_unused: Any) -> Dict[str, Any]:
        if images.ndim != 5:
            raise ValueError("images must have shape [B, N_cam, 3, H, W]")
        if qpos.ndim != 2 or qpos.shape[1] != self.q_dim:
            raise ValueError(f"qpos must have shape [B, {self.q_dim}]")
        batch_size = qpos.shape[0]
        visual_tokens = self.vision_encoder(images)
        z_q = self.joint_encoder(qpos)
        z_motion = qpos.new_zeros(batch_size, self.z_dim)
        policy_tokens = torch.cat(
            [
                visual_tokens,
                z_q[:, None, :],
                self.motion_latent_proj(z_motion)[:, None, :],
            ],
            dim=1,
        )
        memory = self.policy_encoder(policy_tokens)
        decoder_hidden = self.policy_decoder(
            self.future_queries.expand(batch_size, -1, -1),
            memory,
        )
        return {
            "visual_tokens": visual_tokens,
            "z_q": z_q,
            "z_motion": z_motion,
            "decoder_hidden": decoder_hidden,
            "pred_action": self.action_head(decoder_hidden),
        }

    def _reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.future_queries, std=0.02)
