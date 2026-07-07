"""Force-aware ACT policy with only the contact CVAE latent."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from torch import nn

from force_aware_act.models.contact_prior import ContactPriorEncoder
from force_aware_act.models.cross_attention import ForceVisionCrossAttention
from force_aware_act.models.force import TemporalForceEncoder
from force_aware_act.models.heads import ActionHead, ForceHead
from force_aware_act.models.policy import JointMLP
from force_aware_act.models.posterior import ContactPosteriorEncoder
from force_aware_act.models.vision import ResNet18VisionEncoder


class ForceAwareACTContactCVAEPolicy(nn.Module):
    """Force-aware ACT policy with a structurally contact-only latent.

    Token sequence passed to the policy Transformer encoder:
        visual spatial tokens, z_VF, z_q, z_F_online, z_contact.

    Training samples z_contact from q(z_contact | qpos, future actions,
    future forces). Deployment supports only online zero or conditional-prior
    contact latents. Offline posterior-oracle evaluation should encode the
    posterior separately and pass it through contact_latent_override.
    """

    policy_token_names = (
        "visual_tokens",
        "z_VF",
        "z_q",
        "z_F_online",
        "z_contact",
    )

    def __init__(
        self,
        d_model: int = 512,
        z_dim: int = 32,
        q_dim: int = 7,
        action_dim: int = 7,
        force_dim: int = 6,
        chunk_len: int = 50,
        nhead: int = 8,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 2,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        pretrained_resnet18: bool = True,
        freeze_resnet18: bool = False,
        max_force_window_len: int = 256,
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
        if force_dim <= 0:
            raise ValueError("force_dim must be positive")
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
        self.force_dim = force_dim
        self.chunk_len = chunk_len

        self.vision_encoder = ResNet18VisionEncoder(
            d_model=d_model,
            pretrained=pretrained_resnet18,
            freeze_backbone=freeze_resnet18,
        )
        self.joint_encoder = JointMLP(q_dim=q_dim, d_model=d_model)
        self.force_encoder = TemporalForceEncoder(
            force_dim=force_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_window_len=max_force_window_len,
        )
        self.force_vision_cross_attention = ForceVisionCrossAttention(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
        )
        self.contact_posterior = ContactPosteriorEncoder(
            q_dim=q_dim,
            action_dim=action_dim,
            force_dim=force_dim,
            d_model=d_model,
            z_dim=z_dim,
            nhead=nhead,
            num_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_chunk_len=chunk_len,
        )
        self.contact_prior = ContactPriorEncoder(
            d_model=d_model,
            z_dim=z_dim,
            hidden_dim=d_model,
            dropout=dropout,
            use_visual_summary=True,
        )
        self.contact_latent_proj = nn.Linear(z_dim, d_model)

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
        self.force_head = ForceHead(d_model=d_model, z_dim=z_dim, force_dim=force_dim)
        self._reset_parameters()

    def forward(
        self,
        images: torch.Tensor,
        qpos: torch.Tensor,
        force_window: torch.Tensor,
        action_chunk: Optional[torch.Tensor] = None,
        future_force_chunk: Optional[torch.Tensor] = None,
        is_training: bool = True,
        contact_latent_mode: Optional[str] = None,
        deterministic_prior: bool = True,
        contact_latent_override: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        if contact_latent_mode is None:
            contact_latent_mode = "posterior" if is_training else "zero"
        self._validate_online_inputs(images, qpos, force_window)
        self._validate_contact_latent_mode(contact_latent_mode, is_training)
        self._validate_forward_state(
            action_chunk=action_chunk,
            future_force_chunk=future_force_chunk,
            is_training=is_training,
            contact_latent_override=contact_latent_override,
        )

        visual_tokens = self.vision_encoder(images)
        z_q = self.joint_encoder(qpos)
        z_f_online = self.force_encoder(force_window)
        z_vf = self.force_vision_cross_attention(z_f_online, visual_tokens)

        batch_size = qpos.shape[0]
        outputs: Dict[str, Any] = {
            "visual_tokens": visual_tokens,
            "z_q": z_q,
            "z_F_online": z_f_online,
            "z_VF": z_vf,
        }

        if is_training:
            mu_contact, logvar_contact, z_contact = self.encode_contact_posterior(
                qpos,
                action_chunk,
                future_force_chunk,
            )
            visual_summary = visual_tokens.mean(dim=1)
            mu_contact_prior, logvar_contact_prior, z_contact_prior = self.contact_prior(
                z_q=z_q,
                z_F_online=z_f_online,
                z_VF=z_vf,
                visual_summary=visual_summary,
            )
            outputs.update(
                {
                    "mu_contact": mu_contact,
                    "logvar_contact": logvar_contact,
                    "mu_contact_prior": mu_contact_prior,
                    "logvar_contact_prior": logvar_contact_prior,
                    "z_contact_prior": z_contact_prior,
                }
            )
        elif contact_latent_override is not None:
            self._validate_contact_latent_override(contact_latent_override, batch_size)
            z_contact = contact_latent_override
        elif contact_latent_mode == "zero":
            z_contact = qpos.new_zeros(batch_size, self.z_dim)
        else:
            visual_summary = visual_tokens.mean(dim=1)
            mu_contact_prior, logvar_contact_prior, z_contact_prior_sample = self.contact_prior(
                z_q=z_q,
                z_F_online=z_f_online,
                z_VF=z_vf,
                visual_summary=visual_summary,
            )
            z_contact = mu_contact_prior if deterministic_prior else z_contact_prior_sample
            outputs.update(
                {
                    "mu_contact_prior": mu_contact_prior,
                    "logvar_contact_prior": logvar_contact_prior,
                    "z_contact_prior": z_contact,
                    "z_contact_prior_sample": z_contact_prior_sample,
                }
            )

        policy_tokens = self._assemble_policy_tokens(
            visual_tokens=visual_tokens,
            z_vf=z_vf,
            z_q=z_q,
            z_f_online=z_f_online,
            z_contact=z_contact,
        )
        memory = self.policy_encoder(policy_tokens)
        query_tokens = self.future_queries.expand(batch_size, -1, -1)
        decoder_hidden = self.policy_decoder(query_tokens, memory)

        outputs.update(
            {
                "z_contact": z_contact,
                "decoder_hidden": decoder_hidden,
                "pred_action": self.action_head(decoder_hidden),
                "pred_force": self.force_head(decoder_hidden, z_contact),
            }
        )
        return outputs

    def encode_contact_posterior(
        self,
        qpos: torch.Tensor,
        action_chunk: torch.Tensor,
        future_force_chunk: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode qpos, future actions, and future forces into q(z_contact)."""

        return self.contact_posterior(qpos, action_chunk, future_force_chunk)

    def _assemble_policy_tokens(
        self,
        visual_tokens: torch.Tensor,
        z_vf: torch.Tensor,
        z_q: torch.Tensor,
        z_f_online: torch.Tensor,
        z_contact: torch.Tensor,
    ) -> torch.Tensor:
        return torch.cat(
            [
                visual_tokens,
                z_vf[:, None, :],
                z_q[:, None, :],
                z_f_online[:, None, :],
                self.contact_latent_proj(z_contact)[:, None, :],
            ],
            dim=1,
        )

    def _validate_online_inputs(
        self,
        images: torch.Tensor,
        qpos: torch.Tensor,
        force_window: torch.Tensor,
    ) -> None:
        if images.ndim != 5:
            raise ValueError("images must have shape [B, N_cam, 3, H, W]")
        if images.shape[2] != 3:
            raise ValueError(f"images channel dimension must be 3, got {images.shape[2]}")
        if qpos.ndim != 2 or qpos.shape[1] != self.q_dim:
            raise ValueError(f"qpos must have shape [B, {self.q_dim}]")
        if force_window.ndim != 3 or force_window.shape[2] != self.force_dim:
            raise ValueError(f"force_window must have shape [B, L, {self.force_dim}]")
        batch_size = images.shape[0]
        if qpos.shape[0] != batch_size or force_window.shape[0] != batch_size:
            raise ValueError("images, qpos, and force_window must have the same batch size")

    def _validate_contact_latent_mode(self, contact_latent_mode: str, is_training: bool) -> None:
        valid_modes = {"zero", "prior", "posterior"}
        if contact_latent_mode not in valid_modes:
            raise ValueError(
                "contact_latent_mode must be one of: 'zero', 'prior', 'posterior'"
            )
        if is_training and contact_latent_mode != "posterior":
            raise ValueError(
                "ForceAwareACTContactCVAEPolicy training requires contact_latent_mode='posterior'"
            )
        if not is_training and contact_latent_mode == "posterior":
            raise ValueError(
                "contact_latent_mode='posterior' is oracle-only; use encode_contact_posterior "
                "and contact_latent_override for offline evaluation"
            )

    def _validate_forward_state(
        self,
        action_chunk: Optional[torch.Tensor],
        future_force_chunk: Optional[torch.Tensor],
        is_training: bool,
        contact_latent_override: Optional[torch.Tensor],
    ) -> None:
        if is_training:
            if contact_latent_override is not None:
                raise ValueError("contact_latent_override is only supported when is_training=False")
            if action_chunk is None:
                raise ValueError("action_chunk is required when is_training=True")
            if future_force_chunk is None:
                raise ValueError("future_force_chunk is required when is_training=True")
            if action_chunk.ndim != 3 or action_chunk.shape[1:] != (
                self.chunk_len,
                self.action_dim,
            ):
                raise ValueError(
                    "action_chunk must have shape "
                    f"[B, {self.chunk_len}, {self.action_dim}] when is_training=True"
                )
            if future_force_chunk.ndim != 3 or future_force_chunk.shape[1:] != (
                self.chunk_len,
                self.force_dim,
            ):
                raise ValueError(
                    "future_force_chunk must have shape "
                    f"[B, {self.chunk_len}, {self.force_dim}] when is_training=True"
                )
            if action_chunk.shape[0] != future_force_chunk.shape[0]:
                raise ValueError("action_chunk and future_force_chunk must have same batch size")
            return

        if contact_latent_override is not None:
            if action_chunk is not None or future_force_chunk is not None:
                raise ValueError(
                    "future labels must be encoded outside inference before using "
                    "contact_latent_override"
                )
            return
        if action_chunk is not None:
            raise ValueError("action_chunk must be None when is_training=False")
        if future_force_chunk is not None:
            raise ValueError("future_force_chunk must be None when is_training=False")

    def _validate_contact_latent_override(
        self,
        contact_latent_override: torch.Tensor,
        batch_size: int,
    ) -> None:
        if contact_latent_override.ndim != 2:
            raise ValueError("contact_latent_override must have shape [B, z_dim]")
        if contact_latent_override.shape != (batch_size, self.z_dim):
            raise ValueError(
                "contact_latent_override must have shape "
                f"[{batch_size}, {self.z_dim}], got {tuple(contact_latent_override.shape)}"
            )

    def _reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.future_queries, std=0.02)
