"""Conditional contact prior encoder for ForceAwareACT."""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from force_aware_act.models.posterior import reparameterize


class ContactPriorEncoder(nn.Module):
    """Predict a contact latent prior from online state, force, and vision features."""

    def __init__(
        self,
        d_model: int = 512,
        z_dim: int = 32,
        hidden_dim: int = 512,
        dropout: float = 0.1,
        use_visual_summary: bool = True,
    ) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError("d_model must be positive")
        if z_dim <= 0:
            raise ValueError("z_dim must be positive")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if dropout < 0:
            raise ValueError("dropout must be non-negative")

        self.d_model = d_model
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.use_visual_summary = use_visual_summary

        self.trunk = _make_mlp(3 * d_model, hidden_dim, dropout)
        self.visual_trunk = (
            _make_mlp(4 * d_model, hidden_dim, dropout)
            if use_visual_summary
            else None
        )
        self.mu_head = nn.Linear(hidden_dim, z_dim)
        self.logvar_head = nn.Linear(hidden_dim, z_dim)

    def forward(
        self,
        z_q: torch.Tensor,
        z_F_online: torch.Tensor,
        z_VF: torch.Tensor,
        visual_summary: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _validate_feature(z_q, self.d_model, "z_q")
        _validate_feature(z_F_online, self.d_model, "z_F_online")
        _validate_feature(z_VF, self.d_model, "z_VF")
        _validate_batch_size(z_F_online, z_q.shape[0], "z_F_online")
        _validate_batch_size(z_VF, z_q.shape[0], "z_VF")

        if visual_summary is None:
            features = torch.cat([z_q, z_F_online, z_VF], dim=-1)
            hidden = self.trunk(features)
        else:
            if self.visual_trunk is None:
                raise ValueError("visual_summary was provided but use_visual_summary=False")
            _validate_feature(visual_summary, self.d_model, "visual_summary")
            _validate_batch_size(visual_summary, z_q.shape[0], "visual_summary")
            features = torch.cat([z_q, z_F_online, z_VF, visual_summary], dim=-1)
            hidden = self.visual_trunk(features)

        mu = self.mu_head(hidden)
        logvar = self.logvar_head(hidden)
        z = reparameterize(mu, logvar)
        return mu, logvar, z


def _make_mlp(input_dim: int, hidden_dim: int, dropout: float) -> nn.Module:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Dropout(dropout),
    )


def _validate_feature(tensor: torch.Tensor, d_model: int, name: str) -> None:
    if tensor.ndim != 2:
        raise ValueError(f"{name} must have shape [B, d_model]")
    if tensor.shape[1] != d_model:
        raise ValueError(f"{name} last dimension must be {d_model}, got {tensor.shape[1]}")


def _validate_batch_size(tensor: torch.Tensor, batch_size: int, name: str) -> None:
    if tensor.shape[0] != batch_size:
        raise ValueError(f"{name} batch size must be {batch_size}, got {tensor.shape[0]}")
