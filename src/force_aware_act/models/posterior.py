"""Posterior encoder modules for ForceAwareACT latent variables."""

from __future__ import annotations

import torch
from torch import nn


def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Sample a Gaussian latent with the reparameterization trick."""

    if mu.shape != logvar.shape:
        raise ValueError(f"mu and logvar must have the same shape, got {mu.shape} and {logvar.shape}")
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


def kl_normal(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """KL divergence from N(mu, sigma) to standard normal, averaged over batch."""

    if mu.shape != logvar.shape:
        raise ValueError(f"mu and logvar must have the same shape, got {mu.shape} and {logvar.shape}")
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1).mean()


class MotionPosteriorEncoder(nn.Module):
    """Encode qpos and future action chunks into a motion posterior latent."""

    def __init__(
        self,
        q_dim: int = 7,
        action_dim: int = 7,
        d_model: int = 512,
        z_dim: int = 32,
        nhead: int = 8,
        num_layers: int = 2,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_chunk_len: int = 256,
    ) -> None:
        super().__init__()
        _validate_common_config(q_dim, action_dim, d_model, z_dim, nhead, max_chunk_len)

        self.q_dim = q_dim
        self.action_dim = action_dim
        self.d_model = d_model
        self.z_dim = z_dim
        self.max_chunk_len = max_chunk_len

        self.q_proj = nn.Linear(q_dim, d_model)
        self.action_proj = nn.Linear(action_dim, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, max_chunk_len + 2, d_model))
        self.encoder = _make_transformer_encoder(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.mu_head = nn.Linear(d_model, z_dim)
        self.logvar_head = nn.Linear(d_model, z_dim)
        self._reset_parameters()

    def forward(
        self,
        qpos: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _validate_qpos(qpos, self.q_dim)
        _validate_action_chunk(action_chunk, qpos.shape[0], self.action_dim, self.max_chunk_len)

        batch_size, chunk_len, _ = action_chunk.shape
        cls = self.cls_token.expand(batch_size, -1, -1)
        q_token = self.q_proj(qpos).unsqueeze(1)
        action_tokens = self.action_proj(action_chunk)
        tokens = torch.cat([cls, q_token, action_tokens], dim=1)
        tokens = tokens + self.pos_embed[:, : chunk_len + 2]

        cls_output = self.encoder(tokens)[:, 0]
        mu = self.mu_head(cls_output)
        logvar = self.logvar_head(cls_output)
        z = reparameterize(mu, logvar)
        return mu, logvar, z

    def _reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)


class ContactPosteriorEncoder(nn.Module):
    """Encode qpos, future action chunks, and future force chunks into a contact latent."""

    def __init__(
        self,
        q_dim: int = 7,
        action_dim: int = 7,
        force_dim: int = 6,
        d_model: int = 512,
        z_dim: int = 32,
        nhead: int = 8,
        num_layers: int = 2,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_chunk_len: int = 256,
    ) -> None:
        super().__init__()
        _validate_common_config(q_dim, action_dim, d_model, z_dim, nhead, max_chunk_len)
        if force_dim <= 0:
            raise ValueError("force_dim must be positive")

        self.q_dim = q_dim
        self.action_dim = action_dim
        self.force_dim = force_dim
        self.d_model = d_model
        self.z_dim = z_dim
        self.max_chunk_len = max_chunk_len

        self.q_proj = nn.Linear(q_dim, d_model)
        self.action_proj = nn.Linear(action_dim, d_model)
        self.force_proj = nn.Linear(force_dim, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, 2 * max_chunk_len + 2, d_model))
        self.encoder = _make_transformer_encoder(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.mu_head = nn.Linear(d_model, z_dim)
        self.logvar_head = nn.Linear(d_model, z_dim)
        self._reset_parameters()

    def forward(
        self,
        qpos: torch.Tensor,
        action_chunk: torch.Tensor,
        future_force_chunk: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _validate_qpos(qpos, self.q_dim)
        _validate_action_chunk(action_chunk, qpos.shape[0], self.action_dim, self.max_chunk_len)
        _validate_future_force_chunk(
            future_force_chunk=future_force_chunk,
            batch_size=qpos.shape[0],
            chunk_len=action_chunk.shape[1],
            force_dim=self.force_dim,
        )

        batch_size, chunk_len, _ = action_chunk.shape
        cls = self.cls_token.expand(batch_size, -1, -1)
        q_token = self.q_proj(qpos).unsqueeze(1)
        action_tokens = self.action_proj(action_chunk)
        force_tokens = self.force_proj(future_force_chunk)
        tokens = torch.cat([cls, q_token, action_tokens, force_tokens], dim=1)
        tokens = tokens + self.pos_embed[:, : 2 * chunk_len + 2]

        cls_output = self.encoder(tokens)[:, 0]
        mu = self.mu_head(cls_output)
        logvar = self.logvar_head(cls_output)
        z = reparameterize(mu, logvar)
        return mu, logvar, z

    def _reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)


def _make_transformer_encoder(
    d_model: int,
    nhead: int,
    num_layers: int,
    dim_feedforward: int,
    dropout: float,
) -> nn.TransformerEncoder:
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        batch_first=True,
    )
    return nn.TransformerEncoder(encoder_layer, num_layers=num_layers)


def _validate_common_config(
    q_dim: int,
    action_dim: int,
    d_model: int,
    z_dim: int,
    nhead: int,
    max_chunk_len: int,
) -> None:
    if q_dim <= 0:
        raise ValueError("q_dim must be positive")
    if action_dim <= 0:
        raise ValueError("action_dim must be positive")
    if d_model <= 0:
        raise ValueError("d_model must be positive")
    if z_dim <= 0:
        raise ValueError("z_dim must be positive")
    if nhead <= 0:
        raise ValueError("nhead must be positive")
    if d_model % nhead != 0:
        raise ValueError("d_model must be divisible by nhead")
    if max_chunk_len <= 0:
        raise ValueError("max_chunk_len must be positive")


def _validate_qpos(qpos: torch.Tensor, q_dim: int) -> None:
    if qpos.ndim != 2:
        raise ValueError("qpos must have shape [B, q_dim]")
    if qpos.shape[1] != q_dim:
        raise ValueError(f"qpos last dimension must be {q_dim}, got {qpos.shape[1]}")


def _validate_action_chunk(
    action_chunk: torch.Tensor,
    batch_size: int,
    action_dim: int,
    max_chunk_len: int,
) -> None:
    if action_chunk.ndim != 3:
        raise ValueError("action_chunk must have shape [B, K, action_dim]")
    if action_chunk.shape[0] != batch_size:
        raise ValueError(
            f"action_chunk batch size must be {batch_size}, got {action_chunk.shape[0]}"
        )
    if action_chunk.shape[2] != action_dim:
        raise ValueError(
            f"action_chunk last dimension must be {action_dim}, got {action_chunk.shape[2]}"
        )
    if action_chunk.shape[1] > max_chunk_len:
        raise ValueError(
            f"action_chunk length {action_chunk.shape[1]} exceeds max_chunk_len {max_chunk_len}"
        )


def _validate_future_force_chunk(
    future_force_chunk: torch.Tensor,
    batch_size: int,
    chunk_len: int,
    force_dim: int,
) -> None:
    if future_force_chunk.ndim != 3:
        raise ValueError("future_force_chunk must have shape [B, K, force_dim]")
    expected_shape = (batch_size, chunk_len, force_dim)
    if tuple(future_force_chunk.shape) != expected_shape:
        raise ValueError(
            "future_force_chunk must have shape "
            f"[B, K, force_dim] = {expected_shape}, got {tuple(future_force_chunk.shape)}"
        )
