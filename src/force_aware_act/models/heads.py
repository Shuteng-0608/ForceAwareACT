"""Prediction heads for ForceAwareACT."""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn


class ActionHead(nn.Module):
    """Predict future action chunks from decoder hidden states."""

    def __init__(
        self,
        d_model: int = 512,
        action_dim: int = 7,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError("d_model must be positive")
        if action_dim <= 0:
            raise ValueError("action_dim must be positive")

        self.d_model = d_model
        self.action_dim = action_dim
        self.head = _make_head(d_model, action_dim, hidden_dim, dropout)

    def forward(self, decoder_hidden: torch.Tensor) -> torch.Tensor:
        _validate_decoder_hidden(decoder_hidden, self.d_model)
        return self.head(decoder_hidden)


class ForceHead(nn.Module):
    """Predict future force chunks from decoder hidden states and contact latent."""

    def __init__(
        self,
        d_model: int = 512,
        z_dim: int = 32,
        force_dim: int = 6,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError("d_model must be positive")
        if z_dim <= 0:
            raise ValueError("z_dim must be positive")
        if force_dim <= 0:
            raise ValueError("force_dim must be positive")

        self.d_model = d_model
        self.z_dim = z_dim
        self.force_dim = force_dim
        self.head = _make_head(d_model + z_dim, force_dim, hidden_dim, dropout)

    def forward(self, decoder_hidden: torch.Tensor, z_contact: torch.Tensor) -> torch.Tensor:
        _validate_decoder_hidden(decoder_hidden, self.d_model)
        _validate_z_contact(z_contact, decoder_hidden.shape[0], self.z_dim)

        chunk_len = decoder_hidden.shape[1]
        z_contact_rep = z_contact[:, None, :].expand(-1, chunk_len, -1)
        force_input = torch.cat([decoder_hidden, z_contact_rep], dim=-1)
        return self.head(force_input)


def _make_head(
    input_dim: int,
    output_dim: int,
    hidden_dim: Optional[int],
    dropout: float,
) -> nn.Module:
    if hidden_dim is None:
        return nn.Linear(input_dim, output_dim)
    if hidden_dim <= 0:
        raise ValueError("hidden_dim must be positive when provided")
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, output_dim),
    )


def _validate_decoder_hidden(decoder_hidden: torch.Tensor, d_model: int) -> None:
    if decoder_hidden.ndim != 3:
        raise ValueError("decoder_hidden must have shape [B, K, d_model]")
    if decoder_hidden.shape[2] != d_model:
        raise ValueError(
            f"decoder_hidden last dimension must be {d_model}, got {decoder_hidden.shape[2]}"
        )


def _validate_z_contact(z_contact: torch.Tensor, batch_size: int, z_dim: int) -> None:
    if z_contact.ndim != 2:
        raise ValueError("z_contact must have shape [B, z_dim]")
    if z_contact.shape[0] != batch_size:
        raise ValueError(
            f"z_contact batch size must match decoder_hidden batch size {batch_size}, "
            f"got {z_contact.shape[0]}"
        )
    if z_contact.shape[1] != z_dim:
        raise ValueError(f"z_contact last dimension must be {z_dim}, got {z_contact.shape[1]}")
