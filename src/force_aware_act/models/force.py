"""Force encoder modules for ForceAwareACT."""

from __future__ import annotations

import torch
from torch import nn


class TemporalForceEncoder(nn.Module):
    """Encode a past force window into one online force token.

    Input shape:
        force_window: [B, L, force_dim]

    Output shape:
        z_F_online: [B, d_model]
    """

    def __init__(
        self,
        force_dim: int = 6,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 2,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_window_len: int = 256,
    ) -> None:
        super().__init__()
        if force_dim <= 0:
            raise ValueError("force_dim must be positive")
        if d_model <= 0:
            raise ValueError("d_model must be positive")
        if max_window_len <= 0:
            raise ValueError("max_window_len must be positive")

        self.force_dim = force_dim
        self.d_model = d_model
        self.max_window_len = max_window_len

        self.force_proj = nn.Linear(force_dim, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, max_window_len + 1, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self._reset_parameters()

    def forward(self, force_window: torch.Tensor) -> torch.Tensor:
        if force_window.ndim != 3:
            raise ValueError("force_window must have shape [B, L, force_dim]")
        batch_size, window_len, force_dim = force_window.shape
        if force_dim != self.force_dim:
            raise ValueError(
                f"force_window last dimension must be {self.force_dim}, got {force_dim}"
            )
        if window_len > self.max_window_len:
            raise ValueError(
                f"force_window length {window_len} exceeds max_window_len {self.max_window_len}"
            )

        force_tokens = self.force_proj(force_window)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls_tokens, force_tokens], dim=1)
        tokens = tokens + self.pos_embed[:, : window_len + 1]
        encoded = self.encoder(tokens)
        return encoded[:, 0]

    def _reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
