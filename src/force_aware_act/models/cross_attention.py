"""Cross-attention modules for ForceAwareACT."""

from __future__ import annotations

import torch
from torch import nn


class ForceVisionCrossAttention(nn.Module):
    """Attend from an online force token into visual tokens.

    Inputs:
        z_F_online: [B, d_model]
        visual_tokens: [B, N_v, d_model]

    Outputs:
        z_VF: [B, d_model]
        attn_weights: [B, 1, N_v] when ``return_attn=True``
    """

    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 8,
        dropout: float = 0.1,
        return_attn: bool = False,
    ) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError("d_model must be positive")
        if nhead <= 0:
            raise ValueError("nhead must be positive")
        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead")

        self.d_model = d_model
        self.return_attn = return_attn
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )

    def forward(
        self,
        z_F_online: torch.Tensor,
        visual_tokens: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        self._validate_inputs(z_F_online, visual_tokens)

        query = z_F_online.unsqueeze(1)
        attended, attn_weights = self.cross_attn(
            query=query,
            key=visual_tokens,
            value=visual_tokens,
            need_weights=self.return_attn,
        )
        z_vf = attended[:, 0, :]

        if self.return_attn:
            return z_vf, attn_weights
        return z_vf

    def _validate_inputs(self, z_F_online: torch.Tensor, visual_tokens: torch.Tensor) -> None:
        if z_F_online.ndim != 2:
            raise ValueError("z_F_online must have shape [B, d_model]")
        if visual_tokens.ndim != 3:
            raise ValueError("visual_tokens must have shape [B, N_v, d_model]")
        if z_F_online.shape[0] != visual_tokens.shape[0]:
            raise ValueError(
                "z_F_online and visual_tokens must have the same batch size, "
                f"got {z_F_online.shape[0]} and {visual_tokens.shape[0]}"
            )
        if z_F_online.shape[1] != self.d_model:
            raise ValueError(
                f"z_F_online feature dimension must be {self.d_model}, got {z_F_online.shape[1]}"
            )
        if visual_tokens.shape[2] != self.d_model:
            raise ValueError(
                "visual_tokens feature dimension must be "
                f"{self.d_model}, got {visual_tokens.shape[2]}"
            )
