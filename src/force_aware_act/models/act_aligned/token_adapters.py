"""Linear modality adapters with a shared batch-first token contract."""

from __future__ import annotations

import torch
from torch import nn


class SequenceTokenAdapter(nn.Module):
    """Project a sequence from ``input_dim`` to ``d_model``.

    Input: ``[B, S, input_dim]``
    Output: ``[B, S, d_model]``
    """

    def __init__(self, input_dim: int, d_model: int, *, modality_name: str) -> None:
        super().__init__()
        _validate_positive_int(input_dim, "input_dim")
        _validate_positive_int(d_model, "d_model")
        if not modality_name:
            raise ValueError("modality_name must not be empty")
        self.input_dim = input_dim
        self.d_model = d_model
        self.modality_name = modality_name
        self.projection = nn.Linear(input_dim, d_model)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        _validate_floating_tensor(values, self.modality_name)
        if values.ndim != 3 or values.shape[-1] != self.input_dim:
            raise ValueError(
                f"{self.modality_name} must have shape [B, S, {self.input_dim}]"
            )
        return self.projection(values)


class SingleTokenAdapter(nn.Module):
    """Project one feature vector to a one-token sequence.

    Input: ``[B, input_dim]``
    Output: ``[B, 1, d_model]``
    """

    def __init__(self, input_dim: int, d_model: int, *, modality_name: str) -> None:
        super().__init__()
        _validate_positive_int(input_dim, "input_dim")
        _validate_positive_int(d_model, "d_model")
        if not modality_name:
            raise ValueError("modality_name must not be empty")
        self.input_dim = input_dim
        self.d_model = d_model
        self.modality_name = modality_name
        self.projection = nn.Linear(input_dim, d_model)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        _validate_floating_tensor(values, self.modality_name)
        if values.ndim != 2 or values.shape[-1] != self.input_dim:
            raise ValueError(
                f"{self.modality_name} must have shape [B, {self.input_dim}]"
            )
        return self.projection(values).unsqueeze(1)


class QposTokenAdapter(SingleTokenAdapter):
    def __init__(self, q_dim: int, d_model: int) -> None:
        super().__init__(q_dim, d_model, modality_name="qpos")


class ActionTokenAdapter(SequenceTokenAdapter):
    def __init__(self, action_dim: int, d_model: int) -> None:
        super().__init__(action_dim, d_model, modality_name="action")


class ForceTokenAdapter(SequenceTokenAdapter):
    def __init__(self, force_dim: int, d_model: int) -> None:
        super().__init__(force_dim, d_model, modality_name="force")


class LatentTokenAdapter(SingleTokenAdapter):
    def __init__(self, latent_dim: int, d_model: int) -> None:
        super().__init__(latent_dim, d_model, modality_name="latent")


def _validate_positive_int(value: int, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _validate_floating_tensor(tensor: torch.Tensor, name: str) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if not tensor.is_floating_point():
        raise ValueError(f"{name} must be floating point")
