"""Shared causal force-history contract for training and deployment."""

from __future__ import annotations

from typing import Sequence

import torch


CAUSAL_STATE_RATE_FORCE_HISTORY_V1 = (
    "causal_state_rate_last_l_left_padded_normalized_v1"
)


def prepare_causal_state_rate_force_history(
    values: torch.Tensor,
    *,
    window_len: int,
    force_dim: int,
    mean: Sequence[float] | torch.Tensor,
    std: Sequence[float] | torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize the last ``L`` causal samples and left-pad in normalized space.

    ``values`` must be ordered from oldest to newest and include the wrench at
    the current state timestep.  The returned boolean mask follows PyTorch's
    padding convention: ``True`` means that the position is padding.
    """

    if not isinstance(values, torch.Tensor):
        raise TypeError("values must be a torch.Tensor")
    if values.ndim != 2 or values.shape[1] != force_dim:
        raise ValueError(f"values must have shape [N, {force_dim}]")
    if not values.is_floating_point():
        raise ValueError("values must be floating point")
    if values.shape[0] == 0:
        raise ValueError("values must contain the current wrench")
    if window_len <= 0 or force_dim <= 0:
        raise ValueError("window_len and force_dim must be positive")

    mean_tensor = torch.as_tensor(mean, dtype=values.dtype, device=values.device)
    std_tensor = torch.as_tensor(std, dtype=values.dtype, device=values.device)
    if mean_tensor.shape != (force_dim,) or std_tensor.shape != (force_dim,):
        raise ValueError(f"mean and std must have shape [{force_dim}]")
    if not torch.isfinite(mean_tensor).all() or not torch.isfinite(std_tensor).all():
        raise ValueError("mean and std must be finite")
    if torch.any(std_tensor <= 0):
        raise ValueError("std must be positive")

    causal_values = values[-window_len:]
    valid_length = int(causal_values.shape[0])
    history = values.new_zeros(window_len, force_dim)
    padding_mask = torch.ones(window_len, dtype=torch.bool, device=values.device)
    history[-valid_length:] = (causal_values - mean_tensor) / std_tensor
    padding_mask[-valid_length:] = False
    return history, padding_mask
