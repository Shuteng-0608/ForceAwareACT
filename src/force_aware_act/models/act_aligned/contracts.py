"""Tensor layout and sequence-length contracts for ACT-aligned modules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from force_aware_act.models.act_aligned.config import ACTAlignedConfig


CONTACT_POSTERIOR_TOKEN_GROUPS = ("cls", "qpos", "contact_steps")
POLICY_SPECIAL_TOKEN_NAMES = ("z_contact", "qpos", "z_F_online", "z_VF")
MOTION_POSTERIOR_TOKEN_GROUPS = ("cls", "qpos", "motion_steps")
MOTION_POLICY_SPECIAL_TOKEN_NAMES = (
    "z_motion",
    "qpos",
    "z_F_online",
    "z_VF",
)
DUAL_ZERO_POLICY_SPECIAL_TOKEN_NAMES = (
    "qpos",
    "z_F_online",
    "z_VF",
)


@dataclass(frozen=True)
class ACTAlignedShapeContract:
    """Expected batch-first tensor shapes for one configuration."""

    config: ACTAlignedConfig

    def visual_tokens(self, batch_size: int) -> tuple[int, int, int]:
        self._validate_batch_size(batch_size)
        return batch_size, self.config.visual_token_count, self.config.d_model

    def qpos_token(self, batch_size: int) -> tuple[int, int, int]:
        self._validate_batch_size(batch_size)
        return batch_size, 1, self.config.d_model

    def action_tokens(self, batch_size: int) -> tuple[int, int, int]:
        self._validate_batch_size(batch_size)
        return batch_size, self.config.chunk_len, self.config.d_model

    def force_window_tokens(self, batch_size: int) -> tuple[int, int, int]:
        self._validate_batch_size(batch_size)
        return batch_size, self.config.force_window_len, self.config.d_model

    def contact_posterior_tokens(self, batch_size: int) -> tuple[int, int, int]:
        self._validate_batch_size(batch_size)
        return (
            batch_size,
            self.config.contact_posterior_token_count,
            self.config.d_model,
        )

    def motion_posterior_tokens(self, batch_size: int) -> tuple[int, int, int]:
        self._validate_batch_size(batch_size)
        return (
            batch_size,
            self.config.motion_posterior_token_count,
            self.config.d_model,
        )

    def force_encoder_tokens(self, batch_size: int) -> tuple[int, int, int]:
        self._validate_batch_size(batch_size)
        return batch_size, self.config.force_encoder_token_count, self.config.d_model

    def policy_memory(self, batch_size: int) -> tuple[int, int, int]:
        self._validate_batch_size(batch_size)
        return batch_size, self.config.policy_memory_token_count, self.config.d_model

    def decoder_hidden(self, batch_size: int) -> tuple[int, int, int]:
        self._validate_batch_size(batch_size)
        return batch_size, self.config.chunk_len, self.config.d_model

    def action_output(self, batch_size: int) -> tuple[int, int, int]:
        self._validate_batch_size(batch_size)
        return batch_size, self.config.chunk_len, self.config.action_dim

    def force_output(self, batch_size: int) -> tuple[int, int, int]:
        self._validate_batch_size(batch_size)
        return batch_size, self.config.chunk_len, self.config.force_dim

    @staticmethod
    def _validate_batch_size(batch_size: int) -> None:
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")


def require_token_tensor(
    tensor: torch.Tensor,
    *,
    name: str,
    expected_shape: Sequence[int],
) -> None:
    """Require a floating-point batch-first token tensor of an exact shape."""

    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.ndim != 3:
        raise ValueError(f"{name} must use batch-first [B, S, D] layout")
    if tuple(tensor.shape) != tuple(expected_shape):
        raise ValueError(
            f"{name} must have shape {tuple(expected_shape)}, got {tuple(tensor.shape)}"
        )
    if not tensor.is_floating_point():
        raise ValueError(f"{name} must be floating point")


def require_padding_mask(
    mask: torch.Tensor,
    *,
    name: str,
    batch_size: int,
    sequence_length: int,
) -> None:
    """Require a boolean padding mask where True denotes a padded token."""

    expected_shape = (batch_size, sequence_length)
    if not isinstance(mask, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tuple(mask.shape) != expected_shape:
        raise ValueError(
            f"{name} must have shape {expected_shape}, got {tuple(mask.shape)}"
        )
    if mask.dtype is not torch.bool:
        raise ValueError(f"{name} must have dtype torch.bool")
