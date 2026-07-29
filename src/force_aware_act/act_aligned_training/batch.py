"""Strict batch contract for ACT-aligned contact-CVAE training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

import torch

from force_aware_act.models.act_aligned.config import ACTAlignedConfig


@dataclass(frozen=True)
class ACTAlignedBatch:
    """One normalized training batch with explicit causal and future masks."""

    images: torch.Tensor
    qpos: torch.Tensor
    force_history: torch.Tensor
    force_padding_mask: torch.Tensor
    action_chunk: torch.Tensor
    future_force_chunk: torch.Tensor
    future_padding_mask: torch.Tensor

    @property
    def batch_size(self) -> int:
        return self.images.shape[0]

    def validate(self, model_config: ACTAlignedConfig) -> None:
        """Require the exact tensor contract consumed by the new policy."""

        expected_float_shapes = {
            "images": (
                None,
                model_config.num_cameras,
                3,
                model_config.image_height,
                model_config.image_width,
            ),
            "qpos": (None, model_config.q_dim),
            "force_history": (
                None,
                model_config.force_window_len,
                model_config.force_dim,
            ),
            "action_chunk": (
                None,
                model_config.chunk_len,
                model_config.action_dim,
            ),
            "future_force_chunk": (
                None,
                model_config.chunk_len,
                model_config.force_dim,
            ),
        }
        batch_size = None
        reference = None
        for name, expected_shape in expected_float_shapes.items():
            tensor = getattr(self, name)
            _require_shape(tensor, name, expected_shape)
            if not tensor.is_floating_point():
                raise ValueError(f"{name} must be floating point")
            if batch_size is None:
                batch_size = tensor.shape[0]
                reference = tensor
                if batch_size <= 0:
                    raise ValueError("batch size must be positive")
            elif tensor.shape[0] != batch_size:
                raise ValueError("all batch tensors must have the same batch size")
            if tensor.device != reference.device:
                raise ValueError(f"{name} must be on the same device as images")
            if tensor.dtype != reference.dtype:
                raise ValueError(f"{name} must have the same dtype as images")

        mask_shapes = {
            "force_padding_mask": (
                batch_size,
                model_config.force_window_len,
            ),
            "future_padding_mask": (
                batch_size,
                model_config.chunk_len,
            ),
        }
        for name, expected_shape in mask_shapes.items():
            mask = getattr(self, name)
            _require_shape(mask, name, expected_shape)
            if mask.dtype is not torch.bool:
                raise ValueError(f"{name} must have dtype torch.bool")
            if mask.device != reference.device:
                raise ValueError(f"{name} must be on the same device as images")
        if (~self.future_padding_mask).sum().item() == 0:
            raise ValueError("future_padding_mask must contain at least one valid step")

    def to(
        self,
        device: Union[str, torch.device],
        *,
        non_blocking: bool = False,
    ) -> "ACTAlignedBatch":
        """Return the complete batch on one device without changing dtypes."""

        return ACTAlignedBatch(
            images=self.images.to(device, non_blocking=non_blocking),
            qpos=self.qpos.to(device, non_blocking=non_blocking),
            force_history=self.force_history.to(
                device,
                non_blocking=non_blocking,
            ),
            force_padding_mask=self.force_padding_mask.to(
                device,
                non_blocking=non_blocking,
            ),
            action_chunk=self.action_chunk.to(
                device,
                non_blocking=non_blocking,
            ),
            future_force_chunk=self.future_force_chunk.to(
                device,
                non_blocking=non_blocking,
            ),
            future_padding_mask=self.future_padding_mask.to(
                device,
                non_blocking=non_blocking,
            ),
        )


def _require_shape(
    tensor: torch.Tensor,
    name: str,
    expected_shape: tuple,
) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.ndim != len(expected_shape):
        raise ValueError(
            f"{name} must have shape {expected_shape}, got {tuple(tensor.shape)}"
        )
    for actual, expected in zip(tensor.shape, expected_shape):
        if expected is not None and actual != expected:
            raise ValueError(
                f"{name} must have shape {expected_shape}, "
                f"got {tuple(tensor.shape)}"
            )
