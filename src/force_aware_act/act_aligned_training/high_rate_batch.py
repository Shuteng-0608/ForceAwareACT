"""Independent v2 batch contract for native 500 Hz force training."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Union

import torch

from force_aware_act.models.act_aligned.config import ACTAlignedHighRateConfig


@dataclass(frozen=True)
class ACTAlignedHighRateBatch:
    images: torch.Tensor
    qpos: torch.Tensor
    action_chunk: torch.Tensor
    action_padding_mask: torch.Tensor
    online_force_intervals: torch.Tensor
    online_force_relative_time: torch.Tensor
    online_force_sample_padding_mask: torch.Tensor
    online_force_interval_padding_mask: torch.Tensor
    future_force_intervals: torch.Tensor
    future_force_relative_time: torch.Tensor
    future_force_sample_padding_mask: torch.Tensor
    future_force_interval_padding_mask: torch.Tensor
    future_force_target: torch.Tensor

    @property
    def batch_size(self) -> int:
        return int(self.images.shape[0])

    def validate(self, config: ACTAlignedHighRateConfig) -> None:
        b = self.batch_size
        k = config.chunk_len
        h = config.max_online_force_intervals
        s = config.max_force_samples_per_interval
        expected = {
            "images": (b, config.num_cameras, 3, config.image_height, config.image_width),
            "qpos": (b, config.q_dim),
            "action_chunk": (b, k, config.action_dim),
            "action_padding_mask": (b, k),
            "online_force_intervals": (b, h, s, config.force_dim),
            "online_force_relative_time": (b, h, s),
            "online_force_sample_padding_mask": (b, h, s),
            "online_force_interval_padding_mask": (b, h),
            "future_force_intervals": (b, k, s, config.force_dim),
            "future_force_relative_time": (b, k, s),
            "future_force_sample_padding_mask": (b, k, s),
            "future_force_interval_padding_mask": (b, k),
            "future_force_target": (b, k, config.force_dim),
        }
        if b <= 0:
            raise ValueError("batch size must be positive")
        reference = self.images
        float_names = {
            "images",
            "qpos",
            "action_chunk",
            "online_force_intervals",
            "online_force_relative_time",
            "future_force_intervals",
            "future_force_relative_time",
            "future_force_target",
        }
        for item in fields(self):
            name = item.name
            tensor = getattr(self, name)
            if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != expected[name]:
                raise ValueError(f"{name} must have shape {expected[name]}")
            if tensor.device != reference.device:
                raise ValueError(f"{name} must share images device")
            if name in float_names:
                if not tensor.is_floating_point() or tensor.dtype != reference.dtype:
                    raise ValueError(f"{name} must share images floating dtype")
            elif tensor.dtype is not torch.bool:
                raise ValueError(f"{name} must have dtype torch.bool")
        if not torch.equal(
            self.online_force_interval_padding_mask,
            self.online_force_sample_padding_mask.all(dim=-1),
        ):
            raise ValueError("online force interval and sample masks disagree")
        if not torch.equal(
            self.future_force_interval_padding_mask,
            self.future_force_sample_padding_mask.all(dim=-1),
        ):
            raise ValueError("future force interval and sample masks disagree")
        if self.online_force_interval_padding_mask.all(dim=1).any():
            raise ValueError("every sample requires at least one online force interval")
        if self.action_padding_mask.all(dim=1).any():
            raise ValueError("every sample requires at least one valid action")

    def to(
        self,
        device: Union[str, torch.device],
        *,
        non_blocking: bool = False,
    ) -> "ACTAlignedHighRateBatch":
        return ACTAlignedHighRateBatch(
            **{
                item.name: getattr(self, item.name).to(
                    device,
                    non_blocking=non_blocking,
                )
                for item in fields(self)
            }
        )
