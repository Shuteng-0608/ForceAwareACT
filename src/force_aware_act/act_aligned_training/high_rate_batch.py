"""Independent v2 batch contract for native 500 Hz force training."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Optional, Union

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
    future_force_target_padding_mask: Optional[torch.Tensor] = None

    @property
    def force_target_padding_mask(self) -> torch.Tensor:
        """Endpoint validity is independent of other samples in its interval."""
        if self.future_force_target_padding_mask is None:
            return self.future_force_interval_padding_mask
        return self.future_force_target_padding_mask

    @property
    def batch_size(self) -> int:
        return int(self.images.shape[0])

    def validate(self, config: ACTAlignedHighRateConfig) -> None:
        b = self.batch_size
        k = config.chunk_len
        f = config.force_prediction_len
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
            "future_force_intervals": (b, f, s, config.force_dim),
            "future_force_relative_time": (b, f, s),
            "future_force_sample_padding_mask": (b, f, s),
            "future_force_interval_padding_mask": (b, f),
            "future_force_target": (b, f, config.force_dim),
            "future_force_target_padding_mask": (b, f),
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
            if name == "future_force_target_padding_mask" and tensor is None:
                continue
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
        if torch.any(self.future_force_interval_padding_mask & ~self.force_target_padding_mask):
            raise ValueError("a valid force endpoint requires a valid force interval")
        valid_values = {
            "images": self.images,
            "qpos": self.qpos,
            "action_chunk": self.action_chunk[~self.action_padding_mask],
            "online_force_intervals": self.online_force_intervals[~self.online_force_sample_padding_mask],
            "online_force_relative_time": self.online_force_relative_time[~self.online_force_sample_padding_mask],
            "future_force_intervals": self.future_force_intervals[~self.future_force_sample_padding_mask],
            "future_force_relative_time": self.future_force_relative_time[~self.future_force_sample_padding_mask],
            "future_force_target": self.future_force_target[~self.force_target_padding_mask],
        }
        for name, values in valid_values.items():
            if not torch.isfinite(values).all():
                raise ValueError(f"valid {name} values must be finite")

    def to(
        self,
        device: Union[str, torch.device],
        *,
        non_blocking: bool = False,
    ) -> "ACTAlignedHighRateBatch":
        return ACTAlignedHighRateBatch(
            **{
                item.name: None if getattr(self, item.name) is None else getattr(self, item.name).to(
                    device,
                    non_blocking=non_blocking,
                )
                for item in fields(self)
            }
        )
