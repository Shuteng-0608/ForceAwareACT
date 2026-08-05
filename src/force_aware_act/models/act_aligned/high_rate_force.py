"""Shared local encoder for native 500 Hz wrench intervals."""

from __future__ import annotations

import math

import torch
from torch import nn

from force_aware_act.models.act_aligned.config import ACTAlignedHighRateConfig


class ResidualTemporalConvBlock(nn.Module):
    """Mask-safe residual temporal convolution at one dilation."""

    def __init__(self, channels: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(channels)
        self.conv1 = nn.Conv1d(
            channels,
            channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            bias=False,
        )
        self.norm2 = nn.LayerNorm(channels)
        self.conv2 = nn.Conv1d(
            channels,
            channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            bias=False,
        )
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        valid_float = valid.unsqueeze(-1).to(dtype=values.dtype)
        hidden = self.norm1(values) * valid_float
        hidden = self.conv1(hidden.transpose(1, 2)).transpose(1, 2)
        hidden = self.activation(hidden) * valid_float
        hidden = self.norm2(hidden) * valid_float
        hidden = self.conv2(hidden.transpose(1, 2)).transpose(1, 2)
        hidden = self.dropout(hidden) * valid_float
        return self.activation(values + hidden) * valid_float


class ACTAlignedHighRateForceEncoder(nn.Module):
    """Encode each short 500 Hz interval into one ``d_model`` token."""

    def __init__(self, config: ACTAlignedHighRateConfig) -> None:
        super().__init__()
        self.config = config
        self.input_projection = nn.Linear(config.force_dim, config.local_force_dim)
        dilations = tuple(2**index for index in range(config.local_force_encoder_layers))
        self.blocks = nn.ModuleList(
            ResidualTemporalConvBlock(
                config.local_force_dim,
                dilation=dilation,
                dropout=config.dropout,
            )
            for dilation in dilations
        )
        self.output_projection = nn.Linear(
            3 * config.local_force_dim,
            config.d_model,
        )
        self.output_norm = nn.LayerNorm(config.d_model)

    def forward(
        self,
        force_intervals: torch.Tensor,
        relative_times: torch.Tensor,
        sample_padding_mask: torch.Tensor,
        interval_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return tokens with shape ``[B, interval_count, d_model]``."""

        self._validate_inputs(
            force_intervals,
            relative_times,
            sample_padding_mask,
            interval_padding_mask,
        )
        batch_size, interval_count, sample_count, _ = force_intervals.shape
        flat_force = force_intervals.reshape(
            batch_size * interval_count,
            sample_count,
            self.config.force_dim,
        )
        flat_time = relative_times.reshape(batch_size * interval_count, sample_count)
        flat_padding = sample_padding_mask.reshape(
            batch_size * interval_count,
            sample_count,
        )
        valid = ~flat_padding
        hidden = self.input_projection(flat_force)
        hidden = hidden + self._time_encoding(flat_time, hidden.dtype)
        hidden = hidden * valid.unsqueeze(-1).to(dtype=hidden.dtype)
        for block in self.blocks:
            hidden = block(hidden, valid)

        valid_float = valid.unsqueeze(-1).to(dtype=hidden.dtype)
        counts = valid_float.sum(dim=1).clamp_min(1.0)
        mean_pool = (hidden * valid_float).sum(dim=1) / counts
        lowest = torch.finfo(hidden.dtype).min
        max_pool = hidden.masked_fill(~valid.unsqueeze(-1), lowest).max(dim=1).values
        valid_interval = valid.any(dim=1)
        max_pool = torch.where(valid_interval.unsqueeze(-1), max_pool, 0.0)
        last_indices = valid.sum(dim=1).clamp_min(1) - 1
        last_pool = hidden[
            torch.arange(hidden.shape[0], device=hidden.device),
            last_indices,
        ]
        last_pool = last_pool * valid_interval.unsqueeze(-1).to(last_pool.dtype)

        pooled = torch.cat((mean_pool, max_pool, last_pool), dim=-1)
        tokens = self.output_norm(self.output_projection(pooled))
        tokens = tokens * valid_interval.unsqueeze(-1).to(tokens.dtype)
        return tokens.reshape(batch_size, interval_count, self.config.d_model)

    def _time_encoding(
        self,
        relative_times: torch.Tensor,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        coordinates = relative_times * self.config.force_sample_rate_hz
        frequencies = torch.exp(
            torch.arange(
                0,
                self.config.local_force_dim,
                2,
                dtype=torch.float32,
                device=relative_times.device,
            )
            * (-math.log(10000.0) / self.config.local_force_dim)
        )
        angles = coordinates.to(torch.float32).unsqueeze(-1) * frequencies
        encoding = torch.zeros(
            *relative_times.shape,
            self.config.local_force_dim,
            dtype=torch.float32,
            device=relative_times.device,
        )
        encoding[..., 0::2] = angles.sin()
        encoding[..., 1::2] = angles.cos()
        return encoding.to(dtype=dtype)

    def _validate_inputs(
        self,
        force_intervals: torch.Tensor,
        relative_times: torch.Tensor,
        sample_padding_mask: torch.Tensor,
        interval_padding_mask: torch.Tensor,
    ) -> None:
        if not isinstance(force_intervals, torch.Tensor):
            raise TypeError("force_intervals must be a torch.Tensor")
        expected_suffix = (
            self.config.max_force_samples_per_interval,
            self.config.force_dim,
        )
        if force_intervals.ndim != 4 or tuple(force_intervals.shape[-2:]) != expected_suffix:
            raise ValueError(
                "force_intervals must have shape "
                f"[B, I, {expected_suffix[0]}, {expected_suffix[1]}]"
            )
        if not force_intervals.is_floating_point():
            raise ValueError("force_intervals must be floating point")
        batch_size, interval_count, sample_count, _ = force_intervals.shape
        expected_sample_shape = (batch_size, interval_count, sample_count)
        if not isinstance(relative_times, torch.Tensor) or tuple(
            relative_times.shape
        ) != expected_sample_shape:
            raise ValueError(
                f"relative_times must have shape {expected_sample_shape}"
            )
        if not relative_times.is_floating_point():
            raise ValueError("relative_times must be floating point")
        if not isinstance(sample_padding_mask, torch.Tensor) or tuple(
            sample_padding_mask.shape
        ) != expected_sample_shape:
            raise ValueError(
                f"sample_padding_mask must have shape {expected_sample_shape}"
            )
        if sample_padding_mask.dtype is not torch.bool:
            raise ValueError("sample_padding_mask must have dtype torch.bool")
        expected_interval_shape = (batch_size, interval_count)
        if not isinstance(interval_padding_mask, torch.Tensor) or tuple(
            interval_padding_mask.shape
        ) != expected_interval_shape:
            raise ValueError(
                f"interval_padding_mask must have shape {expected_interval_shape}"
            )
        if interval_padding_mask.dtype is not torch.bool:
            raise ValueError("interval_padding_mask must have dtype torch.bool")
        derived_interval_mask = sample_padding_mask.all(dim=-1)
        if not torch.equal(interval_padding_mask, derived_interval_mask):
            raise ValueError(
                "interval_padding_mask must equal sample_padding_mask.all(-1)"
            )
        tensors = (
            relative_times,
            sample_padding_mask,
            interval_padding_mask,
        )
        if any(tensor.device != force_intervals.device for tensor in tensors):
            raise ValueError("all high-rate force inputs must share one device")
        if relative_times.dtype != force_intervals.dtype:
            raise ValueError("relative_times must share force_intervals dtype")
        valid = ~sample_padding_mask
        if not torch.isfinite(force_intervals[valid]).all():
            raise ValueError("valid force samples must be finite")
        if not torch.isfinite(relative_times[valid]).all():
            raise ValueError("valid relative times must be finite")
        valid_prefix = valid.to(torch.int8).diff(dim=-1)
        if torch.any(valid_prefix > 0):
            raise ValueError("valid force samples must be left-aligned in each interval")
