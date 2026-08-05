"""Native 500 Hz dataset kept separate from the v1/motion data path."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch

from force_aware_act.act_aligned_training.data import ACTAlignedHDF5Dataset
from force_aware_act.act_aligned_training.high_rate_batch import (
    ACTAlignedHighRateBatch,
)
from force_aware_act.act_aligned_training.normalization import NormalizationStats
from force_aware_act.act_aligned_training.split import EpisodeRecord
from force_aware_act.high_rate_force import (
    HighRateForceContract,
    build_future_force_intervals,
    build_online_force_intervals,
)
from force_aware_act.models.act_aligned.config import ACTAlignedHighRateConfig


@dataclass(frozen=True)
class ACTAlignedHighRateSample:
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
    episode_id: str
    timestep: int
    online_raw_valid_count: int


class ACTAlignedHighRateHDF5Dataset(ACTAlignedHDF5Dataset):
    """Add timestamp-packed native-force intervals to the shared ACT inputs."""

    def __init__(
        self,
        data_root,
        episodes: Sequence[EpisodeRecord],
        normalization: NormalizationStats,
        model_config: ACTAlignedHighRateConfig,
    ) -> None:
        if not isinstance(model_config, ACTAlignedHighRateConfig):
            raise TypeError("high-rate dataset requires ACTAlignedHighRateConfig")
        super().__init__(data_root, episodes, normalization, model_config)
        self.high_rate_config = model_config
        self.force_contract = HighRateForceContract(
            sample_rate_hz=model_config.force_sample_rate_hz,
            online_window_len=model_config.online_force_window_len,
            max_samples_per_interval=model_config.max_force_samples_per_interval,
            max_online_intervals=model_config.max_online_force_intervals,
        )

    def __getitem__(self, item: int) -> ACTAlignedHighRateSample:
        base = super().__getitem__(item)
        episode_index, timestep = self.index[item]
        record = self.episodes[episode_index]
        handle = self._handle(record)
        state_timestamps = np.asarray(handle["timestamps/state"], dtype=np.float64)
        force_timestamps = np.asarray(handle["timestamps/force"], dtype=np.float64)
        force_values = np.asarray(handle["observations/ft_wrench"], dtype=np.float32)
        online_window, online = build_online_force_intervals(
            force_timestamps,
            force_values,
            state_timestamps,
            state_index=timestep,
            contract=self.force_contract,
        )
        future = build_future_force_intervals(
            force_timestamps,
            force_values,
            state_timestamps,
            state_index=timestep,
            chunk_len=self.high_rate_config.chunk_len,
            max_samples_per_interval=(
                self.high_rate_config.max_force_samples_per_interval
            ),
        )
        online_force = self._normalize_interval_force(
            online.values,
            online.sample_padding_mask,
        )
        future_force = self._normalize_interval_force(
            future.values,
            future.sample_padding_mask,
        )
        future_target = torch.zeros(
            self.high_rate_config.chunk_len,
            self.high_rate_config.force_dim,
            dtype=torch.float32,
        )
        valid_intervals = ~future.interval_padding_mask
        if np.any(valid_intervals):
            interval_indices = np.flatnonzero(valid_intervals)
            last_indices = future.sample_counts[valid_intervals] - 1
            raw_targets = future.values[interval_indices, last_indices]
            future_target[torch.from_numpy(interval_indices)] = (
                self.normalization.normalize_force(torch.from_numpy(raw_targets.copy()))
            )

        return ACTAlignedHighRateSample(
            images=base.images,
            qpos=base.qpos,
            action_chunk=base.action_chunk,
            action_padding_mask=base.future_padding_mask,
            online_force_intervals=online_force,
            online_force_relative_time=torch.from_numpy(online.relative_times.copy()),
            online_force_sample_padding_mask=torch.from_numpy(
                online.sample_padding_mask.copy()
            ),
            online_force_interval_padding_mask=torch.from_numpy(
                online.interval_padding_mask.copy()
            ),
            future_force_intervals=future_force,
            future_force_relative_time=torch.from_numpy(future.relative_times.copy()),
            future_force_sample_padding_mask=torch.from_numpy(
                future.sample_padding_mask.copy()
            ),
            future_force_interval_padding_mask=torch.from_numpy(
                future.interval_padding_mask.copy()
            ),
            future_force_target=future_target,
            episode_id=record.episode_id,
            timestep=timestep,
            online_raw_valid_count=online_window.valid_count,
        )

    def _normalize_interval_force(
        self,
        values: np.ndarray,
        padding_mask: np.ndarray,
    ) -> torch.Tensor:
        tensor = torch.from_numpy(values.copy())
        normalized = self.normalization.normalize_force(tensor)
        return normalized.masked_fill(
            torch.from_numpy(padding_mask).unsqueeze(-1),
            0.0,
        )


def collate_high_rate_samples(
    samples: Sequence[ACTAlignedHighRateSample],
) -> ACTAlignedHighRateBatch:
    if not samples:
        raise ValueError("cannot collate an empty sample sequence")
    names = (
        "images",
        "qpos",
        "action_chunk",
        "action_padding_mask",
        "online_force_intervals",
        "online_force_relative_time",
        "online_force_sample_padding_mask",
        "online_force_interval_padding_mask",
        "future_force_intervals",
        "future_force_relative_time",
        "future_force_sample_padding_mask",
        "future_force_interval_padding_mask",
        "future_force_target",
    )
    return ACTAlignedHighRateBatch(
        **{
            name: torch.stack([getattr(sample, name) for sample in samples])
            for name in names
        }
    )
