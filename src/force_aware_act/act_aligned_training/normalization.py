"""Training-only normalization statistics for ACT-aligned data."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np
import torch

from force_aware_act.act_aligned_training.schema import load_state_aligned_force
from force_aware_act.act_aligned_training.split import EpisodeRecord


@dataclass(frozen=True)
class NormalizationStats:
    qpos_mean: tuple[float, ...]
    qpos_std: tuple[float, ...]
    action_mean: tuple[float, ...]
    action_std: tuple[float, ...]
    force_mean: tuple[float, ...]
    force_std: tuple[float, ...]
    minimum_std: float = 1.0e-6

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "NormalizationStats":
        return cls(
            qpos_mean=tuple(values["qpos_mean"]),
            qpos_std=tuple(values["qpos_std"]),
            action_mean=tuple(values["action_mean"]),
            action_std=tuple(values["action_std"]),
            force_mean=tuple(values["force_mean"]),
            force_std=tuple(values["force_std"]),
            minimum_std=float(values.get("minimum_std", 1.0e-6)),
        )

    def normalize_qpos(self, values: torch.Tensor) -> torch.Tensor:
        return _normalize(values, self.qpos_mean, self.qpos_std)

    def normalize_action(self, values: torch.Tensor) -> torch.Tensor:
        return _normalize(values, self.action_mean, self.action_std)

    def normalize_force(self, values: torch.Tensor) -> torch.Tensor:
        return _normalize(values, self.force_mean, self.force_std)

    def denormalize_action(self, values: torch.Tensor) -> torch.Tensor:
        return _denormalize(values, self.action_mean, self.action_std)

    def denormalize_force(self, values: torch.Tensor) -> torch.Tensor:
        return _denormalize(values, self.force_mean, self.force_std)


def compute_normalization_stats(
    data_root: Path,
    episodes: Iterable[EpisodeRecord],
    *,
    minimum_std: float = 1.0e-6,
) -> NormalizationStats:
    """Compute state-rate statistics from training episodes only."""

    if minimum_std <= 0:
        raise ValueError("minimum_std must be positive")
    qpos_accumulator = _MomentAccumulator(7)
    action_accumulator = _MomentAccumulator(7)
    force_accumulator = _MomentAccumulator(6)
    episode_count = 0
    for record in episodes:
        episode_count += 1
        with h5py.File(record.resolve(data_root), "r") as handle:
            qpos_accumulator.update(handle["observations/joint_pos"][...])
            action_accumulator.update(handle["action"][...])
            force_accumulator.update(load_state_aligned_force(handle))
    if episode_count == 0:
        raise ValueError("normalization requires at least one training episode")
    qpos_mean, qpos_std = qpos_accumulator.finalize(minimum_std)
    action_mean, action_std = action_accumulator.finalize(minimum_std)
    force_mean, force_std = force_accumulator.finalize(minimum_std)
    return NormalizationStats(
        qpos_mean=tuple(qpos_mean.tolist()),
        qpos_std=tuple(qpos_std.tolist()),
        action_mean=tuple(action_mean.tolist()),
        action_std=tuple(action_std.tolist()),
        force_mean=tuple(force_mean.tolist()),
        force_std=tuple(force_std.tolist()),
        minimum_std=float(minimum_std),
    )


def compute_high_rate_normalization_stats(
    data_root: Path,
    episodes: Iterable[EpisodeRecord],
    *,
    minimum_std: float = 1.0e-6,
) -> NormalizationStats:
    """Compute force statistics from every native-rate training sample.

    Joint positions and actions retain the ACT state-rate convention. Force
    statistics deliberately use the complete ``observations/ft_wrench``
    stream so normalization does not discard the information preserved by the
    500 Hz v2 data contract.
    """

    if minimum_std <= 0:
        raise ValueError("minimum_std must be positive")
    qpos_accumulator = _MomentAccumulator(7)
    action_accumulator = _MomentAccumulator(7)
    force_accumulator = _MomentAccumulator(6)
    episode_count = 0
    for record in episodes:
        episode_count += 1
        with h5py.File(record.resolve(data_root), "r") as handle:
            qpos_accumulator.update(handle["observations/joint_pos"][...])
            action_accumulator.update(handle["action"][...])
            force_accumulator.update(handle["observations/ft_wrench"][...])
    if episode_count == 0:
        raise ValueError("normalization requires at least one training episode")
    qpos_mean, qpos_std = qpos_accumulator.finalize(minimum_std)
    action_mean, action_std = action_accumulator.finalize(minimum_std)
    force_mean, force_std = force_accumulator.finalize(minimum_std)
    return NormalizationStats(
        qpos_mean=tuple(qpos_mean.tolist()),
        qpos_std=tuple(qpos_std.tolist()),
        action_mean=tuple(action_mean.tolist()),
        action_std=tuple(action_std.tolist()),
        force_mean=tuple(force_mean.tolist()),
        force_std=tuple(force_std.tolist()),
        minimum_std=float(minimum_std),
    )


class _MomentAccumulator:
    def __init__(self, dimension: int) -> None:
        self.count = 0
        self.total = np.zeros(dimension, dtype=np.float64)
        self.total_square = np.zeros(dimension, dtype=np.float64)

    def update(self, values: np.ndarray) -> None:
        array = np.asarray(values, dtype=np.float64)
        if array.ndim != 2 or array.shape[1] != self.total.shape[0]:
            raise ValueError("normalization array has an unexpected shape")
        self.count += int(array.shape[0])
        self.total += array.sum(axis=0)
        self.total_square += np.square(array).sum(axis=0)

    def finalize(self, minimum_std: float) -> tuple[np.ndarray, np.ndarray]:
        if self.count == 0:
            raise ValueError("normalization accumulator is empty")
        mean = self.total / self.count
        variance = np.maximum(self.total_square / self.count - np.square(mean), 0.0)
        std = np.maximum(np.sqrt(variance), minimum_std)
        return mean.astype(np.float32), std.astype(np.float32)


def _normalize(
    values: torch.Tensor,
    mean_values: tuple[float, ...],
    std_values: tuple[float, ...],
) -> torch.Tensor:
    mean = values.new_tensor(mean_values)
    std = values.new_tensor(std_values)
    return (values - mean) / std


def _denormalize(
    values: torch.Tensor,
    mean_values: tuple[float, ...],
    std_values: tuple[float, ...],
) -> torch.Tensor:
    mean = values.new_tensor(mean_values)
    std = values.new_tensor(std_values)
    return values * std + mean
