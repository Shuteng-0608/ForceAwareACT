"""Normalization statistics utilities for ForceAwareACT datasets."""

from __future__ import annotations

from typing import Iterable, Mapping

import torch
from torch.utils.data import DataLoader


STAT_KEYS = (
    "qpos_mean",
    "qpos_std",
    "action_mean",
    "action_std",
    "force_mean",
    "force_std",
)


class RunningStats:
    """Accumulate feature-wise mean and std from tensors."""

    def __init__(self) -> None:
        self.count = 0
        self.sum = None
        self.sum_sq = None

    def update(self, values: torch.Tensor) -> None:
        if values.numel() == 0:
            return
        values = values.detach().float().reshape(-1, values.shape[-1])
        batch_sum = values.sum(dim=0)
        batch_sum_sq = values.pow(2).sum(dim=0)
        if self.sum is None:
            self.sum = batch_sum
            self.sum_sq = batch_sum_sq
        else:
            self.sum = self.sum + batch_sum
            self.sum_sq = self.sum_sq + batch_sum_sq
        self.count += values.shape[0]

    def mean_std(self, eps: float) -> tuple[torch.Tensor, torch.Tensor]:
        if self.count == 0 or self.sum is None or self.sum_sq is None:
            raise ValueError("cannot compute statistics from zero samples")
        mean = self.sum / self.count
        variance = self.sum_sq / self.count - mean.pow(2)
        variance = torch.clamp(variance, min=0.0)
        std = torch.sqrt(variance).clamp_min(eps)
        return mean, std


def compute_normalization_stats_from_batches(
    batches: Iterable[Mapping[str, torch.Tensor]],
    eps: float = 1.0e-6,
) -> dict[str, torch.Tensor]:
    """Compute qpos/action/force normalization stats from dataset batches."""

    qpos_stats = RunningStats()
    action_stats = RunningStats()
    force_stats = RunningStats()

    for batch in batches:
        _require_batch_key(batch, "qpos")
        _require_batch_key(batch, "action_chunk")
        _require_batch_key(batch, "force_window")
        _require_batch_key(batch, "future_force_chunk")
        qpos_stats.update(batch["qpos"])
        action_stats.update(batch["action_chunk"])
        force_stats.update(batch["force_window"])
        force_stats.update(batch["future_force_chunk"])

    qpos_mean, qpos_std = qpos_stats.mean_std(eps)
    action_mean, action_std = action_stats.mean_std(eps)
    force_mean, force_std = force_stats.mean_std(eps)
    return {
        "qpos_mean": qpos_mean,
        "qpos_std": qpos_std,
        "action_mean": action_mean,
        "action_std": action_std,
        "force_mean": force_mean,
        "force_std": force_std,
    }


def compute_normalization_stats(
    dataset,
    batch_size: int = 64,
    num_workers: int = 0,
    eps: float = 1.0e-6,
) -> dict[str, torch.Tensor]:
    """Compute normalization statistics by iterating over a dataset."""

    if len(dataset) == 0:
        raise ValueError("cannot compute normalization statistics for an empty dataset")
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return compute_normalization_stats_from_batches(dataloader, eps=eps)


def normalize_tensor(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """Normalize tensor features along the last dimension."""

    return (x - _view_stats_for_tensor(mean, x)) / _view_stats_for_tensor(std, x)


def denormalize_tensor(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """Invert ``normalize_tensor`` for matching mean and std."""

    return x * _view_stats_for_tensor(std, x) + _view_stats_for_tensor(mean, x)


def _view_stats_for_tensor(stats: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    if stats.ndim != 1:
        raise ValueError("mean/std must be 1D tensors")
    if x.shape[-1] != stats.shape[0]:
        raise ValueError(
            f"tensor last dimension {x.shape[-1]} does not match stats dimension {stats.shape[0]}"
        )
    return stats.to(device=x.device, dtype=x.dtype).view(*([1] * (x.ndim - 1)), -1)


def _require_batch_key(batch: Mapping[str, torch.Tensor], key: str) -> None:
    if key not in batch:
        raise KeyError(f"batch is missing required key: {key}")
    if not torch.is_tensor(batch[key]):
        raise ValueError(f"batch[{key!r}] must be a torch.Tensor")
