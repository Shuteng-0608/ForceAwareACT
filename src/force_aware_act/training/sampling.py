"""Shared, reproducible training-window sampling contracts."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from torch.utils.data import Dataset


EPISODIC_UNIFORM_SAMPLING = "episodic_uniform"
DENSE_WINDOW_SAMPLING = "dense_windows"
TRAIN_SAMPLING_MODES = (
    EPISODIC_UNIFORM_SAMPLING,
    DENSE_WINDOW_SAMPLING,
)


def validate_train_sampling_mode(value: str) -> str:
    if value not in TRAIN_SAMPLING_MODES:
        raise ValueError(
            f"train sampling mode must be one of {TRAIN_SAMPLING_MODES!r}"
        )
    return value


def sample_uniform_episode_timestep(
    *,
    sampling_seed: int,
    stream_id: int,
    epoch: int,
    episode_index: int,
    episode_steps: int,
) -> int:
    """Select one timestep using a deterministic episode/epoch RNG stream."""

    for name, value in (
        ("sampling_seed", sampling_seed),
        ("stream_id", stream_id),
        ("epoch", epoch),
        ("episode_index", episode_index),
        ("episode_steps", episode_steps),
    ):
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(f"{name} must be an integer")
    if epoch < 0 or episode_index < 0:
        raise ValueError("epoch and episode_index must be non-negative")
    if episode_steps <= 0:
        raise ValueError("episode_steps must be positive")
    seed = np.random.SeedSequence(
        (sampling_seed, stream_id, epoch, episode_index)
    )
    generator = np.random.default_rng(seed)
    return int(generator.integers(0, episode_steps))


class EpisodicUniformWindowDataset(Dataset):
    """View a dense window dataset as one sampled window per episode.

    The wrapped dataset must expose every episode contiguously in the same
    order as ``episodes``.  Sampling is deterministic for a fixed
    ``(seed, stream, epoch, episode)`` tuple and is therefore independent of
    DataLoader worker count and access order.
    """

    def __init__(
        self,
        window_dataset: Dataset,
        episodes: Sequence[Any],
        *,
        sampling_seed: int,
        stream_id: int,
    ) -> None:
        if not episodes:
            raise ValueError("episodic dataset requires episodes")
        self.window_dataset = window_dataset
        self.episodes = tuple(episodes)
        self.sampling_seed = int(sampling_seed)
        self.stream_id = int(stream_id)
        self.epoch = 0
        offsets = []
        total_steps = 0
        for record in self.episodes:
            steps = int(record.num_steps)
            if steps <= 0:
                raise ValueError("every episode must contain at least one step")
            offsets.append(total_steps)
            total_steps += steps
        if len(window_dataset) != total_steps:
            raise ValueError(
                "dense window dataset length does not match episode steps"
            )
        self._episode_offsets = tuple(offsets)

    def __len__(self) -> int:
        return len(self.episodes)

    def set_epoch(self, epoch: int) -> None:
        if not isinstance(epoch, int) or isinstance(epoch, bool) or epoch < 0:
            raise ValueError("epoch must be a non-negative integer")
        self.epoch = epoch

    def sampled_timestep(self, episode_index: int) -> int:
        episode_index = self._normalized_episode_index(episode_index)
        return sample_uniform_episode_timestep(
            sampling_seed=self.sampling_seed,
            stream_id=self.stream_id,
            epoch=self.epoch,
            episode_index=episode_index,
            episode_steps=int(self.episodes[episode_index].num_steps),
        )

    def dense_window_index(self, episode_index: int) -> int:
        episode_index = self._normalized_episode_index(episode_index)
        return (
            self._episode_offsets[episode_index]
            + self.sampled_timestep(episode_index)
        )

    def __getitem__(self, episode_index: int):
        return self.window_dataset[self.dense_window_index(episode_index)]

    def close(self) -> None:
        close = getattr(self.window_dataset, "close", None)
        if close is not None:
            close()

    def _normalized_episode_index(self, episode_index: int) -> int:
        if not isinstance(episode_index, int) or isinstance(episode_index, bool):
            raise TypeError("episode index must be an integer")
        if episode_index < 0:
            episode_index += len(self)
        if episode_index < 0 or episode_index >= len(self):
            raise IndexError("episode index is out of range")
        return episode_index
