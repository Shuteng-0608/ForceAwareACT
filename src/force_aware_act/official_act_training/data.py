"""Official episodic ACT sampling adapted to the compact MuJoCo schema."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import h5py
import numpy as np
import torch
import torch.nn.functional as functional
from torch.utils.data import Dataset

from force_aware_act.act_aligned_training.split import (
    EpisodeRecord,
    discover_episodes,
)
from force_aware_act.models.official_act.config import OfficialACTConfig


OFFICIAL_ACT_SPLIT_VERSION = "official_act_episode_split_v1"
OFFICIAL_ACT_STATS_VERSION = "official_act_normalization_v1"


@dataclass(frozen=True)
class OfficialACTSplitManifest:
    format_version: str
    data_root: str
    split_seed: int
    validation_fraction: float
    train_episodes: tuple[EpisodeRecord, ...]
    validation_episodes: tuple[EpisodeRecord, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "OfficialACTSplitManifest":
        if values.get("format_version") != OFFICIAL_ACT_SPLIT_VERSION:
            raise ValueError("unsupported official ACT split format")

        def records(name: str) -> tuple[EpisodeRecord, ...]:
            return tuple(
                EpisodeRecord(
                    episode_id=item["episode_id"],
                    relative_hdf5_path=item["relative_hdf5_path"],
                    num_steps=int(item["num_steps"]),
                    camera_names=tuple(item["camera_names"]),
                )
                for item in values[name]
            )

        return cls(
            format_version=values["format_version"],
            data_root=values["data_root"],
            split_seed=int(values["split_seed"]),
            validation_fraction=float(values["validation_fraction"]),
            train_episodes=records("train_episodes"),
            validation_episodes=records("validation_episodes"),
        )


@dataclass(frozen=True)
class OfficialACTNormalizationStats:
    format_version: str
    qpos_mean: tuple[float, ...]
    qpos_std: tuple[float, ...]
    action_mean: tuple[float, ...]
    action_std: tuple[float, ...]
    minimum_std: float = 1.0e-2

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(
        cls,
        values: dict[str, Any],
    ) -> "OfficialACTNormalizationStats":
        if values.get("format_version") != OFFICIAL_ACT_STATS_VERSION:
            raise ValueError("unsupported official ACT normalization format")
        return cls(
            format_version=values["format_version"],
            qpos_mean=tuple(values["qpos_mean"]),
            qpos_std=tuple(values["qpos_std"]),
            action_mean=tuple(values["action_mean"]),
            action_std=tuple(values["action_std"]),
            minimum_std=float(values["minimum_std"]),
        )

    def normalize_qpos(self, values: torch.Tensor) -> torch.Tensor:
        return (
            values - values.new_tensor(self.qpos_mean)
        ) / values.new_tensor(self.qpos_std)

    def normalize_action(self, values: torch.Tensor) -> torch.Tensor:
        return (
            values - values.new_tensor(self.action_mean)
        ) / values.new_tensor(self.action_std)


@dataclass(frozen=True)
class OfficialACTBatch:
    images: torch.Tensor
    qpos: torch.Tensor
    action_chunk: torch.Tensor
    padding_mask: torch.Tensor

    @property
    def batch_size(self) -> int:
        return self.images.shape[0]

    def to(self, device: torch.device) -> "OfficialACTBatch":
        return OfficialACTBatch(
            images=self.images.to(device),
            qpos=self.qpos.to(device),
            action_chunk=self.action_chunk.to(device),
            padding_mask=self.padding_mask.to(device),
        )

    def validate(self, config: OfficialACTConfig) -> None:
        expected = {
            "images": (
                None,
                config.num_cameras,
                3,
                config.image_height,
                config.image_width,
            ),
            "qpos": (None, config.q_dim),
            "action_chunk": (
                None,
                config.chunk_len,
                config.action_dim,
            ),
            "padding_mask": (None, config.chunk_len),
        }
        batch_size = self.images.shape[0]
        for name, shape in expected.items():
            value = getattr(self, name)
            if value.ndim != len(shape):
                raise ValueError(f"{name} has an invalid rank")
            for actual, required in zip(value.shape, shape):
                if required is not None and actual != required:
                    raise ValueError(f"{name} has an invalid shape")
            if value.shape[0] != batch_size:
                raise ValueError("official ACT batch sizes must match")
        if self.padding_mask.dtype is not torch.bool:
            raise ValueError("padding_mask must have dtype torch.bool")
        for value in (self.images, self.qpos, self.action_chunk):
            if not value.is_floating_point():
                raise ValueError("official ACT inputs must be floating point")
            if value.device != self.images.device:
                raise ValueError("official ACT inputs must share a device")


def create_official_act_split(
    data_root: Path,
    *,
    validation_fraction: float = 0.2,
    split_seed: int = 1,
) -> OfficialACTSplitManifest:
    """Reproduce official NumPy permutation and 80/20 split semantics."""

    records = list(discover_episodes(data_root))
    generator = np.random.RandomState(split_seed)
    indices = generator.permutation(len(records))
    train_count = int((1.0 - validation_fraction) * len(records))
    if train_count <= 0 or train_count >= len(records):
        raise ValueError("split must contain train and validation episodes")
    train = tuple(records[int(index)] for index in indices[:train_count])
    validation = tuple(records[int(index)] for index in indices[train_count:])
    return OfficialACTSplitManifest(
        format_version=OFFICIAL_ACT_SPLIT_VERSION,
        data_root=str(Path(data_root).resolve()),
        split_seed=split_seed,
        validation_fraction=validation_fraction,
        train_episodes=train,
        validation_episodes=validation,
    )


def compute_official_act_stats(
    data_root: Path,
    episodes: Sequence[EpisodeRecord],
) -> OfficialACTNormalizationStats:
    """Use every episode and PyTorch's official unbiased std behavior."""

    if not episodes:
        raise ValueError("normalization requires episodes")
    qpos_values = []
    action_values = []
    for record in episodes:
        with h5py.File(record.resolve(data_root), "r") as handle:
            qpos_values.append(
                torch.from_numpy(
                    np.asarray(
                        handle["observations/joint_pos"][...],
                        dtype=np.float32,
                    )
                )
            )
            action_values.append(
                torch.from_numpy(
                    np.asarray(handle["action"][...], dtype=np.float32)
                )
            )
    qpos = torch.cat(qpos_values, dim=0)
    action = torch.cat(action_values, dim=0)
    minimum_std = 1.0e-2
    return OfficialACTNormalizationStats(
        format_version=OFFICIAL_ACT_STATS_VERSION,
        qpos_mean=tuple(qpos.mean(dim=0).tolist()),
        qpos_std=tuple(qpos.std(dim=0).clamp_min(minimum_std).tolist()),
        action_mean=tuple(action.mean(dim=0).tolist()),
        action_std=tuple(action.std(dim=0).clamp_min(minimum_std).tolist()),
        minimum_std=minimum_std,
    )


class OfficialACTEpisodicDataset(Dataset):
    """Return one uniformly sampled timestep from each episode per epoch."""

    def __init__(
        self,
        data_root: Path,
        episodes: Sequence[EpisodeRecord],
        stats: OfficialACTNormalizationStats,
        config: OfficialACTConfig,
        *,
        sampling_seed: int,
        stream_id: int,
    ) -> None:
        if not episodes:
            raise ValueError("dataset requires episodes")
        self.data_root = Path(data_root).resolve()
        self.episodes = tuple(episodes)
        self.stats = stats
        self.config = config
        self.sampling_seed = int(sampling_seed)
        self.stream_id = int(stream_id)
        self.epoch = 0
        for record in self.episodes:
            if len(record.camera_names) != config.num_cameras:
                raise ValueError(
                    f"episode {record.episode_id} camera count mismatch"
                )

    def __len__(self) -> int:
        return len(self.episodes)

    def set_epoch(self, epoch: int) -> None:
        if not isinstance(epoch, int) or isinstance(epoch, bool) or epoch < 0:
            raise ValueError("epoch must be a non-negative integer")
        self.epoch = epoch

    def sampled_timestep(self, index: int) -> int:
        record = self.episodes[index]
        seed = np.random.SeedSequence(
            (self.sampling_seed, self.stream_id, self.epoch, index)
        )
        generator = np.random.default_rng(seed)
        return int(generator.integers(0, record.num_steps))

    def __getitem__(self, index: int) -> OfficialACTBatch:
        record = self.episodes[index]
        timestep = self.sampled_timestep(index)
        with h5py.File(record.resolve(self.data_root), "r") as handle:
            images = torch.stack(
                [
                    torch.from_numpy(
                        np.asarray(
                            handle[f"observations/images/{camera_name}"][
                                timestep
                            ],
                            dtype=np.uint8,
                        ).copy()
                    ).permute(2, 0, 1)
                    for camera_name in record.camera_names
                ]
            ).float().div_(255.0)
            if images.shape[-2:] != (
                self.config.image_height,
                self.config.image_width,
            ):
                images = functional.interpolate(
                    images,
                    size=(
                        self.config.image_height,
                        self.config.image_width,
                    ),
                    mode="bilinear",
                    align_corners=False,
                    antialias=True,
                )
            qpos = torch.from_numpy(
                np.asarray(
                    handle["observations/joint_pos"][timestep],
                    dtype=np.float32,
                )
            )
            future_end = min(
                record.num_steps,
                timestep + self.config.chunk_len,
            )
            valid_length = future_end - timestep
            actions = torch.zeros(
                self.config.chunk_len,
                self.config.action_dim,
                dtype=torch.float32,
            )
            actions[:valid_length] = torch.from_numpy(
                np.asarray(
                    handle["action"][timestep:future_end],
                    dtype=np.float32,
                )
            )
        actions = self.stats.normalize_action(actions)
        padding_mask = torch.ones(self.config.chunk_len, dtype=torch.bool)
        padding_mask[:valid_length] = False
        return OfficialACTBatch(
            images=images,
            qpos=self.stats.normalize_qpos(qpos),
            action_chunk=actions,
            padding_mask=padding_mask,
        )


def collate_official_act(
    samples: Sequence[OfficialACTBatch],
) -> OfficialACTBatch:
    if not samples:
        raise ValueError("cannot collate an empty batch")
    return OfficialACTBatch(
        images=torch.stack([sample.images for sample in samples]),
        qpos=torch.stack([sample.qpos for sample in samples]),
        action_chunk=torch.stack(
            [sample.action_chunk for sample in samples]
        ),
        padding_mask=torch.stack(
            [sample.padding_mask for sample in samples]
        ),
    )
