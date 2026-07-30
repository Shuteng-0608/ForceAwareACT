"""Fixed decision-window HDF5 dataset for the ACT-aligned trainer."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Sequence

import h5py
import numpy as np
import torch
import torch.nn.functional as functional
from torch.utils.data import Dataset

from force_aware_act.act_aligned_training.batch import ACTAlignedBatch
from force_aware_act.act_aligned_training.normalization import NormalizationStats
from force_aware_act.act_aligned_training.schema import (
    causal_alignment_indices,
    load_state_aligned_force,
)
from force_aware_act.act_aligned_training.split import EpisodeRecord
from force_aware_act.models.act_aligned.config import ACTAlignedConfig


@dataclass(frozen=True)
class ACTAlignedSample:
    images: torch.Tensor
    qpos: torch.Tensor
    force_history: torch.Tensor
    force_padding_mask: torch.Tensor
    action_chunk: torch.Tensor
    future_force_chunk: torch.Tensor
    future_padding_mask: torch.Tensor
    episode_id: str
    timestep: int


class ACTAlignedHDF5Dataset(Dataset):
    """Expose every state timestep as one deterministic ACT training window."""

    def __init__(
        self,
        data_root: Path,
        episodes: Sequence[EpisodeRecord],
        normalization: NormalizationStats,
        model_config: ACTAlignedConfig,
    ) -> None:
        if not episodes:
            raise ValueError("dataset requires at least one episode")
        if model_config.q_dim != 7 or model_config.action_dim != 7:
            raise ValueError("compact MuJoCo schema requires q_dim=action_dim=7")
        if model_config.force_dim != 6:
            raise ValueError("compact MuJoCo schema requires force_dim=6")
        for record in episodes:
            if len(record.camera_names) != model_config.num_cameras:
                raise ValueError(
                    f"episode {record.episode_id} camera count does not match config"
                )
        self.data_root = Path(data_root).resolve()
        self.episodes = tuple(episodes)
        self.normalization = normalization
        self.model_config = model_config
        self.index = tuple(
            (episode_index, timestep)
            for episode_index, record in enumerate(self.episodes)
            for timestep in range(record.num_steps)
        )
        self._handles: Dict[str, h5py.File] = {}
        self._force_cache: Dict[str, np.ndarray] = {}
        self._image_index_cache: Dict[str, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, item: int) -> ACTAlignedSample:
        episode_index, timestep = self.index[item]
        record = self.episodes[episode_index]
        handle = self._handle(record)
        aligned_force = self._aligned_force(record, handle)
        image_indices = self._image_indices(record, handle)

        images = torch.stack(
            [
                torch.from_numpy(
                    np.asarray(
                        handle[f"observations/images/{camera_name}"][
                            image_indices[timestep]
                        ],
                        dtype=np.uint8,
                    ).copy()
                ).permute(2, 0, 1)
                for camera_name in record.camera_names
            ]
        ).float().div_(255.0)
        target_image_size = (
            self.model_config.image_height,
            self.model_config.image_width,
        )
        if images.shape[-2:] != target_image_size:
            images = functional.interpolate(
                images,
                size=target_image_size,
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
        if self.model_config.imagenet_normalize:
            mean = images.new_tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
            std = images.new_tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
            images = (images - mean) / std

        qpos = torch.from_numpy(
            np.asarray(handle["observations/joint_pos"][timestep], dtype=np.float32)
        )
        qpos = self.normalization.normalize_qpos(qpos)

        force_history = torch.zeros(
            self.model_config.force_window_len,
            self.model_config.force_dim,
            dtype=torch.float32,
        )
        force_padding_mask = torch.ones(
            self.model_config.force_window_len,
            dtype=torch.bool,
        )
        history_start = max(0, timestep - self.model_config.force_window_len + 1)
        history = torch.from_numpy(aligned_force[history_start : timestep + 1].copy())
        history = self.normalization.normalize_force(history)
        force_history[-history.shape[0] :] = history
        force_padding_mask[-history.shape[0] :] = False

        future_end = min(
            record.num_steps,
            timestep + self.model_config.chunk_len,
        )
        valid_future_length = future_end - timestep
        action_chunk = torch.zeros(
            self.model_config.chunk_len,
            self.model_config.action_dim,
            dtype=torch.float32,
        )
        future_force_chunk = torch.zeros(
            self.model_config.chunk_len,
            self.model_config.force_dim,
            dtype=torch.float32,
        )
        future_padding_mask = torch.ones(
            self.model_config.chunk_len,
            dtype=torch.bool,
        )
        valid_action = torch.from_numpy(
            np.asarray(
                handle["action"][timestep:future_end],
                dtype=np.float32,
            )
        )
        valid_force = torch.from_numpy(aligned_force[timestep:future_end].copy())
        action_chunk[:valid_future_length] = self.normalization.normalize_action(
            valid_action
        )
        future_force_chunk[:valid_future_length] = (
            self.normalization.normalize_force(valid_force)
        )
        future_padding_mask[:valid_future_length] = False
        return ACTAlignedSample(
            images=images,
            qpos=qpos,
            force_history=force_history,
            force_padding_mask=force_padding_mask,
            action_chunk=action_chunk,
            future_force_chunk=future_force_chunk,
            future_padding_mask=future_padding_mask,
            episode_id=record.episode_id,
            timestep=timestep,
        )

    def close(self) -> None:
        for handle in self._handles.values():
            handle.close()
        self._handles.clear()

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_handles"] = {}
        state["_force_cache"] = {}
        state["_image_index_cache"] = {}
        return state

    def _handle(self, record: EpisodeRecord) -> h5py.File:
        if record.episode_id not in self._handles:
            self._handles[record.episode_id] = h5py.File(
                record.resolve(self.data_root),
                "r",
            )
        return self._handles[record.episode_id]

    def _aligned_force(
        self,
        record: EpisodeRecord,
        handle: h5py.File,
    ) -> np.ndarray:
        if record.episode_id not in self._force_cache:
            self._force_cache[record.episode_id] = load_state_aligned_force(handle)
        return self._force_cache[record.episode_id]

    def _image_indices(
        self,
        record: EpisodeRecord,
        handle: h5py.File,
    ) -> np.ndarray:
        if record.episode_id not in self._image_index_cache:
            self._image_index_cache[record.episode_id] = causal_alignment_indices(
                handle["timestamps/state"][...],
                handle["timestamps/image"][...],
            )
        return self._image_index_cache[record.episode_id]


def collate_act_aligned_samples(
    samples: Sequence[ACTAlignedSample],
) -> ACTAlignedBatch:
    if not samples:
        raise ValueError("cannot collate an empty sample sequence")
    return ACTAlignedBatch(
        images=torch.stack([sample.images for sample in samples]),
        qpos=torch.stack([sample.qpos for sample in samples]),
        force_history=torch.stack([sample.force_history for sample in samples]),
        force_padding_mask=torch.stack(
            [sample.force_padding_mask for sample in samples]
        ),
        action_chunk=torch.stack([sample.action_chunk for sample in samples]),
        future_force_chunk=torch.stack(
            [sample.future_force_chunk for sample in samples]
        ),
        future_padding_mask=torch.stack(
            [sample.future_padding_mask for sample in samples]
        ),
    )
