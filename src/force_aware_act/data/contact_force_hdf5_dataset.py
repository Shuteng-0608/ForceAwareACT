"""HDF5 dataset reader for contact-dynamics-aware ACT samples."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence, Union

import h5py
import numpy as np
import torch
import torch.nn.functional as torch_functional
from torch.utils.data import Dataset


@dataclass(frozen=True)
class EpisodeIndex:
    """Location of one valid sample inside an episode."""

    episode_path: Path
    state_index: int


@dataclass(frozen=True)
class EpisodeSafeLengths:
    """Safe synchronized lengths for one episode."""

    state_len: int
    image_len: int
    force_len: int
    trim_state: int
    trim_image: int
    trim_force: int
    mismatch_groups: tuple[str, ...]


def nearest_index(timestamps: np.ndarray, target_time: float) -> int:
    """Return the index of the timestamp nearest to ``target_time``."""

    if timestamps.ndim != 1:
        raise ValueError("timestamps must be a 1D array")
    if len(timestamps) == 0:
        raise ValueError("timestamps must not be empty")

    insert_at = int(np.searchsorted(timestamps, target_time, side="left"))
    if insert_at == 0:
        return 0
    if insert_at >= len(timestamps):
        return len(timestamps) - 1

    before = insert_at - 1
    after = insert_at
    if abs(timestamps[after] - target_time) < abs(target_time - timestamps[before]):
        return after
    return before


EpisodePaths = Union[str, Path, Iterable[Union[str, Path]]]
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
TIMESTAMP_STATE_KEYS = ("timestamps/state_episode", "timestamps/state")
TIMESTAMP_FORCE_KEYS = ("timestamps/force_episode", "timestamps/force")
TIMESTAMP_IMAGE_KEYS = ("timestamps/image_episode", "timestamps/image")
ACTION_MODE_TO_DATASET = {
    "joint_pos": "observations/joint_pos",
    "action": "action",
    "joint_pos_command": "actions/joint_pos_command",
    "delta_joint_cmd": "action",
    "delta_joint_pos_command": "actions/joint_pos_command",
}
DELTA_ACTION_MODES = {"delta_joint_cmd", "delta_joint_pos_command"}


def _as_path_list(episode_paths: EpisodePaths) -> list[Path]:
    if isinstance(episode_paths, (str, Path)):
        return [Path(episode_paths)]
    return [Path(path) for path in episode_paths]


def _read_required_array(handle: h5py.File, key: str) -> np.ndarray:
    if key not in handle:
        raise KeyError(f"missing required HDF5 dataset: {key}")
    return np.asarray(handle[key])


def _first_existing_key(handle: h5py.File, keys: Sequence[str]) -> str:
    for key in keys:
        if key in handle:
            return key
    raise KeyError(f"missing required HDF5 dataset; tried: {', '.join(keys)}")


def _read_required_array_any(handle: h5py.File, keys: Sequence[str]) -> np.ndarray:
    return _read_required_array(handle, _first_existing_key(handle, keys))


def _safe_group_length(
    episode_path: Path,
    group_name: str,
    lengths: dict[str, int],
    tolerate_length_mismatch: bool,
    max_length_mismatch: int,
) -> tuple[int, int, bool]:
    minimum = min(lengths.values())
    maximum = max(lengths.values())
    difference = maximum - minimum
    allowed = max_length_mismatch if tolerate_length_mismatch else 0
    if difference > allowed:
        details = ", ".join(f"{key}={value}" for key, value in lengths.items())
        raise ValueError(
            f"{episode_path}: {group_name} group length mismatch {difference} exceeds "
            f"max_length_mismatch={allowed} ({details})"
        )
    if minimum <= 0:
        raise ValueError(f"{episode_path}: {group_name} group has no usable samples")
    return minimum, difference, difference > 0


def get_episode_safe_lengths(
    handle: h5py.File,
    episode_path: Union[str, Path],
    camera_names: Sequence[str] = ("ee_cam", "base_top_cam"),
    tolerate_length_mismatch: bool = True,
    max_length_mismatch: int = 1,
    include_force: bool = True,
) -> EpisodeSafeLengths:
    """Validate synchronization-group lengths and return safe trimmed lengths."""

    episode_path = Path(episode_path)
    if max_length_mismatch < 0:
        raise ValueError("max_length_mismatch must be non-negative")

    state_keys = (
        "observations/ee_pose",
        "observations/joint_pos",
        "observations/joint_vel",
        "observations/joint_torque",
        _first_existing_key(handle, TIMESTAMP_STATE_KEYS),
    )
    image_keys = tuple(f"observations/images/{name}" for name in camera_names) + (
        _first_existing_key(handle, TIMESTAMP_IMAGE_KEYS),
    )
    force_keys = (
        ("observations/ft_wrench", _first_existing_key(handle, TIMESTAMP_FORCE_KEYS))
        if include_force
        else ()
    )

    def lengths_for(keys: Sequence[str]) -> dict[str, int]:
        lengths: dict[str, int] = {}
        for key in keys:
            if key not in handle:
                raise KeyError(f"missing required HDF5 dataset: {key}")
            lengths[key] = len(handle[key])
        return lengths

    state_len, trim_state, state_mismatch = _safe_group_length(
        episode_path,
        "state",
        lengths_for(state_keys),
        tolerate_length_mismatch,
        max_length_mismatch,
    )
    image_len, trim_image, image_mismatch = _safe_group_length(
        episode_path,
        "image",
        lengths_for(image_keys),
        tolerate_length_mismatch,
        max_length_mismatch,
    )
    if include_force:
        force_len, trim_force, force_mismatch = _safe_group_length(
            episode_path,
            "force",
            lengths_for(force_keys),
            tolerate_length_mismatch,
            max_length_mismatch,
        )
    else:
        force_len, trim_force, force_mismatch = 0, 0, False
    mismatch_groups = tuple(
        name
        for name, mismatched in (
            ("state", state_mismatch),
            ("image", image_mismatch),
            ("force", force_mismatch),
        )
        if mismatched
    )
    return EpisodeSafeLengths(
        state_len=state_len,
        image_len=image_len,
        force_len=force_len,
        trim_state=trim_state,
        trim_image=trim_image,
        trim_force=trim_force,
        mismatch_groups=mismatch_groups,
    )


def _sample_past_force_window(
    force_ts: np.ndarray,
    force_values: np.ndarray,
    target_time: float,
    window_len: int,
    window_duration: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample a fixed-length window using only force timestamps <= target time."""

    if window_len <= 0:
        raise ValueError("force_window_len must be positive")
    if window_duration < 0:
        raise ValueError("force_window_duration must be non-negative")

    start_time = target_time - window_duration
    sample_times = np.linspace(start_time, target_time, num=window_len)
    force_indices = np.searchsorted(force_ts, sample_times, side="right") - 1
    force_indices = np.clip(force_indices, 0, len(force_ts) - 1)

    valid_past_mask = force_ts[force_indices] <= target_time
    if not np.all(valid_past_mask):
        raise RuntimeError("force window selected a future force sample")

    return force_values[force_indices], force_indices


class ContactForceHDF5Dataset(Dataset):
    """Dataset for contact-rich manipulation episodes stored as HDF5 files.

    ``action_mode="joint_pos"`` keeps the legacy behavior where the action chunk
    is ``joint_pos[i + 1 : i + K + 1]``. Executable command modes use labels
    aligned to the current decision index, ``command[i : i + K]``.
    """

    def __init__(
        self,
        episode_paths: EpisodePaths,
        camera_names: Sequence[str] = ("ee_cam", "base_top_cam"),
        action_mode: str = "joint_pos",
        chunk_len: int = 50,
        force_window_len: int = 50,
        force_window_duration: float = 0.25,
        image_size: tuple[int, int] = (224, 224),
        normalize_images: bool = True,
        imagenet_normalize: bool = False,
        include_force: bool = True,
        tolerate_length_mismatch: bool = True,
        max_length_mismatch: int = 1,
    ) -> None:
        self.episode_paths = _as_path_list(episode_paths)
        self.camera_names = tuple(camera_names)
        self.action_mode = action_mode
        self.chunk_len = int(chunk_len)
        self.force_window_len = int(force_window_len)
        self.force_window_duration = float(force_window_duration)
        self.image_size = image_size
        self.normalize_images = normalize_images
        self.imagenet_normalize = imagenet_normalize
        self.include_force = bool(include_force)
        self.tolerate_length_mismatch = bool(tolerate_length_mismatch)
        self.max_length_mismatch = int(max_length_mismatch)
        self.action_offset = 1 if action_mode == "joint_pos" else 0
        self.episode_safe_lengths: dict[Path, EpisodeSafeLengths] = {}
        self.episode_action_lengths: dict[Path, int] = {}

        if self.action_mode not in ACTION_MODE_TO_DATASET:
            supported = ", ".join(sorted(ACTION_MODE_TO_DATASET))
            raise ValueError(f"unsupported action_mode={self.action_mode!r}; supported: {supported}")
        if self.chunk_len <= 0:
            raise ValueError("chunk_len must be positive")
        if len(self.image_size) != 2 or self.image_size[0] <= 0 or self.image_size[1] <= 0:
            raise ValueError("image_size must be a positive (height, width) tuple")
        if not self.camera_names:
            raise ValueError("at least one camera name is required")
        if self.imagenet_normalize and not self.normalize_images:
            raise ValueError("imagenet_normalize=True requires normalize_images=True")
        if self.max_length_mismatch < 0:
            raise ValueError("max_length_mismatch must be non-negative")

        self.indices = self._build_indices()

    def __len__(self) -> int:
        return len(self.indices)

    @property
    def action_dim(self) -> int:
        return 7

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample_index = self.indices[index]
        safe_lengths = self.episode_safe_lengths[sample_index.episode_path]
        with h5py.File(sample_index.episode_path, "r") as handle:
            state_ts = _read_required_array_any(handle, TIMESTAMP_STATE_KEYS)[: safe_lengths.state_len]
            image_ts = _read_required_array_any(handle, TIMESTAMP_IMAGE_KEYS)[: safe_lengths.image_len]

            joint_pos = _read_required_array(handle, "observations/joint_pos")[: safe_lengths.state_len]
            joint_vel = _read_required_array(handle, "observations/joint_vel")[: safe_lengths.state_len]
            joint_torque = _read_required_array(handle, "observations/joint_torque")[
                : safe_lengths.state_len
            ]
            ee_pose = _read_required_array(handle, "observations/ee_pose")[: safe_lengths.state_len]

            state_index = sample_index.state_index
            t_state = float(state_ts[state_index])
            image_index = nearest_index(image_ts, t_state)

            images = self._read_images(handle, image_index, safe_lengths.image_len)

            action_chunk = self._read_action_chunk(
                handle=handle,
                joint_pos=joint_pos,
                state_index=state_index,
                action_len=self.episode_action_lengths[sample_index.episode_path],
            )

            force_window = None
            force_indices = None
            future_force_chunk = None
            if self.include_force:
                force_ts = _read_required_array_any(handle, TIMESTAMP_FORCE_KEYS)[
                    : safe_lengths.force_len
                ]
                ft_wrench = _read_required_array(handle, "observations/ft_wrench")[
                    : safe_lengths.force_len
                ]
                force_window, force_indices = _sample_past_force_window(
                    force_ts=force_ts,
                    force_values=ft_wrench,
                    target_time=t_state,
                    window_len=self.force_window_len,
                    window_duration=self.force_window_duration,
                )
                future_force_indices = np.array(
                    [
                        nearest_index(force_ts, float(state_ts[state_index + step]))
                        for step in range(self.chunk_len)
                    ],
                    dtype=np.int64,
                )
                future_force_chunk = ft_wrench[future_force_indices]

        sample = {
            "images": torch.from_numpy(images.astype(np.float32, copy=False)),
            "qpos": torch.from_numpy(joint_pos[state_index].astype(np.float32, copy=False)),
            "qvel": torch.from_numpy(joint_vel[state_index].astype(np.float32, copy=False)),
            "joint_torque": torch.from_numpy(
                joint_torque[state_index].astype(np.float32, copy=False)
            ),
            "ee_pose": torch.from_numpy(ee_pose[state_index].astype(np.float32, copy=False)),
            "action_chunk": torch.from_numpy(action_chunk.astype(np.float32, copy=False)),
            "episode_path": str(sample_index.episode_path),
            "state_index": state_index,
            "t_state": t_state,
            "image_index": image_index,
        }
        if self.include_force:
            sample.update(
                {
                    "force_window": torch.from_numpy(force_window.astype(np.float32, copy=False)),
                    "future_force_chunk": torch.from_numpy(
                        future_force_chunk.astype(np.float32, copy=False)
                    ),
                    "force_indices": force_indices,
                }
            )
        return sample

    def _build_indices(self) -> list[EpisodeIndex]:
        indices: list[EpisodeIndex] = []
        for episode_path in self.episode_paths:
            with h5py.File(episode_path, "r") as handle:
                safe_lengths = get_episode_safe_lengths(
                    handle,
                    episode_path,
                    camera_names=self.camera_names,
                    tolerate_length_mismatch=self.tolerate_length_mismatch,
                    max_length_mismatch=self.max_length_mismatch,
                    include_force=self.include_force,
                )
                self.episode_safe_lengths[episode_path] = safe_lengths
                action_len = self._safe_action_length(handle, episode_path, safe_lengths.state_len)
                self.episode_action_lengths[episode_path] = action_len
                n_state = safe_lengths.state_len
                max_start = min(n_state, action_len) - self.chunk_len - self.action_offset
                if max_start <= 0:
                    continue
                indices.extend(
                    EpisodeIndex(episode_path=episode_path, state_index=state_index)
                    for state_index in range(max_start)
                )
        return indices

    def _read_images(self, handle: h5py.File, image_index: int, image_len: int) -> np.ndarray:
        if image_index >= image_len:
            raise IndexError(f"image index {image_index} exceeds safe image length {image_len}")
        camera_images = []
        for camera_name in self.camera_names:
            key = f"observations/images/{camera_name}"
            if key not in handle:
                raise KeyError(f"missing required HDF5 dataset: {key}")
            image = np.asarray(handle[key][image_index])
            if image.ndim != 3 or image.shape[-1] != 3:
                raise ValueError(f"{key} must have shape [N, H, W, 3]")
            image = self._preprocess_image(image)
            camera_images.append(image)

        return np.stack(camera_images, axis=0).astype(np.float32)

    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        image = image.astype(np.float32)
        if self.normalize_images:
            image = image / 255.0

        image_chw = np.transpose(image, (2, 0, 1))
        image_tensor = torch.from_numpy(image_chw).unsqueeze(0)
        image_tensor = torch_functional.interpolate(
            image_tensor,
            size=self.image_size,
            mode="bilinear",
            align_corners=False,
        )
        image_chw = image_tensor.squeeze(0).numpy()

        if self.imagenet_normalize:
            image_chw = (image_chw - IMAGENET_MEAN[:, None, None]) / IMAGENET_STD[:, None, None]

        return image_chw.astype(np.float32, copy=False)

    def _action_dataset_key(self) -> str:
        return ACTION_MODE_TO_DATASET[self.action_mode]

    def _safe_action_length(
        self,
        handle: h5py.File,
        episode_path: Path,
        state_len: int,
    ) -> int:
        key = self._action_dataset_key()
        if key not in handle:
            raise KeyError(f"{episode_path}: missing action dataset for action_mode={self.action_mode!r}: {key}")

        action_values = handle[key]
        if action_values.ndim != 2 or action_values.shape[1] != self.action_dim:
            raise ValueError(
                f"{episode_path}: {key} must have shape [N, {self.action_dim}] "
                f"for action_mode={self.action_mode!r}; got {action_values.shape}"
            )

        action_len = len(action_values)
        difference = abs(action_len - state_len)
        allowed = self.max_length_mismatch if self.tolerate_length_mismatch else 0
        if difference > allowed:
            raise ValueError(
                f"{episode_path}: action dataset length mismatch {difference} exceeds "
                f"max_length_mismatch={allowed} for action_mode={self.action_mode!r} "
                f"({key}={action_len}, safe_state_len={state_len})"
            )
        if action_len <= 0:
            raise ValueError(f"{episode_path}: {key} has no usable action labels")
        return min(action_len, state_len)

    def _read_action_chunk(
        self,
        handle: h5py.File,
        joint_pos: np.ndarray,
        state_index: int,
        action_len: int,
    ) -> np.ndarray:
        key = self._action_dataset_key()
        action_start = state_index + self.action_offset
        action_stop = action_start + self.chunk_len
        if action_stop > action_len:
            raise IndexError(
                f"action chunk [{action_start}:{action_stop}] exceeds safe action length {action_len}"
            )

        source = _read_required_array(handle, key)[:action_len]
        action_chunk = source[action_start:action_stop]
        if self.action_mode in DELTA_ACTION_MODES:
            action_chunk = action_chunk - joint_pos[state_index][None, :]
        if action_chunk.shape != (self.chunk_len, self.action_dim):
            raise ValueError(
                f"action chunk must have shape [{self.chunk_len}, {self.action_dim}], "
                f"got {action_chunk.shape}"
            )
        return action_chunk
