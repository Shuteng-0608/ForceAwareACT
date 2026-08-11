"""Read-only helpers for the compact MuJoCo HDF5 schema."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import h5py
import numpy as np


EXPECTED_SCHEMA_VERSION = "compact_mujoco_hdf5_v1"
DEFAULT_CAMERA_NAMES = ("ee_cam", "base_top_cam")


@dataclass(frozen=True)
class EpisodeSchema:
    path: str
    num_steps: int
    num_force_samples: int
    num_image_samples: int
    camera_names: Tuple[str, ...]
    image_height: int
    image_width: int


def inspect_episode(path: Path) -> EpisodeSchema:
    """Validate one episode without loading image tensors."""

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"episode file does not exist: {path}")
    with h5py.File(path, "r") as handle:
        schema_version = _as_text(handle.attrs.get("schema_version", ""))
        if schema_version != EXPECTED_SCHEMA_VERSION:
            raise ValueError(
                f"{path} schema_version must be {EXPECTED_SCHEMA_VERSION!r}"
            )
        required = (
            "action",
            "observations/joint_pos",
            "observations/ft_wrench",
            "timestamps/state",
            "timestamps/force",
            "timestamps/image",
        )
        for key in required:
            if key not in handle:
                raise KeyError(f"{path} is missing HDF5 dataset {key!r}")

        state_timestamps = handle["timestamps/state"][...]
        force_timestamps = handle["timestamps/force"][...]
        image_timestamps = handle["timestamps/image"][...]
        if state_timestamps.ndim != 1 or state_timestamps.size == 0:
            raise ValueError(f"{path} state timestamps must have shape [N_state]")
        if force_timestamps.ndim != 1 or force_timestamps.size == 0:
            raise ValueError(f"{path} force timestamps must have shape [N_force]")
        if image_timestamps.ndim != 1 or image_timestamps.size == 0:
            raise ValueError(f"{path} image timestamps must have shape [N_image]")

        num_steps = int(state_timestamps.shape[0])
        num_force_samples = int(force_timestamps.shape[0])
        num_image_samples = int(image_timestamps.shape[0])
        if handle["action"].shape != (num_steps, 7):
            raise ValueError(f"{path} action must have shape [N, 7]")
        if handle["observations/joint_pos"].shape != (num_steps, 7):
            raise ValueError(f"{path} joint_pos must have shape [N, 7]")
        if handle["observations/ft_wrench"].shape != (num_force_samples, 6):
            raise ValueError(f"{path} ft_wrench must have shape [N_force, 6]")

        camera_names = tuple(
            _as_text(value)
            for value in handle["episode_metadata/camera_names"][...]
        )
        if not camera_names:
            raise ValueError(f"{path} must contain at least one camera")
        image_shape = None
        for camera_name in camera_names:
            key = f"observations/images/{camera_name}"
            if key not in handle:
                raise KeyError(f"{path} is missing camera dataset {key!r}")
            shape = handle[key].shape
            if (
                len(shape) != 4
                or shape[0] != num_image_samples
                or shape[-1] != 3
            ):
                raise ValueError(f"{path} camera {camera_name!r} has invalid shape")
            if image_shape is None:
                image_shape = shape[1:3]
            elif shape[1:3] != image_shape:
                raise ValueError(f"{path} camera resolutions must match")

        causal_alignment_indices(state_timestamps, force_timestamps)
        causal_alignment_indices(state_timestamps, image_timestamps)
        return EpisodeSchema(
            path=str(path),
            num_steps=num_steps,
            num_force_samples=num_force_samples,
            num_image_samples=num_image_samples,
            camera_names=camera_names,
            image_height=int(image_shape[0]),
            image_width=int(image_shape[1]),
        )


def causal_alignment_indices(
    target_timestamps: np.ndarray,
    source_timestamps: np.ndarray,
) -> np.ndarray:
    """Select the last source sample not later than each target timestamp."""

    target = np.asarray(target_timestamps, dtype=np.float64)
    source = np.asarray(source_timestamps, dtype=np.float64)
    if target.ndim != 1 or source.ndim != 1:
        raise ValueError("timestamps must be one-dimensional")
    if target.size == 0 or source.size == 0:
        raise ValueError("timestamps must not be empty")
    if np.any(np.diff(target) < 0) or np.any(np.diff(source) < 0):
        raise ValueError("timestamps must be monotonically non-decreasing")
    indices = np.searchsorted(source, target, side="right") - 1
    if np.any(indices < 0):
        raise ValueError("a target timestamp precedes every source timestamp")
    if np.any(source[indices] > target):
        raise RuntimeError("causal timestamp alignment selected a future sample")
    return indices.astype(np.int64, copy=False)


def load_state_aligned_force(handle: h5py.File) -> np.ndarray:
    """Load compensated force at state rate using causal timestamp matching."""

    state_timestamps = handle["timestamps/state"][...]
    force_timestamps = handle["timestamps/force"][...]
    indices = causal_alignment_indices(state_timestamps, force_timestamps)
    return np.asarray(handle["observations/ft_wrench"][indices], dtype=np.float32)


def _as_text(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)
