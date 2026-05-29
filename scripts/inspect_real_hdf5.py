#!/usr/bin/env python3
"""Read-only inspection utility for one ForceAwareACT HDF5 episode."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, Optional, Sequence

import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from force_aware_act.data import ContactForceHDF5Dataset  # noqa: E402


EXPECTED_DATASETS = (
    "observations/ee_pose",
    "observations/joint_pos",
    "observations/joint_vel",
    "observations/joint_torque",
    "observations/ft_wrench",
    "observations/images/ee_cam",
    "observations/images/base_top_cam",
)

TIMESTAMP_DATASETS = (
    ("state", "timestamps/state_episode"),
    ("force", "timestamps/force_episode"),
    ("image", "timestamps/image_episode"),
)


def _decode_value(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.bytes_):
        return value.decode("utf-8")
    return value


def _decode_sequence(values: Iterable) -> list:
    return [_decode_value(value) for value in values]


def _shape_string(handle: h5py.File, key: str) -> str:
    if key not in handle:
        return "missing"
    dataset = handle[key]
    if not isinstance(dataset, h5py.Dataset):
        return "not a dataset"
    return str(tuple(dataset.shape))


def _timestamp_summary(handle: h5py.File, key: str) -> str:
    if key not in handle:
        return "missing"
    values = np.asarray(handle[key])
    if values.ndim != 1:
        return f"length={len(values)}, range=unavailable (not 1D)"
    if len(values) == 0:
        return "length=0, range=empty"
    return f"length={len(values)}, range=[{float(values.min()):.6g}, {float(values.max()):.6g}]"


def _camera_names(handle: h5py.File) -> list[str]:
    candidate_keys = (
        "episode_metadata/camera_names",
        "observations/images/camera_names",
    )
    for key in candidate_keys:
        if key in handle and isinstance(handle[key], h5py.Dataset):
            values = np.asarray(handle[key])
            if values.ndim == 0:
                return [str(_decode_value(values.item()))]
            return [str(value) for value in _decode_sequence(values.tolist())]

    if "observations/images" not in handle:
        return []

    images_group = handle["observations/images"]
    if not isinstance(images_group, h5py.Group):
        return []

    return [
        name
        for name, item in images_group.items()
        if isinstance(item, h5py.Dataset) and name != "camera_names"
    ]


def _print_section(title: str) -> None:
    print(f"\n{title}")
    print("-" * len(title))


def _print_sample_shapes(sample: dict) -> None:
    for key in (
        "images",
        "qpos",
        "qvel",
        "joint_torque",
        "ee_pose",
        "force_window",
        "action_chunk",
        "future_force_chunk",
    ):
        value = sample[key]
        print(f"{key}: {tuple(value.shape)}")


def inspect_episode(path: Path, chunk_len: int, force_window_len: int) -> None:
    with h5py.File(path, "r") as handle:
        _print_section("Top-Level Groups")
        for name in handle.keys():
            print(name)

        _print_section("Expected Dataset Shapes")
        for key in EXPECTED_DATASETS:
            print(f"{key}: {_shape_string(handle, key)}")

        _print_section("Timestamp Lengths And Ranges")
        for label, key in TIMESTAMP_DATASETS:
            print(f"{label} ({key}): {_timestamp_summary(handle, key)}")

        cameras = _camera_names(handle)
        _print_section("Camera Names")
        if cameras:
            for camera_name in cameras:
                print(camera_name)
        else:
            print("missing")

    dataset_kwargs = {
        "chunk_len": chunk_len,
        "force_window_len": force_window_len,
    }
    if cameras:
        dataset_kwargs["camera_names"] = tuple(cameras)

    dataset = ContactForceHDF5Dataset(path, **dataset_kwargs)

    _print_section("Dataset Length")
    print(len(dataset))

    _print_section("Sample Tensor Shapes")
    if len(dataset) == 0:
        print("dataset is empty")
        return
    sample = dataset[0]
    _print_sample_shapes(sample)


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect one HDF5 episode without modifying it.",
    )
    parser.add_argument("episode_path", type=Path, help="Path to one HDF5 episode file.")
    parser.add_argument(
        "--chunk-len",
        type=int,
        default=50,
        help="Action/future-force chunk length used for dataset length and sample shape.",
    )
    parser.add_argument(
        "--force-window-len",
        type=int,
        default=50,
        help="Past-force window length used for sample shape.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    episode_path = args.episode_path.expanduser()
    if not episode_path.exists():
        print(f"error: file does not exist: {episode_path}", file=sys.stderr)
        return 2
    if not episode_path.is_file():
        print(f"error: path is not a file: {episode_path}", file=sys.stderr)
        return 2

    inspect_episode(
        path=episode_path,
        chunk_len=args.chunk_len,
        force_window_len=args.force_window_len,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
