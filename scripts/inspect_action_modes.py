#!/usr/bin/env python3
"""Inspect ContactForceHDF5Dataset action modes on command-labeled episodes."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from force_aware_act.data import ContactForceHDF5Dataset  # noqa: E402


def _find_episodes(data_dir: Path) -> list[Path]:
    if data_dir.name == "episode.hdf5":
        return [data_dir]
    return sorted(data_dir.glob("*/episode.hdf5"))


def _stats(values: np.ndarray) -> str:
    return (
        f"shape={values.shape} "
        f"min={values.min():.6g} mean={values.mean():.6g} max={values.max():.6g}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("mujoco_data/peg_hole_playback_test"))
    parser.add_argument("--camera-names", nargs="+", default=["ee_cam", "base_top_cam"])
    parser.add_argument("--chunk-len", type=int, default=10)
    parser.add_argument("--force-window-len", type=int, default=20)
    parser.add_argument("--sample-index", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    episodes = _find_episodes(args.data_dir)
    if not episodes:
        raise FileNotFoundError(f"no episode.hdf5 files found under {args.data_dir}")

    episode = episodes[0]
    print(f"episode={episode}")
    print(f"episode_count={len(episodes)}")

    modes = ("joint_pos", "action", "joint_pos_command", "delta_joint_cmd")
    samples = {}
    for mode in modes:
        dataset = ContactForceHDF5Dataset(
            episode,
            camera_names=args.camera_names,
            action_mode=mode,
            chunk_len=args.chunk_len,
            force_window_len=args.force_window_len,
        )
        if len(dataset) == 0:
            raise ValueError(f"dataset has no samples for action_mode={mode}")
        sample = dataset[min(args.sample_index, len(dataset) - 1)]
        samples[mode] = sample
        action_chunk = sample["action_chunk"].numpy()
        print(f"action_mode={mode} dataset_length={len(dataset)} {_stats(action_chunk)}")

    action_chunk = samples["action"]["action_chunk"].numpy()
    command_chunk = samples["joint_pos_command"]["action_chunk"].numpy()
    command_match = bool(np.allclose(action_chunk, command_chunk))
    print(f"action_equals_joint_pos_command={command_match}")

    state_index = int(samples["action"]["state_index"])
    with h5py.File(episode, "r") as handle:
        raw_action = np.asarray(handle["action"][state_index : state_index + args.chunk_len])
        current_qpos = np.asarray(handle["observations/joint_pos"][state_index])

    delta_chunk = samples["delta_joint_cmd"]["action_chunk"].numpy()
    expected_delta = raw_action - current_qpos[None, :]
    delta_match = bool(np.allclose(delta_chunk, expected_delta))
    print(f"delta_joint_cmd_matches_action_minus_current_qpos={delta_match}")
    print(f"state_index={state_index}")
    print(f"current_qpos={np.array2string(current_qpos, precision=6, separator=',')}")
    print(f"first_absolute_action={np.array2string(raw_action[0], precision=6, separator=',')}")
    print(f"first_delta_action={np.array2string(delta_chunk[0], precision=6, separator=',')}")

    for mode, sample in samples.items():
        for key in ("images", "qpos", "force_window", "action_chunk", "future_force_chunk"):
            values = sample[key].numpy()
            if not np.isfinite(values).all():
                raise ValueError(f"non-finite values found for action_mode={mode} key={key}")
    print("finite_tensor_check=passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
