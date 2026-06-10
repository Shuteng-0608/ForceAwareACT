#!/usr/bin/env python3
"""Inspect signals and camera frames around one evaluated HDF5 state."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from force_aware_act.data import get_episode_safe_lengths, nearest_index  # noqa: E402


def _read(handle: h5py.File, key: str, length: int) -> np.ndarray:
    if key not in handle:
        raise KeyError(f"missing required HDF5 dataset: {key}")
    return np.asarray(handle[key][:length])


def _parse_frame_offsets(value: str) -> list[int]:
    try:
        return [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError("--frame-offsets must be comma-separated integers") from error


def _load_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _save_plot(
    output_path: Path,
    timestamps: np.ndarray,
    values: np.ndarray,
    labels: Sequence[str],
    selected_time: float,
    ylabel: str,
    title: str,
    past_window_duration: Optional[float] = None,
) -> None:
    plt = _load_matplotlib()
    values_2d = values[:, None] if values.ndim == 1 else values
    for column, label in enumerate(labels):
        plt.plot(timestamps, values_2d[:, column], linewidth=1.2, label=label)
    plt.axvline(selected_time, color="black", linestyle="--", linewidth=1.3, label="selected state")
    if past_window_duration is not None:
        plt.axvspan(
            selected_time - past_window_duration,
            selected_time,
            color="gray",
            alpha=0.12,
            label="past force window",
        )
    plt.xlabel("episode time (s)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(alpha=0.25)
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"saved_plot={output_path}")


def _save_frames(
    handle: h5py.File,
    output_dir: Path,
    camera_names: Sequence[str],
    frame_offsets: Sequence[int],
    state_index: int,
    state_ts: np.ndarray,
    image_ts: np.ndarray,
    image_len: int,
) -> None:
    from PIL import Image

    for offset in frame_offsets:
        offset_state_index = int(np.clip(state_index + offset, 0, len(state_ts) - 1))
        image_index = min(nearest_index(image_ts, float(state_ts[offset_state_index])), image_len - 1)
        for camera_name in camera_names:
            key = f"observations/images/{camera_name}"
            image = np.asarray(handle[key][image_index])
            filename = (
                f"{camera_name}_state{state_index:04d}_offset{offset:+04d}.png"
            )
            output_path = output_dir / filename
            Image.fromarray(image.astype(np.uint8, copy=False)).save(output_path)
            print(
                f"saved_frame={output_path} offset_state_index={offset_state_index} "
                f"image_index={image_index}"
            )


def inspect(args: argparse.Namespace) -> int:
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(args.episode, "r") as handle:
        safe_lengths = get_episode_safe_lengths(
            handle,
            args.episode,
            camera_names=args.camera_names,
            tolerate_length_mismatch=True,
            max_length_mismatch=1,
        )
        state_ts = _read(handle, "timestamps/state_episode", safe_lengths.state_len).astype(
            np.float64
        )
        force_ts = _read(handle, "timestamps/force_episode", safe_lengths.force_len).astype(
            np.float64
        )
        image_ts = _read(handle, "timestamps/image_episode", safe_lengths.image_len).astype(
            np.float64
        )
        if not 0 <= args.state_index < safe_lengths.state_len:
            raise IndexError(
                f"state index {args.state_index} is outside [0, {safe_lengths.state_len - 1}]"
            )

        start = max(0, args.state_index - args.radius)
        end = min(safe_lengths.state_len, args.state_index + args.radius + 1)
        selected_time = float(state_ts[args.state_index])
        state_times = state_ts[start:end]
        force_mask = (force_ts >= state_times[0]) & (force_ts <= state_times[-1])
        force_indices = np.flatnonzero(force_mask)
        if len(force_indices) == 0:
            force_indices = np.array([nearest_index(force_ts, selected_time)])
        force_times = force_ts[force_indices]
        wrench = _read(handle, "observations/ft_wrench", safe_lengths.force_len)[force_indices]

        joint_pos = _read(handle, "observations/joint_pos", safe_lengths.state_len)[start:end]
        joint_vel = _read(handle, "observations/joint_vel", safe_lengths.state_len)[start:end]
        joint_torque = _read(handle, "observations/joint_torque", safe_lengths.state_len)[start:end]
        ee_position = _read(handle, "observations/ee_pose", safe_lengths.state_len)[start:end, :3]

        force_norm = np.linalg.norm(wrench[:, :3], axis=1)
        torque_norm = np.linalg.norm(wrench[:, 3:6], axis=1)
        force_delta = np.gradient(force_norm, force_times) if len(force_times) > 1 else np.zeros(1)
        selected_force_index = nearest_index(force_ts, selected_time)
        selected_window_force_index = int(np.argmin(np.abs(force_times - selected_time)))
        selected_force_norm = float(
            np.linalg.norm(
                _read(handle, "observations/ft_wrench", safe_lengths.force_len)[
                    selected_force_index, :3
                ]
            )
        )

        _save_plot(
            output_dir / "wrench_components.png",
            force_times,
            wrench,
            ("Fx", "Fy", "Fz", "Tx", "Ty", "Tz"),
            selected_time,
            "wrench",
            "Wrench Components",
            args.force_window_duration,
        )
        _save_plot(
            output_dir / "force_norm.png",
            force_times,
            np.column_stack((force_norm, torque_norm)),
            ("force norm", "torque norm"),
            selected_time,
            "norm",
            "Force and Torque Norms",
            args.force_window_duration,
        )
        _save_plot(
            output_dir / "force_delta.png",
            force_times,
            force_delta,
            ("d force norm / dt",),
            selected_time,
            "force norm derivative",
            "Force-Norm Delta",
            args.force_window_duration,
        )
        joint_labels = tuple(f"joint {index}" for index in range(7))
        _save_plot(
            output_dir / "joint_pos.png",
            state_times,
            joint_pos,
            joint_labels,
            selected_time,
            "position",
            "Joint Positions",
        )
        _save_plot(
            output_dir / "joint_vel.png",
            state_times,
            joint_vel,
            joint_labels,
            selected_time,
            "velocity",
            "Joint Velocities",
        )
        _save_plot(
            output_dir / "joint_torque.png",
            state_times,
            joint_torque,
            joint_labels,
            selected_time,
            "torque",
            "Joint Torques",
        )
        _save_plot(
            output_dir / "ee_position.png",
            state_times,
            ee_position,
            ("x", "y", "z"),
            selected_time,
            "position",
            "End-Effector Position",
        )

        if args.save_frames:
            _save_frames(
                handle,
                output_dir,
                args.camera_names,
                args.frame_offsets,
                args.state_index,
                state_ts,
                image_ts,
                safe_lengths.image_len,
            )

        print(f"episode_path={args.episode}")
        print(f"selected_state_index={args.state_index}")
        print(f"selected_timestamp={selected_time:.9g}")
        print(f"state_range=[{start}, {end})")
        print(
            f"force_range=[{int(force_indices[0])}, {int(force_indices[-1]) + 1}) "
            f"time=[{force_times[0]:.9g}, {force_times[-1]:.9g}]"
        )
        for camera_name in args.camera_names:
            print(f"image_shape_{camera_name}={tuple(handle[f'observations/images/{camera_name}'].shape)}")
        print(f"force_norm_at_selected_time={selected_force_norm:.9g}")
        print(f"max_force_norm_in_window={float(np.max(force_norm)):.9g}")
        max_force_norm_index = int(np.argmax(force_norm))
        print(f"max_force_norm_timestamp={float(force_times[max_force_norm_index]):.9g}")
        print(
            "max_force_norm_time_offset="
            f"{float(force_times[max_force_norm_index] - selected_time):.9g}"
        )
        print(f"force_delta_at_selected_time={float(force_delta[selected_window_force_index]):.9g}")
        print(f"max_force_delta_in_window={float(np.max(np.abs(force_delta))):.9g}")
        max_force_delta_index = int(np.argmax(np.abs(force_delta)))
        print(f"max_force_delta_timestamp={float(force_times[max_force_delta_index]):.9g}")
        print(
            "max_force_delta_time_offset="
            f"{float(force_times[max_force_delta_index] - selected_time):.9g}"
        )
        print(f"output_dir={output_dir}")
    return 0


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect signals around one HDF5 state index.")
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--state-index", type=int, required=True)
    parser.add_argument("--radius", type=int, default=40)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--force-window-duration", type=float, default=0.25)
    parser.add_argument("--camera-names", nargs="+", default=("ee_cam", "base_top_cam"))
    parser.add_argument("--save-frames", action="store_true")
    parser.add_argument("--frame-offsets", type=_parse_frame_offsets, default="-10,0,10")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    args.episode = args.episode.expanduser()
    args.output_dir = args.output_dir.expanduser()
    if isinstance(args.frame_offsets, str):
        args.frame_offsets = _parse_frame_offsets(args.frame_offsets)
    if not args.episode.is_file():
        print(f"error: episode does not exist: {args.episode}", file=sys.stderr)
        return 2
    if args.state_index < 0:
        print("error: --state-index must be non-negative", file=sys.stderr)
        return 2
    if args.radius < 0:
        print("error: --radius must be non-negative", file=sys.stderr)
        return 2
    if args.force_window_duration < 0:
        print("error: --force-window-duration must be non-negative", file=sys.stderr)
        return 2
    if not args.camera_names:
        print("error: --camera-names must include at least one camera", file=sys.stderr)
        return 2
    try:
        return inspect(args)
    except Exception as error:
        print(f"error: failed to inspect worst case: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
