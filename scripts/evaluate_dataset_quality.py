#!/usr/bin/env python3
"""Batch quality audit for MuJoCo peg-in-hole HDF5 demonstrations.

The audit is read-only.  It checks whether an episode is usable for training,
whether collection reported task success, and whether force/motion/image signals
look reasonable.  Results are written as one CSV row per episode plus a JSON
summary; thresholds are intentionally exposed as command-line arguments.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Optional, Sequence

import h5py
import numpy as np


REQUIRED = {
    "state": (
        "observations/ee_pose",
        "observations/joint_pos",
        "observations/joint_vel",
        "observations/joint_torque",
        "actions/joint_pos_command",
        "timestamps/state_episode",
    ),
    "force": ("observations/ft_wrench", "timestamps/force_episode"),
    "image": (
        "observations/images/ee_cam",
        "observations/images/base_top_cam",
        "timestamps/image_episode",
    ),
}
NUMERIC_KEYS = tuple(key for keys in REQUIRED.values() for key in keys if "images/" not in key)


def _json_value(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
        return value if math.isfinite(value) else None
    return value


def _add_issue(issues: list[str], text: str) -> None:
    if text not in issues:
        issues.append(text)


def _finite_in_chunks(dataset: h5py.Dataset, chunk: int = 8192) -> bool:
    for start in range(0, len(dataset), chunk):
        if not np.isfinite(np.asarray(dataset[start : start + chunk])).all():
            return False
    return True


def _sample_image_metrics(handle: h5py.File, key: str, samples: int) -> tuple[float, float, float]:
    dataset = handle[key]
    indices = np.unique(np.linspace(0, len(dataset) - 1, min(samples, len(dataset)), dtype=int))
    means, stds, changes = [], [], []
    previous = None
    for index in indices:
        # Subsample spatially: enough for blank/frozen-camera detection without
        # loading hundreds of full RGB frames.
        frame = np.asarray(dataset[int(index)])[::8, ::8].astype(np.float32)
        means.append(float(frame.mean()))
        stds.append(float(frame.std()))
        if previous is not None:
            changes.append(float(np.mean(np.abs(frame - previous))))
        previous = frame
    return float(np.mean(means)), float(np.mean(stds)), float(np.mean(changes or [0.0]))


def evaluate_episode(path: Path, args: argparse.Namespace) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    row: dict[str, Any] = {
        "episode": path.parent.name,
        "path": str(path.resolve()),
        "status": "",
        "n_state": None,
        "n_force": None,
        "n_image": None,
        "duration_s": None,
        "state_hz": None,
        "force_hz": None,
        "image_hz": None,
        "max_state_gap_s": None,
        "max_force_n": None,
        "p95_force_n": None,
        "max_torque_nm": None,
        "max_joint_speed_rad_s": None,
        "p95_joint_speed_rad_s": None,
        "max_command_step_rad": None,
        "max_tracking_error_rad": None,
        "ee_cam_mean": None,
        "ee_cam_std": None,
        "ee_cam_frame_change": None,
        "base_top_cam_mean": None,
        "base_top_cam_std": None,
        "base_top_cam_frame_change": None,
    }
    try:
        with h5py.File(path, "r") as handle:
            row["status"] = str(handle.attrs.get("status", ""))
            if not row["status"]:
                _add_issue(warnings, "missing collection status")
            elif row["status"] != args.success_status:
                _add_issue(errors, f"collection status is {row['status']!r}, not success")

            for keys in REQUIRED.values():
                for key in keys:
                    if key not in handle or not isinstance(handle[key], h5py.Dataset):
                        _add_issue(errors, f"missing dataset: {key}")
            if errors and any(issue.startswith("missing dataset") for issue in errors):
                raise KeyError("required dataset missing")

            group_lengths: dict[str, int] = {}
            for group, keys in REQUIRED.items():
                lengths = [len(handle[key]) for key in keys]
                group_lengths[group] = min(lengths)
                if min(lengths) == 0:
                    _add_issue(errors, f"{group} group is empty")
                if max(lengths) - min(lengths) > args.max_length_mismatch:
                    _add_issue(errors, f"{group} length mismatch: {lengths}")
                elif max(lengths) != min(lengths):
                    _add_issue(warnings, f"{group} length mismatch trimmed: {lengths}")
            row.update(n_state=group_lengths["state"], n_force=group_lengths["force"], n_image=group_lengths["image"])

            for key in NUMERIC_KEYS:
                if not _finite_in_chunks(handle[key]):
                    _add_issue(errors, f"non-finite values: {key}")

            timestamps: dict[str, np.ndarray] = {}
            for group in REQUIRED:
                key = f"timestamps/{group}_episode"
                values = np.asarray(handle[key][: group_lengths[group]], dtype=np.float64)
                timestamps[group] = values
                delta = np.diff(values)
                if len(delta) and np.any(delta <= 0):
                    _add_issue(errors, f"{group} timestamps are not strictly increasing")
                if len(delta):
                    row[f"{group}_hz"] = float(1.0 / np.median(delta))
            state_ts = timestamps["state"]
            row["duration_s"] = float(state_ts[-1] - state_ts[0]) if len(state_ts) > 1 else 0.0
            row["max_state_gap_s"] = float(np.max(np.diff(state_ts))) if len(state_ts) > 1 else 0.0
            if row["duration_s"] < args.min_duration:
                _add_issue(warnings, f"short episode: {row['duration_s']:.3f}s")
            if row["duration_s"] > args.max_duration:
                _add_issue(warnings, f"long episode: {row['duration_s']:.3f}s")
            nominal_dt = np.median(np.diff(state_ts)) if len(state_ts) > 1 else 0.0
            if nominal_dt and row["max_state_gap_s"] > args.max_gap_factor * nominal_dt:
                _add_issue(warnings, f"large state timestamp gap: {row['max_state_gap_s']:.4f}s")

            wrench = np.asarray(handle["observations/ft_wrench"][: group_lengths["force"]], dtype=np.float64)
            force_norm = np.linalg.norm(wrench[:, :3], axis=1)
            torque_norm = np.linalg.norm(wrench[:, 3:], axis=1)
            row["max_force_n"] = float(force_norm.max())
            row["p95_force_n"] = float(np.percentile(force_norm, 95))
            row["max_torque_nm"] = float(torque_norm.max())
            if row["max_force_n"] > args.max_force:
                _add_issue(warnings, f"peak force {row['max_force_n']:.2f}N exceeds {args.max_force:g}N")

            qpos = np.asarray(handle["observations/joint_pos"][: group_lengths["state"]], dtype=np.float64)
            command = np.asarray(handle["actions/joint_pos_command"][: group_lengths["state"]], dtype=np.float64)
            dt = np.diff(state_ts)
            speed = np.max(np.abs(np.diff(qpos, axis=0) / dt[:, None]), axis=1) if len(dt) else np.array([0.0])
            row["max_joint_speed_rad_s"] = float(speed.max())
            row["p95_joint_speed_rad_s"] = float(np.percentile(speed, 95))
            row["max_command_step_rad"] = float(np.max(np.abs(np.diff(command, axis=0)))) if len(command) > 1 else 0.0
            row["max_tracking_error_rad"] = float(np.max(np.abs(command - qpos)))
            if row["max_joint_speed_rad_s"] > args.max_joint_speed:
                _add_issue(warnings, f"joint speed {row['max_joint_speed_rad_s']:.2f}rad/s exceeds limit")
            if row["max_command_step_rad"] > args.max_command_step:
                _add_issue(warnings, f"command step {row['max_command_step_rad']:.3f}rad exceeds limit")

            for camera in ("ee_cam", "base_top_cam"):
                key = f"observations/images/{camera}"
                dataset = handle[key]
                if dataset.ndim != 4 or dataset.shape[-1] != 3 or dataset.dtype != np.uint8:
                    _add_issue(errors, f"invalid image shape/dtype: {key} {dataset.shape} {dataset.dtype}")
                    continue
                mean, std, change = _sample_image_metrics(handle, key, args.image_samples)
                row[f"{camera}_mean"] = mean
                row[f"{camera}_std"] = std
                row[f"{camera}_frame_change"] = change
                if std < args.min_image_std:
                    _add_issue(warnings, f"{camera} appears blank (mean std={std:.2f})")
                if change < args.min_frame_change:
                    _add_issue(warnings, f"{camera} appears frozen (frame change={change:.3f})")
    except Exception as error:
        if not isinstance(error, KeyError) or str(error) != "'required dataset missing'":
            _add_issue(errors, f"cannot read episode: {error}")

    score = max(0, 100 - 40 * len(errors) - 5 * len(warnings))
    if errors:
        quality = "reject"
    elif score >= args.good_score:
        quality = "good"
    else:
        quality = "review"
    row.update(quality=quality, quality_score=score, errors="; ".join(errors), warnings="; ".join(warnings))
    return {key: _json_value(value) for key, value in row.items()}


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", type=Path, nargs="?", default=Path("mujoco_data/peg_hole_fixed"))
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--success-status", default="auto_stop_task_success")
    parser.add_argument("--max-length-mismatch", type=int, default=1)
    parser.add_argument("--min-duration", type=float, default=3.0)
    parser.add_argument("--max-duration", type=float, default=30.0)
    parser.add_argument("--max-gap-factor", type=float, default=3.0)
    parser.add_argument("--max-force", type=float, default=60.0, help="Peak translational force warning threshold (N).")
    parser.add_argument("--max-joint-speed", type=float, default=1.0)
    parser.add_argument("--max-command-step", type=float, default=0.05)
    parser.add_argument("--image-samples", type=int, default=8)
    parser.add_argument("--min-image-std", type=float, default=2.0)
    parser.add_argument("--min-frame-change", type=float, default=0.1)
    parser.add_argument("--good-score", type=int, default=90)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    root = args.data_dir.expanduser().resolve()
    paths = sorted(root.glob("*/episode.hdf5")) if root.is_dir() else ([root] if root.is_file() else [])
    if not paths:
        print(f"error: no episode.hdf5 files found under {root}", file=sys.stderr)
        return 2
    if args.image_samples <= 0 or args.max_length_mismatch < 0:
        print("error: --image-samples must be positive and --max-length-mismatch non-negative", file=sys.stderr)
        return 2

    rows = []
    for index, path in enumerate(paths, 1):
        row = evaluate_episode(path, args)
        rows.append(row)
        print(f"[{index:03d}/{len(paths):03d}] {row['episode']}: {row['quality']} ({row['quality_score']})")

    output_csv = (args.output_csv or root / "quality_report.csv").expanduser().resolve()
    output_json = (args.output_json or root / "quality_summary.json").expanduser().resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    counts = Counter(row["quality"] for row in rows)
    summary = {
        "data_dir": str(root),
        "episode_count": len(rows),
        "quality_counts": dict(sorted(counts.items())),
        "mean_quality_score": float(np.mean([row["quality_score"] for row in rows])),
        "report_csv": str(output_csv),
        "thresholds": {key: value for key, value in vars(args).items() if key not in {"data_dir", "output_csv", "output_json"}},
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    print(f"\nquality_counts={dict(counts)}")
    print(f"csv={output_csv}\njson={output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
