#!/usr/bin/env python3
"""Inspect a collection of ForceAwareACT HDF5 episodes without modifying them."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from force_aware_act.data import ContactForceHDF5Dataset, get_episode_safe_lengths  # noqa: E402
from force_aware_act.utils import resolve_episode_paths  # noqa: E402


REQUIRED_DATASETS = (
    "observations/ee_pose",
    "observations/joint_pos",
    "observations/joint_vel",
    "observations/joint_torque",
    "observations/ft_wrench",
    "observations/images/ee_cam",
    "observations/images/base_top_cam",
    "timestamps/state_episode",
    "timestamps/force_episode",
    "timestamps/image_episode",
)


def _vector_text(values: Optional[np.ndarray]) -> str:
    if values is None:
        return "unavailable"
    return np.array2string(
        np.asarray(values),
        precision=6,
        separator=",",
        suppress_small=False,
        max_line_width=1000,
    )


def _dataset_shape(handle: h5py.File, key: str, problems: list[str]) -> Optional[tuple[int, ...]]:
    if key not in handle:
        problems.append(f"missing {key}")
        return None
    item = handle[key]
    if not isinstance(item, h5py.Dataset):
        problems.append(f"{key} is not a dataset")
        return None
    return tuple(item.shape)


def _check_shape(
    shape: Optional[tuple[int, ...]],
    expected_tail: tuple[int, ...],
    key: str,
    problems: list[str],
) -> None:
    if shape is not None and (len(shape) != len(expected_tail) + 1 or shape[1:] != expected_tail):
        problems.append(f"{key} has shape {shape}, expected [N, {', '.join(map(str, expected_tail))}]")


def _check_timestamp(
    handle: h5py.File,
    key: str,
    problems: list[str],
) -> Optional[np.ndarray]:
    shape = _dataset_shape(handle, key, problems)
    if shape is None:
        return None
    if len(shape) != 1:
        problems.append(f"{key} has shape {shape}, expected [N]")
        return None
    values = np.asarray(handle[key], dtype=np.float64)
    if not np.isfinite(values).all():
        problems.append(f"{key} contains non-finite values")
    if len(values) > 1 and np.any(np.diff(values) < 0):
        problems.append(f"{key} is not monotonically nondecreasing")
    return values


def _duration(state_ts: Optional[np.ndarray]) -> Optional[float]:
    if state_ts is None or len(state_ts) == 0 or not np.isfinite(state_ts).all():
        return None
    return float(state_ts[-1] - state_ts[0])


def inspect_episode(
    episode_index: int,
    path: Path,
    chunk_len: int,
    force_window_len: int,
    force_window_duration: float,
    tolerate_length_mismatch: bool,
    max_length_mismatch: int,
) -> dict[str, Any]:
    problems: list[str] = []
    result: dict[str, Any] = {
        "episode_index": episode_index,
        "path": str(path),
        "n_state": None,
        "n_force": None,
        "n_image": None,
        "duration": None,
        "image_shape": None,
        "force_mean": None,
        "force_std": None,
        "force_min": None,
        "force_max": None,
        "dataset_valid_length": None,
        "episode_metadata_present": False,
        "valid_strict": False,
        "valid_tolerant": False,
        "trim_state": None,
        "trim_image": None,
        "trim_force": None,
        "mismatch_group": "",
    }

    if not path.exists():
        problems.append("file does not exist")
    elif not path.is_file():
        problems.append("path is not a file")
    else:
        try:
            with h5py.File(path, "r") as handle:
                shapes = {key: _dataset_shape(handle, key, problems) for key in REQUIRED_DATASETS}

                for key in (
                    "observations/ee_pose",
                    "observations/joint_pos",
                    "observations/joint_vel",
                    "observations/joint_torque",
                ):
                    _check_shape(shapes[key], (7,), key, problems)
                _check_shape(shapes["observations/ft_wrench"], (6,), "observations/ft_wrench", problems)

                for key in (
                    "observations/images/ee_cam",
                    "observations/images/base_top_cam",
                ):
                    shape = shapes[key]
                    if shape is not None and (len(shape) != 4 or shape[-1] != 3):
                        problems.append(f"{key} has shape {shape}, expected [N, H, W, 3]")

                state_shape = shapes["observations/joint_pos"]
                force_shape = shapes["observations/ft_wrench"]
                image_shape = shapes["observations/images/ee_cam"]
                result["n_state"] = state_shape[0] if state_shape else None
                result["n_force"] = force_shape[0] if force_shape else None
                result["n_image"] = image_shape[0] if image_shape else None
                result["image_shape"] = image_shape[1:] if image_shape and len(image_shape) == 4 else None
                group_keys = {
                    "state": (
                        "observations/ee_pose",
                        "observations/joint_pos",
                        "observations/joint_vel",
                        "observations/joint_torque",
                        "timestamps/state_episode",
                    ),
                    "image": (
                        "observations/images/ee_cam",
                        "observations/images/base_top_cam",
                        "timestamps/image_episode",
                    ),
                    "force": (
                        "observations/ft_wrench",
                        "timestamps/force_episode",
                    ),
                }
                mismatch_groups: list[str] = []
                for group_name, keys in group_keys.items():
                    group_shapes = [shapes[key] for key in keys]
                    if all(shape is not None for shape in group_shapes):
                        lengths = [shape[0] for shape in group_shapes if shape is not None]
                        difference = max(lengths) - min(lengths)
                        result[f"trim_{group_name}"] = difference
                        if difference:
                            mismatch_groups.append(group_name)
                result["mismatch_group"] = ",".join(mismatch_groups)

                state_ts = _check_timestamp(handle, "timestamps/state_episode", problems)
                _check_timestamp(handle, "timestamps/force_episode", problems)
                _check_timestamp(handle, "timestamps/image_episode", problems)
                result["episode_metadata_present"] = "episode_metadata" in handle

                if force_shape is not None and len(force_shape) == 2 and force_shape[1] == 6:
                    force = np.asarray(handle["observations/ft_wrench"], dtype=np.float64)
                    if force.size == 0:
                        problems.append("observations/ft_wrench is empty")
                    elif not np.isfinite(force).all():
                        problems.append("observations/ft_wrench contains non-finite values")
                    else:
                        result["force_mean"] = force.mean(axis=0)
                        result["force_std"] = force.std(axis=0)
                        result["force_min"] = force.min(axis=0)
                        result["force_max"] = force.max(axis=0)

                if not problems:
                    try:
                        strict_lengths = get_episode_safe_lengths(
                            handle,
                            path,
                            tolerate_length_mismatch=False,
                            max_length_mismatch=max_length_mismatch,
                        )
                        result["valid_strict"] = True
                    except ValueError:
                        strict_lengths = None

                    try:
                        safe_lengths = get_episode_safe_lengths(
                            handle,
                            path,
                            tolerate_length_mismatch=True,
                            max_length_mismatch=max_length_mismatch,
                        )
                        result["valid_tolerant"] = True
                        result["trim_state"] = safe_lengths.trim_state
                        result["trim_image"] = safe_lengths.trim_image
                        result["trim_force"] = safe_lengths.trim_force
                        result["mismatch_group"] = ",".join(safe_lengths.mismatch_groups)
                        result["duration"] = _duration(state_ts[: safe_lengths.state_len])
                        force = np.asarray(
                            handle["observations/ft_wrench"][: safe_lengths.force_len],
                            dtype=np.float64,
                        )
                        result["force_mean"] = force.mean(axis=0)
                        result["force_std"] = force.std(axis=0)
                        result["force_min"] = force.min(axis=0)
                        result["force_max"] = force.max(axis=0)
                    except ValueError as error:
                        problems.append(str(error))
                        safe_lengths = strict_lengths
        except Exception as error:
            problems.append(f"failed to read HDF5: {error}")

    result["required_fields_ok"] = (
        result["valid_tolerant"] if tolerate_length_mismatch else result["valid_strict"]
    )
    result["problems"] = problems
    if path.is_file():
        try:
            result["dataset_valid_length"] = len(
                ContactForceHDF5Dataset(
                    path,
                    chunk_len=chunk_len,
                    force_window_len=force_window_len,
                    force_window_duration=force_window_duration,
                    tolerate_length_mismatch=tolerate_length_mismatch,
                    max_length_mismatch=max_length_mismatch,
                )
            )
        except Exception as error:
            result["required_fields_ok"] = False
            if tolerate_length_mismatch:
                result["valid_tolerant"] = False
            else:
                result["valid_strict"] = False
            result["problems"].append(f"dataset initialization failed: {error}")
    return result


def print_episode(result: dict[str, Any]) -> None:
    print(f"\nepisode_index={result['episode_index']}")
    print(f"path={result['path']}")
    print(f"N_state={result['n_state']}")
    print(f"N_force={result['n_force']}")
    print(f"N_image={result['n_image']}")
    print(f"duration={result['duration']}")
    print(f"image_shape={result['image_shape']}")
    print(f"force_mean={_vector_text(result['force_mean'])}")
    print(f"force_std={_vector_text(result['force_std'])}")
    print(f"force_min={_vector_text(result['force_min'])}")
    print(f"force_max={_vector_text(result['force_max'])}")
    print(f"dataset_valid_length={result['dataset_valid_length']}")
    print(f"episode_metadata_present={result['episode_metadata_present']}")
    print(f"valid_strict={result['valid_strict']}")
    print(f"valid_tolerant={result['valid_tolerant']}")
    print(f"trim_state={result['trim_state']}")
    print(f"trim_image={result['trim_image']}")
    print(f"trim_force={result['trim_force']}")
    print(f"mismatch_group={result['mismatch_group']}")
    print(f"required_fields_ok={result['required_fields_ok']}")
    for problem in result["problems"]:
        print(f"problem={problem}")


def _summary_triplet(values: Sequence[float]) -> str:
    if not values:
        return "unavailable"
    array = np.asarray(values, dtype=np.float64)
    return f"min={array.min():.6g} mean={array.mean():.6g} max={array.max():.6g}"


def print_aggregate(results: Sequence[dict[str, Any]]) -> None:
    strict_valid = [result for result in results if result["valid_strict"]]
    tolerant_valid = [result for result in results if result["valid_tolerant"]]
    requiring_trim = [result for result in tolerant_valid if result["mismatch_group"]]
    durations = [result["duration"] for result in results if result["duration"] is not None]
    valid_lengths = [
        result["dataset_valid_length"]
        for result in results
        if result["dataset_valid_length"] is not None
    ]
    print("\nAggregate Summary")
    print("-----------------")
    print(f"episode_count={len(results)}")
    print(f"strict_valid_count={len(strict_valid)}")
    print(f"tolerant_valid_count={len(tolerant_valid)}")
    print(f"invalid_count={len(results) - len(tolerant_valid)}")
    print(f"episodes_requiring_trimming={len(requiring_trim)}")
    print(f"total_state_frames={sum(result['n_state'] or 0 for result in results)}")
    print(f"total_force_frames={sum(result['n_force'] or 0 for result in results)}")
    print(f"duration_seconds={_summary_triplet(durations)}")
    print(f"dataset_valid_length={_summary_triplet(valid_lengths)}")


def write_csv(path: Path, results: Sequence[dict[str, Any]]) -> None:
    fieldnames = (
        "episode_index",
        "path",
        "n_state",
        "n_force",
        "n_image",
        "duration",
        "image_shape",
        "force_mean",
        "force_std",
        "force_min",
        "force_max",
        "dataset_valid_length",
        "episode_metadata_present",
        "valid_strict",
        "valid_tolerant",
        "trim_state",
        "trim_image",
        "trim_force",
        "mismatch_group",
        "required_fields_ok",
        "problems",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            row = dict(result)
            row["image_shape"] = str(result["image_shape"])
            for key in ("force_mean", "force_std", "force_min", "force_max"):
                row[key] = _vector_text(result[key])
            row["problems"] = "; ".join(result["problems"])
            writer.writerow(row)
    print(f"saved_csv={path}")


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect a collection of HDF5 episodes.")
    parser.add_argument("episode_paths", type=Path, nargs="*", help="One or more HDF5 episodes.")
    parser.add_argument("--episode-list", type=Path, default=None)
    parser.add_argument("--chunk-len", type=int, default=10)
    parser.add_argument("--force-window-len", type=int, default=20)
    parser.add_argument("--force-window-duration", type=float, default=0.25)
    parser.add_argument("--tolerate-length-mismatch", action="store_true")
    parser.add_argument("--max-length-mismatch", type=int, default=1)
    parser.add_argument("--output-csv", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.chunk_len <= 0:
        print("error: --chunk-len must be positive", file=sys.stderr)
        return 2
    if args.force_window_len <= 0:
        print("error: --force-window-len must be positive", file=sys.stderr)
        return 2
    if args.force_window_duration < 0:
        print("error: --force-window-duration must be non-negative", file=sys.stderr)
        return 2
    if args.max_length_mismatch < 0:
        print("error: --max-length-mismatch must be non-negative", file=sys.stderr)
        return 2

    try:
        episode_paths = resolve_episode_paths(
            args.episode_paths, args.episode_list, project_root=REPO_ROOT
        )
    except Exception as error:
        print(f"error: failed to resolve episode paths: {error}", file=sys.stderr)
        return 2
    if not episode_paths:
        print("error: provide episode paths or --episode-list", file=sys.stderr)
        return 2

    print(f"chunk_len={args.chunk_len}")
    print(f"force_window_len={args.force_window_len}")
    print(f"force_window_duration={args.force_window_duration}")
    print(f"tolerate_length_mismatch={args.tolerate_length_mismatch}")
    print(f"max_length_mismatch={args.max_length_mismatch}")
    results = [
        inspect_episode(
            episode_index=index,
            path=path,
            chunk_len=args.chunk_len,
            force_window_len=args.force_window_len,
            force_window_duration=args.force_window_duration,
            tolerate_length_mismatch=args.tolerate_length_mismatch,
            max_length_mismatch=args.max_length_mismatch,
        )
        for index, path in enumerate(episode_paths)
    ]
    for result in results:
        print_episode(result)
    print_aggregate(results)

    if args.output_csv is not None:
        output_csv = args.output_csv.expanduser()
        if not output_csv.is_absolute():
            output_csv = REPO_ROOT / output_csv
        write_csv(output_csv.resolve(), results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
