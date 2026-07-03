#!/usr/bin/env python3
"""Run a deterministic grid of hole-position perturbation rollouts."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_mujoco_policy_rollout import DEFAULT_HOLE_BODY_NAME, DEFAULT_HOLE_SITE_NAME  # noqa: E402
from summarize_rollouts import collect_rollouts, write_summary_csv  # noqa: E402

POSITION_SUMMARY_COLUMNS = [
    "point_index",
    "sampling_mode",
    "base_seed",
    "hole_offset_x",
    "hole_offset_y",
    "hole_offset_z",
    "radial_offset",
    "quadrant",
    "success",
    "safe_success",
    "success_step",
    "success_time",
    "stop_reason",
    "final_dist",
    "final_lateral",
    "final_axial",
    "max_force",
    "mean_force",
    "force_gt_20_steps",
    "force_gt_40_steps",
    "output_dir",
    "summary_json",
    "rollout_log_csv",
]


def parse_offset_list(value: str) -> list[float]:
    offsets = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        offsets.append(float(item))
    if not offsets:
        raise ValueError("offset list must contain at least one value")
    return offsets


def signed_mm_token(value_m: float) -> str:
    value_mm = int(round(value_m * 1000.0))
    prefix = "m" if value_mm < 0 else "p"
    return f"{prefix}{abs(value_mm):03d}mm"


def signed_precise_mm_token(value_m: float) -> str:
    value_um = int(round(value_m * 1_000_000.0))
    prefix = "m" if value_um < 0 else "p"
    return f"{prefix}{abs(value_um):06d}mm"


def run_name(x_offset: float, z_offset: float, repeat_index: int) -> str:
    return (
        f"x_{signed_mm_token(x_offset)}_"
        f"z_{signed_mm_token(z_offset)}_"
        f"repeat_{repeat_index:03d}"
    )


def point_run_name(point_index: int, x_offset: float, z_offset: float, repeat_index: int) -> str:
    return (
        f"point_{point_index:03d}_"
        f"x_{signed_precise_mm_token(x_offset)}_"
        f"z_{signed_precise_mm_token(z_offset)}_"
        f"repeat_{repeat_index:03d}"
    )


def latin_hypercube_points(
    num_points: int,
    x_min: float,
    x_max: float,
    z_min: float,
    z_max: float,
    seed: int,
) -> list[tuple[float, float]]:
    rng = np_random(seed)
    strata_x = (np.arange(num_points, dtype=np.float64) + rng.random(num_points)) / num_points
    strata_z = (np.arange(num_points, dtype=np.float64) + rng.random(num_points)) / num_points
    rng.shuffle(strata_x)
    rng.shuffle(strata_z)
    x_values = x_min + strata_x * (x_max - x_min)
    z_values = z_min + strata_z * (z_max - z_min)
    return [(float(x), float(z)) for x, z in zip(x_values, z_values)]


def np_random(seed: int):
    import numpy as np

    return np.random.default_rng(seed)


def random_points(
    num_points: int,
    x_min: float,
    x_max: float,
    z_min: float,
    z_max: float,
    seed: int,
) -> list[tuple[float, float]]:
    rng = np_random(seed)
    x_values = rng.uniform(x_min, x_max, size=num_points)
    z_values = rng.uniform(z_min, z_max, size=num_points)
    return [(float(x), float(z)) for x, z in zip(x_values, z_values)]


def validate_sampling_bounds(args: argparse.Namespace) -> None:
    values = [args.x_min, args.x_max, args.z_min, args.z_max]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("sampling bounds must be finite")
    if args.x_min > args.x_max:
        raise ValueError("--x-min must be <= --x-max")
    if args.z_min > args.z_max:
        raise ValueError("--z-min must be <= --z-max")
    if args.num_points <= 0:
        raise ValueError("--num-points must be positive")


def generate_task_points(args: argparse.Namespace) -> list[dict[str, Any]]:
    task_points = []
    if args.sampling_mode == "grid":
        x_offsets = parse_offset_list(args.x_offsets)
        z_offsets = parse_offset_list(args.z_offsets)
        for repeat in range(1, args.repeats + 1):
            for x_offset in x_offsets:
                for z_offset in z_offsets:
                    point_index = len(task_points) + 1
                    task_points.append(
                        {
                            "point_index": point_index,
                            "hole_offset_x": float(x_offset),
                            "hole_offset_y": float(args.y_offset),
                            "hole_offset_z": float(z_offset),
                            "repeat_index": repeat,
                            "run_name": run_name(x_offset, z_offset, repeat),
                        }
                    )
        return task_points

    validate_sampling_bounds(args)
    if args.sampling_mode == "random":
        points = random_points(args.num_points, args.x_min, args.x_max, args.z_min, args.z_max, args.base_seed)
    elif args.sampling_mode == "latin_hypercube":
        points = latin_hypercube_points(args.num_points, args.x_min, args.x_max, args.z_min, args.z_max, args.base_seed)
    else:
        raise ValueError(f"unknown sampling mode: {args.sampling_mode}")
    for index, (x_offset, z_offset) in enumerate(points, start=1):
        if not (args.x_min <= x_offset <= args.x_max and args.z_min <= z_offset <= args.z_max):
            raise ValueError(f"generated point outside bounds: x={x_offset}, z={z_offset}")
        task_points.append(
            {
                "point_index": index,
                "hole_offset_x": float(x_offset),
                "hole_offset_y": float(args.y_offset),
                "hole_offset_z": float(z_offset),
                "repeat_index": 1,
                "run_name": point_run_name(index, x_offset, z_offset, 1),
            }
        )
    return task_points


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _read_summary(path: Path) -> Optional[dict[str, Any]]:
    if not path.is_file():
        return None
    try:
        with path.open() as summary_file:
            return json.load(summary_file)
    except Exception:
        return None


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as manifest_file:
        json.dump(_json_safe(manifest), manifest_file, indent=2, sort_keys=True)
        manifest_file.write("\n")
    print(f"grid_manifest_json={path}")


def write_task_points_csv(path: Path, task_points: list[dict[str, Any]], args: argparse.Namespace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "point_index",
                "hole_offset_x",
                "hole_offset_y",
                "hole_offset_z",
                "sampling_mode",
                "base_seed",
                "repeat_index",
                "run_name",
                "output_dir",
            ],
        )
        writer.writeheader()
        for point in task_points:
            writer.writerow(
                {
                    "point_index": point["point_index"],
                    "hole_offset_x": point["hole_offset_x"],
                    "hole_offset_y": point["hole_offset_y"],
                    "hole_offset_z": point["hole_offset_z"],
                    "sampling_mode": args.sampling_mode,
                    "base_seed": args.base_seed,
                    "repeat_index": point["repeat_index"],
                    "run_name": point["run_name"],
                    "output_dir": str(args.output_root / point["run_name"]),
                }
            )
    print(f"task_points_csv={path}")


def quadrant(x_offset: float, z_offset: float, eps: float = 1.0e-12) -> str:
    if abs(x_offset) <= eps and abs(z_offset) <= eps:
        return "center"
    if abs(x_offset) <= eps or abs(z_offset) <= eps:
        return "axis"
    if x_offset > 0 and z_offset > 0:
        return "+x+z"
    if x_offset > 0 and z_offset < 0:
        return "+x-z"
    if x_offset < 0 and z_offset > 0:
        return "-x+z"
    return "-x-z"


def wilson_ci(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0:
        return float("nan"), float("nan")
    p_hat = successes / total
    denom = 1.0 + z * z / total
    center = (p_hat + z * z / (2.0 * total)) / denom
    half_width = (
        z
        * math.sqrt((p_hat * (1.0 - p_hat) + z * z / (4.0 * total)) / total)
        / denom
    )
    return max(0.0, center - half_width), min(1.0, center + half_width)


def _summary_row_from_manifest_run(run: dict[str, Any], success_force_threshold: float) -> dict[str, Any]:
    output_dir = Path(run["output_dir"])
    summary_path = output_dir / "summary.json"
    summary = _read_summary(summary_path) or {}
    x_offset = float(run["x_offset"])
    y_offset = float(run["y_offset"])
    z_offset = float(run["z_offset"])
    success = bool(summary.get("success", False)) if summary else False
    max_force = summary.get("max_force_norm", "")
    safe_success = bool(success and max_force != "" and float(max_force) < success_force_threshold)
    return {
        "point_index": run.get("point_index"),
        "sampling_mode": run.get("sampling_mode"),
        "base_seed": run.get("base_seed"),
        "hole_offset_x": x_offset,
        "hole_offset_y": y_offset,
        "hole_offset_z": z_offset,
        "radial_offset": math.sqrt(x_offset * x_offset + z_offset * z_offset),
        "quadrant": quadrant(x_offset, z_offset),
        "success": success,
        "safe_success": safe_success,
        "success_step": summary.get("success_step", ""),
        "success_time": summary.get("success_time", ""),
        "stop_reason": summary.get("stop_reason", run.get("status", "")),
        "final_dist": summary.get("final_peg_to_hole_dist", ""),
        "final_lateral": summary.get("final_peg_to_hole_lateral_error", ""),
        "final_axial": summary.get("final_peg_to_hole_axial_error", ""),
        "max_force": max_force,
        "mean_force": summary.get("mean_force_norm", ""),
        "force_gt_20_steps": summary.get("force_gt_20_steps", ""),
        "force_gt_40_steps": summary.get("force_gt_40_steps", ""),
        "output_dir": output_dir,
        "summary_json": summary_path if summary else "",
        "rollout_log_csv": summary.get("rollout_log_csv", output_dir / "rollout_log.csv" if summary else ""),
    }


def write_position_summary_csv(path: Path, manifest: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    success_force_threshold = float(manifest["success_thresholds"]["success_force_threshold"])
    for run in manifest["runs"]:
        if run.get("status") in {"success", "task_failed", "skipped_existing"}:
            rows.append(_summary_row_from_manifest_run(run, success_force_threshold))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=POSITION_SUMMARY_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in POSITION_SUMMARY_COLUMNS})
    print(f"grid_summary_csv={path}")
    return rows


def _bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _mean(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return sum(finite) / len(finite) if finite else float("nan")


def _median(values: list[float]) -> float:
    finite = sorted(value for value in values if math.isfinite(value))
    if not finite:
        return float("nan")
    mid = len(finite) // 2
    if len(finite) % 2:
        return finite[mid]
    return 0.5 * (finite[mid - 1] + finite[mid])


def _group_rate(rows: list[dict[str, Any]], key_fn) -> dict[str, float]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(key_fn(row)), []).append(row)
    return {
        key: sum(_bool_value(row["success"]) for row in group) / len(group)
        for key, group in sorted(groups.items())
        if group
    }


def radial_bin_label(radial_offset: float) -> str:
    radial_mm = radial_offset * 1000.0
    bins = [
        (0.0, 0.5, "[0,0.5]"),
        (0.5, 1.0, "(0.5,1.0]"),
        (1.0, 1.5, "(1.0,1.5]"),
        (1.5, 2.0, "(1.5,2.0]"),
        (2.0, math.sqrt(8.0), "(2.0,sqrt8]"),
    ]
    for lower, upper, label in bins:
        if radial_mm <= upper and (radial_mm > lower or lower == 0.0):
            return label
    return f">{math.sqrt(8.0):.3g}"


def write_random_position_summary(path: Path, manifest: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    runs = manifest["runs"]
    process_errors = [run for run in runs if run.get("status") == "process_error"]
    completed = rows
    successes = [row for row in completed if _bool_value(row["success"])]
    safe_successes = [row for row in completed if _bool_value(row["safe_success"])]
    lower, upper = wilson_ci(len(successes), len(completed))
    success_times = [float(row["success_time"]) for row in successes if row.get("success_time") not in ("", None)]
    max_forces = [float(row["max_force"]) for row in completed if row.get("max_force") not in ("", None)]
    summary = {
        "sampling_mode": manifest["sampling_mode"],
        "requested_bounds": {
            "x_min": manifest["x_min"],
            "x_max": manifest["x_max"],
            "z_min": manifest["z_min"],
            "z_max": manifest["z_max"],
        },
        "base_seed": manifest["base_seed"],
        "planned_runs": manifest["total_planned_runs"],
        "completed_runs": len(completed),
        "process_error_runs": len(process_errors),
        "completion_rate": len(completed) / manifest["total_planned_runs"] if manifest["total_planned_runs"] else 0.0,
        "successes": len(successes),
        "safe_successes": len(safe_successes),
        "success_rate": len(successes) / len(completed) if completed else 0.0,
        "safe_success_rate": len(safe_successes) / len(completed) if completed else 0.0,
        "success_rate_ci95_lower": lower,
        "success_rate_ci95_upper": upper,
        "mean_success_time": _mean(success_times),
        "median_success_time": _median(success_times),
        "mean_max_force": _mean(max_forces),
        "max_force": max(max_forces) if max_forces else float("nan"),
        "success_rate_by_z_sign": _group_rate(completed, lambda row: "+z" if float(row["hole_offset_z"]) > 0 else ("-z" if float(row["hole_offset_z"]) < 0 else "z0")),
        "success_rate_by_x_sign": _group_rate(completed, lambda row: "+x" if float(row["hole_offset_x"]) > 0 else ("-x" if float(row["hole_offset_x"]) < 0 else "x0")),
        "success_rate_by_quadrant": _group_rate(completed, lambda row: row["quadrant"]),
        "success_rate_by_radial_bin": _group_rate(completed, lambda row: radial_bin_label(float(row["radial_offset"]))),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as json_file:
        json.dump(_json_safe(summary), json_file, indent=2, sort_keys=True)
        json_file.write("\n")
    print(f"random_position_summary_json={path}")
    return summary


def _build_rollout_command(args: argparse.Namespace, output_dir: Path, x_offset: float, y_offset: float, z_offset: float, seed: int) -> list[str]:
    command = [
        args.python_executable,
        str(SCRIPT_DIR / "run_mujoco_policy_rollout.py"),
        "--checkpoint",
        str(args.checkpoint),
        "--normalization-stats",
        str(args.normalization_stats),
        "--model-xml",
        str(args.model_xml),
        "--contact-latent-mode",
        args.contact_latent_mode,
        "--action-mode",
        args.action_mode,
        "--action-select-mode",
        args.action_select_mode,
        "--chunk-len",
        str(args.chunk_len),
        "--force-window-len",
        str(args.force_window_len),
        "--force-window-duration",
        str(args.force_window_duration),
        "--policy-rate-hz",
        str(args.policy_rate_hz),
        "--max-rollout-steps",
        str(args.max_rollout_steps),
        "--max-delta-q",
        str(args.max_delta_q),
        "--force-stop-threshold",
        str(args.force_stop_threshold),
        "--hole-axis-world",
        *(str(value) for value in args.hole_axis_world),
        "--hole-site-name",
        args.hole_site_name,
        "--hole-body-name",
        args.hole_body_name,
        "--hole-offset-frame",
        args.hole_offset_frame,
        # "--hole-offset-x",
        # str(x_offset),
        # "--hole-offset-y",
        # str(y_offset),
        # "--hole-offset-z",
        # str(z_offset),
        f"--hole-offset-x={float(x_offset)!r}",
        f"--hole-offset-y={float(y_offset)!r}",
        f"--hole-offset-z={float(z_offset)!r}",
        "--success-distance-threshold",
        str(args.success_distance_threshold),
        "--success-lateral-threshold",
        str(args.success_lateral_threshold),
        "--success-force-threshold",
        str(args.success_force_threshold),
        "--success-hold-steps",
        str(args.success_hold_steps),
        "--output-dir",
        str(output_dir),
        "--seed",
        str(seed),
        "--execute-actions",
    ]
    if args.save_videos:
        command.append("--save-videos")
    return command


def _summarize(output_root: Path) -> Path:
    summary_csv = output_root / "grid_summary.csv"
    rows = collect_rollouts(output_root, "x_*_z_*_repeat_*")
    write_summary_csv(summary_csv, rows)
    print(f"grid_summary_csv={summary_csv}")
    return summary_csv


def _plot_results(summary_csv: Path, output_root: Path, formats: str) -> None:
    try:
        from plot_hole_grid_results import main as plot_main

        plot_main(
            [
                "--summary-csv",
                str(summary_csv),
                "--output-dir",
                str(output_root / "plots"),
                "--formats",
                formats,
            ]
        )
    except Exception as error:
        print(f"warning: grid plotting failed: {error}")


def run_grid(args: argparse.Namespace) -> dict[str, Any]:
    task_points = generate_task_points(args)
    args.output_root.mkdir(parents=True, exist_ok=True)
    write_task_points_csv(args.output_root / "task_points.csv", task_points, args)
    manifest_path = args.output_root / "grid_manifest.json"
    manifest: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "sampling_mode": args.sampling_mode,
        "num_points": args.num_points,
        "x_min": args.x_min,
        "x_max": args.x_max,
        "z_min": args.z_min,
        "z_max": args.z_max,
        "base_seed": args.base_seed,
        "x_offsets": parse_offset_list(args.x_offsets),
        "y_offset": args.y_offset,
        "z_offsets": parse_offset_list(args.z_offsets),
        "repeats": args.repeats,
        "total_planned_runs": len(task_points),
        "task_points": [
            {
                **point,
                "output_dir": args.output_root / point["run_name"],
                "sampling_mode": args.sampling_mode,
                "base_seed": args.base_seed,
            }
            for point in task_points
        ],
        "policy_config": {
            "checkpoint": args.checkpoint,
            "normalization_stats": args.normalization_stats,
            "model_xml": args.model_xml,
            "contact_latent_mode": args.contact_latent_mode,
            "action_mode": args.action_mode,
            "action_select_mode": args.action_select_mode,
            "chunk_len": args.chunk_len,
            "force_window_len": args.force_window_len,
            "force_window_duration": args.force_window_duration,
            "policy_rate_hz": args.policy_rate_hz,
            "max_rollout_steps": args.max_rollout_steps,
            "max_delta_q": args.max_delta_q,
            "force_stop_threshold": args.force_stop_threshold,
            "hole_site_name": args.hole_site_name,
            "hole_body_name": args.hole_body_name,
        },
        "success_thresholds": {
            "success_distance_threshold": args.success_distance_threshold,
            "success_lateral_threshold": args.success_lateral_threshold,
            "success_force_threshold": args.success_force_threshold,
            "success_hold_steps": args.success_hold_steps,
        },
        "runs": [],
    }

    env = os.environ.copy()
    if args.mujoco_gl:
        env["MUJOCO_GL"] = args.mujoco_gl

    for point in task_points:
        x_offset = point["hole_offset_x"]
        y_offset = point["hole_offset_y"]
        z_offset = point["hole_offset_z"]
        seed = args.base_seed + len(manifest["runs"])
        output_dir = args.output_root / point["run_name"]
        command = _build_rollout_command(
            args,
            output_dir,
            x_offset,
            y_offset,
            z_offset,
            seed,
        )
        run_entry: dict[str, Any] = {
            "point_index": point["point_index"],
            "sampling_mode": args.sampling_mode,
            "base_seed": args.base_seed,
            "x_offset": x_offset,
            "y_offset": y_offset,
            "z_offset": z_offset,
            "repeat_index": point["repeat_index"],
            "seed": seed,
            "output_dir": output_dir,
            "command": command,
            "return_code": None,
            "status": "planned",
            "start_time": None,
            "end_time": None,
        }
        summary_path = output_dir / "summary.json"
        if args.skip_existing and _read_summary(summary_path) is not None:
            run_entry["status"] = "skipped_existing"
            run_entry["return_code"] = 0
            manifest["runs"].append(run_entry)
            continue
        if args.dry_run:
            print(" ".join(command))
            run_entry["status"] = "dry_run"
            manifest["runs"].append(run_entry)
            continue

        run_entry["status"] = "running"
        run_entry["start_time"] = datetime.now(timezone.utc).isoformat()
        completed = subprocess.run(command, env=env, check=False)
        run_entry["end_time"] = datetime.now(timezone.utc).isoformat()
        run_entry["return_code"] = completed.returncode
        if completed.returncode != 0:
            run_entry["status"] = "process_error"
            manifest["runs"].append(run_entry)
            _write_manifest(manifest_path, manifest)
            if not args.continue_on_error:
                break
            continue
        summary = _read_summary(summary_path) or {}
        run_entry["status"] = "success" if summary.get("success") else "task_failed"
        run_entry["stop_reason"] = summary.get("stop_reason")
        manifest["runs"].append(run_entry)
        _write_manifest(manifest_path, manifest)

    _write_manifest(manifest_path, manifest)
    summary_csv = args.output_root / "grid_summary.csv"
    rows = write_position_summary_csv(summary_csv, manifest)
    write_random_position_summary(args.output_root / "random_position_summary.json", manifest, rows)
    if args.plot_results and not args.dry_run:
        _plot_results(summary_csv, args.output_root, args.plot_formats)
    return manifest


def _print_final_counts(manifest: dict[str, Any]) -> None:
    runs = manifest["runs"]
    completed = [run for run in runs if run.get("status") in {"success", "task_failed", "skipped_existing"}]
    successful = [run for run in runs if run.get("status") == "success"]
    failed_task = [run for run in runs if run.get("status") == "task_failed"]
    process_errors = [run for run in runs if run.get("status") == "process_error"]
    denominator = len(completed)
    success_rate = len(successful) / denominator if denominator else 0.0
    completion_rate = denominator / manifest.get("total_planned_runs", len(runs)) if manifest.get("total_planned_runs", len(runs)) else 0.0
    lower, upper = wilson_ci(len(successful), denominator)
    safe_successful = 0
    threshold = float(manifest["success_thresholds"]["success_force_threshold"])
    for run in completed:
        summary = _read_summary(Path(run["output_dir"]) / "summary.json") or {}
        if summary.get("success") and float(summary.get("max_force_norm", float("inf"))) < threshold:
            safe_successful += 1
    print(f"total_planned_runs={manifest.get('total_planned_runs', len(runs))}")
    print(f"requested_points={manifest.get('total_planned_runs', len(runs))}")
    print(f"completed_runs={len(completed)}")
    print(f"completed_points={len(completed)}")
    print(f"successful_runs={len(successful)}")
    print(f"successful_points={len(successful)}")
    print(f"failed_task_runs={len(failed_task)}")
    print(f"failed_task_points={len(failed_task)}")
    print(f"process_error_runs={len(process_errors)}")
    print(f"process_error_points={len(process_errors)}")
    print(f"safe_successful_points={safe_successful}")
    print(f"success_rate={success_rate:.6g}")
    print(f"task_success_rate={success_rate:.6g}")
    print(f"safe_success_rate={(safe_successful / denominator if denominator else 0.0):.6g}")
    print(f"success_rate_ci95_lower={lower:.6g}")
    print(f"success_rate_ci95_upper={upper:.6g}")
    print(f"completion_rate={completion_rate:.6g}")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a hole-position perturbation rollout grid.")
    parser.add_argument(
        "--sampling-mode",
        choices=("grid", "random", "latin_hypercube"),
        default="grid",
    )
    parser.add_argument("--num-points", type=int, default=50)
    parser.add_argument("--x-min", type=float, default=-0.002)
    parser.add_argument("--x-max", type=float, default=0.002)
    parser.add_argument("--z-min", type=float, default=-0.002)
    parser.add_argument("--z-max", type=float, default=0.002)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--normalization-stats", type=Path, required=True)
    parser.add_argument("--model-xml", type=Path, required=True)
    parser.add_argument("--contact-latent-mode", default="zero")
    parser.add_argument("--action-mode", default="action")
    parser.add_argument("--action-select-mode", default="mid")
    parser.add_argument("--chunk-len", type=int, default=10)
    parser.add_argument("--force-window-len", type=int, default=20)
    parser.add_argument("--force-window-duration", type=float, default=0.25)
    parser.add_argument("--policy-rate-hz", type=float, default=30.0)
    parser.add_argument("--max-rollout-steps", type=int, default=900)
    parser.add_argument("--max-delta-q", type=float, default=0.02)
    parser.add_argument("--force-stop-threshold", type=float, default=1000.0)
    parser.add_argument("--hole-axis-world", type=float, nargs=3, default=(0.0, -1.0, 0.0))
    parser.add_argument("--hole-site-name", default=DEFAULT_HOLE_SITE_NAME)
    parser.add_argument("--hole-body-name", default=DEFAULT_HOLE_BODY_NAME)
    parser.add_argument("--hole-offset-frame", choices=("world", "body"), default="world")
    parser.add_argument("--x-offsets", default="-0.002,0.0,0.002")
    parser.add_argument("--y-offset", type=float, default=0.0)
    parser.add_argument("--z-offsets", default="-0.002,0.0,0.002")
    parser.add_argument("--success-distance-threshold", type=float, default=0.005)
    parser.add_argument("--success-lateral-threshold", type=float, default=0.006)
    parser.add_argument("--success-force-threshold", type=float, default=80.0)
    parser.add_argument("--success-hold-steps", type=int, default=15)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--save-videos", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--mujoco-gl")
    parser.add_argument("--plot-results", dest="plot_results", action="store_true", default=True)
    parser.add_argument("--no-plot-results", dest="plot_results", action="store_false")
    parser.add_argument("--plot-formats", default="png")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.repeats <= 0:
        raise ValueError("--repeats must be positive")
    validate_sampling_bounds(args)
    manifest = run_grid(args)
    _print_final_counts(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
