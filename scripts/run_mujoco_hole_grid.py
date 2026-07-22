#!/usr/bin/env python3
"""Run a deterministic grid of hole-position perturbation rollouts."""

from __future__ import annotations

import argparse
import copy
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

from run_mujoco_policy_rollout import (  # noqa: E402
    DEFAULT_HOLE_BODY_NAME,
    DEFAULT_HOLE_SITE_NAME,
    _contact_recovery_config,
    _prepare_rollout_artifacts,
    _resolve_and_validate_rollout_args,
    parse_args as parse_rollout_args,
)
from summarize_rollouts import collect_rollouts, write_summary_csv  # noqa: E402

POSITION_SUMMARY_COLUMNS = [
    "point_index",
    "sampling_mode",
    "base_seed",
    "point_set_seed",
    "rollout_seed_base",
    "rollout_seed",
    "hole_offset_x",
    "hole_offset_y",
    "hole_offset_z",
    "radial_offset",
    "quadrant",
    "success",
    "safe_success",
    "recovery_success",
    "safe_recovery_success",
    "contact_recovery_metrics_valid",
    "contact_event_count",
    "contact_duration_s",
    "force_excess_integral_n_s",
    "hard_force_violation",
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


def resolve_point_set_seed(args: argparse.Namespace) -> int:
    """Return the seed used only to generate random/LHS task points."""
    return int(args.point_set_seed if args.point_set_seed is not None else args.base_seed)


def resolve_rollout_seed_base(args: argparse.Namespace) -> int:
    """Return the first per-run seed, preserving legacy coupled behavior."""
    point_set_seed = resolve_point_set_seed(args)
    return int(
        args.rollout_seed_base
        if args.rollout_seed_base is not None
        else point_set_seed
    )


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


def validate_rollout_parameters(args: argparse.Namespace) -> None:
    """Reject invalid values before a grid or even a dry-run manifest is written."""

    float_names = (
        "temporal_agg_decay",
        "force_window_duration",
        "policy_rate_hz",
        "max_delta_q",
        "force_stop_threshold",
        "y_offset",
        "success_distance_threshold",
        "success_lateral_threshold",
        "success_force_threshold",
        "contact_enter_force_threshold",
        "contact_exit_force_threshold",
        "safe_force_threshold",
        "hard_force_threshold",
    )
    nonfinite = [
        name
        for name in float_names
        if getattr(args, name) is not None
        and not math.isfinite(float(getattr(args, name)))
    ]
    if nonfinite:
        raise ValueError(
            "grid rollout float arguments must be finite: " + ", ".join(nonfinite)
        )
    if args.temporal_agg_decay < 0 or args.force_window_duration < 0:
        raise ValueError("temporal decay and force-window duration must be non-negative")
    if args.policy_rate_hz <= 0 or args.max_delta_q <= 0 or args.force_stop_threshold <= 0:
        raise ValueError("policy rate, max delta, and force-stop threshold must be positive")
    if (
        args.success_distance_threshold <= 0
        or args.success_lateral_threshold <= 0
        or args.success_force_threshold <= 0
        or args.success_hold_steps <= 0
    ):
        raise ValueError("success thresholds and hold steps must be positive")
    if args.chunk_len <= 0 or args.force_window_len <= 0 or args.max_rollout_steps <= 0:
        raise ValueError("chunk/window/rollout lengths must be positive")
    axis = np.asarray(args.hole_axis_world, dtype=np.float64)
    if not np.isfinite(axis).all() or float(np.linalg.norm(axis)) <= 0.0:
        raise ValueError("--hole-axis-world must be a finite nonzero vector")
    for label, raw_offsets in (
        ("--x-offsets", args.x_offsets),
        ("--z-offsets", args.z_offsets),
    ):
        if not all(math.isfinite(value) for value in parse_offset_list(raw_offsets)):
            raise ValueError(f"{label} must contain only finite values")
    _contact_recovery_config(args)


def resolved_sampling_mode(args: argparse.Namespace) -> str:
    return "file" if args.task_points_csv is not None else args.sampling_mode


def read_task_points_csv(path: Path, default_y_offset: float = 0.0) -> list[dict[str, Any]]:
    """Load fixed rollout points whose hole_offset coordinates are in metres."""
    if not path.is_file():
        raise FileNotFoundError(f"task-points CSV does not exist: {path}")
    with path.open(encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None:
            raise ValueError(f"task-points CSV has no header: {path}")
        required = {"hole_offset_x", "hole_offset_z"}
        missing = sorted(required - set(reader.fieldnames))
        if missing:
            raise ValueError(
                f"task-points CSV missing required column(s): {', '.join(missing)}"
            )
        rows = list(reader)
    if not rows:
        raise ValueError(f"task-points CSV contains no points: {path}")

    task_points: list[dict[str, Any]] = []
    seen_coordinates: set[tuple[float, float, float]] = set()
    for expected_index, row in enumerate(rows, start=1):
        raw_index = row.get("point_index", "")
        if raw_index is None or not raw_index.strip():
            point_index = expected_index
        else:
            try:
                numeric_index = float(raw_index)
                point_index = int(numeric_index)
            except ValueError as error:
                raise ValueError(
                    f"task-points CSV row {expected_index} has invalid point_index: {raw_index!r}"
                ) from error
            if numeric_index != point_index or point_index != expected_index:
                raise ValueError(
                    "task-points CSV point_index must be consecutive integers starting at 1; "
                    f"row {expected_index} contains {raw_index!r}"
                )
        try:
            x_offset = float(row["hole_offset_x"])
            z_offset = float(row["hole_offset_z"])
            raw_y = row.get("hole_offset_y", "")
            y_offset = default_y_offset if raw_y is None or not raw_y.strip() else float(raw_y)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"task-points CSV row {expected_index} contains a non-numeric offset"
            ) from error
        coordinates = (x_offset, y_offset, z_offset)
        if not all(math.isfinite(value) for value in coordinates):
            raise ValueError(
                f"task-points CSV row {expected_index} contains a non-finite offset"
            )
        if coordinates in seen_coordinates:
            raise ValueError(
                f"task-points CSV row {expected_index} duplicates an earlier coordinate"
            )
        seen_coordinates.add(coordinates)
        task_points.append(
            {
                "point_index": point_index,
                "hole_offset_x": x_offset,
                "hole_offset_y": y_offset,
                "hole_offset_z": z_offset,
                "repeat_index": 1,
                "run_name": point_run_name(point_index, x_offset, z_offset, 1),
            }
        )
    return task_points


def generate_task_points(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.task_points_csv is not None:
        return read_task_points_csv(args.task_points_csv, args.y_offset)
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
        points = random_points(
            args.num_points,
            args.x_min,
            args.x_max,
            args.z_min,
            args.z_max,
            resolve_point_set_seed(args),
        )
    elif args.sampling_mode == "latin_hypercube":
        points = latin_hypercube_points(
            args.num_points,
            args.x_min,
            args.x_max,
            args.z_min,
            args.z_max,
            resolve_point_set_seed(args),
        )
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
            value = json.load(summary_file)
        return value if isinstance(value, dict) else None
    except Exception:
        return None


def _expected_rollout_contract(
    command: Sequence[str],
    *,
    mujoco_gl: Optional[str] = None,
    contract_template: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Recreate the rollout's exact contract without launching MuJoCo."""

    if len(command) < 3:
        raise ValueError("rollout command is incomplete")
    rollout_args = parse_rollout_args(command[2:])
    _resolve_and_validate_rollout_args(rollout_args)
    if contract_template is not None:
        # Grid commands vary only in these per-point fields. Reusing the
        # already-verified invariant portion avoids deserializing and hashing a
        # potentially large checkpoint once per completed point.
        contract = copy.deepcopy(contract_template)
        contract["runtime"]["seed"] = rollout_args.seed
        contract["runtime"]["hole_offset"] = [
            rollout_args.hole_offset_x,
            rollout_args.hole_offset_y,
            rollout_args.hole_offset_z,
        ]
        contract["runtime"]["mujoco_gl"] = (
            os.environ.get("MUJOCO_GL") if mujoco_gl is None else mujoco_gl
        )
        contract["output_dir"] = str(rollout_args.output_dir.resolve())
        return contract
    _, _, _, contract = _prepare_rollout_artifacts(
        rollout_args,
        mujoco_gl=mujoco_gl,
    )
    return contract


def _validate_existing_summary_contract(
    summary: dict[str, Any],
    command: Sequence[str],
    summary_path: Path,
    *,
    mujoco_gl: Optional[str] = None,
    contract_template: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Authorize --skip-existing only for an exact, auditable contract match."""

    recorded = summary.get("rollout_contract")
    if not isinstance(recorded, dict):
        raise ValueError(
            "refusing --skip-existing for a legacy or incomplete summary without "
            f"rollout_contract: {summary_path}"
        )
    expected = _expected_rollout_contract(
        command,
        mujoco_gl=mujoco_gl,
        contract_template=contract_template,
    )
    try:
        recorded_json = json.dumps(
            recorded,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as error:
        raise ValueError(
            "refusing --skip-existing because rollout_contract is not canonical "
            f"finite JSON: {summary_path}"
        ) from error
    expected_json = json.dumps(
        expected,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    if recorded_json != expected_json:
        raise ValueError(
            "refusing --skip-existing because the existing rollout_contract does "
            f"not exactly match the requested rollout: {summary_path}"
        )
    return expected


def _artifact_stat_signature(args: argparse.Namespace) -> tuple[tuple[int, int, int], ...]:
    """Cheaply detect artifact changes while reusing a verified grid template."""

    signature = []
    for path in (args.checkpoint, args.normalization_stats, args.model_xml):
        stat_result = Path(path).expanduser().resolve().stat()
        signature.append(
            (int(stat_result.st_ino), int(stat_result.st_size), int(stat_result.st_mtime_ns))
        )
    return tuple(signature)


def _contact_recovery_fields(summary: dict[str, Any]) -> dict[str, Any]:
    """Extract canonical contact/recovery fields without inventing legacy values."""

    raw_metrics = summary.get("contact_recovery_metrics", {})
    metrics = raw_metrics if isinstance(raw_metrics, dict) else {}

    def first_present(*keys_and_mappings: tuple[str, dict[str, Any]]) -> Any:
        for key, mapping in keys_and_mappings:
            if key in mapping:
                return mapping[key]
        return ""

    return {
        "recovery_success": first_present(
            ("recovery_success", summary),
            ("recovery_success", metrics),
        ),
        "safe_recovery_success": first_present(
            ("safe_recovery_success", summary),
            ("safe_success", metrics),
        ),
        "contact_recovery_metrics_valid": first_present(
            ("contact_recovery_metrics_valid", summary),
            ("contact_metrics_valid", summary),
            ("metrics_valid", metrics),
        ),
        "contact_event_count": first_present(
            ("contact_event_count", summary),
            ("contact_event_count", metrics),
        ),
        "contact_duration_s": first_present(
            ("contact_duration_s", summary),
            ("contact_duration_s", metrics),
        ),
        "force_excess_integral_n_s": first_present(
            ("force_excess_integral_n_s", summary),
            ("force_excess_integral_n_s", metrics),
        ),
        "hard_force_violation": first_present(
            ("hard_force_violation", summary),
            ("hard_force_violation", metrics),
        ),
    }


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as manifest_file:
        json.dump(_json_safe(manifest), manifest_file, indent=2, sort_keys=True)
        manifest_file.write("\n")
    print(f"grid_manifest_json={path}")


def write_task_points_csv(path: Path, task_points: list[dict[str, Any]], args: argparse.Namespace) -> None:
    point_set_seed = resolve_point_set_seed(args)
    rollout_seed_base = resolve_rollout_seed_base(args)
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
                "point_set_seed",
                "rollout_seed_base",
                "rollout_seed",
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
                    "sampling_mode": resolved_sampling_mode(args),
                    "base_seed": point_set_seed,
                    "point_set_seed": point_set_seed,
                    "rollout_seed_base": rollout_seed_base,
                    "rollout_seed": rollout_seed_base + point["point_index"] - 1,
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
    try:
        finite_max_force = float(max_force)
    except (TypeError, ValueError):
        finite_max_force = float("nan")
    safe_success = bool(
        success
        and math.isfinite(finite_max_force)
        and finite_max_force < success_force_threshold
    )
    contact_recovery_fields = _contact_recovery_fields(summary)
    return {
        "point_index": run.get("point_index"),
        "sampling_mode": run.get("sampling_mode"),
        "base_seed": run.get("base_seed"),
        "point_set_seed": run.get("point_set_seed", run.get("base_seed")),
        "rollout_seed_base": run.get("rollout_seed_base", run.get("base_seed")),
        "rollout_seed": run.get("rollout_seed", run.get("seed")),
        "hole_offset_x": x_offset,
        "hole_offset_y": y_offset,
        "hole_offset_z": z_offset,
        "radial_offset": math.sqrt(x_offset * x_offset + z_offset * z_offset),
        "quadrant": quadrant(x_offset, z_offset),
        "success": success,
        "safe_success": safe_success,
        **contact_recovery_fields,
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
    contact_metrics_reported = [
        row
        for row in completed
        if row.get("contact_recovery_metrics_valid") not in ("", None)
    ]
    contact_metrics_valid = [
        row
        for row in contact_metrics_reported
        if _bool_value(row["contact_recovery_metrics_valid"])
    ]
    recovery_successes = [
        row
        for row in contact_metrics_valid
        if _bool_value(row.get("recovery_success", False))
    ]
    safe_recovery_successes = [
        row
        for row in contact_metrics_valid
        if _bool_value(row.get("safe_recovery_success", False))
    ]
    hard_force_violations = [
        row
        for row in contact_metrics_valid
        if _bool_value(row.get("hard_force_violation", False))
    ]
    lower, upper = wilson_ci(len(successes), len(completed))
    success_times = [float(row["success_time"]) for row in successes if row.get("success_time") not in ("", None)]
    max_forces = [float(row["max_force"]) for row in completed if row.get("max_force") not in ("", None)]
    contact_event_counts = [
        float(row["contact_event_count"])
        for row in contact_metrics_valid
        if row.get("contact_event_count") not in ("", None)
    ]
    contact_durations = [
        float(row["contact_duration_s"])
        for row in contact_metrics_valid
        if row.get("contact_duration_s") not in ("", None)
    ]
    force_excess_integrals = [
        float(row["force_excess_integral_n_s"])
        for row in contact_metrics_valid
        if row.get("force_excess_integral_n_s") not in ("", None)
    ]
    summary = {
        "sampling_mode": manifest["sampling_mode"],
        "requested_bounds": {
            "x_min": manifest["x_min"],
            "x_max": manifest["x_max"],
            "z_min": manifest["z_min"],
            "z_max": manifest["z_max"],
        },
        "base_seed": manifest["base_seed"],
        "point_set_seed": manifest.get("point_set_seed", manifest["base_seed"]),
        "rollout_seed_base": manifest.get("rollout_seed_base", manifest["base_seed"]),
        "planned_runs": manifest["total_planned_runs"],
        "completed_runs": len(completed),
        "process_error_runs": len(process_errors),
        "completion_rate": len(completed) / manifest["total_planned_runs"] if manifest["total_planned_runs"] else 0.0,
        "successes": len(successes),
        "safe_successes": len(safe_successes),
        "success_rate": len(successes) / len(completed) if completed else 0.0,
        "safe_success_rate": len(safe_successes) / len(completed) if completed else 0.0,
        "recovery_successes": len(recovery_successes),
        "safe_recovery_successes": len(safe_recovery_successes),
        "recovery_success_rate": (
            len(recovery_successes) / len(contact_metrics_valid)
            if contact_metrics_valid
            else float("nan")
        ),
        "safe_recovery_success_rate": (
            len(safe_recovery_successes) / len(contact_metrics_valid)
            if contact_metrics_valid
            else float("nan")
        ),
        "recovery_rate_denominator_valid_runs": len(contact_metrics_valid),
        "contact_metrics_reported_runs": len(contact_metrics_reported),
        "contact_metrics_valid_runs": len(contact_metrics_valid),
        "contact_metrics_invalid_runs": (
            len(contact_metrics_reported) - len(contact_metrics_valid)
        ),
        "hard_force_violations": len(hard_force_violations),
        "hard_force_violation_rate": (
            len(hard_force_violations) / len(contact_metrics_valid)
            if contact_metrics_valid
            else float("nan")
        ),
        "success_rate_ci95_lower": lower,
        "success_rate_ci95_upper": upper,
        "mean_success_time": _mean(success_times),
        "median_success_time": _median(success_times),
        "mean_max_force": _mean(max_forces),
        "max_force": max(max_forces) if max_forces else float("nan"),
        "mean_contact_event_count": _mean(contact_event_counts),
        "mean_contact_duration_s": _mean(contact_durations),
        "mean_force_excess_integral_n_s": _mean(force_excess_integrals),
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
        "--device",
        args.device,
        "--model-xml",
        str(args.model_xml),
        "--contact-latent-mode",
        args.contact_latent_mode,
        "--action-mode",
        args.action_mode,
        "--action-select-mode",
        args.action_select_mode,
        "--temporal-agg-decay",
        str(args.temporal_agg_decay),
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
        "--contact-enter-force-threshold",
        str(args.contact_enter_force_threshold),
        "--contact-exit-force-threshold",
        str(args.contact_exit_force_threshold),
        "--contact-min-steps",
        str(args.contact_min_steps),
        "--output-dir",
        str(output_dir),
        "--seed",
        str(seed),
        "--execute-actions",
    ]
    if args.safe_force_threshold is not None:
        command.extend(("--safe-force-threshold", str(args.safe_force_threshold)))
    if args.hard_force_threshold is not None:
        command.extend(("--hard-force-threshold", str(args.hard_force_threshold)))
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
    validate_rollout_parameters(args)
    contact_recovery_config = _contact_recovery_config(args)
    point_set_seed = resolve_point_set_seed(args)
    rollout_seed_base = resolve_rollout_seed_base(args)
    task_points = generate_task_points(args)
    args.output_root.mkdir(parents=True, exist_ok=True)
    write_task_points_csv(args.output_root / "task_points.csv", task_points, args)
    manifest_path = args.output_root / "grid_manifest.json"
    manifest: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "sampling_mode": resolved_sampling_mode(args),
        "num_points": len(task_points),
        "task_points_csv": args.task_points_csv,
        "x_min": args.x_min,
        "x_max": args.x_max,
        "z_min": args.z_min,
        "z_max": args.z_max,
        # Keep base_seed as the point-set seed for historical readers.
        "base_seed": point_set_seed,
        "point_set_seed": point_set_seed,
        "rollout_seed_base": rollout_seed_base,
        "device": args.device,
        "x_offsets": parse_offset_list(args.x_offsets),
        "y_offset": args.y_offset,
        "z_offsets": parse_offset_list(args.z_offsets),
        "repeats": args.repeats,
        "total_planned_runs": len(task_points),
        "task_points": [
            {
                **point,
                "output_dir": args.output_root / point["run_name"],
                "sampling_mode": resolved_sampling_mode(args),
                "base_seed": point_set_seed,
                "point_set_seed": point_set_seed,
                "rollout_seed_base": rollout_seed_base,
                "rollout_seed": rollout_seed_base + point["point_index"] - 1,
            }
            for point in task_points
        ],
        "policy_config": {
            "checkpoint": args.checkpoint,
            "normalization_stats": args.normalization_stats,
            "device": args.device,
            "model_xml": args.model_xml,
            "contact_latent_mode": args.contact_latent_mode,
            "action_mode": args.action_mode,
            "action_select_mode": args.action_select_mode,
            "temporal_agg_decay": args.temporal_agg_decay,
            "chunk_len": args.chunk_len,
            "force_window_len": args.force_window_len,
            "force_window_duration": args.force_window_duration,
            "policy_rate_hz": args.policy_rate_hz,
            "max_rollout_steps": args.max_rollout_steps,
            "max_delta_q": args.max_delta_q,
            "force_stop_threshold": args.force_stop_threshold,
            "contact_enter_force_threshold": args.contact_enter_force_threshold,
            "contact_exit_force_threshold": args.contact_exit_force_threshold,
            "contact_min_steps": args.contact_min_steps,
            "safe_force_threshold": args.safe_force_threshold,
            "hard_force_threshold": args.hard_force_threshold,
            "hole_site_name": args.hole_site_name,
            "hole_body_name": args.hole_body_name,
        },
        "success_thresholds": {
            "success_distance_threshold": args.success_distance_threshold,
            "success_lateral_threshold": args.success_lateral_threshold,
            "success_force_threshold": args.success_force_threshold,
            "success_hold_steps": args.success_hold_steps,
        },
        "contact_recovery_config": contact_recovery_config.to_dict(),
        "runs": [],
    }

    env = os.environ.copy()
    if args.mujoco_gl:
        env["MUJOCO_GL"] = args.mujoco_gl

    skip_contract_template: Optional[dict[str, Any]] = None
    skip_artifact_signature: Optional[tuple[tuple[int, int, int], ...]] = None
    for point in task_points:
        x_offset = point["hole_offset_x"]
        y_offset = point["hole_offset_y"]
        z_offset = point["hole_offset_z"]
        seed = rollout_seed_base + len(manifest["runs"])
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
            "sampling_mode": resolved_sampling_mode(args),
            "base_seed": point_set_seed,
            "point_set_seed": point_set_seed,
            "rollout_seed_base": rollout_seed_base,
            "x_offset": x_offset,
            "y_offset": y_offset,
            "z_offset": z_offset,
            "repeat_index": point["repeat_index"],
            "seed": seed,
            "rollout_seed": seed,
            "output_dir": output_dir,
            "command": command,
            "device": args.device,
            "return_code": None,
            "status": "planned",
            "start_time": None,
            "end_time": None,
        }
        summary_path = output_dir / "summary.json"
        existing_summary = None
        if args.skip_existing and summary_path.exists():
            existing_summary = _read_summary(summary_path)
            if existing_summary is None:
                raise ValueError(
                    "refusing --skip-existing because summary.json is not a valid "
                    "JSON object: "
                    f"{summary_path}"
                )
            current_artifact_signature = _artifact_stat_signature(args)
            if current_artifact_signature != skip_artifact_signature:
                skip_contract_template = None
            skip_contract_template = _validate_existing_summary_contract(
                existing_summary,
                command,
                summary_path,
                mujoco_gl=args.mujoco_gl,
                contract_template=skip_contract_template,
            )
            skip_artifact_signature = current_artifact_signature
        if existing_summary is not None:
            run_entry["status"] = "skipped_existing"
            run_entry["return_code"] = 0
            run_entry.update(_contact_recovery_fields(existing_summary))
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
        run_entry.update(_contact_recovery_fields(summary))
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
        choices=("grid", "random", "latin_hypercube", "file"),
        default="grid",
    )
    parser.add_argument(
        "--task-points-csv",
        type=Path,
        default=None,
        help=(
            "Read fixed points from CSV instead of generating them. Required columns: "
            "hole_offset_x and hole_offset_z in metres; hole_offset_y is optional."
        ),
    )
    parser.add_argument("--num-points", type=int, default=50)
    parser.add_argument("--x-min", type=float, default=-0.002)
    parser.add_argument("--x-max", type=float, default=0.002)
    parser.add_argument("--z-min", type=float, default=-0.002)
    parser.add_argument("--z-max", type=float, default=0.002)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--normalization-stats", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--model-xml", type=Path, required=True)
    parser.add_argument("--contact-latent-mode", choices=("zero", "prior"), default="zero")
    parser.add_argument("--action-mode", default="action")
    parser.add_argument("--action-select-mode", default="mid")
    parser.add_argument("--temporal-agg-decay", type=float, default=0.3)
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
    parser.add_argument("--success-force-threshold", type=float, default=40.0)
    parser.add_argument("--success-hold-steps", type=int, default=15)
    parser.add_argument("--contact-enter-force-threshold", type=float, default=5.0)
    parser.add_argument("--contact-exit-force-threshold", type=float, default=3.0)
    parser.add_argument("--contact-min-steps", type=int, default=2)
    parser.add_argument(
        "--safe-force-threshold",
        type=float,
        default=None,
        help="Optional peak-force ceiling for safe recovery; rollout default applies when omitted.",
    )
    parser.add_argument(
        "--hard-force-threshold",
        type=float,
        default=None,
        help="Optional hard-violation threshold; rollout default applies when omitted.",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument(
        "--base-seed",
        type=int,
        default=0,
        help=(
            "Legacy coupled seed. Used for point generation and as the first "
            "rollout seed unless the separated seed options are provided."
        ),
    )
    parser.add_argument(
        "--point-set-seed",
        type=int,
        default=None,
        help="Seed used only for random/LHS task-point generation.",
    )
    parser.add_argument(
        "--rollout-seed-base",
        type=int,
        default=None,
        help=(
            "First rollout seed. Point i uses rollout-seed-base + i - 1. "
            "Defaults to the resolved point-set seed for legacy behavior."
        ),
    )
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
    if args.sampling_mode == "file" and args.task_points_csv is None:
        raise ValueError("--sampling-mode file requires --task-points-csv")
    if args.task_points_csv is None:
        validate_sampling_bounds(args)
    manifest = run_grid(args)
    _print_final_counts(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
