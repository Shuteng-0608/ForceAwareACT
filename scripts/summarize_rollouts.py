#!/usr/bin/env python3
"""Aggregate MuJoCo rollout summaries into a comparison CSV."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Optional, Sequence


OUTPUT_COLUMNS = [
    "run",
    "success",
    "success_step",
    "success_time",
    "stop_reason",
    "mode",
    "action_select_mode",
    "max_delta_q",
    "steps_executed",
    "final_time",
    "initial_dist",
    "final_dist",
    "min_dist",
    "min_dist_step",
    "initial_axial",
    "final_axial",
    "min_abs_axial",
    "initial_lateral",
    "final_lateral",
    "min_lateral",
    "min_lateral_step",
    "max_force",
    "mean_force",
    "force_gt_5_steps",
    "force_gt_20_steps",
    "force_gt_40_steps",
    "checkpoint",
    "rollout_log_csv",
    "summary_json",
]


def _to_float(value: Any, default: float = float("nan")) -> float:
    if value in (None, ""):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_int(value: Any, default: Optional[int] = None) -> Optional[int]:
    if value in (None, ""):
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _finite_values(rows: list[dict[str, str]], column: str) -> list[tuple[int, float]]:
    values = []
    for index, row in enumerate(rows):
        value = _to_float(row.get(column))
        if value == value:
            values.append((_to_int(row.get("step"), index) or index, value))
    return values


def _min_step(values: list[tuple[int, float]], abs_value: bool = False) -> tuple[Optional[int], float]:
    if not values:
        return None, float("nan")
    key = (lambda item: abs(item[1])) if abs_value else (lambda item: item[1])
    step, value = min(values, key=key)
    return step, abs(value) if abs_value else value


def _count_gt(values: list[float], threshold: float) -> int:
    return sum(value == value and value > threshold for value in values)


def _row_from_summary(run_dir: Path, summary_path: Path) -> dict[str, Any]:
    with summary_path.open() as summary_file:
        summary = json.load(summary_file)
    return {
        "run": run_dir.name,
        "success": bool(summary.get("success", False)),
        "success_step": summary.get("success_step"),
        "success_time": summary.get("success_time"),
        "stop_reason": summary.get("stop_reason", ""),
        "mode": summary.get("contact_latent_mode", ""),
        "action_select_mode": summary.get("action_select_mode", ""),
        "max_delta_q": summary.get("max_delta_q", ""),
        "steps_executed": summary.get("steps_executed", ""),
        "final_time": summary.get("final_time", ""),
        "initial_dist": summary.get("initial_peg_to_hole_dist", ""),
        "final_dist": summary.get("final_peg_to_hole_dist", ""),
        "min_dist": summary.get("min_peg_to_hole_dist", ""),
        "min_dist_step": summary.get("min_peg_to_hole_dist_step", ""),
        "initial_axial": summary.get("initial_peg_to_hole_axial_error", ""),
        "final_axial": summary.get("final_peg_to_hole_axial_error", ""),
        "min_abs_axial": summary.get("min_abs_peg_to_hole_axial_error", ""),
        "initial_lateral": summary.get("initial_peg_to_hole_lateral_error", ""),
        "final_lateral": summary.get("final_peg_to_hole_lateral_error", ""),
        "min_lateral": summary.get("min_peg_to_hole_lateral_error", ""),
        "min_lateral_step": summary.get("min_peg_to_hole_lateral_error_step", ""),
        "max_force": summary.get("max_force_norm", ""),
        "mean_force": summary.get("mean_force_norm", ""),
        "force_gt_5_steps": summary.get("force_gt_5_steps", ""),
        "force_gt_20_steps": summary.get("force_gt_20_steps", ""),
        "force_gt_40_steps": summary.get("force_gt_40_steps", ""),
        "checkpoint": summary.get("checkpoint", ""),
        "rollout_log_csv": summary.get("rollout_log_csv", str(run_dir / "rollout_log.csv")),
        "summary_json": str(summary_path),
    }


def _row_from_csv(run_dir: Path, log_path: Path) -> dict[str, Any]:
    with log_path.open(newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))
    if not rows:
        raise ValueError(f"empty rollout log: {log_path}")

    first = rows[0]
    final = rows[-1]
    distances = _finite_values(rows, "peg_to_hole_dist")
    axial_errors = _finite_values(rows, "peg_to_hole_axial_error")
    lateral_errors = _finite_values(rows, "peg_to_hole_lateral_error")
    force_values = [_to_float(row.get("force_norm")) for row in rows]
    finite_forces = [value for value in force_values if value == value]
    min_dist_step, min_dist = _min_step(distances)
    _, min_abs_axial = _min_step(axial_errors, abs_value=True)
    min_lateral_step, min_lateral = _min_step(lateral_errors)

    success_step = None
    success_time = None
    for row in rows:
        if row.get("stop_reason") == "success" or _to_int(row.get("success_hold_counter"), 0) >= 15:
            success_step = _to_int(row.get("step"))
            success_time = _to_float(row.get("time"))
            break
    success = success_step is not None

    return {
        "run": run_dir.name,
        "success": success,
        "success_step": success_step,
        "success_time": success_time,
        "stop_reason": final.get("stop_reason", ""),
        "mode": final.get("mode", ""),
        "action_select_mode": final.get("action_select_mode", ""),
        "max_delta_q": "",
        "steps_executed": len(rows),
        "final_time": _to_float(final.get("time")),
        "initial_dist": _to_float(first.get("peg_to_hole_dist")),
        "final_dist": _to_float(final.get("peg_to_hole_dist")),
        "min_dist": min_dist,
        "min_dist_step": min_dist_step,
        "initial_axial": _to_float(first.get("peg_to_hole_axial_error")),
        "final_axial": _to_float(final.get("peg_to_hole_axial_error")),
        "min_abs_axial": min_abs_axial,
        "initial_lateral": _to_float(first.get("peg_to_hole_lateral_error")),
        "final_lateral": _to_float(final.get("peg_to_hole_lateral_error")),
        "min_lateral": min_lateral,
        "min_lateral_step": min_lateral_step,
        "max_force": max(finite_forces) if finite_forces else float("nan"),
        "mean_force": sum(finite_forces) / len(finite_forces) if finite_forces else float("nan"),
        "force_gt_5_steps": _count_gt(force_values, 5.0),
        "force_gt_20_steps": _count_gt(force_values, 20.0),
        "force_gt_40_steps": _count_gt(force_values, 40.0),
        "checkpoint": "",
        "rollout_log_csv": str(log_path),
        "summary_json": "",
    }


def collect_rollouts(root: Path, pattern: str) -> list[dict[str, Any]]:
    rows = []
    for run_dir in sorted(path for path in root.glob(pattern) if path.is_dir()):
        summary_path = run_dir / "summary.json"
        log_path = run_dir / "rollout_log.csv"
        if summary_path.is_file():
            rows.append(_row_from_summary(run_dir, summary_path))
        elif log_path.is_file():
            rows.append(_row_from_csv(run_dir, log_path))
    rows.sort(
        key=lambda row: (
            not _to_bool(row.get("success")),
            _to_float(row.get("final_dist")),
            _to_float(row.get("final_lateral")),
            _to_float(row.get("max_force")),
        )
    )
    return rows


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in OUTPUT_COLUMNS})


def print_table(rows: list[dict[str, Any]]) -> None:
    columns = [
        "run",
        "success",
        "stop_reason",
        "action_select_mode",
        "max_delta_q",
        "final_dist",
        "final_lateral",
        "max_force",
    ]
    widths = {
        column: max(len(column), *(len(str(row.get(column, ""))) for row in rows))
        for column in columns
    }
    print("  ".join(column.ljust(widths[column]) for column in columns))
    print("  ".join("-" * widths[column] for column in columns))
    for row in rows:
        print("  ".join(str(row.get(column, "")).ljust(widths[column]) for column in columns))


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize MuJoCo rollout directories.")
    parser.add_argument("--root", type=Path, default=Path("outputs/peg_hole_100"))
    parser.add_argument("--pattern", default="rollout_action_trainzero_20k_bs16*")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/peg_hole_100/rollout_action_trainzero_20k_bs16_summary.csv"),
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    rows = collect_rollouts(args.root, args.pattern)
    write_summary_csv(args.output, rows)
    print_table(rows)
    print(f"summary_csv={args.output}")
    print(f"runs={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
