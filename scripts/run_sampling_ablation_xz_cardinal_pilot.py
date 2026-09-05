#!/usr/bin/env python3
"""Plan or run the four-model fixed-target XZ cardinal-offset pilot."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shlex
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Optional, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
ROLLOUT_SCRIPT = REPO_ROOT / "scripts" / "run_mujoco_policy_rollout.py"
PILOT_VERSION = "sampling_ablation_fixed_target_xz_cardinal_v1"
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "runs"
    / "moving_hole_fixed_peg_sampling_ablation_xz_cardinal_r4mm_h1200_seed0"
)

from scripts.run_sampling_ablation_nominal_rollout_pilot import (  # noqa: E402
    ROLLOUT_PROTOCOL_VERSION,
    NominalPilotSpec,
    build_specs,
    validate_spec_files,
)


@dataclass(frozen=True)
class OffsetPoint:
    point_id: str
    offset_x_m: float
    offset_z_m: float


def build_points(radius_mm: float) -> tuple[OffsetPoint, ...]:
    if not math.isfinite(radius_mm) or radius_mm <= 0.0:
        raise ValueError("radius_mm must be finite and positive")
    radius_m = float(radius_mm) / 1000.0
    token = format(radius_mm, "g").replace(".", "p")
    return (
        OffsetPoint(f"x_p{token}mm", radius_m, 0.0),
        OffsetPoint(f"x_m{token}mm", -radius_m, 0.0),
        OffsetPoint(f"z_p{token}mm", 0.0, radius_m),
        OffsetPoint(f"z_m{token}mm", 0.0, -radius_m),
    )


def build_rollout_command(
    args: argparse.Namespace,
    point: OffsetPoint,
    spec: NominalPilotSpec,
) -> list[str]:
    command = [
        sys.executable,
        str(ROLLOUT_SCRIPT),
        "--checkpoint",
        str(spec.checkpoint),
        "--task-profile",
        "moving_hole_fixed_peg",
        "--terminal-hold-mode",
        "collector",
        "--output-dir",
        str(args.output_dir / point.point_id / spec.model_id),
        "--device",
        args.device,
        "--contact-latent-mode",
        "zero",
        "--action-mode",
        "joint_pos",
        "--action-select-mode",
        "temporal",
        "--temporal-agg-decay",
        "0.01",
        "--policy-rate-hz",
        "30",
        "--max-rollout-steps",
        str(args.max_rollout_steps),
        "--ema-alpha",
        "1",
        "--max-delta-q",
        "0.02",
        "--force-stop-threshold",
        "40",
        "--safe-force-threshold",
        "40",
        f"--hole-offset-x={point.offset_x_m!r}",
        "--hole-offset-y=0.0",
        f"--hole-offset-z={point.offset_z_m!r}",
        "--record-force-metrics",
        "--seed",
        str(args.seed),
    ]
    if args.save_videos:
        command.append("--save-videos")
    if args.execute:
        command.append("--execute-actions")
    return command


def validate_completed_summary(
    summary: dict[str, Any],
    args: argparse.Namespace,
    point: OffsetPoint,
    spec: NominalPilotSpec,
) -> list[str]:
    expected = {
        "rollout_protocol_version": ROLLOUT_PROTOCOL_VERSION,
        "checkpoint": str(spec.checkpoint.resolve()),
        "rollout_mode": "execute",
        "task_profile": "moving_hole_fixed_peg",
        "offset_target_body_name": "fixed_peg_fixture",
        "policy_wrench_source": "compensated",
        "terminal_hold_mode": "collector",
        "seed": args.seed,
        "action_mode": "joint_pos",
        "action_select_mode": "temporal",
        "temporal_agg_decay": 0.01,
        "contact_latent_mode": "zero",
        "deployment_latent_source": "zero",
        "policy_rate_hz": 30.0,
        "max_rollout_steps": args.max_rollout_steps,
        "ema_alpha": 1.0,
        "max_delta_q": 0.02,
        "force_stop_threshold": 40.0,
        "safe_force_threshold": 40.0,
        "hole_offset_x": point.offset_x_m,
        "hole_offset_y": 0.0,
        "hole_offset_z": point.offset_z_m,
        "force_metrics_recorded": True,
    }
    return [
        f"{key}: expected {value!r}, got {summary.get(key)!r}"
        for key, value in expected.items()
        if summary.get(key) != value
    ]


def result_row(
    point: OffsetPoint,
    spec: NominalPilotSpec,
    summary: dict[str, Any],
) -> dict[str, Any]:
    return {
        "point_id": point.point_id,
        "offset_x_mm": point.offset_x_m * 1000.0,
        "offset_z_mm": point.offset_z_m * 1000.0,
        "radial_offset_mm": math.hypot(point.offset_x_m, point.offset_z_m)
        * 1000.0,
        "model_id": spec.model_id,
        "architecture_version": spec.architecture_version,
        "train_sampling_mode": spec.train_sampling_mode,
        "success": summary.get("success"),
        "safe_success": summary.get("safe_success"),
        "task_success_site_reached": summary.get("task_success_site_reached"),
        "terminal_hold_accepted": summary.get("terminal_hold_accepted"),
        "terminal_hold_completion_reason": summary.get(
            "terminal_hold_completion_reason"
        ),
        "stop_reason": summary.get("stop_reason"),
        "steps_executed": summary.get("steps_executed"),
        "success_time": summary.get("success_time"),
        "min_distance_mm": (
            float(summary["min_peg_to_hole_dist"]) * 1000.0
        ),
        "final_distance_mm": (
            float(summary["final_peg_to_hole_dist"]) * 1000.0
        ),
        "max_compensated_force_n": summary.get(
            "force_metrics_compensated_max_force_norm"
        ),
        "summary_json": summary.get("summary_json"),
    }


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_pilot_summary(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    by_model: dict[str, dict[str, Any]] = {}
    for model_id in dict.fromkeys(str(row["model_id"]) for row in rows):
        model_rows = [row for row in rows if row["model_id"] == model_id]
        successful = [row for row in model_rows if row["success"] is True]
        forces = [float(row["max_compensated_force_n"]) for row in model_rows]
        by_model[model_id] = {
            "completed_points": len(model_rows),
            "successes": len(successful),
            "safe_successes": sum(row["safe_success"] is True for row in model_rows),
            "success_rate": len(successful) / len(model_rows),
            "mean_success_time": (
                mean(float(row["success_time"]) for row in successful)
                if successful
                else None
            ),
            "max_compensated_force_n": max(forces),
        }
    payload = {
        "pilot_version": PILOT_VERSION,
        "completed_rollouts": len(rows),
        "by_model": by_model,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plan a four-direction fixed-peg XZ robustness pilot by default; "
            "pass --execute to run all four models at every point."
        )
    )
    from scripts.run_sampling_ablation_nominal_rollout_pilot import (
        DEFAULT_CHECKPOINT_ROOT,
    )

    parser.add_argument(
        "--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--radius-mm", type=float, default=4.0)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--max-rollout-steps", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--save-videos",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    args.checkpoint_root = args.checkpoint_root.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    if not math.isfinite(args.radius_mm) or args.radius_mm <= 0.0:
        parser.error("--radius-mm must be finite and positive")
    if args.max_rollout_steps <= 0:
        parser.error("--max-rollout-steps must be positive")
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    specs = build_specs(args.checkpoint_root)
    points = build_points(args.radius_mm)
    try:
        validate_spec_files(specs)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    planned = [
        (point, spec, build_rollout_command(args, point, spec))
        for point in points
        for spec in specs
    ]
    plan = {
        "pilot_version": PILOT_VERSION,
        "execute_requested": args.execute,
        "output_dir": str(args.output_dir),
        "radius_mm": args.radius_mm,
        "max_rollout_steps": args.max_rollout_steps,
        "seed": args.seed,
        "point_semantics": (
            "world_XZ_translation_of_fixed_peg_fixture; training target position "
            "was always nominal, so this is a local target-location OOD test"
        ),
        "training_collision_error_radii_mm": [4.0, 6.0, 8.0, 10.0],
        "fixed_contract": {
            "task_profile": "moving_hole_fixed_peg",
            "offset_target_body": "fixed_peg_fixture",
            "terminal_hold_mode": "collector",
            "action_select_mode": "temporal",
            "temporal_agg_decay": 0.01,
            "contact_latent_mode": "zero",
            "hard_stop_force_n": 40.0,
            "axial_push": False,
        },
        "runs": [
            {
                "point": asdict(point),
                "model_id": spec.model_id,
                "architecture_version": spec.architecture_version,
                "train_sampling_mode": spec.train_sampling_mode,
                "command": shlex.join(command),
            }
            for point, spec, command in planned
        ],
    }
    print(json.dumps(plan, indent=2, sort_keys=True))
    if not args.execute:
        print("plan_only=true; no rollout actions were executed")
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "pilot_plan.json").write_text(
        json.dumps(plan, indent=2, sort_keys=True) + "\n"
    )
    environment = os.environ.copy()
    environment["MUJOCO_GL"] = "egl"
    existing_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        str(REPO_ROOT / "src")
        if not existing_pythonpath
        else str(REPO_ROOT / "src") + os.pathsep + existing_pythonpath
    )

    rows: list[dict[str, Any]] = []
    for index, (point, spec, command) in enumerate(planned, start=1):
        output_dir = args.output_dir / point.point_id / spec.model_id
        summary_path = output_dir / "summary.json"
        if output_dir.exists():
            if not summary_path.is_file():
                print(
                    f"error: refusing to overwrite incomplete output: {output_dir}",
                    file=sys.stderr,
                )
                return 2
            summary = json.loads(summary_path.read_text())
            errors = validate_completed_summary(summary, args, point, spec)
            if errors:
                print(
                    f"error: existing summary mismatch for {point.point_id}/"
                    f"{spec.model_id}:\n- " + "\n- ".join(errors),
                    file=sys.stderr,
                )
                return 2
            print(
                f"[{index}/{len(planned)}] reusing {point.point_id}/"
                f"{spec.model_id}",
                flush=True,
            )
        else:
            print(
                f"[{index}/{len(planned)}] starting {point.point_id}/"
                f"{spec.model_id}",
                flush=True,
            )
            completed = subprocess.run(command, cwd=REPO_ROOT, env=environment)
            if completed.returncode != 0:
                print(
                    f"error: rollout exited with {completed.returncode}: "
                    f"{point.point_id}/{spec.model_id}",
                    file=sys.stderr,
                )
                return completed.returncode
            summary = json.loads(summary_path.read_text())
            errors = validate_completed_summary(summary, args, point, spec)
            if errors:
                print(
                    f"error: completed summary mismatch for {point.point_id}/"
                    f"{spec.model_id}:\n- " + "\n- ".join(errors),
                    file=sys.stderr,
                )
                return 2
            print(
                f"[{index}/{len(planned)}] completed {point.point_id}/"
                f"{spec.model_id}",
                flush=True,
            )
        rows.append(result_row(point, spec, summary))
        _write_csv(args.output_dir / "aggregate.csv", rows)
        _write_pilot_summary(args.output_dir / "pilot_summary.json", rows)

    print(f"aggregate_csv={args.output_dir / 'aggregate.csv'}")
    print(f"pilot_summary_json={args.output_dir / 'pilot_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
