#!/usr/bin/env python3
"""Run paired ACT/Contact-CVAE Q=1 rollouts on fixed perturbation points."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_paired50_q1_temporal_rollout_pilot import (  # noqa: E402
    DEFAULT_CONTACT_CHECKPOINT,
    DEFAULT_FORCE_HUD_CAMERA,
    DEFAULT_FORCE_HUD_HEIGHT,
    DEFAULT_FORCE_HUD_WIDTH,
    DEFAULT_MODEL_XML,
    DEFAULT_OFFICIAL_CHECKPOINT,
    MODEL_IDS,
    PilotSpec,
    build_rollout_command,
    build_signed_decay_specs,
    validate_completed_summary,
)


EXPERIMENT_VERSION = "paired_fibonacci_temporal_rollouts_v1"
DEFAULT_POINT_SET = REPO_ROOT / "configs/experiments/fibonacci_disk_100_r4mm.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "runs/paired_fibonacci_r4mm_temporal_b1"
DEFAULT_SIGNED_DECAYS = (0.01, 0.05, 0.071, 0.0735, 0.075)


@dataclass(frozen=True)
class TaskPoint:
    point_index: int
    hole_offset_x: float
    hole_offset_y: float
    hole_offset_z: float

    @property
    def radius_mm(self) -> float:
        return 1000.0 * math.hypot(self.hole_offset_x, self.hole_offset_z)


@dataclass(frozen=True)
class PlannedRun:
    configuration_id: str
    model_id: str
    executor_id: str
    signed_decay: float
    point_index: int
    hole_offset_x: float
    hole_offset_y: float
    hole_offset_z: float
    seed: int
    output_dir: Path


def _resolved(path: Path) -> str:
    return str(path.expanduser().resolve())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_task_points(path: Path) -> tuple[TaskPoint, ...]:
    """Load an immutable, consecutively indexed x/y/z point set in metres."""

    if not path.is_file():
        raise FileNotFoundError(f"task-points CSV does not exist: {path}")
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"task-points CSV has no header: {path}")
        required = {"point_index", "hole_offset_x", "hole_offset_z"}
        missing = sorted(required - set(reader.fieldnames))
        if missing:
            raise ValueError(
                "task-points CSV missing required column(s): " + ", ".join(missing)
            )
        rows = list(reader)
    if not rows:
        raise ValueError(f"task-points CSV contains no points: {path}")

    points: list[TaskPoint] = []
    seen_coordinates: set[tuple[float, float, float]] = set()
    for expected_index, row in enumerate(rows, start=1):
        try:
            numeric_index = float(row["point_index"])
            point_index = int(numeric_index)
            x = float(row["hole_offset_x"])
            z = float(row["hole_offset_z"])
            raw_y = row.get("hole_offset_y", "")
            y = 0.0 if raw_y is None or not raw_y.strip() else float(raw_y)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"task-points CSV row {expected_index + 1} contains invalid values"
            ) from error
        if numeric_index != point_index or point_index != expected_index:
            raise ValueError(
                "task-points CSV point_index must be consecutive integers starting at 1"
            )
        coordinates = (x, y, z)
        if not all(math.isfinite(value) for value in coordinates):
            raise ValueError(
                f"task-points CSV row {expected_index + 1} contains non-finite offsets"
            )
        if coordinates in seen_coordinates:
            raise ValueError(
                f"task-points CSV row {expected_index + 1} duplicates an earlier point"
            )
        seen_coordinates.add(coordinates)
        points.append(TaskPoint(point_index, x, y, z))
    return tuple(points)


def select_points(
    points: Sequence[TaskPoint], requested_indices: Optional[Sequence[int]]
) -> tuple[TaskPoint, ...]:
    if not requested_indices:
        return tuple(points)
    if len(set(requested_indices)) != len(requested_indices):
        raise ValueError("point indices must not contain duplicates")
    by_index = {point.point_index: point for point in points}
    unknown = sorted(set(requested_indices) - set(by_index))
    if unknown:
        raise ValueError(f"unknown point indices: {', '.join(map(str, unknown))}")
    requested = set(requested_indices)
    return tuple(point for point in points if point.point_index in requested)


def build_run_matrix(
    points: Sequence[TaskPoint],
    specs: Sequence[PilotSpec],
    output_dir: Path,
    seed_base: int,
) -> tuple[PlannedRun, ...]:
    """Interleave configurations within each point for a paired protocol."""

    runs = []
    for point in points:
        seed = int(seed_base) + point.point_index - 1
        for spec in specs:
            if spec.signed_decay is None:
                raise ValueError("Fibonacci protocol requires signed-temporal specs")
            runs.append(
                PlannedRun(
                    configuration_id=spec.configuration_id,
                    model_id=spec.model_id,
                    executor_id=spec.executor_id,
                    signed_decay=float(spec.signed_decay),
                    point_index=point.point_index,
                    hole_offset_x=point.hole_offset_x,
                    hole_offset_y=point.hole_offset_y,
                    hole_offset_z=point.hole_offset_z,
                    seed=seed,
                    output_dir=(
                        output_dir
                        / spec.configuration_id
                        / f"point_{point.point_index:03d}"
                    ),
                )
            )
    return tuple(runs)


def _point_args(args: argparse.Namespace, run: PlannedRun) -> argparse.Namespace:
    values = vars(args).copy()
    values.update(
        output_dir=run.output_dir.parent.parent,
        hole_offset_x=run.hole_offset_x,
        hole_offset_y=run.hole_offset_y,
        hole_offset_z=run.hole_offset_z,
        seed=run.seed,
        save_videos=False,
        save_force_hud_videos=False,
        force_hud_camera=DEFAULT_FORCE_HUD_CAMERA,
        force_hud_width=DEFAULT_FORCE_HUD_WIDTH,
        force_hud_height=DEFAULT_FORCE_HUD_HEIGHT,
        force_hud_primary_wrench="compensated",
    )
    return argparse.Namespace(**values)


def _replace_option(command: list[str], option: str, value: str) -> None:
    try:
        index = command.index(option)
    except ValueError as error:
        raise ValueError(f"generated rollout command is missing {option}") from error
    command[index + 1] = value


def build_point_rollout_command(
    args: argparse.Namespace,
    spec: PilotSpec,
    run: PlannedRun,
) -> list[str]:
    point_args = _point_args(args, run)
    command = build_rollout_command(point_args, spec)
    _replace_option(command, "--output-dir", str(run.output_dir))
    return command


def validate_run_summary(
    summary: dict[str, Any],
    args: argparse.Namespace,
    spec: PilotSpec,
    run: PlannedRun,
) -> list[str]:
    errors = validate_completed_summary(summary, _point_args(args, run), spec)
    expected_output = _resolved(run.output_dir)
    actual_output = summary.get("output_dir")
    if actual_output is not None and actual_output != expected_output:
        errors.append(
            f"output_dir: expected {expected_output!r}, got {actual_output!r}"
        )
    return errors


def _protocol_payload(
    args: argparse.Namespace,
    point_set_sha256: str,
    all_points: Sequence[TaskPoint],
    selected_points: Sequence[TaskPoint],
    specs: Sequence[PilotSpec],
) -> dict[str, Any]:
    return {
        "experiment_version": EXPERIMENT_VERSION,
        "point_set": {
            "path": _resolved(args.task_points_csv),
            "sha256": point_set_sha256,
            "total_points": len(all_points),
            "selected_point_indices": [point.point_index for point in selected_points],
        },
        "models": {
            "official_act": _resolved(args.official_checkpoint),
            "highrate_contact_v3": _resolved(args.contact_checkpoint),
        },
        "model_xml": _resolved(args.model_xml),
        "configuration_ids": [spec.configuration_id for spec in specs],
        "signed_decays": [spec.signed_decay for spec in specs[:: len(args.selected_model_ids)]],
        "selected_model_ids": list(args.selected_model_ids),
        "seed_contract": {
            "seed_base": args.seed_base,
            "formula": "seed_base + point_index - 1",
            "paired_across_all_configurations": True,
        },
        "fairness_contract": {
            "action_mode": "joint_pos",
            "action_select_mode": "signed_temporal",
            "policy_query_interval": 1,
            "policy_rate_hz": 30.0,
            "max_rollout_steps": args.max_rollout_steps,
            "ema_alpha": 1.0,
            "max_delta_q": 0.02,
            "force_stop_threshold": 100.0,
            "safe_force_threshold": 40.0,
            "success_distance_threshold": 0.003,
            "success_dwell_time": 0.1,
            "contact_latent_mode": "zero",
            "axial_push_enabled": False,
            "videos_saved": False,
        },
    }


def _protocol_fingerprint(protocol: dict[str, Any]) -> str:
    encoded = json.dumps(protocol, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def write_or_validate_plan(
    path: Path,
    protocol: dict[str, Any],
    runs: Sequence[PlannedRun],
) -> dict[str, Any]:
    fingerprint = _protocol_fingerprint(protocol)
    if path.is_file():
        existing = json.loads(path.read_text())
        if existing.get("protocol_fingerprint") != fingerprint:
            raise ValueError(
                "existing experiment plan has a different protocol; use a new output directory"
            )
        return existing
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_fingerprint": fingerprint,
        "protocol": protocol,
        "planned_runs": len(runs),
        "runs": [
            {**asdict(run), "output_dir": _resolved(run.output_dir)} for run in runs
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def _spec_by_id(specs: Sequence[PilotSpec]) -> dict[str, PilotSpec]:
    return {spec.configuration_id: spec for spec in specs}


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-points-csv", type=Path, default=DEFAULT_POINT_SET)
    parser.add_argument("--official-checkpoint", type=Path, default=DEFAULT_OFFICIAL_CHECKPOINT)
    parser.add_argument("--contact-checkpoint", type=Path, default=DEFAULT_CONTACT_CHECKPOINT)
    parser.add_argument("--model-xml", type=Path, default=DEFAULT_MODEL_XML)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    parser.add_argument("--seed-base", type=int, default=0)
    parser.add_argument("--max-rollout-steps", type=int, default=600)
    parser.add_argument("--signed-decay", action="append", type=float, dest="signed_decays")
    parser.add_argument("--model-id", action="append", choices=MODEL_IDS, dest="model_ids")
    parser.add_argument("--point-index", action="append", type=int, dest="point_indices")
    parser.add_argument("--execute-rollouts", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.max_rollout_steps <= 0:
        print("error: --max-rollout-steps must be positive", file=sys.stderr)
        return 2
    signed_decays = tuple(args.signed_decays or DEFAULT_SIGNED_DECAYS)
    if any(not math.isfinite(value) for value in signed_decays):
        print("error: signed decays must be finite", file=sys.stderr)
        return 2
    args.selected_model_ids = tuple(args.model_ids or MODEL_IDS)
    required = [args.task_points_csv, args.model_xml]
    if "official_act" in args.selected_model_ids:
        required.append(args.official_checkpoint)
    if "highrate_contact_v3" in args.selected_model_ids:
        required.append(args.contact_checkpoint)
    for path in required:
        if not path.expanduser().resolve().is_file():
            print(f"error: required file does not exist: {path}", file=sys.stderr)
            return 2
    for name in ("task_points_csv", "official_checkpoint", "contact_checkpoint", "model_xml", "output_dir"):
        setattr(args, name, getattr(args, name).expanduser().resolve())

    try:
        all_points = read_task_points(args.task_points_csv)
        points = select_points(all_points, args.point_indices)
        specs = build_signed_decay_specs(
            args.official_checkpoint,
            args.contact_checkpoint,
            signed_decays,
            args.selected_model_ids,
        )
        runs = build_run_matrix(points, specs, args.output_dir, args.seed_base)
        protocol = _protocol_payload(
            args, _sha256(args.task_points_csv), all_points, points, specs
        )
        plan = write_or_validate_plan(
            args.output_dir / "experiment_plan.json", protocol, runs
        )
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    print(f"experiment_plan={args.output_dir / 'experiment_plan.json'}")
    print(f"protocol_fingerprint={plan['protocol_fingerprint']}")
    print(f"selected_points={len(points)}/{len(all_points)}")
    print(f"selected_configurations={len(specs)}")
    print(f"planned_rollouts={len(runs)}")
    if not args.execute_rollouts:
        print("status=PLAN_ONLY (pass --execute-rollouts to run)")
        return 0

    specs_by_id = _spec_by_id(specs)
    completed = 0
    failed = False
    for index, run in enumerate(runs, start=1):
        spec = specs_by_id[run.configuration_id]
        summary_path = run.output_dir / "summary.json"
        if summary_path.is_file() and args.skip_existing:
            summary = json.loads(summary_path.read_text())
            errors = validate_run_summary(summary, args, spec, run)
            if errors:
                print(
                    f"error: existing run failed contract: {run.output_dir}: "
                    + "; ".join(errors),
                    file=sys.stderr,
                )
                return 2
            completed += 1
            print(f"skipped_complete={run.configuration_id}/point_{run.point_index:03d}")
            continue
        if run.output_dir.exists() and any(run.output_dir.iterdir()):
            print(
                f"error: output directory is non-empty: {run.output_dir}",
                file=sys.stderr,
            )
            return 2
        print(
            f"running={index}/{len(runs)} {run.configuration_id}/point_{run.point_index:03d}",
            flush=True,
        )
        result = subprocess.run(
            build_point_rollout_command(args, spec, run), check=False
        )
        if result.returncode != 0:
            failed = True
            print(
                f"error: rollout exited with {result.returncode}: {run.output_dir}",
                file=sys.stderr,
            )
            if not args.continue_on_error:
                return result.returncode
            continue
        if not summary_path.is_file():
            print(f"error: missing rollout summary: {summary_path}", file=sys.stderr)
            return 2
        summary = json.loads(summary_path.read_text())
        errors = validate_run_summary(summary, args, spec, run)
        if errors:
            print(
                f"error: completed run failed contract: {run.output_dir}: "
                + "; ".join(errors),
                file=sys.stderr,
            )
            return 2
        completed += 1
        print(f"completed={run.configuration_id}/point_{run.point_index:03d}")
    print(f"completed_rollouts={completed}/{len(runs)}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
