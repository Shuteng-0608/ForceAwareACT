#!/usr/bin/env python3
"""Monitor filesystem progress for an x/z multi-seed rollout suite."""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_xz_rollout_suite import (
    ACTION_SELECT_MODES,
    MODEL_SPECS,
    output_dir_for,
    selected_models,
)


COMPLETED_RUN_STATUSES = {"success", "task_failed", "skipped_existing"}
POINT_PATTERN = re.compile(r"point_(\d+)_")


@dataclass(frozen=True)
class ExperimentProgress:
    point_set_seed: int
    rollout_seed_base: int
    model_key: str
    action_select_mode: str
    root: Path
    status: str
    completed_points: int
    attempted_points: int
    process_errors: int
    current_point: int | None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Report the active model, seed configuration, point index, "
            "completed seed configurations, and queued work."
        )
    )
    parser.add_argument("--output-base", type=Path, required=True)
    parser.add_argument("--point-set-seeds", nargs="+", type=int)
    parser.add_argument("--rollout-seed-bases", nargs="+", type=int)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=[spec.key for spec in MODEL_SPECS],
    )
    parser.add_argument(
        "--action-select-modes",
        nargs="+",
        choices=ACTION_SELECT_MODES,
    )
    parser.add_argument("--num-points", type=int)
    parser.add_argument("--offset-mm", type=float)
    parser.add_argument("--max-rollout-steps", type=int, default=900)
    parser.add_argument("--max-delta-q", type=float, default=0.02)
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Refresh continuously until interrupted with Ctrl-C.",
    )
    parser.add_argument("--interval", type=float, default=10.0)
    return parser


def _manual_plan(args: argparse.Namespace) -> dict[str, object]:
    required = {
        "--point-set-seeds": args.point_set_seeds,
        "--rollout-seed-bases": args.rollout_seed_bases,
        "--models": args.models,
        "--num-points": args.num_points,
        "--offset-mm": args.offset_mm,
    }
    missing = [flag for flag, value in required.items() if value is None]
    if missing:
        raise ValueError(
            "suite_plan.json was not found; provide " + ", ".join(missing)
        )
    if args.num_points <= 0:
        raise ValueError("--num-points must be positive")
    if args.interval <= 0:
        raise ValueError("--interval must be positive")
    modes = args.action_select_modes or ["mid"]
    models = [spec.key for spec in selected_models(args.models)]
    configurations = []
    for point_set_seed in args.point_set_seeds:
        for rollout_seed_base in args.rollout_seed_bases:
            configurations.append(
                {
                    "point_set_seed": point_set_seed,
                    "rollout_seed_base": rollout_seed_base,
                    "output_base": str(
                        args.output_base
                        / f"pointset_{point_set_seed}"
                        / f"rollout_{rollout_seed_base}"
                    ),
                }
            )
    return {
        "schema_version": 1,
        "output_base": str(args.output_base),
        "point_set_seeds": list(args.point_set_seeds),
        "rollout_seed_bases": list(args.rollout_seed_bases),
        "seed_configurations": configurations,
        "models": models,
        "action_select_modes": modes,
        "num_points": args.num_points,
        "offset_mm": args.offset_mm,
        "max_rollout_steps": args.max_rollout_steps,
        "max_delta_q": args.max_delta_q,
    }


def load_plan(args: argparse.Namespace) -> tuple[dict[str, object], Path | None]:
    plan_path = args.output_base / "suite_plan.json"
    if plan_path.is_file():
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        return plan, plan_path
    return _manual_plan(args), None


def running_command_lines() -> list[str]:
    result = subprocess.run(
        ["ps", "-eo", "args="],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return []
    return [
        line
        for line in result.stdout.splitlines()
        if "run_mujoco_hole_grid.py" in line
        or "run_mujoco_policy_rollout.py" in line
    ]


def _read_manifest(path: Path) -> dict[str, object] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _csv_row_count(path: Path) -> int:
    try:
        with path.open(newline="") as csv_file:
            return sum(1 for _ in csv.DictReader(csv_file))
    except OSError:
        return 0


def inspect_experiment(
    *,
    point_set_seed: int,
    rollout_seed_base: int,
    model_key: str,
    action_select_mode: str,
    root: Path,
    num_points: int,
    process_lines: Sequence[str],
) -> ExperimentProgress:
    summary_rows = _csv_row_count(root / "grid_summary.csv")
    manifest = _read_manifest(root / "grid_manifest.json") or {}
    runs = manifest.get("runs", [])
    if not isinstance(runs, list):
        runs = []
    completed = sum(
        1 for run in runs if isinstance(run, dict) and run.get("status") in COMPLETED_RUN_STATUSES
    )
    process_errors = sum(
        1 for run in runs if isinstance(run, dict) and run.get("status") == "process_error"
    )
    attempted = len(runs)
    matching_processes = [line for line in process_lines if str(root) in line]
    is_running = bool(matching_processes)
    has_started = bool(runs) or (root / "task_points.csv").is_file() or root.with_name(
        f"{root.name}_console.log"
    ).is_file()

    if summary_rows >= num_points and process_errors == 0:
        status = "complete"
        current_point = None
        completed = num_points
        attempted = max(attempted, num_points)
    elif is_running:
        status = "running"
        point_matches = [
            int(match.group(1))
            for line in matching_processes
            for match in [POINT_PATTERN.search(line)]
            if match is not None
        ]
        current_point = point_matches[0] if point_matches else min(attempted + 1, num_points)
    elif has_started:
        status = "partial"
        current_point = None
    else:
        status = "queued"
        current_point = None

    return ExperimentProgress(
        point_set_seed=point_set_seed,
        rollout_seed_base=rollout_seed_base,
        model_key=model_key,
        action_select_mode=action_select_mode,
        root=root,
        status=status,
        completed_points=completed,
        attempted_points=attempted,
        process_errors=process_errors,
        current_point=current_point,
    )


def collect_progress(
    plan: dict[str, object],
    process_lines: Sequence[str],
) -> list[ExperimentProgress]:
    num_points = int(plan["num_points"])
    offset_mm = float(plan["offset_mm"])
    max_rollout_steps = int(plan["max_rollout_steps"])
    max_delta_q = float(plan["max_delta_q"])
    models = selected_models(plan["models"])
    modes = list(plan["action_select_modes"])
    progress = []
    for configuration in plan["seed_configurations"]:
        point_set_seed = int(configuration["point_set_seed"])
        rollout_seed_base = int(configuration["rollout_seed_base"])
        configuration_base = Path(configuration["output_base"])
        for model in models:
            for mode in modes:
                root = output_dir_for(
                    model,
                    mode,
                    configuration_base,
                    num_points=num_points,
                    offset_mm=offset_mm,
                    max_rollout_steps=max_rollout_steps,
                    max_delta_q=max_delta_q,
                )
                progress.append(
                    inspect_experiment(
                        point_set_seed=point_set_seed,
                        rollout_seed_base=rollout_seed_base,
                        model_key=model.key,
                        action_select_mode=mode,
                        root=root,
                        num_points=num_points,
                        process_lines=process_lines,
                    )
                )
    return progress


def _configuration_rows(
    progress: Sequence[ExperimentProgress],
) -> list[tuple[int, int, str, int, int]]:
    keys = []
    for item in progress:
        key = (item.point_set_seed, item.rollout_seed_base)
        if key not in keys:
            keys.append(key)
    rows = []
    for point_seed, rollout_seed in keys:
        group = [
            item
            for item in progress
            if item.point_set_seed == point_seed
            and item.rollout_seed_base == rollout_seed
        ]
        complete = sum(item.status == "complete" for item in group)
        running = next((item for item in group if item.status == "running"), None)
        if complete == len(group):
            status = "complete"
        elif running is not None:
            status = "running"
        elif any(item.status == "partial" for item in group) or complete:
            status = "partial"
        else:
            status = "queued"
        rows.append((point_seed, rollout_seed, status, complete, len(group)))
    return rows


def render_report(
    plan: dict[str, object],
    progress: Sequence[ExperimentProgress],
    plan_path: Path | None,
) -> str:
    num_points = int(plan["num_points"])
    total_points = len(progress) * num_points
    completed_points = sum(item.completed_points for item in progress)
    running = [item for item in progress if item.status == "running"]
    config_rows = _configuration_rows(progress)
    complete_configs = [row for row in config_rows if row[2] == "complete"]
    queued_configs = [row for row in config_rows if row[2] == "queued"]
    partial_configs = [row for row in config_rows if row[2] == "partial"]

    lines = [
        "=" * 88,
        "ForceAwareACT x/z Rollout Suite Monitor",
        "=" * 88,
        f"time: {datetime.now():%Y-%m-%d %H:%M:%S}",
        f"output_base: {plan['output_base']}",
        f"plan: {plan_path if plan_path is not None else 'CLI fallback'}",
        f"overall_points: {completed_points}/{total_points} "
        f"({100.0 * completed_points / max(total_points, 1):.1f}%)",
        f"seed_configurations: complete={len(complete_configs)} "
        f"partial={len(partial_configs)} running={sum(row[2] == 'running' for row in config_rows)} "
        f"queued={len(queued_configs)} total={len(config_rows)}",
    ]
    if running:
        lines.append("current:")
        for item in running:
            lines.append(
                "  "
                f"point_set_seed={item.point_set_seed} "
                f"rollout_seed_base={item.rollout_seed_base} "
                f"model={item.model_key} mode={item.action_select_mode} "
                f"point={item.current_point}/{num_points} "
                f"completed={item.completed_points}/{num_points}"
            )
    else:
        lines.append("current: no matching rollout process detected")

    lines.extend(
        [
            "-" * 88,
            f"{'point-set':>12} {'rollout-base':>14} {'status':>10} "
            f"{'experiments':>12}  detail",
        ]
    )
    for point_seed, rollout_seed, status, complete, total in config_rows:
        active = next(
            (
                item
                for item in progress
                if item.point_set_seed == point_seed
                and item.rollout_seed_base == rollout_seed
                and item.status == "running"
            ),
            None,
        )
        detail = "-"
        if active is not None:
            detail = (
                f"{active.model_key}:{active.action_select_mode} "
                f"point {active.current_point}/{num_points}"
            )
        elif status == "partial":
            detail = "interrupted or awaiting resume"
        lines.append(
            f"{point_seed:>12} {rollout_seed:>14} {status:>10} "
            f"{complete:>5}/{total:<6}  {detail}"
        )

    if complete_configs:
        lines.append(
            "completed: "
            + ", ".join(f"({row[0]},{row[1]})" for row in complete_configs)
        )
    if queued_configs:
        lines.append(
            "queued: "
            + ", ".join(f"({row[0]},{row[1]})" for row in queued_configs)
        )
    if partial_configs:
        lines.append(
            "partial: "
            + ", ".join(f"({row[0]},{row[1]})" for row in partial_configs)
        )
    lines.append("=" * 88)
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.interval <= 0:
        raise ValueError("--interval must be positive")
    plan, plan_path = load_plan(args)
    try:
        while True:
            progress = collect_progress(plan, running_command_lines())
            print(render_report(plan, progress, plan_path), flush=True)
            if not args.watch:
                return 0
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("monitor stopped")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
