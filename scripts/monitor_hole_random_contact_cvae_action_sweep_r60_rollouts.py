#!/usr/bin/env python3
"""Monitor the Contact-CVAE 1..10 plus temporal fixed-point action sweep."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Sequence


DEFAULT_OUTPUT_ROOT = Path(
    "outputs/hole_random_60mm_hmj/earlystop_train90_splitseed20260716_run1/"
    "rollouts/fibonacci_disk_100_r60mm_contact_action_sweep"
)
DEFAULT_MODES = tuple(str(index) for index in range(1, 11)) + ("temporal",)
DEFAULT_MODELS = ("contact_cvae_zero", "contact_cvae_prior")
TERMINAL_STATUSES = {"completed", "completed_with_errors", "failed", "interrupted"}


@dataclass(frozen=True)
class Experiment:
    key: str
    model: str
    mode: str
    output_dir: Path


@dataclass(frozen=True)
class Progress:
    experiment: Experiment
    status: str
    attempted: int
    valid: int
    successes: int
    safe_successes: int
    process_errors: int
    force_stops: int
    maximum_force: float | None
    timed_attempts: int
    elapsed_seconds: float


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=float, default=10.0)
    parser.add_argument("--stale-after", type=float, default=900.0)
    parser.add_argument("--eta-min-points", type=int, default=3)
    parser.add_argument("--show-log-tail", type=int, default=1)
    return parser


def mode_token(mode: str) -> str:
    return "temporal" if mode == "temporal" else f"action_{int(mode):02d}"


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def read_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def read_status(path: Path) -> str | None:
    lines = read_text(path).splitlines()
    return lines[0].split("\t", 1)[0].strip() if lines else None


def parse_datetime(value: object) -> datetime | None:
    if not value:
        return None
    try:
        result = datetime.fromisoformat(str(value))
    except ValueError:
        return None
    return result.astimezone()


def bool_value(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def process_is_running(path: Path) -> bool:
    try:
        os.kill(int(read_text(path).strip()), 0)
    except (OSError, ValueError):
        return False
    return True


def recently_modified(path: Path, now: datetime, stale_after: float) -> bool:
    try:
        modified = datetime.fromtimestamp(path.stat().st_mtime).astimezone()
    except OSError:
        return False
    age = (now - modified).total_seconds()
    return 0 <= age <= stale_after


def experiments_from_plan(output_root: Path) -> list[Experiment]:
    raw = read_json(output_root / "suite_plan.json").get("experiments", [])
    experiments: list[Experiment] = []
    if isinstance(raw, list):
        for item in raw:
            if not isinstance(item, dict):
                continue
            try:
                experiments.append(
                    Experiment(
                        key=str(item["key"]),
                        model=str(item["model_key"]),
                        mode=str(item["action_select_mode"]),
                        output_dir=Path(str(item["output_dir"])),
                    )
                )
            except KeyError:
                continue
    if experiments:
        return experiments
    return [
        Experiment(
            key=f"{model}__{mode_token(mode)}",
            model=model,
            mode=mode,
            output_dir=output_root / model / mode_token(mode),
        )
        for mode in DEFAULT_MODES
        for model in DEFAULT_MODELS
    ]


def inspect(
    experiment: Experiment,
    pipeline_dir: Path,
    current_experiment: str | None,
    runner_active: bool,
    safe_force_threshold: float,
) -> Progress:
    manifest = read_json(experiment.output_dir / "grid_manifest.json")
    raw_runs = manifest.get("runs", [])
    runs = raw_runs if isinstance(raw_runs, list) else []
    summaries = [
        summary
        for path in sorted(experiment.output_dir.glob("point_*/summary.json"))
        if (summary := read_json(path))
    ]
    successes = sum(bool_value(item.get("success")) for item in summaries)
    safe_successes = sum(
        bool_value(item.get("success"))
        and float(item.get("max_force_norm", float("inf"))) < safe_force_threshold
        for item in summaries
    )
    forces = [
        float(item["max_force_norm"])
        for item in summaries
        if item.get("max_force_norm") is not None
    ]
    process_errors = sum(
        isinstance(run, dict) and run.get("status") == "process_error" for run in runs
    )
    force_stops = sum(
        item.get("stop_reason") == "force_stop_threshold" for item in summaries
    )
    elapsed_seconds = 0.0
    timed_attempts = 0
    for run in runs:
        if not isinstance(run, dict):
            continue
        start = parse_datetime(run.get("start_time"))
        end = parse_datetime(run.get("end_time"))
        if start is not None and end is not None and end > start:
            elapsed_seconds += (end - start).total_seconds()
            timed_attempts += 1

    status = read_status(pipeline_dir / f"{experiment.key}.status")
    if status is None:
        if len(summaries) == 100 and process_errors == 0:
            status = "complete-unmarked"
        elif runs or summaries:
            status = "partial"
        else:
            status = "queued"
    elif status == "completed":
        status = "complete"
    elif status == "running" and not (
        runner_active and current_experiment == experiment.key
    ):
        status = "stale/partial"
    return Progress(
        experiment=experiment,
        status=status,
        attempted=len(runs),
        valid=len(summaries),
        successes=successes,
        safe_successes=safe_successes,
        process_errors=process_errors,
        force_stops=force_stops,
        maximum_force=max(forces) if forces else None,
        timed_attempts=timed_attempts,
        elapsed_seconds=elapsed_seconds,
    )


def format_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    days, remainder = divmod(seconds, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds = divmod(remainder, 60)
    prefix = f"{days}d " if days else ""
    return f"{prefix}{hours:02d}:{minutes:02d}:{seconds:02d}"


def gpu_report() -> list[str]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return ["nvidia-smi unavailable"]
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def log_tail(path: Path, count: int) -> list[str]:
    lines = [line for line in read_text(path).splitlines() if line.strip()]
    return lines[-count:] if count > 0 else []


def render(
    output_root: Path,
    pipeline_status: str,
    runner_state: str,
    current_experiment: str | None,
    grid_pid: str | None,
    progress: Sequence[Progress],
    now: datetime,
    eta_min_points: int,
    log_lines: Sequence[str],
) -> str:
    total = len(progress) * 100
    attempted = sum(item.attempted for item in progress)
    valid = sum(item.valid for item in progress)
    timed = sum(item.timed_attempts for item in progress)
    elapsed = sum(item.elapsed_seconds for item in progress)
    complete = sum(item.status in {"complete", "complete-unmarked"} for item in progress)
    lines = [
        "=" * 148,
        "ForceAwareACT Contact-CVAE Action-Chunk Sweep Monitor (1..10 + temporal)",
        "=" * 148,
        f"time: {now:%Y-%m-%d %H:%M:%S %Z}",
        f"output_root: {output_root}",
        f"pipeline: {pipeline_status} | runner: {runner_state} | experiments: {complete}/{len(progress)}",
        f"current_experiment: {current_experiment or '-'} | grid_pid: {grid_pid or '-'}",
        f"overall_attempts: {attempted}/{total} ({attempted / total * 100 if total else 0:.1f}%) | valid_summaries: {valid}/{total}",
    ]
    if pipeline_status == "completed":
        lines.append("ETA: complete")
    elif pipeline_status in {"failed", "interrupted", "completed_with_errors"}:
        lines.append(f"ETA: unavailable because pipeline is {pipeline_status}")
    elif timed < eta_min_points or elapsed <= 0:
        lines.append(f"ETA: collecting timing; available after {eta_min_points} completed attempts")
    else:
        seconds_per_point = elapsed / timed
        remaining = max(total - attempted, 0)
        eta_seconds = seconds_per_point * remaining
        finish = now + timedelta(seconds=eta_seconds)
        lines.append(
            f"ETA basis: {seconds_per_point:.2f} s/point from {timed} attempts | "
            f"remaining {format_duration(eta_seconds)} -> {finish:%Y-%m-%d %H:%M:%S %Z}"
        )
    lines.extend(
        [
            "-" * 148,
            f"{'model':<22}{'mode':<11}{'status':<16}{'attempts':<11}{'valid':<9}{'success':<12}{'safe<40N':<12}{'errors':<9}{'force_stop':<12}{'maxF(N)':<10}",
        ]
    )
    for item in progress:
        success = f"{item.successes}/{item.valid}" if item.valid else "-"
        safe = f"{item.safe_successes}/{item.valid}" if item.valid else "-"
        maximum_force = "-" if item.maximum_force is None else f"{item.maximum_force:.1f}"
        lines.append(
            f"{item.experiment.model:<22}{item.experiment.mode:<11}{item.status:<16}"
            f"{f'{item.attempted}/100':<11}{f'{item.valid}/100':<9}{success:<12}"
            f"{safe:<12}{item.process_errors:<9}{item.force_stops:<12}{maximum_force:<10}"
        )
    lines.extend(["-" * 148, "GPU:"])
    lines.extend(f"  {line}" for line in gpu_report())
    if log_lines:
        lines.extend(["-" * 148, "Log tail:"])
        lines.extend(f"  {line}" for line in log_lines)
    lines.append("=" * 148)
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.interval <= 0 or args.stale_after <= 0 or args.eta_min_points <= 0:
        raise ValueError("interval, stale threshold and ETA minimum must be positive")
    if args.show_log_tail < 0:
        raise ValueError("--show-log-tail must be non-negative")
    output_root = args.output_root.expanduser().resolve()
    pipeline_dir = output_root / ".pipeline"
    try:
        while True:
            now = datetime.now().astimezone()
            pipeline_status = read_status(pipeline_dir / "pipeline.status") or "not-started"
            current = read_text(pipeline_dir / "current_experiment").strip() or None
            grid_pid = read_text(pipeline_dir / "current_grid.pid").strip() or None
            pid_visible = process_is_running(pipeline_dir / "runner.pid")
            plan_experiments = experiments_from_plan(output_root)
            current_output = next(
                (item.output_dir for item in plan_experiments if item.key == current), None
            )
            activity_paths = [pipeline_dir / "runner.log"]
            if current_output is not None:
                activity_paths.extend(
                    [current_output / "console.log", current_output / "grid_manifest.json"]
                )
            recent = pipeline_status == "running" and any(
                recently_modified(path, now, args.stale_after) for path in activity_paths
            )
            runner_active = pid_visible or recent
            runner_state = (
                "RUNNING"
                if pid_visible
                else "ACTIVE (recent files; PID not visible)"
                if recent
                else "NOT DETECTED"
            )
            plan = read_json(output_root / "suite_plan.json")
            protocol = plan.get("protocol", {})
            force_threshold = (
                float(protocol.get("success_force_threshold", 40.0))
                if isinstance(protocol, dict)
                else 40.0
            )
            progress = [
                inspect(
                    experiment,
                    pipeline_dir,
                    current,
                    runner_active,
                    force_threshold,
                )
                for experiment in plan_experiments
            ]
            current_log = (
                current_output / "console.log"
                if current_output is not None
                else pipeline_dir / "runner.log"
            )
            print(
                ("\033[2J\033[H" if args.watch else "")
                + render(
                    output_root,
                    pipeline_status,
                    runner_state,
                    current,
                    grid_pid,
                    progress,
                    now,
                    args.eta_min_points,
                    log_tail(current_log, args.show_log_tail),
                ),
                flush=True,
            )
            if not args.watch or pipeline_status in TERMINAL_STATUSES:
                return 0
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nmonitor stopped; rollout suite was not interrupted")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
