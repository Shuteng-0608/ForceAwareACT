#!/usr/bin/env python3
"""Monitor the fixed 60 mm Fibonacci rollout suite and estimate completion."""

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
    "outputs/hole_random_60mm_hmj/"
    "earlystop_train90_splitseed20260716_run1/"
    "rollouts/fibonacci_disk_100_r60mm_mid"
)
MODEL_KEYS = (
    "contact_cvae_zero",
    "contact_cvae_prior",
    "motion_cvae",
    "dualzero",
    "act_baseline",
)
TERMINAL_PIPELINE_STATUSES = {
    "completed",
    "completed_with_errors",
    "failed",
    "interrupted",
}


@dataclass(frozen=True)
class RolloutProgress:
    key: str
    status: str
    attempted: int
    valid: int
    successes: int
    safe_successes: int
    process_errors: int
    force_stops: int
    maximum_force: float | None
    current_point: int | None
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
    line = read_text(path).splitlines()
    if not line:
        return None
    return line[0].split("\t", 1)[0].strip() or None


def parse_datetime(value: object) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value))
    except ValueError:
        return None
    return parsed.astimezone() if parsed.tzinfo is not None else parsed.astimezone()


def process_is_running(pid_file: Path) -> bool:
    try:
        pid = int(read_text(pid_file).strip())
        os.kill(pid, 0)
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


def bool_value(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def inspect_model(
    output_root: Path,
    pipeline_dir: Path,
    key: str,
    current_model: str | None,
    runner_active: bool,
    success_force_threshold: float,
) -> RolloutProgress:
    root = output_root / key
    manifest = read_json(root / "grid_manifest.json")
    raw_runs = manifest.get("runs", [])
    runs = raw_runs if isinstance(raw_runs, list) else []
    summaries = []
    for path in sorted(root.glob("point_*/summary.json")):
        summary = read_json(path)
        if summary:
            summaries.append(summary)
    successes = sum(bool_value(summary.get("success")) for summary in summaries)
    safe_successes = sum(
        bool_value(summary.get("success"))
        and float(summary.get("max_force_norm", float("inf"))) < success_force_threshold
        for summary in summaries
    )
    force_stops = sum(
        summary.get("stop_reason") == "force_stop_threshold"
        for summary in summaries
    )
    forces = [
        float(summary["max_force_norm"])
        for summary in summaries
        if summary.get("max_force_norm") is not None
    ]
    process_errors = sum(
        isinstance(run, dict) and run.get("status") == "process_error"
        for run in runs
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

    status = read_status(pipeline_dir / f"{key}.status")
    if status is None:
        if len(summaries) == 100 and process_errors == 0:
            status = "complete-unmarked"
        elif runs or summaries:
            status = "partial"
        else:
            status = "queued"
    elif status == "completed":
        status = "complete"
    elif status == "running" and not (runner_active and current_model == key):
        status = "stale/partial"

    current_point = None
    if status == "running":
        current_point = min(len(runs) + 1, 100)
    return RolloutProgress(
        key=key,
        status=status,
        attempted=len(runs),
        valid=len(summaries),
        successes=successes,
        safe_successes=safe_successes,
        process_errors=process_errors,
        force_stops=force_stops,
        maximum_force=max(forces) if forces else None,
        current_point=current_point,
        timed_attempts=timed_attempts,
        elapsed_seconds=elapsed_seconds,
    )


def format_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    days, remainder = divmod(seconds, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds = divmod(remainder, 60)
    if days:
        return f"{days}d {hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def format_force(value: float | None) -> str:
    return "-" if value is None else f"{value:.1f}"


def log_tail(path: Path, count: int) -> list[str]:
    if count <= 0:
        return []
    lines = [line for line in read_text(path).splitlines() if line.strip()]
    return lines[-count:]


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
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return lines or ["no GPU information"]


def render(
    *,
    output_root: Path,
    pipeline_status: str,
    runner_state: str,
    grid_pid: str | None,
    current_model: str | None,
    progress: Sequence[RolloutProgress],
    now: datetime,
    eta_min_points: int,
    log_lines: Sequence[str],
) -> str:
    attempted = sum(item.attempted for item in progress)
    valid = sum(item.valid for item in progress)
    total_timed = sum(item.timed_attempts for item in progress)
    total_seconds = sum(item.elapsed_seconds for item in progress)
    completed_models = sum(item.status in {"complete", "complete-unmarked"} for item in progress)
    lines = [
        "=" * 132,
        "ForceAwareACT Five-Model Fixed 60 mm Fibonacci Rollout Monitor",
        "=" * 132,
        f"time: {now:%Y-%m-%d %H:%M:%S %Z}",
        f"output_root: {output_root}",
        f"pipeline: {pipeline_status} | runner: {runner_state} | models: {completed_models}/5",
        f"current_model: {current_model or '-'} | grid_pid: {grid_pid or '-'}",
        f"overall_attempts: {attempted}/500 ({attempted / 5:.1f}%) | valid_summaries: {valid}/500",
    ]
    if pipeline_status == "completed":
        lines.append("ETA: complete")
    elif pipeline_status in {"failed", "interrupted", "completed_with_errors"}:
        lines.append(f"ETA: unavailable because pipeline is {pipeline_status}")
    elif total_timed < eta_min_points or total_seconds <= 0:
        lines.append(f"ETA: collecting timing; available after {eta_min_points} completed attempts")
    else:
        seconds_per_point = total_seconds / total_timed
        remaining = max(500 - attempted, 0)
        eta_seconds = seconds_per_point * remaining
        finish = now + timedelta(seconds=eta_seconds)
        lines.append(
            f"ETA basis: {seconds_per_point:.2f} s/point from {total_timed} timed attempts"
        )
        lines.append(
            f"estimated remaining: {format_duration(eta_seconds)} -> "
            f"{finish:%Y-%m-%d %H:%M:%S %Z} ({remaining} attempts)"
        )

    lines.extend(
        [
            "-" * 132,
            (
                f"{'model':<25}{'status':<15}{'attempts':<11}{'valid':<9}"
                f"{'success':<12}{'safe<40N':<12}{'errors':<9}"
                f"{'force_stop':<12}{'maxF(N)':<10}{'current':<10}"
            ),
        ]
    )
    for item in progress:
        success = f"{item.successes}/{item.valid}" if item.valid else "-"
        safe = f"{item.safe_successes}/{item.valid}" if item.valid else "-"
        current = f"{item.current_point}/100" if item.current_point else "-"
        lines.append(
            f"{item.key:<25}{item.status:<15}{f'{item.attempted}/100':<11}"
            f"{f'{item.valid}/100':<9}{success:<12}{safe:<12}"
            f"{item.process_errors:<9}{item.force_stops:<12}"
            f"{format_force(item.maximum_force):<10}{current:<10}"
        )
    lines.extend(["-" * 132, "GPU:"])
    lines.extend(f"  {line}" for line in gpu_report())
    if log_lines:
        lines.extend(["-" * 132, "Current model log tail:"])
        lines.extend(f"  {line}" for line in log_lines)
    lines.append("=" * 132)
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.interval <= 0 or args.stale_after <= 0:
        raise ValueError("--interval and --stale-after must be positive")
    if args.eta_min_points <= 0 or args.show_log_tail < 0:
        raise ValueError("ETA minimum must be positive and log tail non-negative")
    output_root = args.output_root.expanduser().resolve()
    pipeline_dir = output_root / ".pipeline"
    try:
        while True:
            now = datetime.now().astimezone()
            pipeline_status = read_status(pipeline_dir / "pipeline.status") or "not-started"
            current_model = read_text(pipeline_dir / "current_model").strip() or None
            grid_pid = read_text(pipeline_dir / "current_grid.pid").strip() or None
            pid_visible = process_is_running(pipeline_dir / "runner.pid")
            activity_paths = [pipeline_dir / "runner.log"]
            if current_model:
                activity_paths.extend(
                    [
                        output_root / current_model / "console.log",
                        output_root / current_model / "grid_manifest.json",
                    ]
                )
            recent_activity = pipeline_status == "running" and any(
                recently_modified(path, now, args.stale_after)
                for path in activity_paths
            )
            runner_active = pid_visible or recent_activity
            runner_state = (
                "RUNNING"
                if pid_visible
                else "ACTIVE (recent files; PID not visible)"
                if recent_activity
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
                inspect_model(
                    output_root,
                    pipeline_dir,
                    key,
                    current_model,
                    runner_active,
                    force_threshold,
                )
                for key in MODEL_KEYS
            ]
            current_log = (
                output_root / current_model / "console.log"
                if current_model
                else output_root / ".pipeline/runner.log"
            )
            report = render(
                output_root=output_root,
                pipeline_status=pipeline_status,
                runner_state=runner_state,
                grid_pid=grid_pid,
                current_model=current_model,
                progress=progress,
                now=now,
                eta_min_points=args.eta_min_points,
                log_lines=log_tail(current_log, args.show_log_tail),
            )
            if args.watch:
                print("\033[2J\033[H", end="")
            print(report, flush=True)
            if not args.watch or pipeline_status in TERMINAL_PIPELINE_STATUSES:
                return 0
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nmonitor stopped; rollout suite was not interrupted")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
