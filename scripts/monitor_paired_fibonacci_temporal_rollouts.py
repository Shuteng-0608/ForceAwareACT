#!/usr/bin/env python3
"""Monitor a paired Fibonacci temporal rollout experiment without mutation."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Sequence


def _duration(seconds: Optional[float]) -> str:
    if seconds is None:
        return "collecting completed rollout timing"
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:d}:{minutes:02d}:{secs:02d}"


def _pid_alive(pid: Any) -> bool:
    try:
        return int(pid) > 0 and Path(f"/proc/{int(pid)}").exists()
    except (TypeError, ValueError):
        return False


def read_status(output_dir: Path) -> dict[str, Any]:
    plan_path = output_dir / "experiment_plan.json"
    if not plan_path.is_file():
        raise FileNotFoundError(f"experiment plan is missing: {plan_path}")
    plan = json.loads(plan_path.read_text())
    progress_path = output_dir / "progress.json"
    progress = json.loads(progress_path.read_text()) if progress_path.is_file() else {}
    planned = int(plan["planned_runs"])
    completed = int(progress.get("completed_rollouts", 0))
    status = str(progress.get("status", "PLAN_ONLY"))
    pid = progress.get("runner_pid")
    if status == "RUNNING" and not _pid_alive(pid):
        status = "STALE (runner process is not alive)"
    if completed == planned and planned > 0:
        status = "COMPLETED"
    return {
        "output_dir": str(output_dir.resolve()),
        "protocol_fingerprint": plan["protocol_fingerprint"],
        "status": status,
        "pid": pid,
        "planned": planned,
        "completed": completed,
        "remaining": max(0, planned - completed),
        "percent": 100.0 * completed / planned if planned else 0.0,
        "current_run_id": progress.get("current_run_id"),
        "failed": int(progress.get("failed_rollouts", 0)),
        "average_seconds": progress.get("average_seconds_per_new_rollout"),
        "eta_seconds": progress.get("eta_seconds"),
        "updated_at_utc": progress.get("updated_at_utc"),
    }


def format_status(status: dict[str, Any]) -> str:
    width = 32
    filled = min(width, int(round(width * status["percent"] / 100.0)))
    bar = "#" * filled + "-" * (width - filled)
    lines = [
        "=" * 78,
        f"Paired Fibonacci Rollout Monitor | {datetime.now().astimezone():%Y-%m-%d %H:%M:%S %Z}",
        "=" * 78,
        f"output: {status['output_dir']}",
        f"status: {status['status']} | PID={status['pid']}",
        f"progress: [{bar}] {status['completed']}/{status['planned']} ({status['percent']:.2f}%)",
        f"remaining: {status['remaining']} | failed: {status['failed']}",
        f"current: {status['current_run_id'] or 'none'}",
        f"average: {_duration(status['average_seconds'])} per newly completed rollout",
        f"ETA: {_duration(status['eta_seconds'])}",
        f"last update: {status['updated_at_utc'] or 'none'}",
        f"protocol: {status['protocol_fingerprint']}",
        "=" * 78,
    ]
    return "\n".join(lines)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=float, default=10.0)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.interval <= 0:
        print("error: --interval must be positive", file=sys.stderr)
        return 2
    while True:
        try:
            status = read_status(args.output_dir)
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
            print(f"error: {error}", file=sys.stderr)
            return 2
        print(format_status(status), flush=True)
        if not args.watch or status["status"] == "COMPLETED":
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
