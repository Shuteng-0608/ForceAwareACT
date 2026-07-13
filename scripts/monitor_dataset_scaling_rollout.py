#!/usr/bin/env python3
"""Monitor the current dataset-scaling MuJoCo rollout experiment."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence


MODEL_SIZES = (50, 100, 150, 203)
POINT_PATTERN = re.compile(r"point_(\d+)_")
COMPLETED_STATUSES = {
    "success",
    "task_failed",
    "skipped_existing",
}


@dataclass(frozen=True)
class Progress:
    size: int
    root: Path
    status: str
    completed: int
    attempted: int
    process_errors: int
    current_point: int | None
    inference_device: str | None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Monitor the single-GPU sequential dataset-scaling rollout."
        )
    )
    parser.add_argument(
        "--evaluation-root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--num-points",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--watch",
        action="store_true",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=10.0,
    )
    return parser


def process_lines() -> list[str]:
    result = subprocess.run(
        ["ps", "-eo", "pid=,args="],
        capture_output=True,
        text=True,
        check=False,
    )

    if result.returncode != 0:
        return []

    return [
        line.strip()
        for line in result.stdout.splitlines()
        if (
            "run_mujoco_hole_grid.py" in line
            or "run_mujoco_policy_rollout.py" in line
        )
    ]


def read_json(path: Path) -> dict:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8")
        )
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def csv_row_count(path: Path) -> int:
    try:
        with path.open(
            newline="",
            encoding="utf-8",
        ) as file:
            return sum(1 for _ in csv.DictReader(file))
    except OSError:
        return 0


def inspect_model(
    *,
    size: int,
    evaluation_root: Path,
    num_points: int,
    processes: Sequence[str],
) -> Progress:
    root = evaluation_root / f"mix{size}_zero_mid"

    manifest = read_json(root / "grid_manifest.json")
    runs = manifest.get("runs", [])

    if not isinstance(runs, list):
        runs = []

    completed = sum(
        1
        for run in runs
        if isinstance(run, dict)
        and run.get("status") in COMPLETED_STATUSES
    )

    process_errors = sum(
        1
        for run in runs
        if isinstance(run, dict)
        and run.get("status") == "process_error"
    )

    attempted = len(runs)
    summary_rows = csv_row_count(
        root / "grid_summary.csv"
    )

    matching = [
        line
        for line in processes
        if str(root) in line
    ]

    is_running = bool(matching)

    started = any([
        root.exists(),
        (root / "task_points.csv").is_file(),
        evaluation_root.joinpath(
            f"mix{size}_zero_mid.console.log"
        ).is_file(),
        evaluation_root.joinpath(
            f"mix{size}_zero_mid.launch_manifest.txt"
        ).is_file(),
    ])

    if (
        summary_rows >= num_points
        and process_errors == 0
    ):
        status = "complete"
        completed = num_points
        attempted = max(attempted, num_points)
        current_point = None
    elif is_running:
        status = "running"

        point_numbers = [
            int(match.group(1))
            for line in matching
            for match in [POINT_PATTERN.search(line)]
            if match is not None
        ]

        current_point = (
            point_numbers[0]
            if point_numbers
            else min(completed + 1, num_points)
        )
    elif started:
        status = "partial"
        current_point = None
    else:
        status = "queued"
        current_point = None

    inference_device = None

    latest_summary = None
    summaries = list(
        root.glob("point_*/summary.json")
    )

    if summaries:
        latest_summary = max(
            summaries,
            key=lambda path: path.stat().st_mtime,
        )

    if latest_summary is not None:
        summary = read_json(latest_summary)
        value = summary.get("inference_device")

        if value is not None:
            inference_device = str(value)

    return Progress(
        size=size,
        root=root,
        status=status,
        completed=completed,
        attempted=attempted,
        process_errors=process_errors,
        current_point=current_point,
        inference_device=inference_device,
    )


def gpu_process_report() -> list[str]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps="
            "pid,process_name,used_gpu_memory",
            "--format=csv,noheader",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    if result.returncode != 0:
        return ["nvidia-smi unavailable"]

    lines = [
        line.strip()
        for line in result.stdout.splitlines()
        if line.strip()
    ]

    return lines or ["no CUDA compute process detected"]


def render(
    progress: Sequence[Progress],
    evaluation_root: Path,
    num_points: int,
) -> str:
    total = len(progress) * num_points
    completed = sum(item.completed for item in progress)
    errors = sum(item.process_errors for item in progress)

    lines = [
        "=" * 104,
        "ForceAwareACT Dataset-Scaling Rollout Monitor",
        "=" * 104,
        f"time: {datetime.now():%Y-%m-%d %H:%M:%S}",
        f"evaluation_root: {evaluation_root}",
        f"overall: {completed}/{total} "
        f"({100.0 * completed / max(total, 1):.1f}%)",
        f"process_errors: {errors}",
        "-" * 104,
        (
            f"{'model':<10}"
            f"{'status':<12}"
            f"{'completed':<15}"
            f"{'current':<12}"
            f"{'attempted':<12}"
            f"{'errors':<10}"
            f"{'device':<12}"
        ),
    ]

    for item in progress:
        current = (
            str(item.current_point)
            if item.current_point is not None
            else "-"
        )

        device = item.inference_device or "-"

        lines.append(
            f"{f'mix{item.size}':<10}"
            f"{item.status:<12}"
            f"{f'{item.completed}/{num_points}':<15}"
            f"{current:<12}"
            f"{item.attempted:<12}"
            f"{item.process_errors:<10}"
            f"{device:<12}"
        )

    lines.extend([
        "-" * 104,
        "CUDA compute processes:",
    ])

    lines.extend(
        f"  {line}"
        for line in gpu_process_report()
    )

    lines.append("=" * 104)
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.num_points <= 0:
        raise ValueError("--num-points must be positive")

    if args.interval <= 0:
        raise ValueError("--interval must be positive")

    evaluation_root = (
        args.evaluation_root
        .expanduser()
        .resolve()
    )

    try:
        while True:
            processes = process_lines()

            progress = [
                inspect_model(
                    size=size,
                    evaluation_root=evaluation_root,
                    num_points=args.num_points,
                    processes=processes,
                )
                for size in MODEL_SIZES
            ]

            os.system("clear" if args.watch else "true")

            print(
                render(
                    progress,
                    evaluation_root,
                    args.num_points,
                ),
                flush=True,
            )

            if not args.watch:
                return 0

            time.sleep(args.interval)

    except KeyboardInterrupt:
        print("\nmonitor stopped; rollout was not interrupted")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
