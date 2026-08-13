#!/usr/bin/env python3
"""Read-only progress and ETA monitor for ACT-aligned training."""

from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from statistics import mean
from typing import Any, Optional, Sequence


TRAIN_SCRIPT_NAMES = (
    "train_act_aligned_contact_cvae.py",
    "train_act_aligned_high_rate_contact_cvae.py",
    "train_act_aligned_high_rate_motion_cvae.py",
    "train_act_aligned_high_rate_dual_zero.py",
    "train_act_aligned_motion_cvae_control.py",
    "train_official_act.py",
)
DEFAULT_TARGET_STEPS = 24_000
DEFAULT_CHECKPOINT_INTERVAL = 2_000


@dataclass(frozen=True)
class ProcessInfo:
    pid: int
    parent_pid: int
    elapsed_seconds: float
    command: str


@dataclass(frozen=True)
class EtaEstimate:
    completed_steps: int
    steps_per_second: float
    seconds_per_step: float
    remaining_seconds: float
    estimated_total_seconds: float
    estimated_finish: datetime


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Training output directory containing metrics.jsonl.",
    )
    parser.add_argument(
        "--target-steps",
        type=int,
        default=DEFAULT_TARGET_STEPS,
    )
    parser.add_argument(
        "--start-step",
        type=int,
        default=0,
        help="Global step at process start; set this when monitoring a resume.",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=DEFAULT_CHECKPOINT_INTERVAL,
    )
    parser.add_argument(
        "--pid",
        type=int,
        help="Training PID; by default it is detected from --output-dir.",
    )
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=float, default=10.0)
    parser.add_argument(
        "--recent-window",
        type=int,
        default=20,
        help="Number of logged step records used for recent metric means.",
    )
    parser.add_argument(
        "--stale-after",
        type=float,
        default=120.0,
        help="Seconds without a metrics write before a stopped run is stale.",
    )
    return parser


def read_jsonl_records(path: Path) -> list[dict[str, Any]]:
    """Read complete JSON objects while tolerating an in-progress final line."""

    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    records: list[dict[str, Any]] = []
    for line in lines:
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            records.append(value)
    return records


def estimate_eta(
    *,
    current_step: int,
    start_step: int,
    target_steps: int,
    elapsed_seconds: float,
    now: datetime,
) -> Optional[EtaEstimate]:
    completed_steps = current_step - start_step
    if (
        completed_steps <= 0
        or current_step >= target_steps
        or elapsed_seconds <= 0
    ):
        return None
    steps_per_second = completed_steps / elapsed_seconds
    if not math.isfinite(steps_per_second) or steps_per_second <= 0:
        return None
    seconds_per_step = 1.0 / steps_per_second
    remaining_seconds = (target_steps - current_step) * seconds_per_step
    return EtaEstimate(
        completed_steps=completed_steps,
        steps_per_second=steps_per_second,
        seconds_per_step=seconds_per_step,
        remaining_seconds=remaining_seconds,
        estimated_total_seconds=elapsed_seconds + remaining_seconds,
        estimated_finish=now + timedelta(seconds=remaining_seconds),
    )


def _process_table() -> list[ProcessInfo]:
    result = subprocess.run(
        ["ps", "-eo", "pid=,ppid=,etimes=,args="],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        return []
    processes: list[ProcessInfo] = []
    for line in result.stdout.splitlines():
        parts = line.strip().split(maxsplit=3)
        if len(parts) != 4:
            continue
        try:
            pid = int(parts[0])
            parent_pid = int(parts[1])
            elapsed_seconds = float(parts[2])
        except ValueError:
            continue
        processes.append(
            ProcessInfo(pid, parent_pid, elapsed_seconds, parts[3])
        )
    return processes


def _command_output_dir(process: ProcessInfo) -> Optional[Path]:
    try:
        arguments = shlex.split(process.command)
        output_index = arguments.index("--output-dir") + 1
        raw_path = Path(arguments[output_index])
    except (ValueError, IndexError):
        return None
    if raw_path.is_absolute():
        return raw_path.resolve()
    try:
        working_directory = Path(os.readlink(f"/proc/{process.pid}/cwd"))
    except OSError:
        return None
    return (working_directory / raw_path).resolve()


def find_training_process(
    output_dir: Path,
    *,
    requested_pid: Optional[int] = None,
) -> Optional[ProcessInfo]:
    processes = _process_table()
    if requested_pid is not None:
        return next(
            (process for process in processes if process.pid == requested_pid),
            None,
        )

    resolved_output_dir = output_dir.resolve()
    matches = []
    for process in processes:
        if not any(
            script_name in process.command
            for script_name in TRAIN_SCRIPT_NAMES
        ):
            continue
        process_output_dir = _command_output_dir(process)
        if process_output_dir == resolved_output_dir:
            matches.append(process)
    match_pids = {process.pid for process in matches}
    root_matches = [
        process
        for process in matches
        if process.parent_pid not in match_pids
    ]
    if len(root_matches) == 1:
        return root_matches[0]
    if len(root_matches) > 1:
        pids = ", ".join(str(process.pid) for process in root_matches)
        raise RuntimeError(
            "multiple root training processes match the output directory: "
            f"{pids}; pass --pid explicitly"
        )
    return matches[0] if len(matches) == 1 else None


def _format_duration(seconds: float) -> str:
    if not math.isfinite(seconds):
        return "unavailable"
    return str(timedelta(seconds=max(0, round(seconds))))


def _format_bytes(num_bytes: Optional[int]) -> str:
    if num_bytes is None:
        return "n/a"
    return f"{num_bytes / (1024**3):.2f} GiB"


def _progress_bar(current: int, target: int, width: int = 32) -> str:
    fraction = min(max(current / max(target, 1), 0.0), 1.0)
    completed = min(width, int(fraction * width))
    return "[" + "#" * completed + "-" * (width - completed) + "]"


def _metric_mean(
    records: Sequence[dict[str, Any]],
    name: str,
) -> Optional[float]:
    values = [
        record["metrics"][name]
        for record in records
        if isinstance(record.get("metrics"), dict)
        and isinstance(record["metrics"].get(name), (int, float))
    ]
    return mean(values) if values else None


def _format_metric(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.6g}"


def _latest_periodic_checkpoint(output_dir: Path) -> Optional[Path]:
    checkpoints = sorted(output_dir.glob("step_*.pt"))
    return checkpoints[-1] if checkpoints else None


def _read_summary(output_dir: Path) -> Optional[dict[str, Any]]:
    for name in ("training_summary.json", "burn_in_summary.json"):
        path = output_dir / name
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(value, dict):
            return value
    return None


def render_report(
    *,
    output_dir: Path,
    target_steps: int,
    start_step: int,
    checkpoint_interval: int,
    recent_window: int,
    stale_after: float,
    requested_pid: Optional[int],
    previous_sample: Optional[tuple[float, int]] = None,
    now_monotonic: Optional[float] = None,
    now: Optional[datetime] = None,
) -> tuple[str, tuple[float, int]]:
    now_monotonic = time.monotonic() if now_monotonic is None else now_monotonic
    now = datetime.now().astimezone() if now is None else now
    metrics_path = output_dir / "metrics.jsonl"
    records = read_jsonl_records(metrics_path)
    step_records = [
        record for record in records if record.get("record_type") == "step"
    ]
    segment_records = [
        record
        for record in records
        if record.get("record_type") == "epoch_segment"
    ]
    latest = step_records[-1] if step_records else {}
    current_step = int(latest.get("global_step", 0))
    process = find_training_process(
        output_dir,
        requested_pid=requested_pid,
    )
    summary = _read_summary(output_dir)

    try:
        log_age = max(0.0, time.time() - metrics_path.stat().st_mtime)
    except OSError:
        log_age = math.inf

    if current_step >= target_steps or (
        summary is not None and summary.get("passed") is True
    ):
        status = "COMPLETED"
    elif process is not None:
        status = "RUNNING"
    elif log_age <= stale_after:
        status = "RECENT LOG / PROCESS NOT FOUND"
    elif current_step > 0:
        status = "STOPPED OR STALE"
    else:
        status = "WAITING FOR FIRST LOG RECORD"

    progress = 100.0 * current_step / max(target_steps, 1)
    remaining_steps = max(target_steps - current_step, 0)
    lines = [
        "=" * 78,
        f"ACT-Aligned Training Monitor | {now:%Y-%m-%d %H:%M:%S %Z}",
        "=" * 78,
        f"output: {output_dir}",
        (
            f"status: {status}"
            + (
                f" | PID {process.pid} | elapsed "
                f"{_format_duration(process.elapsed_seconds)}"
                if process is not None
                else ""
            )
        ),
        (
            f"progress: {_progress_bar(current_step, target_steps)} "
            f"{current_step:,}/{target_steps:,} ({progress:.2f}%)"
        ),
        (
            f"position: epoch={latest.get('epoch', 'n/a')} "
            f"step_in_epoch={latest.get('step_in_epoch', 'n/a')} "
            f"remaining_steps={remaining_steps:,}"
        ),
    ]

    eta = (
        estimate_eta(
            current_step=current_step,
            start_step=start_step,
            target_steps=target_steps,
            elapsed_seconds=process.elapsed_seconds,
            now=now,
        )
        if process is not None
        else None
    )
    if eta is not None:
        lines.extend(
            [
                (
                    f"average speed: {eta.steps_per_second:.3f} steps/s "
                    f"({eta.seconds_per_step:.3f} s/step)"
                ),
                (
                    f"ETA: {_format_duration(eta.remaining_seconds)} | "
                    f"estimated finish: "
                    f"{eta.estimated_finish:%Y-%m-%d %H:%M:%S %Z}"
                ),
                (
                    "estimated total process time: "
                    f"{_format_duration(eta.estimated_total_seconds)}"
                ),
            ]
        )
    elif status == "COMPLETED":
        lines.append("ETA: complete")
    else:
        lines.append(
            "ETA: unavailable; pass --pid if automatic process detection failed"
        )

    if previous_sample is not None:
        previous_time, previous_step = previous_sample
        sample_seconds = now_monotonic - previous_time
        sample_steps = current_step - previous_step
        if sample_seconds > 0 and sample_steps > 0:
            recent_speed = sample_steps / sample_seconds
            lines.append(
                f"recent sampled speed: {recent_speed:.3f} steps/s "
                f"over {sample_steps} steps"
            )
        else:
            lines.append("recent sampled speed: collecting new logged steps")

    recent = step_records[-recent_window:]
    latest_metrics = latest.get("metrics", {})
    lines.extend(
        [
            "-" * 78,
            (
                "latest metrics: "
                f"total={_format_metric(latest_metrics.get('loss_total'))} "
                f"action={_format_metric(latest_metrics.get('loss_action'))} "
                f"force={_format_metric(latest_metrics.get('loss_force'))} "
                f"posterior_kl="
                f"{_format_metric(latest_metrics.get('loss_posterior_kl'))} "
                f"prior_match="
                f"{_format_metric(latest_metrics.get('loss_prior_match'))} "
                f"grad={_format_metric(latest_metrics.get('gradient_norm'))}"
            ),
            (
                f"last {len(recent)} logged records mean: "
                f"total={_format_metric(_metric_mean(recent, 'loss_total'))} "
                f"action={_format_metric(_metric_mean(recent, 'loss_action'))} "
                f"force={_format_metric(_metric_mean(recent, 'loss_force'))} "
                f"posterior_kl="
                f"{_format_metric(_metric_mean(recent, 'loss_posterior_kl'))} "
                f"prior_match="
                f"{_format_metric(_metric_mean(recent, 'loss_prior_match'))}"
            ),
        ]
    )

    cuda_memory = latest.get("cuda_memory", {})
    lines.append(
        "CUDA memory: "
        f"allocated={_format_bytes(cuda_memory.get('allocated_bytes'))} "
        f"peak_allocated="
        f"{_format_bytes(cuda_memory.get('peak_allocated_bytes'))} "
        f"peak_reserved="
        f"{_format_bytes(cuda_memory.get('peak_reserved_bytes'))}"
    )

    latest_checkpoint = _latest_periodic_checkpoint(output_dir)
    next_checkpoint_step = min(
        (
            (current_step // checkpoint_interval) + 1
        )
        * checkpoint_interval,
        target_steps,
    )
    lines.append(
        "checkpoint: "
        f"latest={latest_checkpoint.name if latest_checkpoint else 'none'} "
        f"next_step={next_checkpoint_step:,}"
    )
    if segment_records:
        validation = segment_records[-1]
        validation_metrics = validation.get("validation", {})
        validation_line = (
            "validation: "
            f"last_step={validation.get('global_step', 'n/a')} "
            f"zero_action_l1="
            f"{_format_metric(validation_metrics.get('deployment_zero_action_l1'))}"
        )
        if "deployment_prior_action_l1" in validation_metrics:
            validation_line += (
                " prior_action_l1="
                f"{_format_metric(validation_metrics.get('deployment_prior_action_l1'))}"
            )
        lines.append(validation_line)
    else:
        lines.append("validation: none yet (first full data epoch is about step 3,460)")

    lines.append(
        f"log freshness: {_format_duration(log_age)} since last metrics write"
    )
    if log_age > stale_after and status != "COMPLETED":
        lines.append(
            f"WARNING: metrics log is stale (threshold {stale_after:.0f}s)"
        )
    lines.append("=" * 78)
    return "\n".join(lines), (now_monotonic, current_step)


def main() -> None:
    args = build_parser().parse_args()
    if args.target_steps <= 0:
        raise SystemExit("--target-steps must be positive")
    if args.start_step < 0 or args.start_step >= args.target_steps:
        raise SystemExit("--start-step must be in [0, target-steps)")
    if args.checkpoint_interval <= 0:
        raise SystemExit("--checkpoint-interval must be positive")
    if args.interval <= 0 or args.recent_window <= 0 or args.stale_after <= 0:
        raise SystemExit(
            "--interval, --recent-window, and --stale-after must be positive"
        )
    if not args.output_dir.is_dir():
        raise SystemExit(f"output directory does not exist: {args.output_dir}")

    previous_sample = None
    while True:
        try:
            report, previous_sample = render_report(
                output_dir=args.output_dir,
                target_steps=args.target_steps,
                start_step=args.start_step,
                checkpoint_interval=args.checkpoint_interval,
                recent_window=args.recent_window,
                stale_after=args.stale_after,
                requested_pid=args.pid,
                previous_sample=previous_sample,
            )
        except RuntimeError as error:
            raise SystemExit(str(error)) from error
        if args.watch and sys.stdout.isatty():
            print("\033[2J\033[H", end="")
        print(report, flush=True)
        if not args.watch:
            break
        try:
            time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\nmonitor stopped; training was not interrupted")
            break


if __name__ == "__main__":
    main()
