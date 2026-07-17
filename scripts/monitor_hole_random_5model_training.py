#!/usr/bin/env python3
"""Monitor the sequential five-model early-stopping training pipeline."""

from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Sequence


MODEL_NAMES = (
    "contact_cvae_zero_seed0",
    "contact_cvae_prior_seed0",
    "motion_cvae_seed0",
    "dualzero_seed0",
    "act_baseline_run0",
)
TERMINAL_PIPELINE_STATUSES = {"completed", "failed", "interrupted"}
STEPS_PER_EPOCH_PATTERN = re.compile(r"^steps_per_epoch=(\d+)$", re.MULTILINE)
EARLY_STOP_PATTERN = re.compile(
    r"early stopping at epoch=(\d+) step=(\d+) "
    r"best_epoch=(\d+) best_step=(\d+)"
)


@dataclass(frozen=True)
class ModelProgress:
    name: str
    status: str
    step: int | None
    epoch: int | None
    batch_in_epoch: int | None
    steps_per_epoch: int | None
    loss_total: float | None
    deploy_loss: float | None
    validation_epoch: int | None
    best_metric: float | None
    best_epoch: int | None
    patience_used: int | None
    deployment_mode: str | None
    stop_reason: str | None
    started_at: datetime | None
    finished_at: datetime | None


@dataclass(frozen=True)
class EtaEstimate:
    seconds_per_step: float
    observed_steps: int
    plateau_remaining_steps: int
    hard_remaining_steps: int
    plateau_seconds: float
    hard_seconds: float


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path(
            "outputs/hole_random_60mm_hmj/"
            "earlystop_train90_splitseed20260716_run1"
        ),
        help="Experiment root containing formal/ and smoke/.",
    )
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=float, default=10.0)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=200000)
    parser.add_argument("--min-epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument(
        "--eta-min-steps",
        type=int,
        default=100,
        help="Wait for this many aggregate logged steps before showing ETA.",
    )
    parser.add_argument(
        "--stale-after",
        type=float,
        default=900.0,
        help="Treat recent log writes as active when PID visibility is restricted.",
    )
    parser.add_argument(
        "--show-log-tail",
        type=int,
        default=1,
        help="Show this many non-empty lines from the current model console.",
    )
    return parser


def read_status(path: Path) -> str | None:
    return read_status_record(path)[0]


def read_status_record(path: Path) -> tuple[str | None, datetime | None]:
    try:
        first_line = path.read_text(encoding="utf-8").splitlines()[0]
    except (OSError, IndexError):
        return None, None
    parts = first_line.split("\t", 1)
    status = parts[0].strip() or None
    timestamp = parse_datetime(parts[1].strip()) if len(parts) > 1 else None
    return status, timestamp


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def parse_datetime(value: str) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.astimezone()
    return parsed


def read_datetime(path: Path) -> datetime | None:
    return parse_datetime(read_text(path).strip())


def read_last_csv_row(path: Path) -> dict[str, str]:
    try:
        with path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
    except (OSError, csv.Error):
        return {}
    return rows[-1] if rows else {}


def optional_int(value: str | None) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(float(value))
    except ValueError:
        return None


def optional_float(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def process_is_running(pid_file: Path) -> bool:
    try:
        pid = int(pid_file.read_text(encoding="utf-8").strip())
        os.kill(pid, 0)
    except (OSError, ValueError):
        return False
    return True


def path_was_modified_recently(
    path: Path,
    *,
    now: datetime,
    stale_after: float,
) -> bool:
    try:
        modified = datetime.fromtimestamp(path.stat().st_mtime).astimezone()
    except OSError:
        return False
    age = (now - modified).total_seconds()
    return 0 <= age <= stale_after


def infer_stop_reason(
    *,
    console: str,
    step: int | None,
    epoch: int | None,
    max_epochs: int,
) -> str | None:
    if EARLY_STOP_PATTERN.search(console):
        return "early_stopping"
    if epoch is not None and epoch >= max_epochs:
        return "max_epochs"
    if "saved final checkpoint:" in console and step is not None:
        return "max_steps"
    return None


def inspect_model(
    *,
    name: str,
    formal_root: Path,
    pipeline_dir: Path,
    current_model: str | None,
    runner_running: bool,
    max_epochs: int,
) -> ModelProgress:
    output_dir = formal_root / name
    raw_status, status_timestamp = read_status_record(
        pipeline_dir / f"{name}.status"
    )
    status = raw_status
    if status is None:
        if (output_dir / "checkpoint.pt").is_file() and (
            output_dir / "checkpoint_best.pt"
        ).is_file():
            status = "complete-unmarked"
        elif output_dir.exists() and any(output_dir.iterdir()):
            status = "partial"
        else:
            status = "queued"
    elif status == "completed":
        status = "complete"
    elif status == "running" and not (
        runner_running and current_model == name
    ):
        status = "stale/partial"

    train_row = read_last_csv_row(output_dir / "train_log.csv")
    validation_row = read_last_csv_row(output_dir / "validation_log.csv")
    console = read_text(output_dir / "console.log")
    steps_match = STEPS_PER_EPOCH_PATTERN.search(console)
    steps_per_epoch = int(steps_match.group(1)) if steps_match else None
    step = optional_int(train_row.get("step"))
    epoch = optional_int(train_row.get("epoch"))

    return ModelProgress(
        name=name,
        status=status,
        step=step,
        epoch=epoch,
        batch_in_epoch=optional_int(train_row.get("batch_in_epoch")),
        steps_per_epoch=steps_per_epoch,
        loss_total=optional_float(train_row.get("loss_total")),
        deploy_loss=optional_float(validation_row.get("deploy_loss")),
        validation_epoch=optional_int(validation_row.get("epoch")),
        best_metric=optional_float(validation_row.get("best_metric")),
        best_epoch=optional_int(validation_row.get("best_epoch")),
        patience_used=optional_int(
            validation_row.get("epochs_without_improvement")
        ),
        deployment_mode=validation_row.get("deployment_mode") or None,
        stop_reason=infer_stop_reason(
            console=console,
            step=step,
            epoch=epoch,
            max_epochs=max_epochs,
        ),
        started_at=(
            read_datetime(pipeline_dir / f"{name}.started_at")
            or (status_timestamp if raw_status == "running" else None)
        ),
        finished_at=read_datetime(pipeline_dir / f"{name}.finished_at"),
    )


def elapsed_seconds(item: ModelProgress, now: datetime) -> float | None:
    if item.started_at is None:
        return None
    end = item.finished_at or (now if item.status == "running" else None)
    if end is None:
        return None
    elapsed = (end - item.started_at).total_seconds()
    return elapsed if elapsed > 0 else None


def plateau_stop_epoch(
    item: ModelProgress,
    *,
    min_epochs: int,
    patience: int,
    max_epochs: int,
) -> int:
    validation_epoch = item.validation_epoch
    patience_used = item.patience_used or 0
    if validation_epoch is None or validation_epoch < min_epochs:
        projected = min_epochs + patience - 1
    else:
        projected = validation_epoch + max(patience - patience_used, 0)
    return min(projected, max_epochs)


def estimate_pipeline_eta(
    progress: Sequence[ModelProgress],
    *,
    now: datetime,
    max_steps: int,
    max_epochs: int,
    min_epochs: int,
    patience: int,
    eta_min_steps: int,
) -> EtaEstimate | None:
    observed_seconds = 0.0
    observed_steps = 0
    for item in progress:
        elapsed = elapsed_seconds(item, now)
        if elapsed is None or item.step is None or item.step <= 0:
            continue
        observed_seconds += elapsed
        observed_steps += item.step

    if observed_steps < eta_min_steps or observed_seconds <= 0:
        return None

    steps_per_epoch = next(
        (
            item.steps_per_epoch
            for item in progress
            if item.steps_per_epoch is not None and item.steps_per_epoch > 0
        ),
        None,
    )
    if steps_per_epoch is None:
        return None

    hard_target = min(max_steps, max_epochs * steps_per_epoch)
    plateau_remaining = 0
    hard_remaining = 0
    terminal_statuses = {"complete", "complete-unmarked", "failed"}
    for item in progress:
        if item.status in terminal_statuses:
            continue
        completed_steps = item.step or 0
        plateau_target = min(
            hard_target,
            plateau_stop_epoch(
                item,
                min_epochs=min_epochs,
                patience=patience,
                max_epochs=max_epochs,
            )
            * steps_per_epoch,
        )
        plateau_remaining += max(plateau_target - completed_steps, 0)
        hard_remaining += max(hard_target - completed_steps, 0)

    seconds_per_step = observed_seconds / observed_steps
    return EtaEstimate(
        seconds_per_step=seconds_per_step,
        observed_steps=observed_steps,
        plateau_remaining_steps=plateau_remaining,
        hard_remaining_steps=hard_remaining,
        plateau_seconds=plateau_remaining * seconds_per_step,
        hard_seconds=hard_remaining * seconds_per_step,
    )


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


def format_optional(value: object, precision: int = 5) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{precision}g}"
    return str(value)


def format_duration(seconds: float) -> str:
    rounded = max(0, int(round(seconds)))
    days, remainder = divmod(rounded, 24 * 60 * 60)
    hours, remainder = divmod(remainder, 60 * 60)
    minutes, seconds = divmod(remainder, 60)
    if days:
        return f"{days}d {hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def format_finish_time(now: datetime, seconds: float) -> str:
    return (now + timedelta(seconds=seconds)).strftime("%Y-%m-%d %H:%M:%S %Z")


def current_log_tail(
    formal_root: Path,
    current_model: str | None,
    line_count: int,
) -> list[str]:
    if not current_model or line_count <= 0:
        return []
    lines = [
        line
        for line in read_text(
            formal_root / current_model / "console.log"
        ).splitlines()
        if line.strip()
    ]
    return lines[-line_count:]


def render(
    *,
    run_root: Path,
    pipeline_status: str,
    runner_state: str,
    current_model: str | None,
    progress: Sequence[ModelProgress],
    max_epochs: int,
    patience: int,
    eta: EtaEstimate | None,
    eta_min_steps: int,
    now: datetime,
    log_tail: Sequence[str],
) -> str:
    completed = sum(item.status in {"complete", "complete-unmarked"} for item in progress)
    lines = [
        "=" * 142,
        "ForceAwareACT Five-Model Early-Stopping Monitor",
        "=" * 142,
        f"time: {datetime.now():%Y-%m-%d %H:%M:%S}",
        f"run_root: {run_root}",
        f"pipeline: {pipeline_status} | runner: {runner_state} | completed: {completed}/5",
        f"current_model: {current_model or '-'}",
    ]
    if pipeline_status == "completed":
        lines.append("ETA: complete")
    elif pipeline_status in {"failed", "interrupted"}:
        lines.append(f"ETA: unavailable because pipeline is {pipeline_status}")
    elif eta is None:
        lines.append(
            "ETA: collecting throughput; available after "
            f"{eta_min_steps} aggregate logged training steps"
        )
    else:
        lines.extend(
            [
                (
                    "ETA basis: "
                    f"{eta.seconds_per_step:.3f} s/step from "
                    f"{eta.observed_steps} observed steps; validation/initialization time included"
                ),
                (
                    "plateau ETA (assuming no future >=0.5% improvement): "
                    f"{format_duration(eta.plateau_seconds)} -> "
                    f"{format_finish_time(now, eta.plateau_seconds)} "
                    f"({eta.plateau_remaining_steps} steps)"
                ),
                (
                    "hard-budget ETA (max_epochs/max_steps): "
                    f"{format_duration(eta.hard_seconds)} -> "
                    f"{format_finish_time(now, eta.hard_seconds)} "
                    f"({eta.hard_remaining_steps} steps)"
                ),
            ]
        )
    lines.extend(
        [
            "-" * 142,
            (
            f"{'model':<31}"
            f"{'status':<15}"
            f"{'epoch':<9}"
            f"{'step':<10}"
            f"{'batch':<15}"
            f"{'train_loss':<13}"
            f"{'deploy':<12}"
            f"{'best':<12}"
            f"{'best_ep':<9}"
            f"{'patience':<11}"
            f"{'mode':<8}"
            f"{'stop':<15}"
            ),
        ]
    )
    for item in progress:
        batch = (
            f"{item.batch_in_epoch}/{item.steps_per_epoch}"
            if item.batch_in_epoch is not None and item.steps_per_epoch is not None
            else "-"
        )
        patience_text = (
            f"{item.patience_used}/{patience}"
            if item.patience_used is not None
            else "-"
        )
        epoch_text = (
            f"{item.epoch}/{max_epochs}" if item.epoch is not None else "-"
        )
        lines.append(
            f"{item.name:<31}"
            f"{item.status:<15}"
            f"{epoch_text:<9}"
            f"{format_optional(item.step):<10}"
            f"{batch:<15}"
            f"{format_optional(item.loss_total):<13}"
            f"{format_optional(item.deploy_loss):<12}"
            f"{format_optional(item.best_metric):<12}"
            f"{format_optional(item.best_epoch):<9}"
            f"{patience_text:<11}"
            f"{format_optional(item.deployment_mode):<8}"
            f"{format_optional(item.stop_reason):<15}"
        )

    lines.extend(["-" * 142, "GPU:"])
    lines.extend(f"  {line}" for line in gpu_report())
    if log_tail:
        lines.extend(["-" * 142, "Current model log tail:"])
        lines.extend(f"  {line}" for line in log_tail)
    lines.append("=" * 142)
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.interval <= 0:
        raise ValueError("--interval must be positive")
    if args.max_epochs <= 0 or args.max_steps <= 0:
        raise ValueError("--max-epochs and --max-steps must be positive")
    if args.min_epochs < 0 or args.patience <= 0:
        raise ValueError("--min-epochs must be non-negative and --patience positive")
    if args.eta_min_steps <= 0:
        raise ValueError("--eta-min-steps must be positive")
    if args.stale_after <= 0:
        raise ValueError("--stale-after must be positive")
    if args.show_log_tail < 0:
        raise ValueError("--show-log-tail must be non-negative")

    run_root = args.run_root.expanduser().resolve()
    formal_root = run_root / "formal"
    pipeline_dir = formal_root / ".pipeline"

    try:
        while True:
            now = datetime.now().astimezone()
            pipeline_status = read_status(pipeline_dir / "pipeline.status") or "not-started"
            current_model = read_text(pipeline_dir / "current_model").strip() or None
            runner_pid_visible = process_is_running(pipeline_dir / "runner.pid")
            activity_paths = [pipeline_dir / "runner.log"]
            if current_model:
                activity_paths.extend(
                    [
                        formal_root / current_model / "console.log",
                        formal_root / current_model / "train_log.csv",
                        formal_root / current_model / "validation_log.csv",
                    ]
                )
            recent_activity = pipeline_status == "running" and any(
                path_was_modified_recently(
                    path,
                    now=now,
                    stale_after=args.stale_after,
                )
                for path in activity_paths
            )
            runner_running = runner_pid_visible or recent_activity
            if runner_pid_visible:
                runner_state = "RUNNING"
            elif recent_activity:
                runner_state = "ACTIVE (recent logs; PID not visible)"
            else:
                runner_state = "NOT DETECTED"
            progress = [
                inspect_model(
                    name=name,
                    formal_root=formal_root,
                    pipeline_dir=pipeline_dir,
                    current_model=current_model,
                    runner_running=runner_running,
                    max_epochs=args.max_epochs,
                )
                for name in MODEL_NAMES
            ]
            eta = estimate_pipeline_eta(
                progress,
                now=now,
                max_steps=args.max_steps,
                max_epochs=args.max_epochs,
                min_epochs=args.min_epochs,
                patience=args.patience,
                eta_min_steps=args.eta_min_steps,
            )
            report = render(
                run_root=run_root,
                pipeline_status=pipeline_status,
                runner_state=runner_state,
                current_model=current_model,
                progress=progress,
                max_epochs=args.max_epochs,
                patience=args.patience,
                eta=eta,
                eta_min_steps=args.eta_min_steps,
                now=now,
                log_tail=current_log_tail(
                    formal_root,
                    current_model,
                    args.show_log_tail,
                ),
            )
            if args.watch:
                print("\033[2J\033[H", end="")
            print(report, flush=True)

            if not args.watch or pipeline_status in TERMINAL_PIPELINE_STATUSES:
                return 0
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nmonitor stopped; training was not interrupted")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
