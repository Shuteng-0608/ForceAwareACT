#!/usr/bin/env python3
"""Read-only progress monitor for one protocol-driven staged training run."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from force_aware_act.training.protocol import StageSpec, load_protocol  # noqa: E402


TRAIN_METRICS = (
    "loss_total",
    "loss_action",
    "loss_force",
    "kl_motion",
    "kl_contact",
    "loss_prior",
    "gradient_norm",
)


@dataclass(frozen=True)
class ValidationProgress:
    count: int
    without_selection: int
    last_stage_step: int
    latest_primary: Mapping[str, str]
    best_primary: Mapping[str, str]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--pid",
        type=int,
        default=None,
        help="Optional explicit training PID; otherwise train_staged.py is discovered.",
    )
    parser.add_argument("--recent-window", type=int, default=200)
    parser.add_argument("--eta-min-steps", type=int, default=100)
    parser.add_argument("--stale-after", type=float, default=900.0)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=float, default=10.0)
    parser.add_argument("--no-gpu", action="store_true")
    return parser


def optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        result = int(value)
    except (TypeError, ValueError):
        return None
    return result


def optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def parse_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0"}:
        return False
    return None


def csv_tail(path: Path, count: int) -> list[dict[str, str]]:
    """Read the header and recent complete rows without scanning a large log."""

    try:
        with path.open("rb") as stream:
            header_bytes = stream.readline()
            if not header_bytes:
                return []
            header = header_bytes.decode("utf-8", errors="replace").rstrip("\r\n")
            stream.seek(0, os.SEEK_END)
            position = stream.tell()
            chunks: list[bytes] = []
            newline_count = 0
            while position > len(header_bytes) and newline_count < count + 2:
                size = min(65536, position - len(header_bytes))
                position -= size
                stream.seek(position)
                chunk = stream.read(size)
                chunks.append(chunk)
                newline_count += chunk.count(b"\n")
    except OSError:
        return []

    body = b"".join(reversed(chunks)).decode("utf-8", errors="replace")
    lines = body.splitlines()
    if position > len(header_bytes) and lines:
        lines = lines[1:]
    lines = [line for line in lines if line.strip() and line != header][-count:]
    try:
        expected_fields = len(next(csv.reader([header])))
        rows = list(csv.DictReader([header, *lines]))
    except csv.Error:
        return []
    return [
        row
        for row in rows
        if None not in row and len(row) == expected_fields
    ]


def read_validation_rows(path: Path) -> list[dict[str, str]]:
    try:
        with path.open(newline="", encoding="utf-8") as stream:
            return [
                row
                for row in csv.DictReader(stream)
                if row.get("validation_index") not in (None, "")
            ]
    except (OSError, csv.Error):
        return []


def summarize_validations(
    rows: Sequence[Mapping[str, str]],
    stage: StageSpec,
) -> ValidationProgress:
    expected_domains = {item.name for item in stage.validation_domains}
    grouped: dict[int, list[Mapping[str, str]]] = {}
    for row in rows:
        index = optional_int(row.get("validation_index"))
        if index is not None:
            grouped.setdefault(index, []).append(row)

    complete: list[tuple[int, list[Mapping[str, str]]]] = []
    for index in sorted(grouped):
        group = grouped[index]
        if {row.get("domain") for row in group} != expected_domains:
            continue
        if len(group) != len(expected_domains):
            continue
        complete.append((index, group))

    without_selection = 0
    last_stage_step = 0
    latest_primary: Mapping[str, str] = {}
    best_primary: Mapping[str, str] = {}
    positive_count = 0
    for index, group in complete:
        primary = next(
            (
                row
                for row in group
                if row.get("domain") == stage.monitor.primary_domain
            ),
            {},
        )
        step = optional_int(primary.get("stage_step")) or 0
        selected = parse_bool(primary.get("selected")) is True
        if index == 0:
            if selected:
                best_primary = primary
            continue
        positive_count += 1
        last_stage_step = step
        latest_primary = primary
        if selected:
            best_primary = primary
            without_selection = 0
        elif step >= stage.monitor.min_stage_steps:
            without_selection += 1

    return ValidationProgress(
        count=positive_count,
        without_selection=without_selection,
        last_stage_step=last_stage_step,
        latest_primary=latest_primary,
        best_primary=best_primary,
    )


def earliest_plateau_stop_step(
    *,
    current_step: int,
    last_validation_step: int,
    min_stage_steps: int,
    validation_every_steps: int,
    patience: int,
    without_selection: int,
    max_steps: int,
) -> int:
    """Return the earliest stop boundary if no future checkpoint is selected."""

    if without_selection >= patience:
        return min(current_step, max_steps)
    needed = patience - without_selection
    if (
        current_step > last_validation_step
        and current_step % validation_every_steps == 0
    ):
        first_candidate = current_step
    else:
        boundary = max(current_step, last_validation_step)
        first_candidate = (
            boundary // validation_every_steps + 1
        ) * validation_every_steps
    first_candidate = max(first_candidate, min_stage_steps)
    target = first_candidate + (needed - 1) * validation_every_steps
    return min(target, max_steps)


def process_elapsed(pid: int) -> float | None:
    result = subprocess.run(
        ["ps", "-p", str(pid), "-o", "etimes="],
        capture_output=True,
        text=True,
        check=False,
    )
    value = result.stdout.strip()
    if result.returncode != 0 or not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def discover_training_process(
    *,
    output_dir: Path,
    protocol_path: Path,
    stage_name: str,
) -> tuple[int, float] | None:
    result = subprocess.run(
        ["ps", "-eo", "pid=,etimes=,args="],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    matches: list[tuple[int, float, str]] = []
    for line in result.stdout.splitlines():
        parts = line.strip().split(maxsplit=2)
        if len(parts) != 3 or "scripts/train_staged.py" not in parts[2]:
            continue
        command = parts[2]
        if f"--stage {stage_name}" not in command:
            continue
        try:
            matches.append((int(parts[0]), float(parts[1]), command))
        except ValueError:
            continue
    for pid, elapsed, command in matches:
        if str(output_dir) in command or str(protocol_path) in command:
            return pid, elapsed
    if len(matches) == 1:
        pid, elapsed, _ = matches[0]
        return pid, elapsed
    return None


def gpu_lines() -> list[str]:
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


def recent_mean(rows: Sequence[Mapping[str, str]], key: str) -> float | None:
    values = [optional_float(row.get(key)) for row in rows]
    finite = [value for value in values if value is not None]
    return sum(finite) / len(finite) if finite else None


def clipped_rate(rows: Sequence[Mapping[str, str]]) -> float | None:
    values = [parse_bool(row.get("gradient_was_clipped")) for row in rows]
    known = [value for value in values if value is not None]
    if not known:
        return None
    return sum(1 for value in known if value) / len(known)


def file_age(path: Path, now: datetime) -> float | None:
    try:
        modified = datetime.fromtimestamp(path.stat().st_mtime).astimezone()
    except OSError:
        return None
    return max(0.0, (now - modified).total_seconds())


def load_completion(path: Path) -> Mapping[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, Mapping) else None


def format_duration(seconds: float) -> str:
    return str(timedelta(seconds=max(0, int(round(seconds)))))


def format_number(value: float | int | None, precision: int = 6) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{precision}g}"
    return str(value)


def render(args: argparse.Namespace) -> str:
    now = datetime.now().astimezone()
    protocol_path = args.protocol.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    protocol = load_protocol(protocol_path)
    stage = protocol.stage(args.stage)
    train_log = output_dir / "train_log.csv"
    validation_log = output_dir / "validation_log.csv"
    train_rows = csv_tail(train_log, args.recent_window)
    train = train_rows[-1] if train_rows else {}
    stage_step = optional_int(train.get("stage_step")) or 0
    global_step = optional_int(train.get("global_step"))
    epoch = optional_int(train.get("epoch"))
    batch_in_epoch = optional_int(train.get("batch_in_epoch"))
    batches_per_epoch = stage.samples_per_epoch // stage.batch_size
    validation = summarize_validations(read_validation_rows(validation_log), stage)

    process: tuple[int, float] | None
    if args.pid is not None:
        elapsed = process_elapsed(args.pid)
        process = (args.pid, elapsed) if elapsed is not None else None
    else:
        process = discover_training_process(
            output_dir=output_dir,
            protocol_path=protocol_path,
            stage_name=stage.name,
        )
    completion = load_completion(output_dir / "stage_completion.json")
    recent_age = file_age(train_log, now)
    if completion is not None:
        status = f"COMPLETE ({completion.get('final_stage_step', stage_step)} steps)"
    elif process is not None:
        status = "RUNNING"
    elif (output_dir / "checkpoint.pt").is_file():
        status = "STOPPED / FINAL CHECKPOINT PRESENT"
    elif recent_age is not None and recent_age <= args.stale_after:
        status = "ACTIVE (recent log; PID not visible)"
    elif output_dir.exists():
        status = "WAITING / STALE"
    else:
        status = "NOT STARTED"

    floor = stage.monitor.min_stage_steps
    plateau_target = earliest_plateau_stop_step(
        current_step=stage_step,
        last_validation_step=validation.last_stage_step,
        min_stage_steps=floor,
        validation_every_steps=stage.validation_every_steps,
        patience=stage.monitor.patience,
        without_selection=validation.without_selection,
        max_steps=stage.max_steps,
    )
    hard_progress = 100.0 * stage_step / stage.max_steps
    floor_progress = 100.0 if floor == 0 else min(100.0, 100.0 * stage_step / floor)
    equivalent_epoch = stage_step / batches_per_epoch
    latest = validation.latest_primary
    best = validation.best_primary

    eta_lines: list[str] = []
    if completion is not None:
        eta_lines.append("ETA: complete")
    elif process is None or stage_step < args.eta_min_steps:
        eta_lines.append(
            f"ETA: collecting timing (requires visible PID and {args.eta_min_steps} steps)"
        )
    else:
        seconds_per_step = process[1] / max(stage_step, 1)
        plateau_seconds = max(0, plateau_target - stage_step) * seconds_per_step
        hard_seconds = max(0, stage.max_steps - stage_step) * seconds_per_step
        eta_lines.extend(
            [
                (
                    f"speed: {seconds_per_step:.3f} s/step "
                    f"({1.0 / seconds_per_step:.3f} steps/s)"
                ),
                (
                    f"plateau ETA: {format_duration(plateau_seconds)} "
                    f"→ {(now + timedelta(seconds=plateau_seconds)):%Y-%m-%d %H:%M:%S}"
                ),
                (
                    f"hard-limit ETA: {format_duration(hard_seconds)} "
                    f"→ {(now + timedelta(seconds=hard_seconds)):%Y-%m-%d %H:%M:%S}"
                ),
            ]
        )

    lines = [
        "=" * 112,
        "ForceAwareACT Staged Training Monitor",
        "=" * 112,
        f"time: {now:%Y-%m-%d %H:%M:%S %Z}",
        f"stage: {stage.name} | status: {status} | PID: {process[0] if process else '-'}",
        f"output: {output_dir}",
        (
            f"step: {stage_step}/{stage.max_steps} ({hard_progress:.2f}%) | "
            f"global_step: {global_step if global_step is not None else '-'}"
        ),
        (
            f"equivalent epoch: {equivalent_epoch:.2f} | "
            f"epoch/batch: {epoch or '-'}/{batch_in_epoch or '-'} "
            f"(batches/epoch={batches_per_epoch})"
        ),
        (
            f"minimum floor: {stage_step}/{floor} ({floor_progress:.2f}%) | "
            f"patience active: {'yes' if stage_step >= floor else 'no'}"
        ),
        (
            f"validation: {validation.count} | "
            f"without selected checkpoint: "
            f"{validation.without_selection}/{stage.monitor.patience} | "
            f"earliest plateau stop: {plateau_target}"
        ),
        *eta_lines,
        "-" * 112,
        f"{'training metric':<24}{'latest':>18}{f'recent-{len(train_rows)} mean':>26}",
    ]
    for metric in TRAIN_METRICS:
        lines.append(
            f"{metric:<24}"
            f"{format_number(optional_float(train.get(metric))):>18}"
            f"{format_number(recent_mean(train_rows, metric)):>26}"
        )
    clip_rate = clipped_rate(train_rows)
    clip_text = "-" if clip_rate is None else f"{100.0 * clip_rate:.4g}%"
    lines.append(
        f"{'gradient clipped rate':<24}{'-':>18}"
        f"{clip_text:>26}"
    )
    lines.extend(
        [
            "-" * 112,
            (
                f"primary validation ({stage.monitor.primary_domain}/"
                f"{stage.monitor.metric}): "
                f"latest={format_number(optional_float(latest.get(stage.monitor.metric)))} "
                f"at step={latest.get('stage_step') or '-'} | "
                f"best={format_number(optional_float(best.get(stage.monitor.metric)))} "
                f"at step={best.get('stage_step') or '-'}"
            ),
            (
                f"latest decision: selected={latest.get('selected') or '-'} "
                f"retention={latest.get('retention_passed') or '-'} "
                f"reason={latest.get('decision_reason') or '-'}"
            ),
            f"learning rates: {latest.get('learning_rates') or '-'}",
            (
                f"log age: {format_duration(recent_age) if recent_age is not None else '-'} | "
                f"validation cadence: {stage.validation_every_steps} steps"
            ),
        ]
    )
    if not args.no_gpu:
        lines.extend(["-" * 112, "GPU:", *(f"  {line}" for line in gpu_lines())])
    if not train_log.is_file():
        lines.append("NOTE: train_log.csv 尚未生成；训练启动后会自动出现。")
    lines.append("=" * 112)
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.interval <= 0:
        raise ValueError("--interval must be positive")
    if args.recent_window <= 0:
        raise ValueError("--recent-window must be positive")
    if args.eta_min_steps <= 0:
        raise ValueError("--eta-min-steps must be positive")
    if args.stale_after <= 0:
        raise ValueError("--stale-after must be positive")
    try:
        while True:
            if args.watch:
                print("\033[2J\033[H", end="")
            print(render(args), flush=True)
            if not args.watch:
                return 0
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\n监控已停止；训练进程未被中断。")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
