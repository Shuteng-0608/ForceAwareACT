#!/usr/bin/env python3
"""Run Contact-CVAE zero/prior over chunk indices 1..10 and temporal aggregation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shlex
import signal
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAINING_ROOT = Path(
    "outputs/hole_random_60mm_hmj/earlystop_train90_splitseed20260716_run1"
)
DEFAULT_STATS = Path(
    "outputs/hole_random_60mm_hmj/train90_val10_seed20260716/"
    "normalization_stats_action_train90.pt"
)
DEFAULT_POINTS = Path("configs/experiments/fibonacci_disk_100_r60mm.csv")
DEFAULT_MODEL_XML = Path("../arm_teleop/model/pangu_all_right.xml")
DEFAULT_MODES = tuple(str(index) for index in range(1, 11)) + ("temporal",)


@dataclass(frozen=True)
class ModelSpec:
    key: str
    checkpoint_relative: str
    contact_latent_mode: str


@dataclass(frozen=True)
class ExperimentSpec:
    key: str
    model_key: str
    checkpoint_relative: str
    contact_latent_mode: str
    action_select_mode: str


MODEL_SPECS = (
    ModelSpec(
        key="contact_cvae_zero",
        checkpoint_relative="formal/contact_cvae_zero_seed0/checkpoint_best.pt",
        contact_latent_mode="zero",
    ),
    ModelSpec(
        key="contact_cvae_prior",
        checkpoint_relative="formal/contact_cvae_prior_seed0/checkpoint_best.pt",
        contact_latent_mode="prior",
    ),
)


def timestamp() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def write_status(path: Path, status: str) -> None:
    atomic_write_text(path, f"{status}\t{timestamp()}\n")


def read_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def count_csv_rows(path: Path) -> int:
    try:
        with path.open(newline="", encoding="utf-8") as stream:
            return sum(1 for _ in csv.DictReader(stream))
    except OSError:
        return 0


def mode_token(mode: str) -> str:
    return "temporal" if mode == "temporal" else f"action_{int(mode):02d}"


def validate_action_modes(modes: Sequence[str], chunk_len: int) -> tuple[str, ...]:
    if not modes:
        raise ValueError("at least one --action-select-mode is required")
    normalized: list[str] = []
    for mode in modes:
        if mode == "temporal":
            normalized.append(mode)
            continue
        if not mode.isdecimal() or not 1 <= int(mode) <= chunk_len:
            raise ValueError(
                f"invalid action selection mode {mode!r}; use 1..{chunk_len} or temporal"
            )
        normalized.append(str(int(mode)))
    if len(set(normalized)) != len(normalized):
        raise ValueError("action selection modes must not contain duplicates")
    return tuple(normalized)


def experiment_specs(modes: Sequence[str]) -> tuple[ExperimentSpec, ...]:
    return tuple(
        ExperimentSpec(
            key=f"{model.key}__{mode_token(mode)}",
            model_key=model.key,
            checkpoint_relative=model.checkpoint_relative,
            contact_latent_mode=model.contact_latent_mode,
            action_select_mode=mode,
        )
        for mode in modes
        for model in MODEL_SPECS
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        nargs="?",
        choices=("preflight", "dry-run", "run"),
        default="preflight",
    )
    parser.add_argument("--training-root", type=Path, default=DEFAULT_TRAINING_ROOT)
    parser.add_argument("--normalization-stats", type=Path, default=DEFAULT_STATS)
    parser.add_argument("--task-points-csv", type=Path, default=DEFAULT_POINTS)
    parser.add_argument("--model-xml", type=Path, default=DEFAULT_MODEL_XML)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=(
            DEFAULT_TRAINING_ROOT
            / "rollouts/fibonacci_disk_100_r60mm_contact_action_sweep"
        ),
    )
    parser.add_argument(
        "--action-select-modes",
        nargs="+",
        default=list(DEFAULT_MODES),
        help="1-based chunk positions and/or temporal (default: 1..10 temporal).",
    )
    parser.add_argument("--temporal-agg-decay", type=float, default=0.3)
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    parser.add_argument("--mujoco-gl", default="egl")
    parser.add_argument("--rollout-seed-base", type=int, default=31000)
    parser.add_argument("--chunk-len", type=int, default=10)
    parser.add_argument("--force-window-len", type=int, default=20)
    parser.add_argument("--force-window-duration", type=float, default=0.25)
    parser.add_argument("--policy-rate-hz", type=float, default=30.0)
    parser.add_argument("--max-rollout-steps", type=int, default=900)
    parser.add_argument("--max-delta-q", type=float, default=0.02)
    parser.add_argument("--force-stop-threshold", type=float, default=1000.0)
    parser.add_argument("--success-distance-threshold", type=float, default=0.005)
    parser.add_argument("--success-lateral-threshold", type=float, default=0.006)
    parser.add_argument("--success-force-threshold", type=float, default=40.0)
    parser.add_argument("--success-hold-steps", type=int, default=15)
    parser.add_argument("--skip-cuda-check", action="store_true")
    parser.add_argument("--save-videos", action="store_true")
    return parser


def checkpoint_path(args: argparse.Namespace, spec: ExperimentSpec) -> Path:
    return args.training_root / spec.checkpoint_relative


def read_fixed_points(path: Path) -> list[tuple[float, float, float]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError(f"point CSV has no header: {path}")
        required = {"hole_offset_x", "hole_offset_y", "hole_offset_z"}
        missing = required - set(reader.fieldnames)
        if missing:
            raise ValueError(f"point CSV is missing columns: {sorted(missing)}")
        rows = list(reader)
    points: list[tuple[float, float, float]] = []
    for index, row in enumerate(rows, start=1):
        if row.get("point_index") and int(row["point_index"]) != index:
            raise ValueError("point_index must be consecutive and start at 1")
        point = tuple(
            float(row[key])
            for key in ("hole_offset_x", "hole_offset_y", "hole_offset_z")
        )
        if not all(math.isfinite(value) for value in point):
            raise ValueError(f"point {index} contains a non-finite offset")
        points.append(point)
    if len(points) != 100 or len(set(points)) != 100:
        raise ValueError("expected exactly 100 unique fixed points")
    maximum_radius = max(math.hypot(point[0], point[2]) for point in points)
    if not 0.059 <= maximum_radius <= 0.060000001:
        raise ValueError(f"fixed points are not the expected 60 mm disk: {maximum_radius}")
    return points


def validate_checkpoint(path: Path, expected_mode: str) -> None:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = checkpoint.get("config", {})
    if config.get("policy_variant") != "force_aware_contact_cvae":
        raise ValueError(f"checkpoint is not Contact-CVAE: {path}")
    if config.get("action_mode") != "action" or config.get("chunk_len") != 10:
        raise ValueError(f"checkpoint action/chunk metadata mismatch: {path}")
    if checkpoint.get("stop_reason") != "best_validation_metric":
        raise ValueError(f"not a validation-best checkpoint: {path}")
    configured_mode = config.get("validation_deployment_mode")
    if configured_mode is not None and configured_mode != expected_mode:
        raise ValueError(
            f"checkpoint deployment mode mismatch: expected {expected_mode}, "
            f"got {configured_mode}"
        )


def validate_inputs(
    args: argparse.Namespace, specs: Sequence[ExperimentSpec]
) -> dict[str, object]:
    if args.chunk_len <= 0 or args.max_rollout_steps <= 0:
        raise ValueError("chunk and rollout lengths must be positive")
    if args.temporal_agg_decay < 0 or not math.isfinite(args.temporal_agg_decay):
        raise ValueError("temporal aggregation decay must be finite and non-negative")
    if args.max_delta_q <= 0 or args.force_stop_threshold <= 0:
        raise ValueError("control and force-stop thresholds must be positive")
    if args.success_force_threshold <= 0 or args.success_hold_steps <= 0:
        raise ValueError("success thresholds must be positive")

    unique_checkpoints = {
        (checkpoint_path(args, spec), spec.contact_latent_mode) for spec in specs
    }
    required = [args.normalization_stats, args.task_points_csv, args.model_xml]
    required.extend(path for path, _ in unique_checkpoints)
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("required files missing:\n" + "\n".join(map(str, missing)))

    points = read_fixed_points(args.task_points_csv)
    stats = torch.load(args.normalization_stats, map_location="cpu", weights_only=False)
    expected_stats = {
        "action_mode": "action",
        "chunk_len": args.chunk_len,
        "force_window_len": args.force_window_len,
        "force_window_duration": args.force_window_duration,
        "image_size": (224, 224),
        "camera_names": ("ee_cam", "base_top_cam"),
    }
    for key, expected in expected_stats.items():
        actual = stats.get(key)
        if key in {"image_size", "camera_names"} and actual is not None:
            actual = tuple(actual)
        if actual != expected:
            raise ValueError(
                f"normalization stats {key} mismatch: expected {expected!r}, got {actual!r}"
            )
    for path, mode in unique_checkpoints:
        validate_checkpoint(path, mode)
    if args.device == "cuda" and not args.skip_cuda_check:
        if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
            raise RuntimeError("CUDA preflight failed: no CUDA device is available")

    metadata = {
        "point_count": len(points),
        "point_sha256": hashlib.sha256(args.task_points_csv.read_bytes()).hexdigest(),
        "maximum_radius_mm": max(
            math.hypot(point[0], point[2]) for point in points
        )
        * 1000.0,
        "experiment_count": len(specs),
        "total_rollouts": len(specs) * len(points),
    }
    print("preflight=passed")
    print(json.dumps(metadata, indent=2))
    return metadata


def output_dir(args: argparse.Namespace, spec: ExperimentSpec) -> Path:
    return args.output_root / spec.model_key / mode_token(spec.action_select_mode)


def build_grid_command(args: argparse.Namespace, spec: ExperimentSpec) -> list[str]:
    command = [
        args.python_executable,
        "scripts/run_mujoco_hole_grid.py",
        "--sampling-mode",
        "file",
        "--task-points-csv",
        str(args.task_points_csv),
        "--num-points",
        "100",
        "--x-min",
        "-0.06",
        "--x-max",
        "0.06",
        "--z-min",
        "-0.06",
        "--z-max",
        "0.06",
        "--checkpoint",
        str(checkpoint_path(args, spec)),
        "--normalization-stats",
        str(args.normalization_stats),
        "--device",
        args.device,
        "--model-xml",
        str(args.model_xml),
        "--contact-latent-mode",
        spec.contact_latent_mode,
        "--action-mode",
        "action",
        "--action-select-mode",
        spec.action_select_mode,
        "--temporal-agg-decay",
        str(args.temporal_agg_decay),
        "--chunk-len",
        str(args.chunk_len),
        "--force-window-len",
        str(args.force_window_len),
        "--force-window-duration",
        str(args.force_window_duration),
        "--policy-rate-hz",
        str(args.policy_rate_hz),
        "--max-rollout-steps",
        str(args.max_rollout_steps),
        "--max-delta-q",
        str(args.max_delta_q),
        "--force-stop-threshold",
        str(args.force_stop_threshold),
        "--hole-axis-world",
        "0",
        "-1",
        "0",
        "--hole-site-name",
        "hole_goal_site",
        "--hole-body-name",
        "wall_task",
        "--hole-offset-frame",
        "world",
        "--y-offset",
        "0",
        "--success-distance-threshold",
        str(args.success_distance_threshold),
        "--success-lateral-threshold",
        str(args.success_lateral_threshold),
        "--success-force-threshold",
        str(args.success_force_threshold),
        "--success-hold-steps",
        str(args.success_hold_steps),
        "--output-root",
        str(output_dir(args, spec)),
        "--base-seed",
        "0",
        "--point-set-seed",
        "0",
        "--rollout-seed-base",
        str(args.rollout_seed_base),
        "--python-executable",
        args.python_executable,
        "--skip-existing",
        "--continue-on-error",
        "--no-plot-results",
    ]
    if args.mujoco_gl:
        command.extend(["--mujoco-gl", args.mujoco_gl])
    if args.save_videos:
        command.append("--save-videos")
    return command


def suite_plan(
    args: argparse.Namespace,
    specs: Sequence[ExperimentSpec],
    point_metadata: dict[str, object],
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "created_at": timestamp(),
        "output_root": str(args.output_root.resolve()),
        "training_root": str(args.training_root.resolve()),
        "normalization_stats": str(args.normalization_stats.resolve()),
        "task_points_csv": str(args.task_points_csv.resolve()),
        **point_metadata,
        "models": [asdict(model) for model in MODEL_SPECS],
        "experiments": [
            {
                **asdict(spec),
                "checkpoint": str(checkpoint_path(args, spec).resolve()),
                "output_dir": str(output_dir(args, spec).resolve()),
            }
            for spec in specs
        ],
        "protocol": {
            "action_mode": "action",
            "action_select_modes": list(args.action_select_modes),
            "temporal_agg_decay": args.temporal_agg_decay,
            "chunk_len": args.chunk_len,
            "force_window_len": args.force_window_len,
            "force_window_duration": args.force_window_duration,
            "policy_rate_hz": args.policy_rate_hz,
            "max_rollout_steps": args.max_rollout_steps,
            "max_delta_q": args.max_delta_q,
            "force_stop_threshold": args.force_stop_threshold,
            "success_distance_threshold": args.success_distance_threshold,
            "success_lateral_threshold": args.success_lateral_threshold,
            "success_force_threshold": args.success_force_threshold,
            "success_hold_steps": args.success_hold_steps,
            "rollout_seed_base": args.rollout_seed_base,
            "device": args.device,
            "model_xml": str(args.model_xml.resolve()),
            "mujoco_gl": args.mujoco_gl,
            "save_videos": args.save_videos,
        },
    }


def write_or_validate_plan(args: argparse.Namespace, plan: dict[str, object]) -> Path:
    path = args.output_root / "suite_plan.json"
    if path.is_file():
        existing = read_json(path)
        old = {key: value for key, value in existing.items() if key != "created_at"}
        new = {key: value for key, value in plan.items() if key != "created_at"}
        if old != new:
            raise ValueError(f"existing suite plan does not match requested protocol: {path}")
        return path
    atomic_write_text(path, json.dumps(plan, indent=2, ensure_ascii=False) + "\n")
    return path


def experiment_is_complete(args: argparse.Namespace, spec: ExperimentSpec) -> bool:
    root = output_dir(args, spec)
    runs = read_json(root / "grid_manifest.json").get("runs", [])
    if not isinstance(runs, list):
        return False
    errors = sum(
        isinstance(run, dict) and run.get("status") == "process_error" for run in runs
    )
    return len(runs) == 100 and errors == 0 and count_csv_rows(root / "grid_summary.csv") == 100


def run_experiment(
    args: argparse.Namespace,
    spec: ExperimentSpec,
    pipeline_dir: Path,
    runner_log,
) -> bool:
    root = output_dir(args, spec)
    root.mkdir(parents=True, exist_ok=True)
    status_path = pipeline_dir / f"{spec.key}.status"
    if experiment_is_complete(args, spec):
        write_status(status_path, "completed")
        print(f"[{timestamp()}] skip complete experiment={spec.key}")
        return True

    command = build_grid_command(args, spec)
    atomic_write_text(
        root / "rollout_command.sh",
        "cd "
        + shlex.quote(str(REPO_ROOT))
        + "\nPYTHONPATH=src "
        + " ".join(shlex.quote(part) for part in command)
        + "\n",
    )
    atomic_write_text(pipeline_dir / "current_experiment", spec.key + "\n")
    atomic_write_text(pipeline_dir / f"{spec.key}.started_at", timestamp() + "\n")
    write_status(status_path, "running")
    message = (
        f"[{timestamp()}] start experiment={spec.key} "
        f"mode={spec.action_select_mode}"
    )
    print(message)
    runner_log.write(message + "\n")
    runner_log.flush()

    environment = os.environ.copy()
    environment["PYTHONPATH"] = "src"
    if args.mujoco_gl:
        environment["MUJOCO_GL"] = args.mujoco_gl
    return_code = 1
    with (root / "console.log").open("a", encoding="utf-8") as console:
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        atomic_write_text(pipeline_dir / "current_grid.pid", f"{process.pid}\n")
        assert process.stdout is not None
        try:
            for line in process.stdout:
                print(line, end="")
                console.write(line)
                console.flush()
            return_code = process.wait()
        except KeyboardInterrupt:
            process.send_signal(signal.SIGINT)
            process.wait()
            raise
        finally:
            (pipeline_dir / "current_grid.pid").unlink(missing_ok=True)

    atomic_write_text(pipeline_dir / f"{spec.key}.finished_at", timestamp() + "\n")
    runs = read_json(root / "grid_manifest.json").get("runs", [])
    runs = runs if isinstance(runs, list) else []
    errors = sum(
        isinstance(run, dict) and run.get("status") == "process_error" for run in runs
    )
    valid = count_csv_rows(root / "grid_summary.csv")
    succeeded = return_code == 0 and len(runs) == 100 and errors == 0 and valid == 100
    atomic_write_text(root / "exit_code.txt", f"{return_code}\n")
    write_status(status_path, "completed" if succeeded else "failed")
    (pipeline_dir / "current_experiment").unlink(missing_ok=True)
    message = (
        f"[{timestamp()}] finish experiment={spec.key} return_code={return_code} "
        f"attempts={len(runs)} valid={valid} process_errors={errors} "
        f"status={'completed' if succeeded else 'failed'}"
    )
    print(message)
    runner_log.write(message + "\n")
    runner_log.flush()
    return succeeded


def run_suite(
    args: argparse.Namespace,
    specs: Sequence[ExperimentSpec],
    point_metadata: dict[str, object],
) -> int:
    args.output_root.mkdir(parents=True, exist_ok=True)
    plan_path = write_or_validate_plan(args, suite_plan(args, specs, point_metadata))
    pipeline_dir = args.output_root / ".pipeline"
    pipeline_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_text(pipeline_dir / "runner.pid", f"{os.getpid()}\n")
    atomic_write_text(pipeline_dir / "started_at", timestamp() + "\n")
    write_status(pipeline_dir / "pipeline.status", "running")
    print(f"suite_plan={plan_path}")

    failures: list[str] = []
    try:
        with (pipeline_dir / "runner.log").open("a", encoding="utf-8") as runner_log:
            for spec in specs:
                if not run_experiment(args, spec, pipeline_dir, runner_log):
                    failures.append(spec.key)
            final_status = "completed" if not failures else "completed_with_errors"
            write_status(pipeline_dir / "pipeline.status", final_status)
            atomic_write_text(pipeline_dir / "finished_at", timestamp() + "\n")
            print(f"[{timestamp()}] suite status={final_status} failures={failures}")
    except KeyboardInterrupt:
        write_status(pipeline_dir / "pipeline.status", "interrupted")
        print("action sweep interrupted; existing point summaries can be resumed")
        return 130
    except Exception:
        write_status(pipeline_dir / "pipeline.status", "failed")
        raise
    finally:
        for name in ("runner.pid", "current_grid.pid", "current_experiment"):
            (pipeline_dir / name).unlink(missing_ok=True)
    return 0 if not failures else 1


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    os.chdir(REPO_ROOT)
    args.action_select_modes = validate_action_modes(
        args.action_select_modes, args.chunk_len
    )
    specs = experiment_specs(args.action_select_modes)
    metadata = validate_inputs(args, specs)
    if args.stage == "preflight":
        return 0
    if args.stage == "dry-run":
        print(f"output_root={args.output_root}")
        for spec in specs:
            print(f"\n[{spec.key}]")
            print(
                "PYTHONPATH=src "
                + " ".join(shlex.quote(part) for part in build_grid_command(args, spec))
            )
        return 0
    return run_suite(args, specs, metadata)


if __name__ == "__main__":
    raise SystemExit(main())
