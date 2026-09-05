#!/usr/bin/env python3
"""Plan or run the four-model nominal scripted200 rollout pilot."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
ROLLOUT_SCRIPT = REPO_ROOT / "scripts" / "run_mujoco_policy_rollout.py"
PILOT_VERSION = "sampling_ablation_nominal_center_pilot_v1"
ROLLOUT_PROTOCOL_VERSION = "collector_terminal_hold_rollout_v5"
DEFAULT_CHECKPOINT_ROOT = (
    REPO_ROOT
    / "outputs"
    / "moving_hole_fixed_peg_scripted200"
    / "sampling_ablation_b16_earlystop"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "runs"
    / "moving_hole_fixed_peg_sampling_ablation_nominal_pilot_seed0"
)
MODEL_CONTRACTS = (
    (
        "official_episodic",
        "01_official_act_episodic_seed0/best_policy.pt",
        "official_act_single_arm_v1",
        "episodic_uniform",
    ),
    (
        "official_dense",
        "02_official_act_dense_seed0/best_policy.pt",
        "official_act_single_arm_v1",
        "dense_windows",
    ),
    (
        "contact_dense",
        "03_contact_cvae_dense_seed0/best.pt",
        "act_aligned_contact_cvae_highrate_force_v2",
        "dense_windows",
    ),
    (
        "contact_episodic",
        "04_contact_cvae_episodic_seed0/best.pt",
        "act_aligned_contact_cvae_highrate_force_v2",
        "episodic_uniform",
    ),
)


@dataclass(frozen=True)
class NominalPilotSpec:
    model_id: str
    checkpoint: Path
    architecture_version: str
    train_sampling_mode: str


def build_specs(checkpoint_root: Path) -> tuple[NominalPilotSpec, ...]:
    return tuple(
        NominalPilotSpec(
            model_id=model_id,
            checkpoint=checkpoint_root / relative_path,
            architecture_version=architecture_version,
            train_sampling_mode=train_sampling_mode,
        )
        for (
            model_id,
            relative_path,
            architecture_version,
            train_sampling_mode,
        ) in MODEL_CONTRACTS
    )


def validate_spec_files(specs: Sequence[NominalPilotSpec]) -> None:
    errors: list[str] = []
    for spec in specs:
        if not spec.checkpoint.is_file():
            errors.append(f"checkpoint is missing: {spec.checkpoint}")
            continue
        metadata_path = spec.checkpoint.parent / "run_metadata.json"
        if not metadata_path.is_file():
            errors.append(f"run metadata is missing: {metadata_path}")
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        actual_mode = metadata.get("train_sampling_mode")
        if actual_mode != spec.train_sampling_mode:
            errors.append(
                f"{spec.model_id} train_sampling_mode: expected "
                f"{spec.train_sampling_mode!r}, got {actual_mode!r}"
            )
        model_metadata = metadata.get("model")
        actual_architecture = (
            model_metadata.get("architecture_version")
            if isinstance(model_metadata, dict)
            else None
        )
        if actual_architecture != spec.architecture_version:
            errors.append(
                f"{spec.model_id} architecture_version: expected "
                f"{spec.architecture_version!r}, got {actual_architecture!r}"
            )
    if errors:
        raise ValueError("invalid pilot inputs:\n- " + "\n- ".join(errors))


def build_rollout_command(
    args: argparse.Namespace,
    spec: NominalPilotSpec,
) -> list[str]:
    command = [
        sys.executable,
        str(ROLLOUT_SCRIPT),
        "--checkpoint",
        str(spec.checkpoint),
        "--task-profile",
        "moving_hole_fixed_peg",
        "--terminal-hold-mode",
        "collector",
        "--output-dir",
        str(args.output_dir / spec.model_id),
        "--device",
        args.device,
        "--contact-latent-mode",
        "zero",
        "--action-mode",
        "joint_pos",
        "--action-select-mode",
        "temporal",
        "--temporal-agg-decay",
        "0.01",
        "--policy-rate-hz",
        "30",
        "--max-rollout-steps",
        str(args.max_rollout_steps),
        "--ema-alpha",
        "1",
        "--max-delta-q",
        "0.02",
        "--force-stop-threshold",
        "40",
        "--safe-force-threshold",
        "40",
        "--hole-offset-x=0.0",
        "--hole-offset-y=0.0",
        "--hole-offset-z=0.0",
        "--record-force-metrics",
        "--seed",
        str(args.seed),
    ]
    if args.save_videos:
        command.append("--save-videos")
    if args.execute:
        command.append("--execute-actions")
    return command


def validate_completed_summary(
    summary: dict[str, Any],
    args: argparse.Namespace,
    spec: NominalPilotSpec,
) -> list[str]:
    expected = {
        "rollout_protocol_version": ROLLOUT_PROTOCOL_VERSION,
        "checkpoint": str(spec.checkpoint.resolve()),
        "rollout_mode": "execute",
        "task_profile": "moving_hole_fixed_peg",
        "policy_wrench_source": "compensated",
        "terminal_hold_mode": "collector",
        "seed": args.seed,
        "action_mode": "joint_pos",
        "action_select_mode": "temporal",
        "temporal_agg_decay": 0.01,
        "contact_latent_mode": "zero",
        "deployment_latent_source": (
            "none" if spec.model_id.startswith("official") else "zero"
        ),
        "policy_rate_hz": 30.0,
        "max_rollout_steps": args.max_rollout_steps,
        "ema_alpha": 1.0,
        "max_delta_q": 0.02,
        "force_stop_threshold": 40.0,
        "safe_force_threshold": 40.0,
        "hole_offset_x": 0.0,
        "hole_offset_y": 0.0,
        "hole_offset_z": 0.0,
        "force_metrics_recorded": True,
    }
    return [
        f"{key}: expected {value!r}, got {summary.get(key)!r}"
        for key, value in expected.items()
        if summary.get(key) != value
    ]


def _aggregate_row(spec: NominalPilotSpec, summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "model_id": spec.model_id,
        "architecture_version": spec.architecture_version,
        "train_sampling_mode": spec.train_sampling_mode,
        "success": summary.get("success"),
        "task_success_site_reached": summary.get("task_success_site_reached"),
        "terminal_hold_accepted": summary.get("terminal_hold_accepted"),
        "terminal_hold_completion_reason": summary.get(
            "terminal_hold_completion_reason"
        ),
        "safe_success": summary.get("safe_success"),
        "stop_reason": summary.get("stop_reason"),
        "steps_executed": summary.get("steps_executed"),
        "success_time": summary.get("success_time"),
        "min_peg_to_hole_dist": summary.get("min_peg_to_hole_dist"),
        "final_peg_to_hole_dist": summary.get("final_peg_to_hole_dist"),
        "max_force_norm": summary.get("max_force_norm"),
        "force_metrics_compensated_max_force_norm": summary.get(
            "force_metrics_compensated_max_force_norm"
        ),
        "policy_inference_time_ms_p95": summary.get(
            "policy_inference_time_ms_p95"
        ),
        "summary_json": summary.get("summary_json"),
    }


def _write_aggregate(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plan the fixed center-point four-model rollout by default; pass "
            "--execute to apply policy actions and run it sequentially."
        )
    )
    parser.add_argument(
        "--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--max-rollout-steps", type=int, default=900)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--save-videos",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually run all four closed-loop rollouts; omitted means plan-only.",
    )
    args = parser.parse_args(argv)
    args.checkpoint_root = args.checkpoint_root.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    if args.max_rollout_steps <= 0:
        parser.error("--max-rollout-steps must be positive")
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    specs = build_specs(args.checkpoint_root)
    try:
        validate_spec_files(specs)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    commands = [build_rollout_command(args, spec) for spec in specs]
    plan = {
        "pilot_version": PILOT_VERSION,
        "execute_requested": args.execute,
        "checkpoint_root": str(args.checkpoint_root),
        "output_dir": str(args.output_dir),
        "seed": args.seed,
        "max_rollout_steps": args.max_rollout_steps,
        "save_videos": args.save_videos,
        "fixed_contract": {
            "task_profile": "moving_hole_fixed_peg",
            "terminal_hold_mode": "collector",
            "offset_xyz_m": [0.0, 0.0, 0.0],
            "action_select_mode": "temporal",
            "temporal_agg_decay": 0.01,
            "contact_latent_mode": "zero",
            "policy_rate_hz": 30.0,
            "max_delta_q": 0.02,
            "ema_alpha": 1.0,
            "hard_stop_force_n": 40.0,
            "force_metrics": True,
            "axial_push": False,
        },
        "models": [
            {
                **asdict(spec),
                "checkpoint": str(spec.checkpoint),
                "command": shlex.join(command),
            }
            for spec, command in zip(specs, commands)
        ],
    }
    print(json.dumps(plan, indent=2, sort_keys=True))
    if not args.execute:
        print("plan_only=true; no rollout actions were executed")
        return 0

    occupied = [
        args.output_dir / spec.model_id
        for spec in specs
        if (args.output_dir / spec.model_id).exists()
    ]
    if occupied:
        print(
            "error: refusing to overwrite existing model output directories:\n- "
            + "\n- ".join(str(path) for path in occupied),
            file=sys.stderr,
        )
        return 2
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "pilot_plan.json").write_text(
        json.dumps(plan, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    environment = os.environ.copy()
    environment["MUJOCO_GL"] = "egl"
    existing_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        str(REPO_ROOT / "src")
        if not existing_pythonpath
        else str(REPO_ROOT / "src") + os.pathsep + existing_pythonpath
    )
    rows: list[dict[str, Any]] = []
    for index, (spec, command) in enumerate(zip(specs, commands), start=1):
        print(f"[{index}/{len(specs)}] starting {spec.model_id}", flush=True)
        completed = subprocess.run(command, cwd=REPO_ROOT, env=environment)
        if completed.returncode != 0:
            print(
                f"error: {spec.model_id} exited with {completed.returncode}",
                file=sys.stderr,
            )
            return completed.returncode
        summary_path = args.output_dir / spec.model_id / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        errors = validate_completed_summary(summary, args, spec)
        if errors:
            print(
                f"error: {spec.model_id} summary contract mismatch:\n- "
                + "\n- ".join(errors),
                file=sys.stderr,
            )
            return 2
        rows.append(_aggregate_row(spec, summary))
        _write_aggregate(args.output_dir / "aggregate.csv", rows)
        print(f"[{index}/{len(specs)}] completed {spec.model_id}", flush=True)

    print(f"aggregate_csv={args.output_dir / 'aggregate.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
