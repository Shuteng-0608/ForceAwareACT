#!/usr/bin/env python3
"""Run predefined or dynamic fixed-point Q=1 temporal-executor sweeps."""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from dataclasses import asdict, dataclass
from decimal import Decimal
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
ROLLOUT_SCRIPT = REPO_ROOT / "scripts" / "run_mujoco_policy_rollout.py"
PILOT_VERSION = "paired50_q1_temporal_rollout_pilot_v5"
ROLLOUT_PROTOCOL_VERSION = "paired_action_executor_rollout_v3"
DEFAULT_FORCE_HUD_CAMERA = "cctv_cam"
DEFAULT_FORCE_HUD_WIDTH = 1280
DEFAULT_FORCE_HUD_HEIGHT = 720
MODEL_IDS = ("official_act", "highrate_contact_v3")
DEFAULT_OUTPUT_DIR = REPO_ROOT / "runs" / "paired50_q1_temporal_rollout_pilot_b1"
DEFAULT_OFFICIAL_CHECKPOINT = (
    REPO_ROOT / "runs" / "paired50_official_act_formal_e2000_b8_seed0" / "best_policy.pt"
)
DEFAULT_CONTACT_CHECKPOINT = (
    REPO_ROOT
    / "runs"
    / "paired50_highrate_contact_v3_formal_s10000_b8_seed0"
    / "best.pt"
)
DEFAULT_MODEL_XML = REPO_ROOT.parent / "arm_teleop" / "model" / "pangu_all_right.xml"


@dataclass(frozen=True)
class PilotSpec:
    configuration_id: str
    model_id: str
    checkpoint: Path
    executor_id: str
    action_select_mode: str
    signed_decay: Optional[float]


def signed_decay_executor_id(signed_decay: float) -> str:
    """Return a stable filesystem-safe identifier for an arbitrary finite k."""

    signed_decay = float(signed_decay)
    if not math.isfinite(signed_decay):
        raise ValueError("signed decay must be finite")
    if signed_decay == 0.0:
        return "signed_k0p0"
    sign = "m" if signed_decay < 0.0 else "p"
    magnitude = format(Decimal(str(abs(signed_decay))).normalize(), "f")
    if "." not in magnitude:
        magnitude += ".0"
    return f"signed_k{sign}{magnitude.replace('.', 'p')}"


def build_signed_decay_specs(
    official_checkpoint: Path,
    contact_checkpoint: Path,
    signed_decays: Sequence[float],
    model_ids: Sequence[str] = MODEL_IDS,
) -> tuple[PilotSpec, ...]:
    """Build an ordered dynamic sweep without a hard-coded k grid."""

    signed_decays = tuple(float(value) for value in signed_decays)
    if not signed_decays:
        raise ValueError("dynamic signed-decay sweep must not be empty")
    if any(not math.isfinite(value) for value in signed_decays):
        raise ValueError("signed decays must be finite")
    if len(set(signed_decays)) != len(signed_decays):
        raise ValueError("signed decays must not contain duplicates")

    model_ids = tuple(model_ids)
    if not model_ids:
        raise ValueError("dynamic sweep model ids must not be empty")
    if len(set(model_ids)) != len(model_ids):
        raise ValueError("dynamic sweep model ids must not contain duplicates")
    unknown_models = sorted(set(model_ids) - set(MODEL_IDS))
    if unknown_models:
        raise ValueError(f"unknown model ids: {', '.join(unknown_models)}")

    checkpoint_by_model = {
        "official_act": official_checkpoint,
        "highrate_contact_v3": contact_checkpoint,
    }
    specs = []
    for signed_decay in signed_decays:
        executor_id = signed_decay_executor_id(signed_decay)
        for model_id in model_ids:
            specs.append(
                PilotSpec(
                    configuration_id=f"{model_id}__{executor_id}",
                    model_id=model_id,
                    checkpoint=checkpoint_by_model[model_id],
                    executor_id=executor_id,
                    action_select_mode="signed_temporal",
                    signed_decay=signed_decay,
                )
            )
    configuration_ids = [spec.configuration_id for spec in specs]
    if len(set(configuration_ids)) != len(configuration_ids):
        raise ValueError("dynamic sweep produced duplicate configuration ids")
    return tuple(specs)


def build_pilot_specs(
    official_checkpoint: Path,
    contact_checkpoint: Path,
) -> tuple[PilotSpec, ...]:
    """Interleave models within each executor to keep comparisons paired."""
    models = (
        ("official_act", official_checkpoint),
        ("highrate_contact_v3", contact_checkpoint),
    )
    executors = (
        ("signed_km0p01", "signed_temporal", -0.01),
        ("signed_k0p0", "signed_temporal", 0.0),
        ("signed_kp0p01", "signed_temporal", 0.01),
        ("signed_kp0p03", "signed_temporal", 0.03),
        ("signed_kp0p04", "signed_temporal", 0.04),
        ("signed_kp0p045", "signed_temporal", 0.045),
        ("signed_kp0p05", "signed_temporal", 0.05),
        ("signed_kp0p055", "signed_temporal", 0.055),
        ("signed_kp0p06", "signed_temporal", 0.06),
        ("signed_kp0p07", "signed_temporal", 0.07),
        ("signed_kp0p08", "signed_temporal", 0.08),
        ("signed_kp0p1", "signed_temporal", 0.1),
        ("signed_kp0p2", "signed_temporal", 0.2),
        ("signed_kp0p3", "signed_temporal", 0.3),
        ("latest_only", "latest_only", None),
    )
    return tuple(
        PilotSpec(
            configuration_id=f"{model_id}__{executor_id}",
            model_id=model_id,
            checkpoint=checkpoint,
            executor_id=executor_id,
            action_select_mode=action_select_mode,
            signed_decay=signed_decay,
        )
        for executor_id, action_select_mode, signed_decay in executors
        for model_id, checkpoint in models
    )


def build_rollout_command(args: argparse.Namespace, spec: PilotSpec) -> list[str]:
    output_dir = args.output_dir / spec.configuration_id
    command = [
        sys.executable,
        str(ROLLOUT_SCRIPT),
        "--checkpoint",
        str(spec.checkpoint),
        "--model-xml",
        str(args.model_xml),
        "--output-dir",
        str(output_dir),
        "--device",
        args.device,
        "--contact-latent-mode",
        "zero",
        "--action-mode",
        "joint_pos",
        "--action-select-mode",
        spec.action_select_mode,
        "--policy-rate-hz",
        "30",
        "--max-rollout-steps",
        str(args.max_rollout_steps),
        "--ema-alpha",
        "1",
        "--max-delta-q",
        "0.02",
        "--force-stop-threshold",
        "100",
        "--safe-force-threshold",
        "40",
        "--success-distance-threshold",
        "0.003",
        "--success-dwell-time",
        "0.1",
        f"--hole-offset-x={float(args.hole_offset_x)!r}",
        f"--hole-offset-y={float(args.hole_offset_y)!r}",
        f"--hole-offset-z={float(args.hole_offset_z)!r}",
        "--seed",
        str(args.seed),
        "--execute-actions",
    ]
    if spec.signed_decay is not None:
        command.append(f"--temporal-agg-decay={float(spec.signed_decay)!r}")
    if args.save_videos:
        command.append("--save-videos")
    if args.save_force_hud_videos:
        command.extend(
            (
                "--save-force-hud-video",
                "--force-hud-camera",
                args.force_hud_camera,
                "--force-hud-width",
                str(args.force_hud_width),
                "--force-hud-height",
                str(args.force_hud_height),
                "--force-hud-primary-wrench",
                args.force_hud_primary_wrench,
            )
        )
    return command


def _resolved(path: Path) -> str:
    return str(path.expanduser().resolve())


def validate_completed_summary(
    summary: dict[str, Any],
    args: argparse.Namespace,
    spec: PilotSpec,
) -> list[str]:
    expected = {
        "rollout_protocol_version": ROLLOUT_PROTOCOL_VERSION,
        "checkpoint": _resolved(spec.checkpoint),
        "model_xml": _resolved(args.model_xml),
        "rollout_mode": "execute",
        "seed": args.seed,
        "action_mode": "joint_pos",
        "action_select_mode": spec.action_select_mode,
        "policy_query_interval": 1,
        "contact_latent_mode": "zero",
        "policy_rate_hz": 30.0,
        "max_rollout_steps": args.max_rollout_steps,
        "ema_alpha": 1.0,
        "max_delta_q": 0.02,
        "force_stop_threshold": 100.0,
        "safe_force_threshold": 40.0,
        "hole_offset_x": args.hole_offset_x,
        "hole_offset_y": args.hole_offset_y,
        "hole_offset_z": args.hole_offset_z,
    }
    errors = [
        f"{key}: expected {value!r}, got {summary.get(key)!r}"
        for key, value in expected.items()
        if summary.get(key) != value
    ]
    expected_signed_decay = spec.signed_decay
    if summary.get("temporal_signed_decay_equivalent") != expected_signed_decay:
        errors.append(
            "temporal_signed_decay_equivalent: expected "
            f"{expected_signed_decay!r}, got "
            f"{summary.get('temporal_signed_decay_equivalent')!r}"
        )
    expected_endpoint = "newest" if spec.action_select_mode == "latest_only" else None
    if summary.get("temporal_endpoint") != expected_endpoint:
        errors.append(
            f"temporal_endpoint: expected {expected_endpoint!r}, got "
            f"{summary.get('temporal_endpoint')!r}"
        )
    if args.save_force_hud_videos:
        expected_hud_path = (
            args.output_dir
            / spec.configuration_id
            / "videos"
            / f"{args.force_hud_camera}_force_hud.mp4"
        ).resolve()
        expected_hud = {
            "force_hud_video_saved": True,
            "force_hud_video_path": str(expected_hud_path),
            "force_hud_camera": args.force_hud_camera,
            "force_hud_resolution": [
                args.force_hud_width,
                args.force_hud_height,
            ],
            "force_hud_primary_wrench": args.force_hud_primary_wrench,
            "force_hud_interval_sampling": (
                "current_policy_state_plus_every_executed_physics_step"
            ),
        }
        errors.extend(
            f"{key}: expected {value!r}, got {summary.get(key)!r}"
            for key, value in expected_hud.items()
            if summary.get(key) != value
        )
        frame_count = summary.get("force_hud_video_frame_count")
        if not isinstance(frame_count, int) or frame_count <= 0:
            errors.append(
                "force_hud_video_frame_count: expected a positive integer, "
                f"got {frame_count!r}"
            )
        if not expected_hud_path.is_file():
            errors.append(f"force HUD video is missing: {expected_hud_path}")
        elif expected_hud_path.stat().st_size <= 0:
            errors.append(f"force HUD video is empty: {expected_hud_path}")
    return errors


def _summary_row(spec: PilotSpec, summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "configuration_id": spec.configuration_id,
        "model_id": spec.model_id,
        "executor_id": spec.executor_id,
        "signed_decay": spec.signed_decay,
        "success": summary.get("success"),
        "safe_success": summary.get("safe_success"),
        "stop_reason": summary.get("stop_reason"),
        "steps_executed": summary.get("steps_executed"),
        "min_peg_to_hole_dist": summary.get("min_peg_to_hole_dist"),
        "final_peg_to_hole_dist": summary.get("final_peg_to_hole_dist"),
        "max_force_norm": summary.get("max_force_norm"),
        "policy_inference_time_ms_p95": summary.get(
            "policy_inference_time_ms_p95"
        ),
        "policy_step_compute_time_ms_p95": summary.get(
            "policy_step_compute_time_ms_p95"
        ),
        "policy_deadline_miss_fraction": summary.get(
            "policy_deadline_miss_fraction"
        ),
    }


def write_aggregate(output_dir: Path, rows: list[dict[str, Any]]) -> Path:
    path = output_dir / "aggregate.csv"
    fieldnames = list(_summary_row(
        PilotSpec("", "", Path("."), "", "", None), {}
    ).keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def collect_completed_rows(
    output_dir: Path,
    args: argparse.Namespace,
    specs: Sequence[PilotSpec],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in specs:
        summary_path = output_dir / spec.configuration_id / "summary.json"
        if not summary_path.is_file():
            continue
        summary = json.loads(summary_path.read_text())
        errors = validate_completed_summary(summary, args, spec)
        if errors:
            raise ValueError(
                f"existing {spec.configuration_id} failed contract: "
                + "; ".join(errors)
            )
        rows.append(_summary_row(spec, summary))
    return rows


def _plan_payload(
    args: argparse.Namespace,
    specs: Sequence[PilotSpec],
    selected_configuration_ids: Sequence[str],
) -> dict[str, Any]:
    return {
        "pilot_version": PILOT_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "execution_enabled": bool(args.execute_rollouts),
        "sweep_mode": args.sweep_mode,
        "requested_signed_decays": args.signed_decays,
        "selected_model_ids": list(args.selected_model_ids),
        "selected_configuration_ids": list(selected_configuration_ids),
        "fairness_contract": {
            "fixed_hole_offset": [
                args.hole_offset_x,
                args.hole_offset_y,
                args.hole_offset_z,
            ],
            "seed": args.seed,
            "policy_query_interval": 1,
            "policy_rate_hz": 30.0,
            "max_rollout_steps": args.max_rollout_steps,
            "contact_latent_mode": "zero",
            "action_mode": "joint_pos",
            "ema_alpha": 1.0,
            "max_delta_q": 0.02,
            "force_stop_threshold": 100.0,
            "safe_force_threshold": 40.0,
            "success_distance_threshold": 0.003,
            "success_dwell_time": 0.1,
            "save_videos": args.save_videos,
            "save_force_hud_videos": args.save_force_hud_videos,
            "force_hud_camera": args.force_hud_camera,
            "force_hud_resolution": [
                args.force_hud_width,
                args.force_hud_height,
            ],
            "force_hud_primary_wrench": args.force_hud_primary_wrench,
        },
        "specifications": [
            {
                **asdict(spec),
                "checkpoint": _resolved(spec.checkpoint),
                "output_dir": str(args.output_dir / spec.configuration_id),
                "command": build_rollout_command(args, spec),
            }
            for spec in specs
        ],
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-checkpoint", type=Path, default=DEFAULT_OFFICIAL_CHECKPOINT)
    parser.add_argument("--contact-checkpoint", type=Path, default=DEFAULT_CONTACT_CHECKPOINT)
    parser.add_argument("--model-xml", type=Path, default=DEFAULT_MODEL_XML)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-rollout-steps", type=int, default=600)
    parser.add_argument("--hole-offset-x", type=float, default=0.0)
    parser.add_argument("--hole-offset-y", type=float, default=0.0)
    parser.add_argument("--hole-offset-z", type=float, default=0.0)
    parser.add_argument(
        "--configuration",
        action="append",
        dest="configurations",
        help="Run only this configuration id; repeat to select multiple.",
    )
    parser.add_argument(
        "--signed-decay",
        action="append",
        type=float,
        dest="signed_decays",
        help=(
            "Build a dynamic signed-temporal sweep at this k; repeat in the "
            "desired execution order. Cannot be combined with --configuration."
        ),
    )
    parser.add_argument(
        "--model-id",
        action="append",
        choices=MODEL_IDS,
        dest="model_ids",
        help=(
            "Model to include in the sweep; repeat for a paired sweep. "
            "Defaults to both models."
        ),
    )
    parser.add_argument(
        "--execute-rollouts",
        action="store_true",
        help="Actually execute policy actions; without this flag only write the plan.",
    )
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--no-save-videos", action="store_false", dest="save_videos")
    parser.add_argument("--save-force-hud-videos", action="store_true")
    parser.add_argument("--force-hud-camera", default=DEFAULT_FORCE_HUD_CAMERA)
    parser.add_argument(
        "--force-hud-width",
        type=int,
        default=DEFAULT_FORCE_HUD_WIDTH,
    )
    parser.add_argument(
        "--force-hud-height",
        type=int,
        default=DEFAULT_FORCE_HUD_HEIGHT,
    )
    parser.add_argument(
        "--force-hud-primary-wrench",
        choices=("raw", "compensated"),
        default="compensated",
    )
    parser.set_defaults(save_videos=True, save_force_hud_videos=False)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.max_rollout_steps <= 0:
        print("error: --max-rollout-steps must be positive", file=sys.stderr)
        return 2
    if args.force_hud_width <= 0 or args.force_hud_height <= 0:
        print("error: force HUD dimensions must be positive", file=sys.stderr)
        return 2
    if args.save_force_hud_videos and not args.force_hud_camera:
        print("error: --force-hud-camera must be non-empty", file=sys.stderr)
        return 2
    if args.signed_decays and args.configurations:
        print(
            "error: --signed-decay cannot be combined with --configuration",
            file=sys.stderr,
        )
        return 2
    if args.signed_decays and any(
        not math.isfinite(value) for value in args.signed_decays
    ):
        print("error: signed decays must be finite", file=sys.stderr)
        return 2
    if args.signed_decays and len(set(args.signed_decays)) != len(
        args.signed_decays
    ):
        print("error: signed decays must not contain duplicates", file=sys.stderr)
        return 2
    if args.model_ids and len(set(args.model_ids)) != len(args.model_ids):
        print("error: model ids must not contain duplicates", file=sys.stderr)
        return 2
    args.selected_model_ids = tuple(args.model_ids or MODEL_IDS)
    required_paths = {"model_xml"}
    if "official_act" in args.selected_model_ids:
        required_paths.add("official_checkpoint")
    if "highrate_contact_v3" in args.selected_model_ids:
        required_paths.add("contact_checkpoint")
    for path_name in ("official_checkpoint", "contact_checkpoint", "model_xml"):
        path = getattr(args, path_name).expanduser().resolve()
        setattr(args, path_name, path)
        if path_name in required_paths and not path.is_file():
            print(f"error: {path_name} does not exist: {path}", file=sys.stderr)
            return 2
    args.output_dir = args.output_dir.expanduser().resolve()
    if args.signed_decays:
        args.sweep_mode = "dynamic_signed_decay"
        try:
            all_specs = build_signed_decay_specs(
                args.official_checkpoint,
                args.contact_checkpoint,
                args.signed_decays,
                args.selected_model_ids,
            )
        except ValueError as error:
            print(f"error: {error}", file=sys.stderr)
            return 2
    else:
        args.sweep_mode = "predefined_matrix"
        args.signed_decays = None
        all_specs = tuple(
            spec
            for spec in build_pilot_specs(
                args.official_checkpoint,
                args.contact_checkpoint,
            )
            if spec.model_id in args.selected_model_ids
        )
    known_ids = {spec.configuration_id for spec in all_specs}
    requested = set(args.configurations or known_ids)
    unknown = sorted(requested - known_ids)
    if unknown:
        print(f"error: unknown configuration ids: {', '.join(unknown)}", file=sys.stderr)
        return 2
    specs = tuple(spec for spec in all_specs if spec.configuration_id in requested)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plan_path = args.output_dir / "pilot_plan.json"
    plan_path.write_text(
        json.dumps(
            _plan_payload(
                args,
                all_specs,
                [spec.configuration_id for spec in specs],
            ),
            indent=2,
        )
        + "\n"
    )
    print(f"pilot_plan={plan_path}")
    for spec in specs:
        print(f"planned={spec.configuration_id}")
    if not args.execute_rollouts:
        print("status=PLAN_ONLY (pass --execute-rollouts to apply policy actions)")
        return 0

    failed = False
    for spec in specs:
        output_dir = args.output_dir / spec.configuration_id
        summary_path = output_dir / "summary.json"
        if summary_path.is_file() and args.skip_existing:
            summary = json.loads(summary_path.read_text())
            errors = validate_completed_summary(summary, args, spec)
            if errors:
                print(
                    f"error: existing {spec.configuration_id} failed contract: "
                    + "; ".join(errors),
                    file=sys.stderr,
                )
                return 2
            print(f"skipped_complete={spec.configuration_id}")
            continue
        if output_dir.exists() and any(output_dir.iterdir()):
            print(
                f"error: output directory is non-empty: {output_dir}; use a new "
                "root or --skip-existing for a valid completed run",
                file=sys.stderr,
            )
            return 2
        print(f"running={spec.configuration_id}", flush=True)
        result = subprocess.run(build_rollout_command(args, spec), check=False)
        if result.returncode != 0:
            failed = True
            print(
                f"error: {spec.configuration_id} exited with {result.returncode}",
                file=sys.stderr,
            )
            if not args.continue_on_error:
                return result.returncode
            continue
        if not summary_path.is_file():
            print(f"error: missing summary after {spec.configuration_id}", file=sys.stderr)
            return 2
        summary = json.loads(summary_path.read_text())
        errors = validate_completed_summary(summary, args, spec)
        if errors:
            print(
                f"error: {spec.configuration_id} failed contract: "
                + "; ".join(errors),
                file=sys.stderr,
            )
            return 2
        try:
            rows = collect_completed_rows(args.output_dir, args, all_specs)
        except ValueError as error:
            print(f"error: {error}", file=sys.stderr)
            return 2
        write_aggregate(args.output_dir, rows)
        print(f"completed={spec.configuration_id}")

    try:
        rows = collect_completed_rows(args.output_dir, args, all_specs)
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    aggregate_path = write_aggregate(args.output_dir, rows)
    print(f"aggregate={aggregate_path}")
    print(f"completed_configurations={len(rows)}/{len(all_specs)}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
