#!/usr/bin/env python3
"""Run a configurable x/z hole-position rollout suite.

This is a thin orchestration wrapper around scripts/run_mujoco_hole_grid.py.
It does not implement rollout logic itself; it only launches the existing grid
runner sequentially for the configured model/action-selection combinations.
After each grid finishes, it can also call scripts/plot_hole_target_map.py to
create target-style rollout maps from the generated grid_summary.csv.

Typical use:

    conda activate forceact
    python scripts/run_xz_rollout_suite.py

The defaults are 50 points, +/-6 mm offsets, and 900 rollout steps. Override
them without changing the script:

    python scripts/run_xz_rollout_suite.py \
        --num-points 25 --offset-mm 4 --max-rollout-steps 600

Preview commands without running:

    python scripts/run_xz_rollout_suite.py --dry-run
"""

from __future__ import annotations

import argparse
import math
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = Path("outputs/peg_hole_100")


@dataclass(frozen=True)
class ModelSpec:
    key: str
    output_token: str
    checkpoint: Path
    contact_latent_mode: str = "zero"


MODEL_SPECS: tuple[ModelSpec, ...] = (
    ModelSpec(
        key="contact_cvae",
        output_token="contact_cvae100k_zero",
        checkpoint=OUTPUT_ROOT / "forceaware_contact_cvae_betac5e4_lp01_trajectory100k/checkpoint.pt",
        contact_latent_mode="zero",
    ),
    ModelSpec(
        key="contact_cvae_prior",
        output_token="contact_cvae100k_prior",
        checkpoint=OUTPUT_ROOT / "forceaware_contact_cvae_betac5e4_lp01_trajectory100k/checkpoint.pt",
        contact_latent_mode="prior",
    ),
    ModelSpec(
        key="motion_cvae",
        output_token="motion_cvae100k",
        checkpoint=OUTPUT_ROOT / "forceaware_motion_cvae_betam5e4_trajectory100k/checkpoint.pt",
        contact_latent_mode="zero",
    ),
    ModelSpec(
        key="dualzero",
        output_token="dualzero100k",
        checkpoint=OUTPUT_ROOT / "forceaware_dualzero_trajectory100k/checkpoint.pt",
        contact_latent_mode="zero",
    ),
    ModelSpec(
        key="act_baseline",
        output_token="act_baseline100k",
        checkpoint=OUTPUT_ROOT / "act_baseline_motion_cvae_betam5e4_trajectory100k/checkpoint.pt",
        contact_latent_mode="zero",
    ),
)


ACTION_SELECT_MODES = ("mid", "temporal")
TEMPORAL_DECAY_NAME = "d03"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run configured policy/latent variants with mid/temporal action "
            "selection on an x/z hole-position suite."
        )
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=[spec.key for spec in MODEL_SPECS],
        default=[spec.key for spec in MODEL_SPECS],
        help=(
            "Policy/latent configurations to run. Default: Contact-CVAE zero "
            "and prior, Motion-CVAE, DualZero, and ACT baseline."
        ),
    )
    parser.add_argument(
        "--action-select-modes",
        nargs="+",
        choices=ACTION_SELECT_MODES,
        default=list(ACTION_SELECT_MODES),
        help="Action selection modes to run. Default: mid temporal.",
    )
    parser.add_argument(
        "--num-points",
        type=int,
        default=50,
        help="Number of sampled hole positions. Default: 50.",
    )
    parser.add_argument(
        "--offset-mm",
        type=float,
        default=6.0,
        help="Symmetric x/z offset bound in millimetres. Default: 6.",
    )
    parser.add_argument("--base-seed", type=int, default=20260702)
    parser.add_argument("--output-base", type=Path, default=OUTPUT_ROOT)
    parser.add_argument(
        "--normalization-stats",
        type=Path,
        default=OUTPUT_ROOT / "normalization_stats_action_all100.pt",
    )
    parser.add_argument("--model-xml", type=Path, default=Path("../arm_teleop/model/pangu_all_right.xml"))
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--mujoco-gl", default=None)
    parser.add_argument("--sampling-mode", choices=("latin_hypercube", "random", "grid"), default="latin_hypercube")
    parser.add_argument("--action-mode", default="action")
    parser.add_argument("--chunk-len", type=int, default=10)
    parser.add_argument("--force-window-len", type=int, default=20)
    parser.add_argument("--force-window-duration", type=float, default=0.25)
    parser.add_argument("--policy-rate-hz", type=float, default=30.0)
    parser.add_argument(
        "--max-rollout-steps",
        type=int,
        default=900,
        help="Maximum steps per rollout. Default: 900.",
    )
    parser.add_argument("--max-delta-q", type=float, default=0.02)
    parser.add_argument("--force-stop-threshold", type=float, default=1000.0)
    parser.add_argument("--success-distance-threshold", type=float, default=0.005)
    parser.add_argument("--success-lateral-threshold", type=float, default=0.006)
    parser.add_argument("--success-force-threshold", type=float, default=80.0)
    parser.add_argument("--success-hold-steps", type=int, default=15)
    parser.add_argument("--y-offset", type=float, default=0.0)
    parser.add_argument("--hole-site-name", default="hole_goal_site")
    parser.add_argument("--hole-body-name", default="wall_task")
    parser.add_argument("--hole-offset-frame", choices=("world", "body"), default="world")
    parser.add_argument("--hole-axis-world", nargs=3, type=float, default=(0.0, -1.0, 0.0))
    parser.add_argument(
        "--skip-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pass --skip-existing to the grid runner. Default: true.",
    )
    parser.add_argument(
        "--continue-on-error",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pass --continue-on-error to the grid runner. Default: true.",
    )
    parser.add_argument(
        "--plot-results",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Let the grid runner create its built-in plots. Default: false.",
    )
    parser.add_argument(
        "--target-maps",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Create target-style maps after each grid_summary.csv is available. Default: true.",
    )
    parser.add_argument("--target-map-ring-step-mm", type=float, default=2.0)
    parser.add_argument(
        "--target-map-max-radius-mm",
        type=float,
        default=None,
        help=(
            "Shared target-map plot radius in millimetres. By default, derive "
            "a clean radius that contains the full square sampling range."
        ),
    )
    parser.add_argument("--target-map-marker-size", type=float, default=44.0)
    parser.add_argument("--target-map-dpi", type=int, default=300)
    parser.add_argument("--target-map-formats", nargs="+", default=["png", "pdf"])
    parser.add_argument(
        "--labeled-target-map",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also create a labeled diagnostic PNG with point indices. Default: true.",
    )
    parser.add_argument(
        "--keep-going",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Continue to the next experiment if one grid command exits nonzero. Default: true.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    return parser


def selected_models(keys: Iterable[str]) -> list[ModelSpec]:
    wanted = set(keys)
    return [spec for spec in MODEL_SPECS if spec.key in wanted]


def _number_token(value: float) -> str:
    text = format(value, ".12g")
    return text.replace("-", "m").replace(".", "p").replace("+", "")


def offset_token(offset_mm: float) -> str:
    return f"{_number_token(offset_mm)}mm"


def delta_q_token(max_delta_q: float) -> str:
    return "dq" + format(max_delta_q, ".12g").replace(".", "").replace("-", "m")


def output_dir_for(
    model: ModelSpec,
    action_select_mode: str,
    output_base: Path,
    *,
    num_points: int = 50,
    offset_mm: float = 6.0,
    max_rollout_steps: int = 900,
    max_delta_q: float = 0.02,
) -> Path:
    suffix = f"{action_select_mode}_{delta_q_token(max_delta_q)}"
    if action_select_mode == "temporal":
        suffix = f"temporal_{TEMPORAL_DECAY_NAME}_{delta_q_token(max_delta_q)}"
    return output_base / (
        f"hole_lhs_{num_points}_xz_{offset_token(offset_mm)}_"
        f"{model.output_token}_{suffix}_maxsteps{max_rollout_steps}"
    )


def output_dir_from_args(
    args: argparse.Namespace,
    model: ModelSpec,
    action_select_mode: str,
) -> Path:
    return output_dir_for(
        model,
        action_select_mode,
        args.output_base,
        num_points=args.num_points,
        offset_mm=args.offset_mm,
        max_rollout_steps=args.max_rollout_steps,
        max_delta_q=args.max_delta_q,
    )


def target_map_limit_mm(args: argparse.Namespace) -> float:
    if args.target_map_max_radius_mm is not None:
        return float(args.target_map_max_radius_mm)
    required_radius = math.sqrt(2.0) * args.offset_mm
    return math.ceil(required_radius / args.target_map_ring_step_mm) * args.target_map_ring_step_mm


def validate_inputs(args: argparse.Namespace, models: Sequence[ModelSpec]) -> None:
    if args.num_points <= 0:
        raise ValueError("--num-points must be positive")
    if args.offset_mm <= 0 or not math.isfinite(args.offset_mm):
        raise ValueError("--offset-mm must be positive and finite")
    if args.max_rollout_steps <= 0:
        raise ValueError("--max-rollout-steps must be positive")
    if args.max_delta_q <= 0 or not math.isfinite(args.max_delta_q):
        raise ValueError("--max-delta-q must be positive and finite")
    if args.target_map_ring_step_mm <= 0 or not math.isfinite(args.target_map_ring_step_mm):
        raise ValueError("--target-map-ring-step-mm must be positive and finite")
    if args.target_map_max_radius_mm is not None:
        if args.target_map_max_radius_mm <= 0 or not math.isfinite(args.target_map_max_radius_mm):
            raise ValueError("--target-map-max-radius-mm must be positive and finite")
        required_radius = math.sqrt(2.0) * args.offset_mm
        if args.target_map_max_radius_mm + 1.0e-9 < required_radius:
            raise ValueError(
                "--target-map-max-radius-mm would not contain the full square "
                f"sampling range; need at least {required_radius:g} mm"
            )
    if args.target_map_marker_size <= 0 or not math.isfinite(args.target_map_marker_size):
        raise ValueError("--target-map-marker-size must be positive and finite")
    if args.target_map_dpi <= 0:
        raise ValueError("--target-map-dpi must be positive")
    supported_formats = {"png", "pdf", "svg"}
    normalized_formats = {item.lower().lstrip(".") for item in args.target_map_formats}
    if not normalized_formats:
        raise ValueError("--target-map-formats must not be empty")
    unsupported_formats = sorted(normalized_formats - supported_formats)
    if unsupported_formats:
        raise ValueError(
            "unsupported target-map format(s): " + ", ".join(unsupported_formats)
        )
    required = [args.normalization_stats, args.model_xml, *[spec.checkpoint for spec in models]]
    missing = [path for path in required if not path.exists()]
    if missing:
        joined = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(f"Required file(s) missing:\n{joined}")


def build_grid_command(args: argparse.Namespace, model: ModelSpec, action_select_mode: str) -> list[str]:
    offset_m = args.offset_mm / 1000.0
    output_root = output_dir_from_args(args, model, action_select_mode)
    cmd = [
        args.python_executable,
        "scripts/run_mujoco_hole_grid.py",
        "--sampling-mode",
        args.sampling_mode,
        "--num-points",
        str(args.num_points),
        "--x-min",
        f"{-offset_m:.6f}",
        "--x-max",
        f"{offset_m:.6f}",
        "--z-min",
        f"{-offset_m:.6f}",
        "--z-max",
        f"{offset_m:.6f}",
        "--base-seed",
        str(args.base_seed),
        "--checkpoint",
        str(model.checkpoint),
        "--normalization-stats",
        str(args.normalization_stats),
        "--model-xml",
        str(args.model_xml),
        "--contact-latent-mode",
        model.contact_latent_mode,
        "--action-mode",
        args.action_mode,
        "--action-select-mode",
        action_select_mode,
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
        *[str(value) for value in args.hole_axis_world],
        "--hole-site-name",
        args.hole_site_name,
        "--hole-body-name",
        args.hole_body_name,
        "--hole-offset-frame",
        args.hole_offset_frame,
        "--y-offset",
        str(args.y_offset),
        "--success-distance-threshold",
        str(args.success_distance_threshold),
        "--success-lateral-threshold",
        str(args.success_lateral_threshold),
        "--success-force-threshold",
        str(args.success_force_threshold),
        "--success-hold-steps",
        str(args.success_hold_steps),
        "--output-root",
        str(output_root),
    ]
    if args.mujoco_gl:
        cmd.extend(["--mujoco-gl", args.mujoco_gl])
    if args.skip_existing:
        cmd.append("--skip-existing")
    if args.continue_on_error:
        cmd.append("--continue-on-error")
    cmd.append("--plot-results" if args.plot_results else "--no-plot-results")
    return cmd


def command_label(model: ModelSpec, action_select_mode: str) -> str:
    return f"{model.key}:{action_select_mode}"


def plot_title_for(model: ModelSpec, action_select_mode: str, offset_mm: float) -> str:
    name = {
        "contact_cvae": "Contact-CVAE zero",
        "contact_cvae_prior": "Contact-CVAE prior",
        "motion_cvae": "Motion-CVAE",
        "dualzero": "DualZero",
        "act_baseline": "ACT baseline",
    }[model.key]
    return f"{name} + {action_select_mode}, +/-{offset_mm:g} mm LHS"


def plot_stem_for(model: ModelSpec, action_select_mode: str, offset_mm: float) -> str:
    return (
        f"{model.output_token}_{action_select_mode}_{offset_token(offset_mm)}"
        "_target_safe_success"
    )


def build_target_map_commands(
    args: argparse.Namespace,
    model: ModelSpec,
    action_select_mode: str,
    output_root: Path,
) -> list[tuple[str, list[str]]]:
    grid_summary_csv = output_root / "grid_summary.csv"
    plots_dir = output_root / "plots"
    stem = plot_stem_for(model, action_select_mode, args.offset_mm)
    title = plot_title_for(model, action_select_mode, args.offset_mm)
    plot_limit_mm = target_map_limit_mm(args)
    base_cmd = [
        args.python_executable,
        "scripts/plot_hole_target_map.py",
        "--grid-summary-csv",
        str(grid_summary_csv),
        "--output-dir",
        str(plots_dir),
        "--output-stem",
        stem,
        "--title",
        title,
        "--ring-step-mm",
        str(args.target_map_ring_step_mm),
        "--max-radius-mm",
        str(plot_limit_mm),
        "--marker-size",
        str(args.target_map_marker_size),
        "--formats",
        *args.target_map_formats,
        "--dpi",
        str(args.target_map_dpi),
    ]
    commands = [(f"{command_label(model, action_select_mode)}:target-map", base_cmd)]
    if args.labeled_target_map:
        labeled_cmd = [
            args.python_executable,
            "scripts/plot_hole_target_map.py",
            "--grid-summary-csv",
            str(grid_summary_csv),
            "--output-dir",
            str(plots_dir),
            "--output-stem",
            f"{stem}_labeled",
            "--title",
            title,
            "--ring-step-mm",
            str(args.target_map_ring_step_mm),
            "--max-radius-mm",
            str(plot_limit_mm),
            "--marker-size",
            str(args.target_map_marker_size),
            "--show-point-index",
            "--show-sampling-boundary",
            "--formats",
            "png",
            "--dpi",
            str(args.target_map_dpi),
        ]
        commands.append((f"{command_label(model, action_select_mode)}:target-map-labeled", labeled_cmd))
    return commands


def print_command(label: str, cmd: Sequence[str], log_path: Path) -> None:
    quoted = " ".join(shlex.quote(str(part)) for part in cmd)
    print(f"\n[{label}]")
    print(f"console_log={log_path}")
    print(f"PYTHONPATH=src {quoted}")


def run_command(label: str, cmd: Sequence[str], log_path: Path, env: dict[str, str]) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n=== Running {label} ===")
    print(f"console_log={log_path}")
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            cmd,
            cwd=REPO_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)
        return_code = process.wait()
    print(f"=== Finished {label}: return_code={return_code} ===")
    return return_code


def run_simple_command(label: str, cmd: Sequence[str], env: dict[str, str]) -> int:
    print(f"\n=== Running {label} ===")
    process = subprocess.run(cmd, cwd=REPO_ROOT, env=env, text=True)
    print(f"=== Finished {label}: return_code={process.returncode} ===")
    return process.returncode


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    models = selected_models(args.models)
    validate_inputs(args, models)

    env = os.environ.copy()
    env["PYTHONPATH"] = "src"
    if args.mujoco_gl:
        env["MUJOCO_GL"] = args.mujoco_gl

    failures: list[tuple[str, int]] = []
    for model in models:
        for action_select_mode in args.action_select_modes:
            label = command_label(model, action_select_mode)
            cmd = build_grid_command(args, model, action_select_mode)
            output_root = output_dir_from_args(args, model, action_select_mode)
            log_path = output_root.with_name(f"{output_root.name}_console.log")
            if args.dry_run:
                print_command(label, cmd, log_path)
                if args.target_maps:
                    for plot_label, plot_cmd in build_target_map_commands(args, model, action_select_mode, output_root):
                        print(f"\n[{plot_label}]")
                        quoted = " ".join(shlex.quote(str(part)) for part in plot_cmd)
                        print(f"PYTHONPATH=src {quoted}")
                continue
            return_code = run_command(label, cmd, log_path, env)
            if return_code != 0:
                failures.append((label, return_code))
                if not args.keep_going:
                    break
            if args.target_maps:
                grid_summary_csv = output_root / "grid_summary.csv"
                if not grid_summary_csv.exists():
                    print(f"Skipping target maps for {label}: missing {grid_summary_csv}")
                else:
                    for plot_label, plot_cmd in build_target_map_commands(args, model, action_select_mode, output_root):
                        plot_return_code = run_simple_command(plot_label, plot_cmd, env)
                        if plot_return_code != 0:
                            failures.append((plot_label, plot_return_code))
                            if not args.keep_going:
                                break
                    if failures and not args.keep_going:
                        break
        if failures and not args.keep_going:
            break

    if failures:
        print("\nFailures:")
        for label, return_code in failures:
            print(f"  {label}: return_code={return_code}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
