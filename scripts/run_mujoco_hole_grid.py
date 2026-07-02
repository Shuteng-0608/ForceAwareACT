#!/usr/bin/env python3
"""Run a deterministic grid of hole-position perturbation rollouts."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from summarize_rollouts import collect_rollouts, write_summary_csv  # noqa: E402


def parse_offset_list(value: str) -> list[float]:
    offsets = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        offsets.append(float(item))
    if not offsets:
        raise ValueError("offset list must contain at least one value")
    return offsets


def signed_mm_token(value_m: float) -> str:
    value_mm = int(round(value_m * 1000.0))
    prefix = "m" if value_mm < 0 else "p"
    return f"{prefix}{abs(value_mm):03d}mm"


def run_name(x_offset: float, z_offset: float, repeat_index: int) -> str:
    return (
        f"x_{signed_mm_token(x_offset)}_"
        f"z_{signed_mm_token(z_offset)}_"
        f"repeat_{repeat_index:03d}"
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _read_summary(path: Path) -> Optional[dict[str, Any]]:
    if not path.is_file():
        return None
    try:
        with path.open() as summary_file:
            return json.load(summary_file)
    except Exception:
        return None


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as manifest_file:
        json.dump(_json_safe(manifest), manifest_file, indent=2, sort_keys=True)
        manifest_file.write("\n")
    print(f"grid_manifest_json={path}")


def _build_rollout_command(args: argparse.Namespace, output_dir: Path, x_offset: float, y_offset: float, z_offset: float, seed: int) -> list[str]:
    command = [
        args.python_executable,
        str(SCRIPT_DIR / "run_mujoco_policy_rollout.py"),
        "--checkpoint",
        str(args.checkpoint),
        "--normalization-stats",
        str(args.normalization_stats),
        "--model-xml",
        str(args.model_xml),
        "--contact-latent-mode",
        args.contact_latent_mode,
        "--action-mode",
        args.action_mode,
        "--action-select-mode",
        args.action_select_mode,
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
        *(str(value) for value in args.hole_axis_world),
        "--hole-site-name",
        args.hole_site_name,
        "--hole-offset-frame",
        args.hole_offset_frame,
        "--hole-offset-x",
        str(x_offset),
        "--hole-offset-y",
        str(y_offset),
        "--hole-offset-z",
        str(z_offset),
        "--success-distance-threshold",
        str(args.success_distance_threshold),
        "--success-lateral-threshold",
        str(args.success_lateral_threshold),
        "--success-force-threshold",
        str(args.success_force_threshold),
        "--success-hold-steps",
        str(args.success_hold_steps),
        "--output-dir",
        str(output_dir),
        "--seed",
        str(seed),
        "--execute-actions",
    ]
    if args.hole_body_name:
        command.extend(["--hole-body-name", args.hole_body_name])
    if args.save_videos:
        command.append("--save-videos")
    return command


def _summarize(output_root: Path) -> Path:
    summary_csv = output_root / "grid_summary.csv"
    rows = collect_rollouts(output_root, "x_*_z_*_repeat_*")
    write_summary_csv(summary_csv, rows)
    print(f"grid_summary_csv={summary_csv}")
    return summary_csv


def _plot_results(summary_csv: Path, output_root: Path, formats: str) -> None:
    try:
        from plot_hole_grid_results import main as plot_main

        plot_main(
            [
                "--summary-csv",
                str(summary_csv),
                "--output-dir",
                str(output_root / "plots"),
                "--formats",
                formats,
            ]
        )
    except Exception as error:
        print(f"warning: grid plotting failed: {error}")


def run_grid(args: argparse.Namespace) -> dict[str, Any]:
    x_offsets = parse_offset_list(args.x_offsets)
    z_offsets = parse_offset_list(args.z_offsets)
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_root / "grid_manifest.json"
    manifest: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "x_offsets": x_offsets,
        "y_offset": args.y_offset,
        "z_offsets": z_offsets,
        "repeats": args.repeats,
        "total_planned_runs": len(x_offsets) * len(z_offsets) * args.repeats,
        "policy_config": {
            "checkpoint": args.checkpoint,
            "normalization_stats": args.normalization_stats,
            "model_xml": args.model_xml,
            "contact_latent_mode": args.contact_latent_mode,
            "action_mode": args.action_mode,
            "action_select_mode": args.action_select_mode,
            "chunk_len": args.chunk_len,
            "force_window_len": args.force_window_len,
            "force_window_duration": args.force_window_duration,
            "policy_rate_hz": args.policy_rate_hz,
            "max_rollout_steps": args.max_rollout_steps,
            "max_delta_q": args.max_delta_q,
            "force_stop_threshold": args.force_stop_threshold,
        },
        "success_thresholds": {
            "success_distance_threshold": args.success_distance_threshold,
            "success_lateral_threshold": args.success_lateral_threshold,
            "success_force_threshold": args.success_force_threshold,
            "success_hold_steps": args.success_hold_steps,
        },
        "runs": [],
    }

    env = os.environ.copy()
    if args.mujoco_gl:
        env["MUJOCO_GL"] = args.mujoco_gl

    for repeat in range(1, args.repeats + 1):
        for x_offset in x_offsets:
            for z_offset in z_offsets:
                seed = args.base_seed + len(manifest["runs"])
                output_dir = args.output_root / run_name(x_offset, z_offset, repeat)
                command = _build_rollout_command(
                    args,
                    output_dir,
                    x_offset,
                    args.y_offset,
                    z_offset,
                    seed,
                )
                run_entry: dict[str, Any] = {
                    "x_offset": x_offset,
                    "y_offset": args.y_offset,
                    "z_offset": z_offset,
                    "repeat_index": repeat,
                    "seed": seed,
                    "output_dir": output_dir,
                    "command": command,
                    "return_code": None,
                    "status": "planned",
                    "start_time": None,
                    "end_time": None,
                }
                summary_path = output_dir / "summary.json"
                if args.skip_existing and _read_summary(summary_path) is not None:
                    run_entry["status"] = "skipped_existing"
                    run_entry["return_code"] = 0
                    manifest["runs"].append(run_entry)
                    continue
                if args.dry_run:
                    print(" ".join(command))
                    run_entry["status"] = "dry_run"
                    manifest["runs"].append(run_entry)
                    continue

                run_entry["status"] = "running"
                run_entry["start_time"] = datetime.now(timezone.utc).isoformat()
                completed = subprocess.run(command, env=env, check=False)
                run_entry["end_time"] = datetime.now(timezone.utc).isoformat()
                run_entry["return_code"] = completed.returncode
                if completed.returncode != 0:
                    run_entry["status"] = "process_error"
                    manifest["runs"].append(run_entry)
                    _write_manifest(manifest_path, manifest)
                    if not args.continue_on_error:
                        break
                    continue
                summary = _read_summary(summary_path) or {}
                run_entry["status"] = "success" if summary.get("success") else "task_failed"
                run_entry["stop_reason"] = summary.get("stop_reason")
                manifest["runs"].append(run_entry)
                _write_manifest(manifest_path, manifest)
            else:
                continue
            break
        else:
            continue
        break

    _write_manifest(manifest_path, manifest)
    summary_csv = _summarize(args.output_root)
    if args.plot_results and not args.dry_run:
        _plot_results(summary_csv, args.output_root, args.plot_formats)
    return manifest


def _print_final_counts(manifest: dict[str, Any]) -> None:
    runs = manifest["runs"]
    completed = [run for run in runs if run.get("status") in {"success", "task_failed", "skipped_existing"}]
    successful = [run for run in runs if run.get("status") == "success"]
    failed_task = [run for run in runs if run.get("status") == "task_failed"]
    process_errors = [run for run in runs if run.get("status") == "process_error"]
    denominator = len(completed)
    success_rate = len(successful) / denominator if denominator else 0.0
    print(f"total_planned_runs={manifest.get('total_planned_runs', len(runs))}")
    print(f"completed_runs={len(completed)}")
    print(f"successful_runs={len(successful)}")
    print(f"failed_task_runs={len(failed_task)}")
    print(f"process_error_runs={len(process_errors)}")
    print(f"success_rate={success_rate:.6g}")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a hole-position perturbation rollout grid.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--normalization-stats", type=Path, required=True)
    parser.add_argument("--model-xml", type=Path, required=True)
    parser.add_argument("--contact-latent-mode", default="zero")
    parser.add_argument("--action-mode", default="action")
    parser.add_argument("--action-select-mode", default="mid")
    parser.add_argument("--chunk-len", type=int, default=10)
    parser.add_argument("--force-window-len", type=int, default=20)
    parser.add_argument("--force-window-duration", type=float, default=0.25)
    parser.add_argument("--policy-rate-hz", type=float, default=30.0)
    parser.add_argument("--max-rollout-steps", type=int, default=900)
    parser.add_argument("--max-delta-q", type=float, default=0.02)
    parser.add_argument("--force-stop-threshold", type=float, default=1000.0)
    parser.add_argument("--hole-axis-world", type=float, nargs=3, default=(0.0, -1.0, 0.0))
    parser.add_argument("--hole-site-name", default="hole_goal_site")
    parser.add_argument("--hole-body-name")
    parser.add_argument("--hole-offset-frame", choices=("world", "body"), default="world")
    parser.add_argument("--x-offsets", default="-0.002,0.0,0.002")
    parser.add_argument("--y-offset", type=float, default=0.0)
    parser.add_argument("--z-offsets", default="-0.002,0.0,0.002")
    parser.add_argument("--success-distance-threshold", type=float, default=0.005)
    parser.add_argument("--success-lateral-threshold", type=float, default=0.006)
    parser.add_argument("--success-force-threshold", type=float, default=80.0)
    parser.add_argument("--success-hold-steps", type=int, default=15)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--save-videos", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--mujoco-gl")
    parser.add_argument("--plot-results", dest="plot_results", action="store_true", default=True)
    parser.add_argument("--no-plot-results", dest="plot_results", action="store_false")
    parser.add_argument("--plot-formats", default="png")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.repeats <= 0:
        raise ValueError("--repeats must be positive")
    manifest = run_grid(args)
    _print_final_counts(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
