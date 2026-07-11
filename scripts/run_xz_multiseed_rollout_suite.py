#!/usr/bin/env python3
"""Run and summarize the x/z rollout suite across independent seed dimensions.

Legacy ``--seeds`` runs remain isolated under ``<output-base>/seed_<seed>/``.
Separated seed runs use ``pointset_<seed>/rollout_<seed>/`` so point generation
and rollout randomness can be varied independently without output collisions.

Example:

    conda activate forceact
    python scripts/run_xz_multiseed_rollout_suite.py \
        --seeds 20260702 20260703 20260704 20260705 20260706 \
        --offset-mm 4 \
        --output-base outputs/peg_hole_100/new_goal_multiseed

Preview without running MuJoCo:

    python scripts/run_xz_multiseed_rollout_suite.py \
        --seeds 20260702 20260703 --dry-run
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_xz_rollout_suite import (
    ACTION_SELECT_MODES,
    MODEL_SPECS,
    output_dir_for,
    selected_models,
)


DEFAULT_OUTPUT_BASE = Path("outputs/peg_hole_100/xz_multiseed_rollouts")
POINT_COLUMNS = ["point_index", "hole_offset_x", "hole_offset_y", "hole_offset_z"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the configurable x/z rollout suite for multiple base seeds, "
            "then aggregate task- and safe-success rates."
        )
    )
    seed_group = parser.add_mutually_exclusive_group(required=True)
    seed_group.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        help=(
            "Legacy coupled seeds. Each value controls both point generation "
            "and the first rollout seed."
        ),
    )
    seed_group.add_argument(
        "--point-set-seeds",
        nargs="+",
        type=int,
        help="Seeds used only to generate independent random/LHS point sets.",
    )
    parser.add_argument(
        "--rollout-seed-bases",
        nargs="+",
        type=int,
        default=None,
        help=(
            "Independent first rollout seeds. With --point-set-seeds, run the "
            "Cartesian product. If omitted, each point-set seed is also used "
            "as its rollout seed base."
        ),
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=[spec.key for spec in MODEL_SPECS],
        default=[spec.key for spec in MODEL_SPECS],
    )
    parser.add_argument(
        "--action-select-modes",
        nargs="+",
        choices=ACTION_SELECT_MODES,
        default=["mid"],
        help=(
            "Action-selection modes to run. Defaults to mid only for multi-seed "
            "safe-success estimates; pass temporal explicitly when needed."
        ),
    )
    parser.add_argument("--num-points", type=int, default=50)
    parser.add_argument("--offset-mm", type=float, default=4.0)
    parser.add_argument("--max-rollout-steps", type=int, default=900)
    parser.add_argument("--max-delta-q", type=float, default=0.02)
    parser.add_argument("--output-base", type=Path, default=DEFAULT_OUTPUT_BASE)
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument(
        "--normalization-stats",
        type=Path,
        default=Path("outputs/peg_hole_100/normalization_stats_action_all100.pt"),
    )
    parser.add_argument(
        "--model-xml",
        type=Path,
        default=Path("../arm_teleop/model/pangu_all_right.xml"),
    )
    parser.add_argument("--mujoco-gl", default=None)
    parser.add_argument(
        "--target-maps",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--keep-going",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Continue to later seeds if one seed-level suite exits nonzero.",
    )
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Skip rollout commands and aggregate existing seed directories.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    point_seeds = selected_point_set_seeds(args)
    if len(set(point_seeds)) != len(point_seeds):
        raise ValueError("point-set seeds must not contain duplicates")
    if args.rollout_seed_bases is not None and len(set(args.rollout_seed_bases)) != len(
        args.rollout_seed_bases
    ):
        raise ValueError("--rollout-seed-bases must not contain duplicates")
    if args.num_points <= 0:
        raise ValueError("--num-points must be positive")
    if args.offset_mm <= 0 or not math.isfinite(args.offset_mm):
        raise ValueError("--offset-mm must be positive and finite")
    if args.max_rollout_steps <= 0:
        raise ValueError("--max-rollout-steps must be positive")
    if args.max_delta_q <= 0 or not math.isfinite(args.max_delta_q):
        raise ValueError("--max-delta-q must be positive and finite")
    if args.dry_run and args.aggregate_only:
        raise ValueError("--dry-run and --aggregate-only cannot be used together")


def seed_output_base(output_base: Path, seed: int) -> Path:
    return output_base / f"seed_{seed}"


def selected_point_set_seeds(args: argparse.Namespace) -> list[int]:
    values = args.point_set_seeds if args.point_set_seeds is not None else args.seeds
    return list(values)


def separated_seed_mode(args: argparse.Namespace) -> bool:
    return args.point_set_seeds is not None or args.rollout_seed_bases is not None


def seed_configurations(args: argparse.Namespace) -> list[tuple[int, int]]:
    point_seeds = selected_point_set_seeds(args)
    if args.rollout_seed_bases is None:
        return [(seed, seed) for seed in point_seeds]
    return list(itertools.product(point_seeds, args.rollout_seed_bases))


def configuration_output_base(
    args: argparse.Namespace,
    point_set_seed: int,
    rollout_seed_base: int,
) -> Path:
    if not separated_seed_mode(args):
        return seed_output_base(args.output_base, point_set_seed)
    return (
        args.output_base
        / f"pointset_{point_set_seed}"
        / f"rollout_{rollout_seed_base}"
    )


def build_suite_plan(args: argparse.Namespace) -> dict[str, object]:
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "output_base": str(args.output_base),
        "point_set_seeds": selected_point_set_seeds(args),
        "rollout_seed_bases": sorted({seed for _, seed in seed_configurations(args)}),
        "seed_configurations": [
            {
                "point_set_seed": point_set_seed,
                "rollout_seed_base": rollout_seed_base,
                "output_base": str(
                    configuration_output_base(args, point_set_seed, rollout_seed_base)
                ),
            }
            for point_set_seed, rollout_seed_base in seed_configurations(args)
        ],
        "models": [model.key for model in selected_models(args.models)],
        "action_select_modes": list(args.action_select_modes),
        "num_points": args.num_points,
        "offset_mm": args.offset_mm,
        "max_rollout_steps": args.max_rollout_steps,
        "max_delta_q": args.max_delta_q,
    }


def write_suite_plan(args: argparse.Namespace) -> Path:
    args.output_base.mkdir(parents=True, exist_ok=True)
    path = args.output_base / "suite_plan.json"
    plan = build_suite_plan(args)
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        comparable_existing = {k: v for k, v in existing.items() if k != "created_at"}
        comparable_new = {k: v for k, v in plan.items() if k != "created_at"}
        if comparable_existing != comparable_new:
            raise ValueError(
                f"existing suite plan does not match requested protocol: {path}"
            )
        return path
    path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def build_seed_command(
    args: argparse.Namespace,
    seed: int,
    rollout_seed_base: int | None = None,
) -> list[str]:
    rollout_seed = seed if rollout_seed_base is None else rollout_seed_base
    command = [
        args.python_executable,
        "scripts/run_xz_rollout_suite.py",
        "--models",
        *args.models,
        "--action-select-modes",
        *args.action_select_modes,
        "--num-points",
        str(args.num_points),
        "--offset-mm",
        str(args.offset_mm),
        "--base-seed",
        str(seed),
        "--point-set-seed",
        str(seed),
        "--rollout-seed-base",
        str(rollout_seed),
        "--max-rollout-steps",
        str(args.max_rollout_steps),
        "--max-delta-q",
        str(args.max_delta_q),
        "--output-base",
        str(configuration_output_base(args, seed, rollout_seed)),
        "--normalization-stats",
        str(args.normalization_stats),
        "--model-xml",
        str(args.model_xml),
        "--skip-existing",
        "--continue-on-error",
        "--keep-going" if args.keep_going else "--no-keep-going",
        "--target-maps" if args.target_maps else "--no-target-maps",
    ]
    if args.mujoco_gl:
        command.extend(["--mujoco-gl", args.mujoco_gl])
    return command


def _parse_bool_series(series: pd.Series, column: str) -> pd.Series:
    true_values = {"true", "1", "yes", "success"}
    false_values = {"false", "0", "no", "failure"}
    parsed: list[bool] = []
    for value in series:
        if isinstance(value, (bool, np.bool_)):
            parsed.append(bool(value))
            continue
        text = str(value).strip().lower()
        if text in true_values:
            parsed.append(True)
        elif text in false_values:
            parsed.append(False)
        else:
            raise ValueError(f"{column} contains unsupported value: {value!r}")
    return pd.Series(parsed, index=series.index, dtype=bool)


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0:
        return float("nan"), float("nan")
    rate = successes / total
    denominator = 1.0 + z * z / total
    center = (rate + z * z / (2.0 * total)) / denominator
    margin = (
        z
        * math.sqrt(rate * (1.0 - rate) / total + z * z / (4.0 * total * total))
        / denominator
    )
    return center - margin, center + margin


def _experiment_root(
    args: argparse.Namespace,
    seed: int,
    model,
    action_select_mode: str,
    rollout_seed_base: int | None = None,
) -> Path:
    rollout_seed = seed if rollout_seed_base is None else rollout_seed_base
    return output_dir_for(
        model,
        action_select_mode,
        configuration_output_base(args, seed, rollout_seed),
        num_points=args.num_points,
        offset_mm=args.offset_mm,
        max_rollout_steps=args.max_rollout_steps,
        max_delta_q=args.max_delta_q,
    )


def experiment_result_complete(root: Path, expected_points: int) -> bool:
    csv_path = root / "grid_summary.csv"
    if not csv_path.is_file():
        return False
    try:
        frame = pd.read_csv(csv_path, usecols=["point_index"])
    except (OSError, ValueError, pd.errors.EmptyDataError):
        return False
    return len(frame) == expected_points


def seed_suite_complete(
    args: argparse.Namespace,
    seed: int,
    rollout_seed_base: int | None = None,
) -> bool:
    models = selected_models(args.models)
    for model in models:
        for action_select_mode in args.action_select_modes:
            root = _experiment_root(
                args,
                seed,
                model,
                action_select_mode,
                rollout_seed_base,
            )
            if not experiment_result_complete(root, args.num_points):
                return False
    return True


def collect_seed_summaries(args: argparse.Namespace) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    models = selected_models(args.models)
    reference_points_by_seed: dict[int, pd.DataFrame] = {}
    for seed, rollout_seed_base in seed_configurations(args):
        for model in models:
            for action_select_mode in args.action_select_modes:
                root = _experiment_root(
                    args,
                    seed,
                    model,
                    action_select_mode,
                    rollout_seed_base,
                )
                csv_path = root / "grid_summary.csv"
                row: dict[str, object] = {
                    "base_seed": seed,
                    "point_set_seed": seed,
                    "rollout_seed_base": rollout_seed_base,
                    "model_key": model.key,
                    "contact_latent_mode": model.contact_latent_mode,
                    "action_select_mode": action_select_mode,
                    "experiment_root": str(root),
                    "status": "missing",
                    "completed_points": 0,
                    "task_successes": 0,
                    "safe_successes": 0,
                    "task_success_rate": float("nan"),
                    "safe_success_rate": float("nan"),
                    "process_errors": float("nan"),
                    "point_set_matches_within_seed": False,
                }
                if not csv_path.is_file():
                    rows.append(row)
                    continue
                frame = pd.read_csv(csv_path)
                required = set(POINT_COLUMNS + ["success", "safe_success"])
                missing = sorted(required - set(frame.columns))
                if missing:
                    raise ValueError(f"{csv_path} missing required column(s): {', '.join(missing)}")
                success = _parse_bool_series(frame["success"], "success")
                safe_success = _parse_bool_series(frame["safe_success"], "safe_success")
                if bool((safe_success & ~success).any()):
                    raise ValueError(f"{csv_path} contains safe success without task success")
                points = frame[POINT_COLUMNS].reset_index(drop=True)
                reference_points = reference_points_by_seed.get(seed)
                if reference_points is None:
                    reference_points_by_seed[seed] = points
                    points_match = True
                else:
                    points_match = points.equals(reference_points)
                summary_path = root / "random_position_summary.json"
                process_errors = float("nan")
                if summary_path.is_file():
                    import json

                    process_errors = int(
                        json.loads(summary_path.read_text(encoding="utf-8")).get(
                            "process_error_runs", 0
                        )
                    )
                completed = len(frame)
                task_count = int(success.sum())
                safe_count = int(safe_success.sum())
                row.update(
                    {
                        "status": "complete" if completed == args.num_points else "incomplete",
                        "completed_points": completed,
                        "task_successes": task_count,
                        "safe_successes": safe_count,
                        "task_success_rate": task_count / completed if completed else float("nan"),
                        "safe_success_rate": safe_count / completed if completed else float("nan"),
                        "process_errors": process_errors,
                        "point_set_matches_within_seed": points_match,
                    }
                )
                rows.append(row)
    return pd.DataFrame(rows)


def aggregate_seed_summaries(per_seed: pd.DataFrame) -> pd.DataFrame:
    available = per_seed[per_seed["completed_points"] > 0].copy()
    if available.empty:
        raise ValueError("no completed grid_summary.csv files were found to aggregate")
    rows: list[dict[str, object]] = []
    group_columns = ["model_key", "contact_latent_mode", "action_select_mode"]
    for keys, group in available.groupby(group_columns, sort=False):
        total = int(group["completed_points"].sum())
        task_successes = int(group["task_successes"].sum())
        safe_successes = int(group["safe_successes"].sum())
        task_lower, task_upper = wilson_interval(task_successes, total)
        safe_lower, safe_upper = wilson_interval(safe_successes, total)
        safe_rates = group["safe_success_rate"].dropna()
        task_rates = group["task_success_rate"].dropna()
        rows.append(
            {
                "model_key": keys[0],
                "contact_latent_mode": keys[1],
                "action_select_mode": keys[2],
                "available_seeds": int(group["base_seed"].nunique()),
                "available_point_set_seeds": int(group["point_set_seed"].nunique()),
                "available_rollout_seed_bases": int(group["rollout_seed_base"].nunique()),
                "available_seed_configurations": int(
                    group[["point_set_seed", "rollout_seed_base"]]
                    .drop_duplicates()
                    .shape[0]
                ),
                "complete_seeds": int((group["status"] == "complete").sum()),
                "total_completed_points": total,
                "task_successes": task_successes,
                "task_success_rate_pooled": task_successes / total,
                "task_success_ci95_lower": task_lower,
                "task_success_ci95_upper": task_upper,
                "task_success_rate_seed_mean": float(task_rates.mean()),
                "task_success_rate_seed_std": (
                    float(task_rates.std(ddof=1)) if len(task_rates) > 1 else float("nan")
                ),
                "safe_successes": safe_successes,
                "safe_success_rate_pooled": safe_successes / total,
                "safe_success_ci95_lower": safe_lower,
                "safe_success_ci95_upper": safe_upper,
                "safe_success_rate_seed_mean": float(safe_rates.mean()),
                "safe_success_rate_seed_std": (
                    float(safe_rates.std(ddof=1)) if len(safe_rates) > 1 else float("nan")
                ),
                "all_point_sets_match_within_seed": bool(
                    group["point_set_matches_within_seed"].all()
                ),
                "total_process_errors": (
                    int(group["process_errors"].sum())
                    if group["process_errors"].notna().all()
                    else float("nan")
                ),
            }
        )
    return pd.DataFrame(rows)


def write_summaries(
    output_base: Path,
    per_seed: pd.DataFrame,
    aggregate: pd.DataFrame,
) -> tuple[Path, Path]:
    summary_dir = output_base / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    per_seed_path = summary_dir / "per_seed_summary.csv"
    aggregate_path = summary_dir / "aggregate_summary.csv"
    for path in (per_seed_path, aggregate_path):
        if path.exists():
            print(f"replacing_existing_summary={path}")
    per_seed.to_csv(per_seed_path, index=False)
    aggregate.to_csv(aggregate_path, index=False)
    return per_seed_path, aggregate_path


def print_aggregate(aggregate: pd.DataFrame) -> None:
    columns = [
        "model_key",
        "contact_latent_mode",
        "action_select_mode",
        "available_seeds",
        "total_completed_points",
        "safe_successes",
        "safe_success_rate_pooled",
        "safe_success_ci95_lower",
        "safe_success_ci95_upper",
        "safe_success_rate_seed_mean",
        "safe_success_rate_seed_std",
    ]
    print("\nSafe-success aggregate:")
    print(aggregate[columns].to_string(index=False, float_format=lambda value: f"{value:.4f}"))


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    validate_args(args)
    failures: list[tuple[int, int, int]] = []
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"

    if not args.dry_run and not args.aggregate_only:
        print(f"suite_plan={write_suite_plan(args)}")

    if not args.aggregate_only:
        for seed, rollout_seed_base in seed_configurations(args):
            if seed_suite_complete(args, seed, rollout_seed_base):
                print(
                    f"\n[point_set_seed={seed},rollout_seed_base={rollout_seed_base}]"
                    "\nskipping_complete_seed_configuration=true"
                )
                continue
            command = build_seed_command(args, seed, rollout_seed_base)
            quoted = " ".join(shlex.quote(part) for part in command)
            print(
                f"\n[point_set_seed={seed},rollout_seed_base={rollout_seed_base}]"
                f"\nPYTHONPATH=src {quoted}"
            )
            if args.dry_run:
                continue
            result = subprocess.run(command, cwd=REPO_ROOT, env=env)
            if result.returncode != 0:
                failures.append((seed, rollout_seed_base, result.returncode))
                if not args.keep_going:
                    break

    if args.dry_run:
        return 0

    per_seed = collect_seed_summaries(args)
    aggregate = aggregate_seed_summaries(per_seed)
    per_seed_path, aggregate_path = write_summaries(args.output_base, per_seed, aggregate)
    print(f"per_seed_summary={per_seed_path}")
    print(f"aggregate_summary={aggregate_path}")
    print_aggregate(aggregate)

    if failures:
        for seed, rollout_seed_base, return_code in failures:
            print(
                f"seed_failure=point_set_seed:{seed},"
                f"rollout_seed_base:{rollout_seed_base},return_code:{return_code}"
            )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
