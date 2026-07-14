#!/usr/bin/env python3
"""Recompute safe-success statistics from completed rollout summaries."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


MODELS = ("mix50", "mix100", "mix150", "mix203")
POINT_PATTERN = re.compile(r"point_(\d+)_")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--evaluation-root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--force-threshold",
        type=float,
        required=True,
    )
    return parser.parse_args()


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value

    if isinstance(value, (int, float, np.integer, np.floating)):
        return bool(value)

    if value is None:
        return False

    text = str(value).strip().lower()

    if text in {"true", "1", "yes", "y"}:
        return True

    if text in {"false", "0", "no", "n", "", "none", "nan"}:
        return False

    raise ValueError(f"Cannot parse boolean value: {value!r}")


def finite_float(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")

    return result if math.isfinite(result) else float("nan")


def first_value(
    data: dict[str, Any],
    candidates: list[str],
) -> Any:
    for name in candidates:
        if name in data and data[name] is not None:
            return data[name]

    return None


def find_column(
    frame: pd.DataFrame,
    candidates: list[str],
) -> str:
    for name in candidates:
        if name in frame.columns:
            return name

    raise KeyError(
        f"Missing columns {candidates}; "
        f"available={frame.columns.tolist()}"
    )


def load_task_points(run_root: Path) -> pd.DataFrame:
    frame = pd.read_csv(run_root / "task_points.csv")

    point_col = find_column(frame, ["point_index"])
    x_col = find_column(
        frame,
        ["hole_offset_x", "x_offset"],
    )
    y_col = find_column(
        frame,
        ["hole_offset_y", "y_offset"],
    )
    z_col = find_column(
        frame,
        ["hole_offset_z", "z_offset"],
    )
    seed_col = find_column(
        frame,
        ["rollout_seed", "seed"],
    )

    result = pd.DataFrame({
        "point_index": pd.to_numeric(frame[point_col]),
        "hole_offset_x": pd.to_numeric(frame[x_col]),
        "hole_offset_y": pd.to_numeric(frame[y_col]),
        "hole_offset_z": pd.to_numeric(frame[z_col]),
        "rollout_seed": pd.to_numeric(frame[seed_col]),
    })

    return (
        result
        .sort_values("point_index")
        .reset_index(drop=True)
    )


def load_model(
    evaluation_root: Path,
    model: str,
    force_threshold: float,
) -> pd.DataFrame:
    run_root = evaluation_root / f"{model}_zero_mid"
    task_points = load_task_points(run_root)

    rows: list[dict[str, Any]] = []

    for summary_path in sorted(
        run_root.glob("point_*/summary.json")
    ):
        match = POINT_PATTERN.search(
            summary_path.parent.name
        )

        if match is None:
            continue

        point_index = int(match.group(1))
        data = json.loads(
            summary_path.read_text(encoding="utf-8")
        )

        success = parse_bool(
            first_value(
                data,
                ["success", "task_success"],
            )
        )

        max_force = finite_float(
            first_value(
                data,
                [
                    "max_force_norm",
                    "maximum_force_norm",
                    "peak_force_norm",
                ],
            )
        )

        safe_success = bool(
            success
            and math.isfinite(max_force)
            and max_force <= force_threshold
        )

        rows.append({
            "point_index": point_index,
            "success": success,
            "safe_success": safe_success,
            "unsafe_success": bool(
                success and not safe_success
            ),
            "max_force_norm": max_force,
            "mean_force_norm": finite_float(
                first_value(
                    data,
                    [
                        "mean_force_norm",
                        "average_force_norm",
                    ],
                )
            ),
            "steps_executed": finite_float(
                first_value(
                    data,
                    [
                        "steps_executed",
                        "num_steps",
                        "rollout_steps",
                        "steps",
                    ],
                )
            ),
            "success_step": finite_float(
                data.get("success_step")
            ),
            "success_time": finite_float(
                data.get("success_time")
            ),
            "stop_reason": data.get("stop_reason"),
            "final_distance": finite_float(
                first_value(
                    data,
                    [
                        "final_peg_to_hole_distance",
                        "final_peg_to_hole_dist",
                        "final_distance",
                    ],
                )
            ),
            "final_axial_error": finite_float(
                first_value(
                    data,
                    [
                        "final_peg_to_hole_axial_error",
                        "final_axial_error",
                    ],
                )
            ),
            "final_lateral_error": finite_float(
                first_value(
                    data,
                    [
                        "final_peg_to_hole_lateral_error",
                        "final_lateral_error",
                    ],
                )
            ),
            "summary_path": str(summary_path),
            "rollout_log_path": str(
                summary_path.parent / "rollout_log.csv"
            ),
        })

    results = pd.DataFrame(rows)

    if len(results) != 100:
        raise RuntimeError(
            f"{model}: expected 100 summary files, "
            f"found {len(results)}"
        )

    if results["point_index"].duplicated().any():
        raise RuntimeError(
            f"{model}: duplicate point_index detected"
        )

    merged = task_points.merge(
        results,
        on="point_index",
        how="left",
        validate="one_to_one",
    )

    if merged["success"].isna().any():
        missing = merged.loc[
            merged["success"].isna(),
            "point_index",
        ].tolist()

        raise RuntimeError(
            f"{model}: missing summaries for {missing}"
        )

    merged.insert(0, "model", model)
    return merged


def exact_mcnemar_p(
    first_only: int,
    second_only: int,
) -> float:
    discordant = first_only + second_only

    if discordant == 0:
        return 1.0

    try:
        from scipy.stats import binomtest
    except ImportError:
        return float("nan")

    return float(
        binomtest(
            min(first_only, second_only),
            n=discordant,
            p=0.5,
            alternative="two-sided",
        ).pvalue
    )


def model_summary(frame: pd.DataFrame) -> dict[str, Any]:
    success = frame["success"].astype(bool)
    safe = frame["safe_success"].astype(bool)
    unsafe = frame["unsafe_success"].astype(bool)

    successful_force = frame.loc[
        success,
        "max_force_norm",
    ].dropna()

    safe_force = frame.loc[
        safe,
        "max_force_norm",
    ].dropna()

    unsafe_force = frame.loc[
        unsafe,
        "max_force_norm",
    ].dropna()

    return {
        "model": frame["model"].iloc[0],
        "total_points": len(frame),
        "task_successes": int(success.sum()),
        "task_success_rate": float(success.mean()),
        "safe_successes": int(safe.sum()),
        "safe_success_rate": float(safe.mean()),
        "unsafe_successes": int(unsafe.sum()),
        "unsafe_success_rate": float(unsafe.mean()),
        "unsafe_share_of_successes": (
            float(unsafe.sum() / success.sum())
            if success.sum()
            else float("nan")
        ),
        "successful_force_median": (
            float(successful_force.median())
            if len(successful_force)
            else float("nan")
        ),
        "successful_force_q75": (
            float(successful_force.quantile(0.75))
            if len(successful_force)
            else float("nan")
        ),
        "successful_force_q90": (
            float(successful_force.quantile(0.90))
            if len(successful_force)
            else float("nan")
        ),
        "successful_force_max": (
            float(successful_force.max())
            if len(successful_force)
            else float("nan")
        ),
        "safe_success_force_median": (
            float(safe_force.median())
            if len(safe_force)
            else float("nan")
        ),
        "unsafe_success_force_median": (
            float(unsafe_force.median())
            if len(unsafe_force)
            else float("nan")
        ),
    }


def paired_comparison(
    wide: pd.DataFrame,
    first: str,
    second: str,
) -> tuple[dict[str, Any], pd.DataFrame]:
    first_success = wide[
        f"{first}_success"
    ].astype(bool)

    second_success = wide[
        f"{second}_success"
    ].astype(bool)

    first_safe = wide[
        f"{first}_safe_success"
    ].astype(bool)

    second_safe = wide[
        f"{second}_safe_success"
    ].astype(bool)

    first_only_success = (
        first_success & ~second_success
    )
    second_only_success = (
        ~first_success & second_success
    )
    both_success = (
        first_success & second_success
    )
    both_fail = (
        ~first_success & ~second_success
    )

    first_only_safe = first_safe & ~second_safe
    second_only_safe = ~first_safe & second_safe

    first_unique_safe = (
        first_only_success & first_safe
    )
    first_unique_unsafe = (
        first_only_success & ~first_safe
    )

    common = wide.loc[both_success].copy()

    force_difference = (
        common[f"{first}_max_force_norm"]
        - common[f"{second}_max_force_norm"]
    )

    row = {
        "comparison": f"{first}_vs_{second}",
        "both_task_success": int(both_success.sum()),
        f"{first}_only_task_success":
            int(first_only_success.sum()),
        f"{second}_only_task_success":
            int(second_only_success.sum()),
        "both_task_fail": int(both_fail.sum()),
        f"task_success_net_gain_{first}": int(
            first_only_success.sum()
            - second_only_success.sum()
        ),
        "task_mcnemar_exact_p": exact_mcnemar_p(
            int(first_only_success.sum()),
            int(second_only_success.sum()),
        ),
        f"{first}_unique_safe_success":
            int(first_unique_safe.sum()),
        f"{first}_unique_unsafe_success":
            int(first_unique_unsafe.sum()),
        f"{first}_unique_unsafe_fraction": (
            float(
                first_unique_unsafe.sum()
                / first_only_success.sum()
            )
            if first_only_success.sum()
            else float("nan")
        ),
        f"{first}_only_safe_success":
            int(first_only_safe.sum()),
        f"{second}_only_safe_success":
            int(second_only_safe.sum()),
        f"safe_success_net_gain_{first}": int(
            first_only_safe.sum()
            - second_only_safe.sum()
        ),
        "safe_mcnemar_exact_p": exact_mcnemar_p(
            int(first_only_safe.sum()),
            int(second_only_safe.sum()),
        ),
        "common_task_success_pairs":
            int(len(common)),
        f"common_{first}_force_median": (
            float(
                common[
                    f"{first}_max_force_norm"
                ].median()
            )
            if len(common)
            else float("nan")
        ),
        f"common_{second}_force_median": (
            float(
                common[
                    f"{second}_max_force_norm"
                ].median()
            )
            if len(common)
            else float("nan")
        ),
        f"common_force_difference_median_{first}_minus_{second}":
            (
                float(force_difference.median())
                if len(force_difference)
                else float("nan")
            ),
    }

    cases = wide.copy()
    cases["comparison"] = (
        f"{first}_vs_{second}"
    )

    conditions = [
        first_only_success & ~first_safe,
        first_only_success & first_safe,
        second_only_success & second_safe,
        second_only_success & ~second_safe,
        both_success & ~first_safe,
        both_success & first_safe & second_safe,
        both_fail,
    ]

    labels = [
        f"{first}_only_unsafe_success",
        f"{first}_only_safe_success",
        f"{second}_only_safe_success",
        f"{second}_only_unsafe_success",
        f"both_success_{first}_unsafe",
        "both_safe_success",
        "both_fail",
    ]

    cases["case_type"] = np.select(
        conditions,
        labels,
        default="other",
    )

    return row, cases


def main() -> int:
    args = parse_args()

    if args.force_threshold <= 0:
        raise ValueError(
            "--force-threshold must be positive"
        )

    root = (
        args.evaluation_root
        .expanduser()
        .resolve()
    )

    tag = (
        f"{args.force_threshold:g}N"
        .replace(".", "p")
    )

    frames = {
        model: load_model(
            root,
            model,
            args.force_threshold,
        )
        for model in MODELS
    }

    summaries = pd.DataFrame([
        model_summary(frames[model])
        for model in MODELS
    ])

    coordinate_columns = [
        "point_index",
        "hole_offset_x",
        "hole_offset_y",
        "hole_offset_z",
        "rollout_seed",
    ]

    wide = frames["mix50"][
        coordinate_columns
    ].copy()

    result_columns = [
        "success",
        "safe_success",
        "unsafe_success",
        "max_force_norm",
        "mean_force_norm",
        "steps_executed",
        "success_step",
        "success_time",
        "stop_reason",
        "final_distance",
        "final_axial_error",
        "final_lateral_error",
        "summary_path",
        "rollout_log_path",
    ]

    for model, frame in frames.items():
        renamed = frame[
            ["point_index", *result_columns]
        ].rename(
            columns={
                column: f"{model}_{column}"
                for column in result_columns
            }
        )

        wide = wide.merge(
            renamed,
            on="point_index",
            how="left",
            validate="one_to_one",
        )

    pair_rows = []
    case_frames = []

    for comparator in ("mix100", "mix203"):
        pair_row, cases = paired_comparison(
            wide,
            "mix150",
            comparator,
        )
        pair_rows.append(pair_row)
        case_frames.append(cases)

    pair_summary = pd.DataFrame(pair_rows)

    candidate_cases = pd.concat(
        case_frames,
        ignore_index=True,
    )

    model_path = (
        root
        / f"safety_model_summary_{tag}.csv"
    )
    pair_path = (
        root
        / f"mix150_pairwise_summary_{tag}.csv"
    )
    pointwise_path = (
        root
        / f"safety_pointwise_results_{tag}.csv"
    )
    cases_path = (
        root
        / f"mix150_candidate_cases_{tag}.csv"
    )

    summaries.to_csv(
        model_path,
        index=False,
    )
    pair_summary.to_csv(
        pair_path,
        index=False,
    )
    wide.to_csv(
        pointwise_path,
        index=False,
    )
    candidate_cases.to_csv(
        cases_path,
        index=False,
    )

    print("=" * 190)
    print(
        f"MODEL SAFETY SUMMARY: "
        f"max_force_norm <= {args.force_threshold:g} N"
    )
    print("=" * 190)
    print(
        summaries.to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )

    print()
    print("=" * 240)
    print("PAIRED MIX150 COMPARISON")
    print("=" * 240)
    print(
        pair_summary.to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )

    print()
    print("=" * 160)
    print("MIX150 RELEVANT CASE COUNTS")
    print("=" * 160)

    relevant = candidate_cases[
        candidate_cases["case_type"].str.contains(
            "mix150",
            regex=False,
        )
    ]

    print(
        relevant.groupby(
            ["comparison", "case_type"]
        ).size().to_string()
    )

    print()
    print(f"saved={model_path}")
    print(f"saved={pair_path}")
    print(f"saved={pointwise_path}")
    print(f"saved={cases_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
