#!/usr/bin/env python3
"""Analyze peg-in-hole contact-stage behavior from rollout logs and HDF5 demos."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

import h5py
import numpy as np


VECTOR_DIM = 7
SUMMARY_COLUMNS = (
    "source",
    "mode",
    "n_steps",
    "first_force_gt_5_step",
    "first_force_gt_5_time",
    "first_force_gt_10_step",
    "first_force_gt_10_time",
    "first_force_gt_20_step",
    "first_force_gt_20_time",
    "first_force_gt_50_step",
    "first_force_gt_50_time",
    "dist_at_force20",
    "axial_at_force20",
    "lateral_at_force20",
    "min_dist",
    "axial_at_min_dist",
    "lateral_at_min_dist",
    "max_force",
    "dist_at_max_force",
    "axial_at_max_force",
    "lateral_at_max_force",
    "mean_lateral_before_contact",
    "mean_lateral_after_contact",
    "mean_force_after_contact",
    "mean_target_delta_after_contact",
    "mean_applied_delta_after_contact",
    "mean_action_delta_from_qpos_after_contact",
    "mean_action_step_delta_after_contact",
    "min_abs_entrance_axial_error",
    "step_at_min_abs_entrance_axial_error",
    "force_rises_near_entrance",
    "lateral_error_decreases_after_contact",
)


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None:
            raise ValueError(f"{path}: CSV has no header")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path}: CSV has no rows")
    return rows


def _column(rows: Sequence[dict[str, str]], name: str, default: float = np.nan) -> np.ndarray:
    values = []
    for row in rows:
        raw = row.get(name, "")
        if raw == "":
            values.append(default)
        else:
            try:
                values.append(float(raw))
            except ValueError:
                values.append(default)
    return np.asarray(values, dtype=np.float64)


def _vector_columns(
    rows: Sequence[dict[str, str]],
    prefix: str,
    dim: int = VECTOR_DIM,
) -> Optional[np.ndarray]:
    names = [f"{prefix}_{index}" for index in range(dim)]
    if not all(name in rows[0] for name in names):
        return None
    return np.stack([_column(rows, name) for name in names], axis=1)


def _safe_nanmean(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(finite.mean()) if len(finite) else float("nan")


def _row_value(values: np.ndarray, index: Optional[int]) -> float:
    if index is None or index < 0 or index >= len(values):
        return float("nan")
    return float(values[index])


def _first_threshold_index(force_norm: np.ndarray, threshold: float) -> Optional[int]:
    indices = np.flatnonzero(force_norm > threshold)
    return int(indices[0]) if len(indices) else None


def _index_of_nanmin(values: np.ndarray) -> Optional[int]:
    finite = np.flatnonzero(np.isfinite(values))
    if not len(finite):
        return None
    return int(finite[np.nanargmin(values[finite])])


def _index_of_nanmax(values: np.ndarray) -> Optional[int]:
    finite = np.flatnonzero(np.isfinite(values))
    if not len(finite):
        return None
    return int(finite[np.nanargmax(values[finite])])


def _command_delta(values_a: Optional[np.ndarray], values_b: Optional[np.ndarray]) -> np.ndarray:
    if values_a is None or values_b is None:
        return np.full(0, np.nan, dtype=np.float64)
    return np.linalg.norm(values_a - values_b, axis=1)


def analyze_contact_arrays(
    source: str,
    mode: str,
    steps: np.ndarray,
    times: np.ndarray,
    dist: np.ndarray,
    axial: np.ndarray,
    lateral: np.ndarray,
    force_norm: np.ndarray,
    hole_entrance_offset: float,
    force_thresholds: Sequence[float],
    target_delta_norm: Optional[np.ndarray] = None,
    applied_delta_norm: Optional[np.ndarray] = None,
    action_delta_from_qpos: Optional[np.ndarray] = None,
    action_step_delta: Optional[np.ndarray] = None,
) -> dict[str, object]:
    """Summarize contact-stage arrays into one table row."""

    entrance_axial_error = axial - hole_entrance_offset
    abs_entrance_axial_error = np.abs(entrance_axial_error)
    threshold_indices = {
        float(threshold): _first_threshold_index(force_norm, float(threshold))
        for threshold in force_thresholds
    }
    force20_index = threshold_indices.get(20.0)
    if force20_index is None and force_thresholds:
        force20_index = threshold_indices.get(float(force_thresholds[0]))

    min_dist_index = _index_of_nanmin(dist)
    max_force_index = _index_of_nanmax(force_norm)
    min_lateral_index = _index_of_nanmin(lateral)
    min_abs_entrance_index = _index_of_nanmin(abs_entrance_axial_error)

    contact_index = threshold_indices.get(5.0)
    if contact_index is None and force_thresholds:
        contact_index = threshold_indices.get(float(force_thresholds[0]))

    before_mask = np.arange(len(force_norm)) < contact_index if contact_index is not None else np.ones(len(force_norm), dtype=bool)
    after_mask = np.arange(len(force_norm)) >= contact_index if contact_index is not None else np.zeros(len(force_norm), dtype=bool)
    near_entrance_mask = np.isfinite(entrance_axial_error) & (np.abs(entrance_axial_error) <= 0.005)
    force_rises_near_entrance = bool(np.any(near_entrance_mask & (force_norm >= 20.0)))
    mean_lateral_before = _safe_nanmean(lateral[before_mask])
    mean_lateral_after = _safe_nanmean(lateral[after_mask])
    lateral_error_decreases_after_contact = bool(
        np.isfinite(mean_lateral_before)
        and np.isfinite(mean_lateral_after)
        and mean_lateral_after < mean_lateral_before
    )

    summary: dict[str, object] = {
        "source": source,
        "mode": mode,
        "n_steps": int(len(steps)),
        "dist_at_force20": _row_value(dist, force20_index),
        "axial_at_force20": _row_value(axial, force20_index),
        "lateral_at_force20": _row_value(lateral, force20_index),
        "min_dist": _row_value(dist, min_dist_index),
        "axial_at_min_dist": _row_value(axial, min_dist_index),
        "lateral_at_min_dist": _row_value(lateral, min_dist_index),
        "max_force": _row_value(force_norm, max_force_index),
        "dist_at_max_force": _row_value(dist, max_force_index),
        "axial_at_max_force": _row_value(axial, max_force_index),
        "lateral_at_max_force": _row_value(lateral, max_force_index),
        "mean_lateral_before_contact": mean_lateral_before,
        "mean_lateral_after_contact": mean_lateral_after,
        "mean_force_after_contact": _safe_nanmean(force_norm[after_mask]),
        "mean_target_delta_after_contact": (
            _safe_nanmean(target_delta_norm[after_mask])
            if target_delta_norm is not None and len(target_delta_norm)
            else float("nan")
        ),
        "mean_applied_delta_after_contact": (
            _safe_nanmean(applied_delta_norm[after_mask])
            if applied_delta_norm is not None and len(applied_delta_norm)
            else float("nan")
        ),
        "mean_action_delta_from_qpos_after_contact": (
            _safe_nanmean(action_delta_from_qpos[after_mask])
            if action_delta_from_qpos is not None and len(action_delta_from_qpos)
            else float("nan")
        ),
        "mean_action_step_delta_after_contact": (
            _safe_nanmean(action_step_delta[after_mask])
            if action_step_delta is not None and len(action_step_delta)
            else float("nan")
        ),
        "min_abs_entrance_axial_error": _row_value(abs_entrance_axial_error, min_abs_entrance_index),
        "step_at_min_abs_entrance_axial_error": _row_value(steps, min_abs_entrance_index),
        "force_rises_near_entrance": force_rises_near_entrance,
        "lateral_error_decreases_after_contact": lateral_error_decreases_after_contact,
    }

    for threshold in force_thresholds:
        threshold_value = float(threshold)
        threshold_key = int(threshold_value) if threshold_value.is_integer() else threshold_value
        index = threshold_indices[threshold_value]
        summary[f"first_force_gt_{threshold_key}_step"] = _row_value(steps, index)
        summary[f"first_force_gt_{threshold_key}_time"] = _row_value(times, index)

    for column in SUMMARY_COLUMNS:
        summary.setdefault(column, float("nan"))
    return summary


def analyze_rollout_csv(
    path: Path,
    hole_entrance_offset: float,
    force_thresholds: Sequence[float],
) -> dict[str, object]:
    rows = _read_csv_rows(path)
    required = (
        "step",
        "time",
        "peg_to_hole_dist",
        "peg_to_hole_axial_error",
        "peg_to_hole_lateral_error",
        "force_norm",
    )
    missing = [name for name in required if name not in rows[0]]
    if missing:
        raise KeyError(f"{path}: missing rollout columns: {', '.join(missing)}")

    target_ctrl = _vector_columns(rows, "target_ctrl")
    applied_ctrl = _vector_columns(rows, "applied_ctrl")
    current_qpos = _vector_columns(rows, "current_qpos")
    if current_qpos is None:
        current_qpos = _vector_columns(rows, "qpos")

    target_delta_norm = _column(rows, "target_ctrl_delta_from_qpos_norm")
    if not np.isfinite(target_delta_norm).any():
        target_delta_norm = _command_delta(target_ctrl, current_qpos)
    applied_delta_norm = _column(rows, "applied_ctrl_delta_from_qpos_norm")
    if not np.isfinite(applied_delta_norm).any():
        applied_delta_norm = _command_delta(applied_ctrl, current_qpos)

    return analyze_contact_arrays(
        source=str(path),
        mode="rollout",
        steps=_column(rows, "step"),
        times=_column(rows, "time"),
        dist=_column(rows, "peg_to_hole_dist"),
        axial=_column(rows, "peg_to_hole_axial_error"),
        lateral=_column(rows, "peg_to_hole_lateral_error"),
        force_norm=_column(rows, "force_norm"),
        hole_entrance_offset=hole_entrance_offset,
        force_thresholds=force_thresholds,
        target_delta_norm=target_delta_norm,
        applied_delta_norm=applied_delta_norm,
    )


def _first_existing_dataset(handle: h5py.File, keys: Sequence[str]) -> Optional[str]:
    for key in keys:
        if key in handle:
            return key
    return None


def _nearest_force_norm_to_state(handle: h5py.File, n_state: int) -> np.ndarray:
    if "observations/ft_wrench" not in handle:
        return np.full(n_state, np.nan, dtype=np.float64)
    ft_wrench = np.asarray(handle["observations/ft_wrench"], dtype=np.float64)
    force_norm = np.linalg.norm(ft_wrench[:, :3], axis=1)
    state_key = _first_existing_dataset(handle, ("timestamps/state_episode", "timestamps/state"))
    force_key = _first_existing_dataset(handle, ("timestamps/force_episode", "timestamps/force"))
    if state_key is None or force_key is None:
        indices = np.linspace(0, len(force_norm) - 1, num=n_state).round().astype(int)
        return force_norm[indices]
    state_ts = np.asarray(handle[state_key], dtype=np.float64)[:n_state]
    force_ts = np.asarray(handle[force_key], dtype=np.float64)
    indices = np.searchsorted(force_ts, state_ts, side="left")
    indices = np.clip(indices, 0, len(force_norm) - 1)
    before = np.clip(indices - 1, 0, len(force_norm) - 1)
    choose_before = np.abs(force_ts[before] - state_ts) < np.abs(force_ts[indices] - state_ts)
    indices = np.where(choose_before, before, indices)
    return force_norm[indices]


def analyze_hdf5_episode(
    path: Path,
    force_thresholds: Sequence[float],
) -> dict[str, object]:
    with h5py.File(path, "r") as handle:
        if "observations/joint_pos" not in handle:
            raise KeyError(f"{path}: missing observations/joint_pos")
        joint_pos = np.asarray(handle["observations/joint_pos"], dtype=np.float64)
        n_state = len(joint_pos)
        action = (
            np.asarray(handle["action"], dtype=np.float64)
            if "action" in handle
            else np.full_like(joint_pos, np.nan)
        )
        time_key = _first_existing_dataset(handle, ("timestamps/state_episode", "timestamps/state"))
        times = (
            np.asarray(handle[time_key], dtype=np.float64)[:n_state]
            if time_key is not None
            else np.arange(n_state, dtype=np.float64)
        )
        force_norm = _nearest_force_norm_to_state(handle, n_state)

    action_delta_from_qpos = np.linalg.norm(action - joint_pos, axis=1)
    action_step_delta = np.full(n_state, np.nan, dtype=np.float64)
    if n_state > 1:
        action_step_delta[:-1] = np.linalg.norm(np.diff(action, axis=0), axis=1)

    return analyze_contact_arrays(
        source=str(path),
        mode="hdf5",
        steps=np.arange(n_state, dtype=np.float64),
        times=times,
        dist=np.full(n_state, np.nan, dtype=np.float64),
        axial=np.full(n_state, np.nan, dtype=np.float64),
        lateral=np.full(n_state, np.nan, dtype=np.float64),
        force_norm=force_norm,
        hole_entrance_offset=0.0,
        force_thresholds=force_thresholds,
        action_delta_from_qpos=action_delta_from_qpos,
        action_step_delta=action_step_delta,
    )


def _write_summary_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    extra_columns = sorted({key for row in rows for key in row} - set(SUMMARY_COLUMNS))
    fieldnames = list(SUMMARY_COLUMNS) + extra_columns
    with path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _print_summary(rows: Sequence[dict[str, object]]) -> None:
    print("Contact Stage Summary")
    print("---------------------")
    for row in rows:
        print(
            " ".join(
                [
                    f"mode={row['mode']}",
                    f"source={row['source']}",
                    f"n_steps={row['n_steps']}",
                    f"force20_step={row.get('first_force_gt_20_step')}",
                    f"min_dist={row.get('min_dist')}",
                    f"max_force={row.get('max_force')}",
                    f"mean_lateral_after_contact={row.get('mean_lateral_after_contact')}",
                    f"mean_target_delta_after_contact={row.get('mean_target_delta_after_contact')}",
                    f"mean_applied_delta_after_contact={row.get('mean_applied_delta_after_contact')}",
                ]
            )
        )


def _load_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError("matplotlib is required for --plot") from error
    return plt


def _plot_rollout(path: Path, output_dir: Path, hole_entrance_offset: float) -> None:
    rows = _read_csv_rows(path)
    time = _column(rows, "time")
    dist = _column(rows, "peg_to_hole_dist")
    axial = _column(rows, "peg_to_hole_axial_error")
    lateral = _column(rows, "peg_to_hole_lateral_error")
    force_norm = _column(rows, "force_norm")
    entrance = axial - hole_entrance_offset
    plt = _load_matplotlib()
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = path.stem

    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    axes[0].plot(time, dist, label="distance")
    axes[0].plot(time, axial, label="axial")
    axes[0].plot(time, lateral, label="lateral")
    axes[0].legend()
    axes[0].set_ylabel("m")
    axes[1].plot(time, force_norm)
    axes[1].set_ylabel("force norm")
    axes[2].plot(time, entrance)
    axes[2].axhline(0.0, color="black", linewidth=1)
    axes[2].set_ylabel("entrance axial error")
    axes[2].set_xlabel("time")
    fig.tight_layout()
    fig.savefig(output_dir / f"{stem}_contact_timeseries.png", dpi=150)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(6, 5))
    axis.scatter(force_norm, lateral, s=12)
    axis.set_xlabel("force norm")
    axis.set_ylabel("lateral error")
    fig.tight_layout()
    fig.savefig(output_dir / f"{stem}_lateral_vs_force.png", dpi=150)
    plt.close(fig)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rollout_logs", type=Path, nargs="*", help="rollout_log.csv files")
    parser.add_argument("--hdf5", type=Path, nargs="*", default=())
    parser.add_argument("--hole-entrance-offset", type=float, default=0.024)
    parser.add_argument("--force-thresholds", type=float, nargs="+", default=(5.0, 10.0, 20.0, 50.0))
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--plot", type=Path, default=None)
    parser.add_argument("--model-xml", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.model_xml is not None:
        print("warning: --model-xml is accepted for future MuJoCo replay but not used in this first version")
    if not args.rollout_logs and not args.hdf5:
        print("error: provide at least one rollout_log.csv or --hdf5 episode", file=sys.stderr)
        return 2
    summaries: list[dict[str, object]] = []
    try:
        for path in args.rollout_logs:
            summaries.append(
                analyze_rollout_csv(
                    path,
                    hole_entrance_offset=args.hole_entrance_offset,
                    force_thresholds=args.force_thresholds,
                )
            )
            if args.plot is not None:
                _plot_rollout(path, args.plot, args.hole_entrance_offset)
        for path in args.hdf5:
            summaries.append(analyze_hdf5_episode(path, force_thresholds=args.force_thresholds))
    except Exception as error:
        print(f"error: contact-stage analysis failed: {error}", file=sys.stderr)
        return 1

    _print_summary(summaries)
    if args.output_csv is not None:
        _write_summary_csv(args.output_csv, summaries)
        print(f"saved_summary_csv={args.output_csv}")
    if args.plot is not None:
        manifest = {"plots_for": [str(path) for path in args.rollout_logs]}
        args.plot.mkdir(parents=True, exist_ok=True)
        (args.plot / "plot_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"saved_plots_dir={args.plot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
