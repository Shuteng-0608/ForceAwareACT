#!/usr/bin/env python3
"""Plot sensor, task-error, and policy-update traces from MuJoCo rollouts."""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "force_aware_act_matplotlib"))

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Thresholds:
    contact_force: float = 5.0
    high_force: float = 20.0
    very_high_force: float = 40.0
    success_distance: float = 0.005
    success_lateral: float = 0.006
    success_force: float = 40.0
    success_hold_steps: int = 15


TASK_ERROR_COLUMNS = (
    ("peg_to_hole_dist", "distance"),
    ("abs_peg_to_hole_axial_error", "|axial error|"),
    ("peg_to_hole_lateral_error", "lateral error"),
)
FORCE_COLUMNS = (
    ("force_norm", "force norm"),
    ("ft_0", "force x"),
    ("ft_1", "force y"),
    ("ft_2", "force z"),
)
ACTION_COLUMNS = (
    ("selected_action_delta_norm_raw_to_current", "selected raw-to-current"),
    ("selected_action_delta_norm_after_clip", "selected after clip"),
    ("selected_action_delta_norm_after_ema", "selected after ema"),
    ("target_ctrl_delta_from_qpos_norm", "target ctrl from qpos"),
    ("applied_ctrl_delta_from_qpos_norm", "applied ctrl from qpos"),
)
PRED_FORCE_COLUMNS = (
    ("pred_force_norm_0", "pred force t0"),
    ("pred_force_norm_mean", "pred force mean"),
    ("pred_force_norm_max", "pred force max"),
)


def _load_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _warn(message: str) -> None:
    print(f"warning: {message}")


def _to_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    if value in (None, ""):
        return default
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if np.isfinite(number) else default


def _to_int(value: Any, default: Optional[int] = None) -> Optional[int]:
    if value in (None, ""):
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def load_rollout_log(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"rollout log does not exist: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"rollout log is empty: {path}")
    for column in df.columns:
        converted = pd.to_numeric(df[column], errors="coerce")
        if converted.notna().any():
            df[column] = converted
    if "peg_to_hole_axial_error" in df.columns:
        df["abs_peg_to_hole_axial_error"] = df["peg_to_hole_axial_error"].abs()
    return df


def load_summary_json(path: Optional[Path]) -> dict[str, Any]:
    if path is None:
        return {}
    if not path.is_file():
        _warn(f"summary JSON not found, continuing without it: {path}")
        return {}
    with path.open() as summary_file:
        return json.load(summary_file)


def choose_x_axis(df: pd.DataFrame) -> tuple[pd.Series, str]:
    if "time" in df.columns:
        return pd.to_numeric(df["time"], errors="coerce"), "time (s)"
    if "step" in df.columns:
        return pd.to_numeric(df["step"], errors="coerce"), "policy step"
    return pd.Series(np.arange(len(df))), "row index"


def find_first_threshold_crossing(series: pd.Series, threshold: float) -> Optional[int]:
    values = pd.to_numeric(series, errors="coerce")
    crossings = values[values > threshold]
    if crossings.empty:
        return None
    return int(crossings.index[0])


def _first_hold_index(condition: pd.Series, hold_steps: int) -> Optional[int]:
    counter = 0
    for index, value in condition.fillna(False).astype(bool).items():
        counter = counter + 1 if value else 0
        if counter >= hold_steps:
            return int(index)
    return None


def _value_at(df: pd.DataFrame, index: Optional[int], column: str) -> Optional[float]:
    if index is None or column not in df.columns or index not in df.index:
        return None
    return _to_float(df.at[index, column])


def _step_at(df: pd.DataFrame, index: Optional[int]) -> Optional[int]:
    if index is None:
        return None
    if "step" in df.columns:
        return _to_int(df.at[index, "step"], index)
    return int(index)


def _time_at(df: pd.DataFrame, index: Optional[int]) -> Optional[float]:
    if index is None:
        return None
    if "time" in df.columns:
        return _to_float(df.at[index, "time"])
    return None


def compute_retroactive_success(
    df: pd.DataFrame,
    thresholds: Thresholds,
    hold_steps: Optional[int] = None,
) -> dict[str, Any]:
    hold_steps = thresholds.success_hold_steps if hold_steps is None else hold_steps
    if "success_hold_counter" in df.columns:
        counters = pd.to_numeric(df["success_hold_counter"], errors="coerce")
        matches = counters[counters >= hold_steps]
        if not matches.empty:
            index = int(matches.index[0])
            return {
                "success": True,
                "success_step": _step_at(df, index),
                "success_time": _time_at(df, index),
                "success_hold_steps_observed": int(counters.max()),
                "success_source": "success_hold_counter",
            }

    if "success_condition" in df.columns:
        condition = df["success_condition"].astype(str).str.lower().isin({"1", "true", "yes"})
    else:
        required = {"peg_to_hole_dist", "peg_to_hole_lateral_error", "force_norm"}
        if not required.issubset(df.columns):
            return {
                "success": False,
                "success_step": None,
                "success_time": None,
                "success_hold_steps_observed": 0,
                "success_source": "unavailable",
            }
        condition = (
            (pd.to_numeric(df["peg_to_hole_dist"], errors="coerce") < thresholds.success_distance)
            & (
                pd.to_numeric(df["peg_to_hole_lateral_error"], errors="coerce")
                < thresholds.success_lateral
            )
            & (pd.to_numeric(df["force_norm"], errors="coerce") < thresholds.success_force)
        )

    index = _first_hold_index(condition, hold_steps)
    max_hold = 0
    counter = 0
    for value in condition.fillna(False).astype(bool):
        counter = counter + 1 if value else 0
        max_hold = max(max_hold, counter)
    return {
        "success": index is not None,
        "success_step": _step_at(df, index),
        "success_time": _time_at(df, index),
        "success_hold_steps_observed": max_hold,
        "success_source": "retroactive_thresholds" if "success_condition" not in df.columns else "success_condition",
    }


def _summary_value(summary: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in summary and summary[key] is not None:
            return summary[key]
    return None


def _first_nonempty_stop_reason(df: pd.DataFrame) -> str:
    if "stop_reason" not in df.columns:
        return ""
    reasons = df["stop_reason"].dropna().astype(str)
    reasons = reasons[reasons.str.len() > 0]
    return str(reasons.iloc[-1]) if not reasons.empty else ""


def compute_markers(
    df: pd.DataFrame,
    summary: dict[str, Any],
    thresholds: Thresholds,
) -> dict[str, Any]:
    markers: dict[str, Any] = {}
    if "force_norm" in df.columns:
        force = pd.to_numeric(df["force_norm"], errors="coerce")
        for name, threshold in (
            ("first_contact", thresholds.contact_force),
            ("first_high_force", thresholds.high_force),
            ("first_very_high_force", thresholds.very_high_force),
        ):
            index = find_first_threshold_crossing(force, threshold)
            markers[f"{name}_step"] = _step_at(df, index)
            markers[f"{name}_time"] = _time_at(df, index)
        max_index = int(force.idxmax()) if force.notna().any() else None
        markers["max_force_step"] = _step_at(df, max_index)
        markers["max_force_time"] = _time_at(df, max_index)
        markers["max_force_norm"] = _value_at(df, max_index, "force_norm")
    else:
        _warn("missing force_norm; force markers unavailable")

    for column, prefix in (
        ("peg_to_hole_dist", "min_dist"),
        ("peg_to_hole_lateral_error", "min_lateral"),
    ):
        if column not in df.columns:
            _warn(f"missing {column}; {prefix} marker unavailable")
            continue
        values = pd.to_numeric(df[column], errors="coerce")
        index = int(values.idxmin()) if values.notna().any() else None
        markers[f"{prefix}_step"] = _step_at(df, index)
        markers[f"{prefix}_time"] = _time_at(df, index)
        markers[prefix] = _value_at(df, index, column)

    final = df.iloc[-1]
    markers["final_dist"] = _to_float(final.get("peg_to_hole_dist"))
    markers["final_lateral"] = _to_float(final.get("peg_to_hole_lateral_error"))
    markers["final_axial"] = _to_float(final.get("peg_to_hole_axial_error"))

    retro_success = compute_retroactive_success(df, thresholds)
    summary_success_step = _summary_value(summary, "success_step")
    summary_success_time = _summary_value(summary, "success_time")
    if summary_success_step is not None or summary_success_time is not None:
        markers["success"] = bool(summary.get("success", True))
        markers["success_step"] = _to_int(summary_success_step)
        markers["success_time"] = _to_float(summary_success_time)
        markers["success_source"] = "summary_json"
    elif "success" in summary:
        markers["success"] = bool(summary.get("success"))
        markers["success_step"] = None
        markers["success_time"] = None
        markers["success_source"] = "summary_json"
    else:
        markers.update(retro_success)
    markers["success_hold_steps_observed"] = summary.get(
        "success_hold_steps_observed",
        retro_success.get("success_hold_steps_observed", 0),
    )
    markers["stop_reason"] = summary.get("stop_reason") or _first_nonempty_stop_reason(df)
    markers["steps_executed"] = _to_int(summary.get("steps_executed"), len(df))
    markers["final_time"] = _to_float(summary.get("final_time"), _time_at(df, int(df.index[-1])))
    markers["thresholds"] = {
        "contact_force": thresholds.contact_force,
        "high_force": thresholds.high_force,
        "very_high_force": thresholds.very_high_force,
        "success_distance": thresholds.success_distance,
        "success_lateral": thresholds.success_lateral,
        "success_force": thresholds.success_force,
        "success_hold_steps": thresholds.success_hold_steps,
    }
    return markers


def save_json_safe(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as json_file:
        json.dump(_json_safe(obj), json_file, indent=2, sort_keys=True)
        json_file.write("\n")
    print(f"saved_json={path}")


def _parse_formats(value: str) -> list[str]:
    formats = [item.strip().lstrip(".") for item in value.split(",") if item.strip()]
    if not formats:
        raise ValueError("--formats must include at least one format")
    return formats


def save_figure(
    fig,
    output_dir: Path,
    stem: str,
    formats: Sequence[str],
    dpi: int,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for fmt in formats:
        path = output_dir / f"{stem}.{fmt}"
        fig.savefig(path, dpi=dpi)
        paths.append(path)
        print(f"saved_plot={path}")
    return paths


def _series(df: pd.DataFrame, column: str, smooth_window: int = 1) -> Optional[pd.Series]:
    if column not in df.columns:
        _warn(f"missing column {column}; skipping curve")
        return None
    values = pd.to_numeric(df[column], errors="coerce")
    if smooth_window > 1:
        return values.rolling(window=smooth_window, center=True, min_periods=1).mean()
    return values


def _plot_columns(
    ax,
    df: pd.DataFrame,
    columns: Sequence[tuple[str, str]],
    smooth_window: int,
    ylabel: str,
) -> bool:
    x_values, _ = choose_x_axis(df)
    plotted = False
    for column, label in columns:
        values = _series(df, column, smooth_window)
        if values is None:
            continue
        ax.plot(x_values, values, linewidth=1.8, label=label)
        plotted = True
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    if plotted:
        ax.legend(loc="best", fontsize="small")
    return plotted


def _marker_x(df: pd.DataFrame, markers: dict[str, Any], name: str) -> Optional[float]:
    time = markers.get(f"{name}_time")
    step = markers.get(f"{name}_step")
    if "time" in df.columns and time is not None:
        return float(time)
    if "step" in df.columns and step is not None:
        return float(step)
    return None


def _add_vertical_markers(ax, df: pd.DataFrame, markers: dict[str, Any]) -> None:
    styles = {
        "first_contact": ("contact", "tab:orange"),
        "max_force": ("max force", "tab:red"),
        "min_lateral": ("min lateral", "tab:green"),
        "min_dist": ("min dist", "tab:blue"),
        "success": ("success", "tab:purple"),
    }
    for name, (label, color) in styles.items():
        marker_name = "success" if name == "success" else name
        if marker_name == "success":
            x_value = markers.get("success_time") if "time" in df.columns else markers.get("success_step")
        else:
            x_value = _marker_x(df, markers, marker_name)
        if x_value is None:
            continue
        ax.axvline(float(x_value), color=color, linestyle="--", linewidth=1.0, alpha=0.75, label=label)


def _dedupe_legend(ax) -> None:
    handles, labels = ax.get_legend_handles_labels()
    seen = set()
    unique_handles = []
    unique_labels = []
    for handle, label in zip(handles, labels):
        if label in seen:
            continue
        seen.add(label)
        unique_handles.append(handle)
        unique_labels.append(label)
    if unique_handles:
        ax.legend(unique_handles, unique_labels, loc="best", fontsize="small")


def _finalize_axes(fig, axes, df: pd.DataFrame, markers: dict[str, Any], x_label: str) -> None:
    for ax in np.atleast_1d(axes):
        _add_vertical_markers(ax, df, markers)
        _dedupe_legend(ax)
    np.atleast_1d(axes)[-1].set_xlabel(x_label)
    fig.tight_layout()


def _save_single_panel(
    df: pd.DataFrame,
    markers: dict[str, Any],
    output_dir: Path,
    stem: str,
    columns: Sequence[tuple[str, str]],
    ylabel: str,
    formats: Sequence[str],
    dpi: int,
    smooth_window: int,
    thresholds: Optional[Thresholds] = None,
) -> list[Path]:
    plt = _load_matplotlib()
    fig, ax = plt.subplots(figsize=(9, 4.8))
    plotted = _plot_columns(ax, df, columns, smooth_window, ylabel)
    if thresholds is not None:
        ax.axhline(thresholds.contact_force, color="tab:orange", linestyle=":", linewidth=1.0, label="contact threshold")
        ax.axhline(thresholds.high_force, color="tab:red", linestyle=":", linewidth=1.0, label="high threshold")
        ax.axhline(thresholds.very_high_force, color="black", linestyle=":", linewidth=1.0, label="very high threshold")
    x_values, x_label = choose_x_axis(df)
    ax.set_xlabel(x_label)
    ax.set_title(stem.replace("_", " "))
    _add_vertical_markers(ax, df, markers)
    _dedupe_legend(ax)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    paths = save_figure(fig, output_dir, stem, formats, dpi) if plotted else []
    plt.close(fig)
    return paths


def plot_single_rollout(
    df: pd.DataFrame,
    summary: dict[str, Any],
    output_dir: Path,
    thresholds: Thresholds,
    formats: Sequence[str],
    dpi: int,
    smooth_window: int,
    show: bool = False,
    combined: bool = True,
) -> dict[str, Any]:
    plt = _load_matplotlib()
    markers = compute_markers(df, summary, thresholds)
    save_json_safe(markers, output_dir / "summary_markers.json")

    _save_single_panel(
        df,
        markers,
        output_dir,
        "task_error_vs_time",
        TASK_ERROR_COLUMNS,
        "task error (m)",
        formats,
        dpi,
        smooth_window,
    )
    _save_single_panel(
        df,
        markers,
        output_dir,
        "force_vs_time",
        FORCE_COLUMNS,
        "force (N)",
        formats,
        dpi,
        smooth_window,
        thresholds,
    )
    _save_single_panel(
        df,
        markers,
        output_dir,
        "action_adjustment_vs_time",
        ACTION_COLUMNS,
        "joint update norm (rad)",
        formats,
        dpi,
        smooth_window,
    )
    _save_single_panel(
        df,
        markers,
        output_dir,
        "predicted_force_vs_time",
        PRED_FORCE_COLUMNS,
        "predicted force norm (N)",
        formats,
        dpi,
        smooth_window,
    )
    qcmd_columns = [(f"qcmd_{index}", f"qcmd {index}") for index in range(7)]
    qpos_columns = [(f"qpos_{index}", f"qpos {index}") for index in range(7)]
    _save_single_panel(
        df,
        markers,
        output_dir,
        "joint_command_vs_time",
        qcmd_columns,
        "joint command (rad)",
        formats,
        dpi,
        smooth_window,
    )
    _save_single_panel(
        df,
        markers,
        output_dir,
        "joint_position_vs_time",
        qpos_columns,
        "joint position (rad)",
        formats,
        dpi,
        smooth_window,
    )

    if combined:
        fig, axes = plt.subplots(4, 1, figsize=(10, 11), sharex=True)
        _plot_columns(axes[0], df, TASK_ERROR_COLUMNS, smooth_window, "task error (m)")
        _plot_columns(axes[1], df, (("force_norm", "force norm"),), smooth_window, "force (N)")
        _plot_columns(
            axes[2],
            df,
            (
                ("selected_action_delta_norm_after_ema", "selected after ema"),
                ("action_delta_norm_after_ema", "action after ema"),
                ("applied_ctrl_delta_from_qpos_norm", "applied ctrl"),
            ),
            smooth_window,
            "update norm (rad)",
        )
        _plot_columns(axes[3], df, PRED_FORCE_COLUMNS, smooth_window, "pred force (N)")
        _, x_label = choose_x_axis(df)
        _finalize_axes(fig, axes, df, markers, x_label)
        fig.suptitle("rollout sensor analysis", y=0.995)
        save_figure(fig, output_dir, "combined_analysis", formats, dpi)
        if show:
            plt.show()
        plt.close(fig)
    return markers


def _sanitize_label(label: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9_.-]+", "_", label.strip())
    return clean.strip("_") or "rollout"


def _resolve_rollout_inputs(
    rollout_dir: Optional[Path],
    rollout_log_csv: Optional[Path] = None,
    summary_json: Optional[Path] = None,
) -> tuple[Path, Optional[Path], Path]:
    if rollout_dir is None and rollout_log_csv is None:
        raise ValueError("provide --rollout-dir or --rollout-log-csv")
    if rollout_dir is not None:
        log_path = rollout_log_csv or rollout_dir / "rollout_log.csv"
        summary_path = summary_json or rollout_dir / "summary.json"
        base_dir = rollout_dir
    else:
        log_path = rollout_log_csv
        summary_path = summary_json
        base_dir = rollout_log_csv.parent
    return log_path, summary_path, base_dir


def _choose_action_compare_column(df: pd.DataFrame) -> Optional[str]:
    for column in (
        "selected_action_delta_norm_after_ema",
        "action_delta_norm_after_ema",
        "applied_ctrl_delta_from_qpos_norm",
    ):
        if column in df.columns:
            return column
    return None


def _overlay_series(
    ax,
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    column: str,
    label_a: str,
    label_b: str,
    smooth_window: int,
    suffix: str = "",
) -> bool:
    plotted = False
    for df, label in ((df_a, label_a), (df_b, label_b)):
        values = _series(df, column, smooth_window)
        if values is None:
            continue
        x_values, _ = choose_x_axis(df)
        ax.plot(x_values, values, linewidth=1.8, label=f"{label} {suffix}".strip())
        plotted = True
    ax.grid(alpha=0.25)
    if plotted:
        ax.legend(loc="best", fontsize="small")
    return plotted


def _save_compare_panel(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    output_dir: Path,
    stem: str,
    specs: Sequence[tuple[str, str]],
    label_a: str,
    label_b: str,
    ylabel: str,
    formats: Sequence[str],
    dpi: int,
    smooth_window: int,
) -> None:
    plt = _load_matplotlib()
    fig, ax = plt.subplots(figsize=(9, 4.8))
    plotted = False
    for column, suffix in specs:
        plotted = _overlay_series(
            ax,
            df_a,
            df_b,
            column,
            label_a,
            label_b,
            smooth_window,
            suffix,
        ) or plotted
    ax.set_title(stem.replace("_", " "))
    ax.set_xlabel(choose_x_axis(df_a)[1])
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    if plotted:
        save_figure(fig, output_dir, stem, formats, dpi)
    plt.close(fig)


def plot_compare_rollouts(
    df_a: pd.DataFrame,
    summary_a: dict[str, Any],
    label_a: str,
    df_b: pd.DataFrame,
    summary_b: dict[str, Any],
    label_b: str,
    output_dir: Path,
    thresholds: Thresholds,
    formats: Sequence[str],
    dpi: int,
    smooth_window: int,
    show: bool = False,
) -> dict[str, Any]:
    plt = _load_matplotlib()
    markers_a = compute_markers(df_a, summary_a, thresholds)
    markers_b = compute_markers(df_b, summary_b, thresholds)
    compare_summary = {
        "label_a": label_a,
        "label_b": label_b,
        "run_a": markers_a,
        "run_b": markers_b,
        "comparison": {
            "success_a": markers_a.get("success"),
            "success_b": markers_b.get("success"),
            "success_time_a": markers_a.get("success_time"),
            "success_time_b": markers_b.get("success_time"),
            "final_dist_a": markers_a.get("final_dist"),
            "final_dist_b": markers_b.get("final_dist"),
            "final_lateral_a": markers_a.get("final_lateral"),
            "final_lateral_b": markers_b.get("final_lateral"),
            "max_force_a": markers_a.get("max_force_norm"),
            "max_force_b": markers_b.get("max_force_norm"),
            "first_contact_time_a": markers_a.get("first_contact_time"),
            "first_contact_time_b": markers_b.get("first_contact_time"),
            "min_lateral_time_a": markers_a.get("min_lateral_time"),
            "min_lateral_time_b": markers_b.get("min_lateral_time"),
        },
    }
    save_json_safe(compare_summary, output_dir / "compare_summary.json")

    _save_compare_panel(
        df_a,
        df_b,
        output_dir,
        "compare_task_error",
        (
            ("peg_to_hole_dist", "distance"),
            ("peg_to_hole_lateral_error", "lateral"),
            ("abs_peg_to_hole_axial_error", "|axial|"),
        ),
        label_a,
        label_b,
        "task error (m)",
        formats,
        dpi,
        smooth_window,
    )
    _save_compare_panel(
        df_a,
        df_b,
        output_dir,
        "compare_force",
        (("force_norm", "force norm"),),
        label_a,
        label_b,
        "force (N)",
        formats,
        dpi,
        smooth_window,
    )
    action_column = _choose_action_compare_column(df_a) or _choose_action_compare_column(df_b)
    if action_column is not None:
        _save_compare_panel(
            df_a,
            df_b,
            output_dir,
            "compare_action_adjustment",
            ((action_column, action_column),),
            label_a,
            label_b,
            "update norm (rad)",
            formats,
            dpi,
            smooth_window,
        )
    else:
        _warn("no action adjustment column found in either rollout; skipping compare_action_adjustment")
    _save_compare_panel(
        df_a,
        df_b,
        output_dir,
        "compare_predicted_force",
        (
            ("pred_force_norm_mean", "pred mean"),
            ("pred_force_norm_max", "pred max"),
        ),
        label_a,
        label_b,
        "predicted force norm (N)",
        formats,
        dpi,
        smooth_window,
    )

    fig, axes = plt.subplots(4, 1, figsize=(10, 11), sharex=False)
    for column, suffix in (("peg_to_hole_dist", "distance"), ("peg_to_hole_lateral_error", "lateral")):
        _overlay_series(axes[0], df_a, df_b, column, label_a, label_b, smooth_window, suffix)
    _overlay_series(axes[1], df_a, df_b, "force_norm", label_a, label_b, smooth_window, "force")
    if action_column is not None:
        _overlay_series(axes[2], df_a, df_b, action_column, label_a, label_b, smooth_window, "action")
    for column, suffix in (("pred_force_norm_mean", "pred mean"), ("pred_force_norm_max", "pred max")):
        _overlay_series(axes[3], df_a, df_b, column, label_a, label_b, smooth_window, suffix)
    for ax, ylabel in zip(
        axes,
        ("task error (m)", "force (N)", "update norm (rad)", "pred force (N)"),
    ):
        ax.set_ylabel(ylabel)
        ax.set_xlabel(choose_x_axis(df_a)[1])
    fig.suptitle(f"{label_a} vs {label_b}", y=0.995)
    fig.tight_layout()
    save_figure(fig, output_dir, "compare_combined_analysis", formats, dpi)
    if show:
        plt.show()
    plt.close(fig)
    return compare_summary


def _default_single_output_dir(
    rollout_dir: Optional[Path],
    log_path: Path,
    output_dir: Optional[Path],
) -> Path:
    if output_dir is not None:
        return output_dir
    if rollout_dir is not None:
        return rollout_dir / "analysis_plots"
    return log_path.parent / "analysis_plots"


def _default_compare_output_dir(
    dir_a: Path,
    dir_b: Path,
    label_a: str,
    label_b: str,
    output_dir: Optional[Path],
) -> Path:
    if output_dir is not None:
        return output_dir
    common_parent = Path(os.path.commonpath([dir_a.resolve(), dir_b.resolve()]))
    if common_parent == dir_a.resolve() or common_parent == dir_b.resolve():
        common_parent = common_parent.parent
    return common_parent / f"comparison_analysis_{_sanitize_label(label_a)}_vs_{_sanitize_label(label_b)}"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot MuJoCo rollout sensor-analysis curves.")
    parser.add_argument("--rollout-dir", type=Path)
    parser.add_argument("--rollout-log-csv", type=Path)
    parser.add_argument("--summary-json", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--contact-force-threshold", type=float, default=5.0)
    parser.add_argument("--high-force-threshold", type=float, default=20.0)
    parser.add_argument("--very-high-force-threshold", type=float, default=40.0)
    parser.add_argument("--success-distance-threshold", type=float, default=0.005)
    parser.add_argument("--success-lateral-threshold", type=float, default=0.006)
    parser.add_argument("--success-force-threshold", type=float, default=40.0)
    parser.add_argument("--success-hold-steps", type=int, default=15)
    parser.add_argument("--smooth-window", type=int, default=1)
    parser.add_argument("--formats", default="png")
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--no-combined", action="store_true")
    parser.add_argument("--compare-rollout-dir-a", type=Path)
    parser.add_argument("--compare-rollout-dir-b", type=Path)
    parser.add_argument("--label-a")
    parser.add_argument("--label-b")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.smooth_window <= 0:
        raise ValueError("--smooth-window must be positive")
    if args.dpi <= 0:
        raise ValueError("--dpi must be positive")
    if args.success_hold_steps <= 0:
        raise ValueError("--success-hold-steps must be positive")
    formats = _parse_formats(args.formats)
    thresholds = Thresholds(
        contact_force=args.contact_force_threshold,
        high_force=args.high_force_threshold,
        very_high_force=args.very_high_force_threshold,
        success_distance=args.success_distance_threshold,
        success_lateral=args.success_lateral_threshold,
        success_force=args.success_force_threshold,
        success_hold_steps=args.success_hold_steps,
    )

    if args.compare_rollout_dir_a is not None or args.compare_rollout_dir_b is not None:
        if args.compare_rollout_dir_a is None or args.compare_rollout_dir_b is None:
            raise ValueError("provide both --compare-rollout-dir-a and --compare-rollout-dir-b")
        label_a = args.label_a or args.compare_rollout_dir_a.name
        label_b = args.label_b or args.compare_rollout_dir_b.name
        log_a, summary_path_a, _ = _resolve_rollout_inputs(args.compare_rollout_dir_a)
        log_b, summary_path_b, _ = _resolve_rollout_inputs(args.compare_rollout_dir_b)
        output_dir = _default_compare_output_dir(
            args.compare_rollout_dir_a,
            args.compare_rollout_dir_b,
            label_a,
            label_b,
            args.output_dir,
        )
        summary_a = load_summary_json(summary_path_a)
        summary_b = load_summary_json(summary_path_b)
        df_a = load_rollout_log(log_a)
        df_b = load_rollout_log(log_b)
        plot_compare_rollouts(
            df_a,
            summary_a,
            label_a,
            df_b,
            summary_b,
            label_b,
            output_dir,
            thresholds,
            formats,
            args.dpi,
            args.smooth_window,
            args.show,
        )
        print(f"output_dir={output_dir}")
        return 0

    log_path, summary_path, _ = _resolve_rollout_inputs(args.rollout_dir, args.rollout_log_csv, args.summary_json)
    output_dir = _default_single_output_dir(args.rollout_dir, log_path, args.output_dir)
    summary = load_summary_json(summary_path)
    df = load_rollout_log(log_path)
    plot_single_rollout(
        df,
        summary,
        output_dir,
        thresholds,
        formats,
        args.dpi,
        args.smooth_window,
        args.show,
        combined=not args.no_combined,
    )
    print(f"output_dir={output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
