#!/usr/bin/env python3
"""Plot heatmaps from hole-position grid rollout summaries."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Optional, Sequence

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "force_aware_act_matplotlib"))

import numpy as np
import pandas as pd


def _load_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _parse_formats(value: str) -> list[str]:
    formats = [item.strip().lstrip(".") for item in value.split(",") if item.strip()]
    if not formats:
        raise ValueError("--formats must include at least one format")
    return formats


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def load_grid_summary(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"grid summary CSV does not exist: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"grid summary CSV is empty: {path}")
    for column in (
        "hole_offset_x",
        "hole_offset_z",
        "hole_offset_y",
        "radial_offset",
        "success_time",
        "final_dist",
        "final_lateral",
        "final_axial",
        "max_force",
        "mean_force",
        "force_gt_20_steps",
        "force_gt_40_steps",
    ):
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    if "success" in df.columns:
        df["success_bool"] = df["success"].map(_to_bool)
    else:
        df["success_bool"] = False
    if "safe_success" in df.columns:
        df["safe_success_bool"] = df["safe_success"].map(_to_bool)
    else:
        df["safe_success_bool"] = df["success_bool"]
    if "radial_offset" not in df.columns:
        df["radial_offset"] = np.sqrt(df["hole_offset_x"] ** 2 + df["hole_offset_z"] ** 2)
    return df


def aggregate_grid_results(df: pd.DataFrame) -> pd.DataFrame:
    required = {"hole_offset_x", "hole_offset_z"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"grid summary missing required columns: {', '.join(sorted(missing))}")

    rows = []
    for (x_offset, z_offset), group in df.groupby(["hole_offset_x", "hole_offset_z"], dropna=False):
        successes = group[group["success_bool"]]
        rows.append(
            {
                "hole_offset_x": float(x_offset),
                "hole_offset_z": float(z_offset),
                "runs": int(len(group)),
                "successes": int(group["success_bool"].sum()),
                "success_rate": float(group["success_bool"].mean()),
                "mean_success_time": float(successes["success_time"].mean()) if not successes.empty else float("nan"),
                "mean_final_dist": float(group["final_dist"].mean()),
                "mean_final_lateral": float(group["final_lateral"].mean()),
                "mean_max_force": float(group["max_force"].mean()),
                "max_observed_force": float(group["max_force"].max()),
                "mean_force_gt_20_steps": float(group["force_gt_20_steps"].mean()),
                "mean_force_gt_40_steps": float(group["force_gt_40_steps"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["hole_offset_z", "hole_offset_x"]).reset_index(drop=True)


def _matrix(table: pd.DataFrame, value_column: str) -> tuple[np.ndarray, list[float], list[float]]:
    x_values = sorted(table["hole_offset_x"].dropna().unique())
    z_values = sorted(table["hole_offset_z"].dropna().unique())
    matrix = np.full((len(z_values), len(x_values)), np.nan, dtype=np.float64)
    for _, row in table.iterrows():
        x_index = x_values.index(row["hole_offset_x"])
        z_index = z_values.index(row["hole_offset_z"])
        matrix[z_index, x_index] = row[value_column]
    return matrix, x_values, z_values


def _save_figure(fig, output_dir: Path, stem: str, formats: Sequence[str], dpi: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        path = output_dir / f"{stem}.{fmt}"
        fig.savefig(path, dpi=dpi)
        print(f"saved_plot={path}")


def _plot_heatmap(
    table: pd.DataFrame,
    value_column: str,
    title: str,
    colorbar_label: str,
    output_dir: Path,
    stem: str,
    formats: Sequence[str],
    dpi: int,
    annotate: bool,
    value_scale: float = 1.0,
) -> None:
    plt = _load_matplotlib()
    matrix, x_values, z_values = _matrix(table, value_column)
    matrix = matrix * value_scale
    fig, ax = plt.subplots(figsize=(6.8, 5.4))
    image = ax.imshow(matrix, origin="lower", aspect="auto")
    ax.set_xticks(np.arange(len(x_values)))
    ax.set_yticks(np.arange(len(z_values)))
    ax.set_xticklabels([f"{value * 1000:.0f}" for value in x_values])
    ax.set_yticklabels([f"{value * 1000:.0f}" for value in z_values])
    ax.set_xlabel("hole x offset (mm)")
    ax.set_ylabel("hole z offset (mm)")
    ax.set_title(title)
    fig.colorbar(image, ax=ax, label=colorbar_label)
    if annotate:
        for z_index in range(matrix.shape[0]):
            for x_index in range(matrix.shape[1]):
                value = matrix[z_index, x_index]
                text = "nan" if np.isnan(value) else f"{value:.3g}"
                ax.text(x_index, z_index, text, ha="center", va="center", color="white")
    fig.tight_layout()
    _save_figure(fig, output_dir, stem, formats, dpi)
    plt.close(fig)


def _scatter_base(df: pd.DataFrame, title: str):
    plt = _load_matplotlib()
    fig, ax = plt.subplots(figsize=(6.4, 5.8))
    ax.axhline(0.0, color="0.7", linewidth=1.0)
    ax.axvline(0.0, color="0.7", linewidth=1.0)
    ax.scatter([0.0], [0.0], marker="+", color="black", s=80, label="nominal")
    ax.set_xlabel("hole x offset (mm)")
    ax.set_ylabel("hole z offset (mm)")
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.25)
    return plt, fig, ax


def _plot_success_scatter(
    df: pd.DataFrame,
    output_dir: Path,
    stem: str,
    success_column: str,
    title: str,
    formats: Sequence[str],
    dpi: int,
) -> None:
    plt, fig, ax = _scatter_base(df, title)
    x_mm = df["hole_offset_x"] * 1000.0
    z_mm = df["hole_offset_z"] * 1000.0
    successes = df[success_column].astype(bool)
    ax.scatter(x_mm[~successes], z_mm[~successes], marker="x", color="tab:red", label="failure")
    ax.scatter(x_mm[successes], z_mm[successes], marker="o", color="tab:green", label="success")
    ax.legend(loc="best", fontsize="small")
    fig.tight_layout()
    _save_figure(fig, output_dir, stem, formats, dpi)
    plt.close(fig)


def _plot_metric_scatter(
    df: pd.DataFrame,
    output_dir: Path,
    stem: str,
    value_column: str,
    title: str,
    colorbar_label: str,
    formats: Sequence[str],
    dpi: int,
    success_only: bool = False,
    value_scale: float = 1.0,
) -> None:
    plot_df = df[df["success_bool"]] if success_only else df
    plot_df = plot_df.dropna(subset=[value_column, "hole_offset_x", "hole_offset_z"])
    if plot_df.empty:
        return
    plt, fig, ax = _scatter_base(plot_df, title)
    scatter = ax.scatter(
        plot_df["hole_offset_x"] * 1000.0,
        plot_df["hole_offset_z"] * 1000.0,
        c=plot_df[value_column] * value_scale,
        cmap="viridis",
        s=48,
    )
    fig.colorbar(scatter, ax=ax, label=colorbar_label)
    ax.legend(loc="best", fontsize="small")
    fig.tight_layout()
    _save_figure(fig, output_dir, stem, formats, dpi)
    plt.close(fig)


def _radial_bin_label(radial_offset: float) -> str:
    radial_mm = radial_offset * 1000.0
    bins = [
        (0.0, 0.5, "[0,0.5]"),
        (0.5, 1.0, "(0.5,1.0]"),
        (1.0, 1.5, "(1.0,1.5]"),
        (1.5, 2.0, "(1.5,2.0]"),
        (2.0, np.sqrt(8.0), "(2.0,sqrt8]"),
    ]
    for lower, upper, label in bins:
        if radial_mm <= upper and (radial_mm > lower or lower == 0.0):
            return label
    return f">{np.sqrt(8.0):.3g}"


def _plot_group_success_rate(
    df: pd.DataFrame,
    output_dir: Path,
    stem: str,
    labels: Sequence[str],
    rates: Sequence[float],
    title: str,
    formats: Sequence[str],
    dpi: int,
) -> None:
    plt = _load_matplotlib()
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.bar(np.arange(len(labels)), rates, color="tab:blue")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("success rate")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    _save_figure(fig, output_dir, stem, formats, dpi)
    plt.close(fig)


def write_scatter_outputs(
    df: pd.DataFrame,
    output_dir: Path,
    formats: Sequence[str],
    dpi: int,
) -> None:
    _plot_success_scatter(
        df,
        output_dir,
        "hole_position_success_scatter",
        "success_bool",
        "Hole position success",
        formats,
        dpi,
    )
    _plot_success_scatter(
        df,
        output_dir,
        "hole_position_safe_success_scatter",
        "safe_success_bool",
        "Hole position safe success",
        formats,
        dpi,
    )
    _plot_metric_scatter(
        df,
        output_dir,
        "hole_position_success_time_scatter",
        "success_time",
        "Success time by hole position",
        "success time (s)",
        formats,
        dpi,
        success_only=True,
    )
    _plot_metric_scatter(df, output_dir, "hole_position_max_force_scatter", "max_force", "Max force by hole position", "max force (N)", formats, dpi)
    _plot_metric_scatter(df, output_dir, "hole_position_final_distance_scatter", "final_dist", "Final distance by hole position", "final distance (mm)", formats, dpi, value_scale=1000.0)
    _plot_metric_scatter(df, output_dir, "hole_position_final_lateral_scatter", "final_lateral", "Final lateral error by hole position", "final lateral error (mm)", formats, dpi, value_scale=1000.0)

    radial_labels = ["[0,0.5]", "(0.5,1.0]", "(1.0,1.5]", "(1.5,2.0]", "(2.0,sqrt8]"]
    radial_groups = df.assign(radial_bin=df["radial_offset"].map(_radial_bin_label))
    radial_rates = [
        float(radial_groups.loc[radial_groups["radial_bin"] == label, "success_bool"].mean())
        if not radial_groups.loc[radial_groups["radial_bin"] == label].empty
        else 0.0
        for label in radial_labels
    ]
    _plot_group_success_rate(
        df,
        output_dir,
        "success_rate_by_radial_bin",
        radial_labels,
        radial_rates,
        "Success rate by radial offset bin",
        formats,
        dpi,
    )

    z_bins = [("-z", df["hole_offset_z"] < 0), ("z=0", df["hole_offset_z"] == 0), ("+z", df["hole_offset_z"] > 0)]
    z_labels = [label for label, _ in z_bins]
    z_rates = [
        float(df.loc[mask, "success_bool"].mean()) if not df.loc[mask].empty else 0.0
        for _, mask in z_bins
    ]
    _plot_group_success_rate(
        df,
        output_dir,
        "success_rate_by_z_bin",
        z_labels,
        z_rates,
        "Success rate by z sign bin",
        formats,
        dpi,
    )


def _cell_record(row: Optional[pd.Series]) -> Optional[dict[str, Any]]:
    if row is None:
        return None
    return {key: _json_safe(value) for key, value in row.to_dict().items()}


def write_outputs(
    summary_csv: Path,
    output_dir: Path,
    formats: Sequence[str],
    dpi: int,
    annotate: bool,
) -> pd.DataFrame:
    df = load_grid_summary(summary_csv)
    write_scatter_outputs(df, output_dir, formats, dpi)
    table = aggregate_grid_results(df)
    output_dir.mkdir(parents=True, exist_ok=True)
    table_path = output_dir / "hole_grid_results_table.csv"
    table.to_csv(table_path, index=False)
    print(f"grid_results_table_csv={table_path}")

    _plot_heatmap(
        table,
        "success_rate",
        "Hole offset success rate",
        "success rate",
        output_dir,
        "hole_offset_success_rate_heatmap",
        formats,
        dpi,
        annotate,
    )
    _plot_heatmap(
        table,
        "mean_success_time",
        "Mean success time",
        "time (s)",
        output_dir,
        "hole_offset_success_time_heatmap",
        formats,
        dpi,
        annotate,
    )
    _plot_heatmap(
        table,
        "mean_max_force",
        "Mean max force",
        "force (N)",
        output_dir,
        "hole_offset_max_force_heatmap",
        formats,
        dpi,
        annotate,
    )
    _plot_heatmap(
        table,
        "mean_final_lateral",
        "Mean final lateral error",
        "lateral error (mm)",
        output_dir,
        "hole_offset_final_lateral_heatmap",
        formats,
        dpi,
        annotate,
        value_scale=1000.0,
    )
    _plot_heatmap(
        table,
        "mean_final_dist",
        "Mean final distance",
        "distance (mm)",
        output_dir,
        "hole_offset_final_distance_heatmap",
        formats,
        dpi,
        annotate,
        value_scale=1000.0,
    )

    best = table.sort_values(["success_rate", "mean_final_dist"], ascending=[False, True]).iloc[0]
    worst = table.sort_values(["success_rate", "mean_final_dist"], ascending=[True, False]).iloc[0]
    summary = {
        "grid_cells": int(len(table)),
        "runs": int(table["runs"].sum()),
        "total_successes": int(table["successes"].sum()),
        "total_success_rate": float(table["successes"].sum() / table["runs"].sum()),
        "best_offset_cell": _cell_record(best),
        "worst_offset_cell": _cell_record(worst),
        "max_tested_abs_x_offset": float(table["hole_offset_x"].abs().max()),
        "max_tested_abs_z_offset": float(table["hole_offset_z"].abs().max()),
        "threshold_configuration": {
            "source": "grid_summary.csv",
        },
    }
    summary_path = output_dir / "hole_grid_summary.json"
    with summary_path.open("w") as summary_file:
        json.dump(_json_safe(summary), summary_file, indent=2, sort_keys=True)
        summary_file.write("\n")
    print(f"hole_grid_summary_json={summary_path}")
    return table


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot hole-position grid heatmaps.")
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--formats", default="png")
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--annotate", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    formats = _parse_formats(args.formats)
    write_outputs(args.summary_csv, args.output_dir, formats, args.dpi, args.annotate)
    if args.show:
        _load_matplotlib().show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
