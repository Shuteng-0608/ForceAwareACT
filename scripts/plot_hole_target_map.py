#!/usr/bin/env python3
"""Plot measured hole-position rollout outcomes as a target-style map."""

from __future__ import annotations

import argparse
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "force_aware_act_matplotlib"))

import numpy as np
import pandas as pd


SUPPORTED_FORMATS = {"png", "pdf", "svg"}


@dataclass(frozen=True)
class TargetMapData:
    point_index: pd.Series
    x_mm: pd.Series
    z_mm: pd.Series
    success: pd.Series

    @property
    def total_points(self) -> int:
        return int(len(self.success))

    @property
    def successful_points(self) -> int:
        return int(self.success.sum())

    @property
    def failed_points(self) -> int:
        return self.total_points - self.successful_points

    @property
    def success_rate(self) -> float:
        return self.successful_points / self.total_points if self.total_points else 0.0

    @property
    def maximum_radius_mm(self) -> float:
        radius = np.hypot(self.x_mm.to_numpy(dtype=float), self.z_mm.to_numpy(dtype=float))
        return float(radius.max())


def _load_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Rectangle

    return plt, Circle, Rectangle


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot measured hole-position rollout outcomes as a target-style spatial map."
    )
    parser.add_argument("--grid-summary-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-stem", default="hole_target_map")
    parser.add_argument("--title", default="")
    parser.add_argument("--ring-step-mm", type=float, default=2.0)
    parser.add_argument("--max-radius-mm", type=float, default=None)
    parser.add_argument("--marker-size", type=float, default=56.0)
    parser.add_argument("--show-point-index", action="store_true")
    parser.add_argument("--show-sampling-boundary", action="store_true")
    parser.add_argument("--formats", nargs="+", default=["png", "pdf"])
    parser.add_argument("--dpi", type=int, default=300)
    return parser


def normalize_formats(formats: Sequence[str]) -> list[str]:
    normalized = [item.strip().lstrip(".").lower() for item in formats if item.strip()]
    if not normalized:
        raise ValueError("--formats must include at least one format")
    unsupported = sorted(set(normalized) - SUPPORTED_FORMATS)
    if unsupported:
        raise ValueError(
            "unsupported output format(s): "
            + ", ".join(unsupported)
            + f"; supported formats: {', '.join(sorted(SUPPORTED_FORMATS))}"
        )
    return normalized


def parse_success_series(series: pd.Series) -> pd.Series:
    true_values = {"1", "true", "yes", "y", "success", "succeeded"}
    false_values = {"0", "false", "no", "n", "failure", "failed"}
    parsed: list[bool] = []
    unknown: list[str] = []
    missing = 0

    for value in series:
        if pd.isna(value):
            missing += 1
            continue
        if isinstance(value, (bool, np.bool_)):
            parsed.append(bool(value))
            continue
        if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
            numeric = float(value)
            if not math.isfinite(numeric):
                unknown.append(str(value))
            elif numeric == 1.0:
                parsed.append(True)
            elif numeric == 0.0:
                parsed.append(False)
            else:
                unknown.append(str(value))
            continue
        text = str(value).strip().lower()
        if not text:
            missing += 1
        elif text in true_values:
            parsed.append(True)
        elif text in false_values:
            parsed.append(False)
        else:
            unknown.append(str(value))

    if missing:
        raise ValueError(f"success column contains {missing} missing value(s)")
    if unknown:
        examples = ", ".join(repr(item) for item in unknown[:5])
        raise ValueError(f"success column contains unknown value(s): {examples}")
    return pd.Series(parsed, index=series.index, dtype=bool)


def resolve_columns(frame: pd.DataFrame) -> dict[str, str]:
    aliases = {
        "point_index": ("point_index", "point", "index"),
        "x": ("hole_offset_x", "x_offset", "offset_x"),
        "z": ("hole_offset_z", "z_offset", "offset_z"),
        "success": ("success", "task_success", "success_bool"),
    }
    resolved: dict[str, str] = {}
    columns = set(frame.columns)
    for key, candidates in aliases.items():
        matches = [column for column in candidates if column in columns]
        if not matches:
            if key == "x":
                raise ValueError("grid summary CSV missing required x-offset column: hole_offset_x")
            if key == "z":
                raise ValueError("grid summary CSV missing required z-offset column: hole_offset_z")
            if key == "success":
                raise ValueError("grid summary CSV missing required success column: success")
            raise ValueError("grid summary CSV missing required point-index column: point_index")
        resolved[key] = matches[0]
    return resolved


def _numeric_column(frame: pd.DataFrame, column: str, label: str) -> pd.Series:
    source = frame[column]
    numeric = pd.to_numeric(source, errors="coerce")
    non_numeric = numeric.isna() & source.notna()
    if bool(non_numeric.any()):
        examples = ", ".join(repr(value) for value in source[non_numeric].head(5).tolist())
        raise ValueError(f"{label} column contains non-numeric value(s): {examples}")
    if bool(numeric.isna().any()):
        raise ValueError(f"{label} column contains missing value(s)")
    finite = np.isfinite(numeric.to_numpy(dtype=float))
    if not bool(finite.all()):
        raise ValueError(f"{label} column contains non-finite value(s)")
    return numeric.astype(float)


def load_target_data(csv_path: Path) -> TargetMapData:
    if not csv_path.is_file():
        raise FileNotFoundError(f"grid summary CSV does not exist: {csv_path}")
    try:
        frame = pd.read_csv(csv_path)
    except pd.errors.EmptyDataError as error:
        raise ValueError(f"grid summary CSV is empty: {csv_path}") from error
    if frame.empty:
        raise ValueError(f"grid summary CSV is empty: {csv_path}")

    columns = resolve_columns(frame)
    point_index = frame[columns["point_index"]]
    if bool(point_index.isna().any()):
        raise ValueError("point_index column contains missing value(s)")
    duplicates = point_index[point_index.duplicated(keep=False)]
    if not duplicates.empty:
        examples = ", ".join(repr(value) for value in duplicates.drop_duplicates().head(5).tolist())
        raise ValueError(f"point_index column contains duplicate value(s): {examples}")

    x_m = _numeric_column(frame, columns["x"], "hole_offset_x")
    z_m = _numeric_column(frame, columns["z"], "hole_offset_z")
    success = parse_success_series(frame[columns["success"]])
    if len(success) == 0:
        raise ValueError("grid summary CSV contains no valid rows")

    return TargetMapData(
        point_index=point_index.reset_index(drop=True),
        x_mm=(x_m * 1000.0).reset_index(drop=True),
        z_mm=(z_m * 1000.0).reset_index(drop=True),
        success=success.reset_index(drop=True),
    )


def compute_symmetric_plot_limit(
    x_mm: Sequence[float],
    z_mm: Sequence[float],
    ring_step_mm: float,
    requested_limit_mm: Optional[float],
) -> float:
    if ring_step_mm <= 0.0 or not math.isfinite(ring_step_mm):
        raise ValueError("--ring-step-mm must be positive and finite")
    x_values = np.asarray(x_mm, dtype=float)
    z_values = np.asarray(z_mm, dtype=float)
    if x_values.size == 0 or z_values.size == 0:
        raise ValueError("at least one point is required to compute plot limits")
    if not (np.isfinite(x_values).all() and np.isfinite(z_values).all()):
        raise ValueError("plot limit inputs must be finite")

    max_abs_x = float(np.max(np.abs(x_values)))
    max_abs_z = float(np.max(np.abs(z_values)))
    max_radius = float(np.max(np.hypot(x_values, z_values)))
    required = max(max_abs_x, max_abs_z, max_radius)

    if requested_limit_mm is not None:
        if requested_limit_mm <= 0.0 or not math.isfinite(requested_limit_mm):
            raise ValueError("--max-radius-mm must be positive and finite")
        if requested_limit_mm + 1.0e-9 < required:
            raise ValueError(
                f"--max-radius-mm={requested_limit_mm:g} would clip input points; "
                f"need at least {required:g} mm"
            )
        return float(requested_limit_mm)

    clean_limit = math.ceil(required / ring_step_mm) * ring_step_mm if required > 0.0 else ring_step_mm
    return float(max(clean_limit, ring_step_mm))


def _ring_radii(plot_limit_mm: float, ring_step_mm: float) -> list[float]:
    count = int(math.floor(plot_limit_mm / ring_step_mm))
    return [ring_step_mm * index for index in range(1, count + 1)]


def create_target_figure(
    data: TargetMapData,
    *,
    title: str,
    ring_step_mm: float,
    plot_limit_mm: float,
    marker_size: float,
    show_point_index: bool,
    show_sampling_boundary: bool,
):
    if marker_size <= 0.0 or not math.isfinite(marker_size):
        raise ValueError("--marker-size must be positive and finite")
    plt, Circle, Rectangle = _load_matplotlib()
    fig, ax = plt.subplots(figsize=(7.0, 6.6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.axhline(0.0, color="0.78", linewidth=0.8, zorder=1)
    ax.axvline(0.0, color="0.78", linewidth=0.8, zorder=1)

    for radius in _ring_radii(plot_limit_mm, ring_step_mm):
        ax.add_patch(Circle((0.0, 0.0), radius, fill=False, edgecolor="0.82", linewidth=0.75, zorder=0))
        label_angle = math.radians(16.0)
        ax.text(
            radius * math.cos(label_angle),
            radius * math.sin(label_angle),
            f"{radius:g} mm",
            fontsize=7.5,
            color="0.45",
            ha="left",
            va="bottom",
            zorder=2,
        )

    if show_sampling_boundary:
        x_min = float(data.x_mm.min())
        x_max = float(data.x_mm.max())
        z_min = float(data.z_mm.min())
        z_max = float(data.z_mm.max())
        ax.add_patch(
            Rectangle(
                (x_min, z_min),
                x_max - x_min,
                z_max - z_min,
                fill=False,
                linestyle="--",
                linewidth=0.9,
                edgecolor="0.35",
                zorder=2,
            )
        )

    failures = ~data.success
    successes = data.success
    if bool(failures.any()):
        ax.scatter(
            data.x_mm[failures],
            data.z_mm[failures],
            s=marker_size,
            marker="o",
            facecolor="#d62728",
            edgecolor="black",
            linewidth=0.55,
            label=f"Failure ({data.failed_points})",
            zorder=4,
        )
    if bool(successes.any()):
        ax.scatter(
            data.x_mm[successes],
            data.z_mm[successes],
            s=marker_size,
            marker="o",
            facecolor="#2ca02c",
            edgecolor="black",
            linewidth=0.55,
            label=f"Success ({data.successful_points})",
            zorder=5,
        )

    ax.scatter([0.0], [0.0], marker="*", s=90.0, color="black", label="Nominal hole", zorder=6)

    if show_point_index:
        for point_index, x_value, z_value in zip(data.point_index, data.x_mm, data.z_mm):
            ax.annotate(
                str(point_index),
                (float(x_value), float(z_value)),
                xytext=(3.0, 3.0),
                textcoords="offset points",
                fontsize=6.5,
                color="0.25",
                zorder=7,
            )

    ax.set_xlim(-plot_limit_mm, plot_limit_mm)
    ax.set_ylim(-plot_limit_mm, plot_limit_mm)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Hole offset x (mm)")
    ax.set_ylabel("Hole offset z (mm)")
    ax.grid(False)
    ax.legend(loc="upper right", frameon=True, framealpha=0.95, fontsize=9)

    stats_title = f"{data.successful_points}/{data.total_points} successful — {data.success_rate * 100.0:.1f}%"
    ax.set_title(f"{title}\n{stats_title}" if title else stats_title)
    fig.tight_layout()
    return fig


def save_figure(
    figure,
    output_dir: Path,
    output_stem: str,
    formats: Sequence[str],
    dpi: int,
) -> list[Path]:
    normalized = normalize_formats(formats)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_files: list[Path] = []
    for fmt in normalized:
        output_path = output_dir / f"{output_stem}.{fmt}"
        if output_path.exists():
            print(f"replacing_existing_output={output_path}")
        figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
        output_files.append(output_path)
    return output_files


def _print_report(input_csv: Path, data: TargetMapData, plot_limit_mm: float, output_files: Sequence[Path]) -> None:
    print(f"input_csv={input_csv}")
    print(f"total_points={data.total_points}")
    print(f"successful_points={data.successful_points}")
    print(f"failed_points={data.failed_points}")
    print(f"success_rate={data.success_rate:.6g}")
    print(f"x_min_mm={float(data.x_mm.min()):.6g}")
    print(f"x_max_mm={float(data.x_mm.max()):.6g}")
    print(f"z_min_mm={float(data.z_mm.min()):.6g}")
    print(f"z_max_mm={float(data.z_mm.max()):.6g}")
    print(f"maximum_radius_mm={data.maximum_radius_mm:.6g}")
    print(f"plot_limit_mm={plot_limit_mm:.6g}")
    print("output_files=" + ",".join(str(path) for path in output_files))


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    formats = normalize_formats(args.formats)
    data = load_target_data(args.grid_summary_csv)
    plot_limit_mm = compute_symmetric_plot_limit(
        data.x_mm,
        data.z_mm,
        args.ring_step_mm,
        args.max_radius_mm,
    )
    fig = create_target_figure(
        data,
        title=args.title,
        ring_step_mm=args.ring_step_mm,
        plot_limit_mm=plot_limit_mm,
        marker_size=args.marker_size,
        show_point_index=args.show_point_index,
        show_sampling_boundary=args.show_sampling_boundary,
    )
    try:
        output_files = save_figure(fig, args.output_dir, args.output_stem, formats, args.dpi)
    finally:
        plt, _, _ = _load_matplotlib()
        plt.close(fig)
    _print_report(args.grid_summary_csv, data, plot_limit_mm, output_files)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
