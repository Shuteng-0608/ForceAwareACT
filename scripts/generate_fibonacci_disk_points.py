#!/usr/bin/env python3
"""Generate a deterministic, area-uniform Fibonacci point set on an x/z disk."""

from __future__ import annotations

import argparse
import csv
import math
import os
import tempfile
from pathlib import Path
from typing import Sequence


DEFAULT_OUTPUT = Path("configs/experiments/fibonacci_disk_100_r4mm.csv")


def fibonacci_disk_points(
    num_points: int,
    radius_mm: float,
    *,
    y_offset_mm: float = 0.0,
    rotation_deg: float = 0.0,
) -> list[dict[str, float | int]]:
    """Return deterministic cell-centred Fibonacci points in metres.

    Squared radii are placed at equal-area cell centres, while successive
    azimuths differ by the golden angle.  The outermost point therefore lies
    just inside the requested radius instead of overweighting the boundary.
    """
    if num_points <= 0:
        raise ValueError("num_points must be positive")
    if radius_mm <= 0.0 or not math.isfinite(radius_mm):
        raise ValueError("radius_mm must be positive and finite")
    if not math.isfinite(y_offset_mm):
        raise ValueError("y_offset_mm must be finite")
    if not math.isfinite(rotation_deg):
        raise ValueError("rotation_deg must be finite")

    radius_m = radius_mm / 1000.0
    y_offset_m = y_offset_mm / 1000.0
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    rotation = math.radians(rotation_deg)
    points: list[dict[str, float | int]] = []
    for zero_based_index in range(num_points):
        radius = radius_m * math.sqrt((zero_based_index + 0.5) / num_points)
        angle = rotation + zero_based_index * golden_angle
        points.append(
            {
                "point_index": zero_based_index + 1,
                "hole_offset_x": radius * math.cos(angle),
                "hole_offset_y": y_offset_m,
                "hole_offset_z": radius * math.sin(angle),
                "radius_mm": radius * 1000.0,
                "angle_deg": math.degrees(angle) % 360.0,
            }
        )
    return points


def write_points_csv(path: Path, points: list[dict[str, float | int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "point_index",
        "hole_offset_x",
        "hole_offset_y",
        "hole_offset_z",
        "radius_mm",
        "angle_deg",
    ]
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(points)


def plot_points(
    path: Path,
    points: list[dict[str, float | int]],
    radius_limit_mm: float,
    *,
    show_point_index: bool = True,
    dpi: int = 300,
) -> None:
    """Plot the generated x/z point layout in millimetres."""
    if dpi <= 0:
        raise ValueError("dpi must be positive")
    os.environ.setdefault(
        "MPLCONFIGDIR",
        str(Path(tempfile.gettempdir()) / "force_aware_act_matplotlib"),
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    x_mm = [float(point["hole_offset_x"]) * 1000.0 for point in points]
    z_mm = [float(point["hole_offset_z"]) * 1000.0 for point in points]
    radii_mm = [float(point["radius_mm"]) for point in points]

    fig, ax = plt.subplots(figsize=(8.0, 8.0), constrained_layout=True)
    ax.add_patch(
        Circle(
            (0.0, 0.0),
            radius_limit_mm,
            facecolor="#f7fafc",
            edgecolor="#1f2937",
            linewidth=1.6,
            zorder=0,
        )
    )
    for ring_radius in range(1, int(math.floor(radius_limit_mm)) + 1):
        ax.add_patch(
            Circle(
                (0.0, 0.0),
                ring_radius,
                fill=False,
                edgecolor="#cbd5e1",
                linewidth=0.8,
                linestyle="--",
                zorder=1,
            )
        )
    scatter = ax.scatter(
        x_mm,
        z_mm,
        c=radii_mm,
        cmap="viridis",
        vmin=0.0,
        vmax=radius_limit_mm,
        s=35,
        edgecolors="white",
        linewidths=0.45,
        zorder=3,
    )
    if show_point_index:
        for point, x_value, z_value in zip(points, x_mm, z_mm):
            ax.annotate(
                str(point["point_index"]),
                (x_value, z_value),
                xytext=(3, 3),
                textcoords="offset points",
                fontsize=5.5,
                color="#111827",
                zorder=4,
            )

    colorbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Radial offset (mm)")
    margin = max(0.35, 0.08 * radius_limit_mm)
    limit = radius_limit_mm + margin
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_aspect("equal", adjustable="box")
    ax.axhline(0.0, color="#94a3b8", linewidth=0.7, zorder=2)
    ax.axvline(0.0, color="#94a3b8", linewidth=0.7, zorder=2)
    ax.set_xlabel("x offset (mm)")
    ax.set_ylabel("z offset (mm)")
    ax.set_title(
        f"Fixed Fibonacci disk point set: N={len(points)}, R={radius_limit_mm:g} mm"
    )
    ax.grid(False)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate fixed Fibonacci disk points. hole_offset_x/y/z are "
            "written in metres for direct rollout consumption."
        )
    )
    parser.add_argument("--num-points", type=int, default=100)
    parser.add_argument("--radius-mm", type=float, default=4.0)
    parser.add_argument("--y-offset-mm", type=float, default=0.0)
    parser.add_argument("--rotation-deg", type=float, default=0.0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--plot-output", type=Path, default=None)
    parser.add_argument(
        "--show-point-index",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show point indices when --plot-output is provided. Default: true.",
    )
    parser.add_argument("--plot-dpi", type=int, default=300)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    points = fibonacci_disk_points(
        args.num_points,
        args.radius_mm,
        y_offset_mm=args.y_offset_mm,
        rotation_deg=args.rotation_deg,
    )
    write_points_csv(args.output, points)
    if args.plot_output is not None:
        plot_points(
            args.plot_output,
            points,
            args.radius_mm,
            show_point_index=args.show_point_index,
            dpi=args.plot_dpi,
        )
    maximum_radius_mm = max(float(point["radius_mm"]) for point in points)
    print(f"task_points_csv={args.output}")
    print(f"num_points={len(points)}")
    print(f"radius_limit_mm={args.radius_mm:.12g}")
    print(f"maximum_point_radius_mm={maximum_radius_mm:.12g}")
    if args.plot_output is not None:
        print(f"point_layout_plot={args.plot_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
