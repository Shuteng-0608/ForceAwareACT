#!/usr/bin/env python3
"""Generate a reproducible area-uniform random point set on an x/z disk."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Sequence

import numpy as np

try:
    from scripts.generate_fibonacci_disk_points import plot_points
except ModuleNotFoundError:  # Direct execution via python scripts/...
    from generate_fibonacci_disk_points import plot_points


DEFAULT_SEED = 20260714
DEFAULT_OUTPUT = Path(
    "configs/experiments/random_disk_100_r60mm_seed20260714.csv"
)


def random_disk_points(
    num_points: int,
    radius_mm: float,
    *,
    seed: int,
    y_offset_mm: float = 0.0,
) -> list[dict[str, float | int]]:
    """Return seeded random points uniformly distributed over disk area."""
    if num_points <= 0:
        raise ValueError("num_points must be positive")
    if radius_mm <= 0.0 or not math.isfinite(radius_mm):
        raise ValueError("radius_mm must be positive and finite")
    if not math.isfinite(y_offset_mm):
        raise ValueError("y_offset_mm must be finite")

    rng = np.random.default_rng(seed)
    radii_mm = radius_mm * np.sqrt(rng.random(num_points))
    angles = 2.0 * math.pi * rng.random(num_points)
    y_offset_m = y_offset_mm / 1000.0
    points: list[dict[str, float | int]] = []
    for index, (radius, angle) in enumerate(zip(radii_mm, angles), start=1):
        points.append(
            {
                "point_index": index,
                "hole_offset_x": float(radius * math.cos(angle) / 1000.0),
                "hole_offset_y": y_offset_m,
                "hole_offset_z": float(radius * math.sin(angle) / 1000.0),
                "radius_mm": float(radius),
                "angle_deg": float(math.degrees(angle) % 360.0),
                "sampling_seed": seed,
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
        "sampling_seed",
    ]
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(points)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate seeded area-uniform random disk points. hole_offset_x/y/z "
            "are written in metres for direct rollout consumption."
        )
    )
    parser.add_argument("--num-points", type=int, default=100)
    parser.add_argument("--radius-mm", type=float, default=60.0)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--y-offset-mm", type=float, default=0.0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--plot-output", type=Path, default=None)
    parser.add_argument(
        "--show-point-index",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--plot-dpi", type=int, default=300)
    parser.add_argument("--plot-ring-step-mm", type=float, default=10.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    points = random_disk_points(
        args.num_points,
        args.radius_mm,
        seed=args.seed,
        y_offset_mm=args.y_offset_mm,
    )
    write_points_csv(args.output, points)
    if args.plot_output is not None:
        plot_points(
            args.plot_output,
            points,
            args.radius_mm,
            show_point_index=args.show_point_index,
            dpi=args.plot_dpi,
            ring_step_mm=args.plot_ring_step_mm,
            title=(
                f"Seeded random disk point set: N={len(points)}, "
                f"R={args.radius_mm:g} mm, seed={args.seed}"
            ),
        )
    maximum_radius_mm = max(float(point["radius_mm"]) for point in points)
    print(f"task_points_csv={args.output}")
    print(f"num_points={len(points)}")
    print(f"radius_limit_mm={args.radius_mm:.12g}")
    print(f"sampling_seed={args.seed}")
    print(f"maximum_point_radius_mm={maximum_radius_mm:.12g}")
    if args.plot_output is not None:
        print(f"point_layout_plot={args.plot_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
