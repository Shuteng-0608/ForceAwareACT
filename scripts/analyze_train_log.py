#!/usr/bin/env python3
"""Analyze a minimal ForceAwareACT CSV training log.

Example:
    PYTHONPATH=src .venv/bin/python scripts/analyze_train_log.py outputs/minimal_train/train_log.csv --plot outputs/minimal_train/loss_curve.png
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from statistics import mean
from typing import Optional, Sequence


REQUIRED_COLUMNS = (
    "step",
    "loss_total",
    "loss_action",
    "loss_force",
    "kl_motion",
    "kl_contact",
    "beta_motion",
    "beta_contact",
)
LOSS_COLUMNS = ("loss_total", "loss_action", "loss_force")
OPTIONAL_LOSS_COLUMNS = ("loss_prior",)
OPTIONAL_METADATA_COLUMNS = ("lambda_prior", "prior_loss_mode")


def _read_log(path: Path) -> tuple[list[dict[str, float | str]], list[str], list[str]]:
    with path.open("r", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None:
            raise ValueError("CSV file has no header")
        missing = [column for column in REQUIRED_COLUMNS if column not in reader.fieldnames]
        if missing:
            raise ValueError(f"CSV file is missing required columns: {', '.join(missing)}")

        optional_loss_columns = [
            column for column in OPTIONAL_LOSS_COLUMNS if column in reader.fieldnames
        ]
        optional_metadata_columns = [
            column for column in OPTIONAL_METADATA_COLUMNS if column in reader.fieldnames
        ]
        rows = []
        for row in reader:
            parsed = {column: float(row[column]) for column in REQUIRED_COLUMNS}
            parsed["step"] = int(parsed["step"])
            for column in optional_loss_columns:
                parsed[column] = float(row[column])
            for column in optional_metadata_columns:
                parsed[column] = row[column]
            rows.append(parsed)
    if not rows:
        raise ValueError("CSV file contains no logged steps")
    return rows, optional_loss_columns, optional_metadata_columns


def _window_mean(rows: list[dict[str, float | str]], column: str) -> float:
    return mean(float(row[column]) for row in rows)


def _percentage_reduction(first_mean: float, last_mean: float) -> float:
    if first_mean == 0:
        return float("nan")
    return 100.0 * (first_mean - last_mean) / abs(first_mean)


def _print_summary(
    rows: list[dict[str, float | str]],
    optional_loss_columns: Sequence[str],
    optional_metadata_columns: Sequence[str],
) -> None:
    first_rows = rows[:20]
    last_rows = rows[-20:]
    min_row = min(rows, key=lambda row: row["loss_total"])
    loss_columns = (*LOSS_COLUMNS, *optional_loss_columns)

    print(f"logged_steps={len(rows)}")
    print(f"prior_loss_found={'loss_prior' in optional_loss_columns}")
    if "lambda_prior" in optional_metadata_columns:
        lambda_values = sorted({str(row["lambda_prior"]) for row in rows})
        print(f"lambda_prior_values={lambda_values}")
    if "prior_loss_mode" in optional_metadata_columns:
        prior_modes = sorted({str(row["prior_loss_mode"]) for row in rows})
        print(f"prior_loss_mode_values={prior_modes}")
    print("\nLoss Summary")
    print("------------")
    for column in loss_columns:
        first_mean = _window_mean(first_rows, column)
        last_mean = _window_mean(last_rows, column)
        reduction = _percentage_reduction(first_mean, last_mean)
        min_column_row = min(rows, key=lambda row: row[column])
        print(
            f"{column}: "
            f"first20_mean={first_mean:.6g} "
            f"last20_mean={last_mean:.6g} "
            f"reduction_pct={reduction:.6g} "
            f"min={float(min_column_row[column]):.6g} "
            f"min_step={int(min_column_row['step'])}"
        )
    print(f"\nmin_loss_total={min_row['loss_total']:.6g} step={int(min_row['step'])}")


def _save_plot(
    rows: list[dict[str, float | str]],
    plot_path: Path,
    optional_loss_columns: Sequence[str],
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not available; skipping plot", file=sys.stderr)
        return

    steps = [row["step"] for row in rows]
    loss_columns = (*LOSS_COLUMNS, *optional_loss_columns)
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    for column in loss_columns:
        plt.plot(steps, [float(row[column]) for row in rows], label=column)
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    print(f"saved_plot={plot_path}")


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze a ForceAwareACT CSV training log.")
    parser.add_argument("log_csv", type=Path, help="Path to train_log.csv.")
    parser.add_argument("--plot", type=Path, default=None, help="Optional output path for a PNG plot.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    log_csv = args.log_csv.expanduser()
    plot_path = args.plot.expanduser() if args.plot is not None else None
    if not log_csv.is_file():
        print(f"error: CSV log does not exist: {log_csv}", file=sys.stderr)
        return 2

    try:
        rows, optional_loss_columns, optional_metadata_columns = _read_log(log_csv)
        _print_summary(rows, optional_loss_columns, optional_metadata_columns)
        if plot_path is not None:
            _save_plot(rows, plot_path, optional_loss_columns)
    except Exception as error:
        print(f"error: failed to analyze train log: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
