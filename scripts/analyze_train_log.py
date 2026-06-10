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


SUPPORTED_METRIC_COLUMNS = (
    "loss_total",
    "loss_action",
    "loss_force",
    "loss_prior",
    "prior_mu_mse",
    "prior_mu_l2",
    "prior_mu_cosine_similarity",
    "kl_motion",
    "kl_contact",
)
FULL_TRAINING_PLOT_COLUMNS = ("loss_total", "loss_action", "loss_force", "loss_prior")
PRIOR_ONLY_PLOT_COLUMNS = (
    "loss_prior",
    "prior_mu_mse",
    "prior_mu_l2",
    "prior_mu_cosine_similarity",
)
OPTIONAL_METADATA_COLUMNS = ("lambda_prior", "prior_loss_mode")
SUMMARY_LOSS_COLUMNS = ("loss_total", "loss_action", "loss_force", "loss_prior")
INDIVIDUAL_PLOT_COLUMNS = (
    "loss_total",
    "loss_action",
    "loss_force",
    "loss_prior",
    "kl_motion",
    "kl_contact",
    "prior_mu_mse",
    "prior_mu_l2",
    "prior_mu_cosine_similarity",
)


def _read_log(path: Path) -> tuple[list[dict[str, float | str]], list[str], list[str], list[str]]:
    with path.open("r", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None:
            raise ValueError("CSV file has no header")
        if "step" not in reader.fieldnames:
            raise ValueError("CSV file is missing required column: step")

        metric_columns = [
            column for column in SUPPORTED_METRIC_COLUMNS if column in reader.fieldnames
        ]
        if not metric_columns:
            raise ValueError(
                "CSV file has no supported metric columns. "
                f"Available columns: {', '.join(reader.fieldnames)}"
            )
        optional_metadata_columns = [
            column for column in OPTIONAL_METADATA_COLUMNS if column in reader.fieldnames
        ]
        rows = []
        for row in reader:
            parsed: dict[str, float | str] = {"step": int(float(row["step"]))}
            for column in metric_columns:
                parsed[column] = float(row[column])
            for column in optional_metadata_columns:
                parsed[column] = row[column]
            rows.append(parsed)
    if not rows:
        raise ValueError("CSV file contains no logged steps")
    return rows, metric_columns, optional_metadata_columns, list(reader.fieldnames)


def _window_mean(rows: list[dict[str, float | str]], column: str) -> float:
    return mean(float(row[column]) for row in rows)


def _percentage_reduction(first_mean: float, last_mean: float) -> float:
    if first_mean == 0:
        return float("nan")
    return 100.0 * (first_mean - last_mean) / abs(first_mean)


def _percentage_increase(first_mean: float, last_mean: float) -> float:
    if first_mean == 0:
        return float("nan")
    return 100.0 * (last_mean - first_mean) / abs(first_mean)


def _print_summary(
    rows: list[dict[str, float | str]],
    metric_columns: Sequence[str],
    optional_metadata_columns: Sequence[str],
) -> None:
    summary_window = min(100, len(rows))
    first_rows = rows[:summary_window]
    last_rows = rows[-summary_window:]

    print(f"logged_steps={len(rows)}")
    print(f"metric_columns={list(metric_columns)}")
    print(f"prior_loss_found={'loss_prior' in metric_columns}")
    if "lambda_prior" in optional_metadata_columns:
        lambda_values = sorted({str(row["lambda_prior"]) for row in rows})
        print(f"lambda_prior_values={lambda_values}")
    if "prior_loss_mode" in optional_metadata_columns:
        prior_modes = sorted({str(row["prior_loss_mode"]) for row in rows})
        print(f"prior_loss_mode_values={prior_modes}")
    print("\nLoss Summary")
    print("------------")
    for column in SUMMARY_LOSS_COLUMNS:
        if column not in metric_columns:
            continue
        first_mean = _window_mean(first_rows, column)
        last_mean = _window_mean(last_rows, column)
        reduction = _percentage_reduction(first_mean, last_mean)
        min_column_row = min(rows, key=lambda row: row[column])
        print(
            f"{column}: "
            f"first100_mean={first_mean:.6g} "
            f"last100_mean={last_mean:.6g} "
            f"reduction_pct={reduction:.6g} "
            f"min={float(min_column_row[column]):.6g} "
            f"min_step={int(min_column_row['step'])}"
        )

    supplemental_columns = [
        column for column in metric_columns if column not in SUMMARY_LOSS_COLUMNS
    ]
    if supplemental_columns:
        print("\nAdditional Metric Summary")
        print("-------------------------")
    for column in supplemental_columns:
        first_mean = _window_mean(first_rows, column)
        last_mean = _window_mean(last_rows, column)
        min_column_row = min(rows, key=lambda row: row[column])
        max_column_row = max(rows, key=lambda row: row[column])
        change_text = (
            f"increase_pct={_percentage_increase(first_mean, last_mean):.6g}"
            if column == "prior_mu_cosine_similarity"
            else f"reduction_pct={_percentage_reduction(first_mean, last_mean):.6g}"
        )
        print(
            f"{column}: "
            f"first100_mean={first_mean:.6g} "
            f"last100_mean={last_mean:.6g} "
            f"{change_text} "
            f"min={float(min_column_row[column]):.6g} "
            f"min_step={int(min_column_row['step'])} "
            f"max={float(max_column_row[column]):.6g} "
            f"max_step={int(max_column_row['step'])}"
        )


def _plot_columns(metric_columns: Sequence[str]) -> list[str]:
    if "loss_total" in metric_columns:
        return [column for column in FULL_TRAINING_PLOT_COLUMNS if column in metric_columns]
    return [column for column in PRIOR_ONLY_PLOT_COLUMNS if column in metric_columns]


def _moving_average(values: Sequence[float], window: int) -> list[float]:
    moving_average: list[float] = []
    running_sum = 0.0
    for index, value in enumerate(values):
        running_sum += value
        if index >= window:
            running_sum -= values[index - window]
        moving_average.append(running_sum / min(index + 1, window))
    return moving_average


def _load_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _save_combined_plot(
    rows: list[dict[str, float | str]],
    plot_path: Path,
    metric_columns: Sequence[str],
    window: int,
    start_index: int = 0,
    title: str = "Training Loss Moving Averages",
) -> None:
    plt = _load_matplotlib()
    steps = [int(row["step"]) for row in rows]
    plot_columns = _plot_columns(metric_columns)
    if not plot_columns:
        raise ValueError("no supported metric columns are available for plotting")
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    for column in plot_columns:
        values = [float(row[column]) for row in rows]
        moving_average = _moving_average(values, window)
        plt.plot(
            steps[start_index:],
            moving_average[start_index:],
            linewidth=2.0,
            label=column,
        )
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.title(title)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    print(f"saved_plot={plot_path}")


def _save_individual_plot(
    rows: list[dict[str, float | str]],
    plot_path: Path,
    column: str,
    window: int,
) -> None:
    plt = _load_matplotlib()
    steps = [int(row["step"]) for row in rows]
    values = [float(row[column]) for row in rows]
    moving_average = _moving_average(values, window)
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.plot(steps, values, alpha=0.2, linewidth=0.7, label=f"{column} raw")
    plt.plot(steps, moving_average, linewidth=2.0, label=f"{column} MA({window})")
    plt.xlabel("step")
    plt.ylabel(column)
    plt.title(column)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    print(f"saved_plot={plot_path}")


def _save_output_dir_plots(
    rows: list[dict[str, float | str]],
    output_dir: Path,
    metric_columns: Sequence[str],
    window: int,
    last_fraction: float,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for column in INDIVIDUAL_PLOT_COLUMNS:
        if column in metric_columns:
            _save_individual_plot(rows, output_dir / f"{column}.png", column, window)

    _save_combined_plot(
        rows,
        output_dir / "loss_moving_average_combined.png",
        metric_columns,
        window,
    )
    start_index = max(0, len(rows) - max(1, int(len(rows) * last_fraction)))
    _save_combined_plot(
        rows,
        output_dir / "loss_moving_average_last_fraction.png",
        metric_columns,
        window,
        start_index=start_index,
        title=f"Training Loss Moving Averages: Last {last_fraction:.0%}",
    )


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze a ForceAwareACT CSV training log.")
    parser.add_argument("log_csv", type=Path, help="Path to train_log.csv.")
    parser.add_argument("--plot", type=Path, default=None, help="Optional output path for a PNG plot.")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--window", type=int, default=100)
    parser.add_argument("--last-fraction", type=float, default=0.2)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    log_csv = args.log_csv.expanduser()
    plot_path = args.plot.expanduser() if args.plot is not None else None
    output_dir = args.output_dir.expanduser() if args.output_dir is not None else None
    if not log_csv.is_file():
        print(f"error: CSV log does not exist: {log_csv}", file=sys.stderr)
        return 2
    if args.window <= 0:
        print("error: --window must be positive", file=sys.stderr)
        return 2
    if not 0.0 < args.last_fraction <= 1.0:
        print("error: --last-fraction must be in (0, 1]", file=sys.stderr)
        return 2

    try:
        rows, metric_columns, optional_metadata_columns, _ = _read_log(log_csv)
        _print_summary(rows, metric_columns, optional_metadata_columns)
        if plot_path is not None:
            _save_combined_plot(rows, plot_path, metric_columns, args.window)
        if output_dir is not None:
            _save_output_dir_plots(
                rows,
                output_dir,
                metric_columns,
                args.window,
                args.last_fraction,
            )
    except ImportError:
        print("error: matplotlib is required for plotting", file=sys.stderr)
        return 1
    except Exception as error:
        print(f"error: failed to analyze train log: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
