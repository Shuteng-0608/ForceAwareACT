import csv

from scripts.evaluate_inference_modes import SAMPLE_OUTPUT_COLUMNS, _write_ranked_cases


def _sample_row(index: int, improvement: float) -> dict[str, object]:
    row: dict[str, object] = {column: 0.0 for column in SAMPLE_OUTPUT_COLUMNS}
    row.update(
        {
            "global_dataset_index": index,
            "episode_path": f"episode_{index}.hdf5",
            "force_prior_improvement_vs_zero": improvement,
        }
    )
    return row


def _read_indices(path) -> list[int]:
    with path.open(newline="") as csv_file:
        return [int(row["global_dataset_index"]) for row in csv.DictReader(csv_file)]


def test_ranked_cases_sort_improvement_smaller_as_worse(tmp_path):
    rows = [_sample_row(0, 0.2), _sample_row(1, -0.4), _sample_row(2, 0.8)]
    worst_path = tmp_path / "worst.csv"
    best_path = tmp_path / "best.csv"

    _write_ranked_cases(
        worst_path,
        rows,
        sort_key="force_prior_improvement_vs_zero",
        top_k=2,
        best=False,
    )
    _write_ranked_cases(
        best_path,
        rows,
        sort_key="force_prior_improvement_vs_zero",
        top_k=2,
        best=True,
    )

    assert _read_indices(worst_path) == [1, 0]
    assert _read_indices(best_path) == [2, 0]
