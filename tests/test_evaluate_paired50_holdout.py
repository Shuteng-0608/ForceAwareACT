import numpy as np
import pytest
import torch

from scripts.evaluate_paired50_holdout import MetricSums, temporal_ensemble_l1


def test_metric_sums_uses_only_valid_unpadded_elements():
    metrics = MetricSums()
    prediction = torch.tensor([[[1.0, 3.0], [100.0, 100.0]]])
    target = torch.zeros_like(prediction)
    valid = torch.tensor([[True, False]])

    metrics.add_absolute_error("error", prediction, target, valid)

    assert metrics.counts["error"] == 2
    assert metrics.means()["error"] == pytest.approx(2.0)


def test_temporal_ensemble_l1_matches_official_candidate_alignment():
    chunks = [
        np.asarray([[1.0], [3.0]], dtype=np.float64),
        np.asarray([[5.0], [7.0]], dtype=np.float64),
    ]
    targets = [
        np.asarray([1.0], dtype=np.float64),
        np.asarray([4.0], dtype=np.float64),
    ]

    total, count = temporal_ensemble_l1(chunks, targets, decay=0.0)

    # Step 0 uses 1. Step 1 averages the aligned values 3 and 5 to obtain 4.
    assert total == pytest.approx(0.0)
    assert count == 2


def test_temporal_ensemble_l1_rejects_mismatched_sequences():
    with pytest.raises(ValueError, match="matching non-empty"):
        temporal_ensemble_l1(
            [np.zeros((2, 1), dtype=np.float64)],
            [],
            decay=0.01,
        )
