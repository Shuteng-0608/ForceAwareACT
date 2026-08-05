import numpy as np
import pytest

from force_aware_act.inference import (
    OFFICIAL_TEMPORAL_AGGREGATION_DECAY,
    OFFICIAL_TEMPORAL_AGGREGATION_VERSION,
    OFFICIAL_TEMPORAL_CANDIDATE_ORDER,
    OFFICIAL_TEMPORAL_WEIGHT_FORMULA,
    OfficialTemporalActionChunkExecutor,
)
from scripts.run_mujoco_policy_rollout import parse_args


def _chunk(prediction_step: int, chunk_len: int = 4) -> np.ndarray:
    indices = np.arange(chunk_len, dtype=np.float64)
    return np.stack(
        (
            prediction_step * 100.0 + indices,
            prediction_step * -10.0 - indices,
        ),
        axis=1,
    )


def _official_reference(
    chunks: list[np.ndarray],
    current_step: int,
    decay: float,
) -> tuple[np.ndarray, np.ndarray]:
    candidates = np.asarray(
        [
            chunk[current_step - prediction_step]
            for prediction_step, chunk in enumerate(chunks)
            if 0 <= current_step - prediction_step < chunk.shape[0]
        ],
        dtype=np.float64,
    )
    weights = np.exp(-decay * np.arange(len(candidates), dtype=np.float64))
    weights /= weights.sum()
    return np.sum(candidates * weights[:, None], axis=0), weights


def test_executor_matches_official_reference_during_warmup_and_eviction():
    executor = OfficialTemporalActionChunkExecutor()
    chunks: list[np.ndarray] = []

    for step in range(9):
        chunks.append(_chunk(step))
        result = executor.update(step, chunks[-1])
        expected_action, expected_weights = _official_reference(
            chunks,
            step,
            OFFICIAL_TEMPORAL_AGGREGATION_DECAY,
        )

        np.testing.assert_allclose(result.action, expected_action)
        np.testing.assert_allclose(result.weights, expected_weights)
        expected_first_step = max(0, step - 3)
        assert result.prediction_steps == tuple(range(expected_first_step, step + 1))
        assert result.chunk_indices == tuple(range(step - expected_first_step, -1, -1))
        assert result.ages == result.chunk_indices
        assert result.num_predictions == min(step + 1, 4)
        assert executor.stored_prediction_count == min(step + 1, 4)


def test_official_weight_direction_favors_oldest_candidate():
    executor = OfficialTemporalActionChunkExecutor()
    result = None
    for step in range(100):
        result = executor.update(step, np.zeros((100, 1), dtype=np.float64))

    assert result is not None
    assert result.num_predictions == 100
    assert result.prediction_steps == tuple(range(100))
    assert result.ages == tuple(range(99, -1, -1))
    assert result.oldest_weight > result.newest_weight
    assert result.oldest_weight == pytest.approx(0.01574093123829339)
    assert result.newest_weight == pytest.approx(0.0058489631431306284)
    assert result.weighted_mean_age == pytest.approx(57.696837354988205)


def test_explicit_population_keeps_a_legitimate_zero_action():
    executor = OfficialTemporalActionChunkExecutor(decay=0.0)
    executor.update(0, np.zeros((2, 1), dtype=np.float64))
    result = executor.update(1, np.asarray([[2.0], [3.0]]))

    assert result.num_predictions == 2
    np.testing.assert_allclose(result.action, [1.0])


def test_executor_requires_exactly_one_prediction_per_step():
    executor = OfficialTemporalActionChunkExecutor()
    executor.update(0, _chunk(0))

    with pytest.raises(ValueError, match="one policy query per step"):
        executor.update(2, _chunk(2))


def test_executor_rejects_shape_changes():
    executor = OfficialTemporalActionChunkExecutor()
    executor.update(0, np.zeros((4, 2), dtype=np.float64))

    with pytest.raises(ValueError, match="shape changed"):
        executor.update(1, np.zeros((5, 2), dtype=np.float64))


@pytest.mark.parametrize("decay", (-0.01, float("nan"), float("inf")))
def test_executor_rejects_invalid_decay(decay):
    with pytest.raises(ValueError, match="finite and non-negative"):
        OfficialTemporalActionChunkExecutor(decay=decay)


def test_executor_metadata_and_rollout_cli_default_are_official():
    args = parse_args(["--checkpoint", "model.pt", "--output-dir", "rollout"])

    assert args.temporal_agg_decay == OFFICIAL_TEMPORAL_AGGREGATION_DECAY
    assert OfficialTemporalActionChunkExecutor.version == (
        OFFICIAL_TEMPORAL_AGGREGATION_VERSION
    )
    assert OfficialTemporalActionChunkExecutor.candidate_order == (
        OFFICIAL_TEMPORAL_CANDIDATE_ORDER
    )
    assert OfficialTemporalActionChunkExecutor.weight_formula == (
        OFFICIAL_TEMPORAL_WEIGHT_FORMULA
    )
