import numpy as np
import pytest

from force_aware_act.inference import (
    OFFICIAL_TEMPORAL_AGGREGATION_DECAY,
    OFFICIAL_TEMPORAL_AGGREGATION_VERSION,
    OFFICIAL_TEMPORAL_CANDIDATE_ORDER,
    OFFICIAL_TEMPORAL_WEIGHT_FORMULA,
    OfficialTemporalActionChunkExecutor,
    RECEDING_CHUNK_EXECUTION_VERSION,
    RECENCY_TEMPORAL_AGGREGATION_VERSION,
    RecencyTemporalActionChunkExecutor,
    RecedingChunkActionExecutor,
    SignedAgeTemporalActionChunkExecutor,
    TemporalEndpointActionChunkExecutor,
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


def test_recency_executor_favors_newest_and_uses_prediction_age():
    executor = RecencyTemporalActionChunkExecutor(decay=0.1)
    result = None
    for step in range(100):
        result = executor.update(step, np.full((100, 1), step, dtype=np.float64))

    assert result is not None
    assert executor.version == RECENCY_TEMPORAL_AGGREGATION_VERSION
    assert result.ages == tuple(range(99, -1, -1))
    assert result.newest_weight > result.oldest_weight
    assert result.weighted_mean_age == pytest.approx(9.50379174567408)


def test_recency_decay_zero_matches_uniform_official_aggregation():
    official = OfficialTemporalActionChunkExecutor(decay=0.0)
    recency = RecencyTemporalActionChunkExecutor(decay=0.0)
    for step in range(5):
        chunk = _chunk(step, chunk_len=5)
        official_result = official.update(step, chunk)
        recency_result = recency.update(step, chunk)

    np.testing.assert_allclose(recency_result.action, official_result.action)
    np.testing.assert_allclose(recency_result.weights, official_result.weights)


def test_signed_negative_official_mapping_is_exact():
    official = OfficialTemporalActionChunkExecutor(decay=0.01)
    signed = SignedAgeTemporalActionChunkExecutor(signed_decay=-0.01)
    for step in range(100):
        chunk = _chunk(step, chunk_len=100)
        official_result = official.update(step, chunk)
        signed_result = signed.update(step, chunk)

    np.testing.assert_allclose(signed_result.action, official_result.action)
    np.testing.assert_allclose(signed_result.weights, official_result.weights)
    assert signed_result.weighted_mean_age == pytest.approx(
        official_result.weighted_mean_age
    )


@pytest.mark.parametrize(
    ("signed_decay", "favored_endpoint"),
    ((-1.0, "oldest"), (0.0, "uniform"), (1.0, "newest")),
)
def test_signed_temporal_direction_and_extremes_are_stable(
    signed_decay, favored_endpoint
):
    executor = SignedAgeTemporalActionChunkExecutor(signed_decay=signed_decay)
    for step in range(100):
        result = executor.update(step, np.full((100, 1), step, dtype=np.float64))

    weights = np.asarray(result.weights)
    assert np.isfinite(weights).all()
    assert weights.sum() == pytest.approx(1.0)
    if favored_endpoint == "oldest":
        assert result.oldest_weight > 0.63
        assert result.oldest_weight > result.newest_weight
    elif favored_endpoint == "newest":
        assert result.newest_weight > 0.63
        assert result.newest_weight > result.oldest_weight
    else:
        np.testing.assert_allclose(weights, np.full(100, 0.01))


@pytest.mark.parametrize("signed_decay", (float("nan"), float("inf")))
def test_signed_temporal_rejects_non_finite_decay(signed_decay):
    with pytest.raises(ValueError, match="signed_decay must be finite"):
        SignedAgeTemporalActionChunkExecutor(signed_decay=signed_decay)


def test_temporal_endpoint_selects_exact_oldest_or_newest_prediction():
    oldest = TemporalEndpointActionChunkExecutor(preference="oldest")
    newest = TemporalEndpointActionChunkExecutor(preference="newest")
    for step in range(4):
        chunk = _chunk(step, chunk_len=4)
        oldest_result = oldest.update(step, chunk)
        newest_result = newest.update(step, chunk)

    np.testing.assert_allclose(oldest_result.action, _chunk(0, 4)[3])
    np.testing.assert_allclose(newest_result.action, _chunk(3, 4)[0])
    assert oldest_result.weighted_mean_age == 3.0
    assert newest_result.weighted_mean_age == 0.0


def test_receding_chunk_query_interval_one_uses_each_fresh_first_action():
    executor = RecedingChunkActionExecutor(query_interval=1)
    selected = []
    for step in range(4):
        assert executor.should_query(step)
        result = executor.update(step, _chunk(step, chunk_len=4))
        selected.append(result.action)
        assert result.query_step == step
        assert result.chunk_index == 0

    assert executor.version == RECEDING_CHUNK_EXECUTION_VERSION
    assert executor.query_count == 4
    np.testing.assert_allclose(selected, [_chunk(step, 4)[0] for step in range(4)])


def test_receding_chunk_executes_in_order_until_next_query():
    executor = RecedingChunkActionExecutor(query_interval=4)
    first = _chunk(0, chunk_len=4)
    second = _chunk(4, chunk_len=4)
    results = []
    for step in range(8):
        chunk = first if step == 0 else second if step == 4 else None
        results.append(executor.update(step, chunk))

    np.testing.assert_allclose([item.action for item in results[:4]], first)
    np.testing.assert_allclose([item.action for item in results[4:]], second)
    assert [item.chunk_index for item in results] == [0, 1, 2, 3, 0, 1, 2, 3]
    assert [item.query_step for item in results] == [0, 0, 0, 0, 4, 4, 4, 4]
    assert executor.query_count == 2


def test_receding_chunk_rejects_missing_or_unexpected_queries():
    executor = RecedingChunkActionExecutor(query_interval=2)
    with pytest.raises(ValueError, match="required on a query step"):
        executor.update(0, None)

    executor = RecedingChunkActionExecutor(query_interval=2)
    executor.update(0, _chunk(0, chunk_len=2))
    with pytest.raises(ValueError, match="omitted on a non-query step"):
        executor.update(1, _chunk(1, chunk_len=2))


def test_receding_chunk_rejects_interval_longer_than_chunk():
    executor = RecedingChunkActionExecutor(query_interval=5)
    with pytest.raises(ValueError, match="cannot exceed"):
        executor.update(0, _chunk(0, chunk_len=4))
