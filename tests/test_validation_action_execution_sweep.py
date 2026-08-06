import numpy as np
import pytest
import torch

from scripts.sweep_paired50_validation_action_execution import (
    OFFICIAL_BASELINE_ID,
    EpisodeCache,
    ExecutorSpec,
    _latest_linear_force_norms,
    aggregate_replay_rows,
    build_rankings,
    build_executor_specs,
    classify_contact_stages,
    replay_action_chunks,
)


def _constant_chunk(values):
    return np.asarray(values, dtype=np.float64).reshape(-1, 1)


def test_default_spec_grid_has_stable_unique_baseline_identifier():
    specs = build_executor_specs(
        (0.0, 0.01, 0.03),
        (0.01, 0.03, 0.1),
        (1, 5, 10, 25, 100),
    )

    assert len(specs) == 11
    assert len({item.executor_id for item in specs}) == len(specs)
    assert OFFICIAL_BASELINE_ID in {item.executor_id for item in specs}


def test_contact_stage_boundaries_are_explicit_and_exhaustive():
    stages = classify_contact_stages(
        np.asarray([0.0, 4.999, 5.0, 19.999, 20.0, 100.0])
    )

    np.testing.assert_array_equal(stages, [0, 0, 1, 1, 2, 2])


def test_official_and_recency_temporal_replay_use_shared_alignment():
    chunks = [
        _constant_chunk([0.0, 10.0, 20.0]),
        _constant_chunk([100.0, 110.0, 120.0]),
        _constant_chunk([200.0, 210.0, 220.0]),
    ]
    official = replay_action_chunks(
        chunks,
        ExecutorSpec("official", "official_temporal", decay=1.0),
    )
    recency = replay_action_chunks(
        chunks,
        ExecutorSpec("recency", "recency_temporal", decay=1.0),
    )

    assert official.actions.shape == (3, 1)
    assert recency.actions.shape == (3, 1)
    assert official.actions[-1, 0] < recency.actions[-1, 0]
    assert official.prediction_ages[-1] > recency.prediction_ages[-1]
    assert official.policy_queried.all()
    assert recency.policy_queried.all()


def test_receding_replay_queries_and_executes_chunk_in_order():
    chunks = [
        _constant_chunk([0.0, 1.0, 2.0, 3.0]),
        _constant_chunk([10.0, 11.0, 12.0, 13.0]),
        _constant_chunk([20.0, 21.0, 22.0, 23.0]),
        _constant_chunk([30.0, 31.0, 32.0, 33.0]),
    ]
    replay = replay_action_chunks(
        chunks,
        ExecutorSpec("q2", "receding_chunk", query_interval=2),
    )

    np.testing.assert_allclose(replay.actions[:, 0], [0.0, 1.0, 20.0, 21.0])
    np.testing.assert_array_equal(replay.policy_queried, [True, False, True, False])
    np.testing.assert_array_equal(replay.chunk_indices, [0, 1, 0, 1])
    np.testing.assert_allclose(replay.prediction_ages, [0.0, 1.0, 0.0, 1.0])


def test_latest_force_uses_last_valid_causal_sample_not_last_storage_slot():
    values = torch.zeros((2, 2, 3, 6), dtype=torch.float32)
    masks = torch.ones((2, 2, 3), dtype=torch.bool)
    values[0, 0, 1, 0] = 3.0
    values[0, 0, 1, 1] = 4.0
    masks[0, 0, 1] = False
    values[0, 1, 0, 0] = 6.0
    values[0, 1, 0, 2] = 8.0
    masks[0, 1, 0] = False
    values[1, 1, 2, 1] = 12.0
    masks[1, 1, 2] = False

    norms = _latest_linear_force_norms(
        values,
        masks,
        force_mean=(0.0,) * 6,
        force_std=(1.0,) * 6,
    )

    torch.testing.assert_close(norms, torch.tensor([10.0, 12.0]))


def test_aggregate_replay_resets_executor_at_episode_boundaries_and_ranks():
    episode = EpisodeCache(
        official_chunks=[_constant_chunk([1.0, 2.0]) for _ in range(2)],
        contact_chunks=[_constant_chunk([0.0, 0.0]) for _ in range(2)],
        current_targets=[np.asarray([0.0]), np.asarray([0.0])],
        current_force_norms=[0.0, 25.0],
    )
    specs = (
        ExecutorSpec(OFFICIAL_BASELINE_ID, "official_temporal", decay=0.01),
        ExecutorSpec("receding_chunk_q1", "receding_chunk", query_interval=1),
    )

    aggregate, per_episode = aggregate_replay_rows(
        {"episode_a": episode, "episode_b": episode}, specs
    )

    assert len(aggregate) == 4
    assert len(per_episode) == 8
    contact_q1 = next(
        row
        for row in aggregate
        if row["model"] == "contact" and row["executor_id"] == "receding_chunk_q1"
    )
    assert contact_q1["action_l1_physical_global"] == pytest.approx(0.0)
    assert contact_q1["policy_query_count"] == 4
    assert contact_q1["free_lt5n_steps"] == 2
    assert contact_q1["contact_ge20n_steps"] == 2

    rankings, paired = build_rankings(aggregate)
    assert set(rankings) == {"official", "contact"}
    assert len(paired) == 2
    assert {row["executor_id"] for row in paired} == {
        OFFICIAL_BASELINE_ID,
        "receding_chunk_q1",
    }
