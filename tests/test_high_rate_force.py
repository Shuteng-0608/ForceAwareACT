import numpy as np
import pytest

from force_aware_act.high_rate_force import (
    DEFAULT_HIGH_RATE_FORCE_CONTRACT,
    HighRateForceContract,
    build_future_force_intervals,
    build_online_force_intervals,
    pack_force_intervals,
    select_causal_force_window,
)


def _series(duration: float = 1.0):
    force_timestamps = np.arange(0.0, duration, 0.002, dtype=np.float64)
    force_values = np.repeat(
        np.arange(force_timestamps.size, dtype=np.float32)[:, None],
        6,
        axis=1,
    )
    state_timestamps = np.arange(0.0, duration, 0.033, dtype=np.float64)
    return force_timestamps, force_values, state_timestamps


def test_causal_window_uses_last_100_native_500hz_samples():
    force_timestamps, force_values, state_timestamps = _series()
    anchor = float(state_timestamps[10])

    window = select_causal_force_window(
        force_timestamps,
        force_values,
        anchor_time=anchor,
        window_len=100,
    )

    expected_end = np.searchsorted(force_timestamps, anchor, side="right")
    expected_indices = np.arange(expected_end - 100, expected_end)
    assert window.valid_count == 100
    np.testing.assert_array_equal(window.values[:, 0], expected_indices)
    np.testing.assert_array_equal(
        window.timestamps,
        force_timestamps[expected_indices],
    )
    assert np.all(window.timestamps <= anchor)


def test_causal_window_left_pads_without_repeating_force_samples():
    force_timestamps, force_values, state_timestamps = _series()

    window = select_causal_force_window(
        force_timestamps,
        force_values,
        anchor_time=float(state_timestamps[1]),
        window_len=100,
    )

    expected_count = int(
        np.searchsorted(force_timestamps, state_timestamps[1], side="right")
    )
    assert window.valid_count == expected_count
    assert window.padding_mask[: 100 - expected_count].all()
    assert not window.padding_mask[100 - expected_count :].any()
    assert np.isnan(window.timestamps[: 100 - expected_count]).all()
    np.testing.assert_array_equal(
        window.values[100 - expected_count :, 0],
        np.arange(expected_count),
    )


def test_interval_boundary_sample_belongs_to_preceding_action_response():
    timestamps = np.asarray([0.0, 0.01, 0.02, 0.03], dtype=np.float64)
    values = np.repeat(np.arange(4, dtype=np.float32)[:, None], 6, axis=1)

    packed = pack_force_intervals(
        timestamps,
        values,
        np.asarray([0.0, 0.02, 0.03]),
        max_samples_per_interval=3,
    )

    assert packed.sample_counts.tolist() == [2, 1]
    np.testing.assert_array_equal(packed.values[0, :2, 0], [1.0, 2.0])
    np.testing.assert_array_equal(packed.values[1, :1, 0], [3.0])


def test_online_grouping_preserves_every_selected_raw_sample_once():
    force_timestamps, force_values, state_timestamps = _series()

    window, intervals = build_online_force_intervals(
        force_timestamps,
        force_values,
        state_timestamps,
        state_index=10,
    )

    assert window.valid_count == DEFAULT_HIGH_RATE_FORCE_CONTRACT.online_window_len
    assert intervals.valid_sample_count == window.valid_count
    recovered = intervals.values[~intervals.sample_padding_mask, 0]
    np.testing.assert_array_equal(recovered, window.values[:, 0])
    assert intervals.sample_counts.max() <= 17


def test_online_training_full_episode_matches_rollout_visible_state_prefix():
    force_timestamps = np.arange(0.0, 0.4, 0.002, dtype=np.float64)
    force_values = np.repeat(
        np.arange(force_timestamps.size, dtype=np.float32)[:, None], 6, axis=1
    )
    state_timestamps = np.asarray(
        [0.0, 0.033, 0.067, 0.100, 0.134, 0.167, 0.200, 0.234],
        dtype=np.float64,
    )

    for state_index in range(len(state_timestamps)):
        full_window, full = build_online_force_intervals(
            force_timestamps,
            force_values,
            state_timestamps,
            state_index=state_index,
        )
        prefix_window, prefix = build_online_force_intervals(
            force_timestamps,
            force_values,
            state_timestamps[: state_index + 1],
            state_index=state_index,
        )
        np.testing.assert_array_equal(full_window.values, prefix_window.values)
        np.testing.assert_array_equal(full.values, prefix.values)
        np.testing.assert_array_equal(full.relative_times, prefix.relative_times)
        np.testing.assert_array_equal(
            full.sample_padding_mask, prefix.sample_padding_mask
        )


def test_future_force_uses_all_samples_in_each_action_response_interval():
    force_timestamps, force_values, state_timestamps = _series()

    intervals = build_future_force_intervals(
        force_timestamps,
        force_values,
        state_timestamps,
        state_index=2,
        chunk_len=5,
        max_samples_per_interval=20,
    )

    assert intervals.values.shape == (5, 20, 6)
    assert set(intervals.sample_counts.tolist()) <= {16, 17}
    for step in range(5):
        left = state_timestamps[2 + step]
        right = state_timestamps[3 + step]
        expected = force_values[(force_timestamps > left) & (force_timestamps <= right)]
        count = intervals.sample_counts[step]
        np.testing.assert_array_equal(intervals.values[step, :count], expected)
        assert np.all(intervals.relative_times[step, :count] > 0.0)
        assert np.all(intervals.relative_times[step, :count] <= right - left + 1e-7)


def test_online_history_and_first_future_interval_are_disjoint():
    force_timestamps, force_values, state_timestamps = _series()
    state_index = 8
    window, _online = build_online_force_intervals(
        force_timestamps,
        force_values,
        state_timestamps,
        state_index=state_index,
    )
    future = build_future_force_intervals(
        force_timestamps,
        force_values,
        state_timestamps,
        state_index=state_index,
        chunk_len=1,
        max_samples_per_interval=20,
    )

    online_indices = set(window.values[~window.padding_mask, 0].astype(int))
    future_indices = set(
        future.values[0, ~future.sample_padding_mask[0], 0].astype(int)
    )
    assert online_indices.isdisjoint(future_indices)
    assert window.timestamps[~window.padding_mask].max() <= state_timestamps[state_index]
    assert future.relative_times[0, 0] > 0.0


def test_future_force_keeps_a_2ms_spike_missed_by_state_rate_sampling():
    force_timestamps, force_values, state_timestamps = _series()
    spike_index = int(np.searchsorted(force_timestamps, state_timestamps[4])) - 3
    force_values[spike_index, 0] = 999.0

    intervals = build_future_force_intervals(
        force_timestamps,
        force_values,
        state_timestamps,
        state_index=2,
        chunk_len=4,
        max_samples_per_interval=20,
    )

    assert 999.0 in intervals.values[..., 0]


def test_future_force_right_pads_missing_episode_tail_intervals():
    force_timestamps, force_values, state_timestamps = _series(duration=0.2)

    intervals = build_future_force_intervals(
        force_timestamps,
        force_values,
        state_timestamps,
        state_index=len(state_timestamps) - 2,
        chunk_len=4,
        max_samples_per_interval=20,
    )

    assert not intervals.interval_padding_mask[0]
    assert intervals.interval_padding_mask[1:].all()
    assert intervals.sample_counts[1:].tolist() == [0, 0, 0]


def test_interval_capacity_overflow_is_never_silently_truncated():
    force_timestamps = np.arange(0.0, 0.05, 0.001, dtype=np.float64)
    force_values = np.zeros((force_timestamps.size, 6), dtype=np.float32)

    with pytest.raises(ValueError, match="capacity exceeded"):
        pack_force_intervals(
            force_timestamps,
            force_values,
            np.asarray([0.0, 0.04]),
            max_samples_per_interval=20,
        )


def test_force_timestamps_must_be_strictly_increasing():
    timestamps = np.asarray([0.0, 0.002, 0.002], dtype=np.float64)
    values = np.zeros((3, 6), dtype=np.float32)

    with pytest.raises(ValueError, match="strictly increasing"):
        select_causal_force_window(
            timestamps,
            values,
            anchor_time=0.002,
            window_len=3,
        )


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"sample_rate_hz": 0.0}, "sample_rate_hz"),
        ({"online_window_len": 0}, "online_window_len"),
        ({"max_samples_per_interval": 0}, "max_samples_per_interval"),
        ({"max_online_intervals": 0}, "max_online_intervals"),
    ],
)
def test_contract_rejects_invalid_capacities(kwargs, match):
    with pytest.raises(ValueError, match=match):
        HighRateForceContract(**kwargs)
