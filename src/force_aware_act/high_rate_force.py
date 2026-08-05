"""Timestamp-driven 500 Hz force windows shared by training and rollout."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


HIGH_RATE_FORCE_CONTRACT_VERSION = "causal_raw_500hz_interval_force_v1"


@dataclass(frozen=True)
class HighRateForceContract:
    """Fixed tensor capacities for the first high-rate force architecture."""

    sample_rate_hz: float = 500.0
    online_window_len: int = 100
    max_samples_per_interval: int = 20
    max_online_intervals: int = 7

    def __post_init__(self) -> None:
        if not np.isfinite(self.sample_rate_hz) or self.sample_rate_hz <= 0.0:
            raise ValueError("sample_rate_hz must be finite and positive")
        for name in (
            "online_window_len",
            "max_samples_per_interval",
            "max_online_intervals",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")


DEFAULT_HIGH_RATE_FORCE_CONTRACT = HighRateForceContract()


@dataclass(frozen=True)
class CausalForceWindow:
    """A left-padded raw-force window ending no later than an anchor time."""

    values: np.ndarray
    timestamps: np.ndarray
    padding_mask: np.ndarray

    @property
    def valid_count(self) -> int:
        return int((~self.padding_mask).sum())


@dataclass(frozen=True)
class PackedForceIntervals:
    """Fixed-capacity force samples grouped by timestamp interval."""

    values: np.ndarray
    relative_times: np.ndarray
    sample_padding_mask: np.ndarray
    interval_padding_mask: np.ndarray
    sample_counts: np.ndarray

    @property
    def valid_sample_count(self) -> int:
        return int(self.sample_counts.sum())


def select_causal_force_window(
    force_timestamps: np.ndarray,
    force_values: np.ndarray,
    *,
    anchor_time: float,
    window_len: int,
) -> CausalForceWindow:
    """Select the last ``window_len`` samples with timestamp <= ``anchor_time``."""

    timestamps, values = _validate_force_series(force_timestamps, force_values)
    if not np.isfinite(anchor_time):
        raise ValueError("anchor_time must be finite")
    if not isinstance(window_len, int) or isinstance(window_len, bool) or window_len <= 0:
        raise ValueError("window_len must be a positive integer")

    causal_end = int(np.searchsorted(timestamps, anchor_time, side="right"))
    causal_start = max(0, causal_end - window_len)
    selected_values = values[causal_start:causal_end]
    selected_timestamps = timestamps[causal_start:causal_end]
    valid_count = int(selected_values.shape[0])
    destination_start = window_len - valid_count

    window_values = np.zeros((window_len, values.shape[1]), dtype=values.dtype)
    window_timestamps = np.full(window_len, np.nan, dtype=np.float64)
    padding_mask = np.ones(window_len, dtype=np.bool_)
    if valid_count:
        window_values[destination_start:] = selected_values
        window_timestamps[destination_start:] = selected_timestamps
        padding_mask[destination_start:] = False
    return CausalForceWindow(
        values=window_values,
        timestamps=window_timestamps,
        padding_mask=padding_mask,
    )


def pack_force_intervals(
    force_timestamps: np.ndarray,
    force_values: np.ndarray,
    interval_boundaries: np.ndarray,
    *,
    max_samples_per_interval: int,
) -> PackedForceIntervals:
    """Pack samples into non-overlapping ``(left, right]`` intervals."""

    timestamps, values = _validate_force_series(
        force_timestamps,
        force_values,
        allow_empty=True,
    )
    boundaries = _validate_timestamps(
        interval_boundaries,
        "interval_boundaries",
        allow_empty=False,
        strictly_increasing=True,
    )
    if boundaries.shape[0] < 2:
        raise ValueError("interval_boundaries must contain at least two values")
    if (
        not isinstance(max_samples_per_interval, int)
        or isinstance(max_samples_per_interval, bool)
        or max_samples_per_interval <= 0
    ):
        raise ValueError("max_samples_per_interval must be a positive integer")

    interval_count = int(boundaries.shape[0] - 1)
    packed_values = np.zeros(
        (interval_count, max_samples_per_interval, values.shape[1]),
        dtype=values.dtype,
    )
    relative_times = np.zeros(
        (interval_count, max_samples_per_interval),
        dtype=np.float32,
    )
    sample_padding_mask = np.ones(
        (interval_count, max_samples_per_interval),
        dtype=np.bool_,
    )
    interval_padding_mask = np.ones(interval_count, dtype=np.bool_)
    sample_counts = np.zeros(interval_count, dtype=np.int64)

    for interval_index, (left, right) in enumerate(
        zip(boundaries[:-1], boundaries[1:])
    ):
        start = int(np.searchsorted(timestamps, left, side="right"))
        end = int(np.searchsorted(timestamps, right, side="right"))
        count = end - start
        if count > max_samples_per_interval:
            raise ValueError(
                "force interval capacity exceeded: "
                f"interval={interval_index}, left={left:.9g}, right={right:.9g}, "
                f"samples={count}, capacity={max_samples_per_interval}"
            )
        if count == 0:
            continue
        packed_values[interval_index, :count] = values[start:end]
        relative_times[interval_index, :count] = (
            timestamps[start:end] - left
        ).astype(np.float32)
        sample_padding_mask[interval_index, :count] = False
        interval_padding_mask[interval_index] = False
        sample_counts[interval_index] = count

    return PackedForceIntervals(
        values=packed_values,
        relative_times=relative_times,
        sample_padding_mask=sample_padding_mask,
        interval_padding_mask=interval_padding_mask,
        sample_counts=sample_counts,
    )


def build_online_force_intervals(
    force_timestamps: np.ndarray,
    force_values: np.ndarray,
    state_timestamps: np.ndarray,
    *,
    state_index: int,
    contract: HighRateForceContract = DEFAULT_HIGH_RATE_FORCE_CONTRACT,
) -> tuple[CausalForceWindow, PackedForceIntervals]:
    """Build the last-100 causal window and its policy-interval grouping."""

    states = _validate_timestamps(
        state_timestamps,
        "state_timestamps",
        allow_empty=False,
        strictly_increasing=True,
    )
    if (
        not isinstance(state_index, int)
        or isinstance(state_index, bool)
        or not 0 <= state_index < states.shape[0]
    ):
        raise IndexError("state_index is outside state_timestamps")
    window = select_causal_force_window(
        force_timestamps,
        force_values,
        anchor_time=float(states[state_index]),
        window_len=contract.online_window_len,
    )
    boundaries = _online_interval_boundaries(
        states,
        state_index=state_index,
        interval_count=contract.max_online_intervals,
    )
    valid = ~window.padding_mask
    packed = pack_force_intervals(
        window.timestamps[valid],
        window.values[valid],
        boundaries,
        max_samples_per_interval=contract.max_samples_per_interval,
    )
    if packed.valid_sample_count != window.valid_count:
        raise RuntimeError(
            "online interval grouping did not preserve every causal force sample: "
            f"window={window.valid_count}, grouped={packed.valid_sample_count}"
        )
    return window, packed


def build_future_force_intervals(
    force_timestamps: np.ndarray,
    force_values: np.ndarray,
    state_timestamps: np.ndarray,
    *,
    state_index: int,
    chunk_len: int,
    max_samples_per_interval: int,
) -> PackedForceIntervals:
    """Build action-response intervals ``(t_j, t_j+1]`` for one chunk."""

    states = _validate_timestamps(
        state_timestamps,
        "state_timestamps",
        allow_empty=False,
        strictly_increasing=True,
    )
    if (
        not isinstance(state_index, int)
        or isinstance(state_index, bool)
        or not 0 <= state_index < states.shape[0]
    ):
        raise IndexError("state_index is outside state_timestamps")
    if not isinstance(chunk_len, int) or isinstance(chunk_len, bool) or chunk_len <= 0:
        raise ValueError("chunk_len must be a positive integer")

    valid_intervals = min(chunk_len, states.shape[0] - state_index - 1)
    force_ts, values = _validate_force_series(force_timestamps, force_values)
    if valid_intervals:
        valid = pack_force_intervals(
            force_ts,
            values,
            states[state_index : state_index + valid_intervals + 1],
            max_samples_per_interval=max_samples_per_interval,
        )
    else:
        valid = _empty_intervals(
            interval_count=0,
            max_samples_per_interval=max_samples_per_interval,
            force_dim=values.shape[1],
            dtype=values.dtype,
        )
    return _right_pad_intervals(valid, interval_count=chunk_len)


def _online_interval_boundaries(
    state_timestamps: np.ndarray,
    *,
    state_index: int,
    interval_count: int,
) -> np.ndarray:
    if state_timestamps.shape[0] >= 2:
        nominal_period = float(np.median(np.diff(state_timestamps)))
    else:
        nominal_period = 1.0 / 30.0
    available_start = max(0, state_index - interval_count)
    available = state_timestamps[available_start : state_index + 1]
    missing = interval_count + 1 - available.shape[0]
    if missing:
        prefix = available[0] - nominal_period * np.arange(
            missing,
            0,
            -1,
            dtype=np.float64,
        )
        available = np.concatenate((prefix, available))
    return available


def _right_pad_intervals(
    intervals: PackedForceIntervals,
    *,
    interval_count: int,
) -> PackedForceIntervals:
    current = intervals.values.shape[0]
    if current > interval_count:
        raise ValueError("cannot pad intervals to a smaller interval count")
    if current == interval_count:
        return intervals
    sample_capacity = intervals.values.shape[1]
    force_dim = intervals.values.shape[2]
    result = _empty_intervals(
        interval_count=interval_count,
        max_samples_per_interval=sample_capacity,
        force_dim=force_dim,
        dtype=intervals.values.dtype,
    )
    if current:
        result.values[:current] = intervals.values
        result.relative_times[:current] = intervals.relative_times
        result.sample_padding_mask[:current] = intervals.sample_padding_mask
        result.interval_padding_mask[:current] = intervals.interval_padding_mask
        result.sample_counts[:current] = intervals.sample_counts
    return result


def _empty_intervals(
    *,
    interval_count: int,
    max_samples_per_interval: int,
    force_dim: int,
    dtype: np.dtype,
) -> PackedForceIntervals:
    return PackedForceIntervals(
        values=np.zeros(
            (interval_count, max_samples_per_interval, force_dim),
            dtype=dtype,
        ),
        relative_times=np.zeros(
            (interval_count, max_samples_per_interval),
            dtype=np.float32,
        ),
        sample_padding_mask=np.ones(
            (interval_count, max_samples_per_interval),
            dtype=np.bool_,
        ),
        interval_padding_mask=np.ones(interval_count, dtype=np.bool_),
        sample_counts=np.zeros(interval_count, dtype=np.int64),
    )


def _validate_force_series(
    timestamps: np.ndarray,
    values: np.ndarray,
    *,
    allow_empty: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    timestamp_array = _validate_timestamps(
        timestamps,
        "force_timestamps",
        allow_empty=allow_empty,
        strictly_increasing=True,
    )
    value_array = np.asarray(values)
    if value_array.ndim != 2 or value_array.shape[1] <= 0:
        raise ValueError("force_values must have shape [N, force_dim]")
    if value_array.shape[0] != timestamp_array.shape[0]:
        raise ValueError("force timestamps and values must have equal length")
    if not np.issubdtype(value_array.dtype, np.floating):
        raise ValueError("force_values must be floating point")
    if not np.isfinite(value_array).all():
        raise ValueError("force_values must be finite")
    return timestamp_array, value_array


def _validate_timestamps(
    timestamps: np.ndarray,
    name: str,
    *,
    allow_empty: bool,
    strictly_increasing: bool,
) -> np.ndarray:
    array = np.asarray(timestamps, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not allow_empty and array.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    differences = np.diff(array)
    if strictly_increasing:
        invalid = differences <= 0.0
    else:
        invalid = differences < 0.0
    if np.any(invalid):
        relation = "strictly increasing" if strictly_increasing else "non-decreasing"
        raise ValueError(f"{name} must be {relation}")
    return array
