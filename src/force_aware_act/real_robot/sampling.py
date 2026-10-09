"""Integer-nanosecond causal observations and fixed, independently masked labels."""

import numpy as np

from force_aware_act.high_rate_force import PackedForceIntervals
from .config import RealRobotDataConfig


def estimate_action_grid_phase_ns(timestamps_ns, period_ns, window_ns):
    """Estimate the stable action clock phase from an initial receive-time window."""
    times = np.asarray(timestamps_ns, dtype=np.int64)
    if times.ndim != 1 or len(times) < 2 or np.any(np.diff(times) <= 0):
        raise ValueError("phase estimation requires at least two strictly increasing action times")
    if type(period_ns) is not int or period_ns <= 0 or type(window_ns) is not int or window_ns <= 0:
        raise ValueError("phase period and window must be positive integer nanoseconds")
    count = int(np.searchsorted(times, int(times[0]) + window_ns, side="right"))
    if count < 2:
        raise ValueError("phase estimation window contains fewer than two actions")
    angle = 2 * np.pi * ((times[:count] - times[0]) % period_ns).astype(np.float64) / period_ns
    mean_vector = np.exp(1j * angle).mean()
    concentration = float(abs(mean_vector))
    if concentration < 0.5:
        raise ValueError("action phase is unstable within the estimation window")
    phase_ns = int(np.rint((np.angle(mean_vector) % (2 * np.pi)) * period_ns / (2 * np.pi))) % period_ns
    return {"method": "circular_mean_initial_window", "phase_ns": phase_ns,
            "period_ns": period_ns, "window_ns": window_ns,
            "window_start_ns": int(times[0]), "window_end_ns": int(times[count - 1]),
            "sample_count": count, "resultant_length": concentration}


def causal_index(timestamps_ns, anchor_ns, max_age_ns):
    """Return a genuinely available recent observation, never a future one."""
    index = int(np.searchsorted(timestamps_ns, int(anchor_ns), side="right")) - 1
    if index < 0 or int(anchor_ns) - int(timestamps_ns[index]) > max_age_ns:
        return None
    return index


def match_time_slots(timestamps_ns, target_ns, tolerance_ns, *, after_ns=None):
    """Match each fixed slot at most once; unmatched slots remain -1, never shift."""
    times = np.asarray(timestamps_ns, dtype=np.int64)
    targets = np.asarray(target_ns, dtype=np.int64)
    if targets.ndim != 1 or times.ndim != 1:
        raise ValueError("time series must be one dimensional")
    if np.any(np.diff(times) <= 0) or np.any(np.diff(targets) <= 0):
        raise ValueError("matching requires strictly increasing time series")
    result = np.full(len(targets), -1, dtype=np.int64)
    previous = -1
    for slot, target in enumerate(targets):
        if not len(times) or target < times[0] or target > times[-1]:
            continue
        insertion = int(np.searchsorted(times, target))
        candidates = [i for i in (insertion - 1, insertion)
                      if previous < i < len(times) and (after_ns is None or times[i] > after_ns)]
        if not candidates:
            continue
        best = min(candidates, key=lambda i: (abs(int(times[i]) - int(target)), i))
        if abs(int(times[best]) - int(target)) <= tolerance_ns:
            result[slot] = best
            previous = best
    return result


def _read_force(values, indices, dimension):
    if hasattr(values, "read"):
        data, valid = values.read(indices)
    else:
        data = np.asarray(values)[indices]
        valid = np.ones(len(data), dtype=np.bool_)
    data = np.asarray(data, dtype=np.float32)
    if data.shape != (len(valid), dimension):
        raise ValueError("force stream dimension differs from its representation")
    return data, np.asarray(valid, dtype=np.bool_)


def pack_torque_history(timestamps_ns, values, anchor_ns, config: RealRobotDataConfig):
    """Keep every actual sample in (anchor-duration, anchor], including real gaps."""
    times = np.asarray(timestamps_ns, dtype=np.int64)
    n, capacity = config.history_intervals, config.max_samples_per_interval
    packed = np.zeros((n, capacity, config.force_dim), dtype=np.float32)
    relative = np.zeros((n, capacity), dtype=np.float32)
    mask = np.ones((n, capacity), dtype=np.bool_)
    counts = np.zeros(n, dtype=np.int64)
    start = int(anchor_ns) - n * config.action_period_ns
    begin, end = np.searchsorted(times, [start, int(anchor_ns)], side="right")
    selected, selected_valid = _read_force(values, slice(begin, end), config.force_dim)
    for interval in range(n):
        left = start + interval * config.action_period_ns
        right = left + config.action_period_ns
        lo = int(np.searchsorted(times, left, side="right"))
        hi = int(np.searchsorted(times, right, side="right"))
        count = hi - lo
        if count > capacity:
            raise ValueError(f"torque history capacity exceeded in interval {interval}: {count}>{capacity}")
        packed[interval, :count] = selected[lo-begin:hi-begin]
        relative[interval, :count] = (times[lo:hi] - left).astype(np.float64) * 1e-9
        mask[interval, :count] = ~selected_valid[lo-begin:hi-begin]
        counts[interval] = int(selected_valid[lo-begin:hi-begin].sum())
    packed[mask] = 0
    return PackedForceIntervals(packed, relative, mask, mask.all(axis=-1), counts)


def future_torque_slot_indices(timestamps_ns, anchor_ns, chunk_len, config: RealRobotDataConfig):
    """Expose the exact raw-source mapping used by future supervision and audits."""
    count = chunk_len * config.samples_per_action
    offsets = np.rint(np.arange(1, count + 1, dtype=np.float64)
                      * (1e9 / config.torque_rate_hz)).astype(np.int64)
    targets = int(anchor_ns) + offsets
    indices = match_time_slots(timestamps_ns, targets,
                              round(config.torque_match_tolerance_ms * 1e6), after_ns=int(anchor_ns))
    return indices.reshape(chunk_len, config.samples_per_action), targets.reshape(chunk_len, config.samples_per_action)


def pack_future_torque(timestamps_ns, values, anchor_ns, chunk_len, config: RealRobotDataConfig):
    """Three fixed torque slots per 10 ms action; missing values keep their slots."""
    indices, _ = future_torque_slot_indices(timestamps_ns, anchor_ns, chunk_len, config)
    shape = (chunk_len, config.max_samples_per_interval)
    packed = np.zeros((*shape, config.force_dim), dtype=np.float32)
    relative = np.zeros(shape, dtype=np.float32)
    mask = np.ones(shape, dtype=np.bool_)
    # Query times and holes keep their nominal positions, including invalid conversions.
    relative[:, :config.samples_per_action] = np.arange(1, config.samples_per_action + 1) / config.torque_rate_hz
    intervals, slots = np.nonzero(indices >= 0)
    selected, valid = _read_force(values, indices[intervals, slots], config.force_dim)
    packed[intervals, slots] = selected
    mask[intervals, slots] = ~valid
    packed[mask] = 0
    return PackedForceIntervals(packed, relative, mask, mask.all(axis=-1), (~mask).sum(axis=-1))


def matching_error_summary(timestamps_ns, targets_ns, indices):
    """Compact absolute matching residuals; never turn missing labels into zeros."""
    targets, indices = np.asarray(targets_ns), np.asarray(indices)
    valid = indices >= 0
    if not np.any(valid):
        return {"matched_count": 0, "missing_count": int(indices.size), "absolute_error_ms": None}
    residual = np.abs(np.asarray(timestamps_ns)[indices[valid]] - targets[valid]).astype(np.float64) * 1e-6
    return {"matched_count": int(valid.sum()), "missing_count": int((~valid).sum()),
            "absolute_error_ms": dict(zip(("p50", "p95", "p99", "max"),
                                            map(float, np.quantile(residual, [.5, .95, .99, 1]))))}
