"""Continuous native-rate force sampling state for online rollout."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from force_aware_act.high_rate_force import HighRateForceContract


@dataclass(frozen=True)
class HighRateForceBufferSnapshot:
    force_timestamps: np.ndarray
    force_values: np.ndarray
    state_timestamps: np.ndarray


class HighRateForceRingBuffer:
    """Sample at 500 Hz continuously while retaining the last 100 values."""

    def __init__(
        self,
        initial_timestamp: float,
        initial_wrench: np.ndarray,
        *,
        contract: HighRateForceContract = HighRateForceContract(),
    ) -> None:
        self.contract = contract
        self.period = 1.0 / contract.sample_rate_hz
        self._force = deque(maxlen=contract.online_window_len)
        self._states = deque(maxlen=contract.max_online_intervals + 1)
        self._validate_wrench(initial_wrench)
        if not np.isfinite(initial_timestamp):
            raise ValueError("initial_timestamp must be finite")
        self._force.append((float(initial_timestamp), np.asarray(initial_wrench, dtype=np.float32).copy()))
        self._next_sample_time = float(initial_timestamp) + self.period
        self.total_samples = 1
        self.physics_observations = 0
        self.skipped_sample_count = 0
        self.duplicate_sample_count = 0

    def observe_physics_step(self, timestamp: float, wrench: np.ndarray) -> bool:
        """Record at most one due sample; return whether a sample was added."""

        self._validate_wrench(wrench)
        self.physics_observations += 1
        timestamp = float(timestamp)
        if not np.isfinite(timestamp) or timestamp <= self._force[-1][0]:
            raise ValueError("physics observation timestamps must be strictly increasing")
        tolerance = max(1.0e-12, self.period * 1.0e-6)
        if timestamp + tolerance < self._next_sample_time:
            return False
        if timestamp - self._next_sample_time >= self.period - tolerance:
            self.skipped_sample_count += 1
            raise RuntimeError(
                "physics observation skipped a required high-rate force sample"
            )
        self._force.append((timestamp, np.asarray(wrench, dtype=np.float32).copy()))
        self._next_sample_time += self.period
        self.total_samples += 1
        return True

    def record_policy_state(self, timestamp: float) -> None:
        timestamp = float(timestamp)
        if not np.isfinite(timestamp):
            raise ValueError("policy state timestamp must be finite")
        if self._states and timestamp <= self._states[-1]:
            raise ValueError("policy state timestamps must be strictly increasing")
        if timestamp < self._force[-1][0]:
            raise ValueError("policy state cannot precede the latest force sample")
        self._states.append(timestamp)

    def snapshot(self) -> HighRateForceBufferSnapshot:
        if not self._states:
            raise RuntimeError("record_policy_state must be called before snapshot")
        return HighRateForceBufferSnapshot(
            force_timestamps=np.asarray([item[0] for item in self._force], dtype=np.float64),
            force_values=np.stack([item[1] for item in self._force]).astype(np.float32, copy=False),
            state_timestamps=np.asarray(self._states, dtype=np.float64),
        )

    @staticmethod
    def _validate_wrench(wrench: np.ndarray) -> None:
        values = np.asarray(wrench)
        if values.shape != (6,) or not np.issubdtype(values.dtype, np.floating):
            raise ValueError("wrench must be a floating array with shape [6]")
        if not np.isfinite(values).all():
            raise ValueError("wrench must be finite")
