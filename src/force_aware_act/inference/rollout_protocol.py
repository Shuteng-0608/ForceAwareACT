"""Shared timing, command post-processing, and task-success rollout protocol."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


ROLLOUT_PROTOCOL_VERSION = "paired_temporal_rollout_v1"
POLICY_STEP_SCHEDULER_VERSION = "cumulative_round_policy_clock_v1"
CONTROL_POSTPROCESS_VERSION = "joint_target_delta_clip_ema_ctrlrange_v1"
TASK_SUCCESS_VERSION = "collector_site_distance_dwell_v1"

DEFAULT_POLICY_RATE_HZ = 30.0
DEFAULT_MAX_ROLLOUT_STEPS = 600
DEFAULT_EMA_ALPHA = 1.0
DEFAULT_MAX_DELTA_Q = 0.02
DEFAULT_FORCE_STOP_THRESHOLD = 120.0
DEFAULT_SAFE_FORCE_THRESHOLD = 100.0
DEFAULT_SUCCESS_DISTANCE_THRESHOLD = 0.003
DEFAULT_SUCCESS_DWELL_TIME = 0.10


class CumulativePolicyStepScheduler:
    """Alternate integer physics-step counts around an exact policy period."""

    version = POLICY_STEP_SCHEDULER_VERSION

    def __init__(self, *, policy_rate_hz: float, physics_timestep: float) -> None:
        if not math.isfinite(policy_rate_hz) or policy_rate_hz <= 0.0:
            raise ValueError("policy_rate_hz must be finite and positive")
        if not math.isfinite(physics_timestep) or physics_timestep <= 0.0:
            raise ValueError("physics_timestep must be finite and positive")
        physics_rate_hz = 1.0 / physics_timestep
        if policy_rate_hz > physics_rate_hz:
            raise ValueError("policy rate cannot exceed the physics rate")
        self.policy_rate_hz = float(policy_rate_hz)
        self.physics_timestep = float(physics_timestep)
        self.ideal_physics_steps = 1.0 / (
            self.policy_rate_hz * self.physics_timestep
        )
        self._interval_count = 0
        self._emitted_physics_steps = 0

    @property
    def interval_count(self) -> int:
        return self._interval_count

    @property
    def emitted_physics_steps(self) -> int:
        return self._emitted_physics_steps

    def next_step_count(self) -> int:
        self._interval_count += 1
        cumulative_target = int(
            math.floor(
                self._interval_count * self.ideal_physics_steps + 0.5
            )
        )
        step_count = cumulative_target - self._emitted_physics_steps
        if step_count <= 0:
            raise RuntimeError("policy scheduler emitted a non-positive step count")
        self._emitted_physics_steps = cumulative_target
        return step_count


@dataclass(frozen=True)
class TaskSuccessUpdate:
    condition: bool
    consecutive_observations: int
    accumulated_time: float
    success: bool
    just_succeeded: bool


class TaskSuccessTracker:
    """Match the collector's site-distance and continuous-dwell semantics."""

    version = TASK_SUCCESS_VERSION

    def __init__(self, *, distance_threshold: float, dwell_time: float) -> None:
        if not math.isfinite(distance_threshold) or distance_threshold <= 0.0:
            raise ValueError("distance_threshold must be finite and positive")
        if not math.isfinite(dwell_time) or dwell_time <= 0.0:
            raise ValueError("dwell_time must be finite and positive")
        self.distance_threshold = float(distance_threshold)
        self.dwell_time = float(dwell_time)
        self.accumulated_time = 0.0
        self.max_accumulated_time = 0.0
        self.consecutive_observations = 0
        self.max_consecutive_observations = 0
        self.success = False
        self._previous_time: float | None = None
        self._previous_condition = False

    def update(self, *, timestamp: float, distance: float) -> TaskSuccessUpdate:
        if not math.isfinite(timestamp):
            raise ValueError("timestamp must be finite")
        if self._previous_time is not None and timestamp < self._previous_time:
            raise ValueError("timestamps must be monotonically non-decreasing")
        condition = bool(
            math.isfinite(distance) and distance <= self.distance_threshold
        )
        if condition:
            self.consecutive_observations += 1
            if self._previous_condition and self._previous_time is not None:
                self.accumulated_time += timestamp - self._previous_time
            else:
                self.accumulated_time = 0.0
        else:
            self.consecutive_observations = 0
            self.accumulated_time = 0.0
        self.max_consecutive_observations = max(
            self.max_consecutive_observations,
            self.consecutive_observations,
        )
        self.max_accumulated_time = max(
            self.max_accumulated_time,
            self.accumulated_time,
        )
        just_succeeded = bool(
            not self.success
            and condition
            and self.accumulated_time + 1.0e-12 >= self.dwell_time
        )
        if just_succeeded:
            self.success = True
        self._previous_time = float(timestamp)
        self._previous_condition = condition
        return TaskSuccessUpdate(
            condition=condition,
            consecutive_observations=self.consecutive_observations,
            accumulated_time=self.accumulated_time,
            success=self.success,
            just_succeeded=just_succeeded,
        )


@dataclass(frozen=True)
class JointPositionPostprocessResult:
    target: np.ndarray
    delta_clipped: np.ndarray
    ema: np.ndarray
    ctrlrange_clipped: np.ndarray
    delta_clip_applied: bool
    ema_modified: bool
    ctrlrange_clip_applied: bool


class JointPositionPostprocessor:
    """Apply one identical joint-target transform to every compared model."""

    version = CONTROL_POSTPROCESS_VERSION

    def __init__(
        self,
        *,
        control_ranges: np.ndarray,
        initial_command: np.ndarray,
        max_delta_q: float,
        ema_alpha: float,
    ) -> None:
        ranges = np.asarray(control_ranges, dtype=np.float64)
        initial = np.asarray(initial_command, dtype=np.float64)
        if ranges.ndim != 2 or ranges.shape[1] != 2:
            raise ValueError("control_ranges must have shape [action_dim, 2]")
        if initial.shape != (ranges.shape[0],):
            raise ValueError("initial_command shape must match control_ranges")
        if not np.isfinite(ranges).all() or np.any(ranges[:, 0] > ranges[:, 1]):
            raise ValueError("control_ranges must be finite and ordered")
        if not np.isfinite(initial).all():
            raise ValueError("initial_command must be finite")
        if not math.isfinite(max_delta_q) or max_delta_q <= 0.0:
            raise ValueError("max_delta_q must be finite and positive")
        if not math.isfinite(ema_alpha) or not 0.0 <= ema_alpha <= 1.0:
            raise ValueError("ema_alpha must be finite and in [0, 1]")
        self.control_ranges = ranges.copy()
        self.max_delta_q = float(max_delta_q)
        self.ema_alpha = float(ema_alpha)
        self.previous_command = initial.copy()

    def process(
        self,
        *,
        target: np.ndarray,
        current_qpos: np.ndarray,
        commit: bool = True,
    ) -> JointPositionPostprocessResult:
        target_values = np.asarray(target, dtype=np.float64)
        qpos = np.asarray(current_qpos, dtype=np.float64)
        expected_shape = self.previous_command.shape
        if target_values.shape != expected_shape or qpos.shape != expected_shape:
            raise ValueError(f"target and current_qpos must have shape {expected_shape}")
        if not np.isfinite(target_values).all() or not np.isfinite(qpos).all():
            raise ValueError("target and current_qpos must be finite")
        clipped_delta = np.clip(
            target_values - qpos,
            -self.max_delta_q,
            self.max_delta_q,
        )
        delta_clipped = qpos + clipped_delta
        ema = (
            self.ema_alpha * delta_clipped
            + (1.0 - self.ema_alpha) * self.previous_command
        )
        ctrlrange_clipped = np.clip(
            ema,
            self.control_ranges[:, 0],
            self.control_ranges[:, 1],
        )
        result = JointPositionPostprocessResult(
            target=target_values.copy(),
            delta_clipped=delta_clipped,
            ema=ema,
            ctrlrange_clipped=ctrlrange_clipped,
            delta_clip_applied=not np.array_equal(
                clipped_delta,
                target_values - qpos,
            ),
            ema_modified=not np.array_equal(ema, delta_clipped),
            ctrlrange_clip_applied=not np.array_equal(ctrlrange_clipped, ema),
        )
        if commit:
            self.previous_command = ctrlrange_clipped.copy()
        return result
