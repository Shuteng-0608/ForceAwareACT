"""Shared timing, command post-processing, and task-success rollout protocol."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


ROLLOUT_PROTOCOL_VERSION = "collector_terminal_hold_rollout_v5"
POLICY_STEP_SCHEDULER_VERSION = "cumulative_round_policy_clock_v1"
CONTROL_POSTPROCESS_VERSION = "joint_target_delta_clip_ema_ctrlrange_v1"
TASK_SUCCESS_VERSION = "collector_site_distance_force_speed_dwell_v2"
TERMINAL_HOLD_VERSION = "collector_adaptive_terminal_hold_v1"

DEFAULT_POLICY_RATE_HZ = 30.0
DEFAULT_MAX_ROLLOUT_STEPS = 600
DEFAULT_EMA_ALPHA = 1.0
DEFAULT_MAX_DELTA_Q = 0.02
DEFAULT_FORCE_STOP_THRESHOLD = 100.0
DEFAULT_SAFE_FORCE_THRESHOLD = 40.0
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
    """Match collector distance, stability-gate, and continuous-dwell semantics."""

    version = TASK_SUCCESS_VERSION

    def __init__(
        self,
        *,
        distance_threshold: float,
        dwell_time: float,
        max_force_norm: float = float("inf"),
        max_lateral_force: float = float("inf"),
        max_arm_speed: float = float("inf"),
        force_axis_world=(0.0, 1.0, 0.0),
    ) -> None:
        if not math.isfinite(distance_threshold) or distance_threshold <= 0.0:
            raise ValueError("distance_threshold must be finite and positive")
        if not math.isfinite(dwell_time) or dwell_time <= 0.0:
            raise ValueError("dwell_time must be finite and positive")
        for name, value in (
            ("max_force_norm", max_force_norm),
            ("max_lateral_force", max_lateral_force),
            ("max_arm_speed", max_arm_speed),
        ):
            if math.isnan(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and not NaN")
        force_axis = np.asarray(force_axis_world, dtype=np.float64)
        if force_axis.shape != (3,) or not np.isfinite(force_axis).all():
            raise ValueError("force_axis_world must be a finite 3-vector")
        force_axis_norm = float(np.linalg.norm(force_axis))
        if force_axis_norm <= 0.0:
            raise ValueError("force_axis_world must be nonzero")
        self.distance_threshold = float(distance_threshold)
        self.dwell_time = float(dwell_time)
        self.max_force_norm = float(max_force_norm)
        self.max_lateral_force = float(max_lateral_force)
        self.max_arm_speed = float(max_arm_speed)
        self.force_axis_world = force_axis / force_axis_norm
        self.accumulated_time = 0.0
        self.max_accumulated_time = 0.0
        self.consecutive_observations = 0
        self.max_consecutive_observations = 0
        self.success = False
        self._previous_time: float | None = None
        self._previous_condition = False

    def update(
        self,
        *,
        timestamp: float,
        distance: float,
        force_world=None,
        arm_speed: float | None = None,
    ) -> TaskSuccessUpdate:
        if not math.isfinite(timestamp):
            raise ValueError("timestamp must be finite")
        if self._previous_time is not None and timestamp < self._previous_time:
            raise ValueError("timestamps must be monotonically non-decreasing")
        force_required = bool(
            math.isfinite(self.max_force_norm)
            or math.isfinite(self.max_lateral_force)
        )
        if force_world is None:
            force_ok = not force_required
        else:
            force = np.asarray(force_world, dtype=np.float64)
            if force.shape != (3,) or not np.isfinite(force).all():
                force_ok = False
            else:
                axial_force = float(np.dot(force, self.force_axis_world))
                lateral_force = force - axial_force * self.force_axis_world
                force_ok = bool(
                    np.linalg.norm(force) <= self.max_force_norm
                    and np.linalg.norm(lateral_force) <= self.max_lateral_force
                )
        speed_required = math.isfinite(self.max_arm_speed)
        speed_ok = bool(
            (arm_speed is not None and math.isfinite(arm_speed)
             and arm_speed <= self.max_arm_speed)
            if speed_required
            else True
        )
        condition = bool(
            math.isfinite(distance)
            and distance <= self.distance_threshold
            and force_ok
            and speed_ok
        )
        if (
            self._previous_time is not None
            and timestamp == self._previous_time
            and condition == self._previous_condition
        ):
            return TaskSuccessUpdate(
                condition=condition,
                consecutive_observations=self.consecutive_observations,
                accumulated_time=self.accumulated_time,
                success=self.success,
                just_succeeded=False,
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
class TerminalHoldUpdate:
    active: bool
    stable: bool
    force_over: bool
    elapsed_time: float
    stable_time: float
    force_over_time: float
    complete: bool
    success: bool
    reason: str
    just_completed: bool


class TerminalHoldTracker:
    """Mirror the collector's adaptive post-success acceptance state machine."""

    version = TERMINAL_HOLD_VERSION

    def __init__(
        self,
        *,
        min_time: float,
        stable_dwell_time: float,
        max_time: float,
        max_force_norm: float,
        max_lateral_force: float,
        force_abort_dwell_time: float,
        stable_max_force_norm: float,
        stable_max_lateral_force: float,
        stable_max_arm_speed: float,
    ) -> None:
        finite_nonnegative = (
            ("min_time", min_time),
            ("stable_dwell_time", stable_dwell_time),
            ("max_time", max_time),
            ("force_abort_dwell_time", force_abort_dwell_time),
        )
        for name, value in finite_nonnegative:
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        for name, value in (
            ("max_force_norm", max_force_norm),
            ("max_lateral_force", max_lateral_force),
            ("stable_max_force_norm", stable_max_force_norm),
            ("stable_max_lateral_force", stable_max_lateral_force),
            ("stable_max_arm_speed", stable_max_arm_speed),
        ):
            if math.isnan(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and not NaN")
        if max_time < min_time:
            raise ValueError("max_time must be greater than or equal to min_time")

        self.min_time = float(min_time)
        self.stable_dwell_time = float(stable_dwell_time)
        self.max_time = float(max_time)
        self.max_force_norm = float(max_force_norm)
        self.max_lateral_force = float(max_lateral_force)
        self.force_abort_dwell_time = float(force_abort_dwell_time)
        self.stable_max_force_norm = float(stable_max_force_norm)
        self.stable_max_lateral_force = float(stable_max_lateral_force)
        self.stable_max_arm_speed = float(stable_max_arm_speed)
        self.active = False
        self.complete = False
        self.success = False
        self.reason = ""
        self.start_time: float | None = None
        self.stable_start_time: float | None = None
        self.force_over_start_time: float | None = None
        self.last_time: float | None = None
        self.elapsed_time = 0.0
        self.stable_time = 0.0
        self.force_over_time = 0.0
        self.max_stable_time = 0.0
        self.max_force_over_time = 0.0

    def start(self, timestamp: float) -> None:
        if not math.isfinite(timestamp):
            raise ValueError("timestamp must be finite")
        if self.active or self.complete:
            raise RuntimeError("terminal hold can only be started once")
        self.active = True
        self.start_time = float(timestamp)
        self.last_time = float(timestamp)

    def update(
        self,
        *,
        timestamp: float,
        force_norm: float,
        lateral_force: float,
        arm_speed: float,
    ) -> TerminalHoldUpdate:
        if not self.active or self.start_time is None:
            raise RuntimeError("terminal hold must be started before update")
        if not math.isfinite(timestamp):
            raise ValueError("timestamp must be finite")
        if self.last_time is not None and timestamp < self.last_time:
            raise ValueError("timestamps must be monotonically non-decreasing")

        if self.complete:
            return self._result(
                stable=False,
                force_over=False,
                just_completed=False,
            )

        values_finite = all(
            math.isfinite(value)
            for value in (force_norm, lateral_force, arm_speed)
        )
        force_over = bool(
            values_finite
            and (
                force_norm > self.max_force_norm
                or lateral_force > self.max_lateral_force
            )
        )
        stable = bool(
            values_finite
            and force_norm <= self.stable_max_force_norm
            and lateral_force <= self.stable_max_lateral_force
            and arm_speed <= self.stable_max_arm_speed
        )
        now = float(timestamp)
        self.elapsed_time = now - self.start_time

        if force_over:
            if self.force_over_start_time is None:
                self.force_over_start_time = now
            self.force_over_time = now - self.force_over_start_time
        else:
            self.force_over_start_time = None
            self.force_over_time = 0.0
        self.max_force_over_time = max(
            self.max_force_over_time,
            self.force_over_time,
        )

        if stable:
            if self.stable_start_time is None:
                self.stable_start_time = now
            self.stable_time = now - self.stable_start_time
        else:
            self.stable_start_time = None
            self.stable_time = 0.0
        self.max_stable_time = max(self.max_stable_time, self.stable_time)

        just_completed = False
        tolerance = 1.0e-12
        # This ordering is deliberate and matches the collector: force abort,
        # stable acceptance, then timeout.
        if self.force_over_time + tolerance >= self.force_abort_dwell_time:
            self.complete = True
            self.success = False
            self.reason = "force_limit"
            just_completed = True
        elif (
            self.elapsed_time + tolerance >= self.min_time
            and stable
            and self.stable_time + tolerance >= self.stable_dwell_time
        ):
            self.complete = True
            self.success = True
            self.reason = "stable"
            just_completed = True
        elif self.elapsed_time + tolerance >= self.max_time:
            self.complete = True
            self.success = False
            self.reason = "stability_timeout"
            just_completed = True

        self.last_time = now
        return self._result(
            stable=stable,
            force_over=force_over,
            just_completed=just_completed,
        )

    def _result(
        self,
        *,
        stable: bool,
        force_over: bool,
        just_completed: bool,
    ) -> TerminalHoldUpdate:
        return TerminalHoldUpdate(
            active=self.active,
            stable=stable,
            force_over=force_over,
            elapsed_time=self.elapsed_time,
            stable_time=self.stable_time,
            force_over_time=self.force_over_time,
            complete=self.complete,
            success=self.success,
            reason=self.reason,
            just_completed=just_completed,
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
