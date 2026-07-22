"""Pure contact, recovery, and force-safety evaluation state machine."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping


def _finite_float(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite number") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _boolean(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


@dataclass(frozen=True)
class ContactRecoveryConfig:
    """Thresholds for contact debouncing, recovery, and safety metrics.

    Force comparisons follow the rollout convention: contact enters at
    ``>= contact_enter_force_n``, exits at ``<= contact_exit_force_n``, task
    success requires ``force < success_force_n``, safe success requires the
    complete-history peak to remain ``< safe_force_n``, and a hard violation
    occurs at ``force > hard_force_n``.
    """

    contact_enter_force_n: float = 5.0
    contact_exit_force_n: float = 3.0
    contact_min_steps: int = 2
    success_force_n: float = 80.0
    safe_force_n: float = 40.0
    hard_force_n: float = 1000.0
    success_hold_steps: int = 15

    def __post_init__(self) -> None:
        numeric_fields = (
            "contact_enter_force_n",
            "contact_exit_force_n",
            "success_force_n",
            "safe_force_n",
            "hard_force_n",
        )
        for field_name in numeric_fields:
            object.__setattr__(
                self,
                field_name,
                _finite_float(getattr(self, field_name), field_name),
            )
        object.__setattr__(
            self,
            "contact_min_steps",
            _positive_int(self.contact_min_steps, "contact_min_steps"),
        )
        object.__setattr__(
            self,
            "success_hold_steps",
            _positive_int(self.success_hold_steps, "success_hold_steps"),
        )

        if self.contact_exit_force_n < 0:
            raise ValueError("contact_exit_force_n must be non-negative")
        if not self.contact_exit_force_n < self.contact_enter_force_n:
            raise ValueError(
                "contact_exit_force_n must be smaller than contact_enter_force_n"
            )
        if not self.contact_enter_force_n < self.safe_force_n:
            raise ValueError("contact_enter_force_n must be smaller than safe_force_n")
        if self.safe_force_n > self.success_force_n:
            raise ValueError("safe_force_n must be no greater than success_force_n")
        if not self.success_force_n < self.hard_force_n:
            raise ValueError("success_force_n must be smaller than hard_force_n")

    def to_dict(self) -> dict[str, object]:
        """Return JSON-native scalar configuration values."""

        return {
            "contact_enter_force_n": self.contact_enter_force_n,
            "contact_exit_force_n": self.contact_exit_force_n,
            "contact_min_steps": self.contact_min_steps,
            "success_force_n": self.success_force_n,
            "safe_force_n": self.safe_force_n,
            "hard_force_n": self.hard_force_n,
            "success_hold_steps": self.success_hold_steps,
        }


@dataclass(frozen=True)
class ContactRecoveryStepState:
    """JSON-serializable scalar snapshot returned by one state-machine step."""

    step: int
    elapsed_time_s: float
    force_n: float
    force_excess_n: float
    contact_active: bool
    contact_transition: str
    contact_ever_confirmed: bool
    contact_enter_counter: int
    contact_exit_counter: int
    contact_event_count: int
    contact_duration_s: float
    peak_force_n: float
    force_excess_integral_n_s: float
    hard_force_this_step: bool
    hard_force_violation: bool
    task_success_condition: bool
    success_condition: bool
    success_hold_counter: int
    recovery_success: bool
    safe_success: bool

    def to_dict(self) -> dict[str, object]:
        return {
            field_name: getattr(self, field_name)
            for field_name in self.__dataclass_fields__
        }


class ContactRecoveryStateMachine:
    """Accumulate contact/recovery/safety metrics from sequential scalar inputs."""

    STATE_VERSION = 1

    def __init__(self, config: ContactRecoveryConfig) -> None:
        if not isinstance(config, ContactRecoveryConfig):
            raise ValueError("config must be a ContactRecoveryConfig")
        self.config = config
        self.reset()

    def reset(self) -> None:
        self._steps = 0
        self._elapsed_time_s = 0.0
        self._in_contact = False
        self._contact_ever_confirmed = False
        self._contact_enter_counter = 0
        self._contact_exit_counter = 0
        self._contact_event_count = 0
        self._first_contact_step = -1
        self._first_contact_time_s = -1.0
        self._contact_duration_s = 0.0
        self._peak_force_n = 0.0
        self._force_excess_integral_n_s = 0.0
        self._hard_force_violation = False
        self._hard_force_violation_step = -1
        self._hard_force_violation_time_s = -1.0
        self._success_hold_counter = 0
        self._max_success_hold_counter = 0
        self._recovery_success = False
        self._recovery_success_step = -1
        self._recovery_success_time_s = -1.0

    @property
    def safe_success(self) -> bool:
        return bool(
            self._recovery_success
            and self._peak_force_n < self.config.safe_force_n
            and not self._hard_force_violation
        )

    def _update_contact_state(self, force_n: float) -> str:
        transition = "none"
        if self._in_contact:
            self._contact_enter_counter = 0
            if force_n <= self.config.contact_exit_force_n:
                self._contact_exit_counter += 1
            else:
                self._contact_exit_counter = 0
            if self._contact_exit_counter >= self.config.contact_min_steps:
                self._in_contact = False
                self._contact_exit_counter = 0
                transition = "exited"
        else:
            self._contact_exit_counter = 0
            if force_n >= self.config.contact_enter_force_n:
                self._contact_enter_counter += 1
            else:
                self._contact_enter_counter = 0
            if self._contact_enter_counter >= self.config.contact_min_steps:
                self._in_contact = True
                self._contact_enter_counter = 0
                self._contact_ever_confirmed = True
                self._contact_event_count += 1
                if self._first_contact_step < 0:
                    self._first_contact_step = self._steps
                    self._first_contact_time_s = self._elapsed_time_s
                transition = "entered"
        return transition

    def update(
        self,
        *,
        force_n: float,
        dt_s: float,
        task_success: bool,
    ) -> ContactRecoveryStepState:
        """Advance one step, rejecting non-finite force/time and negative ``dt``."""

        force_n = _finite_float(force_n, "force_n")
        dt_s = _finite_float(dt_s, "dt_s")
        task_success = _boolean(task_success, "task_success")
        if force_n < 0:
            raise ValueError("force_n must be non-negative")
        if dt_s < 0:
            raise ValueError("dt_s must be non-negative")

        force_excess_n = max(0.0, force_n - self.config.safe_force_n)
        next_elapsed_time_s = self._elapsed_time_s + dt_s
        next_contact_duration_s = self._contact_duration_s + dt_s
        next_force_excess_integral = (
            self._force_excess_integral_n_s + force_excess_n * dt_s
        )
        if not all(
            math.isfinite(value)
            for value in (
                next_elapsed_time_s,
                next_contact_duration_s,
                next_force_excess_integral,
            )
        ):
            raise ValueError("accumulated contact metrics must remain finite")

        self._steps += 1
        self._elapsed_time_s = next_elapsed_time_s
        self._peak_force_n = max(self._peak_force_n, force_n)
        self._force_excess_integral_n_s = next_force_excess_integral

        hard_force_this_step = force_n > self.config.hard_force_n
        if hard_force_this_step and not self._hard_force_violation:
            self._hard_force_violation = True
            self._hard_force_violation_step = self._steps
            self._hard_force_violation_time_s = self._elapsed_time_s

        transition = self._update_contact_state(force_n)
        if self._in_contact:
            self._contact_duration_s = next_contact_duration_s

        success_condition = bool(
            self._contact_ever_confirmed
            and task_success
            and force_n < self.config.success_force_n
        )
        if success_condition:
            self._success_hold_counter += 1
        else:
            self._success_hold_counter = 0
        self._max_success_hold_counter = max(
            self._max_success_hold_counter, self._success_hold_counter
        )
        if (
            not self._recovery_success
            and self._success_hold_counter >= self.config.success_hold_steps
        ):
            self._recovery_success = True
            self._recovery_success_step = self._steps
            self._recovery_success_time_s = self._elapsed_time_s

        return ContactRecoveryStepState(
            step=self._steps,
            elapsed_time_s=self._elapsed_time_s,
            force_n=force_n,
            force_excess_n=force_excess_n,
            contact_active=self._in_contact,
            contact_transition=transition,
            contact_ever_confirmed=self._contact_ever_confirmed,
            contact_enter_counter=self._contact_enter_counter,
            contact_exit_counter=self._contact_exit_counter,
            contact_event_count=self._contact_event_count,
            contact_duration_s=self._contact_duration_s,
            peak_force_n=self._peak_force_n,
            force_excess_integral_n_s=self._force_excess_integral_n_s,
            hard_force_this_step=hard_force_this_step,
            hard_force_violation=self._hard_force_violation,
            task_success_condition=task_success,
            success_condition=success_condition,
            success_hold_counter=self._success_hold_counter,
            recovery_success=self._recovery_success,
            safe_success=self.safe_success,
        )

    def summary(self) -> dict[str, object]:
        """Return a flat mapping containing JSON-native scalar values only."""

        contact_fraction = (
            self._contact_duration_s / self._elapsed_time_s
            if self._elapsed_time_s > 0
            else 0.0
        )
        return {
            "steps": self._steps,
            "elapsed_time_s": self._elapsed_time_s,
            "contact_active_final": self._in_contact,
            "contact_ever_confirmed": self._contact_ever_confirmed,
            "contact_event_count": self._contact_event_count,
            "first_contact_step": self._first_contact_step,
            "first_contact_time_s": self._first_contact_time_s,
            "contact_duration_s": self._contact_duration_s,
            "contact_fraction": contact_fraction,
            "peak_force_n": self._peak_force_n,
            "force_excess_integral_n_s": self._force_excess_integral_n_s,
            "hard_force_violation": self._hard_force_violation,
            "hard_force_violation_step": self._hard_force_violation_step,
            "hard_force_violation_time_s": self._hard_force_violation_time_s,
            "success_hold_steps_observed": self._max_success_hold_counter,
            "recovery_success": self._recovery_success,
            "recovery_success_step": self._recovery_success_step,
            "recovery_success_time_s": self._recovery_success_time_s,
            "safe_success": self.safe_success,
            "contact_enter_force_n": self.config.contact_enter_force_n,
            "contact_exit_force_n": self.config.contact_exit_force_n,
            "contact_min_steps": self.config.contact_min_steps,
            "success_force_n": self.config.success_force_n,
            "safe_force_n": self.config.safe_force_n,
            "hard_force_n": self.config.hard_force_n,
            "success_hold_steps": self.config.success_hold_steps,
        }

    def state_dict(self) -> dict[str, object]:
        """Return complete exact-resume state using flat JSON-native scalars."""

        return {
            "version": self.STATE_VERSION,
            **self.config.to_dict(),
            "steps": self._steps,
            "elapsed_time_s": self._elapsed_time_s,
            "in_contact": self._in_contact,
            "contact_ever_confirmed": self._contact_ever_confirmed,
            "contact_enter_counter": self._contact_enter_counter,
            "contact_exit_counter": self._contact_exit_counter,
            "contact_event_count": self._contact_event_count,
            "first_contact_step": self._first_contact_step,
            "first_contact_time_s": self._first_contact_time_s,
            "contact_duration_s": self._contact_duration_s,
            "peak_force_n": self._peak_force_n,
            "force_excess_integral_n_s": self._force_excess_integral_n_s,
            "hard_force_violation": self._hard_force_violation,
            "hard_force_violation_step": self._hard_force_violation_step,
            "hard_force_violation_time_s": self._hard_force_violation_time_s,
            "success_hold_counter": self._success_hold_counter,
            "max_success_hold_counter": self._max_success_hold_counter,
            "recovery_success": self._recovery_success,
            "recovery_success_step": self._recovery_success_step,
            "recovery_success_time_s": self._recovery_success_time_s,
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        """Restore state after validating configuration and scalar invariants."""

        if not isinstance(state, Mapping) or state.get("version") != self.STATE_VERSION:
            raise ValueError("unsupported contact recovery state")
        for key, expected in self.config.to_dict().items():
            if state.get(key) != expected:
                raise ValueError(
                    f"contact recovery state {key} mismatch: "
                    f"checkpoint={state.get(key)!r} current={expected!r}"
                )

        steps = _nonnegative_int(state.get("steps"), "state steps")
        elapsed_time_s = _finite_float(state.get("elapsed_time_s"), "state elapsed_time_s")
        contact_duration_s = _finite_float(
            state.get("contact_duration_s"), "state contact_duration_s"
        )
        peak_force_n = _finite_float(state.get("peak_force_n"), "state peak_force_n")
        force_excess_integral_n_s = _finite_float(
            state.get("force_excess_integral_n_s"),
            "state force_excess_integral_n_s",
        )
        if min(
            elapsed_time_s,
            contact_duration_s,
            peak_force_n,
            force_excess_integral_n_s,
        ) < 0:
            raise ValueError("contact recovery state accumulated metrics must be non-negative")
        if contact_duration_s > elapsed_time_s + 1.0e-12:
            raise ValueError("state contact_duration_s cannot exceed elapsed_time_s")

        enter_counter = _nonnegative_int(
            state.get("contact_enter_counter"), "state contact_enter_counter"
        )
        exit_counter = _nonnegative_int(
            state.get("contact_exit_counter"), "state contact_exit_counter"
        )
        if enter_counter >= self.config.contact_min_steps:
            raise ValueError("state contact_enter_counter must be below contact_min_steps")
        if exit_counter >= self.config.contact_min_steps:
            raise ValueError("state contact_exit_counter must be below contact_min_steps")

        contact_event_count = _nonnegative_int(
            state.get("contact_event_count"), "state contact_event_count"
        )
        success_hold_counter = _nonnegative_int(
            state.get("success_hold_counter"), "state success_hold_counter"
        )
        max_success_hold_counter = _nonnegative_int(
            state.get("max_success_hold_counter"), "state max_success_hold_counter"
        )
        if success_hold_counter > max_success_hold_counter:
            raise ValueError(
                "state success_hold_counter cannot exceed max_success_hold_counter"
            )
        if max_success_hold_counter > steps:
            raise ValueError("state success hold counters cannot exceed steps")

        in_contact = _boolean(state.get("in_contact"), "state in_contact")
        contact_ever_confirmed = _boolean(
            state.get("contact_ever_confirmed"), "state contact_ever_confirmed"
        )
        hard_force_violation = _boolean(
            state.get("hard_force_violation"), "state hard_force_violation"
        )
        recovery_success = _boolean(
            state.get("recovery_success"), "state recovery_success"
        )
        if in_contact and not contact_ever_confirmed:
            raise ValueError("state cannot be in contact before contact confirmation")
        if (contact_event_count > 0) != contact_ever_confirmed:
            raise ValueError("state contact events must agree with contact confirmation")
        if contact_event_count > steps:
            raise ValueError("state contact_event_count cannot exceed steps")
        if recovery_success and not contact_ever_confirmed:
            raise ValueError("state recovery success requires contact confirmation")
        if in_contact and enter_counter != 0:
            raise ValueError("state contact_enter_counter must be zero while in contact")
        if not in_contact and exit_counter != 0:
            raise ValueError("state contact_exit_counter must be zero outside contact")
        if hard_force_violation != (peak_force_n > self.config.hard_force_n):
            raise ValueError("state hard violation must agree with peak_force_n")
        if recovery_success and max_success_hold_counter < self.config.success_hold_steps:
            raise ValueError("state recovery success requires the configured success hold")
        if not recovery_success and max_success_hold_counter >= self.config.success_hold_steps:
            raise ValueError("state success hold requires recovery_success")

        first_contact_step = self._validated_event_step(
            state.get("first_contact_step"), steps, "first_contact_step"
        )
        hard_step = self._validated_event_step(
            state.get("hard_force_violation_step"),
            steps,
            "hard_force_violation_step",
        )
        recovery_step = self._validated_event_step(
            state.get("recovery_success_step"), steps, "recovery_success_step"
        )
        first_contact_time = self._validated_event_time(
            state.get("first_contact_time_s"), elapsed_time_s, "first_contact_time_s"
        )
        hard_time = self._validated_event_time(
            state.get("hard_force_violation_time_s"),
            elapsed_time_s,
            "hard_force_violation_time_s",
        )
        recovery_time = self._validated_event_time(
            state.get("recovery_success_time_s"),
            elapsed_time_s,
            "recovery_success_time_s",
        )
        if (first_contact_step >= 0) != contact_ever_confirmed:
            raise ValueError("state first contact marker disagrees with contact confirmation")
        if (hard_step >= 0) != hard_force_violation:
            raise ValueError("state hard violation marker disagrees with violation flag")
        if (recovery_step >= 0) != recovery_success:
            raise ValueError("state recovery marker disagrees with recovery flag")
        for marker_name, marker_step, marker_time in (
            ("first contact", first_contact_step, first_contact_time),
            ("hard violation", hard_step, hard_time),
            ("recovery success", recovery_step, recovery_time),
        ):
            if (marker_step >= 0) != (marker_time >= 0):
                raise ValueError(f"state {marker_name} step/time markers disagree")

        self._steps = steps
        self._elapsed_time_s = elapsed_time_s
        self._in_contact = in_contact
        self._contact_ever_confirmed = contact_ever_confirmed
        self._contact_enter_counter = enter_counter
        self._contact_exit_counter = exit_counter
        self._contact_event_count = contact_event_count
        self._first_contact_step = first_contact_step
        self._first_contact_time_s = first_contact_time
        self._contact_duration_s = contact_duration_s
        self._peak_force_n = peak_force_n
        self._force_excess_integral_n_s = force_excess_integral_n_s
        self._hard_force_violation = hard_force_violation
        self._hard_force_violation_step = hard_step
        self._hard_force_violation_time_s = hard_time
        self._success_hold_counter = success_hold_counter
        self._max_success_hold_counter = max_success_hold_counter
        self._recovery_success = recovery_success
        self._recovery_success_step = recovery_step
        self._recovery_success_time_s = recovery_time

    @staticmethod
    def _validated_event_step(value: object, steps: int, name: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"state {name} must be an integer")
        if value < -1 or value > steps or value == 0:
            raise ValueError(f"state {name} must be -1 or in [1, steps]")
        return value

    @staticmethod
    def _validated_event_time(value: object, elapsed_time_s: float, name: str) -> float:
        result = _finite_float(value, f"state {name}")
        if result != -1.0 and not 0.0 <= result <= elapsed_time_s + 1.0e-12:
            raise ValueError(f"state {name} must be -1 or within elapsed time")
        return result
