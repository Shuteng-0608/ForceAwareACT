import json

import pytest

from force_aware_act.evaluation import (
    ContactRecoveryConfig,
    ContactRecoveryStateMachine,
)


def _config(**overrides):
    values = {
        "contact_enter_force_n": 5.0,
        "contact_exit_force_n": 3.0,
        "contact_min_steps": 2,
        "success_force_n": 80.0,
        "safe_force_n": 40.0,
        "hard_force_n": 100.0,
        "success_hold_steps": 2,
    }
    values.update(overrides)
    return ContactRecoveryConfig(**values)


def test_contact_hysteresis_and_min_steps_debounce_enter_and_exit():
    machine = ContactRecoveryStateMachine(_config())

    assert machine.update(force_n=5.5, dt_s=1.0, task_success=False).contact_active is False
    assert machine.update(force_n=4.0, dt_s=1.0, task_success=False).contact_enter_counter == 0
    assert machine.update(force_n=6.0, dt_s=1.0, task_success=False).contact_active is False
    entered = machine.update(force_n=7.0, dt_s=1.0, task_success=False)
    assert entered.contact_active is True
    assert entered.contact_transition == "entered"

    assert machine.update(force_n=4.0, dt_s=1.0, task_success=False).contact_active is True
    assert machine.update(force_n=2.0, dt_s=1.0, task_success=False).contact_active is True
    exited = machine.update(force_n=3.0, dt_s=1.0, task_success=False)

    assert exited.contact_active is False
    assert exited.contact_transition == "exited"
    assert exited.contact_event_count == 1
    assert exited.contact_duration_s == pytest.approx(3.0)
    assert machine.summary()["first_contact_step"] == 4


def test_force_metrics_track_peak_excess_integral_and_hard_violation():
    machine = ContactRecoveryStateMachine(_config(contact_min_steps=1))

    machine.update(force_n=10.0, dt_s=0.1, task_success=False)
    machine.update(force_n=50.0, dt_s=0.2, task_success=False)
    hard = machine.update(force_n=120.0, dt_s=0.1, task_success=False)
    summary = machine.summary()

    assert hard.hard_force_this_step is True
    assert summary["peak_force_n"] == pytest.approx(120.0)
    assert summary["force_excess_integral_n_s"] == pytest.approx(10.0)
    assert summary["hard_force_violation"] is True
    assert summary["hard_force_violation_step"] == 3


def test_recovery_success_requires_confirmed_contact_then_full_success_hold():
    machine = ContactRecoveryStateMachine(_config(contact_min_steps=1, success_hold_steps=2))

    machine.update(force_n=1.0, dt_s=0.1, task_success=True)
    before_contact = machine.update(force_n=1.0, dt_s=0.1, task_success=True)
    assert before_contact.success_hold_counter == 0
    assert before_contact.recovery_success is False

    contact = machine.update(force_n=6.0, dt_s=0.1, task_success=False)
    assert contact.contact_ever_confirmed is True
    first_success = machine.update(force_n=2.0, dt_s=0.1, task_success=True)
    recovery = machine.update(force_n=2.0, dt_s=0.1, task_success=True)

    assert first_success.success_hold_counter == 1
    assert first_success.recovery_success is False
    assert recovery.success_hold_counter == 2
    assert recovery.recovery_success is True
    assert recovery.safe_success is True
    assert machine.summary()["recovery_success_step"] == 5


def test_success_force_and_history_safe_force_have_distinct_roles():
    machine = ContactRecoveryStateMachine(_config(contact_min_steps=1, success_hold_steps=2))
    machine.update(force_n=50.0, dt_s=0.1, task_success=False)

    too_high_for_success = machine.update(force_n=85.0, dt_s=0.1, task_success=True)
    assert too_high_for_success.success_condition is False
    machine.update(force_n=70.0, dt_s=0.1, task_success=True)
    recovery = machine.update(force_n=70.0, dt_s=0.1, task_success=True)

    assert recovery.recovery_success is True
    assert recovery.safe_success is False
    assert recovery.hard_force_violation is False


def test_hard_threshold_uses_strict_rollout_convention():
    machine = ContactRecoveryStateMachine(_config(contact_min_steps=1))

    at_threshold = machine.update(force_n=100.0, dt_s=0.1, task_success=False)
    above_threshold = machine.update(force_n=100.1, dt_s=0.1, task_success=False)

    assert at_threshold.hard_force_violation is False
    assert above_threshold.hard_force_violation is True


@pytest.mark.parametrize(
    ("force_n", "dt_s", "match"),
    [
        (float("nan"), 0.1, "force_n must be finite"),
        (float("inf"), 0.1, "force_n must be finite"),
        (1.0, float("nan"), "dt_s must be finite"),
        (1.0, -0.1, "dt_s must be non-negative"),
        (-1.0, 0.1, "force_n must be non-negative"),
    ],
)
def test_invalid_force_or_dt_fails_explicitly(force_n, dt_s, match):
    machine = ContactRecoveryStateMachine(_config())

    with pytest.raises(ValueError, match=match):
        machine.update(force_n=force_n, dt_s=dt_s, task_success=False)


def test_step_state_summary_and_checkpoint_are_flat_json_scalars():
    machine = ContactRecoveryStateMachine(_config(contact_min_steps=1))
    step = machine.update(force_n=6.0, dt_s=0.1, task_success=False)

    for payload in (step.to_dict(), machine.summary(), machine.state_dict()):
        assert all(isinstance(key, str) for key in payload)
        assert all(isinstance(value, (str, int, float, bool)) for value in payload.values())
        json.dumps(payload, allow_nan=False)


def test_state_round_trip_continues_identically():
    config = _config(contact_min_steps=1, success_hold_steps=3)
    original = ContactRecoveryStateMachine(config)
    original.update(force_n=6.0, dt_s=0.1, task_success=False)
    original.update(force_n=2.0, dt_s=0.1, task_success=True)

    restored = ContactRecoveryStateMachine(config)
    restored.load_state_dict(json.loads(json.dumps(original.state_dict())))

    original_step = original.update(force_n=2.0, dt_s=0.1, task_success=True)
    restored_step = restored.update(force_n=2.0, dt_s=0.1, task_success=True)
    assert restored_step == original_step
    assert restored.summary() == original.summary()


def test_load_rejects_mismatched_configuration():
    state = ContactRecoveryStateMachine(_config()).state_dict()
    other = ContactRecoveryStateMachine(_config(safe_force_n=30.0))

    with pytest.raises(ValueError, match="safe_force_n mismatch"):
        other.load_state_dict(state)


def test_load_rejects_nonfinite_or_inconsistent_accumulated_state():
    machine = ContactRecoveryStateMachine(_config(contact_min_steps=1))
    machine.update(force_n=6.0, dt_s=0.1, task_success=False)
    state = machine.state_dict()
    state["first_contact_time_s"] = -0.5

    with pytest.raises(ValueError, match="first_contact_time_s must be -1"):
        ContactRecoveryStateMachine(machine.config).load_state_dict(state)

    state = machine.state_dict()
    state["elapsed_time_s"] = float("nan")
    with pytest.raises(ValueError, match="elapsed_time_s must be finite"):
        ContactRecoveryStateMachine(machine.config).load_state_dict(state)


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"contact_exit_force_n": 5.0}, "smaller than contact_enter"),
        ({"safe_force_n": 4.0}, "contact_enter_force_n must be smaller"),
        ({"safe_force_n": 90.0}, "no greater than success_force"),
        ({"hard_force_n": 80.0}, "success_force_n must be smaller"),
        ({"contact_min_steps": 0}, "positive integer"),
        ({"success_hold_steps": 0}, "positive integer"),
    ],
)
def test_invalid_threshold_configuration_is_rejected(overrides, match):
    with pytest.raises(ValueError, match=match):
        _config(**overrides)
