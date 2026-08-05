import numpy as np
import pytest

from force_aware_act.inference import (
    CONTROL_POSTPROCESS_VERSION,
    DEFAULT_EMA_ALPHA,
    DEFAULT_FORCE_STOP_THRESHOLD,
    DEFAULT_MAX_DELTA_Q,
    DEFAULT_MAX_ROLLOUT_STEPS,
    DEFAULT_POLICY_RATE_HZ,
    DEFAULT_SAFE_FORCE_THRESHOLD,
    DEFAULT_SUCCESS_DISTANCE_THRESHOLD,
    DEFAULT_SUCCESS_DWELL_TIME,
    POLICY_STEP_SCHEDULER_VERSION,
    ROLLOUT_PROTOCOL_VERSION,
    TASK_SUCCESS_VERSION,
    CumulativePolicyStepScheduler,
    JointPositionPostprocessor,
    TaskSuccessTracker,
)
from scripts.run_mujoco_policy_rollout import (
    _fieldnames,
    _summarize_policy_intervals,
    parse_args,
)


def test_cumulative_scheduler_is_exact_over_one_second_at_30_hz():
    scheduler = CumulativePolicyStepScheduler(
        policy_rate_hz=30.0,
        physics_timestep=0.001,
    )

    counts = [scheduler.next_step_count() for _ in range(30)]

    assert set(counts) == {33, 34}
    assert sum(counts) == 1000
    assert scheduler.emitted_physics_steps == 1000
    assert scheduler.interval_count == 30


def test_cumulative_scheduler_never_accumulates_more_than_half_step_error():
    scheduler = CumulativePolicyStepScheduler(
        policy_rate_hz=30.0,
        physics_timestep=0.001,
    )

    cumulative = 0
    for interval in range(1, 601):
        cumulative += scheduler.next_step_count()
        ideal = interval / (30.0 * 0.001)
        assert abs(cumulative - ideal) <= 0.5 + 1.0e-12


def test_task_success_matches_collector_distance_and_dwell_semantics():
    tracker = TaskSuccessTracker(
        distance_threshold=0.003,
        dwell_time=0.10,
    )

    updates = [
        tracker.update(timestamp=step / 30.0, distance=0.003)
        for step in range(4)
    ]

    assert [update.condition for update in updates] == [True] * 4
    assert not any(update.success for update in updates[:3])
    assert updates[-1].success
    assert updates[-1].just_succeeded
    assert updates[-1].accumulated_time == pytest.approx(0.10)


def test_task_success_resets_dwell_after_distance_failure():
    tracker = TaskSuccessTracker(
        distance_threshold=0.003,
        dwell_time=0.10,
    )
    tracker.update(timestamp=0.0, distance=0.002)
    tracker.update(timestamp=0.05, distance=0.002)
    failed = tracker.update(timestamp=0.08, distance=0.0031)
    restarted = tracker.update(timestamp=0.10, distance=0.002)

    assert not failed.condition
    assert failed.accumulated_time == 0.0
    assert restarted.consecutive_observations == 1
    assert restarted.accumulated_time == 0.0
    assert not restarted.success


def test_default_postprocessor_has_no_ema_and_applies_safety_delta_clip():
    postprocessor = JointPositionPostprocessor(
        control_ranges=np.asarray([[-1.0, 1.0], [-1.0, 1.0]]),
        initial_command=np.zeros(2),
        max_delta_q=0.02,
        ema_alpha=1.0,
    )

    result = postprocessor.process(
        target=np.asarray([0.10, -0.01]),
        current_qpos=np.zeros(2),
    )

    np.testing.assert_allclose(result.delta_clipped, [0.02, -0.01])
    np.testing.assert_allclose(result.ema, result.delta_clipped)
    np.testing.assert_allclose(result.ctrlrange_clipped, result.delta_clipped)
    assert result.delta_clip_applied
    assert not result.ema_modified
    assert not result.ctrlrange_clip_applied


def test_postprocessor_ema_and_ctrlrange_are_ordered_and_auditable():
    postprocessor = JointPositionPostprocessor(
        control_ranges=np.asarray([[-0.01, 0.01]]),
        initial_command=np.asarray([0.0]),
        max_delta_q=0.10,
        ema_alpha=0.5,
    )

    result = postprocessor.process(
        target=np.asarray([0.04]),
        current_qpos=np.asarray([0.0]),
    )

    np.testing.assert_allclose(result.delta_clipped, [0.04])
    np.testing.assert_allclose(result.ema, [0.02])
    np.testing.assert_allclose(result.ctrlrange_clipped, [0.01])
    assert not result.delta_clip_applied
    assert result.ema_modified
    assert result.ctrlrange_clip_applied


def test_postprocessor_can_audit_rejected_command_without_committing_state():
    postprocessor = JointPositionPostprocessor(
        control_ranges=np.asarray([[-1.0, 1.0]]),
        initial_command=np.asarray([0.0]),
        max_delta_q=1.0,
        ema_alpha=0.5,
    )
    postprocessor.process(
        target=np.asarray([0.8]),
        current_qpos=np.asarray([0.0]),
        commit=False,
    )
    accepted = postprocessor.process(
        target=np.asarray([0.4]),
        current_qpos=np.asarray([0.0]),
    )

    np.testing.assert_allclose(accepted.ema, [0.2])


def test_rollout_cli_defaults_match_standard_protocol():
    args = parse_args(["--checkpoint", "model.pt", "--output-dir", "rollout"])

    assert args.policy_rate_hz == DEFAULT_POLICY_RATE_HZ
    assert args.max_rollout_steps == DEFAULT_MAX_ROLLOUT_STEPS
    assert args.ema_alpha == DEFAULT_EMA_ALPHA
    assert args.max_delta_q == DEFAULT_MAX_DELTA_Q
    assert args.force_stop_threshold == DEFAULT_FORCE_STOP_THRESHOLD
    assert args.safe_force_threshold == DEFAULT_SAFE_FORCE_THRESHOLD
    assert args.force_stop_threshold == 100.0
    assert args.safe_force_threshold == 40.0
    assert args.success_distance_threshold == DEFAULT_SUCCESS_DISTANCE_THRESHOLD
    assert args.success_dwell_time == DEFAULT_SUCCESS_DWELL_TIME
    assert args.success_lateral_threshold is None
    assert args.success_hold_steps is None
    assert ROLLOUT_PROTOCOL_VERSION == "paired_temporal_rollout_v1"
    assert POLICY_STEP_SCHEDULER_VERSION
    assert CONTROL_POSTPROCESS_VERSION
    assert TASK_SUCCESS_VERSION


def test_legacy_success_force_flag_is_only_an_alias_for_safe_force():
    args = parse_args(
        [
            "--checkpoint",
            "model.pt",
            "--output-dir",
            "rollout",
            "--success-force-threshold",
            "80",
        ]
    )

    assert args.safe_force_threshold == 80.0


def test_rollout_csv_schema_contains_protocol_diagnostics():
    fields = set(_fieldnames())

    assert {
        "scheduled_physics_steps_this_policy",
        "physics_steps_this_policy",
        "physics_interval_completed",
        "max_force_norm_during_policy_interval",
        "force_stop_time",
        "safety_hold_applied",
        "safety_hold_ctrl_0",
        "delta_clip_applied",
        "ema_modified",
        "ctrlrange_clip_applied",
        "success_hold_time",
    } <= fields


def test_partial_safety_stop_does_not_distort_completed_policy_rate():
    diagnostics = _summarize_policy_intervals(
        scheduled_steps=[33, 34, 33],
        executed_steps=[33, 34, 1],
        physics_timestep=0.001,
    )

    assert diagnostics["physics_steps_total"] == 68
    assert diagnostics["physics_intervals_completed"] == 2
    assert diagnostics["partial_physics_interval_steps"] == 1
    assert diagnostics["physics_steps_per_policy_min"] == 33
    assert diagnostics["physics_steps_per_policy_max"] == 34
    assert diagnostics["achieved_policy_rate_hz"] == pytest.approx(2 / 0.067)
