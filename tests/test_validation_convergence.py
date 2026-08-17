import pytest

from force_aware_act.training import (
    ValidationConvergenceMonitor,
    resolve_convergence_monitor,
)


def test_small_best_improvements_do_not_reset_plateau_patience():
    monitor = ValidationConvergenceMonitor(
        metric_name="action_l1",
        minimum_optimizer_steps=100,
        patience_validations=2,
        min_relative_improvement=0.01,
    )

    first = monitor.update(1.0, global_step=50)
    small = monitor.update(0.995, global_step=100)
    stop = monitor.update(0.994, global_step=110)

    assert first.checkpoint_improved
    assert small.checkpoint_improved
    assert not small.meaningful_improvement
    assert stop.should_stop
    assert monitor.best_metric == pytest.approx(0.994)
    assert monitor.best_step == 110


def test_meaningful_improvement_resets_patience():
    monitor = ValidationConvergenceMonitor(
        metric_name="action_l1",
        minimum_optimizer_steps=0,
        patience_validations=2,
        min_relative_improvement=0.01,
    )
    monitor.update(1.0, global_step=1)
    monitor.update(0.995, global_step=2)
    improved = monitor.update(0.98, global_step=3)

    assert improved.meaningful_improvement
    assert monitor.validations_without_meaningful_improvement == 0


def test_patience_does_not_accumulate_before_minimum_steps():
    monitor = ValidationConvergenceMonitor(
        metric_name="action_l1",
        minimum_optimizer_steps=100,
        patience_validations=1,
        min_relative_improvement=0.01,
    )
    monitor.update(1.0, global_step=10)
    update = monitor.update(1.0, global_step=99)

    assert not update.should_stop
    assert monitor.validations_without_meaningful_improvement == 0


def test_resume_protocol_is_strict_and_restores_state():
    original = ValidationConvergenceMonitor(
        metric_name="action_l1",
        minimum_optimizer_steps=100,
        patience_validations=3,
        min_relative_improvement=0.01,
    )
    original.update(1.0, global_step=100)

    restored = resolve_convergence_monitor(
        metric_name="action_l1",
        minimum_optimizer_steps=None,
        patience_validations=None,
        min_relative_improvement=0.01,
        default_minimum_optimizer_steps=100,
        prior_state=original.to_dict(),
    )

    assert restored is not None
    assert restored.to_dict() == original.to_dict()
    with pytest.raises(ValueError, match="does not match checkpoint"):
        resolve_convergence_monitor(
            metric_name="action_l1",
            minimum_optimizer_steps=100,
            patience_validations=4,
            min_relative_improvement=0.01,
            default_minimum_optimizer_steps=100,
            prior_state=original.to_dict(),
        )


def test_metric_history_can_be_rebased_when_best_artifact_is_missing():
    monitor = ValidationConvergenceMonitor(
        metric_name="action_l1",
        minimum_optimizer_steps=100,
        patience_validations=3,
        min_relative_improvement=0.01,
    )
    monitor.update(1.0, global_step=100)

    monitor.reset_observations()
    update = monitor.update(0.5, global_step=100)

    assert update.checkpoint_improved
    assert monitor.best_metric == 0.5
    assert monitor.best_step == 100
    assert monitor.validations_seen == 1
