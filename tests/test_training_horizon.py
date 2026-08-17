import pytest

from force_aware_act.training import resolve_training_horizon


def test_default_horizon_uses_original_checkpoint_limit():
    horizon = resolve_training_horizon(
        source_optimizer_step_limit=25_000,
        start_global_step=20_000,
        requested_target_optimizer_steps=None,
    )

    assert horizon.target_optimizer_steps == 25_000
    assert not horizon.is_extension


def test_completed_checkpoint_requires_an_explicit_larger_target():
    with pytest.raises(ValueError, match="explicit larger target"):
        resolve_training_horizon(
            source_optimizer_step_limit=25_000,
            start_global_step=25_000,
            requested_target_optimizer_steps=None,
        )


def test_horizon_can_extend_a_completed_checkpoint():
    horizon = resolve_training_horizon(
        source_optimizer_step_limit=25_000,
        start_global_step=25_000,
        requested_target_optimizer_steps=50_000,
    )

    assert horizon.target_optimizer_steps == 50_000
    assert horizon.is_extension
    assert horizon.to_dict()["start_global_step"] == 25_000


def test_official_horizon_requires_whole_epochs():
    horizon = resolve_training_horizon(
        source_optimizer_step_limit=25_000,
        start_global_step=25_000,
        requested_target_optimizer_steps=50_000,
        steps_per_epoch=5,
    )

    assert horizon.target_epochs == 10_000

    with pytest.raises(ValueError, match="divisible"):
        resolve_training_horizon(
            source_optimizer_step_limit=25_000,
            start_global_step=25_000,
            requested_target_optimizer_steps=50_001,
            steps_per_epoch=5,
        )
