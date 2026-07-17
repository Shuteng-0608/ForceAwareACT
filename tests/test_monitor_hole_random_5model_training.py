from datetime import datetime, timedelta, timezone

import pytest

from scripts.monitor_hole_random_5model_training import (
    MODEL_NAMES,
    ModelProgress,
    estimate_pipeline_eta,
    plateau_stop_epoch,
)


def progress(
    name: str,
    *,
    status: str = "queued",
    step: int | None = None,
    validation_epoch: int | None = None,
    patience_used: int | None = None,
    started_at: datetime | None = None,
) -> ModelProgress:
    return ModelProgress(
        name=name,
        status=status,
        step=step,
        epoch=validation_epoch,
        batch_in_epoch=None,
        steps_per_epoch=100,
        loss_total=None,
        deploy_loss=None,
        validation_epoch=validation_epoch,
        best_metric=None,
        best_epoch=None,
        patience_used=patience_used,
        deployment_mode=None,
        stop_reason=None,
        started_at=started_at,
        finished_at=None,
    )


def test_plateau_stop_epoch_matches_early_stopping_counter() -> None:
    queued = progress(MODEL_NAMES[0])
    assert plateau_stop_epoch(
        queued,
        min_epochs=20,
        patience=10,
        max_epochs=100,
    ) == 29

    one_strike_at_minimum = progress(
        MODEL_NAMES[0],
        validation_epoch=20,
        patience_used=1,
    )
    assert plateau_stop_epoch(
        one_strike_at_minimum,
        min_epochs=20,
        patience=10,
        max_epochs=100,
    ) == 29

    improvement_at_minimum = progress(
        MODEL_NAMES[0],
        validation_epoch=20,
        patience_used=0,
    )
    assert plateau_stop_epoch(
        improvement_at_minimum,
        min_epochs=20,
        patience=10,
        max_epochs=100,
    ) == 30


def test_pipeline_eta_reports_plateau_and_hard_budget_bounds() -> None:
    now = datetime(2026, 7, 16, 12, 0, tzinfo=timezone.utc)
    models = [
        progress(
            MODEL_NAMES[0],
            status="running",
            step=100,
            started_at=now - timedelta(seconds=10),
        )
    ] + [progress(name) for name in MODEL_NAMES[1:]]

    estimate = estimate_pipeline_eta(
        models,
        now=now,
        max_steps=10_000,
        max_epochs=100,
        min_epochs=20,
        patience=10,
        eta_min_steps=100,
    )

    assert estimate is not None
    assert estimate.seconds_per_step == pytest.approx(0.1)
    assert estimate.observed_steps == 100
    assert estimate.plateau_remaining_steps == 14_400
    assert estimate.hard_remaining_steps == 49_900
    assert estimate.plateau_seconds == pytest.approx(1_440)
    assert estimate.hard_seconds == pytest.approx(4_990)


def test_pipeline_eta_waits_for_minimum_observation_window() -> None:
    now = datetime(2026, 7, 16, 12, 0, tzinfo=timezone.utc)
    models = [
        progress(
            MODEL_NAMES[0],
            status="running",
            step=99,
            started_at=now - timedelta(seconds=10),
        )
    ] + [progress(name) for name in MODEL_NAMES[1:]]

    assert (
        estimate_pipeline_eta(
            models,
            now=now,
            max_steps=10_000,
            max_epochs=100,
            min_epochs=20,
            patience=10,
            eta_min_steps=100,
        )
        is None
    )
