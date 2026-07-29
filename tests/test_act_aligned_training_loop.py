import pytest
import torch

pytest.importorskip("torchvision")

from force_aware_act.act_aligned_training import (  # noqa: E402
    ACTAlignedBatch,
    ACTAlignedCriterion,
    ACTAlignedTrainingConfig,
    build_act_aligned_optimizer,
    run_training_epoch,
    run_validation_epoch,
)
from force_aware_act.models.act_aligned import (  # noqa: E402
    ACTAlignedConfig,
    ACTAlignedContactCVAEPolicy,
)


def _config():
    return ACTAlignedConfig(
        d_model=32,
        nhead=4,
        dim_feedforward=64,
        dropout=0.0,
        chunk_len=3,
        force_window_len=2,
        image_height=32,
        image_width=32,
        pretrained_backbone=False,
        imagenet_normalize=False,
    )


def _batch(config, padding):
    batch_size = len(padding)
    return ACTAlignedBatch(
        images=torch.randn(batch_size, 2, 3, 32, 32),
        qpos=torch.randn(batch_size, 7),
        force_history=torch.randn(batch_size, 2, 6),
        force_padding_mask=torch.zeros(batch_size, 2, dtype=torch.bool),
        action_chunk=torch.randn(batch_size, 3, 7),
        future_force_chunk=torch.randn(batch_size, 3, 6),
        future_padding_mask=torch.tensor(padding, dtype=torch.bool),
    )


def test_epoch_loops_train_and_report_both_deployment_modes():
    model_config = _config()
    training_config = ACTAlignedTrainingConfig()
    model = ACTAlignedContactCVAEPolicy(model_config)
    criterion = ACTAlignedCriterion(training_config)
    optimizer = build_act_aligned_optimizer(model, training_config)
    batches = [
        _batch(model_config, [[False, False, False], [False, True, True]]),
        _batch(model_config, [[False, False, True]]),
    ]

    train_metrics = run_training_epoch(
        model,
        criterion,
        optimizer,
        batches,
        training_config,
        device=torch.device("cpu"),
    )
    validation_metrics = run_validation_epoch(
        model,
        criterion,
        batches,
        training_config,
        device=torch.device("cpu"),
    )

    assert train_metrics["optimizer_steps"] == 2.0
    assert train_metrics["loss_total"] > 0
    assert validation_metrics["deployment_zero_action_l1"] >= 0
    assert validation_metrics["deployment_prior_action_l1"] >= 0
    assert validation_metrics["posterior_kl_standard"] >= 0
    assert validation_metrics["posterior_zero_action_delta"] >= 0
    assert validation_metrics["prior_zero_force_delta"] >= 0
    assert (
        validation_metrics["posterior_mean_across_sample_variance"] >= 0
    )
    assert validation_metrics["prior_mean_across_sample_variance"] >= 0


def test_training_epoch_can_stop_and_resume_at_an_exact_batch_offset():
    model_config = _config()
    training_config = ACTAlignedTrainingConfig()
    model = ACTAlignedContactCVAEPolicy(model_config)
    criterion = ACTAlignedCriterion(training_config)
    optimizer = build_act_aligned_optimizer(model, training_config)
    batches = [
        _batch(model_config, [[False, False, False]])
        for _ in range(3)
    ]
    callback_steps = []

    first = run_training_epoch(
        model,
        criterion,
        optimizer,
        batches,
        training_config,
        device=torch.device("cpu"),
        max_optimizer_steps=1,
        step_callback=lambda step, metrics: callback_steps.append(
            (step, metrics["loss_total"])
        ),
    )
    second = run_training_epoch(
        model,
        criterion,
        optimizer,
        batches,
        training_config,
        device=torch.device("cpu"),
        max_optimizer_steps=2,
        skip_batches=1,
    )

    assert first["optimizer_steps"] == 1.0
    assert first["batches_skipped"] == 0.0
    assert second["optimizer_steps"] == 2.0
    assert second["batches_skipped"] == 1.0
    assert callback_steps[0][0] == 1
    assert callback_steps[0][1] > 0
    for name in (
        "posterior_mean_abs",
        "prior_mean_abs",
        "posterior_std_mean",
        "prior_std_mean",
        "posterior_prior_mean_l1",
    ):
        assert first[name] >= 0


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"max_optimizer_steps": 0}, "max_optimizer_steps"),
        ({"skip_batches": -1}, "skip_batches"),
    ],
)
def test_training_epoch_rejects_invalid_step_controls(kwargs, message):
    model_config = _config()
    training_config = ACTAlignedTrainingConfig()
    model = ACTAlignedContactCVAEPolicy(model_config)
    criterion = ACTAlignedCriterion(training_config)
    optimizer = build_act_aligned_optimizer(model, training_config)

    with pytest.raises(ValueError, match=message):
        run_training_epoch(
            model,
            criterion,
            optimizer,
            [_batch(model_config, [[False, False, False]])],
            training_config,
            device=torch.device("cpu"),
            **kwargs,
        )
