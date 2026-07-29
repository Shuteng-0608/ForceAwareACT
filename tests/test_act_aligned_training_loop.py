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
