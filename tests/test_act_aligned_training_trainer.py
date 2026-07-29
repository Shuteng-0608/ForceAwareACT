from unittest.mock import patch

import pytest
import torch

pytest.importorskip("torchvision")

from force_aware_act.act_aligned_training import (  # noqa: E402
    ACTAlignedBatch,
    ACTAlignedCriterion,
    ACTAlignedTrainingConfig,
    build_act_aligned_optimizer,
    evaluate_one_batch,
    train_one_step,
)
from force_aware_act.models.act_aligned import (  # noqa: E402
    ACTAlignedConfig,
    ACTAlignedContactCVAEPolicy,
)


def _model_config():
    return ACTAlignedConfig(
        d_model=32,
        nhead=4,
        dim_feedforward=64,
        dropout=0.0,
        chunk_len=6,
        force_window_len=5,
        image_height=64,
        image_width=64,
        pretrained_backbone=False,
        imagenet_normalize=False,
    )


def _batch(config, batch_size=2):
    future_padding_mask = torch.zeros(
        batch_size,
        config.chunk_len,
        dtype=torch.bool,
    )
    future_padding_mask[:, -1] = True
    return ACTAlignedBatch(
        images=torch.randn(
            batch_size,
            config.num_cameras,
            3,
            config.image_height,
            config.image_width,
        ),
        qpos=torch.randn(batch_size, config.q_dim),
        force_history=torch.randn(
            batch_size,
            config.force_window_len,
            config.force_dim,
        ),
        force_padding_mask=torch.zeros(
            batch_size,
            config.force_window_len,
            dtype=torch.bool,
        ),
        action_chunk=torch.randn(
            batch_size,
            config.chunk_len,
            config.action_dim,
        ),
        future_force_chunk=torch.randn(
            batch_size,
            config.chunk_len,
            config.force_dim,
        ),
        future_padding_mask=future_padding_mask,
    )


def test_one_training_step_updates_backbone_and_prediction_head():
    torch.manual_seed(17)
    model_config = _model_config()
    training_config = ACTAlignedTrainingConfig()
    model = ACTAlignedContactCVAEPolicy(model_config)
    criterion = ACTAlignedCriterion(training_config)
    optimizer = build_act_aligned_optimizer(model, training_config)
    batch = _batch(model_config)
    backbone_before = model.vision_backbone.body[0].weight.detach().clone()
    action_head_before = model.action_head.weight.detach().clone()
    posterior_before = model.contact_posterior.mean_head.weight.detach().clone()
    prior_before = model.contact_prior.mean_head.weight.detach().clone()

    metrics = train_one_step(
        model,
        criterion,
        optimizer,
        batch,
        training_config,
    )

    assert not torch.equal(
        model.vision_backbone.body[0].weight,
        backbone_before,
    )
    assert not torch.equal(model.action_head.weight, action_head_before)
    assert not torch.equal(
        model.contact_posterior.mean_head.weight,
        posterior_before,
    )
    assert not torch.equal(model.contact_prior.mean_head.weight, prior_before)
    assert metrics["loss_total"] > 0
    assert metrics["loss_posterior_kl"] >= 0
    assert metrics["loss_prior_match"] >= 0
    assert metrics["gradient_norm"] > 0
    assert metrics["main_learning_rate"] == training_config.learning_rate
    assert (
        metrics["backbone_learning_rate"]
        == training_config.backbone_learning_rate
    )


def test_validation_uses_posterior_mean_and_online_only_deployment_forward():
    model_config = _model_config()
    training_config = ACTAlignedTrainingConfig()
    model = ACTAlignedContactCVAEPolicy(model_config)
    criterion = ACTAlignedCriterion(training_config)
    batch = _batch(model_config)
    model.train()

    with patch.object(
        model,
        "forward_train",
        wraps=model.forward_train,
    ) as posterior_call, patch.object(
        model,
        "forward",
        wraps=model.forward,
    ) as deployment_call:
        metrics = evaluate_one_batch(model, criterion, batch)

    assert posterior_call.call_args.kwargs["sample_posterior"] is False
    assert deployment_call.call_count == 2
    deployment_modes = []
    for call in deployment_call.call_args_list:
        assert len(call.args) == 3
        assert set(call.kwargs) == {
            "force_padding_mask",
            "contact_latent_mode",
            "deterministic_prior",
        }
        assert call.kwargs["deterministic_prior"] is True
        deployment_modes.append(call.kwargs["contact_latent_mode"])
    assert deployment_modes == ["zero", "prior"]
    assert "deployment_zero_action_l1" in metrics
    assert "deployment_zero_force_l1" in metrics
    assert "deployment_prior_action_l1" in metrics
    assert "deployment_prior_force_l1" in metrics
    assert "posterior_kl_standard" in metrics
    assert "posterior_prior_match_kl" in metrics
    assert model.training is True


def test_validation_metrics_are_deterministic_in_eval_paths():
    torch.manual_seed(23)
    model_config = _model_config()
    training_config = ACTAlignedTrainingConfig()
    model = ACTAlignedContactCVAEPolicy(model_config)
    criterion = ACTAlignedCriterion(training_config)
    batch = _batch(model_config)

    first = evaluate_one_batch(model, criterion, batch)
    second = evaluate_one_batch(model, criterion, batch)

    assert first == second


def test_prior_matching_loss_does_not_backpropagate_into_posterior_module():
    model_config = _model_config()
    training_config = ACTAlignedTrainingConfig()
    model = ACTAlignedContactCVAEPolicy(model_config)
    criterion = ACTAlignedCriterion(training_config)
    batch = _batch(model_config)
    outputs = model.forward_train(
        batch.images,
        batch.qpos,
        batch.force_history,
        batch.action_chunk,
        batch.future_force_chunk,
        force_padding_mask=batch.force_padding_mask,
        future_padding_mask=batch.future_padding_mask,
    )

    losses = criterion(
        outputs,
        batch.action_chunk,
        batch.future_force_chunk,
        batch.future_padding_mask,
    )
    losses["loss_prior_match"].backward()

    assert all(
        parameter.grad is None
        for parameter in model.contact_posterior.parameters()
    )
    assert any(
        parameter.grad is not None
        for parameter in model.contact_prior.parameters()
    )


def test_train_step_rejects_mismatched_criterion_config():
    model_config = _model_config()
    model = ACTAlignedContactCVAEPolicy(model_config)
    training_config = ACTAlignedTrainingConfig()
    different_config = ACTAlignedTrainingConfig(posterior_kl_weight=1.0)
    optimizer = build_act_aligned_optimizer(model, training_config)

    with pytest.raises(ValueError, match="same config"):
        train_one_step(
            model,
            ACTAlignedCriterion(different_config),
            optimizer,
            _batch(model_config),
            training_config,
        )
