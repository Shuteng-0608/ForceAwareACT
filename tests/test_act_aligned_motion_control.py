import inspect

import pytest
import torch

pytest.importorskip("torchvision")

from force_aware_act.act_aligned_training import (  # noqa: E402
    ACTAlignedBatch,
    ACTAlignedMotionCriterion,
    ACTAlignedMotionTrainingConfig,
    build_act_aligned_motion_optimizer,
    evaluate_motion_one_batch,
    partition_motion_trainable_parameters,
    run_motion_validation_epoch,
    train_motion_one_step,
)
from force_aware_act.models.act_aligned import (  # noqa: E402
    ACT_ALIGNED_MOTION_CONTROL_ARCHITECTURE_VERSION,
    ACTAlignedConfig,
    ACTAlignedMotionCVAEControlPolicy,
    ACTAlignedMotionPosterior,
)


def _config(**overrides):
    values = {
        "d_model": 32,
        "nhead": 4,
        "dim_feedforward": 64,
        "dropout": 0.0,
        "chunk_len": 6,
        "force_window_len": 5,
        "image_height": 64,
        "image_width": 64,
        "pretrained_backbone": False,
        "imagenet_normalize": False,
    }
    values.update(overrides)
    return ACTAlignedConfig.motion_control(**values)


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


def test_motion_posterior_matches_official_action_only_contract():
    config = _config()
    posterior = ACTAlignedMotionPosterior(config)
    qpos = torch.randn(2, config.q_dim)
    action = torch.randn(2, config.chunk_len, config.action_dim)

    mean, log_variance, latent = posterior(
        qpos,
        action,
        sample=False,
    )

    assert mean.shape == (2, config.latent_dim)
    assert log_variance.shape == mean.shape
    torch.testing.assert_close(latent, mean)
    assert len(posterior.encoder.layers) == config.encoder_layers == 4
    assert not hasattr(posterior, "force_adapter")
    parameters = inspect.signature(posterior.forward).parameters
    assert "future_force_chunk" not in parameters


def test_motion_policy_is_independent_and_deploys_with_zero_latent():
    config = _config()
    model = ACTAlignedMotionCVAEControlPolicy(config).eval()
    batch = _batch(config)

    with torch.no_grad():
        training_outputs = model.forward_train(
            batch.images,
            batch.qpos,
            batch.force_history,
            batch.action_chunk,
            force_padding_mask=batch.force_padding_mask,
            future_padding_mask=batch.future_padding_mask,
            sample_posterior=False,
        )
        deployment_outputs = model(
            batch.images,
            batch.qpos,
            batch.force_history,
            force_padding_mask=batch.force_padding_mask,
        )

    assert not hasattr(model, "contact_prior")
    assert not hasattr(model, "motion_prior")
    assert training_outputs["motion_latent_source"] == "posterior"
    torch.testing.assert_close(
        training_outputs["z_motion"],
        training_outputs["mu_motion"],
    )
    assert deployment_outputs["motion_latent_source"] == "zero"
    torch.testing.assert_close(
        deployment_outputs["z_motion"],
        torch.zeros_like(deployment_outputs["z_motion"]),
    )
    assert deployment_outputs["pred_action"].shape == (
        2,
        config.chunk_len,
        config.action_dim,
    )
    assert deployment_outputs["pred_force"].shape == (
        2,
        config.chunk_len,
        config.force_dim,
    )
    assert "future_force_chunk" not in inspect.signature(
        model.forward_train
    ).parameters
    assert "action_chunk" not in inspect.signature(model.forward).parameters


def test_motion_training_step_and_validation_report_collapse_diagnostics():
    torch.manual_seed(71)
    model_config = _config()
    training_config = ACTAlignedMotionTrainingConfig()
    model = ACTAlignedMotionCVAEControlPolicy(model_config)
    criterion = ACTAlignedMotionCriterion(training_config)
    optimizer = build_act_aligned_motion_optimizer(
        model,
        training_config,
    )
    batch = _batch(model_config)
    posterior_before = (
        model.motion_posterior.mean_head.weight.detach().clone()
    )

    train_metrics = train_motion_one_step(
        model,
        criterion,
        optimizer,
        batch,
        training_config,
    )
    validation_metrics = evaluate_motion_one_batch(
        model,
        criterion,
        batch,
    )

    assert not torch.equal(
        model.motion_posterior.mean_head.weight,
        posterior_before,
    )
    assert train_metrics["loss_total"] > 0
    assert train_metrics["loss_posterior_kl"] >= 0
    assert train_metrics["gradient_norm"] > 0
    assert validation_metrics["posterior_kl_standard"] >= 0
    assert validation_metrics["posterior_zero_action_delta"] >= 0
    assert validation_metrics["posterior_zero_force_delta"] >= 0
    assert validation_metrics["_posterior_mean_sum"].shape == (
        model_config.latent_dim,
    )


def test_motion_validation_loop_aggregates_latent_variance():
    model_config = _config()
    training_config = ACTAlignedMotionTrainingConfig()
    model = ACTAlignedMotionCVAEControlPolicy(model_config)
    criterion = ACTAlignedMotionCriterion(training_config)

    metrics = run_motion_validation_epoch(
        model,
        criterion,
        [_batch(model_config, batch_size=1), _batch(model_config, batch_size=2)],
        training_config,
        device=torch.device("cpu"),
    )

    assert metrics["deployment_zero_action_l1"] >= 0
    assert metrics["posterior_zero_action_delta"] >= 0
    assert metrics["posterior_mean_across_sample_variance"] >= 0
    assert metrics["posterior_total"] >= 0


def test_motion_optimizer_partition_is_complete_and_has_no_prior():
    model = ACTAlignedMotionCVAEControlPolicy(_config())
    partition = partition_motion_trainable_parameters(model)
    grouped_ids = {
        id(parameter)
        for parameters in partition.values()
        for parameter in parameters
    }
    trainable_ids = {
        id(parameter)
        for parameter in model.parameters()
        if parameter.requires_grad
    }

    assert grouped_ids == trainable_ids
    assert id(model.motion_posterior.mean_head.weight) in grouped_ids
    assert all("prior" not in name for name, _parameter in model.named_parameters())


def test_motion_versions_and_metadata_are_explicit():
    model_config = ACTAlignedConfig.motion_control()
    training_config = ACTAlignedMotionTrainingConfig()
    model_metadata = model_config.checkpoint_metadata()
    training_metadata = training_config.checkpoint_metadata()

    assert (
        model_config.architecture_version
        == ACT_ALIGNED_MOTION_CONTROL_ARCHITECTURE_VERSION
    )
    assert model_metadata["motion_posterior_layout"] == "official_action_only"
    assert model_metadata["motion_posterior_inputs"] == (
        "qpos",
        "action_chunk",
    )
    assert model_metadata["deployment_motion_latent"] == "zero"
    assert training_metadata["conditional_prior"] is None
    assert training_metadata["kl_warmup"] is None
    assert training_metadata["free_bits"] is None
