import torch

from force_aware_act.act_aligned_training import (
    ACTAlignedHighRateBatch,
    ACTAlignedHighRateDualZeroCriterion,
    ACTAlignedHighRateDualZeroTrainingConfig,
    ACTAlignedHighRateMotionCriterion,
    ACTAlignedHighRateMotionTrainingConfig,
    build_act_aligned_high_rate_dual_zero_optimizer,
    build_act_aligned_high_rate_motion_optimizer,
    evaluate_high_rate_dual_zero_one_batch,
    evaluate_high_rate_motion_one_batch,
    run_high_rate_dual_zero_validation_epoch,
    run_high_rate_motion_validation_epoch,
    train_high_rate_dual_zero_one_step,
    train_high_rate_motion_one_step,
)
from force_aware_act.models.act_aligned import (
    ACTAlignedHighRateConfig,
    ACTAlignedHighRateDualZeroPolicy,
    ACTAlignedHighRateMotionCVAEPolicy,
)


def _model_config(kind: str):
    factory = {
        "motion": ACTAlignedHighRateConfig.motion_control,
        "dual_zero": ACTAlignedHighRateConfig.dual_zero,
    }[kind]
    return factory(
        d_model=32,
        nhead=4,
        dim_feedforward=64,
        dropout=0.0,
        chunk_len=3,
        local_force_dim=16,
        image_height=32,
        image_width=32,
        pretrained_backbone=False,
        imagenet_normalize=False,
    )


def _batch(config, batch_size=2):
    k = config.chunk_len
    h = config.max_online_force_intervals
    s = config.max_force_samples_per_interval
    online_sample_mask = torch.zeros(batch_size, h, s, dtype=torch.bool)
    online_sample_mask[:, 0, 10:] = True
    future_sample_mask = torch.zeros(batch_size, k, s, dtype=torch.bool)
    future_sample_mask[:, -1] = True
    action_mask = torch.zeros(batch_size, k, dtype=torch.bool)
    action_mask[:, -1] = True
    online_time = (
        torch.arange(s, dtype=torch.float32)
        .div(config.force_sample_rate_hz)
        .view(1, 1, s)
        .expand(batch_size, h, s)
        .clone()
    )
    future_time = online_time[:, :k].clone()
    return ACTAlignedHighRateBatch(
        images=torch.randn(
            batch_size,
            config.num_cameras,
            3,
            config.image_height,
            config.image_width,
        ),
        qpos=torch.randn(batch_size, config.q_dim),
        action_chunk=torch.randn(batch_size, k, config.action_dim),
        action_padding_mask=action_mask,
        online_force_intervals=torch.randn(
            batch_size,
            h,
            s,
            config.force_dim,
        ),
        online_force_relative_time=online_time,
        online_force_sample_padding_mask=online_sample_mask,
        online_force_interval_padding_mask=online_sample_mask.all(dim=-1),
        future_force_intervals=torch.randn(
            batch_size,
            k,
            s,
            config.force_dim,
        ),
        future_force_relative_time=future_time,
        future_force_sample_padding_mask=future_sample_mask,
        future_force_interval_padding_mask=future_sample_mask.all(dim=-1),
        future_force_target=torch.randn(batch_size, k, config.force_dim),
    )


def test_high_rate_motion_training_updates_posterior_and_force_encoder():
    torch.manual_seed(10)
    model_config = _model_config("motion")
    training_config = ACTAlignedHighRateMotionTrainingConfig()
    model = ACTAlignedHighRateMotionCVAEPolicy(model_config)
    criterion = ACTAlignedHighRateMotionCriterion(training_config)
    optimizer = build_act_aligned_high_rate_motion_optimizer(
        model,
        training_config,
    )
    batch = _batch(model_config)
    posterior_before = model.motion_posterior.mean_head.weight.detach().clone()
    force_before = (
        model.high_rate_force_encoder.input_projection.weight.detach().clone()
    )

    metrics = train_high_rate_motion_one_step(
        model,
        criterion,
        optimizer,
        batch,
        training_config,
    )
    validation = evaluate_high_rate_motion_one_batch(model, criterion, batch)

    assert metrics["loss_force_highrate"] >= 0
    assert metrics["loss_posterior_kl"] >= 0
    assert metrics["gradient_norm"] > 0
    assert not torch.equal(
        model.motion_posterior.mean_head.weight,
        posterior_before,
    )
    assert not torch.equal(
        model.high_rate_force_encoder.input_projection.weight,
        force_before,
    )
    assert validation["deployment_zero_force_highrate_l1"] >= 0
    assert validation["posterior_zero_action_delta"] >= 0


def test_high_rate_dual_zero_training_has_only_reconstruction_losses():
    torch.manual_seed(11)
    model_config = _model_config("dual_zero")
    training_config = ACTAlignedHighRateDualZeroTrainingConfig()
    model = ACTAlignedHighRateDualZeroPolicy(model_config)
    criterion = ACTAlignedHighRateDualZeroCriterion(training_config)
    optimizer = build_act_aligned_high_rate_dual_zero_optimizer(
        model,
        training_config,
    )
    batch = _batch(model_config)

    metrics = train_high_rate_dual_zero_one_step(
        model,
        criterion,
        optimizer,
        batch,
        training_config,
    )
    validation = evaluate_high_rate_dual_zero_one_batch(
        model,
        criterion,
        batch,
    )

    assert set(metrics) == {
        "loss_total",
        "loss_action",
        "loss_force",
        "loss_force_highrate",
        "gradient_norm",
        "main_learning_rate",
        "backbone_learning_rate",
    }
    assert metrics["gradient_norm"] > 0
    assert validation["deployment_action_l1"] >= 0
    metadata = training_config.checkpoint_metadata()
    assert metadata["latent_mechanism"] == "none"
    assert "posterior_kl_weight" not in metadata
    assert "prior_match_weight" not in metadata


def test_control_validation_epochs_aggregate_semantic_selection_metrics():
    motion_model_config = _model_config("motion")
    motion_training_config = ACTAlignedHighRateMotionTrainingConfig()
    motion_model = ACTAlignedHighRateMotionCVAEPolicy(motion_model_config)
    motion_criterion = ACTAlignedHighRateMotionCriterion(
        motion_training_config
    )
    motion_metrics = run_high_rate_motion_validation_epoch(
        motion_model,
        motion_criterion,
        [_batch(motion_model_config, batch_size=1)],
        motion_training_config,
        device=torch.device("cpu"),
    )

    dual_model_config = _model_config("dual_zero")
    dual_training_config = ACTAlignedHighRateDualZeroTrainingConfig()
    dual_model = ACTAlignedHighRateDualZeroPolicy(dual_model_config)
    dual_criterion = ACTAlignedHighRateDualZeroCriterion(dual_training_config)
    dual_metrics = run_high_rate_dual_zero_validation_epoch(
        dual_model,
        dual_criterion,
        [_batch(dual_model_config, batch_size=1)],
        dual_training_config,
        device=torch.device("cpu"),
    )

    assert motion_training_config.selection_metric in motion_metrics
    assert motion_metrics["posterior_mean_across_sample_variance"] >= 0
    assert dual_training_config.selection_metric in dual_metrics
    assert dual_metrics["deployment_total"] >= 0
