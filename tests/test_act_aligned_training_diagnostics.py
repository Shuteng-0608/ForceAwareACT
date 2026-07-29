import torch

from force_aware_act.act_aligned_training import (
    ACTAlignedBatch,
    ACTAlignedCriterion,
    ACTAlignedTrainingConfig,
    build_act_aligned_optimizer,
    run_training_preflight,
)
from force_aware_act.models.act_aligned import (
    ACTAlignedConfig,
    ACTAlignedContactCVAEPolicy,
)


def _config():
    return ACTAlignedConfig(
        d_model=16,
        nhead=4,
        dim_feedforward=32,
        dropout=0.0,
        chunk_len=3,
        force_window_len=2,
        image_height=32,
        image_width=32,
        pretrained_backbone=False,
        imagenet_normalize=False,
    )


def _batch(config):
    batch_size = 2
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
        future_padding_mask=torch.zeros(
            batch_size,
            config.chunk_len,
            dtype=torch.bool,
        ),
    )


def test_preflight_audits_shapes_optimizer_gradients_and_prior_isolation():
    config = _config()
    training_config = ACTAlignedTrainingConfig(batch_size=2)
    model = ACTAlignedContactCVAEPolicy(config)
    criterion = ACTAlignedCriterion(training_config)
    optimizer = build_act_aligned_optimizer(model, training_config)

    report = run_training_preflight(
        model,
        criterion,
        optimizer,
        _batch(config),
    )

    assert report["passed"]
    assert report["output_shapes"]["pred_action"] == [2, 3, 7]
    assert report["output_shapes"]["pred_force"] == [2, 3, 6]
    assert report["output_shapes"]["decoder_intermediate"] == [7, 2, 3, 16]
    assert report["deployment"]["zero_latent_source"] == "zero"
    assert report["deployment"]["prior_latent_source"] == "prior_mean"
    assert report["prior_stop_gradient"]["passed"]
    assert (
        report["prior_stop_gradient"][
            "posterior_gradient_norm_from_prior_match"
        ]
        == 0.0
    )
    assert (
        report["prior_stop_gradient"]["prior_gradient_norm_from_prior_match"]
        > 0.0
    )
    assert {group["name"] for group in report["optimizer_groups"]} == {
        "main",
        "backbone",
    }
    for module_report in report["gradients"].values():
        assert module_report["parameters_with_gradient"] > 0
        assert module_report["gradient_norm"] > 0.0


def test_preflight_does_not_take_an_optimizer_step():
    config = _config()
    training_config = ACTAlignedTrainingConfig(batch_size=2)
    model = ACTAlignedContactCVAEPolicy(config)
    criterion = ACTAlignedCriterion(training_config)
    optimizer = build_act_aligned_optimizer(model, training_config)
    before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
    }

    run_training_preflight(model, criterion, optimizer, _batch(config))

    for name, parameter in model.named_parameters():
        torch.testing.assert_close(parameter, before[name])
    assert not optimizer.state
