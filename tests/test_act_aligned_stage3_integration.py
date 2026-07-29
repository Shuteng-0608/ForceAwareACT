import pytest
import torch

pytest.importorskip("torchvision")

from force_aware_act.models.act_aligned import (  # noqa: E402
    ACTAlignedConfig,
    ACTAlignedContactPosterior,
    ACTAlignedContactPrior,
    ACTAlignedForceVisionFusion,
    ACTAlignedOnlineForceEncoder,
    ACTAlignedResNet18Backbone,
    QposTokenAdapter,
)


def test_stage3_online_and_training_only_paths_have_compatible_shapes():
    config = ACTAlignedConfig(
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
    vision = ACTAlignedResNet18Backbone(config)
    qpos_adapter = QposTokenAdapter(config.q_dim, config.d_model)
    online_force_encoder = ACTAlignedOnlineForceEncoder(config)
    fusion = ACTAlignedForceVisionFusion(config)
    prior = ACTAlignedContactPrior(config)
    posterior = ACTAlignedContactPosterior(config)

    batch_size = 2
    images = torch.randn(
        batch_size,
        config.num_cameras,
        3,
        config.image_height,
        config.image_width,
    )
    qpos = torch.randn(batch_size, config.q_dim)
    force_history = torch.randn(
        batch_size,
        config.force_window_len,
        config.force_dim,
    )
    action_chunk = torch.randn(
        batch_size,
        config.chunk_len,
        config.action_dim,
    )
    future_force = torch.randn(
        batch_size,
        config.chunk_len,
        config.force_dim,
    )

    visual_tokens, visual_position = vision(images)
    qpos_feature = qpos_adapter(qpos)[:, 0]
    online_force = online_force_encoder(force_history)
    force_vision = fusion(
        online_force,
        visual_tokens,
        visual_position=visual_position,
    )
    prior_mean, prior_log_variance, deployment_latent = prior(
        qpos_feature,
        online_force,
        force_vision,
        visual_tokens.mean(dim=1),
        deterministic=True,
    )
    posterior_mean, posterior_log_variance, training_latent = posterior(
        qpos,
        action_chunk,
        future_force,
    )

    assert visual_tokens.shape == (
        batch_size,
        config.visual_token_count,
        config.d_model,
    )
    assert qpos_feature.shape == (batch_size, config.d_model)
    assert online_force.shape == (batch_size, config.d_model)
    assert force_vision.shape == (batch_size, config.d_model)
    assert prior_mean.shape == (batch_size, config.latent_dim)
    assert prior_log_variance.shape == prior_mean.shape
    assert deployment_latent.shape == prior_mean.shape
    assert posterior_mean.shape == prior_mean.shape
    assert posterior_log_variance.shape == prior_mean.shape
    assert training_latent.shape == prior_mean.shape
    torch.testing.assert_close(deployment_latent, prior_mean)
