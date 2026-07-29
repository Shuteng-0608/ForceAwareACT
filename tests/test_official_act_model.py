import inspect

import pytest
import torch

pytest.importorskip("torchvision")

from force_aware_act.models.official_act import (  # noqa: E402
    OFFICIAL_ACT_ARCHITECTURE_VERSION,
    OfficialACTConfig,
    OfficialACTPolicy,
    OfficialACTPosterior,
)


def _inputs(config, batch_size=2):
    return {
        "images": torch.rand(
            batch_size,
            config.num_cameras,
            3,
            config.image_height,
            config.image_width,
        ),
        "qpos": torch.randn(batch_size, config.q_dim),
        "action_chunk": torch.randn(
            batch_size,
            config.chunk_len,
            config.action_dim,
        ),
        "padding_mask": torch.zeros(
            batch_size,
            config.chunk_len,
            dtype=torch.bool,
        ),
    }


def test_canonical_config_freezes_official_readme_hyperparameters():
    config = OfficialACTConfig.canonical()

    assert config.architecture_version == OFFICIAL_ACT_ARCHITECTURE_VERSION
    assert config.d_model == 512
    assert config.nhead == 8
    assert config.dim_feedforward == 3200
    assert config.encoder_layers == 4
    assert config.decoder_layers == 7
    assert config.chunk_len == 100
    assert config.latent_dim == 32
    assert (config.image_height, config.image_width) == (480, 640)
    assert config.visual_token_count == 600
    assert config.policy_memory_token_count == 602


def test_posterior_is_exact_action_only_sequence_without_token_types():
    config = OfficialACTConfig.compact_smoke()
    posterior = OfficialACTPosterior(config)
    values = _inputs(config)

    mean, log_variance, latent = posterior(
        values["qpos"],
        values["action_chunk"],
        values["padding_mask"],
        sample=False,
    )

    assert mean.shape == (2, config.latent_dim)
    assert log_variance.shape == mean.shape
    torch.testing.assert_close(latent, mean)
    assert len(posterior.encoder.layers) == 4
    assert not hasattr(posterior, "token_type")
    assert "future_force_chunk" not in inspect.signature(
        posterior.forward
    ).parameters


def test_policy_matches_official_memory_layout_and_output_shapes():
    config = OfficialACTConfig.compact_smoke()
    policy = OfficialACTPolicy(config)
    values = _inputs(config)

    outputs = policy.forward_train(**values, sample_posterior=False)

    assert outputs["motion_latent_source"] == "posterior"
    assert outputs["visual_tokens"].shape == (
        2,
        config.visual_token_count,
        config.d_model,
    )
    assert outputs["memory_tokens"].shape == (
        2,
        config.policy_memory_token_count,
        config.d_model,
    )
    torch.testing.assert_close(
        outputs["memory_tokens"][:, 0],
        policy.latent_output_projection(outputs["z_motion"]),
    )
    torch.testing.assert_close(
        outputs["memory_tokens"][:, 1],
        policy.robot_state_projection(values["qpos"]),
    )
    assert outputs["pred_action"].shape == (
        2,
        config.chunk_len,
        config.action_dim,
    )
    assert outputs["pred_is_pad"].shape == (2, config.chunk_len, 1)
    assert len(policy.policy_encoder.layers) == 4
    assert len(policy.policy_decoder.layers) == 7


def test_deployment_is_zero_latent_and_rejects_future_labels():
    config = OfficialACTConfig.compact_smoke()
    policy = OfficialACTPolicy(config).eval()
    values = _inputs(config)

    with torch.no_grad():
        outputs = policy(values["images"], values["qpos"])

    assert outputs["motion_latent_source"] == "zero"
    torch.testing.assert_close(
        outputs["z_motion"],
        torch.zeros_like(outputs["z_motion"]),
    )
    signature = inspect.signature(policy.forward).parameters
    assert "action_chunk" not in signature
    assert "force_history" not in signature
    assert all("force" not in name for name, _ in policy.named_parameters())


def test_camera_features_are_concatenated_along_width_without_camera_embedding():
    config = OfficialACTConfig.compact_smoke()
    policy = OfficialACTPolicy(config).eval()
    images = torch.rand(
        1,
        config.num_cameras,
        3,
        config.image_height,
        config.image_width,
    )

    with torch.no_grad():
        tokens, positions = policy.backbone(images)

    assert tokens.shape == (
        1,
        config.visual_grid_height
        * config.visual_grid_width_per_camera
        * config.num_cameras,
        config.d_model,
    )
    assert positions.shape == tokens.shape
    assert not hasattr(policy.backbone, "camera_position")
