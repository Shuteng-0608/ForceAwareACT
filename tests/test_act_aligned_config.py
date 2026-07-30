from dataclasses import FrozenInstanceError

import pytest

from force_aware_act.models.act_aligned import (
    ACT_ALIGNED_ARCHITECTURE_VERSION,
    ACTAlignedConfig,
)


def test_canonical_act_config_freezes_official_depth_and_width():
    config = ACTAlignedConfig.canonical_act()

    assert config.architecture_version == ACT_ALIGNED_ARCHITECTURE_VERSION
    assert config.d_model == 512
    assert config.nhead == 8
    assert config.attention_head_dim == 64
    assert config.dim_feedforward == 3200
    assert config.encoder_layers == 4
    assert config.decoder_layers == 7
    assert config.dropout == 0.1
    assert config.activation == "relu"
    assert config.norm_first is False
    assert config.latent_dim == 32
    assert config.chunk_len == 100
    assert (config.image_height, config.image_width) == (480, 640)
    assert config.visual_token_count == 600
    assert config.policy_memory_token_count == 604


def test_compact_smoke_reduces_width_but_preserves_act_depth():
    config = ACTAlignedConfig.compact_smoke()

    assert config.d_model == 128
    assert config.nhead == 4
    assert config.dim_feedforward == 256
    assert config.encoder_layers == 4
    assert config.decoder_layers == 7
    assert config.chunk_len == 100
    assert config.dropout == 0.0
    assert (config.image_height, config.image_width) == (64, 64)
    assert config.pretrained_backbone is False
    assert config.imagenet_normalize is False


def test_config_is_immutable():
    config = ACTAlignedConfig()

    with pytest.raises(FrozenInstanceError):
        config.encoder_layers = 1


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"d_model": 0}, "d_model must be a positive integer"),
        ({"nhead": 0}, "nhead must be a positive integer"),
        ({"d_model": 130, "nhead": 8}, "d_model must be divisible by nhead"),
        ({"d_model": 130, "nhead": 5}, "d_model must be divisible by 4"),
        ({"dropout": 1.0}, "dropout must be in"),
        ({"activation": "gelu"}, "activation='relu'"),
        ({"norm_first": True}, "post-norm"),
        ({"backbone_name": "resnet50"}, "backbone_name='resnet18'"),
        (
            {"pretrained_backbone": True, "imagenet_normalize": False},
            "enabled or disabled together",
        ),
    ],
)
def test_config_rejects_non_aligned_settings(overrides, message):
    with pytest.raises(ValueError, match=message):
        ACTAlignedConfig(**overrides)


def test_checkpoint_metadata_makes_every_layer_count_explicit():
    config = ACTAlignedConfig()
    metadata = config.checkpoint_metadata()

    assert metadata["architecture_version"] == ACT_ALIGNED_ARCHITECTURE_VERSION
    assert metadata["contact_posterior_encoder_layers"] == 4
    assert metadata["online_force_encoder_layers"] == 4
    assert metadata["policy_encoder_layers"] == 4
    assert metadata["policy_decoder_layers"] == 7
    assert metadata["decoder_output_layer"] == "last"
    assert metadata["token_layout"] == "batch_first"
    assert metadata["contact_posterior_layout"] == "time_aligned_action_plus_force"
    assert metadata["deployment_contact_latent"] == "zero"
    assert metadata["deployment_contact_latent_modes"] == (
        "zero",
        "conditional_prior_mean",
        "conditional_prior_sample",
        "offline_override",
    )
    assert metadata["contact_prior_inputs"] == (
        "qpos_feature",
        "online_force_feature",
        "force_vision_feature",
        "visual_summary",
    )
    assert metadata["prediction_head_input"] == "final_decoder_hidden"
    assert metadata["action_head"] == "linear_512_to_7"
    assert metadata["force_head"] == "linear_512_to_6"
    assert metadata["force_head_contact_concat"] is False
