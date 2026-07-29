import pytest

from force_aware_act.act_aligned_training import (
    ACT_ALIGNED_TRAINING_VERSION,
    ACTAlignedTrainingConfig,
)


def test_canonical_training_config_is_act_style_and_explicit():
    config = ACTAlignedTrainingConfig()

    assert config.training_version == ACT_ALIGNED_TRAINING_VERSION
    assert config.learning_rate == 1.0e-5
    assert config.backbone_learning_rate == 1.0e-5
    assert config.weight_decay == 1.0e-4
    assert config.action_loss_weight == 1.0
    assert config.force_loss_weight == 1.0
    assert config.posterior_kl_weight == 10.0
    assert config.prior_match_weight == 1.0
    assert config.prior_match_mode == "detached_gaussian_kl"
    assert config.reference_train_episodes == 90
    assert config.official_reference_epochs == 2000
    assert config.max_optimizer_steps == 24000
    assert config.checkpoint_interval_steps == 2000
    assert config.gradient_clip_norm is None


def test_training_checkpoint_metadata_records_objective_and_validation_modes():
    metadata = ACTAlignedTrainingConfig().checkpoint_metadata()

    assert metadata["optimizer"] == "AdamW"
    assert metadata["scheduler"] is None
    assert "official_equivalent_optimizer_steps" in metadata[
        "duration_semantics"
    ]
    assert "kl_q_standard_normal" in metadata["objective"]
    assert "kl_stopgrad_q_p_conditional" in metadata["objective"]
    assert metadata["posterior_validation_latent"] == "mean"
    assert metadata["deployment_validation_latents"] == (
        "zero",
        "conditional_prior_mean",
    )


def test_training_config_derives_official_equivalent_optimizer_steps():
    config = ACTAlignedTrainingConfig(
        batch_size=10,
        reference_train_episodes=90,
        official_reference_epochs=2000,
    )

    assert config.max_optimizer_steps == 18000


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"learning_rate": 0.0}, "learning_rate must be positive"),
        (
            {"learning_rate": 1.0e-5, "backbone_learning_rate": 1.0e-4},
            "must not exceed",
        ),
        ({"weight_decay": -1.0}, "weight_decay must be non-negative"),
        (
            {"posterior_kl_weight": -1.0},
            "posterior_kl_weight must be non-negative",
        ),
        (
            {"prior_match_weight": -1.0},
            "prior_match_weight must be non-negative",
        ),
        (
            {"prior_match_mode": "mse_mu"},
            "detached_gaussian_kl",
        ),
        ({"adam_beta1": 1.0}, r"adam_beta1 must be in \[0, 1\)"),
        ({"batch_size": 0}, "batch_size must be a positive integer"),
        (
            {"max_optimizer_steps": 20000},
            "ceil\\(reference_train_episodes",
        ),
        ({"gradient_clip_norm": 0.0}, "gradient_clip_norm must be positive"),
        (
            {"selection_metric": "posterior_total"},
            "deployment_zero_action_l1",
        ),
    ],
)
def test_training_config_rejects_invalid_values(overrides, message):
    with pytest.raises(ValueError, match=message):
        ACTAlignedTrainingConfig(**overrides)
