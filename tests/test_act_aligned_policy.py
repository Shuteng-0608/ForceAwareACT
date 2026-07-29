import inspect

import pytest
import torch
from torch import nn

pytest.importorskip("torchvision")

from force_aware_act.models.act_aligned import (  # noqa: E402
    ACTAlignedConfig,
    ACTAlignedContactCVAEPolicy,
    POLICY_SPECIAL_TOKEN_NAMES,
)


def _small_config(**overrides):
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
    return ACTAlignedConfig(**values)


def _online_inputs(config, batch_size=2):
    return {
        "images": torch.randn(
            batch_size,
            config.num_cameras,
            3,
            config.image_height,
            config.image_width,
        ),
        "qpos": torch.randn(batch_size, config.q_dim),
        "force_history": torch.randn(
            batch_size,
            config.force_window_len,
            config.force_dim,
        ),
    }


def _training_inputs(config, batch_size=2):
    return {
        **_online_inputs(config, batch_size),
        "action_chunk": torch.randn(
            batch_size,
            config.chunk_len,
            config.action_dim,
        ),
        "future_force_chunk": torch.randn(
            batch_size,
            config.chunk_len,
            config.force_dim,
        ),
    }


def test_training_path_assembles_complete_policy_and_uses_posterior():
    config = _small_config()
    policy = ACTAlignedContactCVAEPolicy(config)

    outputs = policy.forward_train(**_training_inputs(config))

    assert outputs["contact_latent_source"] == "posterior"
    assert outputs["z_contact"] is outputs["z_contact_posterior"]
    assert outputs["policy_tokens"].shape == (
        2,
        config.policy_memory_token_count,
        config.d_model,
    )
    assert outputs["policy_memory"].shape == outputs["policy_tokens"].shape
    assert outputs["decoder_hidden"].shape == (
        2,
        config.chunk_len,
        config.d_model,
    )
    assert outputs["pred_action"].shape == (
        2,
        config.chunk_len,
        config.action_dim,
    )
    assert outputs["pred_force"].shape == (
        2,
        config.chunk_len,
        config.force_dim,
    )
    assert outputs["mu_contact"].shape == (2, config.latent_dim)
    assert outputs["mu_contact_prior"].shape == (2, config.latent_dim)


def test_mean_posterior_validation_also_uses_deterministic_prior_statistics():
    config = _small_config()
    policy = ACTAlignedContactCVAEPolicy(config).eval()

    with torch.no_grad():
        outputs = policy.forward_train(
            **_training_inputs(config),
            sample_posterior=False,
        )

    torch.testing.assert_close(
        outputs["z_contact_posterior"],
        outputs["mu_contact"],
    )
    torch.testing.assert_close(
        outputs["z_contact_prior"],
        outputs["mu_contact_prior"],
    )


def test_default_deployment_path_uses_zero_and_has_no_future_label_arguments():
    config = _small_config()
    policy = ACTAlignedContactCVAEPolicy(config).eval()
    inputs = _online_inputs(config)

    with torch.no_grad():
        first = policy(**inputs)
        second = policy(**inputs)

    assert first["contact_latent_source"] == "zero"
    torch.testing.assert_close(
        first["z_contact"],
        torch.zeros_like(first["z_contact"]),
    )
    torch.testing.assert_close(first["z_contact"], second["z_contact"])
    torch.testing.assert_close(first["pred_action"], second["pred_action"])
    parameters = inspect.signature(policy.forward).parameters
    assert "action_chunk" not in parameters
    assert "future_force_chunk" not in parameters

    with pytest.raises(TypeError, match="unexpected keyword argument"):
        policy(**inputs, action_chunk=torch.zeros(2, 6, 7))


def test_deployment_can_use_deterministic_or_sampled_conditional_prior():
    config = _small_config()
    policy = ACTAlignedContactCVAEPolicy(config).eval()
    inputs = _online_inputs(config)

    with torch.no_grad():
        deterministic = policy(
            **inputs,
            contact_latent_mode="prior",
            deterministic_prior=True,
        )
        sampled = policy(
            **inputs,
            contact_latent_mode="prior",
            deterministic_prior=False,
        )

    assert deterministic["contact_latent_source"] == "prior_mean"
    torch.testing.assert_close(
        deterministic["z_contact"],
        deterministic["mu_contact_prior"],
    )
    assert sampled["contact_latent_source"] == "prior_sample"
    assert not torch.equal(sampled["z_contact"], sampled["mu_contact_prior"])


def test_deployment_supports_explicit_offline_latent_override():
    config = _small_config()
    policy = ACTAlignedContactCVAEPolicy(config).eval()
    override = torch.randn(2, config.latent_dim)

    with torch.no_grad():
        outputs = policy(
            **_online_inputs(config),
            contact_latent_override=override,
        )

    assert outputs["contact_latent_source"] == "override"
    assert outputs["z_contact"] is override


def test_policy_memory_order_is_frozen_and_positions_match_visual_tail():
    config = _small_config()
    policy = ACTAlignedContactCVAEPolicy(config)
    batch_size = 2
    contact_latent = torch.randn(batch_size, config.latent_dim)
    qpos_token = torch.full((batch_size, 1, config.d_model), 2.0)
    online_force = torch.full((batch_size, config.d_model), 3.0)
    force_vision = torch.full((batch_size, config.d_model), 4.0)
    visual_tokens = torch.full(
        (batch_size, config.visual_token_count, config.d_model),
        5.0,
    )
    visual_position = torch.full_like(visual_tokens, 6.0)

    tokens, position = policy._assemble_policy_input(
        contact_latent=contact_latent,
        qpos_token=qpos_token,
        online_force=online_force,
        force_vision=force_vision,
        visual_tokens=visual_tokens,
        visual_position=visual_position,
    )

    assert policy.policy_special_token_names == (
        "z_contact",
        "qpos",
        "z_F_online",
        "z_VF",
    )
    assert policy.policy_special_token_names == POLICY_SPECIAL_TOKEN_NAMES
    torch.testing.assert_close(
        tokens[:, 0:1],
        policy.contact_latent_adapter(contact_latent),
    )
    torch.testing.assert_close(tokens[:, 1:2], qpos_token)
    torch.testing.assert_close(tokens[:, 2], online_force)
    torch.testing.assert_close(tokens[:, 3], force_vision)
    torch.testing.assert_close(tokens[:, 4:], visual_tokens)
    torch.testing.assert_close(position[:, 4:], visual_position)


def test_parallel_heads_read_the_same_final_decoder_hidden():
    config = _small_config()
    policy = ACTAlignedContactCVAEPolicy(config).eval()
    action_inputs = []
    force_inputs = []
    action_handle = policy.action_head.register_forward_pre_hook(
        lambda _module, args: action_inputs.append(args[0])
    )
    force_handle = policy.force_head.register_forward_pre_hook(
        lambda _module, args: force_inputs.append(args[0])
    )

    with torch.no_grad():
        outputs = policy(
            **_online_inputs(config),
            return_intermediate_decoder=True,
        )
    action_handle.remove()
    force_handle.remove()

    assert isinstance(policy.action_head, nn.Linear)
    assert isinstance(policy.force_head, nn.Linear)
    assert policy.action_head.in_features == config.d_model
    assert policy.force_head.in_features == config.d_model
    assert action_inputs[0] is outputs["decoder_hidden"]
    assert force_inputs[0] is outputs["decoder_hidden"]
    assert outputs["decoder_intermediate"].shape == (
        config.decoder_layers,
        2,
        config.chunk_len,
        config.d_model,
    )
    torch.testing.assert_close(
        outputs["decoder_hidden"],
        outputs["decoder_intermediate"][-1],
    )


def test_gradient_reaches_policy_encoder_all_decoder_layers_and_both_heads():
    config = _small_config()
    policy = ACTAlignedContactCVAEPolicy(config)
    outputs = policy.forward_train(**_training_inputs(config))
    loss = (
        outputs["pred_action"].square().mean()
        + outputs["pred_force"].square().mean()
        + outputs["mu_contact_prior"].square().mean()
        + outputs["logvar_contact_prior"].square().mean()
    )

    loss.backward()

    for layer in policy.policy_encoder.layers:
        assert layer.linear1.weight.grad is not None
        assert torch.count_nonzero(layer.linear1.weight.grad) > 0
    for layer in policy.query_decoder.decoder.layers:
        assert layer.linear1.weight.grad is not None
        assert torch.count_nonzero(layer.linear1.weight.grad) > 0
    assert policy.action_head.weight.grad is not None
    assert torch.count_nonzero(policy.action_head.weight.grad) > 0
    assert policy.force_head.weight.grad is not None
    assert torch.count_nonzero(policy.force_head.weight.grad) > 0


def test_chunk_length_100_controls_queries_and_both_prediction_heads():
    config = _small_config(chunk_len=100)
    policy = ACTAlignedContactCVAEPolicy(config).eval()

    with torch.no_grad():
        outputs = policy(**_online_inputs(config, batch_size=1))

    assert policy.query_decoder.query_embedding.num_embeddings == 100
    assert outputs["decoder_hidden"].shape == (1, 100, config.d_model)
    assert outputs["pred_action"].shape == (1, 100, config.action_dim)
    assert outputs["pred_force"].shape == (1, 100, config.force_dim)


def test_canonical_two_camera_resolution_builds_102_token_policy_memory():
    config = _small_config(image_height=224, image_width=224)
    policy = ACTAlignedContactCVAEPolicy(config).eval()

    with torch.no_grad():
        outputs = policy(**_online_inputs(config, batch_size=1))

    assert config.visual_token_count == 98
    assert outputs["visual_tokens"].shape == (1, 98, config.d_model)
    assert outputs["policy_tokens"].shape == (1, 102, config.d_model)
    assert outputs["policy_position"].shape == (1, 102, config.d_model)
    assert outputs["policy_memory"].shape == (1, 102, config.d_model)


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"qpos": torch.randn(3, 7)}, "same batch size"),
        (
            {"force_history": torch.randn(2, 5, 6, dtype=torch.float64)},
            "same dtype as images",
        ),
    ],
)
def test_policy_rejects_inconsistent_online_context(changes, message):
    config = _small_config()
    inputs = _online_inputs(config)
    inputs.update(changes)

    with pytest.raises(ValueError, match=message):
        ACTAlignedContactCVAEPolicy(config)(**inputs)


def test_policy_rejects_unknown_deployment_latent_mode():
    config = _small_config()

    with pytest.raises(ValueError, match="zero.*prior"):
        ACTAlignedContactCVAEPolicy(config)(
            **_online_inputs(config),
            contact_latent_mode="posterior",
        )
