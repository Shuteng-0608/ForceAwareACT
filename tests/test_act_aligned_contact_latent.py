import inspect

import pytest
import torch
from torch import nn

from force_aware_act.models.act_aligned import (
    ACTAlignedConfig,
    ACTAlignedContactPosterior,
    ACTAlignedContactPrior,
    reparameterize_gaussian,
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


def _posterior_inputs(config, batch_size=2):
    return {
        "qpos": torch.randn(batch_size, config.q_dim),
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


def test_contact_posterior_uses_time_aligned_sequence_and_four_layers():
    config = _small_config()
    posterior = ACTAlignedContactPosterior(config)

    assert posterior.sequence_position.max_length == config.chunk_len + 2
    assert posterior.action_adapter.projection.out_features == config.d_model
    assert posterior.force_adapter.projection.out_features == config.d_model
    assert len(posterior.encoder.layers) == 4

    mean, log_variance, latent = posterior(**_posterior_inputs(config))

    assert mean.shape == (2, config.latent_dim)
    assert log_variance.shape == mean.shape
    assert latent.shape == mean.shape


def test_contact_posterior_encoder_input_pairs_action_and_force_by_timestep():
    class CaptureEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.source = None
            self.position = None
            self.padding_mask = None

        def forward(self, source, *, position, padding_mask):
            self.source = source
            self.position = position
            self.padding_mask = padding_mask
            return source

    config = _small_config()
    posterior = ACTAlignedContactPosterior(config)
    capture = CaptureEncoder()
    posterior.encoder = capture
    inputs = _posterior_inputs(config)
    padding_mask = torch.zeros(2, config.chunk_len, dtype=torch.bool)
    padding_mask[:, -1] = True

    posterior(**inputs, padding_mask=padding_mask, sample=False)

    expected_steps = posterior.action_adapter(
        inputs["action_chunk"]
    ) + posterior.force_adapter(inputs["future_force_chunk"])
    torch.testing.assert_close(capture.source[:, 2:], expected_steps)
    assert capture.source.shape == (
        2,
        config.chunk_len + 2,
        config.d_model,
    )
    assert capture.position.shape == capture.source.shape
    assert capture.padding_mask.shape == (2, config.chunk_len + 2)
    assert not capture.padding_mask[:, :2].any()
    torch.testing.assert_close(capture.padding_mask[:, 2:], padding_mask)


def test_contact_posterior_mask_makes_padded_future_values_irrelevant():
    torch.manual_seed(3)
    config = _small_config()
    posterior = ACTAlignedContactPosterior(config).eval()
    inputs = _posterior_inputs(config)
    changed = {name: value.clone() for name, value in inputs.items()}
    changed["action_chunk"][:, -2:] = 1000.0
    changed["future_force_chunk"][:, -2:] = -1000.0
    padding_mask = torch.zeros(2, config.chunk_len, dtype=torch.bool)
    padding_mask[:, -2:] = True

    first = posterior(**inputs, padding_mask=padding_mask, sample=False)
    second = posterior(**changed, padding_mask=padding_mask, sample=False)

    torch.testing.assert_close(first[0], second[0], atol=1.0e-6, rtol=1.0e-6)
    torch.testing.assert_close(first[1], second[1], atol=1.0e-6, rtol=1.0e-6)


def test_contact_posterior_gradient_reaches_all_four_encoder_layers():
    config = _small_config()
    posterior = ACTAlignedContactPosterior(config)

    mean, log_variance, latent = posterior(**_posterior_inputs(config))
    (mean.square().mean() + log_variance.square().mean() + latent.square().mean()).backward()

    for layer in posterior.encoder.layers:
        assert layer.linear1.weight.grad is not None
        assert torch.count_nonzero(layer.linear1.weight.grad) > 0


def test_contact_prior_is_causal_by_interface_and_deterministic_on_deployment():
    config = _small_config()
    prior = ACTAlignedContactPrior(config).eval()
    features = [torch.randn(2, config.d_model) for _ in range(4)]

    first = prior(*features, deterministic=True)
    second = prior(*features, deterministic=True)

    torch.testing.assert_close(first[0], second[0])
    torch.testing.assert_close(first[1], second[1])
    torch.testing.assert_close(first[2], first[0])
    torch.testing.assert_close(second[2], second[0])
    parameters = inspect.signature(prior.forward).parameters
    assert "action_chunk" not in parameters
    assert "future_force_chunk" not in parameters


def test_contact_prior_can_sample_and_backpropagate():
    config = _small_config()
    prior = ACTAlignedContactPrior(config)
    features = [
        torch.randn(2, config.d_model, requires_grad=True)
        for _ in range(4)
    ]

    mean, log_variance, first_sample = prior(*features, deterministic=False)
    _, _, second_sample = prior(*features, deterministic=False)
    first_sample.square().mean().backward()

    assert mean.shape == (2, config.latent_dim)
    assert log_variance.shape == mean.shape
    assert not torch.equal(first_sample, second_sample)
    assert all(feature.grad is not None for feature in features)


def test_reparameterization_validates_shapes():
    with pytest.raises(ValueError, match="same shape"):
        reparameterize_gaussian(torch.zeros(2, 3), torch.zeros(2, 4))


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"qpos": torch.randn(2, 8)}, "qpos must have shape"),
        (
            {"action_chunk": torch.randn(2, 5, 7)},
            "action_chunk must have shape",
        ),
        (
            {"future_force_chunk": torch.randn(2, 6, 5)},
            "future_force_chunk must have shape",
        ),
    ],
)
def test_contact_posterior_rejects_invalid_shapes(changes, message):
    config = _small_config()
    inputs = _posterior_inputs(config)
    inputs.update(changes)

    with pytest.raises(ValueError, match=message):
        ACTAlignedContactPosterior(config)(**inputs)
