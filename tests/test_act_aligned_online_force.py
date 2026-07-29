import pytest
import torch

from force_aware_act.models.act_aligned import (
    ACTAlignedConfig,
    ACTAlignedOnlineForceEncoder,
)


def _small_config(**overrides):
    values = {
        "d_model": 32,
        "nhead": 4,
        "dim_feedforward": 64,
        "dropout": 0.0,
        "chunk_len": 6,
        "force_window_len": 5,
        "pretrained_backbone": False,
        "imagenet_normalize": False,
    }
    values.update(overrides)
    return ACTAlignedConfig(**values)


def test_online_force_encoder_has_act_depth_and_expected_shape():
    config = _small_config()
    encoder = ACTAlignedOnlineForceEncoder(config)
    force_history = torch.randn(2, config.force_window_len, config.force_dim)

    output = encoder(force_history)

    assert len(encoder.encoder.layers) == 4
    assert encoder.sequence_position.max_length == config.force_window_len + 1
    assert output.shape == (2, config.d_model)


def test_online_force_padding_mask_excludes_padded_samples():
    torch.manual_seed(7)
    config = _small_config()
    encoder = ACTAlignedOnlineForceEncoder(config).eval()
    force_history = torch.randn(2, config.force_window_len, config.force_dim)
    changed = force_history.clone()
    changed[:, :2] = 1000.0
    padding_mask = torch.zeros(2, config.force_window_len, dtype=torch.bool)
    padding_mask[:, :2] = True

    first = encoder(force_history, padding_mask=padding_mask)
    second = encoder(changed, padding_mask=padding_mask)

    torch.testing.assert_close(first, second, atol=1.0e-6, rtol=1.0e-6)


def test_online_force_gradient_reaches_all_encoder_layers():
    config = _small_config()
    encoder = ACTAlignedOnlineForceEncoder(config)
    force_history = torch.randn(2, config.force_window_len, config.force_dim)

    encoder(force_history).square().mean().backward()

    for layer in encoder.encoder.layers:
        assert layer.linear1.weight.grad is not None
        assert torch.count_nonzero(layer.linear1.weight.grad) > 0


@pytest.mark.parametrize(
    "force_history, message",
    [
        (torch.randn(2, 5), "force_history must have shape"),
        (torch.randn(2, 4, 6), "force_history must have shape"),
        (torch.randn(2, 5, 7), "force_history must have shape"),
        (
            torch.ones(2, 5, 6, dtype=torch.int64),
            "force_history must be floating point",
        ),
    ],
)
def test_online_force_rejects_invalid_inputs(force_history, message):
    with pytest.raises(ValueError, match=message):
        ACTAlignedOnlineForceEncoder(_small_config())(force_history)
