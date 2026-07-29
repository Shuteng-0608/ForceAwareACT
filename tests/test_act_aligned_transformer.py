import pytest
import torch
from torch import nn

from force_aware_act.models.act_aligned import (
    ACTAlignedConfig,
    ACTQueryDecoder,
    ACTTransformerDecoder,
    ACTTransformerEncoder,
)


def _small_config(**overrides):
    values = {
        "d_model": 32,
        "nhead": 4,
        "dim_feedforward": 64,
        "dropout": 0.0,
        "chunk_len": 6,
        "pretrained_backbone": False,
        "imagenet_normalize": False,
    }
    values.update(overrides)
    return ACTAlignedConfig(**values)


def test_encoder_has_official_depth_and_no_final_stack_norm():
    config = _small_config()
    encoder = ACTTransformerEncoder(config)

    assert len(encoder.layers) == 4
    assert not hasattr(encoder, "norm")


def test_encoder_preserves_batch_first_shape_and_accepts_padding_mask():
    config = _small_config()
    encoder = ACTTransformerEncoder(config)
    source = torch.randn(2, 9, config.d_model)
    position = torch.randn_like(source)
    padding_mask = torch.zeros(2, 9, dtype=torch.bool)
    padding_mask[:, -2:] = True

    output = encoder(
        source,
        position=position,
        padding_mask=padding_mask,
    )

    assert output.shape == source.shape
    assert torch.isfinite(output).all()


def test_decoder_has_seven_layers_and_shared_final_norm():
    config = _small_config()
    decoder = ACTTransformerDecoder(config)

    assert len(decoder.layers) == 7
    assert isinstance(decoder.norm, nn.LayerNorm)


def test_query_decoder_returns_last_and_all_decoder_layers():
    config = _small_config()
    decoder = ACTQueryDecoder(config)
    memory = torch.randn(2, 11, config.d_model)
    memory_position = torch.randn_like(memory)

    final = decoder(memory, memory_position=memory_position)
    all_layers = decoder(
        memory,
        memory_position=memory_position,
        return_intermediate=True,
    )

    assert final.shape == (2, config.chunk_len, config.d_model)
    assert all_layers.shape == (
        config.decoder_layers,
        2,
        config.chunk_len,
        config.d_model,
    )
    torch.testing.assert_close(final, all_layers[-1])


def test_gradient_reaches_every_encoder_and_decoder_layer():
    config = _small_config()
    encoder = ACTTransformerEncoder(config)
    decoder = ACTQueryDecoder(config)
    source = torch.randn(2, 8, config.d_model)
    position = torch.randn_like(source)

    memory = encoder(source, position=position)
    output = decoder(memory, memory_position=position)
    output.square().mean().backward()

    for layer in encoder.layers:
        assert layer.linear1.weight.grad is not None
        assert torch.count_nonzero(layer.linear1.weight.grad) > 0
    for layer in decoder.decoder.layers:
        assert layer.linear1.weight.grad is not None
        assert torch.count_nonzero(layer.linear1.weight.grad) > 0


@pytest.mark.parametrize(
    "operation, message",
    [
        (
            lambda module, config: module(torch.randn(2, config.d_model, 5)),
            r"\[B, S, 32\]",
        ),
        (
            lambda module, config: module(
                torch.randn(2, 5, config.d_model),
                position=torch.randn(2, 6, config.d_model),
            ),
            "position must have shape",
        ),
        (
            lambda module, config: module(
                torch.randn(2, 5, config.d_model),
                padding_mask=torch.zeros(2, 5),
            ),
            "dtype torch.bool",
        ),
    ],
)
def test_encoder_rejects_invalid_layouts(operation, message):
    config = _small_config()
    encoder = ACTTransformerEncoder(config)

    with pytest.raises(ValueError, match=message):
        operation(encoder, config)


def test_query_count_tracks_canonical_chunk_length():
    config = ACTAlignedConfig.compact_smoke()
    decoder = ACTQueryDecoder(config)

    assert decoder.query_embedding.num_embeddings == 100
