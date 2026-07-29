import pytest
import torch

from force_aware_act.models.act_aligned.position_encoding import (
    CameraPositionEmbedding,
    LearnedSequencePositionEmbedding,
    SinusoidalSequencePositionEncoding,
    SinePositionEncoding2D,
    TokenTypeEmbedding,
)


def test_sine_position_encoding_has_expected_shape_and_spatial_variation():
    encoder = SinePositionEncoding2D(d_model=32)
    features = torch.zeros(2, 16, 7, 7)

    position = encoder(features)

    assert position.shape == (2, 32, 7, 7)
    assert position.dtype == features.dtype
    assert not torch.equal(position[0, :, 0, 0], position[0, :, 0, 1])
    assert not torch.equal(position[0, :, 0, 0], position[0, :, 1, 0])
    torch.testing.assert_close(position[0], position[1])


def test_learned_sequence_positions_match_tokens_and_differ_by_timestep():
    encoder = LearnedSequencePositionEmbedding(max_length=102, d_model=16)
    tokens = torch.zeros(3, 102, 16)

    position = encoder(tokens)

    assert position.shape == tokens.shape
    assert not torch.equal(position[0, 0], position[0, 1])
    torch.testing.assert_close(position[0], position[2])


def test_fixed_sequence_positions_match_official_act_sine_cosine_convention():
    encoder = SinusoidalSequencePositionEncoding(max_length=5, d_model=8)
    tokens = torch.zeros(2, 5, 8, dtype=torch.float64)

    position = encoder(tokens)

    assert position.shape == tokens.shape
    assert position.dtype == tokens.dtype
    assert not any(parameter.requires_grad for parameter in encoder.parameters())
    torch.testing.assert_close(position[0, 0, 0::2], torch.zeros(4, dtype=torch.float64))
    torch.testing.assert_close(position[0, 0, 1::2], torch.ones(4, dtype=torch.float64))
    assert not torch.equal(position[0, 0], position[0, 1])
    torch.testing.assert_close(position[0], position[1])


def test_camera_positions_repeat_spatially_but_differ_across_cameras():
    encoder = CameraPositionEmbedding(num_cameras=2, d_model=8)

    position = encoder(
        batch_size=3,
        spatial_height=2,
        spatial_width=3,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert position.shape == (3, 12, 8)
    torch.testing.assert_close(position[0, 0], position[0, 5])
    torch.testing.assert_close(position[0, 6], position[0, 11])
    assert not torch.equal(position[0, 0], position[0, 6])
    torch.testing.assert_close(position[0], position[2])


def test_token_type_embedding_supports_shared_and_batched_ids():
    encoder = TokenTypeEmbedding(num_token_types=4, d_model=8)

    shared = encoder(torch.tensor([0, 1, 2, 3]))
    batched = encoder(torch.tensor([[0, 1], [2, 3]]))

    assert shared.shape == (4, 8)
    assert batched.shape == (2, 2, 8)
    assert not torch.equal(shared[0], shared[1])


def test_position_modules_reject_invalid_lengths_and_ids():
    sequence = LearnedSequencePositionEmbedding(max_length=4, d_model=8)
    fixed_sequence = SinusoidalSequencePositionEncoding(max_length=4, d_model=8)
    token_type = TokenTypeEmbedding(num_token_types=2, d_model=8)

    with pytest.raises(ValueError, match="exceeds max_length"):
        sequence(torch.zeros(1, 5, 8))
    with pytest.raises(ValueError, match="exceeds max_length"):
        fixed_sequence(torch.zeros(1, 5, 8))
    with pytest.raises(ValueError, match="out-of-range"):
        token_type(torch.tensor([0, 2]))
