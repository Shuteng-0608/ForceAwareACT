import pytest
import torch

from force_aware_act.act_aligned_training import ACTAlignedBatch
from force_aware_act.models.act_aligned import ACTAlignedConfig


def _config():
    return ACTAlignedConfig(
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


def _batch(config, batch_size=2):
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


def test_batch_validates_exact_new_training_contract():
    config = _config()
    batch = _batch(config)

    batch.validate(config)

    assert batch.batch_size == 2
    assert batch.action_chunk.shape == (2, 6, 7)
    assert batch.future_force_chunk.shape == (2, 6, 6)


def test_batch_to_moves_every_tensor_and_preserves_mask_dtypes():
    config = _config()
    moved = _batch(config).to("cpu")

    moved.validate(config)
    assert moved.force_padding_mask.dtype is torch.bool
    assert moved.future_padding_mask.dtype is torch.bool


def test_batch_rejects_all_padded_future_targets():
    config = _config()
    batch = _batch(config)
    invalid = ACTAlignedBatch(
        **{
            **batch.__dict__,
            "future_padding_mask": torch.ones_like(batch.future_padding_mask),
        }
    )

    with pytest.raises(ValueError, match="at least one valid step"):
        invalid.validate(config)


@pytest.mark.parametrize(
    "field, replacement, message",
    [
        ("action_chunk", torch.randn(2, 5, 7), "action_chunk must have shape"),
        (
            "future_force_chunk",
            torch.randn(2, 6, 5),
            "future_force_chunk must have shape",
        ),
        (
            "force_padding_mask",
            torch.zeros(2, 5),
            "force_padding_mask must have dtype torch.bool",
        ),
        ("qpos", torch.randn(3, 7), "same batch size"),
    ],
)
def test_batch_rejects_invalid_contract(field, replacement, message):
    config = _config()
    batch = _batch(config)
    invalid = ACTAlignedBatch(**{**batch.__dict__, field: replacement})

    with pytest.raises(ValueError, match=message):
        invalid.validate(config)
