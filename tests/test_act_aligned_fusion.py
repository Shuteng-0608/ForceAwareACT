import pytest
import torch

from force_aware_act.models.act_aligned import (
    ACTAlignedConfig,
    ACTAlignedForceVisionFusion,
)


def _small_config(**overrides):
    values = {
        "d_model": 32,
        "nhead": 4,
        "dim_feedforward": 64,
        "dropout": 0.0,
        "image_height": 64,
        "image_width": 64,
        "pretrained_backbone": False,
        "imagenet_normalize": False,
    }
    values.update(overrides)
    return ACTAlignedConfig(**values)


def test_force_vision_fusion_shape_attention_and_gradient():
    config = _small_config()
    fusion = ACTAlignedForceVisionFusion(config)
    online_force = torch.randn(2, config.d_model, requires_grad=True)
    visual = torch.randn(
        2,
        config.visual_token_count,
        config.d_model,
        requires_grad=True,
    )
    position = torch.randn_like(visual)

    fused, attention = fusion(
        online_force,
        visual,
        visual_position=position,
        return_attention=True,
    )
    fused.square().mean().backward()

    assert fused.shape == (2, config.d_model)
    assert attention.shape == (2, 1, config.visual_token_count)
    torch.testing.assert_close(
        attention.sum(dim=-1),
        torch.ones(2, 1),
    )
    assert online_force.grad is not None
    assert visual.grad is not None
    assert fusion.cross_attention.in_proj_weight.grad is not None


def test_force_vision_mask_excludes_padded_visual_tokens():
    torch.manual_seed(11)
    config = _small_config()
    fusion = ACTAlignedForceVisionFusion(config).eval()
    online_force = torch.randn(2, config.d_model)
    visual = torch.randn(2, config.visual_token_count, config.d_model)
    changed = visual.clone()
    changed[:, -2:] = 1000.0
    padding_mask = torch.zeros(
        2,
        config.visual_token_count,
        dtype=torch.bool,
    )
    padding_mask[:, -2:] = True

    first = fusion(
        online_force,
        visual,
        visual_padding_mask=padding_mask,
    )
    second = fusion(
        online_force,
        changed,
        visual_padding_mask=padding_mask,
    )

    torch.testing.assert_close(first, second)


@pytest.mark.parametrize(
    "online_force, visual, message",
    [
        (
            torch.randn(2, 31),
            torch.randn(2, 8, 32),
            "online_force_feature must have shape",
        ),
        (
            torch.randn(2, 32),
            torch.randn(2, 9, 32),
            "visual_tokens must have shape",
        ),
        (
            torch.randn(2, 32),
            torch.randn(3, 8, 32),
            "visual_tokens must have shape",
        ),
    ],
)
def test_force_vision_fusion_rejects_invalid_shapes(
    online_force,
    visual,
    message,
):
    with pytest.raises(ValueError, match=message):
        ACTAlignedForceVisionFusion(_small_config())(online_force, visual)
