import pytest
import torch
from torch import nn

pytest.importorskip("torchvision")

from force_aware_act.models.act_aligned import (  # noqa: E402
    ACTAlignedConfig,
    ACTAlignedResNet18Backbone,
    FrozenBatchNorm2d,
)


def _config(**overrides):
    return ACTAlignedConfig.compact_smoke(**overrides)


def test_frozen_batch_norm_has_buffers_but_no_parameters():
    normalization = FrozenBatchNorm2d(8)

    assert dict(normalization.named_parameters()) == {}
    assert set(dict(normalization.named_buffers())) == {
        "weight",
        "bias",
        "running_mean",
        "running_var",
    }


def test_frozen_batch_norm_matches_fixed_batch_norm_formula():
    normalization = FrozenBatchNorm2d(2)
    inputs = torch.randn(3, 2, 4, 5)

    expected = inputs / torch.sqrt(torch.ones(2).reshape(1, 2, 1, 1) + 1.0e-5)

    torch.testing.assert_close(normalization(inputs), expected)


def test_backbone_uses_shared_resnet_and_returns_camera_major_tokens():
    config = _config()
    backbone = ACTAlignedResNet18Backbone(config)
    images = torch.randn(
        2,
        config.num_cameras,
        3,
        config.image_height,
        config.image_width,
    )

    tokens, position = backbone(images)

    assert tokens.shape == (2, config.visual_token_count, config.d_model)
    assert position.shape == tokens.shape
    assert len([module for module in backbone.modules() if isinstance(module, nn.Conv2d)]) == 21
    assert any(isinstance(module, FrozenBatchNorm2d) for module in backbone.body.modules())
    assert not torch.equal(
        position[:, :49],
        position[:, 49:],
    )


def test_backbone_supports_gradient_through_resnet_and_projection():
    config = _config(image_height=64, image_width=64)
    backbone = ACTAlignedResNet18Backbone(config)
    images = torch.randn(1, config.num_cameras, 3, 64, 64)

    tokens, position = backbone(images)
    (tokens.square().mean() + position.square().mean()).backward()

    assert backbone.body[0].weight.grad is not None
    assert backbone.input_projection.weight.grad is not None
    assert backbone.camera_position.embedding.weight.grad is not None


@pytest.mark.parametrize(
    "images, message",
    [
        (torch.randn(2, 3, 224, 224), "images must have shape"),
        (torch.randn(2, 1, 3, 224, 224), "images must have shape"),
        (torch.randn(2, 2, 3, 128, 224), "images must have shape"),
        (
            torch.ones(2, 2, 3, 64, 64, dtype=torch.int64),
            "floating point",
        ),
    ],
)
def test_backbone_rejects_invalid_image_contract(images, message):
    backbone = ACTAlignedResNet18Backbone(_config())

    with pytest.raises(ValueError, match=message):
        backbone(images)
