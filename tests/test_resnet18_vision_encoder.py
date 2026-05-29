import pytest
import torch

pytest.importorskip("torchvision")

from force_aware_act.models import ResNet18VisionEncoder


def test_resnet18_vision_encoder_default_shape():
    encoder = ResNet18VisionEncoder(d_model=512, pretrained=False)
    encoder.eval()
    images = torch.randn(2, 2, 3, 224, 224)

    with torch.no_grad():
        tokens = encoder(images)

    assert tokens.shape == (2, 98, 512)


def test_resnet18_vision_encoder_projection_shape():
    encoder = ResNet18VisionEncoder(d_model=256, pretrained=False)
    encoder.eval()
    images = torch.randn(2, 2, 3, 224, 224)

    with torch.no_grad():
        tokens = encoder(images)

    assert tokens.shape == (2, 98, 256)


def test_resnet18_vision_encoder_freezes_backbone():
    encoder = ResNet18VisionEncoder(pretrained=False, freeze_backbone=True)

    assert all(not parameter.requires_grad for parameter in encoder.backbone.parameters())
