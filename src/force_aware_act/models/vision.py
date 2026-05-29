"""Vision encoder modules for ForceAwareACT."""

from __future__ import annotations

import torch
from torch import nn


def _make_resnet18(pretrained: bool) -> nn.Module:
    try:
        from torchvision.models import ResNet18_Weights, resnet18

        weights = ResNet18_Weights.DEFAULT if pretrained else None
        return resnet18(weights=weights)
    except ImportError as error:
        raise ImportError(
            "ResNet18VisionEncoder requires torchvision. "
            "Install the project with its vision dependencies before using this module."
        ) from error
    except (AttributeError, TypeError):
        from torchvision.models import resnet18

        return resnet18(pretrained=pretrained)


class ResNet18VisionEncoder(nn.Module):
    """Convert multi-camera RGB images into spatial visual tokens.

    Input shape:
        images: [B, N_cam, 3, H, W]

    Output shape for 224x224 inputs:
        visual_tokens: [B, N_cam * 49, d_model]
    """

    backbone_dim = 512

    def __init__(
        self,
        d_model: int = 512,
        pretrained: bool = True,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError("d_model must be positive")

        resnet = _make_resnet18(pretrained=pretrained)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.visual_proj = (
            nn.Identity()
            if d_model == self.backbone_dim
            else nn.Linear(self.backbone_dim, d_model)
        )
        self.d_model = d_model

        if freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 5:
            raise ValueError("images must have shape [B, N_cam, 3, H, W]")
        batch_size, num_cameras, channels, height, width = images.shape
        if channels != 3:
            raise ValueError("images must have 3 RGB channels")

        flat_images = images.reshape(batch_size * num_cameras, channels, height, width)
        features = self.backbone(flat_images)
        tokens = features.flatten(2).transpose(1, 2)
        tokens = tokens.reshape(batch_size, num_cameras * tokens.shape[1], self.backbone_dim)
        return self.visual_proj(tokens)
