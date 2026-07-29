"""Official ACT shared ResNet-18 multi-camera feature layout."""

from __future__ import annotations

import torch
from torch import nn

from force_aware_act.models.act_aligned.backbone import (
    FrozenBatchNorm2d,
    _make_resnet18,
)
from force_aware_act.models.act_aligned.position_encoding import (
    SinePositionEncoding2D,
)
from force_aware_act.models.official_act.config import OfficialACTConfig


class OfficialACTBackbone(nn.Module):
    """Share one ResNet and concatenate camera features along image width."""

    output_channels = 512

    def __init__(self, config: OfficialACTConfig) -> None:
        super().__init__()
        self.config = config
        resnet = _make_resnet18(
            pretrained=config.pretrained_backbone,
            frozen_batch_norm=config.frozen_batch_norm,
        )
        self.body = nn.Sequential(*list(resnet.children())[:-2])
        self.input_projection = nn.Conv2d(
            self.output_channels,
            config.d_model,
            kernel_size=1,
        )
        self.position_encoding = SinePositionEncoding2D(config.d_model)

    def forward(
        self,
        images: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._validate(images)
        features = []
        positions = []
        for camera_index in range(self.config.num_cameras):
            feature = self.input_projection(
                self.body(images[:, camera_index])
            )
            features.append(feature)
            positions.append(self.position_encoding(feature))
        feature_map = torch.cat(features, dim=3)
        position_map = torch.cat(positions, dim=3)
        tokens = feature_map.flatten(2).transpose(1, 2)
        positions_flat = position_map.flatten(2).transpose(1, 2)
        expected = (
            images.shape[0],
            self.config.visual_token_count,
            self.config.d_model,
        )
        if tokens.shape != expected:
            raise RuntimeError(
                f"official visual tokens expected {expected}, "
                f"got {tuple(tokens.shape)}"
            )
        return tokens, positions_flat

    def _validate(self, images: torch.Tensor) -> None:
        if not isinstance(images, torch.Tensor):
            raise TypeError("images must be a torch.Tensor")
        expected = (
            self.config.num_cameras,
            3,
            self.config.image_height,
            self.config.image_width,
        )
        if images.ndim != 5 or tuple(images.shape[1:]) != expected:
            raise ValueError(
                f"images must have shape [B, {expected[0]}, 3, "
                f"{expected[2]}, {expected[3]}]"
            )
        if not images.is_floating_point():
            raise ValueError("images must be floating point")


__all__ = ["FrozenBatchNorm2d", "OfficialACTBackbone"]
