"""ACT-compatible ResNet18 visual tokenization for multiple cameras."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from force_aware_act.models.act_aligned.config import ACTAlignedConfig
from force_aware_act.models.act_aligned.position_encoding import (
    CameraPositionEmbedding,
    SinePositionEncoding2D,
)


class FrozenBatchNorm2d(nn.Module):
    """BatchNorm2d with fixed statistics and affine parameters.

    This follows the FrozenBatchNorm implementation used by the official
    ACT/DETR backbone, including the epsilon placement in the reciprocal
    square root.
    """

    def __init__(self, num_features: int) -> None:
        super().__init__()
        if (
            not isinstance(num_features, int)
            or isinstance(num_features, bool)
            or num_features <= 0
        ):
            raise ValueError("num_features must be a positive integer")
        self.num_features = num_features
        self.register_buffer("weight", torch.ones(num_features))
        self.register_buffer("bias", torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))

    def _load_from_state_dict(
        self,
        state_dict: dict[str, Any],
        prefix: str,
        local_metadata: dict[str, Any],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        state_dict.pop(prefix + "num_batches_tracked", None)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 4 or inputs.shape[1] != self.num_features:
            raise ValueError(
                "inputs must have shape "
                f"[B, {self.num_features}, H, W], got {tuple(inputs.shape)}"
            )
        weight = self.weight.reshape(1, -1, 1, 1)
        bias = self.bias.reshape(1, -1, 1, 1)
        running_var = self.running_var.reshape(1, -1, 1, 1)
        running_mean = self.running_mean.reshape(1, -1, 1, 1)
        scale = weight * (running_var + 1.0e-5).rsqrt()
        offset = bias - running_mean * scale
        return inputs * scale + offset


def _make_resnet18(*, pretrained: bool, frozen_batch_norm: bool) -> nn.Module:
    try:
        from torchvision.models import ResNet18_Weights, resnet18
    except ImportError as exc:
        raise RuntimeError(
            "ACTAlignedResNet18Backbone requires torchvision"
        ) from exc

    norm_layer = FrozenBatchNorm2d if frozen_batch_norm else nn.BatchNorm2d
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    return resnet18(weights=weights, norm_layer=norm_layer)


class ACTAlignedResNet18Backbone(nn.Module):
    """Encode all cameras with one shared ACT-style ResNet18.

    Input images use ``[B, num_cameras, 3, H, W]``. The output token order is
    camera-major, then row-major within each feature grid:
    ``[B, num_cameras * H' * W', d_model]``.

    ImageNet normalization remains a data-pipeline responsibility. The
    ``config.imagenet_normalize`` flag records that input contract; this module
    never normalizes an image a second time.
    """

    output_channels = 512

    def __init__(self, config: ACTAlignedConfig) -> None:
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
        self.spatial_position = SinePositionEncoding2D(config.d_model)
        self.camera_position = CameraPositionEmbedding(
            config.num_cameras,
            config.d_model,
        )

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self._validate_images(images)
        batch_size, num_cameras, channels, height, width = images.shape
        flat_images = images.reshape(
            batch_size * num_cameras,
            channels,
            height,
            width,
        )
        feature_map = self.input_projection(self.body(flat_images))
        _flat_batch, _d_model, grid_height, grid_width = feature_map.shape

        expected_grid = (
            self.config.visual_grid_height,
            self.config.visual_grid_width,
        )
        if (grid_height, grid_width) != expected_grid:
            raise RuntimeError(
                "ResNet18 feature grid differs from the configuration contract: "
                f"expected {expected_grid}, got {(grid_height, grid_width)}"
            )

        spatial_position = self.spatial_position(feature_map)
        tokens = feature_map.flatten(2).transpose(1, 2).reshape(
            batch_size,
            num_cameras * grid_height * grid_width,
            self.config.d_model,
        )
        position = spatial_position.flatten(2).transpose(1, 2).reshape(
            batch_size,
            num_cameras * grid_height * grid_width,
            self.config.d_model,
        )
        position = position + self.camera_position(
            batch_size=batch_size,
            spatial_height=grid_height,
            spatial_width=grid_width,
            device=images.device,
            dtype=feature_map.dtype,
        )
        return tokens, position

    def _validate_images(self, images: torch.Tensor) -> None:
        if not isinstance(images, torch.Tensor):
            raise TypeError("images must be a torch.Tensor")
        expected_shape = (
            self.config.num_cameras,
            3,
            self.config.image_height,
            self.config.image_width,
        )
        if images.ndim != 5 or tuple(images.shape[1:]) != expected_shape:
            raise ValueError(
                "images must have shape "
                f"[B, {expected_shape[0]}, 3, {expected_shape[2]}, "
                f"{expected_shape[3]}], got {tuple(images.shape)}"
            )
        if not images.is_floating_point():
            raise ValueError("images must be floating point")
