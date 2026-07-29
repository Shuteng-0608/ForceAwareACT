"""Spatial, temporal, camera, and token-type position encodings."""

from __future__ import annotations

import math

import torch
from torch import nn


class SinePositionEncoding2D(nn.Module):
    """DETR-style two-dimensional sine position encoding.

    The module accepts a feature map ``[B, C, H, W]`` and returns a positional
    feature map ``[B, d_model, H, W]``. No padding mask is needed for the
    repository's fixed-size resized images.
    """

    def __init__(
        self,
        d_model: int,
        *,
        temperature: float = 10000.0,
        normalize: bool = True,
        scale: float = 2 * math.pi,
    ) -> None:
        super().__init__()
        if not isinstance(d_model, int) or isinstance(d_model, bool) or d_model <= 0:
            raise ValueError("d_model must be a positive integer")
        if d_model % 4 != 0:
            raise ValueError("d_model must be divisible by 4")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if scale <= 0:
            raise ValueError("scale must be positive")
        self.d_model = d_model
        self.num_pos_feats = d_model // 2
        self.temperature = float(temperature)
        self.normalize = bool(normalize)
        self.scale = float(scale)

    def forward(self, feature_map: torch.Tensor) -> torch.Tensor:
        if not isinstance(feature_map, torch.Tensor):
            raise TypeError("feature_map must be a torch.Tensor")
        if feature_map.ndim != 4:
            raise ValueError("feature_map must have shape [B, C, H, W]")
        if not feature_map.is_floating_point():
            raise ValueError("feature_map must be floating point")

        batch_size, _channels, height, width = feature_map.shape
        valid = torch.ones(
            (batch_size, height, width),
            dtype=torch.bool,
            device=feature_map.device,
        )
        y_embed = valid.cumsum(1, dtype=torch.float32)
        x_embed = valid.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1.0e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(
            self.num_pos_feats,
            dtype=torch.float32,
            device=feature_map.device,
        )
        dim_t = self.temperature ** (
            2 * torch.div(dim_t, 2, rounding_mode="floor") / self.num_pos_feats
        )

        pos_x = x_embed[..., None] / dim_t
        pos_y = y_embed[..., None] / dim_t
        pos_x = torch.stack(
            (pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()),
            dim=-1,
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()),
            dim=-1,
        ).flatten(3)
        position = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return position.to(dtype=feature_map.dtype)


class LearnedSequencePositionEmbedding(nn.Module):
    """Learned positions for a batch-first token sequence."""

    def __init__(self, max_length: int, d_model: int) -> None:
        super().__init__()
        _validate_positive_int(max_length, "max_length")
        _validate_positive_int(d_model, "d_model")
        self.max_length = max_length
        self.d_model = d_model
        self.embedding = nn.Embedding(max_length, d_model)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        _validate_token_tensor(tokens, self.d_model, "tokens")
        batch_size, sequence_length, _ = tokens.shape
        if sequence_length > self.max_length:
            raise ValueError(
                f"sequence length {sequence_length} exceeds max_length {self.max_length}"
            )
        indices = torch.arange(sequence_length, device=tokens.device)
        position = self.embedding(indices).to(dtype=tokens.dtype)
        return position.unsqueeze(0).expand(batch_size, -1, -1)


class SinusoidalSequencePositionEncoding(nn.Module):
    """Fixed ACT-style sine/cosine positions for a batch-first sequence."""

    def __init__(
        self,
        max_length: int,
        d_model: int,
        *,
        temperature: float = 10000.0,
    ) -> None:
        super().__init__()
        _validate_positive_int(max_length, "max_length")
        _validate_positive_int(d_model, "d_model")
        if d_model % 2 != 0:
            raise ValueError("d_model must be even")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.max_length = max_length
        self.d_model = d_model
        self.temperature = float(temperature)

        position = torch.arange(max_length, dtype=torch.float32).unsqueeze(1)
        frequency = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(self.temperature) / d_model)
        )
        table = torch.zeros(max_length, d_model, dtype=torch.float32)
        table[:, 0::2] = torch.sin(position * frequency)
        table[:, 1::2] = torch.cos(position * frequency)
        self.register_buffer("table", table.unsqueeze(0), persistent=True)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        _validate_token_tensor(tokens, self.d_model, "tokens")
        batch_size, sequence_length, _ = tokens.shape
        if sequence_length > self.max_length:
            raise ValueError(
                f"sequence length {sequence_length} exceeds max_length {self.max_length}"
            )
        return self.table[:, :sequence_length].to(
            device=tokens.device,
            dtype=tokens.dtype,
        ).expand(batch_size, -1, -1)


class CameraPositionEmbedding(nn.Module):
    """Expand one learned identity embedding over every camera spatial token."""

    def __init__(self, num_cameras: int, d_model: int) -> None:
        super().__init__()
        _validate_positive_int(num_cameras, "num_cameras")
        _validate_positive_int(d_model, "d_model")
        self.num_cameras = num_cameras
        self.d_model = d_model
        self.embedding = nn.Embedding(num_cameras, d_model)

    def forward(
        self,
        *,
        batch_size: int,
        spatial_height: int,
        spatial_width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        _validate_positive_int(batch_size, "batch_size")
        _validate_positive_int(spatial_height, "spatial_height")
        _validate_positive_int(spatial_width, "spatial_width")
        if not dtype.is_floating_point:
            raise ValueError("dtype must be floating point")
        camera_ids = torch.arange(self.num_cameras, device=device)
        camera_position = self.embedding(camera_ids).to(dtype=dtype)
        camera_position = camera_position[:, None, :].expand(
            -1,
            spatial_height * spatial_width,
            -1,
        )
        return camera_position.reshape(1, -1, self.d_model).expand(
            batch_size,
            -1,
            -1,
        )


class TokenTypeEmbedding(nn.Module):
    """Learned embeddings for explicit token-type identifiers."""

    def __init__(self, num_token_types: int, d_model: int) -> None:
        super().__init__()
        _validate_positive_int(num_token_types, "num_token_types")
        _validate_positive_int(d_model, "d_model")
        self.num_token_types = num_token_types
        self.d_model = d_model
        self.embedding = nn.Embedding(num_token_types, d_model)

    def forward(self, type_ids: torch.Tensor) -> torch.Tensor:
        if not isinstance(type_ids, torch.Tensor):
            raise TypeError("type_ids must be a torch.Tensor")
        if type_ids.ndim not in (1, 2):
            raise ValueError("type_ids must have shape [S] or [B, S]")
        if type_ids.dtype != torch.long:
            raise ValueError("type_ids must have dtype torch.long")
        if type_ids.numel() == 0:
            raise ValueError("type_ids must not be empty")
        if type_ids.min().item() < 0 or type_ids.max().item() >= self.num_token_types:
            raise ValueError("type_ids contain an out-of-range token type")
        return self.embedding(type_ids)


def _validate_positive_int(value: int, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _validate_token_tensor(tensor: torch.Tensor, d_model: int, name: str) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.ndim != 3 or tensor.shape[-1] != d_model:
        raise ValueError(f"{name} must have shape [B, S, {d_model}]")
    if not tensor.is_floating_point():
        raise ValueError(f"{name} must be floating point")
