"""Immutable configuration for the faithful official ACT baseline."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any


OFFICIAL_ACT_ARCHITECTURE_VERSION = "official_act_single_arm_v1"


@dataclass(frozen=True)
class OfficialACTConfig:
    """Official ACT hyperparameters with only task dimensions adapted."""

    architecture_version: str = OFFICIAL_ACT_ARCHITECTURE_VERSION
    d_model: int = 512
    nhead: int = 8
    dim_feedforward: int = 3200
    encoder_layers: int = 4
    decoder_layers: int = 7
    dropout: float = 0.1
    latent_dim: int = 32
    q_dim: int = 7
    action_dim: int = 7
    chunk_len: int = 100
    num_cameras: int = 2
    image_height: int = 480
    image_width: int = 640
    backbone_output_stride: int = 32
    backbone_name: str = "resnet18"
    pretrained_backbone: bool = True
    frozen_batch_norm: bool = True
    imagenet_normalize: bool = True

    def __post_init__(self) -> None:
        for name in (
            "d_model",
            "nhead",
            "dim_feedforward",
            "encoder_layers",
            "decoder_layers",
            "latent_dim",
            "q_dim",
            "action_dim",
            "chunk_len",
            "num_cameras",
            "image_height",
            "image_width",
            "backbone_output_stride",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.architecture_version != OFFICIAL_ACT_ARCHITECTURE_VERSION:
            raise ValueError(
                "architecture_version must be "
                f"{OFFICIAL_ACT_ARCHITECTURE_VERSION!r}"
            )
        if self.d_model % self.nhead != 0 or self.d_model % 4 != 0:
            raise ValueError("d_model must be divisible by nhead and by 4")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.backbone_name != "resnet18":
            raise ValueError("official ACT baseline requires ResNet-18")
        if self.pretrained_backbone != self.imagenet_normalize:
            raise ValueError(
                "pretrained_backbone and imagenet_normalize must match"
            )

    @classmethod
    def canonical(cls, **overrides: Any) -> "OfficialACTConfig":
        return cls(**overrides)

    @classmethod
    def compact_smoke(cls, **overrides: Any) -> "OfficialACTConfig":
        values: dict[str, Any] = {
            "d_model": 32,
            "nhead": 4,
            "dim_feedforward": 64,
            "dropout": 0.0,
            "chunk_len": 6,
            "image_height": 64,
            "image_width": 64,
            "pretrained_backbone": False,
            "imagenet_normalize": False,
        }
        values.update(overrides)
        return cls(**values)

    @property
    def visual_grid_height(self) -> int:
        return math.ceil(self.image_height / self.backbone_output_stride)

    @property
    def visual_grid_width_per_camera(self) -> int:
        return math.ceil(self.image_width / self.backbone_output_stride)

    @property
    def visual_token_count(self) -> int:
        return (
            self.num_cameras
            * self.visual_grid_height
            * self.visual_grid_width_per_camera
        )

    @property
    def posterior_token_count(self) -> int:
        return self.chunk_len + 2

    @property
    def policy_memory_token_count(self) -> int:
        return self.visual_token_count + 2

    def checkpoint_metadata(self) -> dict[str, Any]:
        metadata = asdict(self)
        metadata.update(
            {
                "source_implementation": "/home/stw/act",
                "posterior_layout": "cls_qpos_action",
                "posterior_encoder_layers": self.encoder_layers,
                "policy_encoder_layers": self.encoder_layers,
                "policy_decoder_layers": self.decoder_layers,
                "camera_feature_layout": "concatenate_feature_width",
                "policy_memory_layout": "latent_qpos_visual",
                "deployment_latent": "zero",
                "visual_token_count": self.visual_token_count,
                "policy_memory_token_count": self.policy_memory_token_count,
            }
        )
        return metadata
