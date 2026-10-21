"""Immutable configuration for the ACT-aligned model family."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

from force_aware_act.high_rate_force import HIGH_RATE_FORCE_CONTRACT_VERSION


ACT_ALIGNED_ARCHITECTURE_VERSION = "act_aligned_contact_cvae_v1"
ACT_ALIGNED_MOTION_CONTROL_ARCHITECTURE_VERSION = (
    "act_aligned_motion_cvae_control_v1"
)
ACT_ALIGNED_HIGH_RATE_ARCHITECTURE_VERSION = (
    "act_aligned_contact_cvae_highrate_force_v2"
)
ACT_ALIGNED_HIGH_RATE_MOTION_ARCHITECTURE_VERSION = (
    "act_aligned_motion_cvae_highrate_force_v2"
)
ACT_ALIGNED_HIGH_RATE_DUAL_ZERO_ARCHITECTURE_VERSION = (
    "act_aligned_dual_zero_highrate_force_v1"
)

_MOTION_ARCHITECTURE_VERSIONS = {
    ACT_ALIGNED_MOTION_CONTROL_ARCHITECTURE_VERSION,
    ACT_ALIGNED_HIGH_RATE_MOTION_ARCHITECTURE_VERSION,
}
_DUAL_ZERO_ARCHITECTURE_VERSIONS = {
    ACT_ALIGNED_HIGH_RATE_DUAL_ZERO_ARCHITECTURE_VERSION,
}


@dataclass(frozen=True)
class ACTAlignedConfig:
    """Single source of truth for ACT-aligned model dimensions.

    All Transformer encoders deliberately share ``encoder_layers`` as their
    default depth.  A future experiment may introduce explicit per-encoder
    overrides, but it must do so through a new architecture version rather
    than by silently hard-coding a different value in a training script.
    """

    architecture_version: str = ACT_ALIGNED_ARCHITECTURE_VERSION

    d_model: int = 512
    nhead: int = 8
    dim_feedforward: int = 3200
    encoder_layers: int = 4
    decoder_layers: int = 7
    dropout: float = 0.1
    activation: str = "relu"
    norm_first: bool = False

    latent_dim: int = 32
    q_dim: int = 7
    action_dim: int = 7
    force_dim: int = 6
    chunk_len: int = 100
    force_window_len: int = 20

    num_cameras: int = 2
    image_height: int = 480
    image_width: int = 640
    backbone_output_stride: int = 32
    backbone_name: str = "resnet18"
    pretrained_backbone: bool = True
    frozen_batch_norm: bool = True
    imagenet_normalize: bool = True

    def __post_init__(self) -> None:
        positive_int_fields = (
            "d_model",
            "nhead",
            "dim_feedforward",
            "encoder_layers",
            "decoder_layers",
            "latent_dim",
            "q_dim",
            "action_dim",
            "force_dim",
            "chunk_len",
            "force_window_len",
            "num_cameras",
            "image_height",
            "image_width",
            "backbone_output_stride",
        )
        for field_name in positive_int_fields:
            value = getattr(self, field_name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{field_name} must be a positive integer")

        if self.d_model % self.nhead != 0:
            raise ValueError("d_model must be divisible by nhead")
        if self.d_model % 4 != 0:
            raise ValueError("d_model must be divisible by 4 for 2D sine position encoding")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.activation != "relu":
            raise ValueError("ACT-aligned v1 requires activation='relu'")
        if self.norm_first:
            raise ValueError("ACT-aligned v1 requires post-norm Transformer layers")
        supported_versions = {
            ACT_ALIGNED_ARCHITECTURE_VERSION,
            ACT_ALIGNED_MOTION_CONTROL_ARCHITECTURE_VERSION,
            ACT_ALIGNED_HIGH_RATE_ARCHITECTURE_VERSION,
            ACT_ALIGNED_HIGH_RATE_MOTION_ARCHITECTURE_VERSION,
            ACT_ALIGNED_HIGH_RATE_DUAL_ZERO_ARCHITECTURE_VERSION,
        }
        if self.architecture_version not in supported_versions:
            raise ValueError(
                "architecture_version must be one of "
                f"{sorted(supported_versions)!r}"
            )
        if self.backbone_name != "resnet18":
            raise ValueError("ACT-aligned v1 requires backbone_name='resnet18'")
        if self.pretrained_backbone != self.imagenet_normalize:
            raise ValueError(
                "pretrained_backbone and imagenet_normalize must be enabled or "
                "disabled together"
            )

    @classmethod
    def canonical_act(cls, **overrides: Any) -> "ACTAlignedConfig":
        """Return the canonical ACT-depth configuration.

        The defaults match the official ALOHA ACT example's ``chunk_len=100``
        and the repository dataset's native ``480 x 640`` camera resolution.
        Experiments that intentionally use a different prediction horizon or
        image resolution must override it explicitly; checkpoints record both.
        """

        return cls(**overrides)

    @classmethod
    def motion_control(cls, **overrides: Any) -> "ACTAlignedConfig":
        """Return the action-only ACT motion-latent control configuration."""

        values: dict[str, Any] = {
            "architecture_version": (
                ACT_ALIGNED_MOTION_CONTROL_ARCHITECTURE_VERSION
            )
        }
        values.update(overrides)
        return cls(**values)

    @classmethod
    def compact_smoke(cls, **overrides: Any) -> "ACTAlignedConfig":
        """Return a cheap preset for shape, forward, and gradient smoke tests.

        Encoder and decoder depth intentionally remain ACT-aligned.  The preset
        reduces model width and spatial input size, so a smoke test cannot
        accidentally validate a one-layer architecture that differs
        structurally from the canonical model.
        """

        values: dict[str, Any] = {
            "d_model": 128,
            "nhead": 4,
            "dim_feedforward": 256,
            "dropout": 0.0,
            "image_height": 64,
            "image_width": 64,
            "pretrained_backbone": False,
            "imagenet_normalize": False,
        }
        values.update(overrides)
        return cls(**values)

    @property
    def attention_head_dim(self) -> int:
        return self.d_model // self.nhead

    @property
    def visual_grid_height(self) -> int:
        return math.ceil(self.image_height / self.backbone_output_stride)

    @property
    def visual_grid_width(self) -> int:
        return math.ceil(self.image_width / self.backbone_output_stride)

    @property
    def visual_token_count(self) -> int:
        return self.num_cameras * self.visual_grid_height * self.visual_grid_width

    @property
    def contact_posterior_token_count(self) -> int:
        # [CLS, qpos, K time-aligned action-force tokens]
        return self.chunk_len + 2

    @property
    def motion_posterior_token_count(self) -> int:
        # Official ACT layout: [CLS, qpos, K action tokens]
        return self.chunk_len + 2

    @property
    def force_encoder_token_count(self) -> int:
        # [CLS_F, L historical force tokens]
        return self.force_window_len + 1

    @property
    def policy_memory_token_count(self) -> int:
        if self.architecture_version in _DUAL_ZERO_ARCHITECTURE_VERSIONS:
            # [qpos, z_F_online, z_VF, visual tokens]
            return self.visual_token_count + 3
        # [z, qpos, z_F_online, z_VF, visual tokens]
        return self.visual_token_count + 4

    def checkpoint_metadata(self) -> dict[str, Any]:
        """Return explicit, serializable architecture metadata."""

        metadata = asdict(self)
        shared_metadata = {
            "attention_head_dim": self.attention_head_dim,
            "online_force_encoder_layers": self.encoder_layers,
            "policy_encoder_layers": self.encoder_layers,
            "policy_decoder_layers": self.decoder_layers,
            "decoder_output_layer": "last",
            "token_layout": "batch_first",
            "prediction_head_input": "final_decoder_hidden",
            "action_head": f"linear_{self.d_model}_to_{self.action_dim}",
            "force_head": f"linear_{self.d_model}_to_{self.force_dim}",
            "force_head_contact_concat": False,
            "visual_token_count": self.visual_token_count,
            "force_encoder_token_count": self.force_encoder_token_count,
            "policy_memory_token_count": self.policy_memory_token_count,
        }
        metadata.update(shared_metadata)
        if self.architecture_version in _MOTION_ARCHITECTURE_VERSIONS:
            metadata.update(
                {
                    "motion_posterior_encoder_layers": self.encoder_layers,
                    "motion_posterior_layout": "official_action_only",
                    "motion_posterior_inputs": ("qpos", "action_chunk"),
                    "deployment_motion_latent": "zero",
                    "deployment_motion_latent_modes": (
                        "zero",
                        "offline_override",
                    ),
                    "motion_posterior_token_count": (
                        self.motion_posterior_token_count
                    ),
                }
            )
            return metadata

        if self.architecture_version in _DUAL_ZERO_ARCHITECTURE_VERSIONS:
            metadata.update(
                {
                    "latent_mechanism": "none",
                    "uses_motion_latent": False,
                    "uses_contact_latent": False,
                    "deployment_latent_modes": (),
                }
            )
            return metadata

        metadata.update(
            {
                "contact_posterior_encoder_layers": self.encoder_layers,
                "contact_posterior_layout": "time_aligned_action_plus_force",
                "contact_prior_inputs": (
                    "qpos_feature",
                    "online_force_feature",
                    "force_vision_feature",
                    "visual_summary",
                ),
                "deployment_contact_latent": "zero",
                "deployment_contact_latent_modes": (
                    "zero",
                    "conditional_prior_mean",
                    "conditional_prior_sample",
                    "offline_override",
                ),
                "contact_posterior_token_count": self.contact_posterior_token_count,
            }
        )
        return metadata


@dataclass(frozen=True)
class ACTAlignedHighRateConfig(ACTAlignedConfig):
    """ACT-aligned contact policy with native 500 Hz force intervals."""

    architecture_version: str = ACT_ALIGNED_HIGH_RATE_ARCHITECTURE_VERSION
    # None preserves legacy checkpoints: force spans the entire action chunk.
    force_prediction_steps: int | None = None
    strict_force_horizon: bool = False
    force_sample_rate_hz: float = 500.0
    policy_sample_rate_hz: float = 30.0
    online_force_window_len: int = 100
    max_force_samples_per_interval: int = 20
    max_online_force_intervals: int = 7
    # Online history may have wider intervals; future heads retain their capacity.
    max_online_force_samples_per_interval: int | None = None
    online_force_interval_duration_s: float | None = None
    local_force_dim: int = 128
    local_force_encoder_layers: int = 4

    def __post_init__(self) -> None:
        super().__post_init__()
        if type(self.strict_force_horizon) is not bool:
            raise ValueError("strict_force_horizon must be a bool")
        if self.force_prediction_steps is not None:
            if type(self.force_prediction_steps) is not int or not 1 <= self.force_prediction_steps <= self.chunk_len:
                raise ValueError("force_prediction_steps must be an integer between 1 and chunk_len")
        high_rate_versions = {
            ACT_ALIGNED_HIGH_RATE_ARCHITECTURE_VERSION,
            ACT_ALIGNED_HIGH_RATE_MOTION_ARCHITECTURE_VERSION,
            ACT_ALIGNED_HIGH_RATE_DUAL_ZERO_ARCHITECTURE_VERSION,
        }
        if self.architecture_version not in high_rate_versions:
            raise ValueError("high-rate config requires a high-rate architecture version")
        for field_name in ("force_sample_rate_hz", "policy_sample_rate_hz"):
            value = getattr(self, field_name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{field_name} must be finite and positive")
        for field_name in (
            "online_force_window_len",
            "max_force_samples_per_interval",
            "max_online_force_intervals",
            "local_force_dim",
            "local_force_encoder_layers",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{field_name} must be a positive integer")
        if self.local_force_encoder_layers != self.encoder_layers:
            raise ValueError(
                "local_force_encoder_layers must match ACT encoder_layers"
            )
        if self.local_force_dim % 2 != 0:
            raise ValueError("local_force_dim must be even for time encoding")
        capacity = self.max_online_force_samples_per_interval
        if capacity is not None and (type(capacity) is not int or capacity <= 0):
            raise ValueError("max_online_force_samples_per_interval must be a positive integer or None")
        duration = self.online_force_interval_duration_s
        if duration is not None:
            if isinstance(duration, bool) or not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration <= 0:
                raise ValueError("online_force_interval_duration_s must be finite and positive")
            if not math.isclose(duration * self.max_online_force_intervals,
                                self.online_force_window_len / self.force_sample_rate_hz):
                raise ValueError("online interval duration/count must cover the nominal history window")
            if self.online_force_sample_capacity < math.ceil(duration * self.force_sample_rate_hz):
                raise ValueError("online sample capacity cannot hold the nominal history interval")

    @property
    def online_force_sample_capacity(self) -> int:
        return (self.max_force_samples_per_interval if self.max_online_force_samples_per_interval is None
                else self.max_online_force_samples_per_interval)

    @property
    def online_force_interval_s(self) -> float:
        return (1.0 / self.policy_sample_rate_hz if self.online_force_interval_duration_s is None
                else self.online_force_interval_duration_s)

    @property
    def bounds_force_source_timestamps(self) -> bool:
        return self.strict_force_horizon or self.force_prediction_len < self.chunk_len

    @property
    def force_prediction_len(self) -> int:
        return self.chunk_len if self.force_prediction_steps is None else self.force_prediction_steps

    @property
    def force_encoder_token_count(self) -> int:
        return self.max_online_force_intervals + 1

    @classmethod
    def motion_control(cls, **overrides: Any) -> "ACTAlignedHighRateConfig":
        values: dict[str, Any] = {
            "architecture_version": (
                ACT_ALIGNED_HIGH_RATE_MOTION_ARCHITECTURE_VERSION
            )
        }
        values.update(overrides)
        return cls(**values)

    @classmethod
    def dual_zero(cls, **overrides: Any) -> "ACTAlignedHighRateConfig":
        values: dict[str, Any] = {
            "architecture_version": (
                ACT_ALIGNED_HIGH_RATE_DUAL_ZERO_ARCHITECTURE_VERSION
            )
        }
        values.update(overrides)
        return cls(**values)

    def checkpoint_metadata(self) -> dict[str, Any]:
        metadata = super().checkpoint_metadata()
        metadata.update(
            {
                "force_input_contract": (
                    HIGH_RATE_FORCE_CONTRACT_VERSION
                ),
                "strict_force_horizon": self.strict_force_horizon,
                "force_source_timestamps_bounded": self.bounds_force_source_timestamps,
                "force_prediction_steps": self.force_prediction_len,
                "force_prediction_duration_s": self.force_prediction_len / self.policy_sample_rate_hz,
                "force_sample_rate_hz": self.force_sample_rate_hz,
                "policy_sample_rate_hz": self.policy_sample_rate_hz,
                "online_force_window_len": self.online_force_window_len,
                "max_force_samples_per_interval": (
                    self.max_force_samples_per_interval
                ),
                "max_online_force_intervals": self.max_online_force_intervals,
                "online_force_sample_capacity": self.online_force_sample_capacity,
                "online_force_interval_s": self.online_force_interval_s,
                "local_force_dim": self.local_force_dim,
                "local_force_encoder_layers": self.local_force_encoder_layers,
                "shared_local_force_encoder": True,
                "future_force_interval_boundary": "(t_j,t_j+1]",
                "legacy_force_window_len_role": "unused_by_v2",
            }
        )
        if self.architecture_version == ACT_ALIGNED_HIGH_RATE_ARCHITECTURE_VERSION:
            metadata["contact_posterior_layout"] = (
                "time_aligned_action_plus_encoded_500hz_interval_force"
            )
        elif (
            self.architecture_version
            == ACT_ALIGNED_HIGH_RATE_MOTION_ARCHITECTURE_VERSION
        ):
            metadata["motion_posterior_layout"] = "official_action_only"
        elif (
            self.architecture_version
            == ACT_ALIGNED_HIGH_RATE_DUAL_ZERO_ARCHITECTURE_VERSION
        ):
            metadata["policy_special_tokens"] = (
                "qpos",
                "z_F_online",
                "z_VF",
            )
        return metadata


def canonical_model_config(values: dict) -> dict:
    """Compare legacy and explicit full-horizon high-rate configurations."""
    result = dict(values)
    if result.get("architecture_version") in {
        ACT_ALIGNED_HIGH_RATE_ARCHITECTURE_VERSION,
        ACT_ALIGNED_HIGH_RATE_MOTION_ARCHITECTURE_VERSION,
        ACT_ALIGNED_HIGH_RATE_DUAL_ZERO_ARCHITECTURE_VERSION,
    }:
        result.setdefault("strict_force_horizon", False)
        if result.get("force_prediction_steps") is None:
            result["force_prediction_steps"] = result.get("chunk_len", 100)
        if result.get("max_online_force_samples_per_interval") is None:
            result["max_online_force_samples_per_interval"] = result.get("max_force_samples_per_interval", 20)
        if result.get("online_force_interval_duration_s") is None:
            result["online_force_interval_duration_s"] = 1.0 / result.get("policy_sample_rate_hz", 30.0)
    return result
