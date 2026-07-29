"""Training configuration for the action-only motion-latent control."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Optional


ACT_ALIGNED_MOTION_CONTROL_TRAINING_VERSION = (
    "act_aligned_motion_cvae_control_training_v1"
)


@dataclass(frozen=True)
class ACTAlignedMotionTrainingConfig:
    """Official-style CVAE loss with the force-aware prediction heads kept."""

    training_version: str = ACT_ALIGNED_MOTION_CONTROL_TRAINING_VERSION

    learning_rate: float = 1.0e-5
    backbone_learning_rate: float = 1.0e-5
    weight_decay: float = 1.0e-4
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1.0e-8

    action_loss_weight: float = 1.0
    force_loss_weight: float = 1.0
    posterior_kl_weight: float = 10.0

    batch_size: int = 8
    seed: int = 0
    reference_train_episodes: int = 90
    official_reference_epochs: int = 2000
    max_optimizer_steps: Optional[int] = None
    validation_interval: int = 1
    checkpoint_interval_steps: int = 2000
    gradient_clip_norm: Optional[float] = None

    selection_metric: str = "deployment_zero_action_l1"

    def __post_init__(self) -> None:
        if (
            self.training_version
            != ACT_ALIGNED_MOTION_CONTROL_TRAINING_VERSION
        ):
            raise ValueError(
                "training_version must be "
                f"{ACT_ALIGNED_MOTION_CONTROL_TRAINING_VERSION!r}"
            )
        for name in (
            "learning_rate",
            "backbone_learning_rate",
            "adam_epsilon",
            "action_loss_weight",
        ):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ValueError(f"{name} must be numeric")
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        for name in (
            "weight_decay",
            "force_loss_weight",
            "posterior_kl_weight",
        ):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ValueError(f"{name} must be numeric")
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.backbone_learning_rate > self.learning_rate:
            raise ValueError(
                "backbone_learning_rate must not exceed learning_rate"
            )
        for name in ("adam_beta1", "adam_beta2"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ValueError(f"{name} must be numeric")
            if not 0.0 <= value < 1.0:
                raise ValueError(f"{name} must be in [0, 1)")
        for name in (
            "batch_size",
            "reference_train_episodes",
            "official_reference_epochs",
            "validation_interval",
            "checkpoint_interval_steps",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")

        derived_steps = (
            math.ceil(self.reference_train_episodes / self.batch_size)
            * self.official_reference_epochs
        )
        if self.max_optimizer_steps is None:
            object.__setattr__(self, "max_optimizer_steps", derived_steps)
        elif (
            not isinstance(self.max_optimizer_steps, int)
            or isinstance(self.max_optimizer_steps, bool)
            or self.max_optimizer_steps <= 0
        ):
            raise ValueError("max_optimizer_steps must be positive or None")
        elif self.max_optimizer_steps != derived_steps:
            raise ValueError(
                "max_optimizer_steps must equal "
                "ceil(reference_train_episodes / batch_size) * "
                "official_reference_epochs"
            )
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise ValueError("seed must be an integer")
        if self.gradient_clip_norm is not None:
            if (
                not isinstance(self.gradient_clip_norm, (int, float))
                or isinstance(self.gradient_clip_norm, bool)
                or self.gradient_clip_norm <= 0
            ):
                raise ValueError("gradient_clip_norm must be positive or None")
        if self.selection_metric != "deployment_zero_action_l1":
            raise ValueError(
                "selection_metric must be 'deployment_zero_action_l1'"
            )

    def checkpoint_metadata(self) -> dict[str, Any]:
        """Return explicit control-experiment training metadata."""

        metadata = asdict(self)
        metadata.update(
            {
                "optimizer": "AdamW",
                "objective": (
                    "masked_action_l1"
                    "+force_weight*masked_force_l1"
                    "+posterior_kl_weight*kl_q_motion_standard_normal"
                ),
                "scheduler": None,
                "kl_warmup": None,
                "free_bits": None,
                "conditional_prior": None,
                "duration_semantics": (
                    "official_equivalent_optimizer_steps_for_one_random_"
                    "timestep_per_train_episode_per_reference_epoch"
                ),
                "posterior_validation_latent": "mean",
                "deployment_validation_latents": ("zero",),
            }
        )
        return metadata
