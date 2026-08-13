"""Training configuration for native-rate Motion-CVAE."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

from force_aware_act.act_aligned_training.motion_config import (
    ACTAlignedMotionTrainingConfig,
)


ACT_ALIGNED_HIGH_RATE_MOTION_TRAINING_VERSION = (
    "act_aligned_motion_cvae_highrate_training_v1"
)


@dataclass(frozen=True)
class ACTAlignedHighRateMotionTrainingConfig(ACTAlignedMotionTrainingConfig):
    """Action-only posterior objective with native-rate force reconstruction."""

    training_version: str = ACT_ALIGNED_HIGH_RATE_MOTION_TRAINING_VERSION
    high_rate_force_loss_weight: float = 0.5

    def __post_init__(self) -> None:
        if self.training_version != ACT_ALIGNED_HIGH_RATE_MOTION_TRAINING_VERSION:
            raise ValueError(
                "training_version must be "
                f"{ACT_ALIGNED_HIGH_RATE_MOTION_TRAINING_VERSION!r}"
            )
        base_values = {
            field.name: getattr(self, field.name)
            for field in fields(ACTAlignedMotionTrainingConfig)
            if field.name != "training_version"
        }
        validated = ACTAlignedMotionTrainingConfig(**base_values)
        object.__setattr__(
            self,
            "max_optimizer_steps",
            validated.max_optimizer_steps,
        )
        value = self.high_rate_force_loss_weight
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError("high_rate_force_loss_weight must be numeric")
        if value < 0:
            raise ValueError("high_rate_force_loss_weight must be non-negative")

    def checkpoint_metadata(self) -> dict[str, Any]:
        metadata = super().checkpoint_metadata()
        metadata.update(
            {
                "objective": (
                    "masked_action_l1"
                    "+force_weight*masked_interval_endpoint_force_l1"
                    "+high_rate_force_weight*interval_balanced_500hz_force_l1"
                    "+posterior_kl_weight*kl_q_motion_standard_normal"
                ),
                "force_normalization_source": (
                    "all_native_rate_training_split_wrench_samples"
                ),
                "high_rate_force_reduction": (
                    "mean_force_dims_then_mean_valid_samples_per_interval_"
                    "then_mean_valid_intervals"
                ),
            }
        )
        return metadata
