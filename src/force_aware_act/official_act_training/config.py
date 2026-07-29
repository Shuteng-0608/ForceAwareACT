"""Training configuration matching the official ACT README recipe."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


OFFICIAL_ACT_TRAINING_VERSION = "official_act_training_v1"


@dataclass(frozen=True)
class OfficialACTTrainingConfig:
    training_version: str = OFFICIAL_ACT_TRAINING_VERSION
    learning_rate: float = 1.0e-5
    backbone_learning_rate: float = 1.0e-5
    weight_decay: float = 1.0e-4
    action_loss_weight: float = 1.0
    kl_weight: float = 10.0
    batch_size: int = 8
    num_epochs: int = 2000
    seed: int = 0
    split_seed: int = 1
    validation_fraction: float = 0.2
    checkpoint_interval_epochs: int = 100
    selection_metric: str = "official_sampled_validation_loss"

    def __post_init__(self) -> None:
        if self.training_version != OFFICIAL_ACT_TRAINING_VERSION:
            raise ValueError(
                f"training_version must be {OFFICIAL_ACT_TRAINING_VERSION!r}"
            )
        for name in (
            "learning_rate",
            "backbone_learning_rate",
            "action_loss_weight",
            "kl_weight",
        ):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ValueError(f"{name} must be numeric")
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        for name in ("batch_size", "num_epochs", "checkpoint_interval_epochs"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("seed", "split_seed"):
            if not isinstance(getattr(self, name), int) or isinstance(
                getattr(self, name),
                bool,
            ):
                raise ValueError(f"{name} must be an integer")
        if not 0.0 < self.validation_fraction < 1.0:
            raise ValueError("validation_fraction must be in (0, 1)")
        if self.selection_metric != "official_sampled_validation_loss":
            raise ValueError(
                "selection_metric must be "
                "'official_sampled_validation_loss'"
            )

    def checkpoint_metadata(self) -> dict[str, Any]:
        metadata = asdict(self)
        metadata.update(
            {
                "optimizer": "AdamW",
                "scheduler": None,
                "objective": "official_masked_l1+10*kl_q_standard_normal",
                "sampling": "one_uniform_timestep_per_episode_per_epoch",
                "normalization_scope": "all_episodes_official_behavior",
                "normalization_minimum_std": 1.0e-2,
                "validation_order": "before_training_each_epoch",
                "deployment_latent": "zero",
            }
        )
        return metadata
