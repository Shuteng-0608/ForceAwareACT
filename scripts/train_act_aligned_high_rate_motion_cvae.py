#!/usr/bin/env python3
"""Train Motion-CVAE with native 500 Hz force intervals."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from force_aware_act.act_aligned_training import (  # noqa: E402
    ACTAlignedHighRateHDF5Dataset,
    ACTAlignedHighRateMotionCriterion,
    ACTAlignedHighRateMotionTrainingConfig,
    build_act_aligned_high_rate_motion_optimizer,
    collate_high_rate_samples,
    compute_high_rate_normalization_stats,
    run_high_rate_motion_training_epoch,
    run_high_rate_motion_validation_epoch,
)
from force_aware_act.models.act_aligned import (  # noqa: E402
    ACTAlignedHighRateConfig,
    ACTAlignedHighRateMotionCVAEPolicy,
)
from train_act_aligned_contact_cvae import TrainingStack, main  # noqa: E402


def _smoke_model_config() -> ACTAlignedHighRateConfig:
    return ACTAlignedHighRateConfig.motion_control(
        d_model=32,
        nhead=4,
        dim_feedforward=64,
        local_force_dim=32,
        dropout=0.0,
        chunk_len=6,
        image_height=64,
        image_width=64,
        pretrained_backbone=False,
        imagenet_normalize=False,
    )


HIGH_RATE_MOTION_TRAINING_STACK = TrainingStack(
    name="motion_cvae_native_500hz_v1",
    model_config_type=ACTAlignedHighRateConfig,
    training_config_type=ACTAlignedHighRateMotionTrainingConfig,
    policy_type=ACTAlignedHighRateMotionCVAEPolicy,
    criterion_type=ACTAlignedHighRateMotionCriterion,
    build_optimizer=build_act_aligned_high_rate_motion_optimizer,
    run_training_epoch=run_high_rate_motion_training_epoch,
    run_validation_epoch=run_high_rate_motion_validation_epoch,
    smoke_model_config=_smoke_model_config,
    formal_model_config=ACTAlignedHighRateConfig.motion_control,
    dataset_type=ACTAlignedHighRateHDF5Dataset,
    collate_fn=collate_high_rate_samples,
    normalization_fn=compute_high_rate_normalization_stats,
)


if __name__ == "__main__":
    main(HIGH_RATE_MOTION_TRAINING_STACK)
