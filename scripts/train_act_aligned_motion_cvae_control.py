#!/usr/bin/env python3
"""Train the isolated official-style ACT ``z_motion`` control."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from force_aware_act.act_aligned_training import (  # noqa: E402
    ACTAlignedMotionCriterion,
    ACTAlignedMotionTrainingConfig,
    build_act_aligned_motion_optimizer,
    run_motion_training_epoch,
    run_motion_validation_epoch,
)
from force_aware_act.models.act_aligned import (  # noqa: E402
    ACTAlignedConfig,
    ACTAlignedMotionCVAEControlPolicy,
)
from train_act_aligned_contact_cvae import (  # noqa: E402
    TrainingStack,
    main,
)


def _smoke_model_config() -> ACTAlignedConfig:
    return ACTAlignedConfig.motion_control(
        d_model=32,
        nhead=4,
        dim_feedforward=64,
        dropout=0.0,
        chunk_len=6,
        force_window_len=5,
        image_height=64,
        image_width=64,
        pretrained_backbone=False,
        imagenet_normalize=False,
    )


MOTION_CONTROL_TRAINING_STACK = TrainingStack(
    name="official_action_only_motion_cvae_control",
    model_config_type=ACTAlignedConfig,
    training_config_type=ACTAlignedMotionTrainingConfig,
    policy_type=ACTAlignedMotionCVAEControlPolicy,
    criterion_type=ACTAlignedMotionCriterion,
    build_optimizer=build_act_aligned_motion_optimizer,
    run_training_epoch=run_motion_training_epoch,
    run_validation_epoch=run_motion_validation_epoch,
    smoke_model_config=_smoke_model_config,
    formal_model_config=ACTAlignedConfig.motion_control,
)


if __name__ == "__main__":
    main(MOTION_CONTROL_TRAINING_STACK)
