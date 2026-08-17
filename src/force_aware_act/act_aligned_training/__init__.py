"""Independent training stack for the ACT-aligned contact-CVAE."""

from force_aware_act.act_aligned_training.batch import ACTAlignedBatch
from force_aware_act.act_aligned_training.checkpoint import (
    CHECKPOINT_FORMAT_VERSION,
    LoadedCheckpoint,
    TrainingProgress,
    load_act_aligned_checkpoint,
    read_act_aligned_checkpoint,
    save_act_aligned_checkpoint,
)
from force_aware_act.act_aligned_training.config import (
    ACT_ALIGNED_TRAINING_VERSION,
    ACTAlignedTrainingConfig,
)
from force_aware_act.act_aligned_training.losses import (
    ACTAlignedCriterion,
    detached_posterior_prior_kl,
    diagonal_gaussian_kl,
    masked_l1_loss,
    standard_normal_kl,
)
from force_aware_act.act_aligned_training.motion_config import (
    ACT_ALIGNED_MOTION_CONTROL_TRAINING_VERSION,
    ACTAlignedMotionTrainingConfig,
)
from force_aware_act.act_aligned_training.motion_loop import (
    run_motion_training_epoch,
    run_motion_validation_epoch,
)
from force_aware_act.act_aligned_training.motion_losses import (
    ACTAlignedMotionCriterion,
)
from force_aware_act.act_aligned_training.motion_trainer import (
    evaluate_motion_one_batch,
    train_motion_one_step,
)
from force_aware_act.act_aligned_training.data import (
    ACTAlignedHDF5Dataset,
    ACTAlignedSample,
    collate_act_aligned_samples,
)
from force_aware_act.act_aligned_training.high_rate_batch import (
    ACTAlignedHighRateBatch,
)
from force_aware_act.act_aligned_training.high_rate_config import (
    ACT_ALIGNED_HIGH_RATE_TRAINING_VERSION,
    ACTAlignedHighRateTrainingConfig,
)
from force_aware_act.act_aligned_training.high_rate_data import (
    ACTAlignedHighRateHDF5Dataset,
    ACTAlignedHighRateSample,
    collate_high_rate_samples,
)
from force_aware_act.act_aligned_training.high_rate_diagnostics import (
    run_high_rate_training_preflight,
)
from force_aware_act.act_aligned_training.high_rate_control_diagnostics import (
    run_high_rate_control_preflight,
)
from force_aware_act.act_aligned_training.high_rate_losses import (
    ACTAlignedHighRateCriterion,
    masked_interval_balanced_high_rate_l1_loss,
)
from force_aware_act.act_aligned_training.high_rate_control_losses import (
    ACTAlignedHighRateDualZeroCriterion,
    ACTAlignedHighRateMotionCriterion,
)
from force_aware_act.act_aligned_training.high_rate_dual_zero_config import (
    ACT_ALIGNED_HIGH_RATE_DUAL_ZERO_TRAINING_VERSION,
    ACTAlignedHighRateDualZeroTrainingConfig,
)
from force_aware_act.act_aligned_training.high_rate_motion_config import (
    ACT_ALIGNED_HIGH_RATE_MOTION_TRAINING_VERSION,
    ACTAlignedHighRateMotionTrainingConfig,
)
from force_aware_act.act_aligned_training.high_rate_control_loop import (
    run_high_rate_dual_zero_training_epoch,
    run_high_rate_dual_zero_validation_epoch,
    run_high_rate_motion_training_epoch,
    run_high_rate_motion_validation_epoch,
)
from force_aware_act.act_aligned_training.high_rate_control_trainer import (
    evaluate_high_rate_dual_zero_one_batch,
    evaluate_high_rate_motion_one_batch,
    train_high_rate_dual_zero_one_step,
    train_high_rate_motion_one_step,
)
from force_aware_act.act_aligned_training.high_rate_loop import (
    run_high_rate_training_epoch,
    run_high_rate_validation_epoch,
    run_high_rate_physical_action_validation_epoch,
)
from force_aware_act.act_aligned_training.high_rate_trainer import (
    evaluate_high_rate_one_batch,
    train_high_rate_one_step,
)
from force_aware_act.act_aligned_training.diagnostics import (
    run_training_preflight,
)
from force_aware_act.act_aligned_training.loop import (
    run_training_epoch,
    run_validation_epoch,
)
from force_aware_act.act_aligned_training.normalization import (
    NormalizationStats,
    compute_high_rate_normalization_stats,
    compute_normalization_stats,
)
from force_aware_act.act_aligned_training.optimizer import (
    build_act_aligned_motion_optimizer,
    build_act_aligned_high_rate_dual_zero_optimizer,
    build_act_aligned_high_rate_motion_optimizer,
    build_act_aligned_optimizer,
    build_act_aligned_high_rate_optimizer,
    partition_motion_trainable_parameters,
    partition_trainable_parameters,
)
from force_aware_act.act_aligned_training.trainer import (
    evaluate_one_batch,
    train_one_step,
)
from force_aware_act.act_aligned_training.split import (
    EpisodeRecord,
    EpisodeSplitManifest,
    create_episode_split,
    discover_episodes,
)

__all__ = [
    "ACT_ALIGNED_TRAINING_VERSION",
    "ACT_ALIGNED_HIGH_RATE_TRAINING_VERSION",
    "ACT_ALIGNED_HIGH_RATE_DUAL_ZERO_TRAINING_VERSION",
    "ACT_ALIGNED_HIGH_RATE_MOTION_TRAINING_VERSION",
    "ACT_ALIGNED_MOTION_CONTROL_TRAINING_VERSION",
    "ACTAlignedBatch",
    "ACTAlignedCriterion",
    "ACTAlignedHDF5Dataset",
    "ACTAlignedHighRateBatch",
    "ACTAlignedHighRateCriterion",
    "ACTAlignedHighRateDualZeroCriterion",
    "ACTAlignedHighRateDualZeroTrainingConfig",
    "ACTAlignedHighRateHDF5Dataset",
    "ACTAlignedHighRateSample",
    "ACTAlignedHighRateTrainingConfig",
    "ACTAlignedHighRateMotionCriterion",
    "ACTAlignedHighRateMotionTrainingConfig",
    "ACTAlignedMotionCriterion",
    "ACTAlignedMotionTrainingConfig",
    "ACTAlignedSample",
    "ACTAlignedTrainingConfig",
    "CHECKPOINT_FORMAT_VERSION",
    "EpisodeRecord",
    "EpisodeSplitManifest",
    "LoadedCheckpoint",
    "NormalizationStats",
    "TrainingProgress",
    "build_act_aligned_optimizer",
    "build_act_aligned_high_rate_optimizer",
    "build_act_aligned_high_rate_dual_zero_optimizer",
    "build_act_aligned_high_rate_motion_optimizer",
    "build_act_aligned_motion_optimizer",
    "collate_act_aligned_samples",
    "collate_high_rate_samples",
    "compute_normalization_stats",
    "compute_high_rate_normalization_stats",
    "create_episode_split",
    "detached_posterior_prior_kl",
    "diagonal_gaussian_kl",
    "evaluate_one_batch",
    "evaluate_high_rate_one_batch",
    "evaluate_high_rate_dual_zero_one_batch",
    "evaluate_high_rate_motion_one_batch",
    "evaluate_motion_one_batch",
    "discover_episodes",
    "load_act_aligned_checkpoint",
    "masked_l1_loss",
    "masked_interval_balanced_high_rate_l1_loss",
    "partition_trainable_parameters",
    "partition_motion_trainable_parameters",
    "read_act_aligned_checkpoint",
    "run_training_epoch",
    "run_high_rate_training_epoch",
    "run_high_rate_control_preflight",
    "run_high_rate_dual_zero_training_epoch",
    "run_high_rate_dual_zero_validation_epoch",
    "run_high_rate_motion_training_epoch",
    "run_high_rate_motion_validation_epoch",
    "run_high_rate_training_preflight",
    "run_high_rate_validation_epoch",
    "run_high_rate_physical_action_validation_epoch",
    "run_motion_training_epoch",
    "run_motion_validation_epoch",
    "run_training_preflight",
    "run_validation_epoch",
    "save_act_aligned_checkpoint",
    "standard_normal_kl",
    "train_one_step",
    "train_high_rate_one_step",
    "train_high_rate_dual_zero_one_step",
    "train_high_rate_motion_one_step",
    "train_motion_one_step",
]
