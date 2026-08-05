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
from force_aware_act.act_aligned_training.high_rate_data import (
    ACTAlignedHighRateHDF5Dataset,
    ACTAlignedHighRateSample,
    collate_high_rate_samples,
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
    compute_normalization_stats,
)
from force_aware_act.act_aligned_training.optimizer import (
    build_act_aligned_motion_optimizer,
    build_act_aligned_optimizer,
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
    "ACT_ALIGNED_MOTION_CONTROL_TRAINING_VERSION",
    "ACTAlignedBatch",
    "ACTAlignedCriterion",
    "ACTAlignedHDF5Dataset",
    "ACTAlignedHighRateBatch",
    "ACTAlignedHighRateHDF5Dataset",
    "ACTAlignedHighRateSample",
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
    "build_act_aligned_motion_optimizer",
    "collate_act_aligned_samples",
    "collate_high_rate_samples",
    "compute_normalization_stats",
    "create_episode_split",
    "detached_posterior_prior_kl",
    "diagonal_gaussian_kl",
    "evaluate_one_batch",
    "evaluate_motion_one_batch",
    "discover_episodes",
    "load_act_aligned_checkpoint",
    "masked_l1_loss",
    "partition_trainable_parameters",
    "partition_motion_trainable_parameters",
    "read_act_aligned_checkpoint",
    "run_training_epoch",
    "run_motion_training_epoch",
    "run_motion_validation_epoch",
    "run_training_preflight",
    "run_validation_epoch",
    "save_act_aligned_checkpoint",
    "standard_normal_kl",
    "train_one_step",
    "train_motion_one_step",
]
