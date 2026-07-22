"""Training utilities for ForceAwareACT."""

from force_aware_act.training.control import (
    EARLY_STOP_METRICS,
    VALIDATION_DEPLOYMENT_MODES,
    EarlyStoppingState,
    RetentionGatedCheckpointSelector,
    compute_steps_per_epoch,
    evaluate_deployment_metrics,
    evaluate_named_deployment_metrics,
    flatten_named_deployment_metrics,
    resolve_validation_deployment_mode,
    validate_disjoint_episode_splits,
    validate_normalization_training_episodes,
)
from force_aware_act.training.catalog import (
    EpisodePhaseCatalog,
    PhaseCatalog,
    PhaseSegment,
)
from force_aware_act.training.checkpointing import (
    CHECKPOINT_SCHEMA_VERSION,
    INIT_COMPATIBILITY_KEYS,
    RESUME_COMPATIBILITY_KEYS,
    build_checkpoint_v2,
    initialize_model_from_checkpoint,
    resume_training_from_checkpoint,
    save_checkpoint_atomic,
)
from force_aware_act.training.engine import (
    UpdateResult,
    move_batch_to_device,
    normalize_training_batch,
    train_one_update,
)

from force_aware_act.training.losses import (
    compute_act_baseline_loss,
    compute_contact_prior_distillation_loss,
    compute_force_aware_act_loss,
    compute_force_aware_contact_cvae_loss,
    compute_force_aware_motion_cvae_loss,
    linear_warmup,
)
from force_aware_act.training.optim import (
    build_named_parameter_groups,
    build_parameter_groups_from_specs,
    set_batch_norm_eval,
    set_frozen_batch_norm_eval,
    validate_and_clip_gradients,
)
from force_aware_act.training.policies import (
    build_policy,
    compute_policy_training_loss,
    resolved_model_config,
)
from force_aware_act.training.protocol import (
    PROTOCOL_SCHEMA_VERSION,
    ResolvedProtocol,
    load_protocol,
)
from force_aware_act.training.sampling import (
    DomainPhaseBatchSampler,
    SampleDescriptor,
)

__all__ = [
    "EARLY_STOP_METRICS",
    "VALIDATION_DEPLOYMENT_MODES",
    "EarlyStoppingState",
    "RetentionGatedCheckpointSelector",
    "compute_steps_per_epoch",
    "evaluate_deployment_metrics",
    "evaluate_named_deployment_metrics",
    "flatten_named_deployment_metrics",
    "resolve_validation_deployment_mode",
    "validate_disjoint_episode_splits",
    "validate_normalization_training_episodes",
    "compute_act_baseline_loss",
    "compute_force_aware_act_loss",
    "compute_force_aware_contact_cvae_loss",
    "compute_force_aware_motion_cvae_loss",
    "compute_contact_prior_distillation_loss",
    "linear_warmup",
    "PhaseSegment",
    "EpisodePhaseCatalog",
    "PhaseCatalog",
    "CHECKPOINT_SCHEMA_VERSION",
    "INIT_COMPATIBILITY_KEYS",
    "RESUME_COMPATIBILITY_KEYS",
    "build_checkpoint_v2",
    "initialize_model_from_checkpoint",
    "resume_training_from_checkpoint",
    "save_checkpoint_atomic",
    "UpdateResult",
    "move_batch_to_device",
    "normalize_training_batch",
    "train_one_update",
    "build_named_parameter_groups",
    "build_parameter_groups_from_specs",
    "set_batch_norm_eval",
    "set_frozen_batch_norm_eval",
    "validate_and_clip_gradients",
    "build_policy",
    "compute_policy_training_loss",
    "resolved_model_config",
    "PROTOCOL_SCHEMA_VERSION",
    "ResolvedProtocol",
    "load_protocol",
    "DomainPhaseBatchSampler",
    "SampleDescriptor",
]
