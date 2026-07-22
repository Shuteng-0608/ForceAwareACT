"""Policy construction and loss routing shared by staged training."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Tuple

import torch

from force_aware_act.models import (
    ForceAwareACTContactCVAEPolicy,
    ForceAwareACTMotionCVAEPolicy,
    ForceAwareACTPolicy,
)
from force_aware_act.training.losses import (
    compute_force_aware_act_loss,
    compute_force_aware_contact_cvae_loss,
    compute_force_aware_motion_cvae_loss,
    linear_warmup,
)
from force_aware_act.training.protocol import DatasetSpec, ModelSpec, ObjectiveSpec


def resolved_model_config(model_spec: ModelSpec, dataset_spec: DatasetSpec) -> Dict[str, Any]:
    """Return the canonical kwargs used both for construction and checkpoints."""

    return {
        "pretrained_resnet18": model_spec.pretrained_resnet18,
        "d_model": model_spec.d_model,
        "z_dim": model_spec.z_dim,
        "action_dim": model_spec.action_dim,
        "force_dim": model_spec.force_dim,
        "chunk_len": dataset_spec.chunk_len,
        "nhead": model_spec.nhead,
        "num_encoder_layers": model_spec.num_encoder_layers,
        "num_decoder_layers": model_spec.num_decoder_layers,
        "dim_feedforward": model_spec.dim_feedforward,
        "dropout": model_spec.dropout,
        "max_force_window_len": max(dataset_spec.force_window_len, 20),
    }


def build_policy(model_spec: ModelSpec, dataset_spec: DatasetSpec) -> torch.nn.Module:
    """Construct one supported force-aware architecture from canonical config."""

    kwargs = resolved_model_config(model_spec, dataset_spec)
    if model_spec.policy_variant == "force_aware_motion_cvae":
        return ForceAwareACTMotionCVAEPolicy(**kwargs)
    if model_spec.policy_variant == "force_aware_contact_cvae":
        return ForceAwareACTContactCVAEPolicy(**kwargs)
    if model_spec.policy_variant == "force_aware_act":
        return ForceAwareACTPolicy(**kwargs)
    raise ValueError(f"unsupported policy_variant={model_spec.policy_variant!r}")


def _require_tensor(batch: Mapping[str, object], key: str) -> torch.Tensor:
    if key not in batch:
        raise KeyError(f"training batch is missing required key: {key}")
    value = batch[key]
    if not torch.is_tensor(value):
        raise ValueError(f"training batch key {key!r} must be a torch.Tensor")
    return value


def compute_policy_training_loss(
    *,
    model: torch.nn.Module,
    batch: Mapping[str, object],
    policy_variant: str,
    objective: ObjectiveSpec,
    stage_step: int,
) -> Tuple[Mapping[str, Any], Dict[str, Any]]:
    """Run the supervised training-only forward path and compute its loss.

    Future action and force labels are deliberately confined to this function.
    Deployment validation continues to use ``evaluate_deployment_metrics``,
    whose model call supplies neither future actions nor future forces.
    """

    if stage_step <= 0:
        raise ValueError("stage_step must be positive")
    images = _require_tensor(batch, "images")
    qpos = _require_tensor(batch, "qpos")
    force_window = _require_tensor(batch, "force_window")
    action_chunk = _require_tensor(batch, "action_chunk")
    future_force_chunk = _require_tensor(batch, "future_force_chunk")
    beta_motion = linear_warmup(
        stage_step, objective.warmup_steps, objective.beta_motion_max
    )
    beta_contact = linear_warmup(
        stage_step, objective.warmup_steps, objective.beta_contact_max
    )

    if policy_variant == "force_aware_motion_cvae":
        outputs = model(
            images=images,
            qpos=qpos,
            force_window=force_window,
            action_chunk=action_chunk,
            future_force_chunk=future_force_chunk,
            is_training=True,
        )
        losses = compute_force_aware_motion_cvae_loss(
            outputs=outputs,
            action_chunk=action_chunk,
            future_force_chunk=future_force_chunk,
            lambda_force=objective.lambda_force,
            beta_motion=beta_motion,
        )
    elif policy_variant == "force_aware_contact_cvae":
        outputs = model(
            images=images,
            qpos=qpos,
            force_window=force_window,
            action_chunk=action_chunk,
            future_force_chunk=future_force_chunk,
            is_training=True,
            contact_latent_mode=objective.train_contact_latent_mode,
        )
        losses = compute_force_aware_contact_cvae_loss(
            outputs=outputs,
            action_chunk=action_chunk,
            future_force_chunk=future_force_chunk,
            lambda_force=objective.lambda_force,
            beta_contact=beta_contact,
            lambda_prior=objective.lambda_prior,
            prior_loss_mode=objective.prior_loss_mode,
        )
    elif policy_variant == "force_aware_act":
        use_posterior = objective.train_latent_mode == "posterior"
        outputs = model(
            images=images,
            qpos=qpos,
            force_window=force_window,
            action_chunk=action_chunk,
            future_force_chunk=future_force_chunk,
            is_training=True,
            contact_latent_mode=objective.train_latent_mode,
        )
        losses = compute_force_aware_act_loss(
            outputs=outputs,
            action_chunk=action_chunk,
            future_force_chunk=future_force_chunk,
            lambda_force=objective.lambda_force,
            beta_motion=beta_motion if use_posterior else 0.0,
            beta_contact=beta_contact if use_posterior else 0.0,
            lambda_prior=objective.lambda_prior if use_posterior else 0.0,
            prior_loss_mode=objective.prior_loss_mode,
            use_posterior_kl=use_posterior,
        )
    else:
        raise ValueError(f"unsupported policy_variant={policy_variant!r}")

    loss_total = losses.get("loss_total")
    if not torch.is_tensor(loss_total) or loss_total.ndim != 0:
        raise ValueError("policy loss_total must be a scalar torch.Tensor")
    if not bool(torch.isfinite(loss_total).item()):
        raise FloatingPointError("non-finite total training loss")
    metadata = {
        "beta_motion": beta_motion if policy_variant != "force_aware_contact_cvae" else 0.0,
        "beta_contact": beta_contact if policy_variant != "force_aware_motion_cvae" else 0.0,
        "train_latent_mode": objective.train_latent_mode,
        "train_contact_latent_mode": objective.train_contact_latent_mode,
    }
    return losses, metadata
