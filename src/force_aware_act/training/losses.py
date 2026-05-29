"""Loss utilities for ForceAwareACT training."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Union

import torch
import torch.nn.functional as functional

from force_aware_act.models import kl_normal


REQUIRED_OUTPUT_KEYS = (
    "pred_action",
    "pred_force",
    "mu_motion",
    "logvar_motion",
    "mu_contact",
    "logvar_contact",
)


def compute_force_aware_act_loss(
    outputs: Mapping[str, Any],
    action_chunk: torch.Tensor,
    future_force_chunk: torch.Tensor,
    lambda_force: float = 0.1,
    beta_motion: float = 1.0e-4,
    beta_contact: float = 1.0e-4,
) -> Dict[str, Union[torch.Tensor, float]]:
    """Compute the supervised action/force and posterior KL losses."""

    _validate_required_outputs(outputs)
    pred_action = outputs["pred_action"]
    pred_force = outputs["pred_force"]
    _validate_tensor_shape("pred_action", pred_action, action_chunk.shape)
    _validate_tensor_shape("pred_force", pred_force, future_force_chunk.shape)

    loss_action = functional.l1_loss(pred_action, action_chunk)
    loss_force = functional.l1_loss(pred_force, future_force_chunk)
    kl_motion = kl_normal(outputs["mu_motion"], outputs["logvar_motion"])
    kl_contact = kl_normal(outputs["mu_contact"], outputs["logvar_contact"])
    loss_total = (
        loss_action
        + lambda_force * loss_force
        + beta_motion * kl_motion
        + beta_contact * kl_contact
    )

    return {
        "loss_total": loss_total,
        "loss_action": loss_action,
        "loss_force": loss_force,
        "kl_motion": kl_motion,
        "kl_contact": kl_contact,
        "lambda_force": lambda_force,
        "beta_motion": beta_motion,
        "beta_contact": beta_contact,
    }


def linear_warmup(step: int, warmup_steps: int, max_value: float) -> float:
    """Linearly ramp from zero to ``max_value`` over ``warmup_steps``."""

    if warmup_steps <= 0:
        return max_value
    if step >= warmup_steps:
        return max_value
    return max_value * step / warmup_steps


def _validate_required_outputs(outputs: Mapping[str, Any]) -> None:
    for key in REQUIRED_OUTPUT_KEYS:
        if key not in outputs:
            raise KeyError(f"outputs is missing required key: {key}")
        if not isinstance(outputs[key], torch.Tensor):
            raise ValueError(f"outputs[{key!r}] must be a torch.Tensor")


def _validate_tensor_shape(name: str, tensor: torch.Tensor, expected_shape: torch.Size) -> None:
    if tensor.shape != expected_shape:
        raise ValueError(
            f"{name} must have shape {tuple(expected_shape)}, got {tuple(tensor.shape)}"
        )
