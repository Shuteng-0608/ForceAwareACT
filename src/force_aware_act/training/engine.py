"""Small, testable primitives used by the staged training loop."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

import torch

from force_aware_act.data import normalize_tensor
from force_aware_act.training.optim import (
    gradients_are_finite,
    nonfinite_gradient_names,
    set_batch_norm_eval,
    set_frozen_batch_norm_eval,
    validate_and_clip_gradients,
)
from force_aware_act.training.policies import compute_policy_training_loss
from force_aware_act.training.protocol import ObjectiveSpec


NORMALIZATION_KEYS = (
    "qpos_mean",
    "qpos_std",
    "action_mean",
    "action_std",
    "force_mean",
    "force_std",
)


@dataclass(frozen=True)
class UpdateResult:
    """Scalar diagnostics produced by exactly one optimizer update."""

    losses: Mapping[str, float]
    objective: Mapping[str, Any]
    gradient_norm: Optional[float]
    gradient_was_clipped: bool
    gradient_parameter_count: int


def move_batch_to_device(
    batch: Mapping[str, object], device: torch.device
) -> Dict[str, object]:
    """Move tensor values to a device while preserving metadata values."""

    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def validate_normalization_stats(stats: Mapping[str, Any]) -> None:
    """Validate the tensor fields required by model training and rollout."""

    if not isinstance(stats, Mapping):
        raise ValueError("normalization stats must be a mapping")
    for key in NORMALIZATION_KEYS:
        if key not in stats:
            raise KeyError(f"normalization stats is missing required key: {key}")
        value = stats[key]
        if not torch.is_tensor(value) or value.ndim != 1:
            raise ValueError(f"normalization stats {key!r} must be a 1D tensor")
        if not bool(torch.isfinite(value).all().item()):
            raise ValueError(f"normalization stats {key!r} contains non-finite values")
        if key.endswith("_std") and not bool((value > 0).all().item()):
            raise ValueError(f"normalization stats {key!r} must be strictly positive")


def normalize_training_batch(
    batch: Mapping[str, object], stats: Mapping[str, Any]
) -> Dict[str, object]:
    """Normalize online inputs and supervised training targets consistently."""

    validate_normalization_stats(stats)
    normalized = dict(batch)
    for key in ("qpos", "force_window", "action_chunk", "future_force_chunk"):
        if key not in normalized or not torch.is_tensor(normalized[key]):
            raise ValueError(f"training batch {key!r} must be a torch.Tensor")
    normalized["qpos"] = normalize_tensor(
        normalized["qpos"], stats["qpos_mean"], stats["qpos_std"]
    )
    normalized["force_window"] = normalize_tensor(
        normalized["force_window"], stats["force_mean"], stats["force_std"]
    )
    normalized["action_chunk"] = normalize_tensor(
        normalized["action_chunk"], stats["action_mean"], stats["action_std"]
    )
    normalized["future_force_chunk"] = normalize_tensor(
        normalized["future_force_chunk"], stats["force_mean"], stats["force_std"]
    )
    return normalized


def train_one_update(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: Mapping[str, object],
    policy_variant: str,
    objective: ObjectiveSpec,
    stage_step: int,
    max_grad_norm: Optional[float],
    freeze_vision_batch_norm: bool = False,
) -> UpdateResult:
    """Run one finite-checked FP32 training update."""

    model.train()
    set_frozen_batch_norm_eval(model)
    if not isinstance(freeze_vision_batch_norm, bool):
        raise TypeError("freeze_vision_batch_norm must be boolean")
    if freeze_vision_batch_norm:
        vision_encoder = getattr(model, "vision_encoder", None)
        backbone = getattr(vision_encoder, "backbone", None)
        if not isinstance(backbone, torch.nn.Module):
            raise ValueError(
                "freeze_vision_batch_norm requires model.vision_encoder.backbone"
            )
        frozen_batch_norm = set_batch_norm_eval(
            backbone, name_prefix="vision_encoder.backbone"
        )
        if not frozen_batch_norm:
            raise ValueError(
                "freeze_vision_batch_norm requested but the vision backbone has no BatchNorm"
            )
    optimizer.zero_grad(set_to_none=True)
    losses, objective_metadata = compute_policy_training_loss(
        model=model,
        batch=batch,
        policy_variant=policy_variant,
        objective=objective,
        stage_step=stage_step,
    )
    total_loss = losses["loss_total"]
    total_loss.backward()

    gradient_norm = None
    gradient_was_clipped = False
    gradient_parameter_count = sum(
        1
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    )
    if max_grad_norm is not None:
        clip_result = validate_and_clip_gradients(model, max_norm=max_grad_norm)
        gradient_norm = clip_result.total_norm
        gradient_was_clipped = clip_result.was_clipped
        gradient_parameter_count = clip_result.parameter_count
    elif not gradients_are_finite(model):
        invalid = nonfinite_gradient_names(model)
        raise FloatingPointError(
            f"non-finite gradients in {len(invalid)} parameters: " + ", ".join(invalid[:10])
        )
    optimizer.step()

    scalar_losses: Dict[str, float] = {}
    for name, value in losses.items():
        if torch.is_tensor(value):
            if value.ndim == 0:
                scalar_losses[name] = float(value.detach().cpu().item())
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            scalar_losses[name] = float(value)
    return UpdateResult(
        losses=scalar_losses,
        objective=dict(objective_metadata),
        gradient_norm=gradient_norm,
        gradient_was_clipped=gradient_was_clipped,
        gradient_parameter_count=gradient_parameter_count,
    )
