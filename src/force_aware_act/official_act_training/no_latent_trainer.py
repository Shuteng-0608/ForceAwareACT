"""Optimization and validation loops for latent-free official ACT."""

from __future__ import annotations

import math
from statistics import mean
from typing import Callable, Dict, Iterable, Optional

import torch

from force_aware_act.models.official_act import OfficialACTNoLatentPolicy
from force_aware_act.official_act_training.data import OfficialACTBatch
from force_aware_act.official_act_training.no_latent_config import (
    OfficialACTNoLatentTrainingConfig,
)
from force_aware_act.official_act_training.no_latent_losses import (
    OfficialACTNoLatentCriterion,
)


def train_official_act_no_latent_step(
    model: OfficialACTNoLatentPolicy,
    criterion: OfficialACTNoLatentCriterion,
    optimizer: torch.optim.Optimizer,
    batch: OfficialACTBatch,
    config: OfficialACTNoLatentTrainingConfig,
) -> Dict[str, float]:
    _validate_objects(model, criterion, config)
    batch.validate(model.config)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    outputs = model(batch.images, batch.qpos)
    losses = criterion(
        outputs,
        batch.action_chunk,
        batch.padding_mask,
    )
    if not torch.isfinite(losses["loss_total"]):
        raise FloatingPointError("latent-free official ACT loss is not finite")
    losses["loss_total"].backward()
    parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    gradient_norm = _gradient_norm(parameters)
    if not math.isfinite(gradient_norm):
        raise FloatingPointError(
            "latent-free official ACT gradient norm is not finite"
        )
    optimizer.step()
    return {
        **{
            name: float(value.detach().item())
            for name, value in losses.items()
        },
        "loss_action": float(losses["loss_l1"].detach().item()),
        "gradient_norm": gradient_norm,
        "main_learning_rate": _group_lr(optimizer, "main"),
        "backbone_learning_rate": _group_lr(optimizer, "backbone"),
    }


def evaluate_official_act_no_latent_batch(
    model: OfficialACTNoLatentPolicy,
    criterion: OfficialACTNoLatentCriterion,
    batch: OfficialACTBatch,
) -> Dict[str, float]:
    _validate_objects(model, criterion, criterion.config)
    batch.validate(model.config)
    previous_mode = model.training
    model.eval()
    try:
        with torch.no_grad():
            outputs = model(batch.images, batch.qpos)
            losses = criterion(
                outputs,
                batch.action_chunk,
                batch.padding_mask,
            )
    finally:
        model.train(previous_mode)
    return {
        "no_latent_validation_action_l1": float(
            losses["loss_l1"].item()
        ),
        "no_latent_validation_loss": float(losses["loss_total"].item()),
    }


def run_official_act_no_latent_training_epoch(
    model: OfficialACTNoLatentPolicy,
    criterion: OfficialACTNoLatentCriterion,
    optimizer: torch.optim.Optimizer,
    batches: Iterable[OfficialACTBatch],
    config: OfficialACTNoLatentTrainingConfig,
    *,
    device: torch.device,
    global_step_start: int,
    step_callback: Optional[
        Callable[[int, Dict[str, float]], None]
    ] = None,
) -> Dict[str, float]:
    records = []
    for step_in_epoch, batch in enumerate(batches, start=1):
        metrics = train_official_act_no_latent_step(
            model,
            criterion,
            optimizer,
            batch.to(device),
            config,
        )
        records.append(metrics)
        if step_callback is not None:
            step_callback(global_step_start + step_in_epoch, metrics)
    if not records:
        raise ValueError("latent-free official ACT training epoch is empty")
    result = {
        name: mean(record[name] for record in records)
        for name in records[0]
    }
    result["optimizer_steps"] = float(len(records))
    return result


def run_official_act_no_latent_validation_epoch(
    model: OfficialACTNoLatentPolicy,
    criterion: OfficialACTNoLatentCriterion,
    batches: Iterable[OfficialACTBatch],
    *,
    device: torch.device,
) -> Dict[str, float]:
    records = [
        evaluate_official_act_no_latent_batch(
            model,
            criterion,
            batch.to(device),
        )
        for batch in batches
    ]
    if not records:
        raise ValueError("latent-free official ACT validation epoch is empty")
    return {
        name: mean(record[name] for record in records)
        for name in records[0]
    }


def _validate_objects(model, criterion, config) -> None:
    if not isinstance(model, OfficialACTNoLatentPolicy):
        raise TypeError("model must be an OfficialACTNoLatentPolicy")
    if not isinstance(criterion, OfficialACTNoLatentCriterion):
        raise TypeError("criterion must be an OfficialACTNoLatentCriterion")
    if criterion.config != config:
        raise ValueError("criterion and training config must match")


def _gradient_norm(parameters: list[torch.nn.Parameter]) -> float:
    squared = sum(
        float(parameter.grad.detach().float().square().sum().item())
        for parameter in parameters
    )
    return math.sqrt(squared)


def _group_lr(
    optimizer: torch.optim.Optimizer,
    name: str,
) -> float:
    for group in optimizer.param_groups:
        if group.get("name") == name:
            return float(group["lr"])
    raise RuntimeError(f"optimizer is missing group {name!r}")
