"""Official ACT optimization steps, epochs, and latent diagnostics."""

from __future__ import annotations

import math
from statistics import mean
from typing import Any, Callable, Dict, Iterable, Optional

import torch

from force_aware_act.act_aligned_training.losses import masked_l1_loss
from force_aware_act.models.official_act import OfficialACTPolicy
from force_aware_act.official_act_training.config import (
    OfficialACTTrainingConfig,
)
from force_aware_act.official_act_training.data import OfficialACTBatch
from force_aware_act.official_act_training.losses import OfficialACTCriterion


def train_official_act_step(
    model: OfficialACTPolicy,
    criterion: OfficialACTCriterion,
    optimizer: torch.optim.Optimizer,
    batch: OfficialACTBatch,
    config: OfficialACTTrainingConfig,
) -> Dict[str, float]:
    batch.validate(model.config)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    outputs = model.forward_train(
        batch.images,
        batch.qpos,
        batch.action_chunk,
        batch.padding_mask,
        sample_posterior=True,
    )
    losses = criterion(
        outputs,
        batch.action_chunk,
        batch.padding_mask,
    )
    if not torch.isfinite(losses["loss_total"]):
        raise FloatingPointError("official ACT loss is not finite")
    losses["loss_total"].backward()
    parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    gradient_norm = _gradient_norm(parameters)
    if not math.isfinite(gradient_norm):
        raise FloatingPointError("official ACT gradient norm is not finite")
    optimizer.step()
    return {
        **{
            name: float(value.detach().item())
            for name, value in losses.items()
        },
        "loss_action": float(losses["loss_l1"].detach().item()),
        "loss_posterior_kl": float(losses["loss_kl"].detach().item()),
        "posterior_mean_abs": float(
            outputs["mu_motion"].detach().abs().mean().item()
        ),
        "posterior_std_mean": float(
            torch.exp(0.5 * outputs["logvar_motion"].detach())
            .mean()
            .item()
        ),
        "gradient_norm": gradient_norm,
        "main_learning_rate": _group_lr(optimizer, "main"),
        "backbone_learning_rate": _group_lr(optimizer, "backbone"),
    }


def evaluate_official_act_batch(
    model: OfficialACTPolicy,
    criterion: OfficialACTCriterion,
    batch: OfficialACTBatch,
    *,
    include_deployment_diagnostics: bool = True,
) -> Dict[str, Any]:
    """Run official sampled validation plus deterministic latent audits."""

    batch.validate(model.config)
    previous_mode = model.training
    model.eval()
    try:
        with torch.no_grad():
            sampled = model.forward_train(
                batch.images,
                batch.qpos,
                batch.action_chunk,
                batch.padding_mask,
                sample_posterior=True,
            )
            sampled_losses = criterion(
                sampled,
                batch.action_chunk,
                batch.padding_mask,
            )
            diagnostic_metrics = {}
            if include_deployment_diagnostics:
                posterior = model.forward_train(
                    batch.images,
                    batch.qpos,
                    batch.action_chunk,
                    batch.padding_mask,
                    sample_posterior=False,
                )
                zero = model(batch.images, batch.qpos)
                posterior_action = masked_l1_loss(
                    posterior["pred_action"],
                    batch.action_chunk,
                    batch.padding_mask,
                    name="official_posterior_action",
                )
                zero_action = masked_l1_loss(
                    zero["pred_action"],
                    batch.action_chunk,
                    batch.padding_mask,
                    name="official_zero_action",
                )
                posterior_zero_delta = masked_l1_loss(
                    posterior["pred_action"],
                    zero["pred_action"],
                    batch.padding_mask,
                    name="official_posterior_zero_delta",
                )
                diagnostic_metrics = {
                    "posterior_action_l1": float(
                        posterior_action.item()
                    ),
                    "deployment_zero_action_l1": float(
                        zero_action.item()
                    ),
                    "posterior_zero_action_delta": float(
                        posterior_zero_delta.item()
                    ),
                }
    finally:
        model.train(previous_mode)
    return {
        "official_sampled_validation_loss": float(
            sampled_losses["loss_total"].item()
        ),
        "official_sampled_l1": float(sampled_losses["loss_l1"].item()),
        "posterior_kl_standard": float(
            sampled_losses["loss_kl"].item()
        ),
        "posterior_mean_abs": float(
            sampled["mu_motion"].abs().mean().item()
        ),
        "posterior_std_mean": float(
            torch.exp(0.5 * sampled["logvar_motion"]).mean().item()
        ),
        "_posterior_mean_sum": (
            sampled["mu_motion"].double().sum(dim=0).cpu()
        ),
        "_posterior_mean_square_sum": (
            sampled["mu_motion"].double().square().sum(dim=0).cpu()
        ),
        **diagnostic_metrics,
    }


def run_official_act_training_epoch(
    model: OfficialACTPolicy,
    criterion: OfficialACTCriterion,
    optimizer: torch.optim.Optimizer,
    batches: Iterable[OfficialACTBatch],
    config: OfficialACTTrainingConfig,
    *,
    device: torch.device,
    global_step_start: int,
    step_callback: Optional[
        Callable[[int, Dict[str, float]], None]
    ] = None,
) -> Dict[str, float]:
    records = []
    for step_in_epoch, batch in enumerate(batches, start=1):
        metrics = train_official_act_step(
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
        raise ValueError("official ACT training epoch is empty")
    result = {
        name: mean(record[name] for record in records)
        for name in records[0]
    }
    result["optimizer_steps"] = float(len(records))
    return result


def run_official_act_validation_epoch(
    model: OfficialACTPolicy,
    criterion: OfficialACTCriterion,
    batches: Iterable[OfficialACTBatch],
    *,
    device: torch.device,
    include_deployment_diagnostics: bool = True,
) -> Dict[str, float]:
    records = []
    latent_sum = None
    latent_square_sum = None
    sample_count = 0
    for batch in batches:
        batch = batch.to(device)
        metrics = evaluate_official_act_batch(
            model,
            criterion,
            batch,
            include_deployment_diagnostics=(
                include_deployment_diagnostics
            ),
        )
        latent_sum = (
            metrics["_posterior_mean_sum"]
            if latent_sum is None
            else latent_sum + metrics["_posterior_mean_sum"]
        )
        latent_square_sum = (
            metrics["_posterior_mean_square_sum"]
            if latent_square_sum is None
            else latent_square_sum
            + metrics["_posterior_mean_square_sum"]
        )
        sample_count += batch.batch_size
        records.append(
            {
                name: value
                for name, value in metrics.items()
                if not name.startswith("_")
            }
        )
    if not records or latent_sum is None or latent_square_sum is None:
        raise ValueError("official ACT validation epoch is empty")
    result = {
        name: mean(record[name] for record in records)
        for name in records[0]
    }
    latent_mean = latent_sum / sample_count
    latent_mean_square = latent_square_sum / sample_count
    result["posterior_mean_across_sample_variance"] = float(
        (latent_mean_square - latent_mean.square())
        .clamp_min(0.0)
        .mean()
        .item()
    )
    return result


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
