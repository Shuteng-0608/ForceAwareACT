"""Canonical preflight diagnostics for the ACT-aligned training stack."""

from __future__ import annotations

import math
import time
from typing import Any, Dict, Iterable, Mapping

import torch
from torch import nn

from force_aware_act.act_aligned_training.batch import ACTAlignedBatch
from force_aware_act.act_aligned_training.losses import ACTAlignedCriterion
from force_aware_act.models.act_aligned.policy import (
    ACTAlignedContactCVAEPolicy,
)


def run_training_preflight(
    model: ACTAlignedContactCVAEPolicy,
    criterion: ACTAlignedCriterion,
    optimizer: torch.optim.Optimizer,
    batch: ACTAlignedBatch,
) -> Dict[str, Any]:
    """Run a non-mutating forward/backward and deployment-path audit.

    The function populates gradients but deliberately does not call
    ``optimizer.step()``. It therefore checks the complete train graph without
    changing model or optimizer state.
    """

    if not isinstance(model, ACTAlignedContactCVAEPolicy):
        raise TypeError("model must be an ACTAlignedContactCVAEPolicy")
    if not isinstance(criterion, ACTAlignedCriterion):
        raise TypeError("criterion must be an ACTAlignedCriterion")
    if not isinstance(optimizer, torch.optim.Optimizer):
        raise TypeError("optimizer must be a torch optimizer")
    batch.validate(model.config)
    reference_parameter = next(model.parameters())
    if batch.images.device != reference_parameter.device:
        raise ValueError("batch and model must be on the same device")

    modules = _diagnostic_modules(model)
    parameter_report = _parameter_report(model, modules)
    optimizer_report = _optimizer_report(model, optimizer)

    model.train()
    optimizer.zero_grad(set_to_none=True)
    device = batch.images.device
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    start = time.perf_counter()

    outputs = model.forward_train(
        batch.images,
        batch.qpos,
        batch.force_history,
        batch.action_chunk,
        batch.future_force_chunk,
        force_padding_mask=batch.force_padding_mask,
        future_padding_mask=batch.future_padding_mask,
        sample_posterior=True,
        return_intermediate_decoder=True,
    )
    losses = criterion(
        outputs,
        batch.action_chunk,
        batch.future_force_chunk,
        batch.future_padding_mask,
    )
    prior_isolation = _audit_prior_isolation(model, losses["loss_prior_match"])
    losses["loss_total"].backward()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    train_forward_backward_seconds = time.perf_counter() - start

    gradient_report = _gradient_report(modules)
    missing_gradient_modules = [
        name
        for name, report in gradient_report.items()
        if report["trainable_parameters"] > 0
        and report["parameters_with_gradient"] == 0
    ]
    if missing_gradient_modules:
        raise RuntimeError(
            "preflight found modules without gradients: "
            + ", ".join(missing_gradient_modules)
        )

    previous_training_mode = model.training
    model.eval()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    deployment_start = time.perf_counter()
    try:
        with torch.no_grad():
            zero_outputs = model(
                batch.images,
                batch.qpos,
                batch.force_history,
                force_padding_mask=batch.force_padding_mask,
                contact_latent_mode="zero",
            )
            prior_outputs = model(
                batch.images,
                batch.qpos,
                batch.force_history,
                force_padding_mask=batch.force_padding_mask,
                contact_latent_mode="prior",
                deterministic_prior=True,
            )
    finally:
        model.train(previous_training_mode)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    deployment_seconds = time.perf_counter() - deployment_start

    memory = {
        "device": str(device),
        "cuda_peak_allocated_bytes": (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else None
        ),
        "cuda_peak_reserved_bytes": (
            int(torch.cuda.max_memory_reserved(device))
            if device.type == "cuda"
            else None
        ),
    }
    return {
        "passed": True,
        "architecture_version": model.config.architecture_version,
        "batch_size": batch.batch_size,
        "model_config": model.config.checkpoint_metadata(),
        "training_config": criterion.config.checkpoint_metadata(),
        "output_shapes": _output_shapes(outputs),
        "deployment": {
            "zero_latent_source": zero_outputs["contact_latent_source"],
            "prior_latent_source": prior_outputs["contact_latent_source"],
            "zero_action_shape": list(zero_outputs["pred_action"].shape),
            "zero_force_shape": list(zero_outputs["pred_force"].shape),
            "prior_action_shape": list(prior_outputs["pred_action"].shape),
            "prior_force_shape": list(prior_outputs["pred_force"].shape),
        },
        "losses": {
            name: float(value.detach().item())
            for name, value in losses.items()
        },
        "parameters": parameter_report,
        "optimizer_groups": optimizer_report,
        "gradients": gradient_report,
        "prior_stop_gradient": prior_isolation,
        "timing_seconds": {
            "train_forward_backward": train_forward_backward_seconds,
            "two_deployment_forwards": deployment_seconds,
        },
        "memory": memory,
    }


def _diagnostic_modules(
    model: ACTAlignedContactCVAEPolicy,
) -> Mapping[str, nn.Module]:
    return {
        "vision_backbone": model.vision_backbone,
        "qpos_adapter": model.qpos_adapter,
        "online_force_encoder": model.online_force_encoder,
        "force_vision_fusion": model.force_vision_fusion,
        "contact_posterior": model.contact_posterior,
        "contact_prior": model.contact_prior,
        "contact_latent_adapter": model.contact_latent_adapter,
        "policy_special_position": model.policy_special_position,
        "policy_encoder": model.policy_encoder,
        "query_decoder": model.query_decoder,
        "action_head": model.action_head,
        "force_head": model.force_head,
    }


def _parameter_report(
    model: ACTAlignedContactCVAEPolicy,
    modules: Mapping[str, nn.Module],
) -> Dict[str, Any]:
    module_reports: Dict[str, Any] = {}
    grouped_parameter_ids = []
    for name, module in modules.items():
        parameters = list(module.parameters())
        grouped_parameter_ids.extend(id(parameter) for parameter in parameters)
        module_reports[name] = {
            "parameters": sum(parameter.numel() for parameter in parameters),
            "trainable_parameters": sum(
                parameter.numel()
                for parameter in parameters
                if parameter.requires_grad
            ),
        }
    model_parameters = list(model.parameters())
    if len(grouped_parameter_ids) != len(set(grouped_parameter_ids)):
        raise RuntimeError("diagnostic module groups overlap")
    if set(grouped_parameter_ids) != {id(parameter) for parameter in model_parameters}:
        raise RuntimeError("diagnostic module groups do not cover the model")
    return {
        "total": sum(parameter.numel() for parameter in model_parameters),
        "trainable": sum(
            parameter.numel()
            for parameter in model_parameters
            if parameter.requires_grad
        ),
        "by_module": module_reports,
    }


def _optimizer_report(
    model: ACTAlignedContactCVAEPolicy,
    optimizer: torch.optim.Optimizer,
) -> list[Dict[str, Any]]:
    reports = []
    optimizer_ids = []
    for index, group in enumerate(optimizer.param_groups):
        parameters = list(group["params"])
        optimizer_ids.extend(id(parameter) for parameter in parameters)
        reports.append(
            {
                "index": index,
                "name": group.get("name", f"group_{index}"),
                "learning_rate": float(group["lr"]),
                "weight_decay": float(group["weight_decay"]),
                "parameter_tensors": len(parameters),
                "parameters": sum(parameter.numel() for parameter in parameters),
            }
        )
    trainable_ids = {
        id(parameter) for parameter in model.parameters() if parameter.requires_grad
    }
    if len(optimizer_ids) != len(set(optimizer_ids)):
        raise RuntimeError("optimizer groups contain duplicate parameters")
    if set(optimizer_ids) != trainable_ids:
        raise RuntimeError("optimizer groups do not exactly cover trainable parameters")
    return reports


def _audit_prior_isolation(
    model: ACTAlignedContactCVAEPolicy,
    prior_match_loss: torch.Tensor,
) -> Dict[str, Any]:
    posterior_parameters = [
        parameter
        for parameter in model.contact_posterior.parameters()
        if parameter.requires_grad
    ]
    prior_parameters = [
        parameter
        for parameter in model.contact_prior.parameters()
        if parameter.requires_grad
    ]
    gradients = torch.autograd.grad(
        prior_match_loss,
        posterior_parameters + prior_parameters,
        retain_graph=True,
        allow_unused=True,
    )
    posterior_gradients = gradients[: len(posterior_parameters)]
    prior_gradients = gradients[len(posterior_parameters) :]
    posterior_norm = _optional_gradient_norm(posterior_gradients)
    prior_norm = _optional_gradient_norm(prior_gradients)
    passed = posterior_norm == 0.0 and prior_norm > 0.0
    if not passed:
        raise RuntimeError(
            "prior matching must update the prior without updating the posterior"
        )
    return {
        "passed": True,
        "posterior_gradient_norm_from_prior_match": posterior_norm,
        "prior_gradient_norm_from_prior_match": prior_norm,
    }


def _gradient_report(
    modules: Mapping[str, nn.Module],
) -> Dict[str, Dict[str, Any]]:
    reports = {}
    for name, module in modules.items():
        parameters = [
            parameter
            for parameter in module.parameters()
            if parameter.requires_grad
        ]
        gradients = [parameter.grad for parameter in parameters]
        norm = _optional_gradient_norm(gradients)
        if not math.isfinite(norm):
            raise FloatingPointError(f"{name} gradient norm is not finite")
        reports[name] = {
            "trainable_parameter_tensors": len(parameters),
            "trainable_parameters": sum(
                parameter.numel() for parameter in parameters
            ),
            "parameters_with_gradient": sum(
                gradient is not None for gradient in gradients
            ),
            "gradient_norm": norm,
        }
    return reports


def _optional_gradient_norm(
    gradients: Iterable[torch.Tensor | None],
) -> float:
    squared_norm = 0.0
    for gradient in gradients:
        if gradient is not None:
            squared_norm += float(
                gradient.detach().float().square().sum().item()
            )
    return math.sqrt(squared_norm)


def _output_shapes(outputs: Mapping[str, Any]) -> Dict[str, list[int]]:
    names = (
        "visual_tokens",
        "z_F_online",
        "z_VF",
        "mu_contact",
        "logvar_contact",
        "mu_contact_prior",
        "logvar_contact_prior",
        "policy_tokens",
        "policy_memory",
        "decoder_intermediate",
        "decoder_hidden",
        "pred_action",
        "pred_force",
    )
    return {
        name: list(outputs[name].shape)
        for name in names
        if isinstance(outputs.get(name), torch.Tensor)
    }
