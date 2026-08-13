"""Preflight audits for native-rate motion and latent-free controls."""

from __future__ import annotations

import inspect
import math
from typing import Any, Dict

import torch

from force_aware_act.act_aligned_training.high_rate_batch import (
    ACTAlignedHighRateBatch,
)
from force_aware_act.act_aligned_training.high_rate_control_trainer import (
    _criterion_inputs,
    _online_inputs,
)
from force_aware_act.models.act_aligned.high_rate_dual_zero_policy import (
    ACTAlignedHighRateDualZeroPolicy,
)
from force_aware_act.models.act_aligned.high_rate_motion_policy import (
    ACTAlignedHighRateMotionCVAEPolicy,
)


def run_high_rate_control_preflight(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: ACTAlignedHighRateBatch,
) -> Dict[str, Any]:
    """Audit native-force use, graph coverage and latent semantics."""

    if not isinstance(
        model,
        (ACTAlignedHighRateMotionCVAEPolicy, ACTAlignedHighRateDualZeroPolicy),
    ):
        raise TypeError("model must be a native-rate motion or dual-zero policy")
    batch.validate(model.config)
    _validate_optimizer_coverage(model, optimizer)
    is_motion = isinstance(model, ACTAlignedHighRateMotionCVAEPolicy)

    model.train()
    optimizer.zero_grad(set_to_none=True)
    if is_motion:
        outputs = model.forward_train(
            *_online_inputs(batch),
            batch.action_chunk,
            action_padding_mask=batch.action_padding_mask,
            sample_posterior=True,
            return_intermediate_decoder=True,
        )
    else:
        outputs = model(
            *_online_inputs(batch),
            return_intermediate_decoder=True,
        )
    losses = criterion(outputs, *_criterion_inputs(batch))
    losses["loss_total"].backward()
    modules = _diagnostic_modules(model, is_motion=is_motion)
    gradient_report = _gradient_report(modules)

    previous_mode = model.training
    model.eval()
    try:
        with torch.no_grad():
            deployment = model(*_online_inputs(batch))
            intervention_batch, intervention_index = _intervene_online_force(batch)
            intervened = model(*_online_inputs(intervention_batch))
    finally:
        model.train(previous_mode)
    action_delta = float(
        (deployment["pred_action"] - intervened["pred_action"])
        .abs()
        .max()
        .item()
    )
    force_feature_delta = float(
        (deployment["z_F_online"] - intervened["z_F_online"])
        .abs()
        .max()
        .item()
    )
    if action_delta <= 0.0 or force_feature_delta <= 0.0:
        raise RuntimeError("native 500 Hz force intervention did not reach policy")

    if is_motion:
        if not torch.equal(
            deployment["z_motion"],
            torch.zeros_like(deployment["z_motion"]),
        ):
            raise RuntimeError("motion deployment latent is not exact zero")
        if "future_force_intervals" in inspect.signature(
            model.forward_train
        ).parameters:
            raise RuntimeError("motion posterior accepts future force")
        latent_report = {
            "mechanism": "motion_cvae",
            "deployment_source": deployment["motion_latent_source"],
            "deployment_zero_exact": True,
            "posterior_inputs": ("qpos", "action_chunk"),
        }
    else:
        forbidden = ("latent", "posterior", "prior")
        forbidden_modules = [
            name
            for name, _module in model.named_modules()
            if any(token in name for token in forbidden)
        ]
        if forbidden_modules or deployment.get("latent_mechanism") != "none":
            raise RuntimeError("dual-zero policy is not structurally latent-free")
        latent_report = {
            "mechanism": "none",
            "deployment_source": "none",
            "deployment_zero_exact": None,
            "forbidden_modules": forbidden_modules,
        }
    return {
        "passed": True,
        "architecture_version": model.config.architecture_version,
        "training_version": criterion.config.training_version,
        "force_input_contract": model.config.checkpoint_metadata()[
            "force_input_contract"
        ],
        "output_shapes": {
            name: list(outputs[name].shape)
            for name in (
                "pred_action",
                "pred_force",
                "pred_force_highrate",
            )
        },
        "losses": {
            name: float(value.detach().item()) for name, value in losses.items()
        },
        "gradient_modules": gradient_report,
        "native_force_intervention": {
            "passed": True,
            "sample_index": intervention_index,
            "normalized_wrench_delta": 100.0,
            "online_force_feature_max_abs_delta": force_feature_delta,
            "action_max_abs_delta": action_delta,
        },
        "latent": latent_report,
    }


def _validate_optimizer_coverage(model, optimizer) -> None:
    trainable = {
        id(parameter)
        for parameter in model.parameters()
        if parameter.requires_grad
    }
    grouped = [
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    ]
    if len(grouped) != len(set(grouped)) or set(grouped) != trainable:
        raise RuntimeError("optimizer does not exactly cover trainable parameters")


def _diagnostic_modules(model, *, is_motion: bool) -> dict[str, torch.nn.Module]:
    modules = {
        "vision_backbone": model.vision_backbone,
        "qpos_adapter": model.qpos_adapter,
        "shared_high_rate_force_encoder": model.high_rate_force_encoder,
        "online_force_encoder": model.online_force_encoder,
        "force_vision_fusion": model.force_vision_fusion,
        "policy_encoder": model.policy_encoder,
        "query_decoder": model.query_decoder,
        "action_head": model.action_head,
        "force_head": model.force_head,
        "high_rate_force_head": model.high_rate_force_head,
    }
    if is_motion:
        modules.update(
            {
                "motion_posterior": model.motion_posterior,
                "motion_latent_adapter": model.motion_latent_adapter,
            }
        )
    return modules


def _gradient_report(modules) -> Dict[str, Any]:
    report = {}
    for name, module in modules.items():
        parameters = [
            parameter for parameter in module.parameters() if parameter.requires_grad
        ]
        gradients = [
            parameter.grad for parameter in parameters if parameter.grad is not None
        ]
        norm = math.sqrt(
            sum(
                float(gradient.detach().float().square().sum().item())
                for gradient in gradients
            )
        )
        if not gradients or not math.isfinite(norm) or norm <= 0.0:
            raise RuntimeError(f"module {name!r} has no finite nonzero gradient")
        report[name] = {
            "trainable_parameter_tensors": len(parameters),
            "parameters_with_gradient": len(gradients),
            "gradient_norm": norm,
        }
    return report


def _intervene_online_force(
    batch: ACTAlignedHighRateBatch,
) -> tuple[ACTAlignedHighRateBatch, list[int]]:
    valid = (~batch.online_force_sample_padding_mask).nonzero()
    if valid.numel() == 0:
        raise RuntimeError("online force intervention requires one valid sample")
    index = [int(value) for value in valid[-1]]
    force = batch.online_force_intervals.clone()
    force[tuple(index) + (0,)] += 100.0
    return (
        ACTAlignedHighRateBatch(
            **{
                **batch.__dict__,
                "online_force_intervals": force,
            }
        ),
        index,
    )
