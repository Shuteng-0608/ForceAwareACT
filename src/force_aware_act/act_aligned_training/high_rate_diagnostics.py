"""Strict preflight diagnostics for native 500 Hz contact-CVAE training."""

from __future__ import annotations

import math
from typing import Any, Dict

import torch

from force_aware_act.act_aligned_training.high_rate_batch import ACTAlignedHighRateBatch
from force_aware_act.act_aligned_training.high_rate_losses import ACTAlignedHighRateCriterion
from force_aware_act.act_aligned_training.high_rate_trainer import _losses, _training_forward
from force_aware_act.models.act_aligned.high_rate_policy import ACTAlignedHighRateContactCVAEPolicy


def run_high_rate_training_preflight(
    model: ACTAlignedHighRateContactCVAEPolicy,
    criterion: ACTAlignedHighRateCriterion,
    optimizer: torch.optim.Optimizer,
    batch: ACTAlignedHighRateBatch,
) -> Dict[str, Any]:
    """Audit graph connectivity, stop-gradient semantics and deployment inputs."""

    if not isinstance(model, ACTAlignedHighRateContactCVAEPolicy):
        raise TypeError("model must be ACTAlignedHighRateContactCVAEPolicy")
    if not isinstance(criterion, ACTAlignedHighRateCriterion):
        raise TypeError("criterion must be ACTAlignedHighRateCriterion")
    batch.validate(model.config)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer_parameters = [
        parameter for group in optimizer.param_groups for parameter in group["params"]
    ]
    if len({id(parameter) for parameter in trainable}) != len(trainable):
        raise RuntimeError("model registers a trainable parameter more than once")
    if len({id(parameter) for parameter in optimizer_parameters}) != len(optimizer_parameters):
        raise RuntimeError("optimizer registers a parameter more than once")
    if {id(parameter) for parameter in optimizer_parameters} != {id(parameter) for parameter in trainable}:
        raise RuntimeError("optimizer does not exactly cover trainable parameters")

    model.train()
    optimizer.zero_grad(set_to_none=True)
    outputs = _training_forward(model, batch, sample_posterior=True)
    losses = _losses(criterion, outputs, batch)
    isolated_parameters = (
        list(model.contact_posterior.parameters())
        + list(model.high_rate_force_encoder.parameters())
        + list(model.contact_prior.parameters())
    )
    isolated_gradients = torch.autograd.grad(
        losses["loss_prior_match"], isolated_parameters,
        retain_graph=True, allow_unused=True,
    )
    posterior_count = len(list(model.contact_posterior.parameters()))
    shared_count = len(list(model.high_rate_force_encoder.parameters()))
    posterior_norm = _gradient_norm(isolated_gradients[:posterior_count])
    shared_norm = _gradient_norm(
        isolated_gradients[posterior_count:posterior_count + shared_count]
    )
    prior_norm = _gradient_norm(isolated_gradients[posterior_count + shared_count:])
    if posterior_norm != 0.0 or shared_norm != 0.0 or prior_norm <= 0.0:
        raise RuntimeError("prior-match stop-gradient isolation failed")
    losses["loss_total"].backward()

    modules = {
        "vision_backbone": model.vision_backbone,
        "qpos_adapter": model.qpos_adapter,
        "shared_high_rate_force_encoder": model.high_rate_force_encoder,
        "online_force_encoder": model.online_force_encoder,
        "force_vision_fusion": model.force_vision_fusion,
        "contact_posterior": model.contact_posterior,
        "contact_prior": model.contact_prior,
        "policy_encoder": model.policy_encoder,
        "query_decoder": model.query_decoder,
        "action_head": model.action_head,
        "force_head": model.force_head,
        "high_rate_force_head": model.high_rate_force_head,
    }
    gradient_report = {}
    for name, module in modules.items():
        parameters = [parameter for parameter in module.parameters() if parameter.requires_grad]
        norm = _gradient_norm(parameter.grad for parameter in parameters)
        count = sum(parameter.grad is not None for parameter in parameters)
        if parameters and (count == 0 or not math.isfinite(norm)):
            raise RuntimeError(f"module {name!r} has no finite training gradient")
        gradient_report[name] = {
            "trainable_parameter_tensors": len(parameters),
            "parameters_with_gradient": count,
            "gradient_norm": norm,
        }

    previous_mode = model.training
    model.eval()
    try:
        with torch.no_grad():
            zero = model(
                batch.images, batch.qpos,
                batch.online_force_intervals, batch.online_force_relative_time,
                batch.online_force_sample_padding_mask,
                batch.online_force_interval_padding_mask,
                contact_latent_mode="zero",
            )
            future = batch.future_force_intervals.clone()
            valid = (~batch.future_force_sample_padding_mask).nonzero()
            if valid.numel() == 0:
                raise RuntimeError("posterior intervention requires a valid future force sample")
            index = tuple(int(value) for value in valid[0])
            future[index + (0,)] += 100.0
            baseline = _training_forward(model, batch, sample_posterior=False)
            intervened_batch = ACTAlignedHighRateBatch(
                **{
                    **batch.__dict__,
                    "future_force_intervals": future,
                }
            )
            intervened = _training_forward(model, intervened_batch, sample_posterior=False)
    finally:
        model.train(previous_mode)
    if not torch.equal(zero["z_contact"], torch.zeros_like(zero["z_contact"])):
        raise RuntimeError("deployment zero latent is not exact zero")
    posterior_delta = float(
        (baseline["mu_contact"] - intervened["mu_contact"]).abs().max().item()
    )
    if posterior_delta <= 0.0:
        raise RuntimeError("future 500 Hz force intervention did not affect posterior")

    return {
        "passed": True,
        "architecture_version": model.config.architecture_version,
        "training_version": criterion.config.training_version,
        "batch_size": batch.batch_size,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "optimizer_parameter_tensors": len(optimizer_parameters),
        "output_shapes": {
            name: list(outputs[name].shape)
            for name in ("pred_action", "pred_force", "pred_force_highrate", "mu_contact")
        },
        "losses": {name: float(value.detach().item()) for name, value in losses.items()},
        "gradient_modules": gradient_report,
        "prior_stop_gradient": {
            "passed": True,
            "posterior_gradient_norm": posterior_norm,
            "shared_high_rate_encoder_gradient_norm": shared_norm,
            "prior_gradient_norm": prior_norm,
        },
        "future_force_intervention": {
            "passed": True,
            "posterior_mean_max_abs_delta": posterior_delta,
            "intervention_new_information": "one_valid_2ms_force_sample_plus_100_normalized_units",
        },
        "deployment": {
            "input_contract": "online_images_qpos_and_causal_last100_500hz_force_only",
            "contact_latent_source": zero["contact_latent_source"],
            "zero_latent_exact": True,
        },
    }


def _gradient_norm(gradients) -> float:
    return math.sqrt(sum(
        float(gradient.detach().float().square().sum().item())
        for gradient in gradients if gradient is not None
    ))
