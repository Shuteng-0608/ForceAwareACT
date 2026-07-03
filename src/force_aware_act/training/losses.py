"""Loss utilities for ForceAwareACT training."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Union

import torch
import torch.nn.functional as functional

from force_aware_act.models import kl_normal


REQUIRED_OUTPUT_KEYS = (
    "pred_action",
    "pred_force",
)
POSTERIOR_OUTPUT_KEYS = (
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
    lambda_prior: float = 0.0,
    prior_loss_mode: str = "mse_mu",
    use_posterior_kl: bool = True,
) -> Dict[str, Union[torch.Tensor, float, str]]:
    """Compute the supervised action/force and posterior KL losses."""

    _validate_required_outputs(outputs)
    pred_action = outputs["pred_action"]
    pred_force = outputs["pred_force"]
    _validate_tensor_shape("pred_action", pred_action, action_chunk.shape)
    _validate_tensor_shape("pred_force", pred_force, future_force_chunk.shape)

    loss_action = functional.l1_loss(pred_action, action_chunk)
    loss_force = functional.l1_loss(pred_force, future_force_chunk)
    if use_posterior_kl:
        _validate_posterior_outputs(outputs)
        kl_motion = kl_normal(outputs["mu_motion"], outputs["logvar_motion"])
        kl_contact = kl_normal(outputs["mu_contact"], outputs["logvar_contact"])
    else:
        kl_motion = pred_action.new_zeros(())
        kl_contact = pred_action.new_zeros(())
    loss_total = (
        loss_action
        + lambda_force * loss_force
        + beta_motion * kl_motion
        + beta_contact * kl_contact
    )
    loss_prior = pred_action.new_zeros(())
    if lambda_prior > 0:
        if not use_posterior_kl:
            raise ValueError("lambda_prior > 0 requires use_posterior_kl=True")
        _validate_prior_outputs(outputs)
        prior_losses = compute_contact_prior_distillation_loss(
            mu_prior=outputs["mu_contact_prior"],
            logvar_prior=outputs["logvar_contact_prior"],
            mu_posterior=outputs["mu_contact"],
            logvar_posterior=outputs["logvar_contact"],
            mode=prior_loss_mode,
        )
        loss_prior = prior_losses["loss_prior"]
        loss_total = loss_total + lambda_prior * loss_prior

    return {
        "loss_total": loss_total,
        "loss_action": loss_action,
        "loss_force": loss_force,
        "kl_motion": kl_motion,
        "kl_contact": kl_contact,
        "loss_prior": loss_prior,
        "lambda_force": lambda_force,
        "beta_motion": beta_motion,
        "beta_contact": beta_contact,
        "lambda_prior": lambda_prior,
        "prior_loss_mode": prior_loss_mode,
        "use_posterior_kl": use_posterior_kl,
    }


def compute_act_baseline_loss(
    outputs: Mapping[str, Any],
    action_chunk: torch.Tensor,
) -> Dict[str, Union[torch.Tensor, float, str]]:
    """Compute the ACT baseline action reconstruction loss only."""

    if "pred_action" not in outputs:
        raise KeyError("outputs is missing required key: pred_action")
    pred_action = outputs["pred_action"]
    if not isinstance(pred_action, torch.Tensor):
        raise ValueError("outputs['pred_action'] must be a torch.Tensor")
    _validate_tensor_shape("pred_action", pred_action, action_chunk.shape)
    loss_action = functional.l1_loss(pred_action, action_chunk)
    return {
        "loss_total": loss_action,
        "loss_action": loss_action,
        "policy_variant": "act_baseline",
    }


def compute_force_aware_motion_cvae_loss(
    outputs: Mapping[str, Any],
    action_chunk: torch.Tensor,
    future_force_chunk: torch.Tensor,
    lambda_force: float = 0.1,
    beta_motion: float = 1.0e-4,
) -> Dict[str, Union[torch.Tensor, float, str]]:
    """Compute Motion-CVAE action/force supervision plus motion KL only."""

    _validate_required_outputs(outputs)
    for key in ("mu_motion", "logvar_motion"):
        if key not in outputs:
            raise KeyError(f"outputs is missing required motion posterior key: {key}")
        if not isinstance(outputs[key], torch.Tensor):
            raise ValueError(f"outputs[{key!r}] must be a torch.Tensor")

    pred_action = outputs["pred_action"]
    pred_force = outputs["pred_force"]
    _validate_tensor_shape("pred_action", pred_action, action_chunk.shape)
    _validate_tensor_shape("pred_force", pred_force, future_force_chunk.shape)

    loss_action = functional.l1_loss(pred_action, action_chunk)
    loss_force = functional.l1_loss(pred_force, future_force_chunk)
    kl_motion = kl_normal(outputs["mu_motion"], outputs["logvar_motion"])
    loss_total = loss_action + lambda_force * loss_force + beta_motion * kl_motion
    return {
        "loss_total": loss_total,
        "loss_action": loss_action,
        "loss_force": loss_force,
        "kl_motion": kl_motion,
        "lambda_force": lambda_force,
        "beta_motion": beta_motion,
        "policy_variant": "force_aware_motion_cvae",
    }


def compute_contact_prior_distillation_loss(
    mu_prior: torch.Tensor,
    logvar_prior: torch.Tensor,
    mu_posterior: torch.Tensor,
    logvar_posterior: torch.Tensor | None = None,
    mode: str = "mse_mu",
    beta_kl: float = 1.0,
) -> Dict[str, Union[torch.Tensor, str]]:
    """Compute a contact-prior distillation loss against posterior targets."""

    _validate_latent_pair("prior", mu_prior, logvar_prior)
    _validate_latent_tensor("mu_posterior", mu_posterior)
    _validate_tensor_shape("mu_posterior", mu_posterior, mu_prior.shape)

    if mode == "mse_mu":
        loss_prior_mse_mu = functional.mse_loss(mu_prior, mu_posterior.detach())
        return {
            "loss_prior": loss_prior_mse_mu,
            "loss_prior_mse_mu": loss_prior_mse_mu,
            "mode": mode,
        }

    if mode == "kl_q_to_p":
        if logvar_posterior is None:
            raise ValueError("logvar_posterior is required when mode='kl_q_to_p'")
        _validate_latent_pair("posterior", mu_posterior, logvar_posterior)
        _validate_tensor_shape("logvar_posterior", logvar_posterior, mu_prior.shape)
        loss_prior_kl = beta_kl * _kl_normal_q_to_p(
            mu_q=mu_posterior.detach(),
            logvar_q=logvar_posterior.detach(),
            mu_p=mu_prior,
            logvar_p=logvar_prior,
        )
        return {
            "loss_prior": loss_prior_kl,
            "loss_prior_kl": loss_prior_kl,
            "mode": mode,
        }

    raise ValueError("mode must be one of: 'mse_mu', 'kl_q_to_p'")


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


def _validate_posterior_outputs(outputs: Mapping[str, Any]) -> None:
    for key in POSTERIOR_OUTPUT_KEYS:
        if key not in outputs:
            raise KeyError(f"outputs is missing required posterior key: {key}")
        if not isinstance(outputs[key], torch.Tensor):
            raise ValueError(f"outputs[{key!r}] must be a torch.Tensor")


def _validate_prior_outputs(outputs: Mapping[str, Any]) -> None:
    for key in ("mu_contact_prior", "logvar_contact_prior", "mu_contact", "logvar_contact"):
        if key not in outputs:
            raise KeyError(
                "outputs is missing required contact-prior distillation key: "
                f"{key}"
            )
        if not isinstance(outputs[key], torch.Tensor):
            raise ValueError(f"outputs[{key!r}] must be a torch.Tensor")


def _validate_tensor_shape(name: str, tensor: torch.Tensor, expected_shape: torch.Size) -> None:
    if tensor.shape != expected_shape:
        raise ValueError(
            f"{name} must have shape {tuple(expected_shape)}, got {tuple(tensor.shape)}"
        )


def _validate_latent_tensor(name: str, tensor: torch.Tensor) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise ValueError(f"{name} must be a torch.Tensor")
    if tensor.ndim != 2:
        raise ValueError(f"{name} must have shape [B, z_dim]")


def _validate_latent_pair(name: str, mu: torch.Tensor, logvar: torch.Tensor) -> None:
    _validate_latent_tensor(f"mu_{name}", mu)
    _validate_latent_tensor(f"logvar_{name}", logvar)
    if mu.shape != logvar.shape:
        raise ValueError(
            f"mu_{name} and logvar_{name} must have matching shapes, "
            f"got {tuple(mu.shape)} and {tuple(logvar.shape)}"
        )


def _kl_normal_q_to_p(
    mu_q: torch.Tensor,
    logvar_q: torch.Tensor,
    mu_p: torch.Tensor,
    logvar_p: torch.Tensor,
) -> torch.Tensor:
    var_q = logvar_q.exp()
    var_p = logvar_p.exp()
    kl_per_dim = logvar_p - logvar_q + (var_q + (mu_q - mu_p).pow(2)) / var_p - 1.0
    return 0.5 * kl_per_dim.sum(dim=-1).mean()
