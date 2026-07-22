from dataclasses import replace

import pytest
import torch

from force_aware_act.training.policies import compute_policy_training_loss
from force_aware_act.training.protocol import ObjectiveSpec


class _TrainingPolicy(torch.nn.Module):
    def __init__(self, variant):
        super().__init__()
        self.variant = variant
        self.weight = torch.nn.Parameter(torch.tensor(0.0))
        self.last_kwargs = None

    def forward(self, **kwargs):
        self.last_kwargs = kwargs
        batch = kwargs["qpos"].shape[0]
        action = self.weight + torch.zeros(batch, 2, 7)
        force = self.weight + torch.zeros(batch, 2, 6)
        latent = self.weight + torch.zeros(batch, 3)
        outputs = {"pred_action": action, "pred_force": force}
        if self.variant != "force_aware_contact_cvae":
            outputs.update({"mu_motion": latent, "logvar_motion": torch.zeros_like(latent)})
        if self.variant != "force_aware_motion_cvae":
            outputs.update({"mu_contact": latent, "logvar_contact": torch.zeros_like(latent)})
            outputs.update(
                {
                    "mu_contact_prior": latent,
                    "logvar_contact_prior": torch.zeros_like(latent),
                }
            )
        return outputs


def _batch():
    return {
        "images": torch.zeros(2, 1, 3, 4, 4),
        "qpos": torch.zeros(2, 7),
        "force_window": torch.zeros(2, 3, 6),
        "action_chunk": torch.ones(2, 2, 7),
        "future_force_chunk": torch.ones(2, 2, 6),
    }


def _objective():
    return ObjectiveSpec(
        lambda_force=0.1,
        lambda_prior=0.0,
        prior_loss_mode="mse_mu",
        beta_motion_max=1.0e-4,
        beta_contact_max=2.0e-4,
        warmup_steps=10,
        train_latent_mode="posterior",
        train_contact_latent_mode="posterior",
        validation_deployment_mode="auto",
    )


@pytest.mark.parametrize(
    "variant",
    ["force_aware_act", "force_aware_motion_cvae", "force_aware_contact_cvae"],
)
def test_policy_training_routes_supported_variants(variant):
    model = _TrainingPolicy(variant)
    losses, metadata = compute_policy_training_loss(
        model=model,
        batch=_batch(),
        policy_variant=variant,
        objective=_objective(),
        stage_step=5,
    )

    assert losses["loss_total"].ndim == 0
    losses["loss_total"].backward()
    assert model.weight.grad is not None
    assert model.last_kwargs["is_training"] is True
    assert metadata["beta_motion"] == pytest.approx(0.00005 if variant != "force_aware_contact_cvae" else 0.0)


def test_zero_latent_disables_posterior_labels_in_loss_but_remains_training_only():
    model = _TrainingPolicy("force_aware_act")
    objective = replace(_objective(), train_latent_mode="zero")
    losses, _ = compute_policy_training_loss(
        model=model,
        batch=_batch(),
        policy_variant="force_aware_act",
        objective=objective,
        stage_step=1,
    )

    assert losses["kl_motion"].item() == 0.0
    assert losses["kl_contact"].item() == 0.0
    assert model.last_kwargs["contact_latent_mode"] == "zero"


def test_policy_training_rejects_nonfinite_loss():
    model = _TrainingPolicy("force_aware_motion_cvae")
    batch = _batch()
    batch["action_chunk"][0, 0, 0] = float("nan")
    with pytest.raises(FloatingPointError, match="non-finite"):
        compute_policy_training_loss(
            model=model,
            batch=batch,
            policy_variant="force_aware_motion_cvae",
            objective=_objective(),
            stage_step=1,
        )
