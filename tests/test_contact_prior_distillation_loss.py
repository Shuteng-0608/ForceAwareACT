import pytest
import torch

from force_aware_act.training import compute_contact_prior_distillation_loss


def test_contact_prior_distillation_mse_mu_returns_scalar():
    mu_prior = torch.randn(2, 16)
    logvar_prior = torch.zeros(2, 16)
    mu_posterior = torch.randn(2, 16)

    losses = compute_contact_prior_distillation_loss(
        mu_prior,
        logvar_prior,
        mu_posterior,
        mode="mse_mu",
    )

    assert losses["loss_prior"].shape == ()
    assert losses["loss_prior_mse_mu"].shape == ()
    assert losses["mode"] == "mse_mu"


def test_contact_prior_distillation_mse_mu_detaches_posterior_target():
    mu_prior = torch.randn(2, 16, requires_grad=True)
    logvar_prior = torch.zeros(2, 16, requires_grad=True)
    mu_posterior = torch.randn(2, 16, requires_grad=True)

    losses = compute_contact_prior_distillation_loss(
        mu_prior,
        logvar_prior,
        mu_posterior,
        mode="mse_mu",
    )
    losses["loss_prior"].backward()

    assert mu_prior.grad is not None
    assert mu_prior.grad.abs().sum() > 0
    assert mu_posterior.grad is None
    assert logvar_prior.grad is None


def test_contact_prior_distillation_kl_q_to_p_returns_scalar():
    mu_prior = torch.randn(2, 16, requires_grad=True)
    logvar_prior = torch.zeros(2, 16, requires_grad=True)
    mu_posterior = torch.randn(2, 16, requires_grad=True)
    logvar_posterior = torch.randn(2, 16, requires_grad=True)

    losses = compute_contact_prior_distillation_loss(
        mu_prior,
        logvar_prior,
        mu_posterior,
        logvar_posterior=logvar_posterior,
        mode="kl_q_to_p",
        beta_kl=0.5,
    )

    assert losses["loss_prior"].shape == ()
    assert losses["loss_prior_kl"].shape == ()
    assert losses["mode"] == "kl_q_to_p"

    losses["loss_prior"].backward()
    assert mu_prior.grad is not None
    assert mu_prior.grad.abs().sum() > 0
    assert logvar_prior.grad is not None
    assert logvar_prior.grad.abs().sum() > 0
    assert mu_posterior.grad is None
    assert logvar_posterior.grad is None


def test_contact_prior_distillation_rejects_shape_mismatch():
    mu_prior = torch.randn(2, 16)
    logvar_prior = torch.zeros(2, 16)
    mu_posterior = torch.randn(2, 8)

    with pytest.raises(ValueError, match="mu_posterior must have shape"):
        compute_contact_prior_distillation_loss(
            mu_prior,
            logvar_prior,
            mu_posterior,
        )


def test_contact_prior_distillation_rejects_non_matrix_latent():
    mu_prior = torch.randn(2, 4, 16)
    logvar_prior = torch.zeros(2, 4, 16)
    mu_posterior = torch.randn(2, 4, 16)

    with pytest.raises(ValueError, match=r"must have shape \[B, z_dim\]"):
        compute_contact_prior_distillation_loss(
            mu_prior,
            logvar_prior,
            mu_posterior,
        )


def test_contact_prior_distillation_requires_logvar_posterior_for_kl():
    mu_prior = torch.randn(2, 16)
    logvar_prior = torch.zeros(2, 16)
    mu_posterior = torch.randn(2, 16)

    with pytest.raises(ValueError, match="logvar_posterior is required"):
        compute_contact_prior_distillation_loss(
            mu_prior,
            logvar_prior,
            mu_posterior,
            mode="kl_q_to_p",
        )
