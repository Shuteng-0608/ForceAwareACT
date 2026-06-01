import pytest
import torch

from force_aware_act.models import ContactPriorEncoder


def test_contact_prior_encoder_without_visual_summary_shapes():
    encoder = ContactPriorEncoder(
        d_model=128,
        z_dim=16,
        hidden_dim=64,
        dropout=0.0,
    )
    z_q = torch.randn(2, 128)
    z_F_online = torch.randn(2, 128)
    z_VF = torch.randn(2, 128)

    mu, logvar, z = encoder(z_q, z_F_online, z_VF)

    assert mu.shape == (2, 16)
    assert logvar.shape == (2, 16)
    assert z.shape == (2, 16)


def test_contact_prior_encoder_with_visual_summary_shapes():
    encoder = ContactPriorEncoder(
        d_model=128,
        z_dim=16,
        hidden_dim=64,
        dropout=0.0,
        use_visual_summary=True,
    )
    z_q = torch.randn(2, 128)
    z_F_online = torch.randn(2, 128)
    z_VF = torch.randn(2, 128)
    visual_summary = torch.randn(2, 128)

    mu, logvar, z = encoder(z_q, z_F_online, z_VF, visual_summary=visual_summary)

    assert mu.shape == (2, 16)
    assert logvar.shape == (2, 16)
    assert z.shape == (2, 16)


def test_contact_prior_encoder_rejects_mismatched_batch_size():
    encoder = ContactPriorEncoder(d_model=128, z_dim=16, hidden_dim=64)
    z_q = torch.randn(2, 128)
    z_F_online = torch.randn(3, 128)
    z_VF = torch.randn(2, 128)

    with pytest.raises(ValueError, match="batch size"):
        encoder(z_q, z_F_online, z_VF)


def test_contact_prior_encoder_rejects_mismatched_feature_dim():
    encoder = ContactPriorEncoder(d_model=128, z_dim=16, hidden_dim=64)
    z_q = torch.randn(2, 128)
    z_F_online = torch.randn(2, 64)
    z_VF = torch.randn(2, 128)

    with pytest.raises(ValueError, match="last dimension"):
        encoder(z_q, z_F_online, z_VF)


def test_contact_prior_encoder_rejects_visual_summary_when_disabled():
    encoder = ContactPriorEncoder(
        d_model=128,
        z_dim=16,
        hidden_dim=64,
        use_visual_summary=False,
    )
    z_q = torch.randn(2, 128)
    z_F_online = torch.randn(2, 128)
    z_VF = torch.randn(2, 128)
    visual_summary = torch.randn(2, 128)

    with pytest.raises(ValueError, match="use_visual_summary=False"):
        encoder(z_q, z_F_online, z_VF, visual_summary=visual_summary)


def test_contact_prior_encoder_gradients_flow_through_inputs_and_parameters():
    encoder = ContactPriorEncoder(
        d_model=64,
        z_dim=8,
        hidden_dim=32,
        dropout=0.0,
    )
    z_q = torch.randn(2, 64, requires_grad=True)
    z_F_online = torch.randn(2, 64, requires_grad=True)
    z_VF = torch.randn(2, 64, requires_grad=True)
    visual_summary = torch.randn(2, 64, requires_grad=True)

    mu, logvar, z = encoder(z_q, z_F_online, z_VF, visual_summary=visual_summary)
    loss = mu.pow(2).mean() + logvar.pow(2).mean() + z.pow(2).mean()
    loss.backward()

    assert z_q.grad is not None
    assert z_q.grad.abs().sum() > 0
    assert z_F_online.grad is not None
    assert z_F_online.grad.abs().sum() > 0
    assert z_VF.grad is not None
    assert z_VF.grad.abs().sum() > 0
    assert visual_summary.grad is not None
    assert visual_summary.grad.abs().sum() > 0
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in encoder.parameters()
    )


def test_contact_prior_encoder_mu_logvar_are_deterministic_independent_of_sampling():
    encoder = ContactPriorEncoder(
        d_model=64,
        z_dim=8,
        hidden_dim=32,
        dropout=0.0,
    )
    z_q = torch.randn(2, 64)
    z_F_online = torch.randn(2, 64)
    z_VF = torch.randn(2, 64)

    mu_1, logvar_1, z_1 = encoder(z_q, z_F_online, z_VF)
    mu_2, logvar_2, z_2 = encoder(z_q, z_F_online, z_VF)

    assert mu_1.shape == z_1.shape == (2, 8)
    assert mu_2.shape == z_2.shape == (2, 8)
    assert torch.allclose(mu_1, mu_2)
    assert torch.allclose(logvar_1, logvar_2)
