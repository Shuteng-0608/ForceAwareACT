import pytest
import torch

from force_aware_act.models import (
    ContactPosteriorEncoder,
    MotionPosteriorEncoder,
    kl_normal,
)


def test_motion_posterior_encoder_default_shapes():
    encoder = MotionPosteriorEncoder(dropout=0.0)
    qpos = torch.randn(2, 7)
    action_chunk = torch.randn(2, 50, 7)

    mu, logvar, z = encoder(qpos, action_chunk)

    assert mu.shape == (2, 32)
    assert logvar.shape == (2, 32)
    assert z.shape == (2, 32)


def test_contact_posterior_encoder_default_shapes():
    encoder = ContactPosteriorEncoder(dropout=0.0)
    qpos = torch.randn(2, 7)
    action_chunk = torch.randn(2, 50, 7)
    future_force_chunk = torch.randn(2, 50, 6)

    mu, logvar, z = encoder(qpos, action_chunk, future_force_chunk)

    assert mu.shape == (2, 32)
    assert logvar.shape == (2, 32)
    assert z.shape == (2, 32)


def test_posterior_encoders_support_smaller_model_and_latent_dims():
    motion_encoder = MotionPosteriorEncoder(
        d_model=128,
        z_dim=16,
        nhead=4,
        num_layers=1,
        dim_feedforward=256,
        dropout=0.0,
    )
    contact_encoder = ContactPosteriorEncoder(
        d_model=128,
        z_dim=16,
        nhead=4,
        num_layers=1,
        dim_feedforward=256,
        dropout=0.0,
    )
    qpos = torch.randn(2, 7)
    action_chunk = torch.randn(2, 50, 7)
    future_force_chunk = torch.randn(2, 50, 6)

    motion_mu, motion_logvar, motion_z = motion_encoder(qpos, action_chunk)
    contact_mu, contact_logvar, contact_z = contact_encoder(
        qpos,
        action_chunk,
        future_force_chunk,
    )

    assert motion_mu.shape == (2, 16)
    assert motion_logvar.shape == (2, 16)
    assert motion_z.shape == (2, 16)
    assert contact_mu.shape == (2, 16)
    assert contact_logvar.shape == (2, 16)
    assert contact_z.shape == (2, 16)


def test_motion_posterior_encoder_rejects_long_chunk():
    encoder = MotionPosteriorEncoder(max_chunk_len=4)
    qpos = torch.randn(2, 7)
    action_chunk = torch.randn(2, 5, 7)

    with pytest.raises(ValueError, match="exceeds max_chunk_len"):
        encoder(qpos, action_chunk)


def test_contact_posterior_encoder_rejects_long_chunk():
    encoder = ContactPosteriorEncoder(max_chunk_len=4)
    qpos = torch.randn(2, 7)
    action_chunk = torch.randn(2, 5, 7)
    future_force_chunk = torch.randn(2, 5, 6)

    with pytest.raises(ValueError, match="exceeds max_chunk_len"):
        encoder(qpos, action_chunk, future_force_chunk)


def test_contact_posterior_encoder_rejects_future_force_shape_mismatch():
    encoder = ContactPosteriorEncoder(max_chunk_len=8)
    qpos = torch.randn(2, 7)
    action_chunk = torch.randn(2, 5, 7)
    future_force_chunk = torch.randn(2, 4, 6)

    with pytest.raises(ValueError, match="future_force_chunk must have shape"):
        encoder(qpos, action_chunk, future_force_chunk)


def test_kl_normal_returns_scalar():
    mu = torch.zeros(2, 32)
    logvar = torch.zeros(2, 32)

    kl = kl_normal(mu, logvar)

    assert kl.shape == ()


def test_motion_posterior_encoder_gradients_flow():
    encoder = MotionPosteriorEncoder(
        d_model=64,
        z_dim=8,
        nhead=4,
        num_layers=1,
        dim_feedforward=128,
        dropout=0.0,
    )
    qpos = torch.randn(2, 7, requires_grad=True)
    action_chunk = torch.randn(2, 8, 7, requires_grad=True)

    mu, logvar, z = encoder(qpos, action_chunk)
    loss = mu.pow(2).mean() + logvar.pow(2).mean() + z.pow(2).mean()
    loss.backward()

    assert qpos.grad is not None
    assert qpos.grad.abs().sum() > 0
    assert action_chunk.grad is not None
    assert action_chunk.grad.abs().sum() > 0
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in encoder.parameters()
    )


def test_contact_posterior_encoder_gradients_flow():
    encoder = ContactPosteriorEncoder(
        d_model=64,
        z_dim=8,
        nhead=4,
        num_layers=1,
        dim_feedforward=128,
        dropout=0.0,
    )
    qpos = torch.randn(2, 7, requires_grad=True)
    action_chunk = torch.randn(2, 8, 7, requires_grad=True)
    future_force_chunk = torch.randn(2, 8, 6, requires_grad=True)

    mu, logvar, z = encoder(qpos, action_chunk, future_force_chunk)
    loss = mu.pow(2).mean() + logvar.pow(2).mean() + z.pow(2).mean()
    loss.backward()

    assert qpos.grad is not None
    assert qpos.grad.abs().sum() > 0
    assert action_chunk.grad is not None
    assert action_chunk.grad.abs().sum() > 0
    assert future_force_chunk.grad is not None
    assert future_force_chunk.grad.abs().sum() > 0
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in encoder.parameters()
    )
