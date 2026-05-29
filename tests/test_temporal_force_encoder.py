import pytest
import torch

from force_aware_act.models import TemporalForceEncoder


def test_temporal_force_encoder_default_shape():
    encoder = TemporalForceEncoder(dropout=0.0)
    force_window = torch.randn(2, 50, 6)

    z_force = encoder(force_window)

    assert z_force.shape == (2, 512)


def test_temporal_force_encoder_projected_shape():
    encoder = TemporalForceEncoder(d_model=256, nhead=8, dropout=0.0)
    force_window = torch.randn(2, 50, 6)

    z_force = encoder(force_window)

    assert z_force.shape == (2, 256)


def test_temporal_force_encoder_rejects_long_window():
    encoder = TemporalForceEncoder(max_window_len=4)
    force_window = torch.randn(2, 5, 6)

    with pytest.raises(ValueError, match="exceeds max_window_len"):
        encoder(force_window)


def test_temporal_force_encoder_gradients_flow():
    encoder = TemporalForceEncoder(d_model=64, nhead=4, num_layers=1, dim_feedforward=128)
    force_window = torch.randn(2, 8, 6, requires_grad=True)

    output = encoder(force_window)
    loss = output.pow(2).mean()
    loss.backward()

    assert force_window.grad is not None
    assert force_window.grad.abs().sum() > 0
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in encoder.parameters()
    )
