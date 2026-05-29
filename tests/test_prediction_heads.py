import pytest
import torch

from force_aware_act.models import ActionHead, ForceHead


def test_action_head_default_shape():
    head = ActionHead(d_model=512, action_dim=7)
    decoder_hidden = torch.randn(2, 50, 512)

    pred_action = head(decoder_hidden)

    assert pred_action.shape == (2, 50, 7)


def test_force_head_default_shape():
    head = ForceHead(d_model=512, z_dim=32, force_dim=6)
    decoder_hidden = torch.randn(2, 50, 512)
    z_contact = torch.randn(2, 32)

    pred_force = head(decoder_hidden, z_contact)

    assert pred_force.shape == (2, 50, 6)


def test_prediction_heads_smaller_dimensions_work():
    action_head = ActionHead(d_model=128, action_dim=7, hidden_dim=64)
    force_head = ForceHead(d_model=128, z_dim=16, force_dim=6, hidden_dim=64)
    decoder_hidden = torch.randn(2, 50, 128)
    z_contact = torch.randn(2, 16)

    pred_action = action_head(decoder_hidden)
    pred_force = force_head(decoder_hidden, z_contact)

    assert pred_action.shape == (2, 50, 7)
    assert pred_force.shape == (2, 50, 6)


def test_force_head_output_changes_when_contact_latent_changes():
    head = ForceHead(d_model=32, z_dim=8, force_dim=6)
    decoder_hidden = torch.randn(2, 5, 32)
    z_contact = torch.randn(2, 8)
    z_contact_changed = z_contact + 1.0

    pred_force = head(decoder_hidden, z_contact)
    pred_force_changed = head(decoder_hidden, z_contact_changed)

    assert not torch.allclose(pred_force, pred_force_changed)


def test_force_head_rejects_batch_mismatch():
    head = ForceHead(d_model=512, z_dim=32)
    decoder_hidden = torch.randn(2, 50, 512)
    z_contact = torch.randn(3, 32)

    with pytest.raises(ValueError, match="batch size"):
        head(decoder_hidden, z_contact)


def test_action_head_rejects_decoder_hidden_feature_mismatch():
    head = ActionHead(d_model=512)
    decoder_hidden = torch.randn(2, 50, 256)

    with pytest.raises(ValueError, match="decoder_hidden last dimension"):
        head(decoder_hidden)


def test_force_head_gradients_flow():
    head = ForceHead(d_model=64, z_dim=8, force_dim=6, hidden_dim=32)
    decoder_hidden = torch.randn(2, 10, 64, requires_grad=True)
    z_contact = torch.randn(2, 8, requires_grad=True)

    loss = head(decoder_hidden, z_contact).pow(2).mean()
    loss.backward()

    assert decoder_hidden.grad is not None
    assert decoder_hidden.grad.abs().sum() > 0
    assert z_contact.grad is not None
    assert z_contact.grad.abs().sum() > 0
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in head.parameters()
    )
