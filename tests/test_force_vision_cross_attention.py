import pytest
import torch

from force_aware_act.models import ForceVisionCrossAttention


def test_force_vision_cross_attention_default_shape():
    module = ForceVisionCrossAttention(d_model=512, nhead=8, dropout=0.0)
    z_force = torch.randn(2, 512)
    visual_tokens = torch.randn(2, 98, 512)

    z_vf = module(z_force, visual_tokens)

    assert z_vf.shape == (2, 512)


def test_force_vision_cross_attention_returns_attention_weights():
    module = ForceVisionCrossAttention(
        d_model=512,
        nhead=8,
        dropout=0.0,
        return_attn=True,
    )
    z_force = torch.randn(2, 512)
    visual_tokens = torch.randn(2, 98, 512)

    z_vf, attn_weights = module(z_force, visual_tokens)

    assert z_vf.shape == (2, 512)
    assert attn_weights.shape == (2, 1, 98)


def test_force_vision_cross_attention_projected_dim():
    module = ForceVisionCrossAttention(d_model=256, nhead=8, dropout=0.0)
    z_force = torch.randn(2, 256)
    visual_tokens = torch.randn(2, 98, 256)

    z_vf = module(z_force, visual_tokens)

    assert z_vf.shape == (2, 256)


def test_force_vision_cross_attention_rejects_batch_mismatch():
    module = ForceVisionCrossAttention(d_model=512)
    z_force = torch.randn(2, 512)
    visual_tokens = torch.randn(3, 98, 512)

    with pytest.raises(ValueError, match="same batch size"):
        module(z_force, visual_tokens)


def test_force_vision_cross_attention_rejects_feature_mismatch():
    module = ForceVisionCrossAttention(d_model=512)
    z_force = torch.randn(2, 512)
    visual_tokens = torch.randn(2, 98, 256)

    with pytest.raises(ValueError, match="visual_tokens feature dimension"):
        module(z_force, visual_tokens)


def test_force_vision_cross_attention_gradients_flow():
    module = ForceVisionCrossAttention(d_model=64, nhead=4, dropout=0.0)
    z_force = torch.randn(2, 64, requires_grad=True)
    visual_tokens = torch.randn(2, 16, 64, requires_grad=True)

    loss = module(z_force, visual_tokens).sum()
    loss.backward()

    assert z_force.grad is not None
    assert torch.any(z_force.grad != 0)
    assert visual_tokens.grad is not None
    assert torch.any(visual_tokens.grad != 0)
