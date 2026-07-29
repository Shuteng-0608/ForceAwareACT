import pytest
import torch

from force_aware_act.models.act_aligned.token_adapters import (
    ActionTokenAdapter,
    ForceTokenAdapter,
    LatentTokenAdapter,
    QposTokenAdapter,
)


def test_modality_adapters_produce_batch_first_tokens():
    qpos_adapter = QposTokenAdapter(q_dim=7, d_model=32)
    action_adapter = ActionTokenAdapter(action_dim=7, d_model=32)
    force_adapter = ForceTokenAdapter(force_dim=6, d_model=32)
    latent_adapter = LatentTokenAdapter(latent_dim=8, d_model=32)

    assert qpos_adapter(torch.randn(2, 7)).shape == (2, 1, 32)
    assert action_adapter(torch.randn(2, 100, 7)).shape == (2, 100, 32)
    assert force_adapter(torch.randn(2, 20, 6)).shape == (2, 20, 32)
    assert latent_adapter(torch.randn(2, 8)).shape == (2, 1, 32)


def test_sequence_adapter_preserves_batch_and_sequence_axes():
    adapter = ActionTokenAdapter(action_dim=7, d_model=16)
    values = torch.randn(3, 11, 7, requires_grad=True)

    tokens = adapter(values)
    tokens.square().mean().backward()

    assert tokens.shape == (3, 11, 16)
    assert values.grad is not None
    assert values.grad.abs().sum() > 0
    assert adapter.projection.weight.grad is not None


@pytest.mark.parametrize(
    ("adapter", "bad_input", "message"),
    [
        (QposTokenAdapter(7, 16), torch.randn(2, 1, 7), "qpos must have shape"),
        (ActionTokenAdapter(7, 16), torch.randn(2, 7), "action must have shape"),
        (ForceTokenAdapter(6, 16), torch.randn(2, 4, 7), "force must have shape"),
        (LatentTokenAdapter(8, 16), torch.ones(2, 8, dtype=torch.long), "floating point"),
    ],
)
def test_modality_adapters_reject_invalid_layout_or_dtype(adapter, bad_input, message):
    with pytest.raises(ValueError, match=message):
        adapter(bad_input)
