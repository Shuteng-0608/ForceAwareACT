import pytest
import torch

pytest.importorskip("torchvision")

from force_aware_act.models import ForceAwareACTPolicy
from force_aware_act.training import compute_force_aware_act_loss, linear_warmup


def _make_outputs():
    return {
        "pred_action": torch.tensor([[[1.0, 2.0]]]),
        "pred_force": torch.tensor([[[3.0, 4.0, 5.0]]]),
        "mu_motion": torch.zeros(1, 2),
        "logvar_motion": torch.zeros(1, 2),
        "mu_contact": torch.ones(1, 2),
        "logvar_contact": torch.zeros(1, 2),
    }


def _add_prior_outputs(outputs):
    outputs.update(
        {
            "mu_contact_prior": torch.full((1, 2), 2.0),
            "logvar_contact_prior": torch.zeros(1, 2),
        }
    )
    return outputs


def test_force_aware_act_loss_terms_are_scalar_tensors():
    outputs = _make_outputs()
    action_chunk = torch.zeros(1, 1, 2)
    future_force_chunk = torch.zeros(1, 1, 3)

    losses = compute_force_aware_act_loss(outputs, action_chunk, future_force_chunk)

    for key in ("loss_total", "loss_action", "loss_force", "kl_motion", "kl_contact"):
        assert isinstance(losses[key], torch.Tensor)
        assert losses[key].shape == ()
    assert losses["loss_prior"].shape == ()
    assert losses["lambda_prior"] == 0.0
    assert losses["prior_loss_mode"] == "mse_mu"


def test_force_aware_act_loss_total_matches_weighted_sum():
    outputs = _make_outputs()
    action_chunk = torch.zeros(1, 1, 2)
    future_force_chunk = torch.zeros(1, 1, 3)

    losses = compute_force_aware_act_loss(
        outputs,
        action_chunk,
        future_force_chunk,
        lambda_force=0.5,
        beta_motion=0.25,
        beta_contact=0.125,
    )

    expected = (
        losses["loss_action"]
        + 0.5 * losses["loss_force"]
        + 0.25 * losses["kl_motion"]
        + 0.125 * losses["kl_contact"]
    )
    torch.testing.assert_close(losses["loss_total"], expected)


def test_force_aware_act_loss_zero_latent_mode_disables_kl_terms():
    outputs = {
        "pred_action": torch.tensor([[[1.0, 2.0]]]),
        "pred_force": torch.tensor([[[3.0, 4.0, 5.0]]]),
    }
    action_chunk = torch.zeros(1, 1, 2)
    future_force_chunk = torch.zeros(1, 1, 3)

    losses = compute_force_aware_act_loss(
        outputs,
        action_chunk,
        future_force_chunk,
        lambda_force=0.5,
        beta_motion=100.0,
        beta_contact=100.0,
        use_posterior_kl=False,
    )

    expected = losses["loss_action"] + 0.5 * losses["loss_force"]
    torch.testing.assert_close(losses["loss_total"], expected)
    torch.testing.assert_close(losses["kl_motion"], torch.zeros(()))
    torch.testing.assert_close(losses["kl_contact"], torch.zeros(()))
    assert losses["use_posterior_kl"] is False


def test_force_aware_act_loss_zero_latent_mode_rejects_prior_distillation():
    outputs = {
        "pred_action": torch.tensor([[[1.0, 2.0]]]),
        "pred_force": torch.tensor([[[3.0, 4.0, 5.0]]]),
    }
    action_chunk = torch.zeros(1, 1, 2)
    future_force_chunk = torch.zeros(1, 1, 3)

    with pytest.raises(ValueError, match="lambda_prior"):
        compute_force_aware_act_loss(
            outputs,
            action_chunk,
            future_force_chunk,
            lambda_prior=0.1,
            use_posterior_kl=False,
        )


def test_force_aware_act_loss_total_unchanged_when_lambda_prior_zero():
    outputs = _make_outputs()
    action_chunk = torch.zeros(1, 1, 2)
    future_force_chunk = torch.zeros(1, 1, 3)

    losses = compute_force_aware_act_loss(
        outputs,
        action_chunk,
        future_force_chunk,
        lambda_force=0.5,
        beta_motion=0.25,
        beta_contact=0.125,
        lambda_prior=0.0,
    )

    expected = (
        losses["loss_action"]
        + 0.5 * losses["loss_force"]
        + 0.25 * losses["kl_motion"]
        + 0.125 * losses["kl_contact"]
    )
    torch.testing.assert_close(losses["loss_total"], expected)
    torch.testing.assert_close(losses["loss_prior"], torch.zeros(()))


def test_force_aware_act_loss_total_includes_weighted_prior_loss():
    outputs = _add_prior_outputs(_make_outputs())
    action_chunk = torch.zeros(1, 1, 2)
    future_force_chunk = torch.zeros(1, 1, 3)

    losses = compute_force_aware_act_loss(
        outputs,
        action_chunk,
        future_force_chunk,
        lambda_force=0.5,
        beta_motion=0.25,
        beta_contact=0.125,
        lambda_prior=0.75,
        prior_loss_mode="mse_mu",
    )

    expected = (
        losses["loss_action"]
        + 0.5 * losses["loss_force"]
        + 0.25 * losses["kl_motion"]
        + 0.125 * losses["kl_contact"]
        + 0.75 * losses["loss_prior"]
    )
    torch.testing.assert_close(losses["loss_total"], expected)
    assert losses["lambda_prior"] == 0.75
    assert losses["prior_loss_mode"] == "mse_mu"


def test_linear_warmup_values():
    assert linear_warmup(step=0, warmup_steps=10, max_value=0.5) == 0.0
    assert linear_warmup(step=5, warmup_steps=10, max_value=0.5) == 0.25
    assert linear_warmup(step=10, warmup_steps=10, max_value=0.5) == 0.5
    assert linear_warmup(step=1, warmup_steps=0, max_value=0.5) == 0.5


def test_force_aware_act_loss_missing_keys_raise_clear_errors():
    action_chunk = torch.zeros(1, 1, 2)
    future_force_chunk = torch.zeros(1, 1, 3)

    outputs = _make_outputs()
    del outputs["pred_force"]
    with pytest.raises(KeyError, match="pred_force"):
        compute_force_aware_act_loss(outputs, action_chunk, future_force_chunk)

    outputs = _make_outputs()
    del outputs["mu_contact"]
    with pytest.raises(KeyError, match="mu_contact"):
        compute_force_aware_act_loss(outputs, action_chunk, future_force_chunk)


def test_force_aware_act_loss_missing_prior_outputs_raise_when_enabled():
    outputs = _make_outputs()
    action_chunk = torch.zeros(1, 1, 2)
    future_force_chunk = torch.zeros(1, 1, 3)

    with pytest.raises(KeyError, match="mu_contact_prior"):
        compute_force_aware_act_loss(
            outputs,
            action_chunk,
            future_force_chunk,
            lambda_prior=0.1,
        )


def test_force_aware_act_loss_one_batch_forward_backward():
    model = ForceAwareACTPolicy(
        d_model=128,
        z_dim=16,
        chunk_len=10,
        nhead=4,
        num_encoder_layers=1,
        num_decoder_layers=1,
        dim_feedforward=256,
        dropout=0.0,
        pretrained_resnet18=False,
    )
    images = torch.randn(2, 2, 3, 224, 224)
    qpos = torch.randn(2, 7)
    force_window = torch.randn(2, 20, 6)
    action_chunk = torch.randn(2, 10, 7)
    future_force_chunk = torch.randn(2, 10, 6)

    outputs = model(
        images=images,
        qpos=qpos,
        force_window=force_window,
        action_chunk=action_chunk,
        future_force_chunk=future_force_chunk,
        is_training=True,
    )
    losses = compute_force_aware_act_loss(
        outputs,
        action_chunk,
        future_force_chunk,
        lambda_force=0.1,
        beta_motion=1.0e-4,
        beta_contact=1.0e-4,
        lambda_prior=0.1,
        prior_loss_mode="mse_mu",
    )
    losses["loss_total"].backward()

    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in model.parameters()
    )
