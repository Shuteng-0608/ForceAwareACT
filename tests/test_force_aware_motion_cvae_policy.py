import torch

import pytest

pytest.importorskip("torchvision")

from force_aware_act.models import ForceAwareACTMotionCVAEPolicy
from force_aware_act.training import compute_force_aware_motion_cvae_loss


def _make_policy(chunk_len=4):
    return ForceAwareACTMotionCVAEPolicy(
        d_model=128,
        z_dim=16,
        action_dim=7,
        force_dim=6,
        chunk_len=chunk_len,
        nhead=4,
        num_encoder_layers=1,
        num_decoder_layers=1,
        dim_feedforward=256,
        dropout=0.0,
        pretrained_resnet18=False,
    )


def _make_inputs(batch_size=2, num_cameras=2, chunk_len=4):
    return {
        "images": torch.randn(batch_size, num_cameras, 3, 224, 224),
        "qpos": torch.randn(batch_size, 7),
        "force_window": torch.randn(batch_size, 20, 6),
        "action_chunk": torch.randn(batch_size, chunk_len, 7),
        "future_force_chunk": torch.randn(batch_size, chunk_len, 6),
    }


def test_motion_cvae_training_forward_shapes_and_outputs():
    model = _make_policy(chunk_len=4)
    model.eval()
    inputs = _make_inputs(chunk_len=4)

    with torch.no_grad():
        outputs = model(
            images=inputs["images"],
            qpos=inputs["qpos"],
            force_window=inputs["force_window"],
            action_chunk=inputs["action_chunk"],
            future_force_chunk=inputs["future_force_chunk"],
            is_training=True,
        )

    assert outputs["pred_action"].shape == (2, 4, 7)
    assert outputs["pred_force"].shape == (2, 4, 6)
    assert outputs["visual_tokens"].shape == (2, 98, 128)
    assert outputs["z_q"].shape == (2, 128)
    assert outputs["z_F_online"].shape == (2, 128)
    assert outputs["z_VF"].shape == (2, 128)
    assert outputs["z_motion"].shape == (2, 16)
    assert outputs["decoder_hidden"].shape == (2, 4, 128)
    assert outputs["mu_motion"].shape == (2, 16)
    assert outputs["logvar_motion"].shape == (2, 16)
    assert model.policy_token_names == (
        "visual_tokens",
        "z_VF",
        "z_q",
        "z_F_online",
        "z_motion",
    )
    assert not any("contact" in key or "prior" in key for key in outputs)


def test_motion_cvae_inference_uses_exact_zero_motion_and_no_posteriors():
    model = _make_policy(chunk_len=4)
    model.eval()
    inputs = _make_inputs(chunk_len=4)

    with torch.no_grad():
        outputs = model(
            images=inputs["images"],
            qpos=inputs["qpos"],
            force_window=inputs["force_window"],
            action_chunk=None,
            future_force_chunk=None,
            is_training=False,
        )

    assert outputs["pred_action"].shape == (2, 4, 7)
    assert outputs["pred_force"].shape == (2, 4, 6)
    assert "mu_motion" not in outputs
    assert "logvar_motion" not in outputs
    assert torch.count_nonzero(outputs["z_motion"]) == 0
    assert not any("contact" in key or "prior" in key for key in outputs)


def test_motion_cvae_has_no_contact_or_prior_modules_or_parameters():
    model = _make_policy(chunk_len=4)

    assert not hasattr(model, "contact_posterior")
    assert not hasattr(model, "contact_prior")
    assert not hasattr(model, "contact_latent_proj")
    assert not any("contact" in name or "prior" in name for name, _ in model.named_modules())
    assert not any("contact" in name or "prior" in name for name, _ in model.named_parameters())


def test_motion_cvae_inference_rejects_future_labels():
    model = _make_policy(chunk_len=4)
    inputs = _make_inputs(chunk_len=4)

    with pytest.raises(ValueError, match="action_chunk must be None"):
        model(
            images=inputs["images"],
            qpos=inputs["qpos"],
            force_window=inputs["force_window"],
            action_chunk=inputs["action_chunk"],
            future_force_chunk=None,
            is_training=False,
        )

    with pytest.raises(ValueError, match="future_force_chunk must be None"):
        model(
            images=inputs["images"],
            qpos=inputs["qpos"],
            force_window=inputs["force_window"],
            action_chunk=None,
            future_force_chunk=inputs["future_force_chunk"],
            is_training=False,
        )


def test_motion_cvae_training_requires_supervised_chunks():
    model = _make_policy(chunk_len=4)
    inputs = _make_inputs(chunk_len=4)

    with pytest.raises(ValueError, match="action_chunk is required"):
        model(
            images=inputs["images"],
            qpos=inputs["qpos"],
            force_window=inputs["force_window"],
            action_chunk=None,
            future_force_chunk=inputs["future_force_chunk"],
            is_training=True,
        )

    with pytest.raises(ValueError, match="future_force_chunk is required"):
        model(
            images=inputs["images"],
            qpos=inputs["qpos"],
            force_window=inputs["force_window"],
            action_chunk=inputs["action_chunk"],
            future_force_chunk=None,
            is_training=True,
        )


def test_motion_cvae_force_head_gets_fixed_zero_auxiliary_vector(monkeypatch):
    model = _make_policy(chunk_len=4)
    model.eval()
    inputs = _make_inputs(chunk_len=4)
    captured = {}
    original_forward = model.force_head.forward

    def capture_forward(decoder_hidden, auxiliary):
        captured["auxiliary"] = auxiliary
        return original_forward(decoder_hidden, auxiliary)

    monkeypatch.setattr(model.force_head, "forward", capture_forward)

    with torch.no_grad():
        model(
            images=inputs["images"],
            qpos=inputs["qpos"],
            force_window=inputs["force_window"],
            action_chunk=inputs["action_chunk"],
            future_force_chunk=inputs["future_force_chunk"],
            is_training=True,
        )

    assert captured["auxiliary"].shape == (2, 16)
    assert torch.count_nonzero(captured["auxiliary"]) == 0


def test_motion_cvae_loss_has_no_contact_or_prior_terms():
    outputs = {
        "pred_action": torch.tensor([[[1.0, 2.0]]]),
        "pred_force": torch.tensor([[[3.0, 4.0, 5.0]]]),
        "mu_motion": torch.zeros(1, 2),
        "logvar_motion": torch.zeros(1, 2),
    }
    action_chunk = torch.zeros(1, 1, 2)
    future_force_chunk = torch.zeros(1, 1, 3)

    losses = compute_force_aware_motion_cvae_loss(
        outputs,
        action_chunk,
        future_force_chunk,
        lambda_force=0.5,
        beta_motion=0.25,
    )

    expected = losses["loss_action"] + 0.5 * losses["loss_force"] + 0.25 * losses["kl_motion"]
    torch.testing.assert_close(losses["loss_total"], expected)
    assert "kl_contact" not in losses
    assert "loss_prior" not in losses
    assert losses["policy_variant"] == "force_aware_motion_cvae"
