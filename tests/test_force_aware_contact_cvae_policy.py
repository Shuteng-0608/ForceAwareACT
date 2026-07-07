import argparse
import importlib.util
from pathlib import Path

import pytest
import torch

pytest.importorskip("torchvision")

from force_aware_act.models import ForceAwareACTContactCVAEPolicy
from force_aware_act.training import compute_force_aware_contact_cvae_loss
from scripts import evaluate_contact_cvae_modes as contact_eval
from scripts import train_minimal
from scripts.run_mujoco_policy_rollout import _build_policy_from_checkpoint, _policy_has_contact_prior


def _make_policy(chunk_len=3):
    return ForceAwareACTContactCVAEPolicy(
        d_model=32,
        z_dim=8,
        action_dim=7,
        force_dim=6,
        chunk_len=chunk_len,
        nhead=4,
        num_encoder_layers=1,
        num_decoder_layers=1,
        dim_feedforward=64,
        dropout=0.0,
        pretrained_resnet18=False,
        max_force_window_len=8,
    )


def _make_inputs(batch_size=2, num_cameras=2, chunk_len=3):
    return {
        "images": torch.randn(batch_size, num_cameras, 3, 64, 64),
        "qpos": torch.randn(batch_size, 7),
        "force_window": torch.randn(batch_size, 5, 6),
        "action_chunk": torch.randn(batch_size, chunk_len, 7),
        "future_force_chunk": torch.randn(batch_size, chunk_len, 6),
    }


def _training_outputs(model, inputs):
    return model(
        images=inputs["images"],
        qpos=inputs["qpos"],
        force_window=inputs["force_window"],
        action_chunk=inputs["action_chunk"],
        future_force_chunk=inputs["future_force_chunk"],
        is_training=True,
    )


def _assert_no_motion(model, outputs=None):
    assert not hasattr(model, "motion_posterior")
    assert not hasattr(model, "motion_latent_proj")
    assert not any("motion" in name for name, _parameter in model.named_parameters())
    if outputs is not None:
        assert not any("motion" in key for key in outputs)


def test_contact_cvae_training_forward_shapes_and_no_motion_outputs():
    model = _make_policy()
    model.eval()
    inputs = _make_inputs()

    with torch.no_grad():
        outputs = _training_outputs(model, inputs)

    assert outputs["pred_action"].shape == (2, 3, 7)
    assert outputs["pred_force"].shape == (2, 3, 6)
    assert outputs["z_contact"].shape == (2, 8)
    assert outputs["mu_contact"].shape == (2, 8)
    assert outputs["logvar_contact"].shape == (2, 8)
    assert outputs["mu_contact_prior"].shape == (2, 8)
    assert outputs["logvar_contact_prior"].shape == (2, 8)
    assert outputs["z_contact_prior"].shape == (2, 8)
    assert model.policy_token_names == (
        "visual_tokens",
        "z_VF",
        "z_q",
        "z_F_online",
        "z_contact",
    )
    _assert_no_motion(model, outputs)


def test_contact_cvae_zero_and_prior_deployment_modes():
    model = _make_policy()
    model.eval()
    inputs = _make_inputs()

    with torch.no_grad():
        zero_outputs = model(
            images=inputs["images"],
            qpos=inputs["qpos"],
            force_window=inputs["force_window"],
            action_chunk=None,
            future_force_chunk=None,
            is_training=False,
            contact_latent_mode="zero",
        )
        prior_outputs = model(
            images=inputs["images"],
            qpos=inputs["qpos"],
            force_window=inputs["force_window"],
            action_chunk=None,
            future_force_chunk=None,
            is_training=False,
            contact_latent_mode="prior",
        )
        sampled_prior_outputs = model(
            images=inputs["images"],
            qpos=inputs["qpos"],
            force_window=inputs["force_window"],
            action_chunk=None,
            future_force_chunk=None,
            is_training=False,
            contact_latent_mode="prior",
            deterministic_prior=False,
        )

    assert zero_outputs["pred_action"].shape == (2, 3, 7)
    assert zero_outputs["pred_force"].shape == (2, 3, 6)
    assert torch.count_nonzero(zero_outputs["z_contact"]) == 0
    assert "mu_contact_prior" not in zero_outputs
    torch.testing.assert_close(prior_outputs["z_contact"], prior_outputs["mu_contact_prior"])
    assert prior_outputs["z_contact_prior"].shape == (2, 8)
    assert sampled_prior_outputs["z_contact"].shape == (2, 8)
    assert sampled_prior_outputs["z_contact"] is sampled_prior_outputs["z_contact_prior"]
    _assert_no_motion(model, prior_outputs)


def test_contact_cvae_posterior_oracle_uses_encode_and_override():
    model = _make_policy()
    model.eval()
    inputs = _make_inputs()

    with torch.no_grad():
        mu_contact, logvar_contact, z_contact_sample = model.encode_contact_posterior(
            inputs["qpos"],
            inputs["action_chunk"],
            inputs["future_force_chunk"],
        )
        outputs = model(
            images=inputs["images"],
            qpos=inputs["qpos"],
            force_window=inputs["force_window"],
            action_chunk=None,
            future_force_chunk=None,
            is_training=False,
            contact_latent_override=mu_contact,
        )

    assert mu_contact.shape == (2, 8)
    assert logvar_contact.shape == (2, 8)
    assert z_contact_sample.shape == (2, 8)
    assert outputs["z_contact"] is mu_contact
    assert outputs["pred_action"].shape == (2, 3, 7)


def test_contact_cvae_rejects_deployment_posterior_mode_and_future_labels():
    model = _make_policy()
    inputs = _make_inputs()

    with pytest.raises(ValueError, match="oracle-only"):
        model(
            images=inputs["images"],
            qpos=inputs["qpos"],
            force_window=inputs["force_window"],
            action_chunk=None,
            future_force_chunk=None,
            is_training=False,
            contact_latent_mode="posterior",
        )
    with pytest.raises(ValueError, match="action_chunk must be None"):
        model(
            images=inputs["images"],
            qpos=inputs["qpos"],
            force_window=inputs["force_window"],
            action_chunk=inputs["action_chunk"],
            future_force_chunk=None,
            is_training=False,
            contact_latent_mode="zero",
        )


def test_contact_cvae_loss_terms_and_no_motion_terms():
    outputs = {
        "pred_action": torch.tensor([[[1.0, 2.0]]]),
        "pred_force": torch.tensor([[[3.0, 4.0, 5.0]]]),
        "mu_contact": torch.ones(1, 2),
        "logvar_contact": torch.zeros(1, 2),
        "mu_contact_prior": torch.zeros(1, 2, requires_grad=True),
        "logvar_contact_prior": torch.zeros(1, 2, requires_grad=True),
    }
    action_chunk = torch.zeros(1, 1, 2)
    future_force_chunk = torch.zeros(1, 1, 3)

    losses = compute_force_aware_contact_cvae_loss(
        outputs,
        action_chunk,
        future_force_chunk,
        lambda_force=0.5,
        beta_contact=0.25,
        lambda_prior=0.75,
    )

    expected = (
        losses["loss_action"]
        + 0.5 * losses["loss_force"]
        + 0.25 * losses["kl_contact"]
        + 0.75 * losses["loss_prior"]
    )
    torch.testing.assert_close(losses["loss_total"], expected)
    assert "kl_motion" not in losses
    assert "beta_motion" not in losses


def test_contact_cvae_gradients_follow_approved_paths():
    model = _make_policy()
    inputs = _make_inputs(batch_size=1)
    outputs = _training_outputs(model, inputs)

    loss_action = torch.nn.functional.l1_loss(outputs["pred_action"], inputs["action_chunk"])
    model.zero_grad(set_to_none=True)
    loss_action.backward(retain_graph=True)
    assert model.contact_latent_proj.weight.grad is not None
    assert model.contact_latent_proj.weight.grad.abs().sum() > 0

    model.zero_grad(set_to_none=True)
    loss_force = torch.nn.functional.l1_loss(outputs["pred_force"], inputs["future_force_chunk"])
    loss_force.backward(retain_graph=True)
    assert model.contact_latent_proj.weight.grad is not None
    assert model.contact_latent_proj.weight.grad.abs().sum() > 0

    model.zero_grad(set_to_none=True)
    losses = compute_force_aware_contact_cvae_loss(
        outputs,
        inputs["action_chunk"],
        inputs["future_force_chunk"],
        lambda_prior=0.0,
    )
    losses["loss_total"].backward(retain_graph=True)
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in model.contact_posterior.parameters()
    )
    assert all(parameter.grad is None for parameter in model.contact_prior.parameters())

    model.zero_grad(set_to_none=True)
    losses = compute_force_aware_contact_cvae_loss(
        outputs,
        inputs["action_chunk"],
        inputs["future_force_chunk"],
        lambda_prior=0.5,
    )
    losses["loss_total"].backward()
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in model.contact_prior.parameters()
    )


def test_contact_cvae_strict_state_dict_roundtrip():
    model = _make_policy()
    loaded = _make_policy()

    loaded.load_state_dict(model.state_dict(), strict=True)


def test_contact_cvae_training_config_and_simplenamespace_default():
    args = argparse.Namespace(
        episode_paths=[Path("/tmp/episode.hdf5")],
        action_mode="action",
        policy_variant="force_aware_contact_cvae",
        train_latent_mode="posterior",
        chunk_len=3,
        force_window_len=5,
        force_window_duration=0.25,
        image_size=(64, 64),
        camera_names=("ee_cam", "base_top_cam"),
        imagenet_normalize=False,
        batch_size=1,
        num_workers=0,
        max_steps=1,
        learning_rate=1.0e-4,
        lambda_force=0.1,
        lambda_prior=0.2,
        prior_loss_mode="mse_mu",
        beta_motion_max=123.0,
        beta_contact_max=1.0e-4,
        warmup_steps=1,
        output_dir=Path("/tmp/out"),
        log_csv=Path("/tmp/out/log.csv"),
        device="cpu",
        normalization_stats=None,
    )

    config = train_minimal._config_from_args(args)

    assert config["policy_variant"] == "force_aware_contact_cvae"
    assert config["uses_motion_latent"] is False
    assert config["uses_contact_latent"] is True
    assert config["train_contact_latent_mode"] == "posterior"
    assert config["deployment_contact_latent_modes"] == ["zero", "prior"]


def test_contact_cvae_rollout_dispatch_and_contact_prior_capability():
    checkpoint = {
        "config": {
            "policy_variant": "force_aware_contact_cvae",
            "model": {
                "pretrained_resnet18": False,
                "d_model": 32,
                "z_dim": 8,
                "action_dim": 7,
                "force_dim": 6,
                "chunk_len": 3,
                "nhead": 4,
                "num_encoder_layers": 1,
                "num_decoder_layers": 1,
                "dim_feedforward": 64,
                "dropout": 0.0,
                "max_force_window_len": 8,
            },
        }
    }

    model = _build_policy_from_checkpoint(checkpoint, force_window_len=5, chunk_len=3)

    assert isinstance(model, ForceAwareACTContactCVAEPolicy)
    assert _policy_has_contact_prior("force_aware_act")
    assert _policy_has_contact_prior("force_aware_contact_cvae")
    assert not _policy_has_contact_prior("force_aware_motion_cvae")
    assert not _policy_has_contact_prior("act_baseline")


def test_contact_cvae_evaluator_metrics_and_episode_identifier():
    outputs = contact_eval.ContactModeOutputs(
        zero={
            "pred_action": torch.tensor([[[1.0, 3.0]]]),
            "pred_force": torch.tensor([[[1.0, 2.0, 3.0]]]),
        },
        prior={
            "pred_action": torch.tensor([[[2.0, 4.0]]]),
            "pred_force": torch.tensor([[[2.0, 3.0, 4.0]]]),
            "mu_contact_prior": torch.tensor([[1.0, 0.0]]),
        },
        posterior={
            "pred_action": torch.tensor([[[0.0, 2.0]]]),
            "pred_force": torch.tensor([[[0.0, 1.0, 2.0]]]),
        },
        mu_contact=torch.tensor([[0.0, 1.0]]),
        logvar_contact=torch.zeros(1, 2),
        z_contact_posterior=torch.tensor([[0.0, 1.0]]),
    )

    metrics = contact_eval.compute_sample_metrics(
        outputs,
        action_target=torch.zeros(1, 1, 2),
        force_target=torch.zeros(1, 1, 3),
    )

    torch.testing.assert_close(metrics["action_l1_zero"], torch.tensor([2.0]))
    torch.testing.assert_close(metrics["action_l1_prior"], torch.tensor([3.0]))
    torch.testing.assert_close(metrics["action_l1_posterior"], torch.tensor([1.0]))
    torch.testing.assert_close(
        metrics["pred_action_prior_posterior_mean_abs_diff"],
        torch.tensor([2.0]),
    )
    assert torch.isfinite(metrics["mu_prior_to_mu_posterior_cosine"]).all()
    assert (
        contact_eval.episode_identifier("/data/run_001/episode.hdf5")
        == "run_001"
    )


def test_contact_cvae_parameter_audit_invariants():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "audit_model_components.py"
    spec = importlib.util.spec_from_file_location("audit_model_components_contact", script_path)
    audit_model_components = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(audit_model_components)
    config = dict(audit_model_components.DEFAULT_SYNTHETIC_CONFIG)
    config.update(
        {
            "d_model": 32,
            "z_dim": 8,
            "chunk_len": 3,
            "nhead": 4,
            "num_encoder_layers": 1,
            "num_decoder_layers": 1,
            "dim_feedforward": 64,
            "max_force_window_len": 8,
        }
    )

    audit = audit_model_components.create_audit(
        config=config,
        device="cpu",
        policy_variant="force_aware_contact_cvae",
    )

    components = audit["components"]
    assert components["motion_latent_modules"]["total_parameters"] == 0
    assert components["contact_latent_prior_posterior"]["total_parameters"] > 0
    assert components["force_temporal_encoder"]["total_parameters"] > 0
    assert components["force_vision_fusion"]["total_parameters"] > 0
    assert components["force_head"]["total_parameters"] > 0
    assert components["other_unclassified"]["total_parameters"] == 0


def test_contact_cvae_exports_are_available():
    from force_aware_act.models import ForceAwareACTContactCVAEPolicy as ExportedModel
    from force_aware_act.training import (
        compute_force_aware_contact_cvae_loss as exported_loss,
    )

    assert ExportedModel is ForceAwareACTContactCVAEPolicy
    assert exported_loss is compute_force_aware_contact_cvae_loss
