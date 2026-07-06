from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
import torch.nn.functional as functional

pytest.importorskip("torchvision")

from force_aware_act.data import ContactForceHDF5Dataset
from force_aware_act.models import ACTPolicyBaseline
from force_aware_act.training import compute_act_baseline_loss, linear_warmup
from scripts.audit_model_components import (
    DEFAULT_ACT_SYNTHETIC_CONFIG,
    create_audit,
)
from scripts.run_mujoco_policy_rollout import _run_mode


def _make_policy(chunk_len=4):
    return ACTPolicyBaseline(
        d_model=32,
        z_dim=8,
        q_dim=7,
        action_dim=7,
        chunk_len=chunk_len,
        nhead=4,
        num_encoder_layers=1,
        num_decoder_layers=1,
        dim_feedforward=64,
        dropout=0.0,
        pretrained_resnet18=False,
    )


def _make_inputs(batch_size=2, num_cameras=2, chunk_len=4):
    return {
        "images": torch.randn(batch_size, num_cameras, 3, 64, 64),
        "qpos": torch.randn(batch_size, 7),
        "action_chunk": torch.randn(batch_size, chunk_len, 7),
    }


def test_act_policy_baseline_forward_accepts_no_force_inputs():
    model = _make_policy(chunk_len=4)
    model.eval()
    inputs = _make_inputs(chunk_len=4)

    with torch.no_grad():
        outputs = model(images=inputs["images"], qpos=inputs["qpos"], is_training=False)

    assert outputs["pred_action"].shape == (2, 4, 7)
    assert outputs["visual_tokens"].shape[0] == 2
    assert outputs["z_q"].shape == (2, 32)
    assert outputs["z_motion"].shape == (2, 8)
    assert torch.count_nonzero(outputs["z_motion"]) == 0
    assert "pred_force" not in outputs


def test_act_policy_baseline_instantiates_motion_posterior():
    model = _make_policy(chunk_len=4)

    assert hasattr(model, "motion_posterior")
    assert any(name.startswith("motion_posterior.") for name, _ in model.named_parameters())


def test_act_policy_baseline_has_no_force_or_contact_modules_or_state_keys():
    model = _make_policy(chunk_len=4)
    forbidden = ("force", "contact")

    module_names = [name for name, _module in model.named_modules()]
    state_keys = list(model.state_dict())
    parameter_names = [name for name, _parameter in model.named_parameters()]

    assert all(not any(token in name for token in forbidden) for name in module_names)
    assert all(not any(token in name for token in forbidden) for name in state_keys)
    assert all(not any(token in name for token in forbidden) for name in parameter_names)


def test_training_requires_action_chunk_and_returns_posterior_stats():
    model = _make_policy(chunk_len=4)
    inputs = _make_inputs(chunk_len=4)

    with pytest.raises(ValueError, match="action_chunk is required"):
        model(images=inputs["images"], qpos=inputs["qpos"], is_training=True)

    outputs = model(
        images=inputs["images"],
        qpos=inputs["qpos"],
        action_chunk=inputs["action_chunk"],
        is_training=True,
    )

    assert outputs["mu_motion"].shape == (2, 8)
    assert outputs["logvar_motion"].shape == (2, 8)
    assert outputs["z_motion"].shape == (2, 8)
    assert torch.isfinite(outputs["mu_motion"]).all()
    assert torch.isfinite(outputs["logvar_motion"]).all()
    assert not torch.equal(outputs["z_motion"], torch.zeros_like(outputs["z_motion"]))


def test_action_cvae_backprop_reaches_act_baseline_components():
    model = _make_policy(chunk_len=4)
    inputs = _make_inputs(chunk_len=4)

    outputs = model(
        images=inputs["images"],
        qpos=inputs["qpos"],
        action_chunk=inputs["action_chunk"],
        is_training=True,
    )
    losses = compute_act_baseline_loss(outputs, inputs["action_chunk"], beta_motion=0.1)
    loss = losses["loss_total"]
    loss.backward()

    prefixes = {
        "motion_posterior": False,
        "motion_latent_proj": False,
        "vision_encoder.backbone": False,
        "joint_encoder": False,
        "policy_encoder": False,
        "policy_decoder": False,
        "action_head": False,
    }
    for name, parameter in model.named_parameters():
        if parameter.grad is None or parameter.grad.abs().sum() == 0:
            continue
        for prefix in prefixes:
            if name.startswith(prefix):
                prefixes[prefix] = True

    assert all(prefixes.values())


def test_deployment_zero_latent_and_posterior_mean_override():
    model = _make_policy(chunk_len=4)
    model.eval()
    inputs = _make_inputs(chunk_len=4)

    with torch.no_grad():
        zero_outputs = model(images=inputs["images"], qpos=inputs["qpos"], is_training=False)
        mu_motion, _logvar_motion, _z = model.encode_motion_posterior(
            inputs["qpos"],
            inputs["action_chunk"],
        )
        posterior_outputs = model(
            images=inputs["images"],
            qpos=inputs["qpos"],
            action_chunk=None,
            is_training=False,
            motion_latent_override=mu_motion,
        )

    torch.testing.assert_close(zero_outputs["z_motion"], torch.zeros_like(zero_outputs["z_motion"]))
    torch.testing.assert_close(posterior_outputs["z_motion"], mu_motion)
    assert not torch.allclose(zero_outputs["pred_action"], posterior_outputs["pred_action"])


def test_deployment_rejects_action_labels_and_bad_override_shape():
    model = _make_policy(chunk_len=4)
    inputs = _make_inputs(chunk_len=4)

    with pytest.raises(ValueError, match="action_chunk must be None"):
        model(
            images=inputs["images"],
            qpos=inputs["qpos"],
            action_chunk=inputs["action_chunk"],
            is_training=False,
        )
    with pytest.raises(ValueError, match="motion_latent_override"):
        model(
            images=inputs["images"],
            qpos=inputs["qpos"],
            is_training=False,
            motion_latent_override=torch.zeros(2, 7),
        )


def test_act_baseline_loss_total_and_warmup_values():
    outputs = {
        "pred_action": torch.tensor([[[1.0, 2.0]]]),
        "mu_motion": torch.tensor([[1.0, 0.0]]),
        "logvar_motion": torch.zeros(1, 2),
    }
    action_chunk = torch.zeros(1, 1, 2)
    losses = compute_act_baseline_loss(outputs, action_chunk, beta_motion=0.25)

    expected = losses["loss_action"] + 0.25 * losses["kl_motion"]
    torch.testing.assert_close(losses["loss_total"], expected)
    assert losses["kl_motion"].item() >= 0.0
    assert linear_warmup(step=0, warmup_steps=10, max_value=0.5) == 0.0
    assert linear_warmup(step=5, warmup_steps=10, max_value=0.5) == 0.25
    assert linear_warmup(step=10, warmup_steps=10, max_value=0.5) == 0.5
    assert linear_warmup(step=11, warmup_steps=10, max_value=0.5) == 0.5


def test_act_checkpoint_save_load_reproduces_eval_outputs(tmp_path):
    config = {
        "d_model": 32,
        "z_dim": 8,
        "q_dim": 7,
        "action_dim": 7,
        "chunk_len": 4,
        "nhead": 4,
        "num_encoder_layers": 1,
        "num_decoder_layers": 1,
        "dim_feedforward": 64,
        "dropout": 0.0,
        "pretrained_resnet18": False,
        "freeze_resnet18": False,
    }
    model = ACTPolicyBaseline(**config)
    model.eval()
    inputs = _make_inputs(chunk_len=4)
    with torch.no_grad():
        expected = model(
            images=inputs["images"],
            qpos=inputs["qpos"],
            is_training=False,
        )["pred_action"]

    checkpoint_path = tmp_path / "act_checkpoint.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {
                "policy_variant": "act_baseline",
                "act_baseline_version": ACTPolicyBaseline.act_baseline_version,
                "uses_force": False,
                "uses_contact_latent": False,
                "motion_latent_mode": "posterior_train_zero_deploy",
                "model": config,
            },
            "step": 1,
        },
        checkpoint_path,
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    loaded = ACTPolicyBaseline(**checkpoint["config"]["model"])
    loaded.load_state_dict(checkpoint["model_state_dict"])
    loaded.eval()

    with torch.no_grad():
        actual = loaded(images=inputs["images"], qpos=inputs["qpos"], is_training=False)[
            "pred_action"
        ]

    torch.testing.assert_close(actual, expected)


def test_act_rollout_run_mode_does_not_pass_force_to_policy():
    class CapturingACTPolicy(ACTPolicyBaseline):
        def __init__(self):
            super().__init__(
                d_model=32,
                z_dim=8,
                q_dim=7,
                action_dim=7,
                chunk_len=4,
                nhead=4,
                num_encoder_layers=1,
                num_decoder_layers=1,
                dim_feedforward=64,
                dropout=0.0,
                pretrained_resnet18=False,
            )
            self.forward_called = False

        def forward(self, images, qpos, action_chunk=None, is_training=True, motion_latent_override=None):
            self.forward_called = True
            assert action_chunk is None
            assert is_training is False
            return super().forward(
                images=images,
                qpos=qpos,
                action_chunk=action_chunk,
                is_training=is_training,
                motion_latent_override=motion_latent_override,
            )

    model = CapturingACTPolicy()
    model.eval()
    inputs = _make_inputs(chunk_len=4)
    force_window = torch.randn(2, 20, 6)

    with torch.no_grad():
        outputs = _run_mode(model, inputs["images"], inputs["qpos"], force_window, "prior")

    assert model.forward_called
    assert outputs["pred_action"].shape == (2, 4, 7)
    assert "pred_force" not in outputs


def test_act_parameter_audit_has_no_force_contact_or_unclassified_parameters():
    config = dict(DEFAULT_ACT_SYNTHETIC_CONFIG)
    config.update(
        {
            "d_model": 32,
            "z_dim": 8,
            "chunk_len": 4,
            "nhead": 4,
            "num_encoder_layers": 1,
            "num_decoder_layers": 1,
            "dim_feedforward": 64,
        }
    )
    audit = create_audit(config=config, device="cpu", policy_variant="act_baseline")

    assert audit["components"]["force_temporal_encoder"]["total_parameters"] == 0
    assert audit["components"]["force_vision_fusion"]["total_parameters"] == 0
    assert audit["components"]["force_head"]["total_parameters"] == 0
    assert audit["components"]["contact_latent_prior_posterior"]["total_parameters"] == 0
    assert audit["components"]["motion_latent_modules"]["total_parameters"] > 0
    assert audit["components"]["other_unclassified"]["total_parameters"] == 0
    assert audit["unclassified_parameter_names"] == []


def _write_no_force_episode(path: Path) -> None:
    n_state = 12
    n_image = 12
    height = 16
    width = 16
    state_ts = np.arange(n_state, dtype=np.float32) * 0.1
    image_ts = np.arange(n_image, dtype=np.float32) * 0.1
    joint_pos = np.arange(n_state * 7, dtype=np.float32).reshape(n_state, 7)

    with h5py.File(path, "w") as handle:
        timestamps = handle.create_group("timestamps")
        timestamps.create_dataset("state_episode", data=state_ts)
        timestamps.create_dataset("image_episode", data=image_ts)
        observations = handle.create_group("observations")
        observations.create_dataset("joint_pos", data=joint_pos)
        observations.create_dataset("joint_vel", data=joint_pos + 100.0)
        observations.create_dataset("joint_torque", data=joint_pos + 200.0)
        observations.create_dataset("ee_pose", data=joint_pos + 300.0)
        images = observations.create_group("images")
        images.create_dataset(
            "ee_cam",
            data=np.zeros((n_image, height, width, 3), dtype=np.uint8),
        )
        images.create_dataset(
            "base_top_cam",
            data=np.zeros((n_image, height, width, 3), dtype=np.uint8),
        )
        handle.create_dataset("action", data=joint_pos + 1000.0)


def test_act_dataset_path_can_skip_force_fields(tmp_path):
    episode_path = tmp_path / "no_force_episode.hdf5"
    _write_no_force_episode(episode_path)

    dataset = ContactForceHDF5Dataset(
        episode_path,
        action_mode="action",
        chunk_len=4,
        image_size=(32, 32),
        include_force=False,
    )
    sample = dataset[0]

    assert sample["images"].shape == (2, 3, 32, 32)
    assert sample["qpos"].shape == (7,)
    assert sample["action_chunk"].shape == (4, 7)
    assert "force_window" not in sample
    assert "future_force_chunk" not in sample
