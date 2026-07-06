import argparse
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

pytest.importorskip("torchvision")

from scripts import train_act_baseline
from scripts.train_minimal import checkpoint_step_path, resolve_checkpoint_steps


def _write_episode(path: Path) -> None:
    n_state = 12
    n_image = 12
    height = 16
    width = 16
    state_ts = np.arange(n_state, dtype=np.float32) * 0.1
    image_ts = np.arange(n_image, dtype=np.float32) * 0.1
    joint_pos = np.arange(n_state * 7, dtype=np.float32).reshape(n_state, 7) * 0.01

    with h5py.File(path, "w") as handle:
        timestamps = handle.create_group("timestamps")
        timestamps.create_dataset("state_episode", data=state_ts)
        timestamps.create_dataset("image_episode", data=image_ts)
        observations = handle.create_group("observations")
        observations.create_dataset("joint_pos", data=joint_pos)
        observations.create_dataset("joint_vel", data=joint_pos + 1.0)
        observations.create_dataset("joint_torque", data=joint_pos + 2.0)
        observations.create_dataset("ee_pose", data=joint_pos + 3.0)
        images = observations.create_group("images")
        images.create_dataset(
            "ee_cam",
            data=np.zeros((n_image, height, width, 3), dtype=np.uint8),
        )
        images.create_dataset(
            "base_top_cam",
            data=np.zeros((n_image, height, width, 3), dtype=np.uint8),
        )
        handle.create_dataset("action", data=joint_pos + 10.0)


def _base_args(
    episode_path: Path,
    output_dir: Path,
    *,
    max_steps: int,
    save_every: int,
):
    return argparse.Namespace(
        episode_paths=[episode_path],
        camera_names=("ee_cam", "base_top_cam"),
        action_mode="joint_pos",
        chunk_len=4,
        image_size=(64, 64),
        imagenet_normalize=False,
        batch_size=1,
        num_workers=0,
        max_steps=max_steps,
        learning_rate=1.0e-4,
        beta_motion_max=1.0e-4,
        warmup_steps=2,
        save_every=save_every,
        save_steps=[],
        d_model=32,
        z_dim=8,
        nhead=4,
        num_encoder_layers=1,
        num_decoder_layers=1,
        dim_feedforward=64,
        dropout=0.0,
        output_dir=output_dir,
        log_csv=output_dir / "train_log.csv",
        device="cpu",
        normalization_stats=None,
    )


def test_train_act_baseline_help_includes_save_every(capsys):
    with pytest.raises(SystemExit) as exc_info:
        train_act_baseline.parse_args(["--help"])

    captured = capsys.readouterr()
    assert exc_info.value.code == 0
    assert "--save-every" in captured.out
    assert "--save-steps" in captured.out


def test_act_save_defaults_and_disable_match_train_minimal():
    args = train_act_baseline.parse_args(["episode.hdf5"])

    assert args.save_every == 0
    assert args.save_steps == []
    assert resolve_checkpoint_steps(max_steps=5, save_every=args.save_every, save_steps=[]) == []


def test_act_checkpoint_schedule_rejects_invalid_intervals():
    with pytest.raises(ValueError, match="--save-every must be non-negative"):
        resolve_checkpoint_steps(max_steps=5, save_every=-1, save_steps=[])
    with pytest.raises(ValueError, match="positive integers"):
        resolve_checkpoint_steps(max_steps=5, save_every=0, save_steps=[0])


def test_act_training_writes_periodic_checkpoints_at_expected_steps(tmp_path):
    episode_path = tmp_path / "episode.hdf5"
    output_dir = tmp_path / "act_periodic"
    _write_episode(episode_path)
    args = _base_args(episode_path, output_dir, max_steps=5, save_every=2)

    assert train_act_baseline.train(args) == 0

    checkpoint_names = sorted(path.name for path in output_dir.glob("checkpoint*.pt"))
    assert checkpoint_names == [
        "checkpoint.pt",
        "checkpoint_step_00000002.pt",
        "checkpoint_step_00000004.pt",
    ]
    assert checkpoint_step_path(output_dir, 2).name == "checkpoint_step_00000002.pt"
    assert checkpoint_step_path(output_dir, 4).name == "checkpoint_step_00000004.pt"

    expected_steps = {
        "checkpoint_step_00000002.pt": 2,
        "checkpoint_step_00000004.pt": 4,
        "checkpoint.pt": 5,
    }
    for filename, expected_step in expected_steps.items():
        checkpoint = torch.load(output_dir / filename, map_location="cpu")
        assert {"model_state_dict", "optimizer_state_dict", "config", "step"} <= set(checkpoint)
        assert checkpoint["step"] == expected_step
        assert checkpoint["config"]["policy_variant"] == "act_baseline"
        assert checkpoint["config"]["act_baseline_version"] == "motion_cvae_v1"
        assert checkpoint["config"]["motion_latent_mode"] == "posterior_train_zero_deploy"
        assert checkpoint["config"]["uses_force"] is False
        assert checkpoint["config"]["uses_contact_latent"] is False
        assert checkpoint["config"]["save_every"] == 2
        assert checkpoint["config"]["intermediate_checkpoint_steps"] == (2, 4)


def test_act_final_divisible_periodic_checkpoint_matches_checkpoint_pt(tmp_path):
    episode_path = tmp_path / "episode.hdf5"
    output_dir = tmp_path / "act_final_periodic"
    _write_episode(episode_path)
    args = _base_args(episode_path, output_dir, max_steps=4, save_every=2)

    assert train_act_baseline.train(args) == 0

    periodic = torch.load(output_dir / "checkpoint_step_00000004.pt", map_location="cpu")
    final = torch.load(output_dir / "checkpoint.pt", map_location="cpu")

    assert periodic["step"] == 4
    assert final["step"] == 4
    assert periodic["config"] == final["config"]
    assert periodic["model_state_dict"].keys() == final["model_state_dict"].keys()
    for name, tensor in periodic["model_state_dict"].items():
        assert torch.equal(tensor, final["model_state_dict"][name]), name
    assert periodic["optimizer_state_dict"].keys() == final["optimizer_state_dict"].keys()
