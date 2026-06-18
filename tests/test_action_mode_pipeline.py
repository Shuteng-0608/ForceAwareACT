from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import torch

from force_aware_act.data import ContactForceHDF5Dataset, compute_normalization_stats
from scripts.compute_normalization_stats import compute_and_save
from scripts.evaluate_inference_modes import (
    _build_evaluation_dataset,
    _validate_normalization_action_mode as validate_eval_stats_action_mode,
)
from scripts.train_minimal import (
    _build_training_dataset,
    _validate_normalization_action_mode as validate_train_stats_action_mode,
)


def _write_action_mode_episode(path: Path) -> None:
    n_state = 32
    n_force = 96
    n_image = 32
    state_ts = np.arange(n_state, dtype=np.float32) / 30.0
    force_ts = np.arange(n_force, dtype=np.float32) / 90.0
    image_ts = state_ts.copy()

    joint_pos = np.arange(n_state * 7, dtype=np.float32).reshape(n_state, 7) * 0.01
    joint_vel = joint_pos + 1.0
    joint_torque = joint_pos + 2.0
    ee_pose = joint_pos + 3.0
    ft_wrench = np.arange(n_force * 6, dtype=np.float32).reshape(n_force, 6) * 0.01
    action = joint_pos + 100.0
    images = np.zeros((n_image, 8, 8, 3), dtype=np.uint8)

    with h5py.File(path, "w") as handle:
        timestamps = handle.create_group("timestamps")
        timestamps.create_dataset("state_episode", data=state_ts)
        timestamps.create_dataset("force_episode", data=force_ts)
        timestamps.create_dataset("image_episode", data=image_ts)

        observations = handle.create_group("observations")
        observations.create_dataset("joint_pos", data=joint_pos)
        observations.create_dataset("joint_vel", data=joint_vel)
        observations.create_dataset("joint_torque", data=joint_torque)
        observations.create_dataset("ee_pose", data=ee_pose)
        observations.create_dataset("ft_wrench", data=ft_wrench)
        image_group = observations.create_group("images")
        image_group.create_dataset("ee_cam", data=images)
        image_group.create_dataset("base_top_cam", data=images)

        handle.create_dataset("action", data=action)
        actions = handle.create_group("actions")
        actions.create_dataset("joint_pos_command", data=action)


def _dataset_args(episode_path: Path, action_mode: str) -> SimpleNamespace:
    return SimpleNamespace(
        episode_paths=[episode_path],
        camera_names=("ee_cam", "base_top_cam"),
        action_mode=action_mode,
        chunk_len=4,
        force_window_len=5,
        force_window_duration=0.1,
        image_size=(224, 224),
        imagenet_normalize=False,
    )


def test_training_dataset_builder_passes_action_mode_action(tmp_path):
    episode_path = tmp_path / "episode.hdf5"
    _write_action_mode_episode(episode_path)

    dataset = _build_training_dataset(_dataset_args(episode_path, "action"))

    assert dataset.action_mode == "action"
    sample = dataset[3]
    np.testing.assert_allclose(sample["action_chunk"].numpy()[0], sample["qpos"].numpy() + 100.0)


def test_training_dataset_builder_passes_action_mode_delta_joint_cmd(tmp_path):
    episode_path = tmp_path / "episode.hdf5"
    _write_action_mode_episode(episode_path)

    dataset = _build_training_dataset(_dataset_args(episode_path, "delta_joint_cmd"))
    sample = dataset[3]
    expected = (
        ContactForceHDF5Dataset(
            episode_path,
            action_mode="action",
            chunk_len=4,
            force_window_len=5,
        )[3]["action_chunk"].numpy()
        - sample["qpos"].numpy()[None, :]
    )

    assert dataset.action_mode == "delta_joint_cmd"
    np.testing.assert_allclose(sample["action_chunk"].numpy(), expected)


def test_normalization_stats_metadata_records_action_mode(tmp_path):
    episode_path = tmp_path / "episode.hdf5"
    output_path = tmp_path / "stats.pt"
    _write_action_mode_episode(episode_path)
    args = SimpleNamespace(
        episode_paths=[episode_path],
        episode_list=None,
        output=output_path,
        action_mode="action",
        chunk_len=4,
        force_window_len=5,
        force_window_duration=0.1,
        image_size=(224, 224),
        camera_names=("ee_cam", "base_top_cam"),
        imagenet_normalize=False,
        batch_size=4,
        num_workers=0,
        eps=1.0e-6,
    )

    assert compute_and_save(args) == 0
    stats = torch.load(output_path, map_location="cpu")

    assert stats["action_mode"] == "action"
    assert stats["chunk_len"] == 4
    assert stats["force_window_len"] == 5
    assert stats["camera_names"] == ("ee_cam", "base_top_cam")


def test_action_stats_differ_between_joint_pos_and_action_modes(tmp_path):
    episode_path = tmp_path / "episode.hdf5"
    _write_action_mode_episode(episode_path)
    joint_pos_dataset = ContactForceHDF5Dataset(
        episode_path,
        action_mode="joint_pos",
        chunk_len=4,
        force_window_len=5,
    )
    action_dataset = ContactForceHDF5Dataset(
        episode_path,
        action_mode="action",
        chunk_len=4,
        force_window_len=5,
    )

    joint_pos_stats = compute_normalization_stats(joint_pos_dataset, batch_size=8)
    action_stats = compute_normalization_stats(action_dataset, batch_size=8)

    assert not torch.allclose(joint_pos_stats["action_mean"], action_stats["action_mean"])


def test_mismatched_normalization_action_mode_raises_clear_error():
    stats = {"action_mode": "joint_pos"}

    with pytest.raises(ValueError, match="action_mode mismatch"):
        validate_train_stats_action_mode(stats, "action")
    with pytest.raises(ValueError, match="action_mode mismatch"):
        validate_eval_stats_action_mode(stats, "delta_joint_cmd")


def test_evaluation_dataset_builder_passes_action_mode(tmp_path):
    episode_path = tmp_path / "episode.hdf5"
    _write_action_mode_episode(episode_path)

    dataset = _build_evaluation_dataset(_dataset_args(episode_path, "action"))

    assert dataset.action_mode == "action"
    assert dataset[0]["action_chunk"].shape == (4, 7)
