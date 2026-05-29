from pathlib import Path

import h5py
import numpy as np

from force_aware_act.data import ContactForceHDF5Dataset


def _write_fake_episode(path: Path) -> None:
    n_state = 100
    n_force = 500
    n_image = 50
    height = 64
    width = 64

    state_ts = np.arange(n_state, dtype=np.float32) * 0.1
    force_ts = np.arange(n_force, dtype=np.float32) * 0.02
    image_ts = np.arange(n_image, dtype=np.float32) * 0.2

    joint_pos = np.arange(n_state * 7, dtype=np.float32).reshape(n_state, 7)
    joint_vel = joint_pos + 1000.0
    joint_torque = joint_pos + 2000.0
    ee_pose = joint_pos + 3000.0
    ft_wrench = np.arange(n_force * 6, dtype=np.float32).reshape(n_force, 6)

    ee_cam = np.zeros((n_image, height, width, 3), dtype=np.uint8)
    base_top_cam = np.zeros((n_image, height, width, 3), dtype=np.uint8)
    for idx in range(n_image):
        ee_cam[idx, :, :, :] = idx
        base_top_cam[idx, :, :, :] = idx + 50

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

        images = observations.create_group("images")
        images.create_dataset("ee_cam", data=ee_cam)
        images.create_dataset("base_top_cam", data=base_top_cam)


def test_contact_force_hdf5_dataset_shapes(tmp_path):
    episode_path = tmp_path / "episode.hdf5"
    _write_fake_episode(episode_path)

    dataset = ContactForceHDF5Dataset(
        [episode_path],
        chunk_len=10,
        force_window_len=8,
        force_window_duration=0.14,
    )

    sample = dataset[5]

    assert len(dataset) == 89
    assert sample["images"].shape == (2, 3, 64, 64)
    assert sample["qpos"].shape == (7,)
    assert sample["qvel"].shape == (7,)
    assert sample["joint_torque"].shape == (7,)
    assert sample["ee_pose"].shape == (7,)
    assert sample["force_window"].shape == (8, 6)
    assert sample["action_chunk"].shape == (10, 7)
    assert sample["future_force_chunk"].shape == (10, 6)


def test_contact_force_hdf5_dataset_timestamp_alignment(tmp_path):
    episode_path = tmp_path / "episode.hdf5"
    _write_fake_episode(episode_path)

    dataset = ContactForceHDF5Dataset(
        episode_path,
        chunk_len=4,
        force_window_len=5,
        force_window_duration=0.08,
        normalize_images=False,
    )

    sample = dataset[5]

    assert sample["state_index"] == 5
    assert sample["t_state"] == np.float32(0.5)
    assert sample["image_index"] == 2
    np.testing.assert_array_equal(sample["qpos"].numpy(), np.arange(35, 42, dtype=np.float32))
    np.testing.assert_array_equal(
        sample["action_chunk"].numpy(),
        np.arange(42, 70, dtype=np.float32).reshape(4, 7),
    )

    force_indices = sample["force_indices"]
    assert np.all(np.arange(500, dtype=np.float32)[force_indices] * 0.02 <= sample["t_state"])
    np.testing.assert_array_equal(force_indices, np.array([21, 22, 23, 24, 25]))
    np.testing.assert_array_equal(
        sample["force_window"].numpy(),
        np.arange(126, 156, dtype=np.float32).reshape(5, 6),
    )

    expected_future_force_indices = np.array([25, 30, 35, 40])
    np.testing.assert_array_equal(
        sample["future_force_chunk"].numpy(),
        np.arange(500 * 6, dtype=np.float32)
        .reshape(500, 6)[expected_future_force_indices],
    )
