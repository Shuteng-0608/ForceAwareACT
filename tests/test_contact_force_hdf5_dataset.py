from pathlib import Path

import h5py
import numpy as np
import pytest

from force_aware_act.data import ContactForceHDF5Dataset


def _write_fake_episode(path: Path, include_actions: bool = False) -> None:
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
    action = joint_pos + 10_000.0

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

        if include_actions:
            handle.create_dataset("action", data=action)
            actions = handle.create_group("actions")
            actions.create_dataset("joint_pos_command", data=action)


def _shorten_dataset(path: Path, key: str, remove_count: int) -> None:
    with h5py.File(path, "r+") as handle:
        values = np.asarray(handle[key])
        del handle[key]
        handle.create_dataset(key, data=values[:-remove_count])


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

    assert len(dataset) == 90
    assert sample["images"].shape == (2, 3, 224, 224)
    np.testing.assert_allclose(sample["images"][0].numpy(), np.full((3, 224, 224), 2 / 255.0))
    np.testing.assert_allclose(
        sample["images"][1].numpy(),
        np.full((3, 224, 224), 52 / 255.0),
    )
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

    assert sample["images"].shape == (2, 3, 224, 224)
    np.testing.assert_array_equal(sample["images"][0].numpy(), np.full((3, 224, 224), 2.0))
    np.testing.assert_array_equal(sample["images"][1].numpy(), np.full((3, 224, 224), 52.0))
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


def test_action_mode_joint_pos_still_works_without_command_labels(tmp_path):
    episode_path = tmp_path / "episode.hdf5"
    _write_fake_episode(episode_path, include_actions=False)

    dataset = ContactForceHDF5Dataset(
        episode_path,
        action_mode="joint_pos",
        chunk_len=4,
        force_window_len=5,
    )

    sample = dataset[5]

    np.testing.assert_array_equal(
        sample["action_chunk"].numpy(),
        np.arange(42, 70, dtype=np.float32).reshape(4, 7),
    )


def test_action_mode_action_uses_root_command_labels(tmp_path):
    episode_path = tmp_path / "episode.hdf5"
    _write_fake_episode(episode_path, include_actions=True)

    dataset = ContactForceHDF5Dataset(
        episode_path,
        action_mode="action",
        chunk_len=4,
        force_window_len=5,
    )

    sample = dataset[5]
    expected = np.arange(35, 63, dtype=np.float32).reshape(4, 7) + 10_000.0

    assert len(dataset) == 97
    assert sample["action_chunk"].shape == (4, 7)
    np.testing.assert_array_equal(sample["action_chunk"].numpy(), expected)


def test_action_mode_joint_pos_command_uses_semantic_command_copy(tmp_path):
    episode_path = tmp_path / "episode.hdf5"
    _write_fake_episode(episode_path, include_actions=True)

    action_dataset = ContactForceHDF5Dataset(
        episode_path,
        action_mode="action",
        chunk_len=4,
        force_window_len=5,
    )
    command_dataset = ContactForceHDF5Dataset(
        episode_path,
        action_mode="joint_pos_command",
        chunk_len=4,
        force_window_len=5,
    )

    np.testing.assert_array_equal(
        action_dataset[5]["action_chunk"].numpy(),
        command_dataset[5]["action_chunk"].numpy(),
    )


def test_action_mode_delta_joint_cmd_is_command_minus_current_qpos(tmp_path):
    episode_path = tmp_path / "episode.hdf5"
    _write_fake_episode(episode_path, include_actions=True)

    dataset = ContactForceHDF5Dataset(
        episode_path,
        action_mode="delta_joint_cmd",
        chunk_len=4,
        force_window_len=5,
    )

    sample = dataset[5]
    current_qpos = np.arange(35, 42, dtype=np.float32)
    expected_action = np.arange(35, 63, dtype=np.float32).reshape(4, 7) + 10_000.0
    expected_delta = expected_action - current_qpos[None, :]

    np.testing.assert_array_equal(sample["action_chunk"].numpy(), expected_delta)


def test_action_mode_delta_joint_pos_command_is_command_minus_current_qpos(tmp_path):
    episode_path = tmp_path / "episode.hdf5"
    _write_fake_episode(episode_path, include_actions=True)

    dataset = ContactForceHDF5Dataset(
        episode_path,
        action_mode="delta_joint_pos_command",
        chunk_len=4,
        force_window_len=5,
    )

    sample = dataset[5]
    current_qpos = np.arange(35, 42, dtype=np.float32)
    expected_action = np.arange(35, 63, dtype=np.float32).reshape(4, 7) + 10_000.0
    expected_delta = expected_action - current_qpos[None, :]

    np.testing.assert_array_equal(sample["action_chunk"].numpy(), expected_delta)


def test_command_action_mode_missing_dataset_raises_clear_error(tmp_path):
    episode_path = tmp_path / "episode.hdf5"
    _write_fake_episode(episode_path, include_actions=False)

    with pytest.raises(KeyError, match="missing action dataset"):
        ContactForceHDF5Dataset(episode_path, action_mode="action")


def test_command_action_mode_length_mismatch_raises(tmp_path):
    episode_path = tmp_path / "episode.hdf5"
    _write_fake_episode(episode_path, include_actions=True)
    _shorten_dataset(episode_path, "action", 2)

    with pytest.raises(ValueError, match="action dataset length mismatch"):
        ContactForceHDF5Dataset(episode_path, action_mode="action")


def test_command_action_mode_returns_expected_shapes_and_finite_tensors(tmp_path):
    episode_path = tmp_path / "episode.hdf5"
    _write_fake_episode(episode_path, include_actions=True)

    dataset = ContactForceHDF5Dataset(
        episode_path,
        action_mode="action",
        chunk_len=10,
        force_window_len=20,
    )
    sample = dataset[5]

    assert sample["images"].shape == (2, 3, 224, 224)
    assert sample["qpos"].shape == (7,)
    assert sample["force_window"].shape == (20, 6)
    assert sample["action_chunk"].shape == (10, 7)
    assert sample["future_force_chunk"].shape == (10, 6)
    for key in ("qpos", "force_window", "action_chunk", "future_force_chunk"):
        assert np.isfinite(sample[key].numpy()).all()


def test_contact_force_hdf5_dataset_imagenet_normalization(tmp_path):
    episode_path = tmp_path / "episode.hdf5"
    _write_fake_episode(episode_path)

    dataset = ContactForceHDF5Dataset(
        episode_path,
        chunk_len=4,
        force_window_len=5,
        force_window_duration=0.08,
        imagenet_normalize=True,
    )

    sample = dataset[5]

    expected = np.array(
        [
            (2 / 255.0 - 0.485) / 0.229,
            (2 / 255.0 - 0.456) / 0.224,
            (2 / 255.0 - 0.406) / 0.225,
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(
        sample["images"][0].numpy(),
        expected[:, None, None] * np.ones((3, 224, 224), dtype=np.float32),
        rtol=1e-6,
    )


def test_perfectly_aligned_episode_has_no_safe_length_trimming(tmp_path):
    episode_path = tmp_path / "episode.hdf5"
    _write_fake_episode(episode_path)

    dataset = ContactForceHDF5Dataset(episode_path, chunk_len=10)
    safe_lengths = dataset.episode_safe_lengths[episode_path]

    assert len(dataset) == 90
    assert safe_lengths.trim_state == 0
    assert safe_lengths.trim_image == 0
    assert safe_lengths.trim_force == 0
    assert safe_lengths.mismatch_groups == ()


def test_one_frame_image_timestamp_mismatch_works_in_tolerant_mode(tmp_path):
    episode_path = tmp_path / "episode.hdf5"
    _write_fake_episode(episode_path)
    _shorten_dataset(episode_path, "timestamps/image_episode", 1)

    dataset = ContactForceHDF5Dataset(episode_path, chunk_len=10)

    assert dataset.episode_safe_lengths[episode_path].image_len == 49
    assert dataset.episode_safe_lengths[episode_path].trim_image == 1
    assert dataset[len(dataset) - 1]["image_index"] < 49


def test_one_frame_base_top_camera_mismatch_works_in_tolerant_mode(tmp_path):
    episode_path = tmp_path / "episode.hdf5"
    _write_fake_episode(episode_path)
    _shorten_dataset(episode_path, "observations/images/base_top_cam", 1)

    dataset = ContactForceHDF5Dataset(episode_path, chunk_len=10)
    sample = dataset[len(dataset) - 1]

    assert dataset.episode_safe_lengths[episode_path].image_len == 49
    assert sample["images"].shape == (2, 3, 224, 224)
    assert sample["image_index"] < 49


def test_one_frame_force_timestamp_mismatch_works_in_tolerant_mode(tmp_path):
    episode_path = tmp_path / "episode.hdf5"
    _write_fake_episode(episode_path)
    _shorten_dataset(episode_path, "timestamps/force_episode", 1)

    dataset = ContactForceHDF5Dataset(episode_path, chunk_len=10)
    sample = dataset[len(dataset) - 1]

    assert dataset.episode_safe_lengths[episode_path].force_len == 499
    assert dataset.episode_safe_lengths[episode_path].trim_force == 1
    assert sample["force_indices"].max() < 499


def test_length_mismatch_strict_mode_raises(tmp_path):
    episode_path = tmp_path / "episode.hdf5"
    _write_fake_episode(episode_path)
    _shorten_dataset(episode_path, "timestamps/image_episode", 1)

    with pytest.raises(ValueError, match="image group length mismatch"):
        ContactForceHDF5Dataset(
            episode_path,
            tolerate_length_mismatch=False,
        )


def test_length_mismatch_above_maximum_raises(tmp_path):
    episode_path = tmp_path / "episode.hdf5"
    _write_fake_episode(episode_path)
    _shorten_dataset(episode_path, "timestamps/image_episode", 2)

    with pytest.raises(ValueError, match="exceeds max_length_mismatch=1"):
        ContactForceHDF5Dataset(
            episode_path,
            tolerate_length_mismatch=True,
            max_length_mismatch=1,
        )


def test_dataset_preflight_rejects_nonfinite_numeric_values(tmp_path):
    episode_path = tmp_path / "episode.hdf5"
    _write_fake_episode(episode_path, include_actions=True)
    with h5py.File(episode_path, "r+") as handle:
        handle["observations/joint_pos"][4, 0] = np.nan

    with pytest.raises(ValueError, match="joint_pos contains non-finite"):
        ContactForceHDF5Dataset(episode_path, action_mode="action")


def test_latest_past_alignment_never_uses_future_images_and_enforces_lag(tmp_path):
    episode_path = tmp_path / "episode.hdf5"
    _write_fake_episode(episode_path)
    dataset = ContactForceHDF5Dataset(
        episode_path,
        chunk_len=4,
        image_alignment="latest_past",
        max_image_lag_seconds=0.05,
    )

    assert dataset.indices
    assert all(index.state_index % 2 == 0 for index in dataset.indices)
    sample = dataset[3]
    with h5py.File(episode_path, "r") as handle:
        image_time = handle["timestamps/image_episode"][sample["image_index"]]
    assert float(image_time) <= sample["t_state"]
    assert sample["t_state"] - float(image_time) <= 0.05

    inclusive = ContactForceHDF5Dataset(
        episode_path,
        chunk_len=4,
        image_alignment="latest_past",
        max_image_lag_seconds=0.1,
    )
    assert len(inclusive) == 96


def test_episode_with_exactly_one_action_chunk_is_not_dropped(tmp_path):
    episode_path = tmp_path / "episode.hdf5"
    _write_fake_episode(episode_path, include_actions=True)

    command_dataset = ContactForceHDF5Dataset(
        episode_path,
        action_mode="action",
        chunk_len=100,
    )
    joint_position_dataset = ContactForceHDF5Dataset(
        episode_path,
        action_mode="joint_pos",
        chunk_len=99,
    )

    assert len(command_dataset) == 1
    assert command_dataset.indices[0].state_index == 0
    assert len(joint_position_dataset) == 1
    assert joint_position_dataset.indices[0].state_index == 0


def test_dataset_skips_decisions_without_any_past_force_sample(tmp_path):
    episode_path = tmp_path / "episode.hdf5"
    _write_fake_episode(episode_path)
    with h5py.File(episode_path, "r+") as handle:
        handle["timestamps/force_episode"][:] += np.float32(0.2)

    dataset = ContactForceHDF5Dataset(episode_path, chunk_len=4)
    assert dataset.indices[0].state_index == 2
