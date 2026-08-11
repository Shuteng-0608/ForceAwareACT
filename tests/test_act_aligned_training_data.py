import json
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np
import pytest
import torch

from force_aware_act.act_aligned_training import (
    ACTAlignedHDF5Dataset,
    ACTAlignedHighRateHDF5Dataset,
    collate_high_rate_samples,
    collate_act_aligned_samples,
    compute_normalization_stats,
    create_episode_split,
    discover_episodes,
)
from force_aware_act.act_aligned_training.checkpoint import (
    CHECKPOINT_FORMAT_VERSION,
)
from force_aware_act.act_aligned_training.schema import (
    inspect_episode,
    load_state_aligned_force,
)
from force_aware_act.inference import RolloutPolicyAdapter
from force_aware_act.models.act_aligned import (
    ACTAlignedConfig,
    ACTAlignedContactCVAEPolicy,
    ACTAlignedHighRateConfig,
)


def _write_episode(root: Path, name: str, offset: float) -> None:
    episode_dir = root / name
    episode_dir.mkdir()
    path = episode_dir / "episode.hdf5"
    state_time = np.arange(4, dtype=np.float64)
    force_time = np.arange(0.0, 4.0, 0.5, dtype=np.float64)
    with h5py.File(path, "w") as handle:
        handle.attrs["schema_version"] = "compact_mujoco_hdf5_v1"
        handle.create_dataset(
            "action",
            data=np.arange(28, dtype=np.float64).reshape(4, 7) + offset,
        )
        observations = handle.create_group("observations")
        observations.create_dataset(
            "joint_pos",
            data=np.arange(28, dtype=np.float64).reshape(4, 7) + offset,
        )
        observations.create_dataset(
            "ft_wrench",
            data=np.repeat(
                (np.arange(8, dtype=np.float64) + offset)[:, None],
                6,
                axis=1,
            ),
        )
        images = observations.create_group("images")
        for camera_index, camera_name in enumerate(("ee_cam", "base_top_cam")):
            images.create_dataset(
                camera_name,
                data=np.full(
                    (4, 6, 8, 3),
                    20 + camera_index * 30,
                    dtype=np.uint8,
                ),
            )
        timestamps = handle.create_group("timestamps")
        timestamps.create_dataset("state", data=state_time)
        timestamps.create_dataset("image", data=state_time)
        timestamps.create_dataset("force", data=force_time)
        metadata = handle.create_group("episode_metadata")
        metadata.create_dataset(
            "camera_names",
            data=np.asarray([b"ee_cam", b"base_top_cam"]),
        )
    (episode_dir / "metadata.json").write_text(
        json.dumps(
            {
                "schema_version": "compact_mujoco_hdf5_v1",
                "n_state": 4,
            }
        ),
        encoding="utf-8",
    )


def _replace_with_async_images(root: Path, name: str) -> None:
    path = root / name / "episode.hdf5"
    image_time = np.asarray([0.0, 1.0, 3.0], dtype=np.float64)
    with h5py.File(path, "a") as handle:
        del handle["timestamps/image"]
        handle["timestamps"].create_dataset("image", data=image_time)
        for camera_index, camera_name in enumerate(
            ("ee_cam", "base_top_cam")
        ):
            key = f"observations/images/{camera_name}"
            del handle[key]
            frames = np.stack(
                [
                    np.full(
                        (6, 8, 3),
                        10 * image_index + 20 + camera_index * 30,
                        dtype=np.uint8,
                    )
                    for image_index in range(len(image_time))
                ]
            )
            handle["observations/images"].create_dataset(
                camera_name,
                data=frames,
            )


def _config(**overrides):
    values = {
        "d_model": 32,
        "nhead": 4,
        "dim_feedforward": 64,
        "dropout": 0.0,
        "chunk_len": 3,
        "force_window_len": 2,
        "image_height": 8,
        "image_width": 8,
        "pretrained_backbone": False,
        "imagenet_normalize": False,
    }
    values.update(overrides)
    return ACTAlignedConfig(
        **values,
    )


def _high_rate_config(**overrides):
    values = {
        "d_model": 32,
        "nhead": 4,
        "dim_feedforward": 64,
        "dropout": 0.0,
        "chunk_len": 3,
        "local_force_dim": 16,
        "image_height": 8,
        "image_width": 8,
        "pretrained_backbone": False,
        "imagenet_normalize": False,
    }
    values.update(overrides)
    return ACTAlignedHighRateConfig(**values)


def test_split_stats_and_dataset_are_deterministic_causal_and_time_aligned(tmp_path):
    _write_episode(tmp_path, "episode_a", 0.0)
    _write_episode(tmp_path, "episode_b", 100.0)
    first_manifest = create_episode_split(
        tmp_path,
        validation_fraction=0.5,
        seed=4,
    )
    second_manifest = create_episode_split(
        tmp_path,
        validation_fraction=0.5,
        seed=4,
    )

    assert first_manifest == second_manifest
    assert {
        item.episode_id for item in first_manifest.train_episodes
    }.isdisjoint(
        item.episode_id for item in first_manifest.validation_episodes
    )
    stats = compute_normalization_stats(
        tmp_path,
        first_manifest.train_episodes,
    )
    train_offset = (
        0.0
        if first_manifest.train_episodes[0].episode_id == "episode_a"
        else 100.0
    )
    assert min(stats.qpos_mean) >= train_offset
    assert max(stats.qpos_mean) < train_offset + 28

    dataset = ACTAlignedHDF5Dataset(
        tmp_path,
        first_manifest.train_episodes,
        stats,
        _config(),
    )
    first = dataset[0]
    second = dataset[1]
    last = dataset[3]

    assert first.images.shape == (2, 3, 8, 8)
    assert first.images.dtype is torch.float32
    assert first.force_padding_mask.tolist() == [True, False]
    assert second.force_padding_mask.tolist() == [False, False]
    assert last.future_padding_mask.tolist() == [False, True, True]
    raw_second_force = stats.denormalize_force(second.force_history)[-1]
    torch.testing.assert_close(
        raw_second_force,
        torch.full((6,), train_offset + 2.0),
    )
    raw_future_force = stats.denormalize_force(second.future_force_chunk)[0]
    torch.testing.assert_close(raw_future_force, raw_second_force)
    batch = collate_act_aligned_samples([first, last])
    batch.validate(_config())
    dataset.close()


def test_discovery_validates_all_episode_records(tmp_path):
    _write_episode(tmp_path, "episode_a", 0.0)
    _write_episode(tmp_path, "episode_b", 10.0)

    records = discover_episodes(tmp_path)

    assert len(records) == 2
    assert records[0].num_steps == 4
    assert records[0].camera_names == ("ee_cam", "base_top_cam")


def test_async_image_count_is_validated_and_causally_aligned(tmp_path):
    _write_episode(tmp_path, "episode_async", 0.0)
    _write_episode(tmp_path, "episode_sync", 100.0)
    _replace_with_async_images(tmp_path, "episode_async")

    schema = inspect_episode(tmp_path / "episode_async" / "episode.hdf5")
    assert schema.num_steps == 4
    assert schema.num_image_samples == 3
    records = discover_episodes(tmp_path)
    async_record = next(
        record for record in records if record.episode_id == "episode_async"
    )
    stats = compute_normalization_stats(tmp_path, (async_record,))
    dataset = ACTAlignedHDF5Dataset(
        tmp_path,
        (async_record,),
        stats,
        _config(),
    )

    sample = dataset[2]

    assert sample.timestep == 2
    torch.testing.assert_close(
        sample.images[0],
        torch.full_like(sample.images[0], 30.0 / 255.0),
    )
    dataset.close()


def test_async_image_timestamp_count_must_match_every_camera(tmp_path):
    _write_episode(tmp_path, "episode_bad", 0.0)
    _replace_with_async_images(tmp_path, "episode_bad")
    path = tmp_path / "episode_bad" / "episode.hdf5"
    with h5py.File(path, "a") as handle:
        del handle["observations/images/ee_cam"]
        handle["observations/images"].create_dataset(
            "ee_cam",
            data=np.zeros((2, 6, 8, 3), dtype=np.uint8),
        )

    with pytest.raises(ValueError, match="camera 'ee_cam' has invalid shape"):
        inspect_episode(path)


def test_dataset_preserves_native_images_without_interpolation(tmp_path):
    _write_episode(tmp_path, "episode_a", 0.0)
    _write_episode(tmp_path, "episode_b", 10.0)
    manifest = create_episode_split(
        tmp_path,
        validation_fraction=0.5,
        seed=0,
    )
    stats = compute_normalization_stats(
        tmp_path,
        manifest.train_episodes,
    )
    dataset = ACTAlignedHDF5Dataset(
        tmp_path,
        manifest.train_episodes,
        stats,
        _config(image_height=6, image_width=8),
    )

    with patch(
        "force_aware_act.act_aligned_training.data.functional.interpolate",
        wraps=torch.nn.functional.interpolate,
    ) as interpolate:
        sample = dataset[0]

    interpolate.assert_not_called()
    torch.testing.assert_close(
        sample.images[0],
        torch.full((3, 6, 8), 20.0 / 255.0),
    )
    torch.testing.assert_close(
        sample.images[1],
        torch.full((3, 6, 8), 50.0 / 255.0),
    )
    dataset.close()


def test_training_and_rollout_share_identical_causal_force_windows(tmp_path):
    _write_episode(tmp_path, "episode_a", 0.0)
    _write_episode(tmp_path, "episode_b", 10.0)
    manifest = create_episode_split(
        tmp_path,
        validation_fraction=0.5,
        seed=0,
    )
    record = manifest.train_episodes[0]
    stats = compute_normalization_stats(tmp_path, (record,))
    config = _config(force_window_len=2)
    dataset = ACTAlignedHDF5Dataset(tmp_path, (record,), stats, config)
    checkpoint = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "architecture_version": config.architecture_version,
        "model_config": config.__dict__,
        "model_state": ACTAlignedContactCVAEPolicy(config).state_dict(),
        "normalization": stats.to_dict(),
    }
    adapter = RolloutPolicyAdapter.from_checkpoint(
        checkpoint,
        device=torch.device("cpu"),
    )
    with h5py.File(record.resolve(tmp_path), "r") as handle:
        aligned_force = load_state_aligned_force(handle)

    for timestep in range(record.num_steps):
        training_sample = dataset[timestep]
        rollout_history, rollout_mask = adapter.prepare_force_history(
            aligned_force[: timestep + 1]
        )
        torch.testing.assert_close(
            rollout_history[0],
            training_sample.force_history,
            atol=0.0,
            rtol=0.0,
        )
        assert torch.equal(
            rollout_mask[0],
            training_sample.force_padding_mask,
        )

    dataset.close()


def test_high_rate_dataset_is_independent_and_preserves_action_force_masks(tmp_path):
    _write_episode(tmp_path, "episode_a", 0.0)
    _write_episode(tmp_path, "episode_b", 10.0)
    manifest = create_episode_split(tmp_path, validation_fraction=0.5, seed=0)
    record = manifest.train_episodes[0]
    stats = compute_normalization_stats(tmp_path, (record,))
    config = _high_rate_config()
    dataset = ACTAlignedHighRateHDF5Dataset(
        tmp_path,
        (record,),
        stats,
        config,
    )

    first = dataset[0]
    last = dataset[record.num_steps - 1]
    batch = collate_high_rate_samples((first, last))
    batch.validate(config)

    assert first.online_raw_valid_count == 1
    assert first.online_force_intervals.shape == (7, 20, 6)
    assert first.future_force_intervals.shape == (3, 20, 6)
    assert not first.action_padding_mask[0]
    assert not first.future_force_interval_padding_mask[0]
    assert not last.action_padding_mask[0]
    assert last.future_force_interval_padding_mask.all()
    assert last.action_padding_mask[1:].all()
    dataset.close()
