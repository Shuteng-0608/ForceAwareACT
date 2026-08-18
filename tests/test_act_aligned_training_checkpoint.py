import random
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch

pytest.importorskip("torchvision")

from force_aware_act.act_aligned_training import (  # noqa: E402
    CHECKPOINT_FORMAT_VERSION,
    ACTAlignedTrainingConfig,
    EpisodeRecord,
    EpisodeSplitManifest,
    NormalizationStats,
    TrainingProgress,
    build_act_aligned_optimizer,
    load_act_aligned_checkpoint,
    save_act_aligned_checkpoint,
)
from force_aware_act.models.act_aligned import (  # noqa: E402
    ACTAlignedConfig,
    ACTAlignedContactCVAEPolicy,
)
from force_aware_act.act_aligned_training.checkpoint import (  # noqa: E402
    _restore_rng_state,
)


def _model_config():
    return ACTAlignedConfig(
        d_model=32,
        nhead=4,
        dim_feedforward=64,
        dropout=0.0,
        chunk_len=3,
        force_window_len=2,
        image_height=32,
        image_width=32,
        pretrained_backbone=False,
        imagenet_normalize=False,
    )


def _stats():
    return NormalizationStats(
        qpos_mean=(0.0,) * 7,
        qpos_std=(1.0,) * 7,
        action_mean=(0.0,) * 7,
        action_std=(1.0,) * 7,
        force_mean=(0.0,) * 6,
        force_std=(1.0,) * 6,
    )


def _manifest():
    train = EpisodeRecord("train", "train/episode.hdf5", 4, ("a", "b"))
    validation = EpisodeRecord("val", "val/episode.hdf5", 4, ("a", "b"))
    return EpisodeSplitManifest(
        "act_aligned_episode_split_v1",
        "/data",
        0,
        0.5,
        (train,),
        (validation,),
    )


def test_checkpoint_round_trip_restores_model_optimizer_progress_and_rng(tmp_path):
    model_config = _model_config()
    training_config = ACTAlignedTrainingConfig()
    model = ACTAlignedContactCVAEPolicy(model_config)
    optimizer = build_act_aligned_optimizer(model, training_config)
    generator = torch.Generator().manual_seed(9)
    experiment_manifest = {
        "format_version": "paired_episode_subset_v1",
        "dataset_fingerprint": "fingerprint",
    }
    path = tmp_path / "checkpoint.pt"
    torch.manual_seed(7)
    random.seed(7)
    np.random.seed(7)

    save_act_aligned_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        training_config=training_config,
        progress=TrainingProgress(3, 17, 0.25, step_in_epoch=5),
        normalization=_stats(),
        split_manifest=_manifest(),
        dataloader_generator=generator,
        experiment_manifest=experiment_manifest,
        run_control={"target_optimizer_steps": 24_000},
    )
    expected_random = torch.rand(4)
    with torch.no_grad():
        model.action_head.weight.add_(10.0)

    loaded = load_act_aligned_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        training_config=training_config,
    )
    actual_random = torch.rand(4)

    torch.testing.assert_close(actual_random, expected_random)
    assert loaded.progress == TrainingProgress(3, 17, 0.25, step_in_epoch=5)
    assert loaded.normalization == _stats()
    assert loaded.split_manifest == _manifest()
    payload = torch.load(path, weights_only=False)
    assert payload["experiment_manifest"] == experiment_manifest
    assert payload["run_control"] == {
        "target_optimizer_steps": 24_000,
    }
    torch.testing.assert_close(
        loaded.dataloader_generator_state,
        generator.get_state(),
    )


def test_rng_restore_normalizes_map_location_and_visible_gpu_count():
    expected_cpu = torch.get_rng_state()
    expected_cuda = torch.arange(8, dtype=torch.uint8)
    loaded_cpu = Mock(spec=torch.Tensor)
    loaded_cuda = Mock(spec=torch.Tensor)
    ignored_loaded_cuda = Mock(spec=torch.Tensor)
    loaded_cpu.detach.return_value.cpu.return_value = expected_cpu
    loaded_cuda.detach.return_value.cpu.return_value = expected_cuda
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": loaded_cpu,
        "cuda": [loaded_cuda, ignored_loaded_cuda],
    }

    with patch(
        "force_aware_act.act_aligned_training.checkpoint."
        "torch.set_rng_state"
    ) as set_cpu, patch(
        "force_aware_act.act_aligned_training.checkpoint."
        "torch.cuda.is_available",
        return_value=True,
    ), patch(
        "force_aware_act.act_aligned_training.checkpoint."
        "torch.cuda.device_count",
        return_value=1,
    ), patch(
        "force_aware_act.act_aligned_training.checkpoint."
        "torch.cuda.set_rng_state_all"
    ) as set_cuda:
        _restore_rng_state(state)

    set_cpu.assert_called_once_with(expected_cpu)
    set_cuda.assert_called_once_with([expected_cuda])
    ignored_loaded_cuda.detach.assert_not_called()


def test_checkpoint_loader_normalizes_dataloader_generator_state(tmp_path):
    model = ACTAlignedContactCVAEPolicy(_model_config())
    config = ACTAlignedTrainingConfig()
    optimizer = build_act_aligned_optimizer(model, config)
    path = tmp_path / "checkpoint.pt"
    save_act_aligned_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        training_config=config,
        progress=TrainingProgress(1, 2, 0.5),
        normalization=_stats(),
        split_manifest=_manifest(),
        dataloader_generator=torch.Generator().manual_seed(9),
    )
    expected = torch.load(path, weights_only=False)[
        "dataloader_generator_state"
    ]

    loaded = load_act_aligned_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        training_config=config,
        restore_rng=False,
    )

    assert loaded.dataloader_generator_state is not None
    assert loaded.dataloader_generator_state.device.type == "cpu"
    assert loaded.dataloader_generator_state.dtype is torch.uint8
    torch.testing.assert_close(loaded.dataloader_generator_state, expected)


def test_checkpoint_rejects_mismatched_training_config(tmp_path):
    model = ACTAlignedContactCVAEPolicy(_model_config())
    config = ACTAlignedTrainingConfig()
    optimizer = build_act_aligned_optimizer(model, config)
    path = tmp_path / "checkpoint.pt"
    save_act_aligned_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        training_config=config,
        progress=TrainingProgress(0, 0, float("inf")),
        normalization=_stats(),
        split_manifest=_manifest(),
    )

    with pytest.raises(ValueError, match="training_config"):
        load_act_aligned_checkpoint(
            path,
            model=model,
            optimizer=optimizer,
            training_config=ACTAlignedTrainingConfig(prior_match_weight=2.0),
        )


def test_checkpoint_v2_records_cursor_and_reader_migrates_v1(tmp_path):
    model = ACTAlignedContactCVAEPolicy(_model_config())
    config = ACTAlignedTrainingConfig()
    optimizer = build_act_aligned_optimizer(model, config)
    path = tmp_path / "checkpoint.pt"
    save_act_aligned_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        training_config=config,
        progress=TrainingProgress(2, 9, 0.5, step_in_epoch=3),
        normalization=_stats(),
        split_manifest=_manifest(),
    )
    payload = torch.load(path, weights_only=False)

    assert payload["format_version"] == CHECKPOINT_FORMAT_VERSION
    assert payload["progress"]["step_in_epoch"] == 3

    payload["format_version"] = "act_aligned_checkpoint_v1"
    payload["progress"].pop("step_in_epoch")
    torch.save(payload, path)
    loaded = load_act_aligned_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        training_config=config,
        restore_rng=False,
    )

    assert loaded.progress == TrainingProgress(2, 9, 0.5, step_in_epoch=0)


def test_reader_migrates_v1_training_schedule_to_step_semantics(tmp_path):
    model = ACTAlignedContactCVAEPolicy(_model_config())
    config = ACTAlignedTrainingConfig()
    optimizer = build_act_aligned_optimizer(model, config)
    path = tmp_path / "legacy_training.pt"
    save_act_aligned_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        training_config=config,
        progress=TrainingProgress(0, 200, 0.3, step_in_epoch=200),
        normalization=_stats(),
        split_manifest=_manifest(),
    )
    payload = torch.load(path, weights_only=False)
    legacy = dict(payload["training_config"])
    legacy["training_version"] = (
        "act_aligned_conditional_cvae_training_v1"
    )
    legacy["num_epochs"] = legacy.pop("official_reference_epochs")
    legacy["checkpoint_interval"] = 100
    legacy.pop("reference_train_episodes")
    legacy.pop("max_optimizer_steps")
    legacy.pop("checkpoint_interval_steps")
    payload["training_version"] = legacy["training_version"]
    payload["training_config"] = legacy
    torch.save(payload, path)

    migrated_config = ACTAlignedTrainingConfig(
        reference_train_episodes=1,
    )
    loaded = load_act_aligned_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        training_config=migrated_config,
        restore_rng=False,
    )

    assert loaded.progress.global_step == 200
    assert migrated_config.max_optimizer_steps == 2000
