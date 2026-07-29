import random

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
    torch.testing.assert_close(
        loaded.dataloader_generator_state,
        generator.get_state(),
    )


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
