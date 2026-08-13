from dataclasses import replace

import pytest
import torch

pytest.importorskip("torchvision")

from force_aware_act.act_aligned_training.split import EpisodeRecord  # noqa: E402
from force_aware_act.models.official_act import (  # noqa: E402
    OfficialACTNoLatentConfig,
    OfficialACTNoLatentPolicy,
)
from force_aware_act.official_act_training import (  # noqa: E402
    OfficialACTBatch,
    OfficialACTNoLatentCriterion,
    OfficialACTNoLatentTrainingConfig,
    OfficialACTNormalizationStats,
    OfficialACTSplitManifest,
    build_official_act_no_latent_optimizer,
    evaluate_official_act_no_latent_batch,
    load_official_act_no_latent_checkpoint,
    official_masked_l1,
    read_official_act_no_latent_checkpoint,
    save_official_act_no_latent_checkpoint,
    train_official_act_no_latent_step,
)


def _model_config():
    return OfficialACTNoLatentConfig.compact_smoke(
        chunk_len=3,
        image_height=32,
        image_width=32,
    )


def _batch(config, batch_size=2):
    return OfficialACTBatch(
        images=torch.rand(
            batch_size,
            config.num_cameras,
            3,
            config.image_height,
            config.image_width,
        ),
        qpos=torch.randn(batch_size, config.q_dim),
        action_chunk=torch.randn(
            batch_size,
            config.chunk_len,
            config.action_dim,
        ),
        padding_mask=torch.tensor(
            [[False, False, True], [False, True, True]],
            dtype=torch.bool,
        )[:batch_size],
    )


def _normalization():
    return OfficialACTNormalizationStats(
        format_version="official_act_normalization_v1",
        qpos_mean=(0.0,) * 7,
        qpos_std=(1.0,) * 7,
        action_mean=(0.0,) * 7,
        action_std=(1.0,) * 7,
    )


def _split():
    record = EpisodeRecord(
        episode_id="episode_000",
        relative_hdf5_path="episode_000/episode.hdf5",
        num_steps=4,
        camera_names=("front", "wrist"),
    )
    return OfficialACTSplitManifest(
        format_version="official_act_episode_split_v1",
        data_root="/data",
        split_seed=1,
        validation_fraction=0.5,
        train_episodes=(record,),
        validation_episodes=(record,),
    )


def test_no_latent_objective_is_only_official_masked_action_l1():
    config = OfficialACTNoLatentTrainingConfig(action_loss_weight=2.5)
    criterion = OfficialACTNoLatentCriterion(config)
    prediction = torch.tensor([[[1.0], [100.0]]])
    target = torch.zeros_like(prediction)
    mask = torch.tensor([[False, True]])

    losses = criterion({"pred_action": prediction}, target, mask)

    expected = official_masked_l1(prediction, target, mask)
    torch.testing.assert_close(losses["loss_l1"], expected)
    torch.testing.assert_close(losses["loss_total"], 2.5 * expected)
    assert set(losses) == {"loss_total", "loss_l1"}
    metadata = config.checkpoint_metadata()
    assert metadata["latent_mechanism"] == "none"
    assert "kl" not in metadata["objective"].lower()


def test_no_latent_training_step_updates_model_without_kl_metrics():
    torch.manual_seed(3)
    model_config = _model_config()
    training_config = OfficialACTNoLatentTrainingConfig(
        learning_rate=1.0e-3,
        backbone_learning_rate=1.0e-3,
        batch_size=2,
        num_epochs=1,
        checkpoint_interval_epochs=1,
    )
    model = OfficialACTNoLatentPolicy(model_config)
    criterion = OfficialACTNoLatentCriterion(training_config)
    optimizer = build_official_act_no_latent_optimizer(
        model,
        training_config,
    )
    before = model.action_head.weight.detach().clone()

    metrics = train_official_act_no_latent_step(
        model,
        criterion,
        optimizer,
        _batch(model_config),
        training_config,
    )

    assert torch.isfinite(torch.tensor(list(metrics.values()))).all()
    assert metrics["gradient_norm"] > 0.0
    assert not torch.equal(before, model.action_head.weight.detach())
    assert all("kl" not in name and "posterior" not in name for name in metrics)
    assert {group["name"] for group in optimizer.param_groups} == {
        "main",
        "backbone",
    }


def test_no_latent_validation_is_deterministic_and_restores_mode():
    torch.manual_seed(5)
    model_config = _model_config()
    training_config = OfficialACTNoLatentTrainingConfig()
    model = OfficialACTNoLatentPolicy(model_config).train()
    criterion = OfficialACTNoLatentCriterion(training_config)
    batch = _batch(model_config)

    first = evaluate_official_act_no_latent_batch(model, criterion, batch)
    second = evaluate_official_act_no_latent_batch(model, criterion, batch)

    assert first == second
    assert model.training
    assert set(first) == {
        "no_latent_validation_action_l1",
        "no_latent_validation_loss",
    }


def test_no_latent_checkpoint_round_trip_is_strict(tmp_path):
    model_config = _model_config()
    training_config = OfficialACTNoLatentTrainingConfig(
        batch_size=2,
        num_epochs=1,
        checkpoint_interval_epochs=1,
    )
    model = OfficialACTNoLatentPolicy(model_config)
    optimizer = build_official_act_no_latent_optimizer(
        model,
        training_config,
    )
    path = tmp_path / "checkpoint.pt"
    save_official_act_no_latent_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        training_config=training_config,
        normalization=_normalization(),
        split_manifest=_split(),
        epoch=2,
        global_step=9,
        best_metric=0.25,
        best_epoch=1,
    )
    payload = read_official_act_no_latent_checkpoint(path)
    reloaded = OfficialACTNoLatentPolicy(model_config)
    reloaded_optimizer = build_official_act_no_latent_optimizer(
        reloaded,
        training_config,
    )

    loaded = load_official_act_no_latent_checkpoint(
        path,
        model=reloaded,
        optimizer=reloaded_optimizer,
        training_config=training_config,
        map_location="cpu",
    )

    assert loaded["progress"] == payload["progress"]
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, reloaded.state_dict()[name])
    mismatched = replace(training_config, learning_rate=2.0e-5)
    with pytest.raises(ValueError, match="training config mismatch"):
        load_official_act_no_latent_checkpoint(
            path,
            model=reloaded,
            optimizer=reloaded_optimizer,
            training_config=mismatched,
            map_location="cpu",
        )
