import h5py
import numpy as np
import torch

from force_aware_act.act_aligned_training import (
    ACTAlignedHighRateBatch,
    ACTAlignedHighRateCriterion,
    ACTAlignedHighRateTrainingConfig,
    EpisodeRecord,
    EpisodeSplitManifest,
    NormalizationStats,
    TrainingProgress,
    build_act_aligned_high_rate_optimizer,
    compute_high_rate_normalization_stats,
    load_act_aligned_checkpoint,
    masked_interval_balanced_high_rate_l1_loss,
    run_high_rate_physical_action_validation_epoch,
    save_act_aligned_checkpoint,
    train_high_rate_one_step,
)
from force_aware_act.models.act_aligned import (
    ACTAlignedHighRateConfig,
    ACTAlignedHighRateContactCVAEPolicy,
)


def _model_config():
    return ACTAlignedHighRateConfig(
        d_model=32,
        nhead=4,
        dim_feedforward=64,
        local_force_dim=16,
        dropout=0.0,
        chunk_len=3,
        image_height=32,
        image_width=32,
        pretrained_backbone=False,
        imagenet_normalize=False,
    )


def _training_config(**overrides):
    values = {
        "batch_size": 2,
        "reference_train_episodes": 2,
        "official_reference_epochs": 2,
    }
    values.update(overrides)
    return ACTAlignedHighRateTrainingConfig(**values)


def _batch(config, batch_size=2):
    b, k = batch_size, config.chunk_len
    h, s, f = (
        config.max_online_force_intervals,
        config.max_force_samples_per_interval,
        config.force_dim,
    )
    online_sample_mask = torch.zeros(b, h, s, dtype=torch.bool)
    online_interval_mask = torch.zeros(b, h, dtype=torch.bool)
    future_sample_mask = torch.zeros(b, k, s, dtype=torch.bool)
    future_sample_mask[:, :, 17:] = True
    future_interval_mask = future_sample_mask.all(dim=-1)
    future_force = torch.randn(b, k, s, f).masked_fill(
        future_sample_mask.unsqueeze(-1), 0.0
    )
    return ACTAlignedHighRateBatch(
        images=torch.randn(b, config.num_cameras, 3, config.image_height, config.image_width),
        qpos=torch.randn(b, config.q_dim),
        action_chunk=torch.randn(b, k, config.action_dim),
        action_padding_mask=torch.zeros(b, k, dtype=torch.bool),
        online_force_intervals=torch.randn(b, h, s, f),
        online_force_relative_time=torch.arange(s).float().mul(0.002).view(1, 1, s).expand(b, h, s).clone(),
        online_force_sample_padding_mask=online_sample_mask,
        online_force_interval_padding_mask=online_interval_mask,
        future_force_intervals=future_force,
        future_force_relative_time=torch.arange(s).float().mul(0.002).view(1, 1, s).expand(b, k, s).clone(),
        future_force_sample_padding_mask=future_sample_mask,
        future_force_interval_padding_mask=future_interval_mask,
        future_force_target=future_force[:, :, 16].clone(),
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


def test_interval_balanced_loss_does_not_overweight_17_sample_interval():
    target = torch.zeros(1, 2, 20, 1)
    prediction = torch.zeros_like(target)
    prediction[:, 0, :16] = 1.0
    prediction[:, 1, :17] = 3.0
    sample_mask = torch.ones(1, 2, 20, dtype=torch.bool)
    sample_mask[:, 0, :16] = False
    sample_mask[:, 1, :17] = False
    interval_mask = torch.zeros(1, 2, dtype=torch.bool)

    loss = masked_interval_balanced_high_rate_l1_loss(
        prediction, target, sample_mask, interval_mask, name="test"
    )

    torch.testing.assert_close(loss, torch.tensor(2.0))


def test_all_invalid_high_rate_loss_is_differentiable_zero():
    prediction = torch.randn(1, 2, 3, 1, requires_grad=True)
    target = torch.zeros_like(prediction)
    sample_mask = torch.ones(1, 2, 3, dtype=torch.bool)
    interval_mask = torch.ones(1, 2, dtype=torch.bool)
    loss = masked_interval_balanced_high_rate_l1_loss(
        prediction, target, sample_mask, interval_mask, name="empty"
    )
    loss.backward()

    assert loss.item() == 0.0
    torch.testing.assert_close(prediction.grad, torch.zeros_like(prediction))


def test_high_rate_normalization_uses_every_native_force_sample(tmp_path):
    episode_dir = tmp_path / "episode_0"
    episode_dir.mkdir()
    path = episode_dir / "episode.hdf5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("observations/joint_pos", data=np.zeros((2, 7), np.float32))
        handle.create_dataset("action", data=np.zeros((2, 7), np.float32))
        force = np.zeros((4, 6), np.float32)
        force[1, 0] = 100.0
        handle.create_dataset("observations/ft_wrench", data=force)
    record = EpisodeRecord("episode_0", "episode_0/episode.hdf5", 2, ())

    stats = compute_high_rate_normalization_stats(tmp_path, (record,))

    assert stats.force_mean[0] == 25.0
    assert stats.force_std[0] > 40.0


def test_high_rate_train_step_updates_explicit_high_rate_head():
    torch.manual_seed(0)
    config = _model_config()
    training_config = _training_config()
    model = ACTAlignedHighRateContactCVAEPolicy(config)
    criterion = ACTAlignedHighRateCriterion(training_config)
    optimizer = build_act_aligned_high_rate_optimizer(model, training_config)
    before = model.high_rate_force_head.weight.detach().clone()

    metrics = train_high_rate_one_step(
        model, criterion, optimizer, _batch(config), training_config
    )

    assert metrics["loss_force_highrate"] > 0.0
    assert not torch.equal(before, model.high_rate_force_head.weight.detach())


def test_high_rate_physical_action_validation_uses_action_std():
    config = _model_config()
    model = ACTAlignedHighRateContactCVAEPolicy(config)
    for parameter in model.parameters():
        parameter.data.zero_()
    batch = _batch(config, batch_size=2)
    batch.action_chunk.fill_(1.0)

    metrics = run_high_rate_physical_action_validation_epoch(
        model,
        [batch],
        (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0),
        device=torch.device("cpu"),
    )

    assert metrics["deployment_zero_action_l1_physical"] == 4.0
    assert metrics["full_validation_windows"] == 2.0


def test_prior_match_cannot_update_shared_high_rate_encoder():
    config = _model_config()
    training_config = _training_config()
    model = ACTAlignedHighRateContactCVAEPolicy(config)
    criterion = ACTAlignedHighRateCriterion(training_config)
    batch = _batch(config, batch_size=1)
    outputs = model.forward_train(
        batch.images, batch.qpos,
        batch.online_force_intervals, batch.online_force_relative_time,
        batch.online_force_sample_padding_mask, batch.online_force_interval_padding_mask,
        batch.action_chunk, batch.future_force_intervals,
        batch.future_force_relative_time, batch.future_force_sample_padding_mask,
        batch.future_force_interval_padding_mask,
        action_padding_mask=batch.action_padding_mask, sample_posterior=False,
    )
    losses = criterion(
        outputs, batch.action_chunk, batch.future_force_target,
        batch.future_force_intervals, batch.action_padding_mask,
        batch.future_force_interval_padding_mask,
        batch.future_force_sample_padding_mask,
    )

    losses["loss_prior_match"].backward()

    assert all(parameter.grad is None for parameter in model.high_rate_force_encoder.parameters())
    assert all(parameter.grad is None for parameter in model.contact_posterior.parameters())
    assert any(parameter.grad is not None for parameter in model.contact_prior.parameters())


def test_high_rate_checkpoint_round_trip_is_strict(tmp_path):
    model_config = _model_config()
    training_config = _training_config()
    model = ACTAlignedHighRateContactCVAEPolicy(model_config)
    optimizer = build_act_aligned_high_rate_optimizer(model, training_config)
    record = EpisodeRecord("train", "train/episode.hdf5", 2, ())
    manifest = EpisodeSplitManifest(
        "act_aligned_episode_split_v1", str(tmp_path), 0, 0.0, (record,), ()
    )
    path = tmp_path / "v3.pt"
    save_act_aligned_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        training_config=training_config,
        progress=TrainingProgress(1, 1, 0.5),
        normalization=_stats(),
        split_manifest=manifest,
    )

    loaded = load_act_aligned_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        training_config=training_config,
        restore_rng=False,
    )

    assert loaded.progress.global_step == 1
