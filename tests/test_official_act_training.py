import random
from unittest.mock import Mock, patch

import h5py
import numpy as np
import pytest
import torch

pytest.importorskip("torchvision")

from force_aware_act.act_aligned_training.split import EpisodeRecord
from force_aware_act.models.official_act import (
    OfficialACTConfig,
    OfficialACTPolicy,
)
from force_aware_act.official_act_training import (
    OfficialACTBatch,
    OfficialACTCriterion,
    OfficialACTEpisodicDataset,
    OfficialACTNormalizationStats,
    OfficialACTSplitManifest,
    OfficialACTTrainingConfig,
    build_official_act_optimizer,
    collate_official_act,
    evaluate_official_act_batch,
    official_masked_l1,
    load_official_act_checkpoint,
    save_official_act_checkpoint,
    train_official_act_step,
)
from force_aware_act.official_act_training.data import (
    OFFICIAL_ACT_STATS_VERSION,
)
from force_aware_act.official_act_training.checkpoint import _restore_rng


def _stats():
    return OfficialACTNormalizationStats(
        format_version=OFFICIAL_ACT_STATS_VERSION,
        qpos_mean=(0.0,) * 7,
        qpos_std=(1.0,) * 7,
        action_mean=(0.0,) * 7,
        action_std=(1.0,) * 7,
    )


def _write_episode(path, *, num_steps=8):
    path.parent.mkdir(parents=True)
    with h5py.File(path, "w") as handle:
        handle.create_dataset(
            "observations/joint_pos",
            data=np.arange(num_steps * 7, dtype=np.float32).reshape(
                num_steps,
                7,
            ),
        )
        handle.create_dataset(
            "action",
            data=np.linspace(
                -1,
                1,
                num_steps * 7,
                dtype=np.float32,
            ).reshape(num_steps, 7),
        )
        for camera_index, name in enumerate(("cam_a", "cam_b")):
            handle.create_dataset(
                f"observations/images/{name}",
                data=np.full(
                    (num_steps, 64, 64, 3),
                    20 + camera_index,
                    dtype=np.uint8,
                ),
            )


def _record():
    return EpisodeRecord(
        episode_id="episode",
        relative_hdf5_path="episode/episode.hdf5",
        num_steps=8,
        camera_names=("cam_a", "cam_b"),
    )


def _batch(config, batch_size=2):
    mask = torch.zeros(batch_size, config.chunk_len, dtype=torch.bool)
    mask[:, -2:] = True
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
        padding_mask=mask,
    )


def test_official_loss_keeps_padding_in_mean_denominator():
    prediction = torch.ones(1, 2, 1)
    target = torch.zeros_like(prediction)
    mask = torch.tensor([[False, True]])

    loss = official_masked_l1(prediction, target, mask)

    assert loss.item() == pytest.approx(0.5)


def test_episodic_dataset_has_one_item_per_episode_and_epoch_sampling(tmp_path):
    _write_episode(tmp_path / "episode" / "episode.hdf5")
    config = OfficialACTConfig.compact_smoke()
    dataset = OfficialACTEpisodicDataset(
        tmp_path,
        [_record()],
        _stats(),
        config,
        sampling_seed=0,
        stream_id=0,
    )
    first_timestep = dataset.sampled_timestep(0)
    first = dataset[0]
    dataset.set_epoch(1)
    second_timestep = dataset.sampled_timestep(0)

    assert len(dataset) == 1
    assert 0 <= first_timestep < 8
    assert 0 <= second_timestep < 8
    assert first.images.shape == (2, 3, 64, 64)
    assert first.action_chunk.shape == (6, 7)
    assert (~first.padding_mask).sum().item() == min(
        6,
        8 - first_timestep,
    )


def test_training_step_and_validation_use_official_and_zero_paths():
    torch.manual_seed(5)
    model_config = OfficialACTConfig.compact_smoke()
    training_config = OfficialACTTrainingConfig(num_epochs=2)
    model = OfficialACTPolicy(model_config)
    criterion = OfficialACTCriterion(training_config)
    optimizer = build_official_act_optimizer(model, training_config)
    batch = _batch(model_config)
    before = model.action_head.weight.detach().clone()

    train_metrics = train_official_act_step(
        model,
        criterion,
        optimizer,
        batch,
        training_config,
    )
    validation = evaluate_official_act_batch(model, criterion, batch)

    assert not torch.equal(model.action_head.weight, before)
    assert train_metrics["loss_total"] > 0
    assert train_metrics["loss_kl"] >= 0
    assert train_metrics["gradient_norm"] > 0
    assert validation["official_sampled_validation_loss"] > 0
    assert validation["deployment_zero_action_l1"] >= 0
    assert validation["posterior_zero_action_delta"] >= 0


def test_collate_preserves_official_batch_contract():
    config = OfficialACTConfig.compact_smoke()
    first = _batch(config, batch_size=1)
    second = _batch(config, batch_size=1)

    combined = collate_official_act(
        [
            OfficialACTBatch(
                images=first.images[0],
                qpos=first.qpos[0],
                action_chunk=first.action_chunk[0],
                padding_mask=first.padding_mask[0],
            ),
            OfficialACTBatch(
                images=second.images[0],
                qpos=second.qpos[0],
                action_chunk=second.action_chunk[0],
                padding_mask=second.padding_mask[0],
            ),
        ]
    )

    combined.validate(config)
    assert combined.batch_size == 2


def test_official_checkpoint_round_trip_is_strict(tmp_path):
    model_config = OfficialACTConfig.compact_smoke()
    training_config = OfficialACTTrainingConfig(num_epochs=2)
    model = OfficialACTPolicy(model_config)
    optimizer = build_official_act_optimizer(model, training_config)
    manifest = OfficialACTSplitManifest(
        format_version="official_act_episode_split_v1",
        data_root=str(tmp_path),
        split_seed=1,
        validation_fraction=0.5,
        train_episodes=(_record(),),
        validation_episodes=(_record(),),
    )
    path = tmp_path / "checkpoint.pt"
    expected = model.action_head.weight.detach().clone()

    save_official_act_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        training_config=training_config,
        normalization=_stats(),
        split_manifest=manifest,
        epoch=1,
        global_step=2,
        best_metric=0.5,
    )
    with torch.no_grad():
        model.action_head.weight.add_(10)
    payload = load_official_act_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        training_config=training_config,
        map_location="cpu",
    )

    torch.testing.assert_close(model.action_head.weight, expected)
    assert payload["progress"] == {
        "epoch": 1,
        "global_step": 2,
        "best_metric": 0.5,
        "best_epoch": -1,
    }


def test_rng_restore_moves_loaded_cpu_and_cuda_states_back_to_cpu():
    expected_cpu = torch.get_rng_state()
    expected_cuda = torch.arange(8, dtype=torch.uint8)
    loaded_cpu = Mock(spec=torch.Tensor)
    loaded_cuda = Mock(spec=torch.Tensor)
    loaded_cpu.detach.return_value.cpu.return_value = expected_cpu
    loaded_cuda.detach.return_value.cpu.return_value = expected_cuda
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": loaded_cpu,
        "cuda": [loaded_cuda],
    }

    with patch(
        "force_aware_act.official_act_training.checkpoint."
        "torch.set_rng_state"
    ) as set_cpu, patch(
        "force_aware_act.official_act_training.checkpoint."
        "torch.cuda.is_available",
        return_value=True,
    ), patch(
        "force_aware_act.official_act_training.checkpoint."
        "torch.cuda.set_rng_state_all"
    ) as set_cuda:
        _restore_rng(state)

    set_cpu.assert_called_once_with(expected_cpu)
    set_cuda.assert_called_once_with([expected_cuda])
