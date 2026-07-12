from pathlib import Path
import csv
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import torch

from force_aware_act.data import ContactForceHDF5Dataset, compute_normalization_stats
import scripts.train_minimal as train_minimal
from scripts.compute_normalization_stats import compute_and_save
from scripts.evaluate_inference_modes import (
    _build_evaluation_dataset,
    _validate_normalization_action_mode as validate_eval_stats_action_mode,
)
from scripts.train_minimal import (
    _build_training_dataset,
    build_checkpoint_payload,
    checkpoint_step_path,
    parse_args as parse_train_args,
    resolve_checkpoint_steps,
    save_checkpoint_atomic,
    train,
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


def test_train_latent_mode_default_is_posterior():
    args = parse_train_args(["episode.hdf5"])

    assert args.train_latent_mode == "posterior"
    assert args.save_every == 0
    assert args.save_steps == []


def test_checkpoint_schedule_default_is_empty():
    assert resolve_checkpoint_steps(max_steps=20, save_every=0, save_steps=[]) == []


def test_checkpoint_schedule_periodic_steps_include_final_divisible_step():
    assert resolve_checkpoint_steps(max_steps=20000, save_every=5000, save_steps=[]) == [
        5000,
        10000,
        15000,
        20000,
    ]


def test_checkpoint_schedule_explicit_steps():
    assert resolve_checkpoint_steps(
        max_steps=20000,
        save_every=0,
        save_steps=[3000, 5000, 8000],
    ) == [3000, 5000, 8000]


def test_checkpoint_schedule_combines_sorts_and_deduplicates():
    assert resolve_checkpoint_steps(
        max_steps=10000,
        save_every=5000,
        save_steps=[8000, 3000, 5000, 3000],
    ) == [3000, 5000, 8000, 10000]


def test_checkpoint_schedule_save_every_larger_than_max_steps_uses_explicit_only():
    assert resolve_checkpoint_steps(
        max_steps=10000,
        save_every=20000,
        save_steps=[3000],
    ) == [3000]


@pytest.mark.parametrize(
    ("save_every", "save_steps", "match"),
    [
        (-1, [], "--save-every must be non-negative"),
        (0, [0], "positive integers"),
        (0, [-5], "positive integers"),
        (0, [21], "<= --max-steps"),
    ],
)
def test_checkpoint_schedule_rejects_invalid_inputs(save_every, save_steps, match):
    with pytest.raises(ValueError, match=match):
        resolve_checkpoint_steps(max_steps=20, save_every=save_every, save_steps=save_steps)


def test_checkpoint_step_path_uses_eight_digits(tmp_path):
    assert checkpoint_step_path(tmp_path, 1).name == "checkpoint_step_00000001.pt"
    assert checkpoint_step_path(tmp_path, 5000).name == "checkpoint_step_00005000.pt"
    assert checkpoint_step_path(tmp_path, 20000).name == "checkpoint_step_00020000.pt"


def test_checkpoint_payload_and_atomic_save_round_trip_and_replace(tmp_path):
    checkpoint_path = tmp_path / "checkpoint.pt"
    temporary_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
    config = {"model": {"type": "linear"}, "max_steps": 5}

    first_payload = build_checkpoint_payload(
        model=model,
        optimizer=optimizer,
        config=config,
        step=1,
    )
    save_checkpoint_atomic(first_payload, checkpoint_path)
    assert torch.load(checkpoint_path, map_location="cpu")["step"] == 1

    second_payload = build_checkpoint_payload(
        model=model,
        optimizer=optimizer,
        config=config,
        step=5,
    )
    save_checkpoint_atomic(second_payload, checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    assert checkpoint["step"] == 5
    assert {"model_state_dict", "optimizer_state_dict", "config", "step"} <= set(checkpoint)
    assert checkpoint["config"] == config
    assert not temporary_path.exists()


def test_train_zero_latent_mode_writes_log_columns(tmp_path):
    episode_path = tmp_path / "episode.hdf5"
    output_dir = tmp_path / "train_zero"
    log_csv = output_dir / "train_log.csv"
    _write_action_mode_episode(episode_path)
    args = SimpleNamespace(
        episode_paths=[episode_path],
        camera_names=("ee_cam", "base_top_cam"),
        action_mode="action",
        train_latent_mode="zero",
        chunk_len=4,
        force_window_len=5,
        force_window_duration=0.1,
        image_size=(224, 224),
        imagenet_normalize=False,
        batch_size=1,
        num_workers=0,
        max_steps=1,
        learning_rate=1.0e-4,
        lambda_force=0.1,
        lambda_prior=0.1,
        prior_loss_mode="mse_mu",
        beta_motion_max=1.0e-4,
        beta_contact_max=1.0e-4,
        warmup_steps=100,
        save_every=0,
        save_steps=[],
        output_dir=output_dir,
        log_csv=log_csv,
        device="cpu",
        normalization_stats=None,
    )

    assert train(args) == 0
    with log_csv.open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert rows[0]["train_latent_mode"] == "zero"
    assert rows[0]["uses_posterior_latent"] == "False"
    assert rows[0]["uses_zero_latent"] == "True"
    assert float(rows[0]["kl_motion"]) == 0.0
    assert float(rows[0]["kl_contact"]) == 0.0
    checkpoint = torch.load(output_dir / "checkpoint.pt", map_location="cpu")
    assert checkpoint["config"]["policy_variant"] == "force_aware_act"
    assert checkpoint["config"]["training_seed"] == 0
    assert checkpoint["config"]["dataloader_seed"] == 1
    assert checkpoint["config"]["deterministic_enabled"] is False
    assert checkpoint["config"]["initial_model_sha256"] == checkpoint["initial_model_sha256"]
    assert checkpoint["config"]["torch_num_threads"] == checkpoint["torch_num_threads"]
    assert (
        checkpoint["config"]["torch_num_interop_threads"]
        == checkpoint["torch_num_interop_threads"]
    )
    assert args.policy_variant == "force_aware_act"


def test_train_intermediate_checkpoints_are_saved_after_optimizer_steps(
    tmp_path,
    monkeypatch,
):
    episode_path = tmp_path / "episode.hdf5"
    output_dir = tmp_path / "trajectory"
    log_csv = output_dir / "train_log.csv"
    _write_action_mode_episode(episode_path)
    optimizer_step_calls = []

    class CountingAdamW(torch.optim.AdamW):
        def step(self, closure=None):
            optimizer_step_calls.append(len(optimizer_step_calls) + 1)
            return super().step(closure=closure)

    monkeypatch.setattr(train_minimal, "AdamW", CountingAdamW)
    args = SimpleNamespace(
        episode_paths=[episode_path],
        camera_names=("ee_cam", "base_top_cam"),
        action_mode="action",
        policy_variant="force_aware_motion_cvae",
        train_latent_mode="posterior",
        chunk_len=4,
        force_window_len=5,
        force_window_duration=0.1,
        image_size=(224, 224),
        imagenet_normalize=False,
        batch_size=1,
        num_workers=0,
        max_steps=4,
        learning_rate=1.0e-4,
        lambda_force=0.1,
        lambda_prior=0.0,
        prior_loss_mode="mse_mu",
        beta_motion_max=1.0e-4,
        beta_contact_max=1.0e-4,
        warmup_steps=100,
        save_every=0,
        save_steps=[1, 2, 4],
        output_dir=output_dir,
        log_csv=log_csv,
        device="cpu",
        normalization_stats=None,
    )

    assert train(args) == 0
    assert optimizer_step_calls == [1, 2, 3, 4]

    expected_steps = {
        "checkpoint_step_00000001.pt": 1,
        "checkpoint_step_00000002.pt": 2,
        "checkpoint_step_00000004.pt": 4,
        "checkpoint.pt": 4,
    }
    checkpoints = {}
    for filename, expected_step in expected_steps.items():
        checkpoint = torch.load(output_dir / filename, map_location="cpu")
        checkpoints[filename] = checkpoint
        assert checkpoint["step"] == expected_step
        assert {"model_state_dict", "optimizer_state_dict", "config", "step"} <= set(checkpoint)

    with log_csv.open(newline="") as handle:
        assert len(list(csv.DictReader(handle))) == 4

    step4_state = checkpoints["checkpoint_step_00000004.pt"]["model_state_dict"]
    final_state = checkpoints["checkpoint.pt"]["model_state_dict"]
    assert step4_state.keys() == final_state.keys()
    for name, tensor in step4_state.items():
        assert torch.equal(tensor, final_state[name]), name
