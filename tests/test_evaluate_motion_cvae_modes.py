import csv
import math
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

pytest.importorskip("torchvision")

from force_aware_act.models import ForceAwareACTMotionCVAEPolicy
from scripts import evaluate_motion_cvae_modes as evaluator


def _make_policy(chunk_len: int = 4) -> ForceAwareACTMotionCVAEPolicy:
    return ForceAwareACTMotionCVAEPolicy(
        d_model=32,
        z_dim=8,
        action_dim=7,
        force_dim=6,
        chunk_len=chunk_len,
        nhead=4,
        num_encoder_layers=1,
        num_decoder_layers=1,
        dim_feedforward=64,
        dropout=0.0,
        pretrained_resnet18=False,
        max_force_window_len=8,
    )


def _make_batch(batch_size: int = 2, chunk_len: int = 4) -> dict[str, torch.Tensor]:
    return {
        "images": torch.randn(batch_size, 2, 3, 224, 224),
        "qpos": torch.randn(batch_size, 7),
        "force_window": torch.randn(batch_size, 5, 6),
        "action_chunk": torch.randn(batch_size, chunk_len, 7),
        "future_force_chunk": torch.randn(batch_size, chunk_len, 6),
    }


def _write_episode(path: Path) -> None:
    n_state = 12
    n_force = 36
    state_ts = np.arange(n_state, dtype=np.float32) / 30.0
    force_ts = np.arange(n_force, dtype=np.float32) / 90.0
    image_ts = state_ts.copy()
    joint_pos = np.arange(n_state * 7, dtype=np.float32).reshape(n_state, 7) * 0.01
    joint_vel = joint_pos + 1.0
    joint_torque = joint_pos + 2.0
    ee_pose = joint_pos + 3.0
    ft_wrench = np.arange(n_force * 6, dtype=np.float32).reshape(n_force, 6) * 0.01
    action = joint_pos + 10.0
    images = np.zeros((n_state, 8, 8, 3), dtype=np.uint8)

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


def _write_stats(path: Path) -> None:
    torch.save(
        {
            "qpos_mean": torch.zeros(7),
            "qpos_std": torch.ones(7),
            "action_mean": torch.zeros(7),
            "action_std": torch.ones(7),
            "force_mean": torch.zeros(6),
            "force_std": torch.ones(6),
            "action_mode": "action",
        },
        path,
    )


def _write_checkpoint(path: Path, chunk_len: int = 4) -> None:
    model = _make_policy(chunk_len=chunk_len)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": {},
            "step": 1,
            "config": {
                "policy_variant": "force_aware_motion_cvae",
                "model": {
                    "pretrained_resnet18": False,
                    "d_model": 32,
                    "z_dim": 8,
                    "action_dim": 7,
                    "force_dim": 6,
                    "chunk_len": chunk_len,
                    "nhead": 4,
                    "num_encoder_layers": 1,
                    "num_decoder_layers": 1,
                    "dim_feedforward": 64,
                    "dropout": 0.0,
                    "max_force_window_len": 8,
                },
            },
        },
        path,
    )


def test_motion_cvae_zero_and_posterior_mean_modes_have_expected_shapes():
    model = _make_policy()
    model.eval()
    batch = _make_batch()

    outputs = evaluator.run_motion_modes(model, batch, posterior_mode="mean")

    assert outputs.zero["pred_action"].shape == (2, 4, 7)
    assert outputs.posterior["pred_action"].shape == (2, 4, 7)
    assert outputs.zero["pred_force"].shape == (2, 4, 6)
    assert outputs.posterior["pred_force"].shape == (2, 4, 6)
    assert torch.count_nonzero(outputs.zero["z_motion"]) == 0
    torch.testing.assert_close(outputs.z_motion_posterior, outputs.mu_motion)


def test_posterior_mean_mode_is_deterministic_and_uses_mu_motion():
    torch.manual_seed(7)
    model = _make_policy()
    model.eval()
    batch = _make_batch()

    first = evaluator.run_motion_modes(model, batch, posterior_mode="mean")
    second = evaluator.run_motion_modes(model, batch, posterior_mode="mean")

    torch.testing.assert_close(first.z_motion_posterior, first.mu_motion)
    torch.testing.assert_close(second.z_motion_posterior, second.mu_motion)
    torch.testing.assert_close(first.posterior["pred_action"], second.posterior["pred_action"])
    torch.testing.assert_close(first.posterior["pred_force"], second.posterior["pred_force"])


def test_zero_mode_rejects_future_labels_but_accepts_explicit_motion_override():
    model = _make_policy()
    model.eval()
    batch = _make_batch()

    with pytest.raises(ValueError, match="action_chunk must be None"):
        model(
            images=batch["images"],
            qpos=batch["qpos"],
            force_window=batch["force_window"],
            action_chunk=batch["action_chunk"],
            future_force_chunk=None,
            is_training=False,
        )

    override = torch.zeros(batch["qpos"].shape[0], model.z_dim)
    outputs = model(
        images=batch["images"],
        qpos=batch["qpos"],
        force_window=batch["force_window"],
        action_chunk=None,
        future_force_chunk=None,
        is_training=False,
        motion_latent_override=override,
    )
    torch.testing.assert_close(outputs["z_motion"], override)


def test_metric_helpers_are_numerically_correct_and_finite():
    outputs = evaluator.MotionModeOutputs(
        zero={
            "pred_action": torch.tensor([[[1.0, 3.0]], [[3.0, 5.0]]]),
            "pred_force": torch.tensor([[[1.0, 2.0, 3.0]], [[3.0, 4.0, 5.0]]]),
            "z_motion": torch.zeros(2, 2),
        },
        posterior={
            "pred_action": torch.tensor([[[0.0, 2.0]], [[1.0, 1.0]]]),
            "pred_force": torch.tensor([[[1.0, 1.0, 1.0]], [[1.0, 1.0, 1.0]]]),
            "z_motion": torch.ones(2, 2),
        },
        mu_motion=torch.tensor([[0.0, 0.0], [1.0, 2.0]]),
        logvar_motion=torch.zeros(2, 2),
        z_motion_posterior=torch.tensor([[0.0, 0.0], [1.0, 2.0]]),
    )
    metrics = evaluator.compute_sample_metrics(
        outputs,
        action_target=torch.zeros(2, 1, 2),
        force_target=torch.zeros(2, 1, 3),
    )

    torch.testing.assert_close(metrics["action_l1_zero"], torch.tensor([2.0, 4.0]))
    torch.testing.assert_close(metrics["action_l1_posterior"], torch.tensor([1.0, 1.0]))
    torch.testing.assert_close(metrics["force_l1_zero"], torch.tensor([2.0, 4.0]))
    torch.testing.assert_close(metrics["force_l1_posterior"], torch.tensor([1.0, 1.0]))
    torch.testing.assert_close(
        metrics["pred_action_zero_posterior_mean_abs_diff"],
        torch.tensor([1.0, 3.0]),
    )
    torch.testing.assert_close(
        evaluator.compute_safe_ratio(torch.tensor([1.0]), torch.tensor([0.0])),
        torch.tensor([1.0 / evaluator.RATIO_EPSILON]),
    )
    for key in ("kl_motion", "mu_motion_l2", "mu_motion_abs_mean", "logvar_motion_mean", "posterior_std_mean"):
        assert torch.isfinite(metrics[key]).all()


def test_checkpoint_dispatch_builds_motion_cvae_strictly(tmp_path):
    checkpoint_path = tmp_path / "checkpoint.pt"
    _write_checkpoint(checkpoint_path)
    args = evaluator.parse_args(
        [
            "--checkpoint",
            str(checkpoint_path),
            "--normalization-stats",
            str(tmp_path / "stats.pt"),
            "--chunk-len",
            "4",
        ]
    )

    model = evaluator.load_motion_cvae_checkpoint(checkpoint_path, args, torch.device("cpu"))

    assert isinstance(model, ForceAwareACTMotionCVAEPolicy)
    assert model.chunk_len == 4


def test_csv_output_and_end_to_end_cpu_smoke(tmp_path, capsys):
    episode_path = tmp_path / "episode.hdf5"
    episode_list = tmp_path / "episodes.txt"
    checkpoint_path = tmp_path / "checkpoint.pt"
    stats_path = tmp_path / "stats.pt"
    output_csv = tmp_path / "nested" / "motion_eval.csv"
    _write_episode(episode_path)
    episode_list.write_text(f"{episode_path}\n")
    _write_checkpoint(checkpoint_path)
    _write_stats(stats_path)

    result = evaluator.main(
        [
            "--episode-list",
            str(episode_list),
            "--checkpoint",
            str(checkpoint_path),
            "--normalization-stats",
            str(stats_path),
            "--action-mode",
            "action",
            "--batch-size",
            "2",
            "--max-batches",
            "2",
            "--chunk-len",
            "4",
            "--force-window-len",
            "5",
            "--force-window-duration",
            "0.1",
            "--image-size",
            "224",
            "224",
            "--camera-names",
            "ee_cam",
            "base_top_cam",
            "--posterior-mode",
            "mean",
            "--device",
            "cpu",
            "--output-csv",
            str(output_csv),
        ]
    )

    assert result == 0
    captured = capsys.readouterr()
    assert "Motion-CVAE inference mode evaluation" in captured.out
    assert "posterior_mode=mean" in captured.out
    assert output_csv.is_file()
    with output_csv.open(newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))
    assert len(rows) == 4
    assert set(evaluator.CSV_COLUMNS).issubset(rows[0])
    for row in rows:
        assert row["episode_path"] == str(episode_path)
        assert row["episode_identifier"] == "episode"
        assert row["timestep_index"] != ""
        for column in evaluator.METRIC_COLUMNS:
            assert math.isfinite(float(row[column]))
