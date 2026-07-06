import csv
import math
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

pytest.importorskip("torchvision")

from force_aware_act.models import ACTPolicyBaseline
from scripts import evaluate_act_baseline_modes as evaluator


def _make_policy(chunk_len: int = 4) -> ACTPolicyBaseline:
    return ACTPolicyBaseline(
        d_model=32,
        z_dim=8,
        q_dim=7,
        action_dim=7,
        chunk_len=chunk_len,
        nhead=4,
        num_encoder_layers=1,
        num_decoder_layers=1,
        dim_feedforward=64,
        dropout=0.0,
        pretrained_resnet18=False,
    )


def _make_batch(batch_size: int = 2, chunk_len: int = 4) -> dict[str, torch.Tensor]:
    return {
        "images": torch.randn(batch_size, 2, 3, 64, 64),
        "qpos": torch.randn(batch_size, 7),
        "action_chunk": torch.randn(batch_size, chunk_len, 7),
    }


def _write_episode(path: Path) -> None:
    n_state = 12
    state_ts = np.arange(n_state, dtype=np.float32) / 30.0
    image_ts = state_ts.copy()
    joint_pos = np.arange(n_state * 7, dtype=np.float32).reshape(n_state, 7) * 0.01
    images = np.zeros((n_state, 8, 8, 3), dtype=np.uint8)

    with h5py.File(path, "w") as handle:
        timestamps = handle.create_group("timestamps")
        timestamps.create_dataset("state_episode", data=state_ts)
        timestamps.create_dataset("image_episode", data=image_ts)
        observations = handle.create_group("observations")
        observations.create_dataset("joint_pos", data=joint_pos)
        observations.create_dataset("joint_vel", data=joint_pos + 1.0)
        observations.create_dataset("joint_torque", data=joint_pos + 2.0)
        observations.create_dataset("ee_pose", data=joint_pos + 3.0)
        image_group = observations.create_group("images")
        image_group.create_dataset("ee_cam", data=images)
        image_group.create_dataset("base_top_cam", data=images)
        handle.create_dataset("action", data=joint_pos + 10.0)


def _write_stats(path: Path) -> None:
    torch.save(
        {
            "qpos_mean": torch.zeros(7),
            "qpos_std": torch.ones(7),
            "action_mean": torch.zeros(7),
            "action_std": torch.ones(7),
            "action_mode": "action",
        },
        path,
    )


def _checkpoint_config(chunk_len: int = 4) -> dict[str, object]:
    return {
        "policy_variant": "act_baseline",
        "act_baseline_version": ACTPolicyBaseline.act_baseline_version,
        "uses_force": False,
        "uses_contact_latent": False,
        "motion_latent_mode": "posterior_train_zero_deploy",
        "model": {
            "pretrained_resnet18": False,
            "freeze_resnet18": False,
            "d_model": 32,
            "z_dim": 8,
            "q_dim": 7,
            "action_dim": 7,
            "chunk_len": chunk_len,
            "nhead": 4,
            "num_encoder_layers": 1,
            "num_decoder_layers": 1,
            "dim_feedforward": 64,
            "dropout": 0.0,
        },
    }


def _write_checkpoint(path: Path, chunk_len: int = 4, *, legacy: bool = False) -> None:
    model = _make_policy(chunk_len=chunk_len)
    config = _checkpoint_config(chunk_len=chunk_len)
    if legacy:
        config.pop("act_baseline_version")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": {},
            "step": 1,
            "config": config,
        },
        path,
    )


def test_run_act_modes_zero_inference_uses_no_action_labels_and_posterior_mean_override():
    class CapturingACTPolicy(ACTPolicyBaseline):
        def __init__(self):
            super().__init__(
                d_model=32,
                z_dim=8,
                q_dim=7,
                action_dim=7,
                chunk_len=4,
                nhead=4,
                num_encoder_layers=1,
                num_decoder_layers=1,
                dim_feedforward=64,
                dropout=0.0,
                pretrained_resnet18=False,
            )
            self.forward_calls = []

        def forward(
            self,
            images,
            qpos,
            action_chunk=None,
            is_training=True,
            motion_latent_override=None,
        ):
            self.forward_calls.append(
                {
                    "action_chunk": action_chunk,
                    "is_training": is_training,
                    "has_override": motion_latent_override is not None,
                }
            )
            return super().forward(
                images=images,
                qpos=qpos,
                action_chunk=action_chunk,
                is_training=is_training,
                motion_latent_override=motion_latent_override,
            )

    model = CapturingACTPolicy()
    model.eval()
    batch = _make_batch()

    outputs = evaluator.run_act_modes(model, batch, posterior_mode="mean")

    assert [call["action_chunk"] for call in model.forward_calls] == [None, None]
    assert [call["is_training"] for call in model.forward_calls] == [False, False]
    assert [call["has_override"] for call in model.forward_calls] == [False, True]
    torch.testing.assert_close(outputs.zero["z_motion"], torch.zeros_like(outputs.zero["z_motion"]))
    torch.testing.assert_close(outputs.z_motion_posterior, outputs.mu_motion)
    torch.testing.assert_close(outputs.posterior["z_motion"], outputs.mu_motion)


def test_act_mode_metrics_are_numerically_correct_and_finite():
    outputs = evaluator.ACTModeOutputs(
        zero={
            "pred_action": torch.tensor([[[1.0, 3.0]], [[3.0, 5.0]]]),
            "z_motion": torch.zeros(2, 2),
        },
        posterior={
            "pred_action": torch.tensor([[[0.0, 2.0]], [[1.0, 1.0]]]),
            "z_motion": torch.ones(2, 2),
        },
        mu_motion=torch.tensor([[0.0, 0.0], [1.0, 2.0]]),
        logvar_motion=torch.zeros(2, 2),
        z_motion_posterior=torch.tensor([[0.0, 0.0], [1.0, 2.0]]),
    )
    metrics = evaluator.compute_sample_metrics(
        outputs,
        action_target=torch.zeros(2, 1, 2),
    )

    torch.testing.assert_close(metrics["action_l1_zero"], torch.tensor([2.0, 4.0]))
    torch.testing.assert_close(metrics["action_l1_posterior"], torch.tensor([1.0, 1.0]))
    torch.testing.assert_close(
        metrics["pred_action_zero_posterior_mean_abs_diff"],
        torch.tensor([1.0, 3.0]),
    )
    for key in evaluator.METRIC_COLUMNS:
        assert torch.isfinite(metrics[key]).all()


def test_checkpoint_dispatch_requires_corrected_act_cvae_marker(tmp_path):
    checkpoint_path = tmp_path / "checkpoint.pt"
    stats_path = tmp_path / "stats.pt"
    _write_checkpoint(checkpoint_path)
    args = evaluator.parse_args(
        [
            "--checkpoint",
            str(checkpoint_path),
            "--normalization-stats",
            str(stats_path),
            "--chunk-len",
            "4",
        ]
    )

    model = evaluator.load_act_baseline_checkpoint(checkpoint_path, args, torch.device("cpu"))

    assert isinstance(model, ACTPolicyBaseline)
    assert model.chunk_len == 4

    legacy_path = tmp_path / "legacy.pt"
    _write_checkpoint(legacy_path, legacy=True)
    with pytest.raises(ValueError, match="Legacy zero-latent"):
        evaluator.load_act_baseline_checkpoint(legacy_path, args, torch.device("cpu"))


def test_csv_output_episode_identifier_and_end_to_end_cpu_smoke(tmp_path, capsys):
    episode_dir = tmp_path / "episode_000123"
    episode_dir.mkdir()
    episode_path = episode_dir / "episode.hdf5"
    episode_list = tmp_path / "episodes.txt"
    checkpoint_path = tmp_path / "checkpoint.pt"
    stats_path = tmp_path / "stats.pt"
    output_csv = tmp_path / "nested" / "act_eval.csv"
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
            "--image-size",
            "64",
            "64",
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
    assert "ACT baseline inference mode evaluation" in captured.out
    assert "posterior_mode=mean" in captured.out
    assert output_csv.is_file()
    with output_csv.open(newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))
    assert len(rows) == 4
    assert set(evaluator.CSV_COLUMNS).issubset(rows[0])
    for row in rows:
        assert row["episode_path"] == str(episode_path)
        assert row["episode_identifier"] == "episode_000123"
        assert row["timestep_index"] != ""
        for column in evaluator.METRIC_COLUMNS:
            assert math.isfinite(float(row[column]))
