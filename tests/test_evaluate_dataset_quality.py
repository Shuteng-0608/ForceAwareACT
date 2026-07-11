import argparse
from pathlib import Path

import h5py
import numpy as np

from scripts.evaluate_dataset_quality import evaluate_episode


def _args():
    return argparse.Namespace(
        success_status="auto_stop_task_success", max_length_mismatch=1,
        min_duration=0.1, max_duration=30.0, max_gap_factor=3.0,
        max_force=80.0, max_joint_speed=1.0, max_command_step=0.05,
        image_samples=3, min_image_std=2.0, min_frame_change=0.1, good_score=90,
    )


def _episode(path: Path, status="auto_stop_task_success"):
    n, nf = 6, 11
    with h5py.File(path, "w") as f:
        f.attrs["status"] = status
        obs = f.create_group("observations")
        obs.create_dataset("ee_pose", data=np.zeros((n, 7)))
        q = np.linspace(0, 0.02, n)[:, None] * np.ones((1, 7))
        obs.create_dataset("joint_pos", data=q)
        obs.create_dataset("joint_vel", data=np.zeros((n, 7)))
        obs.create_dataset("joint_torque", data=np.zeros((n, 7)))
        obs.create_dataset("ft_wrench", data=np.zeros((nf, 6)))
        images = obs.create_group("images")
        rng = np.random.default_rng(1)
        images.create_dataset("ee_cam", data=rng.integers(0, 255, (n, 16, 16, 3), dtype=np.uint8))
        images.create_dataset("base_top_cam", data=rng.integers(0, 255, (n, 16, 16, 3), dtype=np.uint8))
        actions = f.create_group("actions")
        actions.create_dataset("joint_pos_command", data=q)
        ts = f.create_group("timestamps")
        ts.create_dataset("state_episode", data=np.arange(n) * 0.1)
        ts.create_dataset("image_episode", data=np.arange(n) * 0.1)
        ts.create_dataset("force_episode", data=np.arange(nf) * 0.05)


def test_good_episode(tmp_path):
    directory = tmp_path / "ep"
    directory.mkdir()
    path = directory / "episode.hdf5"
    _episode(path)
    row = evaluate_episode(path, _args())
    assert row["quality"] == "good"
    assert row["n_state"] == 6
    assert row["errors"] == ""


def test_failed_collection_is_rejected(tmp_path):
    directory = tmp_path / "ep"
    directory.mkdir()
    path = directory / "episode.hdf5"
    _episode(path, status="controller_shutdown")
    row = evaluate_episode(path, _args())
    assert row["quality"] == "reject"
    assert "not success" in row["errors"]


def test_corrupt_file_is_rejected(tmp_path):
    path = tmp_path / "episode.hdf5"
    path.write_bytes(b"not hdf5")
    row = evaluate_episode(path, _args())
    assert row["quality"] == "reject"
    assert "cannot read episode" in row["errors"]
