from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import Dataset

from scripts import train_contact_prior_stage2 as stage2


class _TinyDataset(Dataset):
    def __init__(self, *_args, **_kwargs) -> None:
        pass

    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "images": torch.zeros(1, 3, 2, 2),
            "qpos": torch.full((7,), float(index + 1)),
            "force_window": torch.zeros(2, 6),
            "action_chunk": torch.ones(2, 7),
            "future_force_chunk": torch.ones(2, 6),
        }


class _TinyPolicy(torch.nn.Module):
    def __init__(self, **_kwargs) -> None:
        super().__init__()
        self.contact_prior = torch.nn.Linear(7, 2, bias=False)
        self.frozen = torch.nn.Linear(7, 2, bias=False)

    def forward(self, *, qpos, action_chunk=None, future_force_chunk=None, **_kwargs):
        mu_prior = self.contact_prior(qpos)
        mu_contact = torch.ones_like(mu_prior)
        batch_size = qpos.shape[0]
        action_shape = (batch_size, 2, 7)
        force_shape = (batch_size, 2, 6)
        signal = mu_prior.mean(dim=-1, keepdim=True).unsqueeze(-1)
        return {
            "mu_contact_prior": mu_prior,
            "logvar_contact_prior": torch.zeros_like(mu_prior),
            "mu_contact": mu_contact,
            "logvar_contact": torch.zeros_like(mu_contact),
            "pred_action": signal.expand(action_shape),
            "pred_force": signal.expand(force_shape),
        }


def test_stage2_validation_writes_best_checkpoint_and_infers_stage1_semantics(
    tmp_path,
    monkeypatch,
):
    checkpoint_path = tmp_path / "stage1.pt"
    stats_path = tmp_path / "stats.pt"
    output_dir = tmp_path / "stage2"
    torch.save(
        {
            "model_state_dict": _TinyPolicy().state_dict(),
            "config": {
                "action_mode": "action",
                "lambda_force": 0.2,
                "model": {"chunk_len": 2},
            },
        },
        checkpoint_path,
    )
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
        stats_path,
    )
    monkeypatch.setattr(stage2, "ContactForceHDF5Dataset", _TinyDataset)
    monkeypatch.setattr(stage2, "ForceAwareACTPolicy", _TinyPolicy)
    args = SimpleNamespace(
        episode_paths=[Path("train.hdf5")],
        val_episode_paths=[Path("val.hdf5")],
        checkpoint=checkpoint_path,
        normalization_stats=stats_path,
        output_dir=output_dir,
        log_csv=output_dir / "train_log.csv",
        validation_log=output_dir / "validation_log.csv",
        max_steps=1,
        max_epochs=None,
        val_every_epochs=1,
        early_stop_patience=2,
        early_stop_min_epochs=1,
        early_stop_min_delta=0.005,
        early_stop_metric="deploy_loss",
        batch_size=1,
        learning_rate=1.0e-3,
        prior_loss_mode="mse_mu",
        action_mode=None,
        lambda_force=None,
        chunk_len=2,
        force_window_len=2,
        force_window_duration=0.1,
        image_size=(2, 2),
        camera_names=("camera",),
        device="cpu",
    )

    assert stage2.train(args) == 0

    best = torch.load(output_dir / "checkpoint_best.pt", map_location="cpu")
    final = torch.load(output_dir / "checkpoint.pt", map_location="cpu")
    assert args.action_mode == "action"
    assert args.lambda_force == 0.2
    assert best["config"]["training_stage"] == "contact_prior_stage2"
    assert best["config"]["validation_deployment_mode"] == "prior"
    assert best["stop_reason"] == "best_validation_metric"
    assert final["stop_reason"] == "max_steps"
