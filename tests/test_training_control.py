from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from force_aware_act.training import (
    EarlyStoppingState,
    compute_steps_per_epoch,
    evaluate_deployment_metrics,
    resolve_validation_deployment_mode,
    validate_disjoint_episode_splits,
    validate_normalization_training_episodes,
)


class _ValidationDataset(Dataset):
    def __len__(self) -> int:
        return 3

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        value = float(index + 1)
        return {
            "images": torch.zeros(1, 3, 2, 2),
            "qpos": torch.zeros(7),
            "force_window": torch.zeros(2, 6),
            "action_chunk": torch.full((2, 7), value),
            "future_force_chunk": torch.full((2, 6), 2.0),
        }


class _ZeroDeploymentModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.mixed_mode_child = torch.nn.Dropout()
        self.grad_enabled_during_forward: list[bool] = []

    def forward(self, **kwargs):
        self.grad_enabled_during_forward.append(torch.is_grad_enabled())
        batch_size = kwargs["qpos"].shape[0]
        device = kwargs["qpos"].device
        return {
            "pred_action": torch.zeros(batch_size, 2, 7, device=device) + self.anchor,
            "pred_force": torch.zeros(batch_size, 2, 6, device=device) + self.anchor,
        }


def test_steps_per_epoch_uses_partial_final_batch():
    assert compute_steps_per_epoch(29_977, 16) == 1_874


def test_early_stopping_uses_relative_delta_and_patience():
    state = EarlyStoppingState(patience=2, min_epochs=2, min_delta=0.01)

    assert state.update(1.0, epoch=1, step=10) == (True, False)
    assert state.update(0.995, epoch=2, step=20) == (False, False)
    assert state.update(0.994, epoch=3, step=30) == (False, True)
    assert state.best_metric == 1.0
    assert state.best_epoch == 1


def test_deployment_mode_resolution_is_variant_aware():
    assert resolve_validation_deployment_mode(
        policy_variant="force_aware_contact_cvae",
        requested_mode="auto",
        train_latent_mode="posterior",
        lambda_prior=0.1,
    ) == "prior"
    assert resolve_validation_deployment_mode(
        policy_variant="force_aware_act",
        requested_mode="auto",
        train_latent_mode="zero",
        lambda_prior=0.1,
    ) == "zero"
    assert resolve_validation_deployment_mode(
        policy_variant="act_baseline",
        requested_mode="auto",
    ) == "zero"


def test_episode_split_overlap_is_rejected(tmp_path):
    shared = tmp_path / "shared.hdf5"
    with pytest.raises(ValueError, match="overlap_count=1"):
        validate_disjoint_episode_splits([shared], [shared])


def test_normalization_episode_provenance_must_match_training_split(tmp_path):
    train = tmp_path / "train.hdf5"
    other = tmp_path / "other.hdf5"
    validate_normalization_training_episodes({"episode_paths": [str(train)]}, [train])
    with pytest.raises(ValueError, match="do not match"):
        validate_normalization_training_episodes({"episode_paths": [str(other)]}, [train])


def test_validation_is_sample_weighted_deterministic_and_restores_train_mode():
    model = _ZeroDeploymentModel()
    model.train()
    metrics = evaluate_deployment_metrics(
        model=model,
        dataloader=DataLoader(_ValidationDataset(), batch_size=2, shuffle=False),
        device=torch.device("cpu"),
        policy_variant="force_aware_motion_cvae",
        deployment_mode="zero",
        normalization_stats=None,
        lambda_force=0.1,
    )

    assert metrics["action_l1"] == pytest.approx(2.0)
    assert metrics["force_l1"] == pytest.approx(2.0)
    assert metrics["deploy_loss"] == pytest.approx(2.2)
    assert metrics["num_samples"] == 3
    assert model.training is True
    assert model.grad_enabled_during_forward == [False, False]


def test_validation_restores_mixed_submodule_training_states():
    model = _ZeroDeploymentModel()
    model.eval()
    model.mixed_mode_child.train()
    evaluate_deployment_metrics(
        model=model,
        dataloader=DataLoader(_ValidationDataset(), batch_size=3),
        device=torch.device("cpu"),
        policy_variant="force_aware_motion_cvae",
        deployment_mode="zero",
        normalization_stats=None,
        lambda_force=0.1,
    )

    assert model.training is False
    assert model.mixed_mode_child.training is True

