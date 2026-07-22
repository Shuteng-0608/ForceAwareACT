import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from force_aware_act.training.control import (
    RetentionGatedCheckpointSelector,
    evaluate_deployment_metrics,
    evaluate_named_deployment_metrics,
    flatten_named_deployment_metrics,
)


class _ConstantValidationDataset(Dataset):
    def __init__(self, value, length=2):
        self.value = float(value)
        self.length = int(length)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return {
            "images": torch.zeros(1, 3, 2, 2),
            "qpos": torch.zeros(7),
            "force_window": torch.zeros(2, 6),
            "action_chunk": torch.full((2, 7), self.value),
            "future_force_chunk": torch.full((2, 6), self.value * 2.0),
        }


class _ZeroDeploymentModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))

    def forward(self, **kwargs):
        batch_size = kwargs["qpos"].shape[0]
        return {
            "pred_action": torch.zeros(batch_size, 2, 7) + self.anchor,
            "pred_force": torch.zeros(batch_size, 2, 6) + self.anchor,
        }


class _UnequalEpisodeDataset(Dataset):
    samples = (("short", 0.0), ("long", 2.0), ("long", 2.0), ("long", 2.0))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        episode, value = self.samples[index]
        return {
            "images": torch.zeros(1, 3, 2, 2),
            "qpos": torch.zeros(7),
            "force_window": torch.zeros(2, 6),
            "action_chunk": torch.full((2, 7), value),
            "future_force_chunk": torch.full((2, 6), value),
            "episode_path": episode,
        }


def _evaluate_kwargs(model):
    return {
        "model": model,
        "device": torch.device("cpu"),
        "policy_variant": "force_aware_motion_cvae",
        "deployment_mode": "zero",
        "normalization_stats": None,
        "lambda_force": 0.1,
    }


def test_named_validation_keeps_domain_metrics_separate_and_restores_mode():
    model = _ZeroDeploymentModel()
    model.train()
    dataloaders = {
        "r60_retention": DataLoader(_ConstantValidationDataset(1.0), batch_size=1),
        "r2_contact": DataLoader(_ConstantValidationDataset(3.0, length=3), batch_size=2),
    }

    metrics = evaluate_named_deployment_metrics(
        dataloaders=dataloaders,
        **_evaluate_kwargs(model),
    )

    assert set(metrics) == {"r60_retention", "r2_contact"}
    assert metrics["r60_retention"]["action_l1"] == pytest.approx(1.0)
    assert metrics["r60_retention"]["force_l1"] == pytest.approx(2.0)
    assert metrics["r60_retention"]["deploy_loss"] == pytest.approx(1.2)
    assert metrics["r2_contact"]["deploy_loss"] == pytest.approx(3.6)
    assert metrics["r2_contact"]["num_samples"] == 3
    assert model.training is True


def test_single_named_domain_matches_legacy_evaluator():
    model = _ZeroDeploymentModel()
    dataloader = DataLoader(_ConstantValidationDataset(2.0), batch_size=2)
    legacy = evaluate_deployment_metrics(
        dataloader=dataloader,
        **_evaluate_kwargs(model),
    )
    named = evaluate_named_deployment_metrics(
        dataloaders={"validation": dataloader},
        **_evaluate_kwargs(model),
    )

    assert named == {"validation": legacy}


def test_episode_uniform_validation_does_not_overweight_long_episodes():
    model = _ZeroDeploymentModel()
    dataloader = DataLoader(_UnequalEpisodeDataset(), batch_size=3)
    sample_weighted = evaluate_deployment_metrics(
        dataloader=dataloader,
        aggregation="sample",
        **_evaluate_kwargs(model),
    )
    episode_uniform = evaluate_deployment_metrics(
        dataloader=dataloader,
        aggregation="episode_uniform",
        **_evaluate_kwargs(model),
    )

    assert sample_weighted["action_l1"] == pytest.approx(1.5)
    assert episode_uniform["action_l1"] == pytest.approx(1.0)
    assert episode_uniform["num_episodes"] == 2.0


def test_named_metrics_flatten_for_structured_logs():
    flattened = flatten_named_deployment_metrics(
        {
            "r60": {"deploy_loss": 1.2, "num_samples": 5.0},
            "r2": {"deploy_loss": 0.7, "num_samples": 5.0},
        }
    )

    assert flattened == {
        "r60/deploy_loss": 1.2,
        "r60/num_samples": 5.0,
        "r2/deploy_loss": 0.7,
        "r2/num_samples": 5.0,
    }


def test_retention_gate_blocks_better_contact_metric_when_broad_domain_forgets():
    selector = RetentionGatedCheckpointSelector(
        objective_domain="r2_contact",
        retention_domain="r60_retention",
        retention_baseline=1.0,
        max_relative_degradation=0.1,
        min_relative_improvement=0.05,
    )
    selected = selector.update(
        {
            "r2_contact": {"deploy_loss": 0.8},
            "r60_retention": {"deploy_loss": 1.05},
        },
        epoch=1,
        step=100,
    )
    blocked = selector.update(
        {
            "r2_contact": {"deploy_loss": 0.6},
            "r60_retention": {"deploy_loss": 1.11},
        },
        epoch=2,
        step=200,
    )

    assert selected.selected is True
    assert selected.retention_passed is True
    assert blocked.selected is False
    assert blocked.objective_improved is True
    assert blocked.retention_passed is False
    assert blocked.reason == "retention_gate_failed"
    assert selector.best_objective_value == pytest.approx(0.8)
    assert selector.best_epoch == 1


def test_retention_gate_requires_configured_relative_objective_improvement():
    selector = RetentionGatedCheckpointSelector(
        objective_domain="r2",
        retention_domain="r60",
        retention_baseline=2.0,
        max_absolute_degradation=0.2,
        min_relative_improvement=0.1,
    )
    assert selector.update(
        {"r2": {"deploy_loss": 1.0}, "r60": {"deploy_loss": 2.1}},
        epoch=1,
        step=10,
    ).selected

    too_small = selector.update(
        {"r2": {"deploy_loss": 0.95}, "r60": {"deploy_loss": 2.0}},
        epoch=2,
        step=20,
    )
    enough = selector.update(
        {"r2": {"deploy_loss": 0.89}, "r60": {"deploy_loss": 2.0}},
        epoch=3,
        step=30,
    )

    assert too_small.selected is False
    assert too_small.reason == "objective_not_improved"
    assert enough.selected is True
    assert selector.best_step == 30


def test_retention_selector_state_round_trip_preserves_best_checkpoint():
    selector = RetentionGatedCheckpointSelector(
        objective_domain="r2",
        retention_domain="r60",
        retention_baseline=1.0,
        max_relative_degradation=0.05,
    )
    selector.update(
        {"r2": {"deploy_loss": 0.7}, "r60": {"deploy_loss": 1.02}},
        epoch=4,
        step=400,
    )
    restored = RetentionGatedCheckpointSelector(
        objective_domain="r2",
        retention_domain="r60",
        retention_baseline=1.0,
        max_relative_degradation=0.05,
    )

    restored.load_state_dict(selector.state_dict())

    assert restored.checkpoint_metadata() == selector.checkpoint_metadata()


def test_retention_selector_rejects_missing_or_nonfinite_named_metrics():
    selector = RetentionGatedCheckpointSelector(
        objective_domain="r2",
        retention_domain="r60",
        retention_baseline=1.0,
    )

    with pytest.raises(ValueError, match="missing validation domain"):
        selector.update({"r2": {"deploy_loss": 0.7}}, epoch=1, step=1)
    with pytest.raises(ValueError, match="must be finite"):
        selector.update(
            {"r2": {"deploy_loss": float("nan")}, "r60": {"deploy_loss": 1.0}},
            epoch=1,
            step=1,
        )
