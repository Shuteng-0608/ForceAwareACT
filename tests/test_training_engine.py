import pytest
import torch

from force_aware_act.training.engine import (
    normalize_training_batch,
    train_one_update,
    validate_normalization_stats,
)
from force_aware_act.training.protocol import ObjectiveSpec


class _MotionPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, **kwargs):
        batch = kwargs["qpos"].shape[0]
        latent = self.weight + torch.zeros(batch, 2)
        return {
            "pred_action": self.weight + torch.zeros(batch, 2, 7),
            "pred_force": self.weight + torch.zeros(batch, 2, 6),
            "mu_motion": latent,
            "logvar_motion": torch.zeros_like(latent),
        }


class _MotionPolicyWithVisionBatchNorm(_MotionPolicy):
    def __init__(self):
        super().__init__()
        self.vision_encoder = torch.nn.Module()
        self.vision_encoder.backbone = torch.nn.Sequential(torch.nn.BatchNorm2d(3))

    def forward(self, **kwargs):
        self.vision_encoder.backbone(kwargs["images"][:, 0])
        return super().forward(**kwargs)


def _stats():
    return {
        "qpos_mean": torch.ones(7),
        "qpos_std": torch.full((7,), 2.0),
        "action_mean": torch.ones(7),
        "action_std": torch.full((7,), 2.0),
        "force_mean": torch.ones(6),
        "force_std": torch.full((6,), 2.0),
    }


def _batch():
    return {
        "images": torch.zeros(1, 1, 3, 4, 4),
        "qpos": torch.full((1, 7), 3.0),
        "force_window": torch.full((1, 3, 6), 3.0),
        "action_chunk": torch.full((1, 2, 7), 3.0),
        "future_force_chunk": torch.full((1, 2, 6), 3.0),
        "episode_path": ["episode.hdf5"],
    }


def _objective():
    return ObjectiveSpec(
        lambda_force=0.1,
        lambda_prior=0.0,
        prior_loss_mode="mse_mu",
        beta_motion_max=0.0,
        beta_contact_max=0.0,
        warmup_steps=0,
        train_latent_mode="posterior",
        train_contact_latent_mode="posterior",
        validation_deployment_mode="zero",
    )


def test_normalization_preserves_metadata_and_normalizes_all_training_tensors():
    normalized = normalize_training_batch(_batch(), _stats())
    for key in ("qpos", "force_window", "action_chunk", "future_force_chunk"):
        assert torch.equal(normalized[key], torch.ones_like(normalized[key]))
    assert normalized["episode_path"] == ["episode.hdf5"]


def test_normalization_rejects_nonpositive_std():
    stats = _stats()
    stats["force_std"][0] = 0.0
    with pytest.raises(ValueError, match="strictly positive"):
        validate_normalization_stats(stats)


def test_train_one_update_changes_parameter_and_reports_gradient():
    model = _MotionPolicy()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    result = train_one_update(
        model=model,
        optimizer=optimizer,
        batch=_batch(),
        policy_variant="force_aware_motion_cvae",
        objective=_objective(),
        stage_step=1,
        max_grad_norm=0.5,
    )

    assert model.weight.item() > 0.0
    assert result.losses["loss_total"] == pytest.approx(3.3)
    assert result.gradient_norm is not None
    assert result.gradient_parameter_count == 1


def test_train_one_update_rejects_nonfinite_gradient_without_clipping():
    model = _MotionPolicy()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    batch = _batch()
    batch["action_chunk"][0, 0, 0] = float("nan")
    with pytest.raises(FloatingPointError, match="non-finite"):
        train_one_update(
            model=model,
            optimizer=optimizer,
            batch=batch,
            policy_variant="force_aware_motion_cvae",
            objective=_objective(),
            stage_step=1,
            max_grad_norm=None,
        )


def test_train_one_update_can_freeze_vision_batch_norm_running_stats():
    model = _MotionPolicyWithVisionBatchNorm()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    batch = _batch()
    batch["images"].fill_(5.0)
    before = model.vision_encoder.backbone[0].running_mean.clone()
    train_one_update(
        model=model,
        optimizer=optimizer,
        batch=batch,
        policy_variant="force_aware_motion_cvae",
        objective=_objective(),
        stage_step=1,
        max_grad_norm=0.5,
        freeze_vision_batch_norm=True,
    )
    assert torch.equal(model.vision_encoder.backbone[0].running_mean, before)
    assert model.vision_encoder.backbone[0].weight.grad is None
