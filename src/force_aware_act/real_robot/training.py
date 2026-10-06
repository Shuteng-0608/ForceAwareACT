"""Shared real-robot training profile; the existing simulation recipes stay intact."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from force_aware_act.act_aligned_training.high_rate_config import ACTAlignedHighRateTrainingConfig
from force_aware_act.act_aligned_training.high_rate_control_losses import (
    ACTAlignedHighRateDualZeroCriterion, ACTAlignedHighRateMotionCriterion,
)
from force_aware_act.act_aligned_training.high_rate_dual_zero_config import ACTAlignedHighRateDualZeroTrainingConfig
from force_aware_act.act_aligned_training.high_rate_losses import ACTAlignedHighRateCriterion
from force_aware_act.act_aligned_training.high_rate_motion_config import ACTAlignedHighRateMotionTrainingConfig
from force_aware_act.act_aligned_training.losses import masked_l1_loss, standard_normal_kl
from force_aware_act.models.act_aligned import (
    ACTAlignedHighRateConfig, ACTAlignedHighRateContactCVAEPolicy,
    ACTAlignedHighRateDualZeroPolicy, ACTAlignedHighRateMotionCVAEPolicy,
)
from force_aware_act.models.official_act import OfficialACTConfig, OfficialACTPolicy


MODEL_KINDS = ("contact", "motion", "dual-zero", "act")


@dataclass(frozen=True)
class RealRobotOptimizationConfig:
    """Optimizer/loss settings only; the runner owns the explicit step budget."""

    learning_rate: float = 1e-5
    backbone_learning_rate: float = 1e-5
    weight_decay: float = 1e-4
    action_loss_weight: float = 1.0
    force_loss_weight: float = 1.0
    high_rate_force_loss_weight: float = 0.5
    posterior_kl_weight: float = 10.0
    prior_match_weight: float = 1.0
    gradient_clip_norm: float = 1.0

    def __post_init__(self):
        for name, value in vars(self).items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{name} must be finite numeric")
            if value < 0 or (name in ("learning_rate", "backbone_learning_rate", "action_loss_weight", "gradient_clip_norm") and value == 0):
                raise ValueError(f"invalid {name}")
        if self.backbone_learning_rate > self.learning_rate:
            raise ValueError("backbone learning rate must not exceed main learning rate")


def make_model_config(kind, data_config, *, compact=False, chunk_len=100, pretrained=True):
    if kind not in MODEL_KINDS:
        raise ValueError(f"unknown model kind: {kind}")
    values = dict(
        q_dim=7, action_dim=7, num_cameras=len(data_config.camera_names),
        image_height=data_config.image_height, image_width=data_config.image_width,
        chunk_len=chunk_len, pretrained_backbone=pretrained and not compact,
        imagenet_normalize=pretrained and not compact,
    )
    if compact:
        # Keep the actual 640x480 inputs; reduce network width/depth for smoke runs.
        values.update(d_model=32, nhead=4, dim_feedforward=64,
                      encoder_layers=1, decoder_layers=1, dropout=0.0)
    if kind == "act":
        return OfficialACTConfig(**values)
    values.update(
        force_dim=7, force_sample_rate_hz=data_config.torque_rate_hz,
        policy_sample_rate_hz=data_config.action_rate_hz,
        online_force_window_len=round(data_config.history_duration_s * data_config.torque_rate_hz),
        max_online_force_intervals=round(data_config.history_duration_s * data_config.action_rate_hz),
        max_force_samples_per_interval=data_config.max_samples_per_interval,
    )
    if compact:
        values.update(local_force_dim=16, local_force_encoder_layers=1)
    factory = {"contact": ACTAlignedHighRateConfig,
               "motion": ACTAlignedHighRateConfig.motion_control,
               "dual-zero": ACTAlignedHighRateConfig.dual_zero}[kind]
    return factory(**values)


def build_model(kind: str, model_config: dict[str, Any], *, initialize_pretrained=True):
    """Loading a complete checkpoint must not download unrelated backbone weights."""
    config_type = OfficialACTConfig if kind == "act" else ACTAlignedHighRateConfig
    config = config_type(**model_config)
    classes = {"contact": ACTAlignedHighRateContactCVAEPolicy,
               "motion": ACTAlignedHighRateMotionCVAEPolicy,
               "dual-zero": ACTAlignedHighRateDualZeroPolicy, "act": OfficialACTPolicy}
    if kind not in classes:
        raise ValueError(f"unknown model kind: {kind}")
    if initialize_pretrained or not config.pretrained_backbone:
        return classes[kind](config)
    # The backbone only uses this flag during construction; restore the saved config
    # on every module that kept the temporary dataclass, including preprocessing flags.
    from dataclasses import replace
    temporary = replace(config, pretrained_backbone=False, imagenet_normalize=False)
    model = classes[kind](temporary)
    for module in model.modules():
        if getattr(module, "config", None) is temporary:
            module.config = config
    return model


class RealRobotACTCriterion(nn.Module):
    """ACT network with versioned valid-element loss for missing real labels."""

    def __init__(self, config: RealRobotOptimizationConfig):
        super().__init__()
        self.config = config

    def forward(self, outputs, target, mask):
        l1 = masked_l1_loss(outputs["pred_action"], target, mask, name="real_robot_action")
        kl = standard_normal_kl(outputs["mu_motion"], outputs["logvar_motion"])
        return {"loss_action": l1, "loss_posterior_kl": kl,
                "loss_total": self.config.action_loss_weight * l1 + self.config.posterior_kl_weight * kl}


def build_criterion(kind, settings):
    if kind == "act":
        return RealRobotACTCriterion(settings)
    values = {name: getattr(settings, name) for name in (
        "action_loss_weight", "force_loss_weight", "high_rate_force_loss_weight")}
    if kind != "dual-zero":
        values["posterior_kl_weight"] = settings.posterior_kl_weight
    if kind == "contact":
        values["prior_match_weight"] = settings.prior_match_weight
    config_type, criterion_type = {
        "contact": (ACTAlignedHighRateTrainingConfig, ACTAlignedHighRateCriterion),
        "motion": (ACTAlignedHighRateMotionTrainingConfig, ACTAlignedHighRateMotionCriterion),
        "dual-zero": (ACTAlignedHighRateDualZeroTrainingConfig, ACTAlignedHighRateDualZeroCriterion),
    }[kind]
    return criterion_type(config_type(**values))


def build_optimizer(model, settings):
    backbone = model.backbone if isinstance(model, OfficialACTPolicy) else model.vision_backbone
    backbone_parameters = [p for p in backbone.body.parameters() if p.requires_grad]
    backbone_ids = {id(p) for p in backbone_parameters}
    main = [p for p in model.parameters() if p.requires_grad and id(p) not in backbone_ids]
    return torch.optim.AdamW([
        {"name": "main", "params": main, "lr": settings.learning_rate},
        {"name": "backbone", "params": backbone_parameters, "lr": settings.backbone_learning_rate},
    ], lr=settings.learning_rate, weight_decay=settings.weight_decay)


def online_inputs(batch):
    return (batch.images, batch.qpos, batch.online_force_intervals,
            batch.online_force_relative_time, batch.online_force_sample_padding_mask,
            batch.online_force_interval_padding_mask)


def forward_training(kind, model, criterion, batch, *, sample_posterior=True):
    if kind == "act":
        # Intentionally pass no torque tensor to the ACT baseline.
        outputs = model.forward_train(batch.images, batch.qpos, batch.action_chunk,
                                      batch.action_padding_mask, sample_posterior=sample_posterior)
        return outputs, criterion(outputs, batch.action_chunk, batch.action_padding_mask)
    batch.validate(model.config)
    if kind == "contact":
        outputs = model.forward_train(
            *online_inputs(batch), batch.action_chunk, batch.future_force_intervals,
            batch.future_force_relative_time, batch.future_force_sample_padding_mask,
            batch.future_force_interval_padding_mask,
            action_padding_mask=batch.action_padding_mask, sample_posterior=sample_posterior)
    elif kind == "motion":
        outputs = model.forward_train(*online_inputs(batch), batch.action_chunk,
                                     action_padding_mask=batch.action_padding_mask,
                                     sample_posterior=sample_posterior)
    else:
        outputs = model(*online_inputs(batch))
    losses = criterion(
        outputs, batch.action_chunk, batch.future_force_target, batch.future_force_intervals,
        batch.action_padding_mask, batch.future_force_interval_padding_mask,
        batch.future_force_sample_padding_mask,
        force_target_padding_mask=batch.force_target_padding_mask)
    return outputs, losses


def train_step(kind, model, criterion, optimizer, batch, settings):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    _, losses = forward_training(kind, model, criterion, batch)
    if not torch.isfinite(losses["loss_total"]):
        raise FloatingPointError("non-finite training loss")
    losses["loss_total"].backward()
    parameters = [p for p in model.parameters() if p.requires_grad and p.grad is not None]
    if not parameters:
        raise RuntimeError("no model gradients")
    gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, settings.gradient_clip_norm,
                                                  error_if_nonfinite=True)
    optimizer.step()
    return {**{name: float(value.detach()) for name, value in losses.items()},
            "gradient_norm": float(gradient_norm)}


@torch.no_grad()
def deployment_predictions(kind, model, batch):
    if kind == "act":
        return model(batch.images, batch.qpos)
    if kind == "contact":
        return model(*online_inputs(batch), contact_latent_mode="zero")
    return model(*online_inputs(batch))


@torch.no_grad()
def evaluate(kind, model, batches, normalization, device, *, detailed=False):
    """Global valid-scalar deployment errors, never posterior-assisted validation."""
    previous = model.training
    model.eval()
    sums = {"action": 0.0, "torque": 0.0, "torque_endpoint": 0.0}
    counts = dict.fromkeys(sums, 0)
    windows = 0
    action_errors = []
    hold_sum, hold_count = 0.0, 0
    horizon_sums, horizon_counts = {}, {}
    try:
        for batch in batches:
            batch = batch.to(device)
            outputs = deployment_predictions(kind, model, batch)
            windows += batch.batch_size
            if detailed:
                error = ((outputs["pred_action"] - batch.action_chunk) * batch.action_chunk.new_tensor(normalization.action_std)).abs()
                action_errors.append(error[~batch.action_padding_mask].cpu().numpy())
                current_q = batch.qpos * batch.qpos.new_tensor(normalization.qpos_std) + batch.qpos.new_tensor(normalization.qpos_mean)
                target_q = normalization.denormalize_action(batch.action_chunk)
                hold = (current_q[:, None] - target_q).abs()[~batch.action_padding_mask]
                hold_sum += float(hold.double().sum())
                hold_count += hold.numel()
                for slot in (0, 10, 50, 99):
                    if slot < error.shape[1]:
                        valid = error[:, slot][~batch.action_padding_mask[:, slot]]
                        horizon_sums[slot] = horizon_sums.get(slot, 0.0) + float(valid.double().sum())
                        horizon_counts[slot] = horizon_counts.get(slot, 0) + valid.numel()
            entries = [("action", outputs["pred_action"], batch.action_chunk,
                        batch.action_padding_mask, normalization.action_std)]
            if kind != "act":
                entries.extend([
                    ("torque", outputs["pred_force_highrate"], batch.future_force_intervals,
                     batch.future_force_sample_padding_mask, normalization.force_std),
                    ("torque_endpoint", outputs["pred_force"], batch.future_force_target,
                     batch.force_target_padding_mask, normalization.force_std),
                ])
            for name, predicted, target, mask, std in entries:
                error = ((predicted - target) * predicted.new_tensor(std)).abs()[~mask]
                if not torch.isfinite(error).all():
                    raise FloatingPointError(f"non-finite validation {name}")
                sums[name] += float(error.double().sum())
                counts[name] += error.numel()
    finally:
        model.train(previous)
    result = {"windows": windows,
            **{f"{name}_mae_{'rad' if name == 'action' else 'nm'}":
               sums[name] / counts[name] if counts[name] else None for name in sums},
            **{f"{name}_valid_scalars": count for name, count in counts.items()}}
    if detailed:
        import numpy as np
        errors = np.concatenate(action_errors)
        result.update(action_mae_per_joint_rad=errors.mean(axis=0, dtype=np.float64).tolist(),
                      action_p95_per_joint_rad=np.quantile(errors, .95, axis=0).tolist(),
                      hold_current_qpos_mae_rad=hold_sum / hold_count,
                      horizon_action_mae_rad={str(slot): horizon_sums[slot] / horizon_counts[slot]
                                              if horizon_counts[slot] else None for slot in horizon_sums})
    return result
