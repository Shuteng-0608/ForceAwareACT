"""Six-axis sensor diagnostics: dimensionless training losses and separate N/N*m errors."""

import numpy as np
import torch

from force_aware_act.act_aligned_training.high_rate_losses import (
    masked_optional_l1_loss, masked_interval_balanced_high_rate_l1_loss,
)


@torch.no_grad()
def training_components(outputs, batch):
    """Use exactly the existing reductions; these detached values only add logging."""
    result = {}
    for label, section in (("force", slice(0, 3)), ("moment", slice(3, 6))):
        result[f"loss_{label}_xyz"] = masked_optional_l1_loss(
            outputs["pred_force"][..., section], batch.future_force_target[..., section],
            batch.force_target_padding_mask, name=label)
        result[f"loss_{label}_highrate_xyz"] = masked_interval_balanced_high_rate_l1_loss(
            outputs["pred_force_highrate"][..., section], batch.future_force_intervals[..., section],
            batch.future_force_sample_padding_mask, batch.future_force_interval_padding_mask, name=label)
    return result


def last_observed_wrench(batch):
    values = batch.online_force_intervals.flatten(1, 2)
    valid = ~batch.online_force_sample_padding_mask.flatten(1, 2)
    positions = torch.arange(values.shape[1], device=values.device).expand(valid.shape)
    last = positions.masked_fill(~valid, -1).amax(1)
    if bool((last < 0).any()):
        raise ValueError("wrench hold baseline requires a valid causal observation")
    return values[torch.arange(len(values), device=values.device), last]


def _summarize(sums):
    result = {}
    for key, (total, count) in sums.items():
        result[key] = float(total.sum() / (count * len(total))) if count else None
        result[key + "_valid_scalars"] = count * len(total)
        result[key + "_per_axis"] = (total / count).tolist() if count else None
    return result


@torch.no_grad()
def evaluate_wrench6(kind, model, batches, normalization, device, *, detailed=False, force_evaluation_steps=None):
    from .training import deployment_predictions
    sums = {}
    windows = 0

    def add(key, prediction, target, padding, std):
        error = ((prediction - target) * prediction.new_tensor(std)).abs()[~padding]
        if not torch.isfinite(error).all():
            raise FloatingPointError("nonfinite unmasked wrench evaluation: " + key)
        total = error.double().sum(0).cpu().numpy()
        previous, count = sums.get(key, (np.zeros_like(total), 0))
        sums[key] = (previous + total, count + len(error))

    previous = model.training
    model.eval()
    try:
        for batch in batches:
            batch = batch.to(device)
            outputs = deployment_predictions(kind, model, batch)
            windows += batch.batch_size
            add("action_mae_rad", outputs["pred_action"], batch.action_chunk,
                batch.action_padding_mask, normalization.action_std)
            current = batch.qpos * batch.qpos.new_tensor(normalization.qpos_std) + batch.qpos.new_tensor(normalization.qpos_mean)
            hold_q = (current - current.new_tensor(normalization.action_mean)) / current.new_tensor(normalization.action_std)
            add("hold_current_qpos_mae_rad", hold_q[:, None].expand_as(batch.action_chunk), batch.action_chunk,
                batch.action_padding_mask, normalization.action_std)
            if detailed:
                for slot in (0, 10, 19, 50, 99):
                    if slot < batch.action_chunk.shape[1]:
                        add(f"action_step{slot}_mae_rad", outputs["pred_action"][:, slot:slot+1],
                            batch.action_chunk[:, slot:slot+1], batch.action_padding_mask[:, slot:slot+1], normalization.action_std)
            if kind == "act":
                continue
            hold = last_observed_wrench(batch)
            count = force_evaluation_steps or model.config.force_prediction_len
            for suffix, prediction, target, padding in (
                ("", outputs["pred_force_highrate"], batch.future_force_intervals, batch.future_force_sample_padding_mask),
                ("_endpoint", outputs["pred_force"], batch.future_force_target, batch.force_target_padding_mask),
            ):
                prediction, target, padding = prediction[:, :count], target[:, :count], padding[:, :count]
                if prediction.shape != target.shape:
                    raise ValueError("wrench predictions and labels disagree")
                hold_target = hold.view(len(hold), *([1] * (target.ndim - 2)), 6).expand_as(target)
                add(f"wrench{suffix}_normalized_mae", prediction, target, padding, [1.] * 6)
                for label, section, unit in (("force", slice(0, 3), "n"), ("moment", slice(3, 6), "nm")):
                    std = normalization.force_std[section]
                    add(f"{label}{suffix}_mae_{unit}", prediction[..., section], target[..., section], padding, std)
                    add(f"{label}_hold{suffix}_mae_{unit}", hold_target[..., section], target[..., section], padding, std)
                    add(f"{label}{suffix}_normalized_mae", prediction[..., section], target[..., section], padding, [1.] * 3)
    finally:
        model.train(previous)
    result = {"windows": windows, "force_representation": "end_effector_wrench6",
              "force_component_units": ["N"] * 3 + ["N*m"] * 3, **_summarize(sums)}
    # Keep the action count key shared with legacy evaluation consumers.
    result["action_valid_scalars"] = result["action_mae_rad_valid_scalars"]
    if kind != "act":
        result.update(force_evaluation_steps=count, force_evaluation_duration_s=count / model.config.policy_sample_rate_hz)
    return result


def aggregate_wrench6_metrics(per_episode):
    rows = list(per_episode.values())
    if any(r.get("force_representation") != "end_effector_wrench6" for r in rows):
        raise ValueError("mixed force representations in episode evaluation")
    result = {"windows": sum(r["windows"] for r in rows), "per_episode": per_episode,
              "force_representation": "end_effector_wrench6", "force_component_units": rows[0]["force_component_units"]}
    keys = [key for key in rows[0] if key + "_valid_scalars" in rows[0]]
    for key in keys:
        counts = [r[key + "_valid_scalars"] for r in rows]
        total = sum(counts)
        result[key + "_valid_scalars"] = total
        result[key] = sum(r[key] * c for r, c in zip(rows, counts) if c) / total if total else None
        values = [r[key] for r in rows if r[key] is not None]
        result["episode_macro_" + key] = sum(values) / len(values) if values else None
        valid = [(r[key + "_per_axis"], c) for r, c in zip(rows, counts) if c]
        result[key + "_per_axis"] = (sum(np.asarray(v) * c for v, c in valid) / total).tolist() if total else None
    result["action_valid_scalars"] = result["action_mae_rad_valid_scalars"]
    for key in ("force_evaluation_steps", "force_evaluation_duration_s"):
        if key in rows[0]:
            if any(r.get(key) != rows[0][key] for r in rows):
                raise ValueError("episodes must use the same force evaluation horizon")
            result[key] = rows[0][key]
    return result
