"""Seven-joint, missing-slot contracts shared by real-robot training models."""

import math
from dataclasses import replace

import pytest
import torch

from force_aware_act.act_aligned_training import (
    ACTAlignedHighRateBatch,
    ACTAlignedHighRateCriterion,
    ACTAlignedHighRateTrainingConfig,
    ACTAlignedHighRateMotionCriterion,
    ACTAlignedHighRateMotionTrainingConfig,
    ACTAlignedHighRateDualZeroCriterion,
    ACTAlignedHighRateDualZeroTrainingConfig,
    build_act_aligned_high_rate_optimizer,
    build_act_aligned_high_rate_motion_optimizer,
    build_act_aligned_high_rate_dual_zero_optimizer,
    run_high_rate_training_epoch,
    run_high_rate_validation_epoch,
    run_high_rate_motion_training_epoch,
    run_high_rate_motion_validation_epoch,
    run_high_rate_dual_zero_training_epoch,
    run_high_rate_dual_zero_validation_epoch,
)
from force_aware_act.act_aligned_training.high_rate_data import (
    ACTAlignedHighRateSample,
    collate_high_rate_samples,
)
from force_aware_act.act_aligned_training.high_rate_losses import (
    masked_interval_balanced_high_rate_l1_loss,
)
from force_aware_act.models.act_aligned import (
    ACTAlignedHighRateConfig,
    ACTAlignedHighRateContactCVAEPolicy,
    ACTAlignedHighRateMotionCVAEPolicy,
    ACTAlignedHighRateDualZeroPolicy,
)
from force_aware_act.models.act_aligned.high_rate_force import ACTAlignedHighRateForceEncoder


def _config(kind="contact"):
    factory = {
        "contact": ACTAlignedHighRateConfig,
        "motion": ACTAlignedHighRateConfig.motion_control,
        "dual_zero": ACTAlignedHighRateConfig.dual_zero,
    }[kind]
    return factory(
        d_model=16, nhead=4, dim_feedforward=32, encoder_layers=1,
        decoder_layers=1, local_force_encoder_layers=1, local_force_dim=8,
        force_dim=7, force_sample_rate_hz=300, policy_sample_rate_hz=100,
        max_force_samples_per_interval=8, max_online_force_intervals=2,
        chunk_len=3, dropout=0, image_height=32, image_width=32,
        pretrained_backbone=False, imagenet_normalize=False,
    )


def _batch(config, *, missing_future=False):
    b, h, k, s, f = 1, config.max_online_force_intervals, config.chunk_len, 8, 7
    online_mask = torch.ones(b, h, s, dtype=torch.bool)
    online_mask[:, :, [0, 2]] = False
    future_mask = torch.ones(b, k, s, dtype=torch.bool)
    if not missing_future:
        future_mask[:, :, [0, 2]] = False
    return ACTAlignedHighRateBatch(
        images=torch.randn(b, 2, 3, 32, 32), qpos=torch.randn(b, 7),
        action_chunk=torch.randn(b, k, 7),
        action_padding_mask=torch.tensor([[False, True, False]]),
        online_force_intervals=torch.randn(b, h, s, f),
        online_force_relative_time=torch.arange(s).float().div(300).expand(b, h, s).clone(),
        online_force_sample_padding_mask=online_mask,
        online_force_interval_padding_mask=online_mask.all(-1),
        future_force_intervals=torch.randn(b, k, s, f),
        future_force_relative_time=torch.arange(s).float().div(300).expand(b, k, s).clone(),
        future_force_sample_padding_mask=future_mask,
        future_force_interval_padding_mask=future_mask.all(-1),
        future_force_target=torch.randn(b, k, f),
        future_force_target_padding_mask=future_mask[:, :, 2].clone(),
    )


def test_internal_holes_use_last_valid_slot_and_ignore_missing_payloads():
    config = _config()
    encoder = ACTAlignedHighRateForceEncoder(config)
    encoder.blocks = torch.nn.ModuleList()
    values = torch.randn(1, 1, 8, 7)
    mask = torch.ones(1, 1, 8, dtype=torch.bool)
    mask[:, :, [0, 2]] = False
    values[mask] = float("nan")
    times = torch.arange(8).float().div(300).reshape(1, 1, 8)
    times[mask] = float("nan")
    captured = []
    handle = encoder.output_projection.register_forward_pre_hook(
        lambda module, inputs: captured.append(inputs[0].detach().clone())
    )
    output = encoder(values, times, mask, mask.all(-1))
    handle.remove()
    expected_last = encoder.input_projection(values[:, :, 2])[:, 0]
    expected_last += encoder._time_encoding(times[:, :, 2], values.dtype)[:, 0]
    torch.testing.assert_close(captured[0][:, -config.local_force_dim:], expected_last)
    assert torch.isfinite(output).all()


def test_endpoint_mask_is_independent_and_collates_with_legacy_fallback():
    batch = _batch(_config())
    endpoint_mask = torch.ones_like(batch.future_force_interval_padding_mask)
    batch = replace(batch, future_force_target_padding_mask=endpoint_mask)
    batch.validate(_config())
    moved = batch.to("cpu")
    assert moved.force_target_padding_mask.all()
    criterion = ACTAlignedHighRateDualZeroCriterion(ACTAlignedHighRateDualZeroTrainingConfig())
    outputs = {
        "pred_action": batch.action_chunk.clone(),
        "pred_force": torch.ones_like(batch.future_force_target),
        "pred_force_highrate": batch.future_force_intervals + 1,
    }
    losses = criterion(
        outputs, batch.action_chunk, batch.future_force_target,
        batch.future_force_intervals, batch.action_padding_mask,
        batch.future_force_interval_padding_mask, batch.future_force_sample_padding_mask,
        batch.force_target_padding_mask,
    )
    assert losses["loss_force"].item() == 0
    assert losses["loss_force_highrate"].item() == pytest.approx(1)
    fields = {
        name: getattr(batch, name)[0]
        for name in batch.__dataclass_fields__
        if name != "future_force_target_padding_mask"
    }
    legacy = ACTAlignedHighRateSample(**fields, episode_id="legacy", timestep=0, online_raw_valid_count=4)
    explicit = replace(legacy, future_force_target_padding_mask=endpoint_mask[0])
    collated = collate_high_rate_samples([legacy, explicit])
    assert not collated.force_target_padding_mask[0].any()
    assert collated.force_target_padding_mask[1].all()
    collated.validate(_config())


def test_missing_auxiliary_slots_have_zero_gradient_even_with_nan_payload():
    prediction = torch.ones(1, 1, 8, 7, requires_grad=True)
    target = torch.full_like(prediction, float("nan"))
    target[:, :, [0, 2]] = 0
    mask = torch.ones(1, 1, 8, dtype=torch.bool)
    mask[:, :, [0, 2]] = False
    loss = masked_interval_balanced_high_rate_l1_loss(
        prediction, target, mask, mask.all(-1), name="torque"
    )
    loss.backward()
    assert loss.item() == pytest.approx(1)
    assert not prediction.grad[mask].any()
    assert prediction.grad[~mask].abs().sum() > 0


def test_endpoint_and_highrate_epoch_metrics_use_separate_valid_counts():
    from force_aware_act.act_aligned_training.high_rate_control_loop import _HighRateDualZeroAccumulator

    first = _batch(_config())
    first = replace(first, future_force_target_padding_mask=torch.tensor([[False, True, True]]))
    second = _batch(_config())
    sample_mask = second.future_force_sample_padding_mask.clone()
    sample_mask[:, -1] = True
    second = replace(
        second, future_force_sample_padding_mask=sample_mask,
        future_force_interval_padding_mask=sample_mask.all(-1),
        future_force_target_padding_mask=torch.tensor([[False, False, True]]),
    )
    accumulator = _HighRateDualZeroAccumulator(ACTAlignedHighRateDualZeroTrainingConfig())
    for batch, endpoint, highrate in ((first, 10.0, 4.0), (second, 20.0, 8.0)):
        accumulator.add_validation(batch, {
            "deployment_action_l1": 1.0,
            "deployment_force_l1": endpoint,
            "deployment_force_highrate_l1": highrate,
        })
    result = accumulator.validation_result()
    assert result["deployment_force_l1"] == pytest.approx(50 / 3)
    assert result["deployment_force_highrate_l1"] == pytest.approx(28 / 5)
    assert result["force_target_valid_count"] == 3
    assert result["force_highrate_valid_interval_count"] == 5


def test_shared_action_loss_ignores_nan_missing_targets_and_rejects_valid_nan():
    from force_aware_act.act_aligned_training.losses import masked_l1_loss

    prediction = torch.ones(1, 3, 7, requires_grad=True)
    target = torch.zeros_like(prediction)
    target[:, 1] = float("nan")
    mask = torch.tensor([[False, True, False]])
    loss = masked_l1_loss(prediction, target, mask, name="action")
    loss.backward()
    assert loss.item() == pytest.approx(1)
    assert not prediction.grad[:, 1].any()
    target[:, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        masked_l1_loss(prediction, target, mask, name="action")


@pytest.mark.parametrize("kind", ["contact", "motion", "official"])
def test_missing_action_labels_cannot_leak_through_posteriors(kind):
    from force_aware_act.models.act_aligned.contact_latent import ACTAlignedHighRateContactPosterior
    from force_aware_act.models.act_aligned.motion_latent import ACTAlignedMotionPosterior
    from force_aware_act.models.official_act.config import OfficialACTConfig
    from force_aware_act.models.official_act.posterior import OfficialACTPosterior

    config = _config("motion" if kind == "motion" else "contact")
    if kind == "official":
        config = OfficialACTConfig(d_model=16, nhead=4, dim_feedforward=32,
                                   encoder_layers=1, dropout=0, chunk_len=3,
                                   pretrained_backbone=False, imagenet_normalize=False)
        posterior = OfficialACTPosterior(config)
    else:
        posterior = (ACTAlignedHighRateContactPosterior if kind == "contact" else ACTAlignedMotionPosterior)(config)
    posterior.eval()
    qpos = torch.randn(1, 7)
    action = torch.randn(1, 3, 7)
    mask = torch.tensor([[False, True, False]])
    altered = action.clone()
    altered[:, 1] = float("nan")
    if kind == "contact":
        force_tokens = torch.randn(1, 3, config.d_model)
        force_mask = torch.tensor([[False, False, True]])
        changed_force = force_tokens.clone()
        changed_force[:, 1:] = float("nan")
        expected = posterior(qpos, action, force_tokens, action_padding_mask=mask,
                             future_force_interval_padding_mask=force_mask, sample=False)
        actual = posterior(qpos, altered, changed_force, action_padding_mask=mask,
                           future_force_interval_padding_mask=force_mask, sample=False)
    else:
        expected = posterior(qpos, action, padding_mask=mask, sample=False)
        actual = posterior(qpos, altered, padding_mask=mask, sample=False)
    for observed, reference in zip(actual, expected):
        torch.testing.assert_close(observed, reference)
        assert torch.isfinite(observed).all()


@pytest.mark.parametrize("kind", ["contact", "motion", "dual_zero"])
def test_seven_joint_training_and_validation_allow_all_missing_future_torque(kind):
    factories = {
        "contact": (ACTAlignedHighRateContactCVAEPolicy, ACTAlignedHighRateTrainingConfig,
                    ACTAlignedHighRateCriterion, build_act_aligned_high_rate_optimizer,
                    run_high_rate_training_epoch, run_high_rate_validation_epoch),
        "motion": (ACTAlignedHighRateMotionCVAEPolicy, ACTAlignedHighRateMotionTrainingConfig,
                   ACTAlignedHighRateMotionCriterion, build_act_aligned_high_rate_motion_optimizer,
                   run_high_rate_motion_training_epoch, run_high_rate_motion_validation_epoch),
        "dual_zero": (ACTAlignedHighRateDualZeroPolicy, ACTAlignedHighRateDualZeroTrainingConfig,
                      ACTAlignedHighRateDualZeroCriterion, build_act_aligned_high_rate_dual_zero_optimizer,
                      run_high_rate_dual_zero_training_epoch, run_high_rate_dual_zero_validation_epoch),
    }
    model_cls, train_config_cls, criterion_cls, optimizer_fn, train_fn, eval_fn = factories[kind]
    config = _config(kind)
    model = model_cls(config)
    train_config = train_config_cls()
    criterion = criterion_cls(train_config)
    optimizer = optimizer_fn(model, train_config)
    batch = _batch(config, missing_future=True)
    training = train_fn(model, criterion, optimizer, [batch], train_config, device=torch.device("cpu"))
    validation = eval_fn(model, criterion, [batch], train_config, device=torch.device("cpu"))
    assert all(math.isfinite(value) for value in training.values())
    assert all(math.isfinite(value) for value in validation.values())
    assert training["force_target_valid_count"] == 0
    assert validation["force_highrate_valid_interval_count"] == 0
    assert not any(name.endswith(("force_l1", "force_highrate_l1")) for name in validation)
    assert any(name.endswith("action_l1") for name in validation)
