from dataclasses import asdict
from argparse import Namespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from force_aware_act.act_aligned_training.checkpoint import (
    CHECKPOINT_FORMAT_VERSION,
)
from force_aware_act.inference import (
    ACT_ALIGNED_HIGH_RATE_DUAL_ZERO_ROLLOUT_KIND,
    ACT_ALIGNED_HIGH_RATE_MOTION_ROLLOUT_KIND,
    ACT_ALIGNED_HIGH_RATE_ROLLOUT_KIND,
    ACT_ALIGNED_ROLLOUT_KIND,
    HighRateForceRingBuffer,
    NO_FORCE_HISTORY_CONTRACT,
    OFFICIAL_ACT_ROLLOUT_KIND,
    RolloutPolicyAdapter,
    checkpoint_uses_rollout_adapter,
)
from force_aware_act.models.act_aligned import (
    ACTAlignedConfig,
    ACTAlignedContactCVAEPolicy,
    ACTAlignedHighRateConfig,
    ACTAlignedHighRateContactCVAEPolicy,
    ACTAlignedHighRateDualZeroPolicy,
    ACTAlignedHighRateMotionCVAEPolicy,
)
from force_aware_act.models.official_act import (
    OfficialACTConfig,
    OfficialACTPolicy,
)
from force_aware_act.official_act_training.checkpoint import (
    OFFICIAL_ACT_CHECKPOINT_VERSION,
)
from scripts.run_mujoco_policy_rollout import _resolve_checkpoint_contract


def _normalization(*, include_force: bool) -> dict:
    values = {
        "qpos_mean": tuple(float(value) for value in range(7)),
        "qpos_std": (2.0,) * 7,
        "action_mean": (10.0,) * 7,
        "action_std": (3.0,) * 7,
    }
    if include_force:
        values.update(
            {
                "force_mean": tuple(float(value) for value in range(6)),
                "force_std": (4.0,) * 6,
            }
        )
    return values


def _official_checkpoint(*, include_format: bool = True) -> dict:
    config = OfficialACTConfig.compact_smoke(
        encoder_layers=1,
        decoder_layers=1,
        chunk_len=3,
        image_height=32,
        image_width=48,
    )
    payload = {
        "architecture_version": config.architecture_version,
        "model_config": asdict(config),
        "model_state": OfficialACTPolicy(config).state_dict(),
        "normalization": _normalization(include_force=False),
    }
    if include_format:
        payload["format_version"] = OFFICIAL_ACT_CHECKPOINT_VERSION
    return payload


def _contact_checkpoint() -> dict:
    config = ACTAlignedConfig.compact_smoke(
        d_model=32,
        dim_feedforward=64,
        encoder_layers=1,
        decoder_layers=1,
        chunk_len=3,
        force_window_len=4,
        image_height=32,
        image_width=48,
    )
    return {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "architecture_version": config.architecture_version,
        "model_config": asdict(config),
        "model_state": ACTAlignedContactCVAEPolicy(config).state_dict(),
        "normalization": _normalization(include_force=True),
    }


def _high_rate_checkpoint() -> dict:
    config = ACTAlignedHighRateConfig(
        d_model=32,
        nhead=4,
        dim_feedforward=64,
        local_force_dim=16,
        dropout=0.0,
        chunk_len=3,
        image_height=32,
        image_width=48,
        pretrained_backbone=False,
        imagenet_normalize=False,
    )
    return {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "architecture_version": config.architecture_version,
        "model_config": asdict(config),
        "model_state": ACTAlignedHighRateContactCVAEPolicy(config).state_dict(),
        "normalization": _normalization(include_force=True),
    }


def _high_rate_control_checkpoint(kind: str) -> dict:
    factory, policy_type = {
        "motion": (
            ACTAlignedHighRateConfig.motion_control,
            ACTAlignedHighRateMotionCVAEPolicy,
        ),
        "dual_zero": (
            ACTAlignedHighRateConfig.dual_zero,
            ACTAlignedHighRateDualZeroPolicy,
        ),
    }[kind]
    config = factory(
        d_model=32,
        nhead=4,
        dim_feedforward=64,
        local_force_dim=16,
        dropout=0.0,
        chunk_len=3,
        image_height=32,
        image_width=48,
        pretrained_backbone=False,
        imagenet_normalize=False,
    )
    return {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "architecture_version": config.architecture_version,
        "model_config": asdict(config),
        "model_state": policy_type(config).state_dict(),
        "normalization": _normalization(include_force=True),
    }


@pytest.mark.parametrize("include_format", (True, False))
def test_official_adapter_supports_full_and_best_policy_checkpoints(
    include_format,
):
    checkpoint = _official_checkpoint(include_format=include_format)
    assert checkpoint_uses_rollout_adapter(checkpoint)
    adapter = RolloutPolicyAdapter.from_checkpoint(
        checkpoint,
        device=torch.device("cpu"),
    )
    assert adapter.kind == OFFICIAL_ACT_ROLLOUT_KIND
    assert adapter.chunk_len == 3
    assert not adapter.uses_force_history
    assert adapter.force_history_contract == NO_FORCE_HISTORY_CONTRACT

    native_images = torch.rand(2, 3, 32, 48)
    with patch("torch.nn.functional.interpolate") as interpolate:
        images = adapter.prepare_images(native_images)
    interpolate.assert_not_called()
    qpos = adapter.prepare_qpos(np.arange(7, dtype=np.float32))
    output = adapter.forward(images, qpos)
    action, force = adapter.denormalize_predictions(output)

    assert images.shape == (1, 2, 3, 32, 48)
    assert torch.equal(qpos, torch.zeros_like(qpos))
    assert action.shape == (3, 7)
    assert force.shape == (3, 6)
    assert np.isnan(force).all()


def test_contact_adapter_reproduces_causal_left_padded_force_contract():
    adapter = RolloutPolicyAdapter.from_checkpoint(
        _contact_checkpoint(),
        device=torch.device("cpu"),
    )
    assert adapter.kind == ACT_ALIGNED_ROLLOUT_KIND
    assert adapter.force_window_len == 4
    assert adapter.force_history_contract == (
        "causal_state_rate_last_l_left_padded_normalized_v1"
    )

    raw = np.stack(
        (
            np.arange(6, dtype=np.float32),
            np.arange(6, dtype=np.float32) + 4.0,
        )
    )
    history, padding_mask = adapter.prepare_force_history(raw)
    expected = torch.as_tensor(
        (raw - np.arange(6, dtype=np.float32)) / 4.0
    )
    assert history.shape == (1, 4, 6)
    assert torch.equal(history[0, :2], torch.zeros(2, 6))
    assert torch.allclose(history[0, 2:], expected)
    assert torch.equal(
        padding_mask,
        torch.tensor([[True, True, False, False]]),
    )

    images = adapter.prepare_images(torch.rand(2, 3, 32, 48))
    qpos = adapter.prepare_qpos(np.arange(7, dtype=np.float32))
    first = adapter.forward(
        images,
        qpos,
        force_history=history,
        force_padding_mask=padding_mask,
    )
    second = adapter.forward(
        images,
        qpos,
        force_history=history,
        force_padding_mask=padding_mask,
    )
    action, force = adapter.denormalize_predictions(first)
    assert first["contact_latent_source"] == "zero"
    diagnostics = adapter.deployment_diagnostics(
        first,
        force_padding_mask=padding_mask,
        requested_latent_mode="zero",
    )
    assert diagnostics == {
        "latent_name": "z_contact",
        "latent_source": "zero",
        "latent_max_abs": 0.0,
        "force_history_valid_samples": 2,
        "force_history_padding_samples": 2,
    }
    assert torch.equal(first["pred_action"], second["pred_action"])
    assert action.shape == (3, 7)
    assert force.shape == (3, 6)


def test_high_rate_adapter_uses_continuous_500hz_ring_and_training_packer():
    adapter = RolloutPolicyAdapter.from_checkpoint(
        _high_rate_checkpoint(), device=torch.device("cpu")
    )
    assert adapter.kind == ACT_ALIGNED_HIGH_RATE_ROLLOUT_KIND
    assert adapter.uses_high_rate_force_history
    assert not adapter.uses_state_rate_force_history
    assert adapter.force_window_len == 100
    assert adapter.force_history_contract == (
        "causal_raw_500hz_last_100_grouped_state_intervals_v2"
    )

    buffer = HighRateForceRingBuffer(0.0, np.zeros(6, dtype=np.float32))
    policy_times_ms = {33, 67, 100, 133, 167, 200}
    buffer.record_policy_state(0.0)
    for millisecond in range(1, 201):
        wrench = np.full(6, millisecond, dtype=np.float32)
        buffer.observe_physics_step(millisecond / 1000.0, wrench)
        if millisecond in policy_times_ms:
            buffer.record_policy_state(millisecond / 1000.0)
    snapshot = buffer.snapshot()
    intervals, relative_time, sample_mask, interval_mask = (
        adapter.prepare_high_rate_force_history(
            snapshot.force_timestamps,
            snapshot.force_values,
            snapshot.state_timestamps,
        )
    )

    assert buffer.total_samples == 101
    assert snapshot.force_values.shape == (100, 6)
    assert intervals.shape == (1, 7, 20, 6)
    assert relative_time.shape == (1, 7, 20)
    assert int((~sample_mask).sum()) == 100
    assert int((~interval_mask).sum()) == 6
    images = adapter.prepare_images(torch.rand(2, 3, 32, 48))
    qpos = adapter.prepare_qpos(np.arange(7, dtype=np.float32))
    output = adapter.forward(
        images,
        qpos,
        online_force_intervals=intervals,
        online_force_relative_time=relative_time,
        online_force_sample_padding_mask=sample_mask,
        online_force_interval_padding_mask=interval_mask,
        contact_latent_mode="zero",
    )
    assert output["contact_latent_source"] == "zero"
    assert torch.equal(output["z_contact"], torch.zeros_like(output["z_contact"]))
    assert output["pred_action"].shape == (1, 3, 7)


@pytest.mark.parametrize(
    ("kind", "expected_kind", "latent_name"),
    (
        (
            "motion",
            ACT_ALIGNED_HIGH_RATE_MOTION_ROLLOUT_KIND,
            "z_motion",
        ),
        (
            "dual_zero",
            ACT_ALIGNED_HIGH_RATE_DUAL_ZERO_ROLLOUT_KIND,
            "none",
        ),
    ),
)
def test_high_rate_control_adapters_use_native_force_and_zero_only_mode(
    kind,
    expected_kind,
    latent_name,
):
    adapter = RolloutPolicyAdapter.from_checkpoint(
        _high_rate_control_checkpoint(kind),
        device=torch.device("cpu"),
    )
    assert adapter.kind == expected_kind
    assert adapter.uses_high_rate_force_history
    assert not adapter.has_contact_prior
    online_force, relative_time, sample_mask, interval_mask = (
        _interval_tensors_for_adapter(adapter)
    )
    images = adapter.prepare_images(torch.rand(2, 3, 32, 48))
    qpos = adapter.prepare_qpos(np.arange(7, dtype=np.float32))

    output = adapter.forward(
        images,
        qpos,
        online_force_intervals=online_force,
        online_force_relative_time=relative_time,
        online_force_sample_padding_mask=sample_mask,
        online_force_interval_padding_mask=interval_mask,
    )
    diagnostics = adapter.deployment_diagnostics(
        output,
        force_padding_mask=sample_mask.flatten(1),
        requested_latent_mode="zero",
    )

    assert diagnostics["latent_name"] == latent_name
    assert diagnostics["latent_source"] in {"zero", "none"}
    assert diagnostics["latent_max_abs"] == 0.0
    assert diagnostics["force_history_valid_samples"] == int(
        (~sample_mask).sum().item()
    )
    with pytest.raises(ValueError, match="require zero latent mode"):
        adapter.forward(
            images,
            qpos,
            online_force_intervals=online_force,
            online_force_relative_time=relative_time,
            online_force_sample_padding_mask=sample_mask,
            online_force_interval_padding_mask=interval_mask,
            contact_latent_mode="prior",
        )


def _interval_tensors_for_adapter(adapter):
    config = adapter.config
    interval_count = config.max_online_force_intervals
    sample_count = config.max_force_samples_per_interval
    force = torch.randn(1, interval_count, sample_count, config.force_dim)
    relative_time = (
        torch.arange(sample_count, dtype=torch.float32)
        .div(config.force_sample_rate_hz)
        .view(1, 1, sample_count)
        .expand(1, interval_count, sample_count)
        .clone()
    )
    sample_mask = torch.zeros(
        1,
        interval_count,
        sample_count,
        dtype=torch.bool,
    )
    interval_mask = sample_mask.all(dim=-1)
    return force, relative_time, sample_mask, interval_mask


def test_high_rate_ring_rejects_a_missed_500hz_sampling_deadline():
    buffer = HighRateForceRingBuffer(0.0, np.zeros(6, dtype=np.float32))

    with pytest.raises(RuntimeError, match="skipped"):
        buffer.observe_physics_step(0.004, np.ones(6, dtype=np.float32))


def test_adapter_rejects_nonzero_deployment_latent_marked_as_zero():
    adapter = RolloutPolicyAdapter.from_checkpoint(
        _contact_checkpoint(),
        device=torch.device("cpu"),
    )
    output = {
        "z_contact": torch.ones(1, adapter.config.latent_dim),
        "contact_latent_source": "zero",
    }
    padding_mask = torch.zeros(1, adapter.force_window_len, dtype=torch.bool)

    with pytest.raises(RuntimeError, match="zero latent is not exactly zero"):
        adapter.deployment_diagnostics(
            output,
            force_padding_mask=padding_mask,
            requested_latent_mode="zero",
        )


def test_adapter_rejects_missing_embedded_normalization():
    checkpoint = _official_checkpoint()
    checkpoint.pop("normalization")
    with pytest.raises(KeyError, match="embedded normalization"):
        RolloutPolicyAdapter.from_checkpoint(
            checkpoint,
            device=torch.device("cpu"),
        )


def test_official_adapter_rejects_prior_latent_mode():
    adapter = RolloutPolicyAdapter.from_checkpoint(
        _official_checkpoint(),
        device=torch.device("cpu"),
    )
    images = adapter.prepare_images(torch.rand(2, 3, 32, 48))
    qpos = adapter.prepare_qpos(np.arange(7, dtype=np.float32))
    with pytest.raises(ValueError, match="must be zero"):
        adapter.forward(images, qpos, contact_latent_mode="prior")


def _rollout_args(**overrides) -> Namespace:
    values = {
        "checkpoint": torch.tensor(0),
        "normalization_stats": None,
        "chunk_len": None,
        "force_window_len": None,
        "force_window_duration": None,
        "policy_rate_hz": 30.0,
        "image_height": 480,
        "image_width": 640,
        "contact_latent_mode": "zero",
        "action_mode": "joint_pos",
    }
    values.update(overrides)
    return Namespace(**values)


def test_contact_checkpoint_contract_is_inferred_without_external_stats():
    args = _rollout_args()
    adapter, stats, normalization_reference = _resolve_checkpoint_contract(
        args,
        _contact_checkpoint(),
        torch.device("cpu"),
    )

    assert adapter.kind == ACT_ALIGNED_ROLLOUT_KIND
    assert stats is None
    assert normalization_reference.startswith("embedded:")
    assert args.chunk_len == 3
    assert args.force_window_len == 4
    assert args.force_window_duration == pytest.approx(3.0 / 30.0)
    assert (args.image_height, args.image_width) == (32, 48)


def test_high_rate_checkpoint_contract_uses_500hz_window_duration():
    args = _rollout_args()
    adapter, stats, _ = _resolve_checkpoint_contract(
        args, _high_rate_checkpoint(), torch.device("cpu")
    )

    assert adapter.kind == ACT_ALIGNED_HIGH_RATE_ROLLOUT_KIND
    assert stats is None
    assert args.force_window_len == 100
    assert args.force_window_duration == pytest.approx(99.0 / 500.0)


@pytest.mark.parametrize("kind", ("motion", "dual_zero"))
def test_high_rate_control_checkpoint_contract_rejects_prior_mode(kind):
    args = _rollout_args(contact_latent_mode="prior")

    with pytest.raises(ValueError, match="has no contact prior"):
        _resolve_checkpoint_contract(
            args,
            _high_rate_control_checkpoint(kind),
            torch.device("cpu"),
        )


def test_high_rate_checkpoint_rejects_policy_rate_different_from_training():
    args = _rollout_args(policy_rate_hz=25.0)

    with pytest.raises(ValueError, match="policy_sample_rate_hz"):
        _resolve_checkpoint_contract(
            args, _high_rate_checkpoint(), torch.device("cpu")
        )


def test_checkpoint_contract_rejects_silent_chunk_override():
    args = _rollout_args(chunk_len=100)
    with pytest.raises(ValueError, match="does not match checkpoint"):
        _resolve_checkpoint_contract(
            args,
            _official_checkpoint(),
            torch.device("cpu"),
        )


def test_official_checkpoint_contract_has_no_force_input():
    args = _rollout_args()
    adapter, stats, _ = _resolve_checkpoint_contract(
        args,
        _official_checkpoint(),
        torch.device("cpu"),
    )

    assert adapter.kind == OFFICIAL_ACT_ROLLOUT_KIND
    assert stats is None
    assert args.force_window_len == 0
    assert args.force_window_duration == 0.0
