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
    ACT_ALIGNED_ROLLOUT_KIND,
    NO_FORCE_HISTORY_CONTRACT,
    OFFICIAL_ACT_ROLLOUT_KIND,
    RolloutPolicyAdapter,
    checkpoint_uses_rollout_adapter,
)
from force_aware_act.models.act_aligned import (
    ACTAlignedConfig,
    ACTAlignedContactCVAEPolicy,
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
