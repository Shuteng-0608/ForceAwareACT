from argparse import Namespace

import numpy as np
import pytest
import torch

from force_aware_act.data import canonical_json_sha256, normalize_tensor
from force_aware_act.evaluation import ContactRecoveryConfig, ContactRecoveryStateMachine
from force_aware_act.training.checkpointing import build_checkpoint_v2

from scripts.run_mujoco_policy_rollout import (
    SUMMARY_REQUIRED_KEYS,
    _control_command_is_finite,
    _contact_recovery_config,
    _finalize_contact_recovery_summary,
    _interpret_selected_action,
    _position_success_condition,
    _recovery_observation_is_valid,
    _resolve_and_validate_rollout_args,
    _resolve_inference_device,
    _run_mode,
    _stats_to_device,
    _success_condition,
    _update_success_hold_counter,
    _selected_action_delta_norm_raw_to_current,
    _selected_action_index,
    _validate_stats_action_mode,
    _validate_rollout_artifact_semantics,
    _validate_summary_schema,
    parse_args,
)


class _ColocationCheckingPolicy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))

    def forward(self, images, qpos, force_window, **kwargs):
        expected_device = self.weight.device
        assert images.device == expected_device
        assert qpos.device == expected_device
        assert force_window.device == expected_device
        return {
            "pred_action": qpos.new_zeros((qpos.shape[0], 2, 7)),
            "pred_force": force_window.new_zeros((qpos.shape[0], 2, 6)),
        }


def _assert_inference_colocation(device: torch.device) -> None:
    model = _ColocationCheckingPolicy().to(device).eval()
    stats = _stats_to_device(
        {
            "qpos_mean": torch.zeros(7),
            "qpos_std": torch.ones(7),
            "force_mean": torch.zeros(6),
            "force_std": torch.ones(6),
        },
        device,
    )
    images = torch.zeros((1, 2, 3, 16, 16), device=device)
    qpos = normalize_tensor(
        torch.zeros((1, 7), device=device),
        stats["qpos_mean"],
        stats["qpos_std"],
    )
    force_window = normalize_tensor(
        torch.zeros((1, 20, 6), device=device),
        stats["force_mean"],
        stats["force_std"],
    )

    output = _run_mode(model, images, qpos, force_window, "zero")

    model_device = next(model.parameters()).device
    assert model_device.type == device.type
    assert all(value.device == model_device for value in stats.values())
    assert output["pred_action"].device == model_device
    assert output["pred_force"].device == model_device
    assert output["pred_action"].is_inference()


def test_explicit_cpu_device_resolution_and_inference_colocation():
    device = _resolve_inference_device("cpu")

    assert device == torch.device("cpu")
    _assert_inference_colocation(device)


def test_auto_device_resolution(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert _resolve_inference_device("auto") == torch.device("cuda")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert _resolve_inference_device("auto") == torch.device("cpu")


def test_explicit_cuda_fails_clearly_when_unavailable(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="CUDA is not available"):
        _resolve_inference_device("cuda")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_cuda_inference_colocation():
    _assert_inference_colocation(_resolve_inference_device("cuda"))


def test_absolute_action_mode_target_ctrl_is_prediction():
    qpos = np.asarray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    pred_action = np.asarray([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])

    target_ctrl = _interpret_selected_action(pred_action, qpos, "action")

    np.testing.assert_allclose(target_ctrl, pred_action)


def test_legacy_joint_pos_mode_target_ctrl_is_prediction():
    qpos = np.asarray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    pred_action = np.asarray([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])

    target_ctrl = _interpret_selected_action(pred_action, qpos, "joint_pos")

    np.testing.assert_allclose(target_ctrl, pred_action)


def test_delta_joint_cmd_target_ctrl_is_current_qpos_plus_prediction():
    qpos = np.asarray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    pred_delta = np.asarray([0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7])

    target_ctrl = _interpret_selected_action(pred_delta, qpos, "delta_joint_cmd")

    np.testing.assert_allclose(target_ctrl, qpos + pred_delta)


def test_delta_joint_pos_command_target_ctrl_is_current_qpos_plus_prediction():
    qpos = np.asarray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    pred_delta = np.asarray([0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7])

    target_ctrl = _interpret_selected_action(pred_delta, qpos, "delta_joint_pos_command")

    np.testing.assert_allclose(target_ctrl, qpos + pred_delta)


def test_selected_action_delta_norm_uses_raw_delta_for_delta_modes():
    qpos = np.asarray([10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0])
    pred_delta = np.asarray([0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7])

    value = _selected_action_delta_norm_raw_to_current(pred_delta, qpos, "delta_joint_cmd")

    assert value == pytest.approx(np.linalg.norm(pred_delta))


def test_selected_action_delta_norm_uses_prediction_minus_qpos_for_absolute_modes():
    qpos = np.asarray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    pred_action = np.asarray([1.1, 1.8, 3.3, 3.6, 5.5, 5.4, 7.7])

    value = _selected_action_delta_norm_raw_to_current(pred_action, qpos, "action")

    assert value == pytest.approx(np.linalg.norm(pred_action - qpos))


def test_one_based_action_chunk_selection_maps_to_zero_based_indices():
    assert [_selected_action_index(10, str(index)) for index in range(1, 11)] == list(
        range(10)
    )
    assert _selected_action_index(10, "first") == 0
    assert _selected_action_index(10, "mid") == 5
    assert _selected_action_index(10, "last") == 9
    assert _selected_action_index(10, "temporal") == -1


@pytest.mark.parametrize("mode", ["0", "11", "unknown"])
def test_action_chunk_selection_rejects_out_of_range_or_unknown_modes(mode):
    with pytest.raises(ValueError, match="action selection|unknown action"):
        _selected_action_index(10, mode)


def test_stats_action_mode_mismatch_raises_clear_error():
    with pytest.raises(ValueError, match="action_mode mismatch"):
        _validate_stats_action_mode({"action_mode": "joint_pos"}, "action")


def test_missing_stats_action_mode_allows_legacy_joint_pos():
    _validate_stats_action_mode({}, "joint_pos")


def test_missing_stats_action_mode_rejects_command_modes():
    with pytest.raises(ValueError, match="do not contain action_mode metadata"):
        _validate_stats_action_mode({}, "delta_joint_cmd")


def test_success_condition_requires_distance_lateral_and_force_thresholds():
    assert _success_condition(
        peg_to_hole_dist=0.004,
        peg_to_hole_lateral_error=0.005,
        force_norm=50.0,
        distance_threshold=0.005,
        lateral_threshold=0.006,
        force_threshold=80.0,
    )
    assert not _success_condition(0.005, 0.005, 50.0, 0.005, 0.006, 80.0)
    assert not _success_condition(0.004, 0.006, 50.0, 0.005, 0.006, 80.0)
    assert not _success_condition(0.004, 0.005, 80.0, 0.005, 0.006, 80.0)


def test_recovery_geometry_and_cli_threshold_defaults_are_separate_from_force():
    assert _position_success_condition(0.004, 0.005, 0.005, 0.006)
    assert not _position_success_condition(0.005, 0.005, 0.005, 0.006)
    config = _contact_recovery_config(
        Namespace(
            success_force_threshold=40.0,
            safe_force_threshold=None,
            force_stop_threshold=300.0,
            hard_force_threshold=None,
            contact_enter_force_threshold=5.0,
            contact_exit_force_threshold=3.0,
            contact_min_steps=2,
            success_hold_steps=15,
        )
    )
    assert config.safe_force_n == 40.0
    assert config.hard_force_n == 1000.0
    assert config.success_hold_steps == 15


def test_low_legacy_force_stop_does_not_invalidate_default_recovery_thresholds():
    config = _contact_recovery_config(
        Namespace(
            success_force_threshold=40.0,
            safe_force_threshold=None,
            force_stop_threshold=20.0,
            hard_force_threshold=None,
            contact_enter_force_threshold=5.0,
            contact_exit_force_threshold=3.0,
            contact_min_steps=2,
            success_hold_steps=15,
        )
    )

    assert config.hard_force_n == 1000.0


def test_invalid_force_observation_fails_recovery_summary_closed():
    tracker = ContactRecoveryStateMachine(
        ContactRecoveryConfig(
            contact_enter_force_n=5.0,
            contact_exit_force_n=3.0,
            contact_min_steps=1,
            success_force_n=40.0,
            safe_force_n=40.0,
            hard_force_n=1000.0,
            success_hold_steps=1,
        )
    )
    tracker.update(force_n=6.0, dt_s=0.1, task_success=False)
    tracker.update(force_n=2.0, dt_s=0.1, task_success=True)

    summary = _finalize_contact_recovery_summary(tracker, 1)

    assert summary["recovery_success_observed"] is True
    assert summary["safe_success_observed"] is True
    assert summary["metrics_valid"] is False
    assert summary["invalid_observation_count"] == 1
    assert summary["recovery_success"] is False
    assert summary["safe_success"] is False


@pytest.mark.parametrize(
    "invalid_field",
    (
        "force_n",
        "peg_to_hole_dist",
        "peg_to_hole_lateral_error",
        "observation_time",
        "dt_s",
    ),
)
def test_recovery_observation_rejects_each_nonfinite_input(invalid_field):
    observation = {
        "force_n": 6.0,
        "peg_to_hole_dist": 0.004,
        "peg_to_hole_lateral_error": 0.003,
        "observation_time": 1.0,
        "dt_s": 0.1,
    }
    observation[invalid_field] = float("nan")

    assert not _recovery_observation_is_valid(**observation)


def test_recovery_observation_rejects_negative_or_reversed_time():
    base = {
        "force_n": 6.0,
        "peg_to_hole_dist": 0.004,
        "peg_to_hole_lateral_error": 0.003,
        "observation_time": 1.0,
        "dt_s": 0.1,
    }
    assert _recovery_observation_is_valid(**base)
    assert not _recovery_observation_is_valid(**{**base, "dt_s": -0.01})
    assert not _recovery_observation_is_valid(**{**base, "observation_time": -0.01})


def test_control_command_final_gate_rejects_nonfinite_or_wrong_shape():
    assert _control_command_is_finite(np.zeros(7))
    assert not _control_command_is_finite(np.asarray([0.0] * 6 + [np.inf]))
    assert not _control_command_is_finite(np.zeros(6))


@pytest.mark.parametrize(
    ("option", "value"),
    (
        ("--temporal-agg-decay", "nan"),
        ("--force-window-duration", "inf"),
        ("--policy-rate-hz", "nan"),
        ("--ema-alpha", "inf"),
        ("--max-delta-q", "nan"),
        ("--force-stop-threshold", "inf"),
        ("--success-distance-threshold", "nan"),
        ("--success-lateral-threshold", "inf"),
        ("--success-force-threshold", "nan"),
        ("--contact-enter-force-threshold", "inf"),
        ("--contact-exit-force-threshold", "nan"),
        ("--safe-force-threshold", "inf"),
        ("--hard-force-threshold", "nan"),
        ("--hole-offset-x", "inf"),
        ("--hole-offset-y", "nan"),
        ("--hole-offset-z", "inf"),
        ("--axial-push-speed", "nan"),
        ("--axial-push-start-dist", "inf"),
        ("--axial-push-stop-force", "nan"),
    ),
)
def test_rollout_rejects_every_nonfinite_float_argument(
    tmp_path, option, value
):
    checkpoint = tmp_path / "checkpoint.pt"
    stats = tmp_path / "stats.pt"
    model_xml = tmp_path / "model.xml"
    checkpoint.touch()
    stats.touch()
    model_xml.touch()
    args = parse_args(
        [
            "--checkpoint",
            str(checkpoint),
            "--normalization-stats",
            str(stats),
            "--model-xml",
            str(model_xml),
            "--output-dir",
            str(tmp_path / "output"),
            option,
            value,
        ]
    )

    with pytest.raises(ValueError, match="must be finite"):
        _resolve_and_validate_rollout_args(args)


def _formal_stats() -> dict[str, object]:
    stats: dict[str, object] = {
        "qpos_mean": torch.zeros(7),
        "qpos_std": torch.ones(7),
        "action_mean": torch.zeros(7),
        "action_std": torch.ones(7),
        "force_mean": torch.zeros(6),
        "force_std": torch.ones(6),
        "action_mode": "action",
        "chunk_len": 10,
        "force_window_len": 20,
        "force_window_duration": 0.25,
        "camera_names": list(("ee_cam", "base_top_cam")),
        "image_size": [224, 224],
        "imagenet_normalize": False,
        "normalization_config": {"method": "balanced_raw"},
        "population_identities": [{"episode_uuid": "episode-1"}],
    }
    stats["normalization_config_sha256"] = canonical_json_sha256(
        stats["normalization_config"]
    )
    stats["population_sha256"] = canonical_json_sha256(
        stats["population_identities"]
    )
    descriptor = {
        "normalization_config_sha256": stats["normalization_config_sha256"],
        "population_sha256": stats["population_sha256"],
        "statistics": {
            key: {
                "dtype": str(stats[key].dtype),
                "shape": list(stats[key].shape),
                "values": stats[key].tolist(),
            }
            for key in (
                "qpos_mean",
                "qpos_std",
                "action_mean",
                "action_std",
                "force_mean",
                "force_std",
            )
        },
    }
    stats["normalization_content_sha256"] = canonical_json_sha256(descriptor)
    return stats


def _formal_checkpoint(stats: dict[str, object]) -> dict[str, object]:
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-4)
    normalization_hash = str(stats["normalization_content_sha256"])
    config = {
        "policy_variant": "force_aware_act",
        "action_mode": "action",
        "chunk_len": 10,
        "force_window_len": 20,
        "force_window_duration": 0.25,
        "camera_names": ["ee_cam", "base_top_cam"],
        "image_size": [224, 224],
        "imagenet_normalize": False,
        "normalization_sha256": normalization_hash,
        "model": {"chunk_len": 10},
    }
    return build_checkpoint_v2(
        model=model,
        optimizer=optimizer,
        config=config,
        global_step=1,
        stage_step=1,
        epoch=0,
        step_in_epoch=1,
        stage_name="stage1",
        stage_index=0,
        protocol_sha256="1" * 64,
        normalization_sha256=normalization_hash,
    )


def _formal_runtime_args() -> Namespace:
    return Namespace(
        action_mode="action",
        chunk_len=10,
        force_window_len=20,
        force_window_duration=0.25,
        image_size=224,
    )


def test_formal_v2_rollout_binds_checkpoint_runtime_and_normalization():
    stats = _formal_stats()
    checkpoint = _formal_checkpoint(stats)

    verification = _validate_rollout_artifact_semantics(
        checkpoint, stats, _formal_runtime_args()
    )

    assert verification["status"] == "v2_verified"
    assert verification["normalization_semantic_sha256"] == stats[
        "normalization_content_sha256"
    ]


def test_formal_v2_rollout_rejects_checkpoint_integrity_tampering():
    stats = _formal_stats()
    checkpoint = _formal_checkpoint(stats)
    checkpoint["model_state_dict"]["weight"] = torch.full((1, 1), 5.0)

    with pytest.raises(ValueError, match="model_state_dict"):
        _validate_rollout_artifact_semantics(
            checkpoint, stats, _formal_runtime_args()
        )


def test_formal_checkpoint_cannot_downgrade_into_legacy_unverified_mode():
    stats = _formal_stats()
    checkpoint = _formal_checkpoint(stats)
    checkpoint.pop("schema_version")

    with pytest.raises(ValueError, match="formal checkpoint markers"):
        _validate_rollout_artifact_semantics(
            checkpoint, stats, _formal_runtime_args()
        )


def test_formal_v2_rollout_rejects_runtime_semantic_mismatch():
    stats = _formal_stats()
    checkpoint = _formal_checkpoint(stats)
    args = _formal_runtime_args()
    args.force_window_len = 21

    with pytest.raises(ValueError, match="checkpoint/runtime semantic mismatch"):
        _validate_rollout_artifact_semantics(checkpoint, stats, args)


def test_formal_v2_rollout_rejects_normalization_content_tampering():
    stats = _formal_stats()
    checkpoint = _formal_checkpoint(stats)
    stats["force_mean"] = torch.ones(6)

    with pytest.raises(ValueError, match="normalization semantic SHA256 mismatch"):
        _validate_rollout_artifact_semantics(
            checkpoint, stats, _formal_runtime_args()
        )


def test_summary_schema_includes_contact_recovery_outputs():
    assert {
        "recovery_success",
        "safe_recovery_success",
        "contact_recovery_metrics_valid",
        "contact_recovery_metrics",
    }.issubset(
        SUMMARY_REQUIRED_KEYS
    )
    assert {
        "checkpoint_file_sha256",
        "normalization_stats_file_sha256",
        "model_xml_file_sha256",
        "rollout_contract",
    }.issubset(SUMMARY_REQUIRED_KEYS)


def test_success_hold_counter_resets_on_failed_step():
    counter = 0
    for condition in [True, True, False, True, True, True]:
        counter = _update_success_hold_counter(counter, condition)

    assert counter == 3


def test_summary_schema_validation_accepts_required_keys():
    summary = {key: None for key in SUMMARY_REQUIRED_KEYS}

    _validate_summary_schema(summary)


def test_summary_schema_validation_rejects_missing_key():
    summary = {key: None for key in SUMMARY_REQUIRED_KEYS}
    summary.pop("summary_json", None)

    with pytest.raises(KeyError, match="summary is missing required keys"):
        _validate_summary_schema(summary)
