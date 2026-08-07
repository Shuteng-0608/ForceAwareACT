import numpy as np
import pytest
import torch

from force_aware_act.data import normalize_tensor
from force_aware_act.inference import (
    SignedAgeTemporalActionChunkExecutor,
    TemporalEndpointActionChunkExecutor,
)

from scripts.run_mujoco_policy_rollout import (
    LATEST_ONLY_ACTION_SELECT_MODE,
    SIGNED_TEMPORAL_ACTION_SELECT_MODE,
    SUMMARY_REQUIRED_KEYS,
    _build_temporal_executor,
    _interpret_selected_action,
    _measure_policy_inference,
    _resolve_inference_device,
    _run_mode,
    _signed_temporal_decay_equivalent,
    _stats_to_device,
    _summarize_latency_ms,
    _selected_action_delta_norm_raw_to_current,
    _selected_action_index,
    _validate_stats_action_mode,
    _validate_summary_schema,
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
    assert _selected_action_index(10, "recency_temporal") == -1
    assert _selected_action_index(10, "signed_temporal") == -1
    assert _selected_action_index(10, "latest_only") == -1
    assert _selected_action_index(10, "receding_chunk") == -1


def test_temporal_executor_builder_exposes_q1_signed_and_latest_modes():
    signed = _build_temporal_executor(SIGNED_TEMPORAL_ACTION_SELECT_MODE, -0.3)
    latest = _build_temporal_executor(LATEST_ONLY_ACTION_SELECT_MODE, 123.0)

    assert isinstance(signed, SignedAgeTemporalActionChunkExecutor)
    assert signed.signed_decay == -0.3
    assert isinstance(latest, TemporalEndpointActionChunkExecutor)
    assert latest.preference == "newest"


def test_temporal_modes_report_equivalent_signed_age_parameter():
    assert _signed_temporal_decay_equivalent("temporal", 0.01) == -0.01
    assert _signed_temporal_decay_equivalent("recency_temporal", 0.03) == 0.03
    assert _signed_temporal_decay_equivalent("signed_temporal", -0.3) == -0.3
    assert _signed_temporal_decay_equivalent("latest_only", 0.01) is None


def test_cpu_policy_inference_timer_and_latency_summary():
    output, elapsed_ms = _measure_policy_inference(
        lambda: {"value": 3},
        torch.device("cpu"),
    )
    summary = _summarize_latency_ms([1.0, 2.0, 3.0, float("nan")])

    assert output == {"value": 3}
    assert elapsed_ms >= 0.0
    assert summary == pytest.approx(
        {"mean": 2.0, "p50": 2.0, "p95": 2.9, "max": 3.0}
    )


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


def test_summary_schema_validation_accepts_required_keys():
    summary = {key: None for key in SUMMARY_REQUIRED_KEYS}

    _validate_summary_schema(summary)


def test_summary_schema_validation_rejects_missing_key():
    summary = {key: None for key in SUMMARY_REQUIRED_KEYS}
    summary.pop("summary_json", None)

    with pytest.raises(KeyError, match="summary is missing required keys"):
        _validate_summary_schema(summary)
