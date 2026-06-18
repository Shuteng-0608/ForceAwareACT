import numpy as np
import pytest

from scripts.run_mujoco_policy_rollout import (
    _interpret_selected_action,
    _selected_action_delta_norm_raw_to_current,
    _validate_stats_action_mode,
)


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


def test_stats_action_mode_mismatch_raises_clear_error():
    with pytest.raises(ValueError, match="action_mode mismatch"):
        _validate_stats_action_mode({"action_mode": "joint_pos"}, "action")


def test_missing_stats_action_mode_allows_legacy_joint_pos():
    _validate_stats_action_mode({}, "joint_pos")


def test_missing_stats_action_mode_rejects_command_modes():
    with pytest.raises(ValueError, match="do not contain action_mode metadata"):
        _validate_stats_action_mode({}, "delta_joint_cmd")
