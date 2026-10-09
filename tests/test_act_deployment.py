"""ACT entry identity and shared manual execution gates; no device imports."""
import hashlib
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from scripts.deployment import run_act_controller as act
from scripts.deployment import run_contact_controller as shared


def test_act_preview_is_disabled_and_retains_external_safety(capsys):
    assert act.main(["--goal-id", "x13y07"]) == 0
    preview = json.loads(capsys.readouterr().out)
    assert preview["model_kind"] == "act"
    assert preview["policy_inputs"] == ["RGB d455", "RGB d405", "qpos_rad"]
    assert preview["external_torque_safety_enabled"]
    assert not preview["hardware_connected"] and not preview["commands_sent"]
    assert preview["checkpoint_sha256"] == shared.VALIDATED_CHECKPOINTS["act"]


def test_contact_entry_remains_pinned_to_contact(capsys):
    assert shared.main(["--goal-id", "x13y07"]) == 0
    preview = json.loads(capsys.readouterr().out)
    assert preview["model_kind"] == "contact"
    assert "external_torque_history" in preview["policy_inputs"]
    assert preview["checkpoint_sha256"] == shared.CHECKPOINT_SHA256


def test_act_enabled_run_requires_hardware_config_and_confirmation():
    with pytest.raises(SystemExit):
        act.main(["--goal-id", "x13y07", "--enable-control"])


def test_act_wrapper_dispatches_only_act(monkeypatch):
    calls = []
    def fake_main(argv, *, policy_kind):
        calls.append((argv, policy_kind))
        return 7
    monkeypatch.setattr(act, "_shared_main", fake_main)
    assert act.main(["--goal-id", "x13y07"]) == 7
    assert calls == [(["--goal-id", "x13y07"], "act")]


def test_act_checkpoint_check_dispatches_only_act(monkeypatch):
    from scripts.deployment import check_act_checkpoint as check
    calls = []
    def fake_main(argv, *, model_kind):
        calls.append((argv, model_kind))
    monkeypatch.setattr(check, "_shared_main", fake_main)
    check.main(["--checkpoint", "fixture.pt"])
    assert calls == [(["--checkpoint", "fixture.pt"], "act")]


def test_act_prediction_does_not_depend_on_external_torque():
    from dataclasses import asdict
    import numpy as np
    from force_aware_act.inference.real_robot import RealRobotPolicyAdapter, RealRobotObservationBuffer
    from force_aware_act.real_robot.config import RealRobotDataConfig
    from force_aware_act.real_robot.training import build_model, make_model_config
    config = RealRobotDataConfig(image_height=32, image_width=32, history_duration_s=.02)
    model = build_model("act", asdict(make_model_config("act", config, compact=True, chunk_len=3)),
                        initialize_pretrained=False)
    stats = {f"{name}_{key}": [value]*7 for name in ("qpos", "action")
             for key, value in (("mean", 0.), ("std", 1.))}
    adapter = RealRobotPolicyAdapter(model, model_kind="act", normalization=stats,
                                    data_config=config, data_contract=config.contract_dict())
    outputs = []
    for torque in (0., 20.):
        buffer = RealRobotObservationBuffer(config)
        buffer.append_state(1_000_000_000, np.zeros(7))
        buffer.append_external_torque(1_000_000_000, np.full(7, torque))
        for camera in config.camera_names:
            buffer.append_image(camera, 1_000_000_000, np.zeros((32,32,3), dtype=np.uint8))
        inputs = adapter.prepare_observation(buffer, 1_000_000_000)
        assert inputs.online_force_intervals is None
        outputs.append(adapter.predict_prepared(inputs)["pred_action"])
    np.testing.assert_array_equal(*outputs)


def test_checkpoint_pin_is_model_specific_and_checked_before_devices(tmp_path, monkeypatch):
    path = tmp_path / "checkpoint.pt"
    path.write_bytes(b"small identity fixture")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setitem(shared.VALIDATED_CHECKPOINTS, "act", digest)
    assert shared.validate_checkpoint(path, "act") == digest
    with pytest.raises(ValueError, match="validated contact"):
        shared.validate_checkpoint(path, "contact")
    _, selected, options = shared.parse_arguments(["--goal-id", "x13y07"])
    args = SimpleNamespace(enable_control=True, operator_confirmed_no_contact=True, checkpoint=path)
    with pytest.raises(ValueError, match="validated contact"):
        shared._run_live_owned(args, selected, replace(options, allow_hardware=True))
