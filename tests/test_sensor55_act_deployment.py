"""Sensor55 ACT identity, policy inputs and read-only reporting; no devices."""
from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts.deployment import run_sensor55_act_x14y08 as entry
from scripts.deployment.shadow_sensor_readonly import prediction_record
from force_aware_act.inference.real_robot import RealRobotObservationBuffer, RealRobotPolicyAdapter
from force_aware_act.real_robot.config import RealRobotDataConfig, WRENCH_UNITS
from force_aware_act.real_robot.training import build_model, make_model_config


def test_act1_preview_selects_new_library_checkpoint_and_keeps_hardware_disabled(capsys):
    assert entry.main([]) == 0
    preview = json.loads(capsys.readouterr().out)
    assert preview['model_kind'] == 'act'
    assert preview['checkpoint_sha256'] == entry.EXPECTED_MODEL_SHA
    assert preview['model_manifest']['model_library_id'] == 1
    assert preview['model_manifest']['force_input'] == 'none'
    assert preview['policy_inputs'] == ['RGB d455', 'RGB d405', 'qpos_rad']
    assert preview['session']['ft_force_hard_n'] == [20, 20, 60]
    assert preview['session']['servo_off_after_standstill'] is True
    assert preview['session']['servo_off_on_fault'] is True
    assert preview['goal_selection']['goal_id'] == 'x14y08'
    assert preview['goal_selection']['config']['action_chunk_len'] == 20
    assert preview['goal_selection']['config']['max_prediction_age_ms'] == 200
    assert 'fresh observations and [20,7] prediction' in preview['sequence']
    assert not preview['hardware_connected'] and not preview['commands_sent']
    args = entry.controller_arguments([])
    checkpoint = args[args.index('--checkpoint')+1]
    assert '/model_library/1mm_x14y08_55_v3_20261026/' in checkpoint
    assert checkpoint.endswith('/01_act_none_no_history_zero/best.pt')
    with pytest.raises(SystemExit):
        entry.controller_arguments(['--checkpoint', 'old_baseline.pt'])


def test_act_hardware_entry_rejects_a_session_without_immediate_protection(tmp_path):
    old = json.loads((entry.ROOT/'configs/deployment/command_session_sensor55_candidate.json').read_text())
    old.pop('servo_off_on_fault')
    old['allow_hardware'] = True
    path = tmp_path/'previous_session.json'
    path.write_text(json.dumps(old))
    with pytest.raises(SystemExit):
        entry.controller_arguments(['--enable-control', '--session-config', str(path)])


@pytest.mark.parametrize('field,value', [
    ('model_kind', 'contact'),
    ('checkpoint_sha256', '0'*64),
    ('action_chunk_len', 100),
    ('force_frame', 'base'),
])
def test_act_entry_rejects_wrong_model_contract(tmp_path, monkeypatch, field, value):
    binding = json.loads(entry.BINDING.read_text())
    manifest = json.loads((entry.ROOT/binding['model_manifest']).read_text())
    manifest[field] = value
    manifest_path = tmp_path/'manifest.json'
    manifest_path.write_text(json.dumps(manifest))
    binding['model_manifest'] = str(manifest_path)
    binding_path = tmp_path/'binding.json'
    binding_path.write_text(json.dumps(binding))
    monkeypatch.setattr(entry, 'BINDING', binding_path)
    with pytest.raises(ValueError, match='pinned sensor55 act'):
        entry.controller_arguments([])


def test_sensor55_act_output_and_readonly_record_ignore_force_history():
    config = RealRobotDataConfig(image_height=32, image_width=32,
        force_representation='end_effector_wrench6', wrench_component_units=WRENCH_UNITS,
        wrench_frame='right_ft_measurement', force_std_floor=(.01,)*6)
    model = build_model('act', asdict(make_model_config('act', config, compact=True, chunk_len=20)),
                        initialize_pretrained=False)
    norm = {f'{name}_{key}': [value]*7 for name in ('qpos', 'action')
            for key, value in (('mean', 0.), ('std', 1.))}
    adapter = RealRobotPolicyAdapter(model, model_kind='act', normalization=norm,
        data_config=config, data_contract=config.contract_dict())
    predictions = []
    for force in (0., 100.):
        buffer = RealRobotObservationBuffer(config)
        anchor = 1_000_000_000
        buffer.append_state(anchor, np.zeros(7))
        buffer.append_sensor_wrench(anchor, np.full(6, force), frame_id='right_ft_measurement')
        for name in config.camera_names:
            buffer.append_image(name, anchor, np.zeros((32,32,3), dtype=np.uint8))
        inputs = adapter.prepare_observation(buffer, anchor)
        assert inputs.online_force_intervals is None
        output = adapter.predict_prepared(inputs)
        assert output['pred_action'].shape == (20, 7)
        row = prediction_record(inputs, output, anchor, 1.)
        assert row['valid_force_samples'] == 0 and row['force_history_used'] is False
        assert row['prediction_shapes'] == {'pred_action': [20, 7]}
        assert row['motor_commands_sent'] is False
        predictions.append(output['pred_action'])
    np.testing.assert_array_equal(*predictions)


def test_contact_readonly_record_still_counts_force_samples():
    source = SimpleNamespace(qpos=np.zeros(7), anchor_ns=1_000_000_000,
        state_timestamp_ns=999_000_000, image_timestamps_ns=(990_000_000, 980_000_000))
    mask = torch.tensor([[[False, True], [False, False]]])
    inputs = SimpleNamespace(source=source, online_force_sample_padding_mask=mask,
        online_force_intervals=torch.zeros((1,2,2,6)))
    row = prediction_record(inputs, {'pred_action': np.zeros((20,7))}, 1_000_000_000, 3.)
    assert row['valid_force_samples'] == 3 and row['force_history_used'] is True
    assert row['state_age_ms'] == 1. and row['image_age_ms'] == [10., 20.]
