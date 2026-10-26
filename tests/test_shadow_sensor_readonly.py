"""Read-only integration contract and HUD without invented joint tare."""
import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.deployment.shadow_sensor_readonly import main, parse_arguments, append_feedback, stats
from force_aware_act.visualization.deployment_hud import DeploymentHud, read_config


def sample():
    return dict(received_monotonic_ns=1_000_000_000, frame_serial=12, qpos_rad=[0.]*7,
        sensor_ft=dict(valid=True,source_ns=1_000_000_000,frame_serial=12,
            wrench_si=[1.,2.,3.,.1,.2,.3],frame_id='right_ft_measurement',
            sensor_to_flange_rotation=np.eye(3).tolist()))


def test_preview_and_explicit_live_confirmation(capsys):
    assert main([])==0
    assert json.loads(capsys.readouterr().out)['hardware_connected'] is False
    with pytest.raises(SystemExit):parse_arguments(['--connect'])
    with pytest.raises(SystemExit):parse_arguments(['--seconds','nan'])


def test_same_frame_sensor_only_and_causal_availability():
    calls=[]
    buffer=SimpleNamespace(append_sensor_wrench=lambda *a,**kw:calls.append(('sensor',a,kw)),
                           append_state=lambda *a,**kw:calls.append(('state',a,kw)))
    clock=SimpleNamespace(map=lambda t:t+50)
    row=sample()
    assert append_feedback(buffer,row,1_001_000_000,clock)
    assert [c[0] for c in calls]==['sensor','state']
    np.testing.assert_equal(calls[0][1][1],row['sensor_ft']['wrench_si'])
    assert calls[0][2]['available_ns']==1_001_000_050
    calls.clear()
    assert not append_feedback(buffer,row,1_030_000_000,clock)
    assert not append_feedback(buffer,row,999_000_000,clock)
    row['sensor_ft']['valid']=False
    assert not append_feedback(buffer,row,1_001_000_000,clock)
    assert not calls
    row=sample();row['sensor_ft']['frame_serial']=13
    with pytest.raises(ValueError,match='frame mismatch'):
        append_feedback(buffer,row,1_001_000_000,clock)


def test_sensor_hud_needs_no_joint_tare_or_goal():
    goal=dict(goal_id='not selected',target_position_m=None,
              config={'limits':{'torque_soft_nm':[1.]*7}})
    hud=DeploymentHud(read_config('configs/deployment/force_hud_sensor.json'),goal)
    packet=dict(record=sample(),bias_nm=None,bias_valid=True,state='READONLY_PREDICTING',
                calibration_status='valid',stop_reason=None,source_mode='readonly_feedback')
    hud.update(packet,1_001_000_000)
    assert hud.status(1_001_000_000)['valid']
    assert hud.estimate['zeroed_tau_nm'] is None
    assert hud.estimate['distance_mm'] is None
    assert hud.render(1_001_000_000)[0].shape==(720,1280,3)
    packet=copy.deepcopy(packet);packet['bias_valid']=False
    hud.update(packet,1_002_000_000)
    assert not hud.status(1_002_000_000)['valid']


def test_stats_accepts_numpy_intervals():
    assert stats(np.array([]))=={'count':0}
    assert stats(np.array([1.,2.]))['max']==2.
