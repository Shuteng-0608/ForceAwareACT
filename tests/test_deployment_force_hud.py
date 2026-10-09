"""HUD data semantics, causal display, independent process; no robot connection."""
import json
import socket
import time
from urllib.request import urlopen

import cv2
import numpy as np
import pytest

from force_aware_act.control.hud_telemetry import HudTelemetry
from force_aware_act.control.wrench_estimation import flange_jacobian, equivalent_wrench
from force_aware_act.visualization.deployment_hud import DeploymentHud, HudProcess, read_config
from force_aware_act.visualization.tianji_hud_render import RingMapping
from scripts.deployment.run_contact_controller import parse_arguments, main
from scripts.deployment.observe_force_hud_readonly import calibration_candidate

Q=np.array([-1.5,-1.34,1.58,-1.43,.22,-.065,.034])
GOAL={'goal_id':'x13y07','target_position_m':[.517,-.317,.250],
      'config':{'limits':{'torque_soft_nm':[17.5,6.5,5.,11.,1.,6.5,2.]}}}


def packet(t=1_000_000_000, tau=None):
    tau=np.zeros(7) if tau is None else np.asarray(tau)
    bias=np.arange(7)*.1
    return {'record':{'received_monotonic_ns':t,'qpos_rad':Q.tolist(),
                     'external_raw_nm':(tau+bias).tolist(),'internal_torque_nm':[999.]*7},
            'bias_nm':bias.tolist(),'bias_valid':True,'state':'RUNNING',
            'calibration_status':'valid','stop_reason':None}


def test_latest_snapshot_overwrites_without_backlog_or_waiting():
    channel=HudTelemetry(); assert channel.read() is None
    assert channel.publish({'frame':1}); assert channel.publish({'frame':2})
    assert channel.read()=={'frame':2}
    channel.lock.acquire()
    try:
        assert not channel.publish({'frame':3})
        assert channel.read() is None
    finally:
        channel.lock.release()
    assert channel.read()=={'frame':2}


def test_hud_uses_external_estimate_and_exact_session_bias():
    tau=flange_jacobian(Q).T@np.array([2.,3.,4.,.1,.2,.3])
    hud=DeploymentHud(read_config(),GOAL); p=packet(tau=tau)
    hud.update(p,p['record']['received_monotonic_ns'])
    np.testing.assert_allclose(hud.estimate['zeroed_tau_nm'],tau,atol=1e-15)
    np.testing.assert_allclose(hud.estimate['wrench_base'],[2.,3.,4.,.1,.2,.3],atol=1e-12)
    assert hud.status(1_000_000_000)['valid']
    image,_=hud.render(1_000_000_000); assert image.shape==(720,1280,3)


def test_high_residual_remains_visible_with_diagnostic():
    j=flange_jacobian(Q); _,_,vt=np.linalg.svd(j,full_matrices=True)
    tau=vt[-1]+j.T@np.array([.1,.2,.3,0,0,0])
    hud=DeploymentHud(read_config(),GOAL); hud.update(packet(tau=tau),1_000_000_000)
    assert hud.estimate['relative_residual']>.5
    assert hud.status(1_000_000_000)['valid']
    assert 'HIGH RESIDUAL' in hud.estimate['diagnostics']
    assert RingMapping(**read_config()['mapping']).project(hud.model.arms['right']) is not None


def test_only_causal_samples_drive_filter_and_duplicate_does_not_refresh_age():
    hud=DeploymentHud(read_config(),GOAL); j=flange_jacobian(Q)
    first=np.array([1.,0,0,0,0,0]); second=np.array([10.,0,0,0,0,0])
    hud.update(packet(tau=j.T@first),1_000_000_000)
    p=packet(t=1_080_000_000,tau=j.T@second); hud.update(p,1_080_000_000)
    expected=first+(1-np.exp(-1))*(second-first)
    np.testing.assert_allclose(hud.model.arms['right'].filtered,expected,atol=1e-12)
    hud.update(p,1_100_000_000)
    np.testing.assert_allclose(hud.model.arms['right'].filtered,expected,atol=1e-12)
    assert not hud.status(1_300_000_000)['valid']


def test_calibration_required_and_context_change_hides_old_estimate():
    hud=DeploymentHud(read_config(),GOAL); p=packet(); p['bias_nm']=None
    hud.update(p,1_000_000_000); assert not hud.status(1_000_000_000)['valid']
    p=packet(t=1_010_000_000); hud.update(p,1_010_000_000)
    assert hud.status(1_010_000_000)['valid']
    p['bias_valid']=False; hud.update(p,1_020_000_000)
    assert not hud.status(1_020_000_000)['valid'] and hud.estimate is None


def test_hud_preview_never_starts_devices(capsys):
    assert main(['--goal-id','x13y07','--hud'])==0
    preview=json.loads(capsys.readouterr().out)
    assert preview['hud']['requested'] and not preview['hud']['started']
    assert not preview['hardware_connected'] and not preview['commands_sent']
    with pytest.raises(SystemExit):
        parse_arguments(['--goal-id','x13y07','--hud','--hud-port','0'])


def test_display_process_http_switches_only_presentation(tmp_path):
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0)); port=sock.getsockname()[1]
    channel=HudTelemetry()
    process=HudProcess(channel,read_config(),GOAL,tmp_path/'hud',port)
    try:
        for _ in range(10):
            now=time.monotonic_ns(); channel.publish(packet(t=now,tau=flange_jacobian(Q).T@np.ones(6)))
            time.sleep(.02)
        with urlopen(process.url+'/status.json',timeout=2) as response:
            status=json.load(response)
        assert status['valid'] and status['state']=='RUNNING'
        with urlopen(process.url+'/frame.jpg',timeout=2) as response:
            decoded=cv2.imdecode(np.frombuffer(response.read(),dtype=np.uint8),cv2.IMREAD_COLOR)
        assert decoded.shape==(720,1280,3)
        with urlopen(process.url+'/view?mode=bars&raw=1',timeout=2) as response:
            assert response.status==200
        time.sleep(.04)
        with urlopen(process.url+'/status.json',timeout=2) as response:
            status=json.load(response)
        assert status['display_mode']=='bars' and status['raw_display']
    finally:
        report=process.close()
    assert not report['errors']
    assert not report['torch_imported'] and not report['camera_library_imported']
    assert not report['robot_transport_imported'] and not report['commands_sent']


def test_readonly_zero_uses_complete_stationary_window_and_records_context():
    rows=[]
    for t in range(1_000_000_000,3_010_000_000,10_000_000):
        r=packet(t)['record'];r.update(qvel_rad_s=[0.]*7,state=0,tool_kine=[0.]*6,tool_dyn=[0.]*10)
        rows.append(r)
    assert calibration_candidate(rows[:-1]) is None
    out=calibration_candidate(rows)
    np.testing.assert_allclose(out['bias_nm'],np.arange(7)*.1)
    assert out['end_ns']-out['start_ns']==2_000_000_000
    assert out['context'][0]==0
