"""Local mathematical and nominal data-path tests. No robot SDK is loaded."""
from dataclasses import asdict, replace
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from force_aware_act.control.sensor_ft import SensorFT, MODEL_SHA256
from force_aware_act.control.kinematics import right_flange_transform
from force_aware_act.control.command_session import CommandSession, SessionOptions
from force_aware_act.control.deployment_model import adapter_contract
from force_aware_act.control.goal_selection import select_goal
from force_aware_act.control.execution import TemporalExecutor, ActionUnavailable
from force_aware_act.inference.real_robot import RealRobotObservationBuffer, RealRobotPolicyAdapter
from force_aware_act.real_robot.config import RealRobotDataConfig
from force_aware_act.real_robot.training import build_model, make_model_config
from force_aware_act.visualization.deployment_hud import DeploymentHud, read_config
from scripts.deployment.build_ft_compensation import build

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def library(tmp_path_factory):
    return build(tmp_path_factory.mktemp('ft_math')/'compensation.so')


@pytest.fixture
def calibration(tmp_path):
    data = dict(schema_version=1, kind='real_telemetry', validation_passed=True,
        model_sha256=MODEL_SHA256, gravity_base_m_s2=[0., -9.81, 0.],
        parameters=dict(mass_kg=.3, com_sensor_m=[.01, -.02, .05],
                        force_bias_N=[1., 2., 3.], torque_bias_Nm=[.01, .02, .03]),
        sensor=dict(axes_confirmed=True, frame_id='right_ft_measurement',
                    force_sign=1., torque_sign=1., R_flange_sensor=[[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]]))
    path = tmp_path/'current.yaml'
    path.write_text(yaml.safe_dump(data))
    return path, data


def make_row(data, q, ns, residual=None):
    rotation = right_flange_transform(q)[:3,:3] @ np.asarray(data['sensor']['R_flange_sensor'])
    parameters = data['parameters']
    gravity = parameters['mass_kg'] * rotation.T @ data['gravity_base_m_s2']
    gravity6 = np.r_[gravity, np.cross(parameters['com_sensor_m'], gravity)]
    fixed = np.r_[parameters['force_bias_N'], parameters['torque_bias_Nm']]
    measured = (gravity6+fixed+(np.zeros(6) if residual is None else residual))
    return dict(qpos_rad=np.asarray(q).tolist(), qvel_rad_s=[0.]*7,
                received_monotonic_ns=ns, frame_serial=ns//10_000_000,
                state=3, low_speed_flag=1, ft_channel=216,
                ft_raw_counts=(measured*10000.).tolist())


def test_gravity_updates_with_same_frame_pose_and_task_tare_is_fixed(library, calibration, tmp_path):
    path, data = calibration
    sensor = SensorFT(path, library)
    q = np.array([.1, -.3, .2, -.4, .5, -.6, .7])
    offset = np.array([.3, -.2, .1, .002, -.001, .003])
    try:
        for n in range(501):
            row = make_row(data, q, n*10_000_000, offset)
            out = sensor.process(row, collect=True)
            assert not out['valid']
            np.testing.assert_allclose(out['compensated_si'], offset, atol=1e-14)
        assert sensor.capture(5_000_000_000, tmp_path/'tare')
        np.testing.assert_allclose(sensor.tare['bias_N_Nm'], offset, atol=1e-14)
        external = np.array([2.,3.,4.,.1,.2,.3])
        q += .07
        row = make_row(data, q, 5_010_000_000, offset+external)
        out = sensor.process(row)
        np.testing.assert_allclose(out['wrench_si'], external, atol=1e-13)
        assert out['frame_serial'] == row['frame_serial'] and out['source_ns'] == row['received_monotonic_ns']
        assert len(list((tmp_path/'tare').glob('*.yaml'))) == 1
        assert sensor.capture(5_010_000_000, tmp_path/'tare')
        np.testing.assert_allclose(sensor.tare['bias_N_Nm'], offset, atol=1e-14)
    finally:
        sensor.close()


def test_each_new_session_loads_overwritten_calibration(library, calibration):
    path, data = calibration
    first = SensorFT(path, library)
    data['parameters']['mass_kg'] = .42
    path.write_text(yaml.safe_dump(data))
    second = SensorFT(path, library)
    try:
        assert first.report()['parameters']['mass_kg'] == .3
        assert second.report()['parameters']['mass_kg'] == .42
        assert second.tare is None
        row = make_row(data, np.zeros(7), 10_000_000)
        np.testing.assert_allclose(second.process(row)['compensated_si'], 0, atol=1e-14)
    finally:
        first.close(); second.close()


def sensor_config(dim):
    return RealRobotDataConfig.from_dict(json.loads((ROOT/('configs/real_robot/right_arm_sensor_force3.json'
        if dim == 3 else 'configs/real_robot/right_arm_wrench6.json')).read_text()))


@pytest.mark.parametrize('dim', [3,6])
@pytest.mark.parametrize('kind', ['contact','motion','act'])
def test_sensor_model_input_units_channels_normalization_and_causality(dim, kind):
    config = replace(sensor_config(dim), image_height=32, image_width=32)
    horizon = {} if kind == "act" else dict(force_prediction_steps=20, strict_force_horizon=True)
    mc = make_model_config(kind, config, compact=True, chunk_len=20, pretrained=False, **horizon)
    model = build_model(kind, asdict(mc), initialize_pretrained=False)
    stats = {f'{name}_{key}':[v]*size for name,size in [('qpos',7),('action',7),('force',dim)]
             for key,v in [('mean',1.),('std',2.)]}
    adapter = RealRobotPolicyAdapter(model, model_kind=kind, normalization=stats,
                                    data_config=config, data_contract=config.contract_dict())
    buffer = RealRobotObservationBuffer(config)
    anchor = 1_000_000_000
    wrench = np.array([2.,4.,6.,.2,.4,.6])
    for ns in range(anchor-200_000_000, anchor+1, 3_333_333):
        buffer.append_state(ns, np.zeros(7))
        buffer.append_sensor_wrench(ns, wrench, frame_id=config.wrench_frame)
    for name in config.camera_names:
        buffer.append_image(name, anchor, np.zeros((32,32,3),dtype=np.uint8))
    before = adapter.prepare_observation(buffer, anchor)
    assert before.source.torque_values.shape[1] == dim
    np.testing.assert_allclose(before.source.torque_values[-1], wrench[:dim])
    buffer.append_sensor_wrench(anchor+1, np.ones(6)*99, frame_id=config.wrench_frame)
    after = adapter.prepare_observation(buffer, anchor)
    if kind != 'act':
        torch.testing.assert_close(before.online_force_intervals, after.online_force_intervals)
        values = before.online_force_intervals[~before.online_force_sample_padding_mask]
        torch.testing.assert_close(values, torch.tensor((wrench[:dim]-1)/2, dtype=torch.float32).expand_as(values))
    with torch.no_grad():
        prediction = adapter.predict_prepared(after)
    assert prediction['pred_action'].shape == (20,7)
    assert np.isfinite(prediction['pred_action']).all()
    contract = adapter_contract(adapter)
    assert contract == json.loads(json.dumps(contract))
    assert contract['force_dim'] == dim and contract['action_chunk_len'] == 20


def test_nominal_controller_waits_for_sensor_tare_and_hud_uses_sensor_origin(library, calibration, tmp_path):
    spec = importlib.util.spec_from_file_location('command_helpers', Path(__file__).with_name('test_command_session.py'))
    helper = importlib.util.module_from_spec(spec); spec.loader.exec_module(helper)
    path, data = calibration
    selected = select_goal('x13y07')
    q = (np.array(selected.config.limits.joint_min_rad)+selected.config.limits.joint_max_rad)/2
    transport = helper.JournalTransport(q)
    sensor = SensorFT(path, library)
    options = SessionOptions(ft_calibration_file=str(path), ft_compensation_library=str(library))
    session = CommandSession(selected, options, transport, run_id='sensor_nominal', origin_ns=0,
        enable_control=True, operator_confirmed_no_contact=True, sensor_ft=sensor, output=tmp_path)
    try:
        for ns in range(0, 8_000_000_000, 10_000_000):
            row = transport.row(ns)
            source = make_row(data, q, ns)
            row.update({k:source[k] for k in ['ft_channel','ft_raw_counts','low_speed_flag']})
            session.feedback(row, ns); session.poll(ns)
            if session.state == 'READY': break
        assert session.state == 'READY', session.report()
        assert sensor.tare['span_s'] >= 5
        assert row['sensor_ft']['valid'] and session.last['sensor_ft']['valid']
        assert [call[0] for call in transport.calls] == ['parameters','mode']
        config = read_config(ROOT/'configs/deployment/force_hud_sensor.json')
        hud = DeploymentHud(config, selected.to_dict())
        row['sensor_ft']['wrench_si'] = [1.,2.,3.,.1,.2,.3]
        from force_aware_act.control.hud_telemetry import session_snapshot
        hud.update(session_snapshot(session, row, ns), ns)
        image, status = hud.render(ns)
        assert status['valid'], status
        assert status['estimate']['reference_point'] == 'sensor'
        r = right_flange_transform(q)[:3,:3] @ np.array(data['sensor']['R_flange_sensor'])
        np.testing.assert_allclose(status['estimate']['wrench_base'], np.r_[r @ [1.,2.,3.],r @ [.1,.2,.3]])
        assert image.shape == (720,1280,3)
        assert status['estimate']['relative_residual'] is None
    finally:
        sensor.close()


@pytest.fixture
def stopped_sensor_session(library, calibration, tmp_path):
    spec = importlib.util.spec_from_file_location('stop_helpers', Path(__file__).with_name('test_command_session.py'))
    helper = importlib.util.module_from_spec(spec); spec.loader.exec_module(helper)
    path, data = calibration
    selected = select_goal('x14y08', ROOT/'configs/deployment/profiles/contact_sensor55_x14y08_v1')
    q = (np.array(selected.config.limits.joint_min_rad)+selected.config.limits.joint_max_rad)/2
    transport = helper.JournalTransport(q)
    sensor = SensorFT(path, library)
    offset = np.array([.1,.2,.3,.001,.002,.003])
    options = SessionOptions(ft_calibration_file=str(path), ft_compensation_library=str(library),
                             ft_force_hard_n=(20.,20.,60.), ft_torque_hard_nm=(1.15,2.55,.24))
    session = CommandSession(selected, options, transport, run_id='stop_sensor', origin_ns=0,
        enable_control=True, operator_confirmed_no_contact=True, sensor_ft=sensor, output=tmp_path)

    def row(ns, wrench):
        value = transport.row(ns)
        source = make_row(data, q, ns, offset+np.asarray(wrench))
        value.update({k:source[k] for k in ['ft_channel','ft_raw_counts','low_speed_flag']})
        return value

    try:
        for ns in range(0, 8_000_000_000, 10_000_000):
            session.feedback(row(ns, np.zeros(6)), ns); session.poll(ns)
            if session.state == 'READY': break
        assert session.state == 'READY', session.report()
        yield session, transport, ns, row
    finally:
        sensor.close()


@pytest.mark.parametrize('servo_off', [False, True])
def test_tared_sensor_remains_observable_after_revocation_and_latches_hard_force(stopped_sensor_session, servo_off):
    session, transport, start, row = stopped_sensor_session
    session.options = replace(session.options, servo_off_after_standstill=servo_off)
    stop = start+10_000_000
    session.stop('operator_stop', stop)
    for ns in range(stop+10_000_000, stop+1_000_000_000, 10_000_000):
        frame = row(ns, [0.,0.,-70.,0.,0.,0.])
        session.feedback(frame, ns)
        assert frame['sensor_ft']['valid'] and session.tick(ns, ns) is None
        if session.terminal: break
    report = session.report()
    assert report['outcome'] == 'aborted' and report['stop']['reason'] == 'operator_stop'
    violation = report['stop']['sensor_hard_after_stop']['violations'][0]
    assert violation['axis'] == 'Fz' and violation['limit'] == 60
    assert violation['value'] == pytest.approx(-70)
    assert report['stop']['sensor_monitor_after_stop']['peak_abs_N_Nm'][2] == pytest.approx(70)
    assert session.packet_counts['target'] == 0 and transport.revoked and session.bias is None
    assert [x[0] for x in transport.calls] == ['parameters','mode']+(['servo_off'] if servo_off else [])
    assert session.state == ('SERVO_OFF_CONFIRMED' if servo_off else 'STOPPED_HOLDING_LAST_TARGET')
    from force_aware_act.control.hud_telemetry import session_snapshot
    packet = session_snapshot(session, frame, ns)
    assert packet['bias_valid']  # Sensor contact must remain visible after IDLE.
    if servo_off: assert packet['bias_nm'] is None
    hud = DeploymentHud(read_config(ROOT/'configs/deployment/force_hud_sensor.json'),session.selection.to_dict())
    hud.update(packet,ns)
    assert hud.status(ns)['valid']
    np.testing.assert_allclose(hud.estimate['wrench_sensor'],[0,0,-70,0,0,0],atol=1e-12)


def test_post_stop_force_read_failure_is_not_hidden_by_later_recovery(stopped_sensor_session):
    session, transport, start, row = stopped_sensor_session
    session.stop('operator_stop', start+10_000_000)
    ns = start+20_000_000
    frame = row(ns, np.zeros(6)); frame['ft_channel'] = 0
    session.feedback(frame, ns)
    assert 'sensor_ft' not in frame and session.stop_request['sensor_ft_error_after_stop']
    session.feedback(row(ns+10_000_000, np.zeros(6)), ns+10_000_000)
    status = session.stop_request['sensor_monitor_after_stop']
    assert status['error_samples'] == 1 and status['valid_samples'] == 1
    assert session.report()['outcome'] == 'aborted' and session.packet_counts['target'] == 0


def test_post_stop_monitor_records_force_when_additional_thresholds_are_unset(stopped_sensor_session):
    session, transport, start, row = stopped_sensor_session
    session.options = replace(session.options, ft_force_hard_n=None, ft_torque_hard_nm=None)
    session.stop('operator_stop', start+10_000_000)
    ns = start+20_000_000
    session.feedback(row(ns, [0,0,-70,0,0,0]), ns)
    assert session.stop_request['sensor_monitor_after_stop']['valid_samples'] == 1
    assert 'sensor_hard_after_stop' not in session.stop_request
    assert session.report()['outcome'] == 'ended' and session.packet_counts['target'] == 0


def test_twenty_point_chunk_uses_timestamp_indices_and_expires():
    config = replace(select_goal('x13y07').config, action_chunk_len=20)
    executor = TemporalExecutor(0,'unit',config)
    executor.submit(np.tile(np.arange(20)[:,None],(1,7)), query_id=1, start_ns=0,
                    observed_ns=0,available_ns=0,run_id='unit',now_ns=0)
    assert executor.sample(190_000_000)['action_rad'] == [19.]*7
    with pytest.raises(ActionUnavailable): executor.sample(200_000_000)


def test_sensor_checkpoint_manifest_and_preview_need_no_devices(tmp_path, calibration, library, capsys):
    from scripts.deployment.prepare_sensor_model import main as prepare
    from scripts.deployment.run_contact_controller import main as preview
    path, _ = calibration
    config = replace(sensor_config(3), image_height=32, image_width=32)
    mc = make_model_config('contact', config, compact=True, chunk_len=20, pretrained=False,
                           force_prediction_steps=20, strict_force_horizon=True)
    model = build_model('contact', asdict(mc), initialize_pretrained=False)
    stats = {f'{name}_{key}':[v]*size for name,size in [('qpos',7),('action',7),('force',3)]
             for key,v in [('mean',0.),('std',1.)]}
    checkpoint = tmp_path/'model.pt'
    torch.save(dict(format_version='real_robot_training_v1',model_kind='contact',model_config=asdict(mc),
        data_config=config.to_dict(),data_contract=config.contract_dict(),normalization=stats,
        model_state=model.state_dict(), split_manifest={}, source_fingerprints={}, training_config={},
        run_config={}, optimizer_state={}, global_step=0, rng_state={}, sampler_state={}), checkpoint)
    manifest = tmp_path/'deployment.json'
    prepare(['--checkpoint',str(checkpoint),'--output',str(manifest)])
    capsys.readouterr()
    result = preview(['--goal-id','x13y07','--session-config',str(ROOT/'configs/deployment/command_session_sensor_ft.json'),
        '--model-manifest',str(manifest),'--ft-calibration',str(path),'--ft-library',str(library),'--hud'])
    assert result == 0
    report = json.loads(capsys.readouterr().out)
    assert not report['hardware_connected'] and not report['commands_sent']
    assert report['policy_inputs'][-1] == 'sensor_wrench_history'
    assert any('[20,7]' in item for item in report['sequence'])
    assert report['session']['allow_hardware'] is False


def test_readonly_adapter_only_calls_feedback_api(tmp_path, monkeypatch):
    import ctypes
    from types import SimpleNamespace
    from force_aware_act.control.sensor_readonly import SensorReadOnlyTransport
    from force_aware_act.control.sdk_transport import sha256
    shim=tmp_path/'readonly.so';sdk=tmp_path/'sdk.so'
    shim.write_bytes(b'fixture');sdk.write_bytes(b'fixture sdk')
    options=replace(SessionOptions(), expected_sdk_version=100343014, expected_sdk_sha256=sha256(sdk))
    shim.with_suffix('.json').write_text(json.dumps(dict(read_only=True,abi=3,
        shim_sha256=sha256(shim),sdk_sha256=sha256(sdk),sdk_path=str(sdk))))
    calls=[]
    def fn(name,result):
        def call(*args):calls.append(name);return result
        return call
    serial=0
    def read(values,length):
        nonlocal serial
        serial+=1;calls.append('read');assert length==79
        values[0]=serial;values[72]=216
        return 1
    lib=SimpleNamespace(fa_read_only=fn('readonly',1),fa_abi_version=fn('abi',3),
        fa_sdk_version=fn('version',100343014),fa_open=fn('open',1),
        fa_select_ft_telemetry=fn('select216',1),fa_read=read,fa_close=fn('release',1))
    monkeypatch.setattr(ctypes,'CDLL',lambda path:lib)
    transport=SensorReadOnlyTransport(shim,sdk,options)
    transport.open('192.168.1.190')
    row=transport.read();transport.close()
    assert row['ft_channel']==216 and len(row['ft_raw_counts'])==6
    assert calls==['readonly','abi','version','open','select216','read','read','read','release']
    assert not hasattr(transport,'target') and not hasattr(transport,'servo_off')


def test_readonly_observer_preview_and_nominal_compensation_without_tare(library, calibration, tmp_path, monkeypatch, capsys):
    from scripts.deployment.observe_sensor_ft_readonly import main
    from force_aware_act.control import sensor_readonly
    import time
    path,data=calibration
    args=['--transport',str(tmp_path/'not_loaded.so'),'--sdk',str(tmp_path/'not_loaded_sdk.so'),
          '--ft-library',str(library),'--ft-calibration',str(path),'--seconds','.03']
    assert main(args)==0
    assert not json.loads(capsys.readouterr().out)['hardware_connected']
    calls=[]
    class FeedbackFixture:
        def __init__(self,*args):pass
        def open(self,address):calls.append('open')
        def read(self):
            row=make_row(data,np.zeros(7),time.monotonic_ns())
            row.update(error_code=0,impedance_type=1,tool_kine=[0.]*6,tool_dyn=[0.]*10)
            return row
        def close(self):calls.append('release')
    monkeypatch.setattr(sensor_readonly,'SensorReadOnlyTransport',FeedbackFixture)
    output=tmp_path/'observation'
    assert main(args+['--connect','--output',str(output)])==0
    report=json.loads((output/'report.json').read_text())
    assert report['samples']>1 and report['sensor_ft']['tare'] is None
    assert report['motor_commands_sent'] is False and report['mode_or_parameters_changed'] is False
    assert report['sdk_released'] and calls==['open','release']
    np.testing.assert_allclose(report['compensated_mean_N_Nm'],0,atol=1e-12)
