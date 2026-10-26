"""Sensor FT + RGB + policy + HUD acceptance. Predictions are logged only.

The sole SDK owner is observe_sensor_ft_readonly.py with the motor-free shim.
Default preview does not import drivers, load weights, or connect hardware.
"""
import argparse
import json
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SERIALS = {'d455': '261822304024', 'd405': '260322275961'}


def parse_arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('checkpoint', 'transport', 'sdk', 'ft-library', 'ft-calibration', 'output'):
        p.add_argument('--'+key, type=Path)
    p.add_argument('--session-config', type=Path, default=ROOT/'configs/deployment/command_session_sensor_ft.json')
    p.add_argument('--robot-ip', default='192.168.1.190')
    p.add_argument('--seconds', type=float, default=120.)
    p.add_argument('--hud-port', type=int, default=8787)
    p.add_argument('--connect', action='store_true')
    p.add_argument('--operator-confirmed-no-contact', action='store_true')
    a = p.parse_args(argv)
    if not 0 < a.seconds <= 120 or not 1 <= a.hud_port <= 65535:
        p.error('seconds must be in (0,120], HUD port in 1..65535')
    if a.connect:
        if not a.operator_confirmed_no_contact:
            p.error('live tare requires stationary/no-contact confirmation')
        if any(getattr(a, key) is None for key in ('checkpoint','transport','sdk','ft_library','ft_calibration','output')):
            p.error('connected acceptance requires all input paths and a new output directory')
        if a.output.exists():
            p.error('output already exists')
    return a


def append_feedback(buffer, row, available_ns, clock):
    """Validate same-frame compensated/tared SI input before causal buffering."""
    sensor = row['sensor_ft']
    ns = row['received_monotonic_ns']
    if not sensor['valid']:
        return False
    if sensor['source_ns'] != ns or sensor['frame_serial'] != row['frame_serial']:
        raise ValueError('sensor/actual-joint frame mismatch')
    if not 0 <= available_ns-ns <= 20_000_000:
        return False
    buffer.append_sensor_wrench(clock.map(ns), np.asarray(sensor['wrench_si']),
        frame_id=sensor['frame_id'], available_ns=clock.map(available_ns))
    buffer.append_state(clock.map(ns), np.asarray(row['qpos_rad']), available_ns=clock.map(available_ns))
    return True


def stats(values):
    return dict(count=len(values), p50=float(np.percentile(values,50)),
                p99=float(np.percentile(values,99)), max=float(max(values))) if len(values) else dict(count=0)


def main(argv=None):
    args = parse_arguments(argv)
    if not args.connect:
        print(json.dumps(dict(mode='disabled_preview', hardware_connected=False,
                              motor_commands_sent=False, predictions_only=True)))
        return 0
    import torch
    import pyrealsense2 as rs
    from force_aware_act.real_robot.checkpoint import read_checkpoint
    from force_aware_act.inference.real_robot import RealRobotPolicyAdapter, RealRobotObservationBuffer, ObservationUnavailableError
    from force_aware_act.inference.realtime_gc import StaticGraphGCGuard
    from force_aware_act.control.deployment_model import adapter_contract, checkpoint_digest
    from force_aware_act.control.hud_telemetry import HudTelemetry
    from force_aware_act.visualization.deployment_hud import HudProcess, read_config
    from scripts.deployment.shadow_contact_readonly import ReceiveClock

    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(mode='sensor_policy_hud_readonly', motor_commands_sent=False,
                  predictions_only=True, query_hz=50, seconds_requested=args.seconds, errors=[])
    errors = report['errors']
    stop = threading.Event()
    handlers = {s:signal.signal(s, lambda *_: stop.set()) for s in (signal.SIGINT,signal.SIGTERM)}
    process = hud = guard = None
    pipelines = []; threads = []; files = []
    counts = dict(feedback=0, accepted=0, stale_ipc=0, d455=0, d405=0, predictions=0, skipped=0)
    camera_times = {name:[] for name in SERIALS}; ipc_ms=[]; model_ms=[]; work_ms=[]; prediction_times=[]
    first_valid = []; first_prediction = None; elapsed = 0.
    try:
        torch.set_num_threads(4)
        checkpoint = read_checkpoint(args.checkpoint)
        adapter = RealRobotPolicyAdapter.from_checkpoint(checkpoint, device='cuda')
        report['model'] = dict(adapter_contract(adapter), checkpoint=str(args.checkpoint.resolve()),
                               checkpoint_sha256=checkpoint_digest(args.checkpoint), global_step=checkpoint.get('global_step'))
        del checkpoint
        warm = RealRobotObservationBuffer(adapter.data_config)
        anchor = 1_000_000_000
        for t in range(anchor-300_000_000, anchor, 3_333_333):
            warm.append_state(t, np.asarray(adapter.normalization['qpos_mean']))
            warm.append_sensor_wrench(t, np.zeros(6), frame_id=adapter.data_config.wrench_frame)
        for name in SERIALS:
            warm.append_image(name, anchor-1_000_000, np.zeros((480,640,3),dtype=np.uint8))
        for _ in range(5):
            adapter.predict(warm, anchor)
        torch.cuda.synchronize()
        del warm
        buffer = RealRobotObservationBuffer(adapter.data_config, image_capacity=4)
        lock = threading.Lock(); clock = ReceiveClock(); channel = HudTelemetry()
        goal = dict(goal_id='not selected', target_position_m=None,
                    config={'limits':{'torque_soft_nm':[17.5,6.5,5.,11.,1.,6.5,2.]}})
        hud = HudProcess(channel, read_config(ROOT/'configs/deployment/force_hud_sensor.json'),
                         goal, args.output/'hud', args.hud_port)
        report['hud_url'] = hud.url
        report['clock'] = dict(kind='fixed_epoch_offset_from_monotonic_receive',offset_ns=clock.offset_ns)
        guard = StaticGraphGCGuard().start()
        for name, serial in SERIALS.items():
            pipeline=rs.pipeline(); config=rs.config(); config.enable_device(serial)
            config.enable_stream(rs.stream.color,640,480,rs.format.rgb8,30)
            config.enable_stream(rs.stream.depth,640,480,rs.format.z16,30)
            profile=pipeline.start(config); pipelines.append((name,pipeline))
            for sensor in profile.get_device().query_sensors():
                if sensor.supports(rs.option.enable_auto_exposure):
                    sensor.set_option(rs.option.enable_auto_exposure,1)
            time.sleep(3)
        stderr = (args.output/'observer.stderr').open('x'); files.append(stderr)
        command = [sys.executable,'-u',str(ROOT/'scripts/deployment/observe_sensor_ft_readonly.py'),
                   '--connect','--tare','--operator-confirmed-no-contact','--stream',
                   '--seconds',str(args.seconds+40),'--robot-ip',args.robot_ip,
                   '--output',str(args.output/'feedback')]
        for key in ('transport','sdk','ft_library','ft_calibration','session_config'):
            command += ['--'+key.replace('_','-'),str(getattr(args,key).resolve())]
        process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=stderr,text=True,bufsize=1)

        def robot_worker():
            last_hud=0
            try:
                for line in process.stdout:
                    if stop.is_set() or not line.startswith('{"kind":"feedback",'):
                        continue
                    row=json.loads(line)['record']; now=time.monotonic_ns(); sensor=row['sensor_ft']
                    counts['feedback']+=1; ipc_ms.append((now-row['received_monotonic_ns'])/1e6)
                    if now-last_hud >= 10_000_000:
                        channel.publish(dict(record=row,bias_nm=None,bias_valid=sensor['valid'],
                            state='READONLY_PREDICTING' if sensor['valid'] else 'READONLY_TARING',
                            calibration_status='valid' if sensor['valid'] else 'collecting',
                            stop_reason=None,source_mode='readonly_feedback'))
                        last_hud=now
                    with lock:
                        accepted=append_feedback(buffer,row,now,clock)
                    if accepted:
                        counts['accepted']+=1
                        if not first_valid:first_valid.append(now)
                    elif sensor['valid']:counts['stale_ipc']+=1
                if not stop.is_set():raise RuntimeError('read-only observer exited unexpectedly')
            except Exception as exc:
                errors.append('feedback: '+str(exc)); stop.set()
                # Drain until EOF so observer can write its final report and release SDK.
                for _ in process.stdout:pass

        def camera_worker():
            last={name:None for name in SERIALS}
            try:
                while not stop.is_set():
                    for name,pipeline in pipelines:
                        frames=pipeline.poll_for_frames()
                        if not frames:continue
                        frame=frames.get_color_frame()
                        if not frame or frame.get_frame_number()==last[name]:continue
                        ns=time.monotonic_ns(); rgb=np.asanyarray(frame.get_data()).copy()
                        available=time.monotonic_ns()
                        with lock:buffer.append_image(name,clock.map(ns),rgb,available_ns=clock.map(available))
                        last[name]=frame.get_frame_number();counts[name]+=1;camera_times[name].append(ns)
                    time.sleep(.001)
            except Exception as exc:
                errors.append('camera: '+str(exc));stop.set()

        for fn in (robot_worker,camera_worker):
            thread=threading.Thread(target=fn,daemon=True);threads.append(thread);thread.start()
        ready_deadline=time.monotonic()+30
        while not stop.is_set() and (not first_valid or time.monotonic_ns()-first_valid[0]<300_000_000):
            if time.monotonic()>ready_deadline:raise RuntimeError('tare/history not ready within 30 s; inspect feedback report')
            stop.wait(.05)
        if stop.is_set():raise RuntimeError('read-only startup interrupted')
        print('READONLY_READY '+json.dumps({'hud':hud.url,'model':adapter.model_kind,'force_dim':adapter.data_config.force_dim}),flush=True)
        started=time.monotonic(); next_query=started
        with (args.output/'predictions.jsonl').open('x') as log:
            while not stop.is_set() and time.monotonic()-started<args.seconds:
                begin=time.perf_counter(); ns=time.monotonic_ns()
                try:
                    with lock:snapshot=buffer.observation_at(clock.map(ns))
                    inputs=adapter.prepare_snapshot(snapshot)
                    torch.cuda.synchronize(); model_begin=time.perf_counter()
                    output=adapter.predict_prepared(inputs,contact_latent_mode='zero')
                    torch.cuda.synchronize(); latency=(time.perf_counter()-model_begin)*1000
                    model_ms.append(latency); prediction_times.append(ns); counts['predictions']+=1
                    if first_prediction is None:first_prediction=ns
                    source=inputs.source
                    row=dict(anchor_ns=ns,model_ms=latency,motor_commands_sent=False,
                        qpos_rad=source.qpos.tolist(),state_age_ms=(source.anchor_ns-source.state_timestamp_ns)/1e6,
                        image_age_ms=[(source.anchor_ns-t)/1e6 for t in source.image_timestamps_ns],
                        valid_force_samples=int((~inputs.online_force_sample_padding_mask).sum()),
                        prediction_shapes={k:list(v.shape) for k,v in output.items() if hasattr(v,'shape')},
                        pred_action_rad=output['pred_action'].tolist())
                except ObservationUnavailableError as exc:
                    counts['skipped']+=1;row=dict(anchor_ns=ns,skip=str(exc))
                log.write(json.dumps(row)+'\n')
                work_ms.append((time.perf_counter()-begin)*1000)
                next_query=max(next_query+.02,time.monotonic())
                stop.wait(max(0,next_query-time.monotonic()))
        elapsed=time.monotonic()-started
    except Exception as exc:
        errors.append(type(exc).__name__+': '+str(exc))
    finally:
        stop.set()
        if process is not None:
            if process.poll() is None:process.terminate()
            try:process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                errors.append('observer cleanup timeout; SDK release unconfirmed')
            report['observer_exit_code']=process.poll()
        for thread in threads:thread.join(2)
        for name,pipeline in pipelines:
            try:pipeline.stop()
            except Exception as exc:errors.append(name+' cleanup: '+str(exc))
        if hud is not None:
            report['hud']=hud.close()
            errors.extend('HUD: '+e for e in report['hud'].get('errors',[]))
        if guard is not None:report['gc']=guard.snapshot();guard.close()
        for f in files:f.close()
        for s,handler in handlers.items():signal.signal(s,handler)
        path=args.output/'feedback/report.json'
        if path.exists():
            report['observer']=json.loads(path.read_text())
            errors.extend('observer: '+e for e in report['observer']['errors'])
            if not report['observer']['sdk_released']:errors.append('SDK release unconfirmed')
        report.update(counts=counts,inference_elapsed_s=elapsed,
            completed_hz=counts['predictions']/elapsed if elapsed else 0,
            model_ms=stats(model_ms),full_loop_ms=stats(work_ms),ipc_ms=stats(ipc_ms),
            prediction_interval_ms=stats(np.diff(prediction_times)/1e6),
            camera_interval_ms={name:stats(np.diff(ts)/1e6) for name,ts in camera_times.items()})
        report['passed']=bool(not errors and counts['predictions'] and counts['skipped']==0
                              and elapsed>=args.seconds and report.get('observer',{}).get('sdk_released'))
        (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({'passed':report['passed'],'counts':counts,'errors':errors,'report':str(args.output/'report.json')}),flush=True)
    return int(not report['passed'])


if __name__=='__main__':raise SystemExit(main())
