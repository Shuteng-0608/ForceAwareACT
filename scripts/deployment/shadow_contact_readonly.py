"""Receive hardware observations, software-zero force, log predictions. No control API."""
import argparse
from collections import deque
import hashlib
import json
from pathlib import Path
import subprocess
import threading
import time

import numpy as np


def estimate_bias(rows, *, min_span_s=1.6, max_std_nm=.15):
    """Operator must confirm no contact separately; this verifies stationarity only."""
    if len(rows) < 20:
        raise ValueError('not enough distinct feedback samples')
    frames=[r['frame_serial'] for r in rows]
    if len(set(frames))!=len(frames):
        raise ValueError('calibration requires distinct controller frames')
    times = np.asarray([r['received_monotonic_ns'] for r in rows],dtype=np.int64)
    if (times[-1]-times[0])/1e9 < min_span_s or np.any(np.diff(times)<=0) or np.max(np.diff(times))>200_000_000:
        raise ValueError('calibration window is short, stale or unordered')
    q=np.asarray([r['qpos_rad'] for r in rows]);dq=np.asarray([r['qvel_rad_s'] for r in rows]);tau=np.asarray([r['external_raw_nm'] for r in rows])
    if not all(np.isfinite(v).all() for v in [q,dq,tau]):
        raise ValueError('nonfinite calibration data')
    metrics={'samples':len(rows),'span_s':float((times[-1]-times[0])/1e9),
             'max_velocity_rad_s':float(np.abs(dq).max()),
             'position_range_rad':np.ptp(q,axis=0).tolist(),'torque_std_nm':tau.std(axis=0).tolist(),
             'max_std_nm':max_std_nm,'max_position_range_rad':.005,'velocity_limit_rad_s':.01}
    if metrics['max_velocity_rad_s']>.01 or np.ptp(q,axis=0).max()>.005 or tau.std(axis=0).max()>max_std_nm:
        raise ValueError('calibration is unstable: '+json.dumps(metrics))
    return tau.mean(axis=0), metrics


class ReceiveClock:
    """Freeze epoch offset once; keep availability ordered despite wall-clock adjustment."""
    def __init__(self):
        self.offset_ns=time.time_ns()-time.monotonic_ns()
    def map(self, monotonic_ns):
        return int(monotonic_ns)+self.offset_ns


def summarize_rate_stage(target_hz, elapsed_s, cycles, successes, skipped, work_ms, model_ms, before, after):
    """Rate limit applies to the full receive/prepare/infer/log loop, not GPU only."""
    period_ms=1000/target_hz
    misses=sum(value>period_ms for value in work_ms)
    delta={key:after[key]-before[key] for key in before}
    rate=successes/elapsed_s
    skip_fraction=skipped/max(cycles,1)
    miss_fraction=misses/max(cycles,1)
    stale_fraction=delta['robot_stale_ipc']/max(delta['robot_distinct']+delta['robot_stale_ipc'],1)
    p99=float(np.percentile(work_ms,99)) if work_ms else None
    checks={'completed_rate_at_least_98pct':rate>=target_hz*.98,
            'deadline_miss_at_most_1pct':miss_fraction<=.01,
            'skips_at_most_0_5pct':skip_fraction<=.005,
            'full_loop_p99_within_period':p99 is not None and p99<=period_ms,
            'robot_fresh_rate_at_least_270hz':delta['robot_distinct']/elapsed_s>=270,
            'cameras_at_least_27hz':min(delta['d455'],delta['d405'])/elapsed_s>=27,
            'stale_ipc_at_most_1pct':stale_fraction<=.01}
    return {'target_hz':target_hz,'elapsed_s':elapsed_s,'cycles':cycles,'predictions':successes,
            'actual_completed_hz':rate,'skipped':skipped,'skip_fraction':skip_fraction,
            'deadline_misses':misses,'deadline_miss_fraction':miss_fraction,
            'full_loop_ms_p50':float(np.percentile(work_ms,50)) if work_ms else None,
            'full_loop_ms_p99':p99,'model_ms_mean':float(np.mean(model_ms)) if model_ms else None,
            'model_ms_p99':float(np.percentile(model_ms,99)) if model_ms else None,
            'stream_counts':delta,'stale_ipc_fraction':stale_fraction,'checks':checks,
            'stable':all(checks.values())}


def parse_arguments(argv=None):
    """Resolve the immutable per-run goal before loading any hardware SDK."""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--reader',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--robot-ip',default='192.168.1.190')
    parser.add_argument('--seconds',type=float,default=120)
    parser.add_argument('--query-hz',type=float,default=5)
    parser.add_argument('--rate-sweep',default='',help='comma-separated ascending Hz; each stage uses --seconds')
    parser.add_argument('--confirm-seconds',type=float,default=60)
    parser.add_argument('--operator-confirmed-no-contact',action='store_true')
    selection=parser.add_mutually_exclusive_group()
    selection.add_argument('--controller-config',type=Path,
                        help='optional independent shadow executor/safety/goal monitor; never commands hardware')
    selection.add_argument('--goal-id',help='explicit hole label, e.g. x14y08; enables shadow goal monitoring')
    parser.add_argument('--controller-profile',type=Path,
                        help='profile directory for --goal-id; default: repository contact_right_step_v1')
    parser.add_argument('--describe-goal',action='store_true',
                        help='print the selected goal/config snapshot and exit without model or devices')
    args=parser.parse_args(argv)
    if (args.controller_profile is not None or args.describe_goal) and args.goal_id is None:
        parser.error('--controller-profile and --describe-goal require --goal-id')
    controller_config=None;goal_selection=None
    try:
        if args.goal_id is not None:
            from force_aware_act.control.goal_selection import DEFAULT_PROFILE,select_goal
            goal_selection=select_goal(args.goal_id,args.controller_profile or DEFAULT_PROFILE)
            controller_config=goal_selection.config
        elif args.controller_config is not None:
            from force_aware_act.control.contracts import ShadowConfig
            controller_config=ShadowConfig.read(args.controller_config)
    except (OSError,ValueError,TypeError,KeyError) as exc:
        parser.error(str(exc))
    if args.describe_goal:
        return args,controller_config,goal_selection
    missing=[f'--{name}' for name in ('checkpoint','reader','output') if getattr(args,name) is None]
    if missing:
        parser.error('required for a run: '+', '.join(missing))
    if not args.operator_confirmed_no_contact:
        parser.error('manual stationary/no-contact confirmation required for software calibration')
    if not 0<args.seconds<=180 or not 0<args.query_hz<=120:
        parser.error('seconds must be in (0,180], query-hz in (0,120]')
    rates=[float(x) for x in args.rate_sweep.split(',')] if args.rate_sweep else []
    if controller_config is not None:
        if rates or args.query_hz!=controller_config.query_hz:
            parser.error('shadow controller requires a fixed 50 Hz query rate')
    if args.output.exists():
        parser.error('output directory already exists; use a fresh run directory')
    if rates and (rates!=sorted(set(rates)) or not all(0<x<=120 for x in rates) or not 0<args.confirm_seconds<=180):
        parser.error('rate sweep must be increasing unique Hz in (0,120]; confirm seconds in (0,180]')
    return args,controller_config,goal_selection


def main(argv=None):
    args,controller_config,goal_selection=parse_arguments(argv)
    selection_snapshot=goal_selection.to_dict() if goal_selection is not None else None
    if args.describe_goal:
        print(json.dumps(selection_snapshot,indent=2))
        return 0
    if selection_snapshot is not None:
        print('GOAL_SELECTION '+json.dumps(selection_snapshot),flush=True)
    rates=[float(x) for x in args.rate_sweep.split(',')] if args.rate_sweep else []
    total_seconds=(len(rates)+1)*args.seconds+args.confirm_seconds if rates else args.seconds
    profile=json.loads((args.reader.resolve().parent/'configs/readonly_profile.json').read_text())
    adopted=profile['adopted_readonly_parameters']
    if adopted['robot_ip']!=args.robot_ip or adopted['enabled_arms']!='right':
        raise ValueError('readonly profile arm/IP mismatch')
    import ctypes
    lib=ctypes.CDLL(str(args.reader.resolve().parent/'sdk_validated/libMarvinSDK.so'))
    lib.OnGetSDKVersion.restype=ctypes.c_long
    sdk_version=lib.OnGetSDKVersion()
    if sdk_version!=adopted['expected_sdk_version']:
        raise ValueError('SDK version differs from bridge_both_test profile')
    import torch
    import pyrealsense2 as rs
    from force_aware_act.real_robot.checkpoint import read_checkpoint
    from force_aware_act.inference.real_robot import RealRobotPolicyAdapter,RealRobotObservationBuffer,ObservationUnavailableError
    torch.set_num_threads(4)
    adapter=RealRobotPolicyAdapter.from_checkpoint(read_checkpoint(args.checkpoint),device='cuda')
    if adapter.model_kind!='contact':raise ValueError('Contact checkpoint required')
    # Cold CUDA initialization must happen before starting device reception.
    warm_buffer=RealRobotObservationBuffer(adapter.data_config)
    warm_anchor=1_000_000_000
    for t in np.arange(warm_anchor-210_000_000,warm_anchor,3_333_333):
        warm_buffer.append_state(int(t),np.asarray(adapter.normalization['qpos_mean']))
        warm_buffer.append_external_torque(int(t),np.asarray(adapter.normalization['force_mean']))
    for name in adapter.data_config.camera_names:
        warm_buffer.append_image(name,warm_anchor-1_000_000,np.zeros((480,640,3),dtype=np.uint8))
    for _ in range(5):
        adapter.predict(warm_buffer,warm_anchor)
    torch.cuda.synchronize()
    args.output.mkdir(parents=True,exist_ok=True)
    if selection_snapshot is not None:
        (args.output/'goal_selection.json').write_text(json.dumps(selection_snapshot,indent=2)+'\n')
    buffer=RealRobotObservationBuffer(adapter.data_config,image_capacity=4)
    lock=threading.Lock();stop=threading.Event();errors=[];clock=ReceiveClock();bias={};calibration_rows=deque(maxlen=800)
    robot_latest={};counts={'robot_distinct':0,'robot_repeated':0,'robot_stale_ipc':0,'d455':0,'d405':0}
    rows_log=(args.output/'feedback_samples.jsonl').open('w');prediction_log=(args.output/'predictions.jsonl').open('w')
    pipelines=[];worker_threads=[];process=None;shadow_monitor=None
    report={'schema_version':1,'commands_sent':False,'operator_confirmed_no_contact':True,
            'raw_rgb_no_jpeg':True,'clock_mapping':'fixed_epoch_offset_from_host_monotonic_receive_v1',
            'clock_offset_ns':clock.offset_ns,'input_timestamp_key':adapter.data_config.timestamp_key,
            'query_hz':args.query_hz,'seconds_requested':args.seconds,'errors':errors,'counts':counts,
            'readonly_profile':profile,'sdk_version':sdk_version}
    if selection_snapshot is not None:
        report['goal_selection']=selection_snapshot
    def worker_guard(fn):
        try:fn()
        except Exception as exc:
            errors.append({'worker':fn.__name__,'type':type(exc).__name__,'message':str(exc)});stop.set()
            if shadow_monitor is not None:
                try:shadow_monitor.publish('fault',reason='upstream_'+fn.__name__+'_error')
                except RuntimeError:pass
            if fn is robot_worker and process is not None:
                # Keep draining so SDK stdout cannot block graceful OnRelease.
                for _ in process.stdout:
                    pass
    def robot_worker():
        last_frame=None;last_context=None;last_log=0
        for line in process.stdout:
            if not line.startswith('{'):continue
            record=json.loads(line)
            if record.get('kind')=='summary':
                report['reader_summary']=record
                continue
            if stop.is_set():continue
            if record.get('kind')!='feedback':continue
            arm=record['arms'][1];received=record['received_monotonic_ns'];available=time.monotonic_ns()
            if arm['frame_serial']<0:continue
            if arm['frame_serial']==last_frame:
                counts['robot_repeated']+=1;continue
            if last_frame is not None and arm['frame_serial']<last_frame and not(last_frame>900000 and arm['frame_serial']<100000):
                raise RuntimeError('controller frame serial reset; calibration invalid')
            last_frame=arm['frame_serial']
            context=(arm['state'],tuple(arm['tool_kine']),tuple(arm['tool_dyn']))
            if last_context is not None and context!=last_context:
                raise RuntimeError('controller state/tool changed; calibration invalid')
            last_context=context
            if arm['error_code']!=0:raise RuntimeError('controller error; no recovery attempted')
            if available-received>200_000_000:raise RuntimeError('feedback IPC delay exceeded 200 ms; calibration invalid')
            if available-received>20_000_000:
                counts['robot_stale_ipc']+=1
                continue
            sample={**arm,'received_monotonic_ns':received,'received_epoch_ns':clock.map(received)}
            with lock:
                robot_latest.clear();robot_latest.update(sample)
                calibration_rows.append(sample)
                buffer.append_state(clock.map(received),np.asarray(arm['qpos_rad']),available_ns=clock.map(available))
                if 'value' in bias:
                    corrected=np.asarray(arm['external_raw_nm'])-bias['value']
                    buffer.append_external_torque(clock.map(received),corrected,available_ns=clock.map(available))
                    if shadow_monitor is not None:
                        shadow_monitor.publish('feedback',value={
                            'timestamp_ns':received,'frame_serial':arm['frame_serial'],
                            'q_rad':arm['qpos_rad'],'dq_rad_s':arm['qvel_rad_s'],
                            'external_nm':corrected.tolist(),'force_timestamp_ns':received,
                            'bias_valid':True,'context':json.dumps(context,allow_nan=False),
                            'error_code':arm['error_code'],'raw_external_nm':arm['external_raw_nm'],
                            'internal_nm':arm['internal_torque_nm']})
                counts['robot_distinct']+=1
                if available-last_log>=1_000_000_000:
                    last_log=available
                    sample['bias_valid']='value' in bias
                    if 'value' in bias:
                        sample['bias_nm']=bias['value'].tolist();sample['external_corrected_nm']=corrected.tolist()
                    rows_log.write(json.dumps(sample)+'\n');rows_log.flush()
        if not stop.is_set():
            errors.append({'worker':'robot_worker','message':'reader exited unexpectedly'});stop.set()
    def camera_worker():
        last={name:None for name,_ in pipelines}
        while not stop.is_set():
            for name,pipeline in pipelines:
                frames=pipeline.poll_for_frames()
                if not frames:continue
                frame=frames.get_color_frame()
                if not frame or frame.get_frame_number()==last[name]:continue
                received=time.monotonic_ns();rgb=np.asanyarray(frame.get_data()).copy();available=time.monotonic_ns()
                with lock:buffer.append_image(name,clock.map(received),rgb,available_ns=clock.map(available))
                if shadow_monitor is not None:
                    shadow_monitor.publish('camera',name=name,received_ns=received)
                last[name]=frame.get_frame_number();counts[name]+=1
            time.sleep(.001)
    latencies=[];skipped=0;predictions=0
    try:
        if controller_config is not None:
            from force_aware_act.control.runtime import ShadowProcess
            shadow_monitor=ShadowProcess(controller_config,args.output/'controller')
            report['controller']={'run_id':shadow_monitor.run_id,'origin_monotonic_ns':shadow_monitor.origin_ns,
                                  'config':controller_config.to_dict(),'backend':'journal_only_no_motion_api'}
        serials={'d455':'261822304024','d405':'260322275961'}
        for name in adapter.data_config.camera_names:
            pipeline=rs.pipeline();config=rs.config();config.enable_device(serials[name])
            config.enable_stream(rs.stream.color,640,480,rs.format.rgb8,30);config.enable_stream(rs.stream.depth,640,480,rs.format.z16,30)
            profile=pipeline.start(config);pipelines.append((name,pipeline))
            for sensor in profile.get_device().query_sensors():
                if sensor.supports(rs.option.enable_auto_exposure):sensor.set_option(rs.option.enable_auto_exposure,1)
            time.sleep(3)
        process=subprocess.Popen([str(args.reader.resolve()),args.robot_ip,str(total_seconds+30),'--stream'],stdout=subprocess.PIPE,stderr=(args.output/'robot_reader.stderr').open('w'),text=True,bufsize=1)
        for fn in [robot_worker,camera_worker]:
            thread=threading.Thread(target=worker_guard,args=(fn,),daemon=True);worker_threads.append(thread);thread.start()
        deadline=time.monotonic()+10
        while time.monotonic()<deadline and not stop.is_set():
            time.sleep(.1)
            with lock:
                candidates=list(calibration_rows)
            if candidates and candidates[-1]['received_monotonic_ns']-candidates[0]['received_monotonic_ns']>=2_000_000_000:
                value,metrics=estimate_bias(candidates,min_span_s=.8*adopted['joint_zero_window_sec'],max_std_nm=adopted['joint_zero_max_std_nm'])
                with lock:bias['value']=value
                report['calibration']={'bias_nm':value.tolist(),'metrics':metrics,'captured_unix_ns':time.time_ns(),
                                       'arm':1,'source':'m_EST_Joint_Force','method':'fixed_mean_raw_stationary_window_v1'}
                (args.output/'calibration.json').write_text(json.dumps(report['calibration'],indent=2)+'\n')
                break
        if 'value' not in bias:raise RuntimeError('could not establish software calibration')
        time.sleep(.3)
        stages=[(10,3,'warmup')]+[(hz,args.seconds,'sweep') for hz in rates] if rates else [(args.query_hz,args.seconds,'fixed')]
        report['rate_stages']=[]
        last_passed=None
        stage_index=0
        while stage_index<len(stages) and not stop.is_set():
            hz,duration,kind=stages[stage_index]
            start=time.monotonic();next_query=start;before=dict(counts)
            work=[];models=[];stage_predictions=0;stage_skips=0;cycles=0
            while not stop.is_set() and time.monotonic()-start<duration:
                cycle_begin=time.perf_counter();cycles+=1
                try:
                    query_mono_ns=time.monotonic_ns()
                    if shadow_monitor is not None:
                        query_mono_ns=shadow_monitor.origin_ns+(query_mono_ns-shadow_monitor.origin_ns)//20_000_000*20_000_000
                    with lock:
                        snapshot=buffer.observation_at(clock.map(query_mono_ns))
                    inputs=adapter.prepare_snapshot(snapshot)
                    torch.cuda.synchronize();begin=time.perf_counter()
                    output=adapter.predict_prepared(inputs,contact_latent_mode='zero')
                    torch.cuda.synchronize();elapsed=(time.perf_counter()-begin)*1000
                    latencies.append(elapsed);models.append(elapsed);predictions+=1;stage_predictions+=1
                    if shadow_monitor is not None:
                        shadow_monitor.publish('prediction',actions=output['pred_action'].tolist(),metadata={
                            'query_id':predictions,'start_ns':query_mono_ns,'observed_ns':query_mono_ns,
                            'available_ns':time.monotonic_ns(),'run_id':shadow_monitor.run_id})
                    source=inputs.source
                    row={'anchor_ns':source.anchor_ns,'completed_ns':clock.map(time.monotonic_ns()),
                         'target_hz':hz,'stage_kind':kind,'model_ms':elapsed,'qpos_rad':source.qpos.tolist(),
                         'state_age_ms':(source.anchor_ns-source.state_timestamp_ns)/1e6,
                         'image_age_ms':[(source.anchor_ns-t)/1e6 for t in source.image_timestamps_ns],
                         'valid_force_samples':int((~inputs.online_force_sample_padding_mask).sum()),
                         'commands_sent':False}
                    # Benchmark logs compact output to avoid storing hundreds of MB of text.
                    if rates:
                        row['first_pred_action_rad']=output['pred_action'][0].tolist()
                        row['pred_action_shape']=list(output['pred_action'].shape)
                    else:
                        row['pred_action_rad']=output['pred_action'].tolist()
                    prediction_log.write(json.dumps(row)+'\n');prediction_log.flush()
                except ObservationUnavailableError as exc:
                    skipped+=1;stage_skips+=1
                    prediction_log.write(json.dumps({'skip':str(exc),'target_hz':hz,'stage_kind':kind,'commands_sent':False})+'\n');prediction_log.flush()
                work.append((time.perf_counter()-cycle_begin)*1000)
                if shadow_monitor is not None:
                    # Fixed 20 ms query phase on the same 10 ms execution grid.
                    # If the calculation overruns, skip slots without queuing catch-up work.
                    current=time.monotonic_ns()
                    next_slot=max(query_mono_ns+20_000_000,
                                  shadow_monitor.origin_ns+((current-shadow_monitor.origin_ns)//20_000_000+1)*20_000_000)
                    time.sleep(max(0,(next_slot-time.monotonic_ns())/1e9))
                else:
                    next_query=max(next_query+1/hz,time.monotonic());time.sleep(max(0,next_query-time.monotonic()))
            measured=summarize_rate_stage(hz,time.monotonic()-start,cycles,stage_predictions,stage_skips,work,models,before,dict(counts))
            measured['kind']=kind
            report['rate_stages'].append(measured)
            print('RATE_STAGE '+json.dumps(measured),flush=True)
            if rates and kind in ('sweep','refine'):
                if measured['stable']:
                    last_passed=hz
                else:
                    stages=stages[:stage_index+1]
                    if kind=='sweep' and last_passed is not None:
                        stages.append(((last_passed+hz)/2,args.seconds,'refine'))
                if stage_index==len(stages)-1 and last_passed is not None:
                    stages.append((last_passed,args.confirm_seconds,'confirm'))
            stage_index+=1
        confirmed=[r['target_hz'] for r in report['rate_stages'] if r['kind']=='confirm' and r['stable']]
        report['highest_confirmed_stable_hz']=max(confirmed) if confirmed and not errors else None
    except Exception as exc:
        errors.append({'stage':'main','type':type(exc).__name__,'message':str(exc)})
        if shadow_monitor is not None:
            try:shadow_monitor.publish('fault',reason='upstream_main_error')
            except RuntimeError:pass
    finally:
        stop.set()
        if shadow_monitor is not None:
            try:shadow_monitor.close()
            except Exception as exc:errors.append({'stage':'shadow_cleanup','message':str(exc)})
        if process is not None:
            process.terminate()
            try:process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                errors.append({'stage':'cleanup','message':'SDK release timed out'});process.kill();process.wait()
            report['reader_returncode']=process.returncode
        for thread in worker_threads:thread.join(timeout=5)
        for _,pipeline in reversed(pipelines):pipeline.stop()
        if shadow_monitor is not None:
            controller_report=args.output/'controller/report.json'
            if controller_report.exists():
                report['controller_report']=json.loads(controller_report.read_text())
        rows_log.close();prediction_log.close()
        warm=latencies[3:]
        report.update(predictions=predictions,skipped_observations=skipped,
                      latency_ms={'all':latencies,'warm_mean':float(np.mean(warm)) if warm else None,
                                  'warm_p95':float(np.percentile(warm,95)) if warm else None},
                      calibration_valid_for_completed_run=bool(bias) and not errors,
                      calibration_valid_at_end=False,calibration_invalidated_reason='reader_disconnect')
        (args.output/'shadow_report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    return int(bool(errors) or predictions==0)


if __name__=='__main__':raise SystemExit(main())
