"""Observe same-frame sensor compensation with a motor-free native transport.

Default is a disabled preview. --connect only permits feedback connection and
channel 216 selection, never motor targets, parameter/mode changes or servo off.
Software tare is optional and requires a separate current no-contact confirmation.
"""
import argparse
import json
from pathlib import Path
import signal
import time

import numpy as np
from force_aware_act.control.command_session import SessionOptions
from force_aware_act.control.sensor_ft import SensorFT

ROOT=Path(__file__).resolve().parents[2]


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--transport',type=Path,required=True)
    p.add_argument('--sdk',type=Path,required=True)
    p.add_argument('--ft-library',type=Path,required=True)
    p.add_argument('--ft-calibration',type=Path,required=True)
    p.add_argument('--session-config',type=Path,default=ROOT/'configs/deployment/command_session_sensor_ft.json')
    p.add_argument('--robot-ip',default='192.168.1.190')
    p.add_argument('--seconds',type=float,default=10.)
    p.add_argument('--output',type=Path)
    p.add_argument('--connect',action='store_true')
    p.add_argument('--tare',action='store_true')
    p.add_argument('--operator-confirmed-no-contact',action='store_true')
    args=p.parse_args(argv)
    if not np.isfinite(args.seconds) or not 0<args.seconds<=180:p.error('seconds must be in (0,180]')
    if args.tare and not args.operator_confirmed_no_contact:p.error('tare needs current stationary/no-contact confirmation')
    if args.connect and (args.output is None or args.output.exists()):p.error('connected observation needs a new output directory')
    options=SessionOptions.read(args.session_config)
    sensor=SensorFT(args.ft_calibration,args.ft_library,window_s=options.ft_tare_window_s)
    if not args.connect:
        try:
            print(json.dumps(dict(mode='disabled_preview',hardware_connected=False,motor_commands_sent=False,
                tare_requested=args.tare,calibration=sensor.report()),indent=2))
        finally:sensor.close()
        return 0
    from force_aware_act.control.sensor_readonly import SensorReadOnlyTransport
    args.output.mkdir(parents=True,exist_ok=False)
    transport=None; errors=[]; times=[]; processing=[]; values=[]; last=None; context=None; stopped=False
    connected=False; released=False
    def stop(*_):
        nonlocal stopped
        stopped=True
    handlers={s:signal.signal(s,stop) for s in (signal.SIGINT,signal.SIGTERM)}
    started=time.monotonic()
    try:
        transport=SensorReadOnlyTransport(args.transport,args.sdk,options)
        transport.open(args.robot_ip);connected=True;started=time.monotonic()
        with (args.output/'feedback.jsonl').open('x') as stream:
            while not stopped and time.monotonic()-started<args.seconds:
                row=transport.read();ns=row['received_monotonic_ns']
                if row['frame_serial']!=last:
                    if times and ns<=times[-1]:raise ValueError('feedback clock not increasing')
                    if last is not None and row['frame_serial']<last and not (last>900000 and row['frame_serial']<100000):
                        raise ValueError('feedback sequence reset')
                    if row['error_code']:raise ValueError('controller reports an error; no recovery attempted')
                    current=(row['state'],row['impedance_type'],tuple(row['tool_kine']),tuple(row['tool_dyn']))
                    if context is not None and current!=context:raise ValueError('mode/tool context changed')
                    context=current;last=row['frame_serial']
                    result=sensor.process(row,collect=args.tare and sensor.tare is None)
                    if args.tare and sensor.tare is None and sensor.capture(ns,args.output/'tare'):
                        result=sensor.process(row)
                    times.append(ns);processing.append(result['processing_ms']);values.append(result['compensated_si'])
                    stream.write(json.dumps(row)+'\n')
                time.sleep(.002)
    except Exception as exc:errors.append(str(exc))
    finally:
        if transport is not None:
            try:transport.close();released=True
            except Exception as exc:errors.append(str(exc))
        for s,handler in handlers.items():signal.signal(s,handler)
        gaps=np.diff(times)/1e6
        report=dict(mode='sensor_feedback_only',hardware_connected=connected,sdk_released=released,
            motor_commands_sent=False,mode_or_parameters_changed=False,servo_commands_sent=False,
            telemetry_selection='OnSetUserSpcfData_B(216)',elapsed_s=time.monotonic()-started,
            samples=len(times),observed_hz=(len(times)-1)/((times[-1]-times[0])/1e9) if len(times)>1 else None,
            interval_ms_p99=float(np.percentile(gaps,99)) if len(gaps) else None,
            interval_ms_max=float(max(gaps)) if len(gaps) else None,
            compensation_ms_p99=float(np.percentile(processing,99)) if processing else None,
            compensated_mean_N_Nm=np.mean(values,axis=0).tolist() if values else None,
            sensor_ft=sensor.report(),errors=errors)
        sensor.close()
        (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report,indent=2))
    return int(bool(errors) or len(times)<2 or (args.tare and report['sensor_ft']['tare'] is None))


if __name__=='__main__':raise SystemExit(main())
