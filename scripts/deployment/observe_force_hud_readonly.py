"""Observe the verified feedback-only binary with HUD; never enables motors.

The binary only connects/reads/releases. A new software display zero requires
the operator's current stationary/no-contact confirmation, and a stable window.
This zero belongs to this observation session, not a later enabled rollout.
"""
import argparse
from collections import deque
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np

from force_aware_act.control.goal_selection import select_goal
from force_aware_act.control.hud_telemetry import HudTelemetry
from force_aware_act.visualization.deployment_hud import DEFAULT_CONFIG,HudProcess,read_config

VALIDATED_READER_SHA256='460b4c149d69cfb3223542dced69f5d6046e5e37f7c73750f3fc160cffe6ee73'


def calibration_candidate(rows):
    if len(rows)<2 or rows[-1]['received_monotonic_ns']-rows[0]['received_monotonic_ns']<2_000_000_000:
        return None
    gaps=np.diff([r['received_monotonic_ns'] for r in rows])
    if np.any(gaps<=0) or np.any(gaps>20_000_000):
        return None
    raw=np.asarray([r['external_raw_nm'] for r in rows])
    q=np.asarray([r['qpos_rad'] for r in rows]);dq=np.asarray([r['qvel_rad_s'] for r in rows])
    if raw.shape[1:]!=(7,) or not np.isfinite(raw).all() or not np.isfinite(q).all() or not np.isfinite(dq).all():
        return None
    if np.abs(dq).max()>.01 or np.ptp(q,axis=0).max()>.005 or raw.std(axis=0).max()>.15:
        return None
    context=lambda r:(r['state'],tuple(r['tool_kine']),tuple(r['tool_dyn']))
    if any(context(r)!=context(rows[0]) for r in rows):
        return None
    return {'bias_nm':raw.mean(axis=0).tolist(),'start_ns':rows[0]['received_monotonic_ns'],
            'end_ns':rows[-1]['received_monotonic_ns'],'samples':len(rows),
            'source':'m_EST_Joint_Force','context':context(rows[-1]),
            'std_nm':raw.std(axis=0).tolist()}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reader',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--goal-id',required=True)
    p.add_argument('--controller-profile',type=Path)
    p.add_argument('--config',type=Path,default=DEFAULT_CONFIG)
    p.add_argument('--robot-ip',default='192.168.1.190')
    p.add_argument('--seconds',type=float,default=120)
    p.add_argument('--port',type=int,default=8787)
    p.add_argument('--operator-confirmed-no-contact',action='store_true')
    args=p.parse_args(argv)
    if not args.operator_confirmed_no_contact:
        p.error('current stationary/no-contact confirmation is required for software zero')
    if not np.isfinite(args.seconds) or not 3<=args.seconds<=180:
        p.error('seconds must be 3..180')
    if not 1<=args.port<=65535:
        p.error('port must be 1..65535')
    if hashlib.sha256(args.reader.read_bytes()).hexdigest()!=VALIDATED_READER_SHA256:
        p.error('reader differs from the validated feedback-only binary')
    goal=select_goal(args.goal_id,args.controller_profile) if args.controller_profile else select_goal(args.goal_id)
    args.output.mkdir(parents=True,exist_ok=False)
    channel=HudTelemetry(); hud=HudProcess(channel,read_config(args.config),goal.to_dict(),args.output/'hud',args.port)
    reader=None; rows=deque(); calibration=None; errors=[]; summary=None; frames=0
    last_frame=None; next_publish=0; context=None; hud_report=None
    print('Read-only observer HUD: '+hud.url,flush=True)
    try:
        with (args.output/'reader.stderr').open('x') as stderr, (args.output/'feedback.jsonl').open('x') as log:
            reader=subprocess.Popen([str(args.reader.resolve()),args.robot_ip,str(args.seconds),'--stream'],
                                    stdout=subprocess.PIPE,stderr=stderr,text=True,bufsize=1)
            for line in reader.stdout:
                if not line.startswith('{'):
                    continue
                row=json.loads(line)
                if row.get('kind')=='summary':
                    summary=row;continue
                if row.get('kind')!='feedback':
                    continue
                record={**row['arms'][1],'received_monotonic_ns':row['received_monotonic_ns']}
                if record['frame_serial']==last_frame:
                    continue
                last_frame=record['frame_serial'];frames+=1
                log.write(json.dumps(record)+'\n')
                now=time.monotonic_ns()
                new_context=(record['state'],tuple(record['tool_kine']),tuple(record['tool_dyn']))
                if context is not None and context!=new_context:
                    errors.append('state/tool context changed; observation calibration invalidated')
                    calibration=None
                    reader.terminate();break
                context=new_context
                if record['error_code']!=0:
                    errors.append('controller reports an error; no recovery attempted')
                    calibration=None
                    reader.terminate();break
                if now-record['received_monotonic_ns']<=20_000_000:
                    rows.append(record)
                    while len(rows)>2 and rows[-1]['received_monotonic_ns']-rows[1]['received_monotonic_ns']>=2_000_000_000:
                        rows.popleft()
                    if calibration is None:
                        calibration=calibration_candidate(rows)
                        if calibration:
                            (args.output/'calibration.json').write_text(json.dumps(calibration,indent=2)+'\n')
                            print('Software display zero valid; no controller parameters changed.',flush=True)
                if now>=next_publish:
                    channel.publish({'record':record,'state':'READONLY_OBSERVING' if calibration else 'READONLY_CALIBRATING',
                        'source_mode':'readonly_feedback',
                        'bias_nm':calibration['bias_nm'] if calibration else None,'bias_valid':calibration is not None,
                        'calibration_status':'valid' if calibration else 'collecting',
                        'stop_reason':None,'published_ns':now})
                    next_publish=now+33_333_333
            reader.stdout.close()
            reader.wait(timeout=5)
    except KeyboardInterrupt:
        if reader:
            reader.terminate()
    except Exception as exc:
        errors.append(str(exc))
    finally:
        if reader and reader.poll() is None:
            reader.terminate()
            if reader.stdout:
                for _ in reader.stdout:
                    pass  # Drain SDK output so OnRelease cannot block on a pipe.
            reader.wait(timeout=5)
        time.sleep(.08)
        hud_report=hud.close()
        result={'feedback_samples':frames,'reader_exit_code':reader.returncode if reader else None,
                'reader_summary':summary,'calibration':calibration,'hud':hud_report,'errors':errors,
                'reader_sha256':hashlib.sha256(args.reader.read_bytes()).hexdigest(),
                'commands_sent':False,'control_parameters_changed':False,'mode_changed':False,
                'enable_or_servo_off_requested':False}
        (args.output/'report.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result,indent=2))
    return int(bool(errors) or not calibration or not summary or reader.returncode!=0 or bool(hud_report['errors']))


if __name__=='__main__':
    raise SystemExit(main())
