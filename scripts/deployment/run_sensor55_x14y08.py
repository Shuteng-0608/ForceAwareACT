"""Pinned sensor55 Contact entry. Default previews only; no checkpoint override.

Actual hardware execution requires an explicitly enabled per-run session,
--enable-control and current --operator-confirmed-no-contact. The delegated
controller verifies the full checkpoint digest before opening the robot.
"""
import argparse
from datetime import datetime
import json
from pathlib import Path
import uuid
from zoneinfo import ZoneInfo

ROOT=Path(__file__).resolve().parents[2]
BINDING=ROOT/'configs/deployment/sensor55_model_binding.json'
EXPECTED_MODEL_SHA='5ae5f9d2f9abc6d82e04e8a4e7a3ea3e1aeec6f2599ccfdf6d0f2574fd699875'


def controller_arguments(argv=None, *, binding_path=None,
                         expected_model_sha=EXPECTED_MODEL_SHA, model_kind='contact'):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--session-config',type=Path)
    p.add_argument('--output',type=Path,help='optional; enabled runs default to a timestamped reports directory')
    p.add_argument('--seconds',type=float,default=3.,help='policy test up to 10s; preparation up to 3s')
    p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--enable-control',action='store_true')
    p.add_argument('--operator-confirmed-no-contact',action='store_true')
    a=p.parse_args(argv)
    limit=3 if a.prepare_only else 10
    if not 0<a.seconds<=limit:p.error(f'this acceptance mode is limited to {limit} seconds')
    b=json.loads((binding_path or BINDING).read_text());manifest=ROOT/b['model_manifest']
    m=json.loads(manifest.read_text())
    if (m['checkpoint_sha256']!=expected_model_sha or b['checkpoint_sha256']!=expected_model_sha
            or m['model_kind']!=model_kind or m['force_dim']!=6 or m['action_chunk_len']!=20
            or m['force_frame']!='right_ft_measurement'):
        raise ValueError(f'pinned sensor55 {model_kind} model contract changed')
    if a.enable_control and b.get('require_servo_off_on_fault'):
        from force_aware_act.control.command_session import SessionOptions
        options=SessionOptions.read(a.session_config or ROOT/b['session_config'])
        if not options.servo_off_on_fault:
            p.error('this entry requires servo_off_on_fault=true; create a new session from the current candidate config')
    if a.output is None and a.enable_control:
        stamp=datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S_%f')
        phase='prepare' if a.prepare_only else 'rollout'
        a.output=ROOT/'reports'/f'sensor55_{model_kind}_x14y08_{phase}_{a.seconds:g}s_{stamp}_{uuid.uuid4().hex[:6]}'
    args=['--goal-id','x14y08','--model-manifest',str(manifest),
          '--checkpoint',b['checkpoint'],'--controller-profile',str(ROOT/b['controller_profile']),
          '--session-config',str(a.session_config or ROOT/b['session_config']),
          '--transport',str(ROOT/b['transport']),'--sdk',b['sdk'],'--robot-ip','192.168.1.190',
          '--seconds',str(a.seconds),'--hud']
    if a.output is not None:args+=['--output',str(a.output)]
    for flag in ('prepare_only','enable_control','operator_confirmed_no_contact'):
        if getattr(a,flag):args+=['--'+flag.replace('_','-')]
    return args


def main(argv=None):
    from scripts.deployment.run_contact_controller import main as controller_main
    return controller_main(controller_arguments(argv))


if __name__=='__main__':raise SystemExit(main())
