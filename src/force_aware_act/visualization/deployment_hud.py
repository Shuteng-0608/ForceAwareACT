"""Independent ROS-free deployment HUD: log replay or read-only telemetry.

No camera, CUDA or robot SDK is imported. All geometry/filtering/rendering is
performed in the display process. The localhost HTTP interface controls only
presentation. High residuals remain visible, explicitly labelled as estimates.
"""
from __future__ import annotations

import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import multiprocessing as mp
from pathlib import Path
import threading
import time
import sys
from types import SimpleNamespace as NS
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

from force_aware_act.control.kinematics import right_flange_transform
from force_aware_act.control.wrench_estimation import flange_jacobian, equivalent_wrench
from .tianji_hud_render import AXES, HudModel, RingMapping, render_hud


DEFAULT_CONFIG = Path(__file__).resolve().parents[3]/'configs/deployment/force_hud.json'


def read_config(path=DEFAULT_CONFIG):
    config = json.loads(Path(path).read_text())
    if config['schema_version'] != 'deployment_force_hud_v1':
        raise ValueError('unsupported HUD config')
    if not 1 <= config['display_rate_hz'] <= 30:
        raise ValueError('HUD display rate must be 1..30 Hz')
    if config['show_high_residual_samples'] is not True:
        raise ValueError('deployment HUD requires explicit full residual display')
    if config['mapping']['frame'] != config['wrench_frame']:
        raise ValueError('task axes and estimated wrench must use the same frame')
    RingMapping(**config['mapping'])
    HudModel(timeout=config['input_timeout_s'], smoothing_sec=config['smoothing_time_constant_s'])
    return config


class DeploymentHud:
    """Causal display adapter. q/tau [7] -> base wrench [6] -> uint8 BGR image."""
    def __init__(self, config, goal):
        self.config, self.goal = config, goal
        self.model = HudModel(timeout=config['input_timeout_s'],
                              smoothing_sec=config['smoothing_time_constant_s'])
        self.mapping = RingMapping(**config['mapping'])
        self.packet = None
        self.estimate = None
        self.last_source_ns = None
        self.mode, self.raw_display = config['display_mode'], False

    def update(self, packet, now_ns):
        self.packet = packet
        record = packet.get('record')
        if not record or packet.get('bias_nm') is None:
            self.estimate = None
            self.model.invalidate('WAITING for deployment software calibration')
            return
        if not packet.get('bias_valid', True):
            self.estimate = None
            self.model.invalidate('Calibration context changed / no current zero')
            return
        source_ns = record['received_monotonic_ns']
        if source_ns == self.last_source_ns:
            return  # Repeated frame cannot refresh age or advance the filter.
        self.last_source_ns = source_ns
        try:
            q = np.asarray(record['qpos_rad'], dtype=float)
            bias = np.asarray(packet['bias_nm'], dtype=float)
            tau = np.asarray(record['external_raw_nm'], dtype=float)-bias
            if q.shape != (7,) or tau.shape != (7,) or bias.shape != (7,):
                raise ValueError('expected seven joints and seven calibration biases')
            out = equivalent_wrench(flange_jacobian(q), tau)
            w = out['wrench_base']
            transform = right_flange_transform(q)
            distance = np.linalg.norm(transform[:3, 3]/1000.-self.goal['target_position_m'])*1000.
            issues = []
            if out['relative_residual'] > .5:
                issues.append('HIGH RESIDUAL')
            if out['scaled_condition'] > 200 or out['rank'] < 6:
                issues.append('ILL CONDITIONED')
            if np.linalg.norm(w[:3]) > 1000 or np.linalg.norm(w[3:]) > 100:
                issues.append('LARGE ESTIMATE')
            self.estimate = {'source_ns': source_ns, 'qpos_rad': q.tolist(),
                'zeroed_tau_nm': tau.tolist(), 'wrench_base': w.tolist(),
                'relative_residual': out['relative_residual'], 'condition': out['scaled_condition'],
                'rank': out['rank'], 'diagnostics': issues, 'distance_mm': float(distance)}
            finite = (np.isfinite(w).all() and math.isfinite(out['scaled_condition'])
                      and math.isfinite(out['relative_residual']))
            if not finite:
                self.estimate = None
                self.model.invalidate('INVALID: non-finite estimate or condition')
                return
            fields = {'schema_version': '1', 'convention_confirmed': 'false',
                      'wrench_frame': self.config['wrench_frame'],
                      'scaled_condition': str(out['scaled_condition']),
                      'relative_residual': str(out['relative_residual'])}
            fields.update({'raw_'+axis: str(value) for axis, value in zip(AXES, w)})
            status = NS(name='right', level=1 if finite else 2,
                        message=' / '.join(issues) or 'UNVERIFIED equivalent flange estimate',
                        values=[NS(key=k, value=v) for k, v in fields.items()])
            left = NS(name='left', level=2, message='not displayed', values=[NS(key='schema_version', value='1')])
            report = NS(header=NS(stamp=NS(sec=source_ns//1_000_000_000,
                                          nanosec=source_ns%1_000_000_000)), status=[left, status])
            self.model.ingest(report, now_ns, now_ns/1e9)
        except (ValueError, KeyError, TypeError, np.linalg.LinAlgError) as exc:
            self.estimate = None
            self.model.invalidate('INVALID: '+str(exc))

    def status(self, now_ns):
        self.model.refresh(now_ns, now_ns/1e9)
        record = (self.packet or {}).get('record')
        age = (now_ns-record['received_monotonic_ns'])/1e9 if record else None
        view = self.model.arms['right']
        return {'state': (self.packet or {}).get('state', 'WAITING'),
                'calibration_status': (self.packet or {}).get('calibration_status', 'waiting'),
                'stop_reason': (self.packet or {}).get('stop_reason'),
                'goal_id': self.goal['goal_id'], 'source_age_s': age,
                'source_mode':(self.packet or {}).get('source_mode','deployment_feedback'),
                'valid': view.valid, 'message': view.message,
                'display_mode': self.mode, 'raw_display': self.raw_display,
                'filtered_wrench_base': view.filtered.tolist() if view.valid else None,
                'estimate': self.estimate, 'hardware_connected_by_hud': False,
                'commands_sent_by_hud': False}

    def render(self, now_ns):
        status = self.status(now_ns)
        view = self.model.arms['right']
        saved = view.filtered
        if self.raw_display and view.valid:
            view.filtered = view.raw
        try:
            image = render_hud(self.model, status['source_age_s'] if status['source_age_s'] is not None else math.nan,
                               self.config['width'], self.config['height'],
                               self.config['force_display_range_N'], self.config['torque_display_range_Nm'],
                               self.mode, {'left': RingMapping(), 'right': self.mapping}, True, 'right')
        finally:
            view.filtered = saved
        # Original right-arm renderer is centered. Side panels keep its layout.
        scale_x, scale_y = image.shape[1]/1280, image.shape[0]/720
        def text(value, x, y, scale=.48, color=(225,225,225)):
            cv2.putText(image, str(value), (int(x*scale_x),int(y*scale_y)), cv2.FONT_HERSHEY_SIMPLEX,
                        scale*min(scale_x,scale_y), color, 1, cv2.LINE_AA)
        label={'saved_log_replay':'LOG REPLAY','readonly_feedback':'READ ONLY'}.get(status['source_mode'],'DEPLOYMENT')
        text(label+' / '+self.goal['goal_id'], 22, 142, .64)
        text(status['state'], 22, 176, .52)
        text('ZERO: '+status['calibration_status'], 22, 205)
        text('VIEW: '+('RAW' if self.raw_display else f"CAUSAL LPF {self.config['smoothing_time_constant_s']*1000:g}ms"), 22, 234)
        reason = status['stop_reason']
        if reason:
            text('STOP: '+reason[:34], 22, 270, .4, (90,190,240))
        text('JOINT DISTURBANCE / Nm', 22, 310, .51)
        record = (self.packet or {}).get('record')
        fresh = record and 0 <= status['source_age_s'] <= self.config['input_timeout_s']
        torque = self.estimate['zeroed_tau_nm'] if self.estimate and fresh else None
        limits = self.goal['config']['limits']
        for j in range(7):
            y = 350+j*35
            value = f'{torque[j]:+.3f}' if torque is not None else '--'
            text(f'J{j+1}  {value}', 24, y)
            if torque is not None:
                ratio = min(abs(torque[j])/limits['torque_soft_nm'][j], 1.)
                color = (70,175,235) if abs(torque[j]) >= limits['torque_soft_nm'][j] else (210,185,80)
                cv2.rectangle(image, (int(166*scale_x),int((y-12)*scale_y)),
                              (int((166+130*ratio)*scale_x),int((y-3)*scale_y)), color, -1)
        if view.valid and self.estimate:
            e = self.estimate
            text('BASE COMPONENTS / '+('RAW' if self.raw_display else 'LPF'), 970, 143, .45)
            values = view.raw if self.raw_display else view.filtered
            for j, axis in enumerate(AXES):
                text(f'{axis}: {values[j]:+.3f} '+('N' if j<3 else 'Nm'), 970, 185+j*31, .5)
            text(f'|F| RAW: {np.linalg.norm(view.raw[:3]):.2f} N', 970, 405)
            text(f'|M| RAW: {np.linalg.norm(view.raw[3:]):.3f} Nm', 970, 437)
            text(f'RESIDUAL: {e["relative_residual"]:.1%}', 970, 480)
            text(f'CONDITION: {e["condition"]:.2f}', 970, 512)
            text(f'GOAL: {e["distance_mm"]:.1f} mm', 970, 544)
            if e['diagnostics']:
                text(' / '.join(e['diagnostics'])[:30], 970, 582, .4, (90,190,240))
        else:
            text('NO FRESH ZEROED ESTIMATE', 970, 200, .4, (90,190,240))
        # The deployment bias already zeros the shared source, no second tare.
        cv2.rectangle(image, (int(340*scale_x),int(564*scale_y)),
                      (int(937*scale_x),int(594*scale_y)), (44,37,29), -1)
        text('Deployment joint zero | no additional display tare', 363, 583, .44)
        return image, status


HTML = b'''<!doctype html><meta charset="utf-8"><title>Tianji Force HUD</title>
<style>body{margin:0;background:#171310;color:#ddd;font:16px sans-serif}header{padding:10px}
button{margin:3px;padding:8px}img{display:block;width:min(100vw,1280px);height:auto}</style>
<header>Tianji deployment HUD | display only
<button onclick="view('ring')">Ring</button><button onclick="view('bars')">Six-axis bars</button>
<button onclick="raw=!raw;view(mode)">Raw / filtered</button><span id="state"></span></header>
<img id="frame" alt="Waiting for HUD">
<script>let raw=false,mode='ring',busy=false;
async function view(m){mode=m;await fetch('/view?mode='+m+'&raw='+(raw?1:0))}
setInterval(async()=>{if(busy)return;busy=true;try{let r=await fetch('/frame.jpg',{cache:'no-store'});
if(r.ok){let u=URL.createObjectURL(await r.blob());let old=frame.src;frame.src=u;
if(old.startsWith('blob:'))URL.revokeObjectURL(old)}}catch(e){}finally{busy=false}},33);
setInterval(async()=>{try{let s=await(await fetch('/status.json',{cache:'no-store'})).json();
document.getElementById('state').textContent=' | '+s.source_mode+' | '+s.state+' | source '+
(s.source_age_s===null?'--':Math.round(s.source_age_s*1000))+' ms'}catch(e){
document.getElementById('state').textContent=' | HUD disconnected'}},500);</script>'''


class HudServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def display_worker(channel, stop, ready, config, goal, output, port):
    """Independent CPU renderer and localhost web server; no SDK references."""
    output = Path(output)
    server = None
    errors, times, frames, skipped = [], [], 0, 0
    state = {'jpeg': b'', 'status': {}, 'mode': config['display_mode'], 'raw': False}
    cache_lock = threading.Lock()
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass
        def do_GET(self):
            parsed = urlparse(self.path)
            if parsed.path == '/':
                data, content = HTML, 'text/html; charset=utf-8'
            elif parsed.path == '/view':
                query = parse_qs(parsed.query)
                mode, raw = query.get('mode', ['ring'])[0], query.get('raw', ['0'])[0]
                if mode not in ('ring','bars') or raw not in ('0','1'):
                    self.send_error(400); return
                with cache_lock:
                    state['mode'], state['raw'] = mode, raw=='1'
                data, content = b'view updated', 'text/plain'
            elif parsed.path in ('/frame.jpg','/status.json'):
                with cache_lock:
                    data = state['jpeg'] if parsed.path == '/frame.jpg' else json.dumps(state['status']).encode()
                content = 'image/jpeg' if parsed.path == '/frame.jpg' else 'application/json'
            else:
                self.send_error(404); return
            self.send_response(200)
            self.send_header('Content-Type', content)
            self.send_header('Cache-Control', 'no-store')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            try:
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):
                pass
    try:
        cv2.setNumThreads(1)
        server = HudServer(('127.0.0.1',port), Handler)
        model = DeploymentHud(config, goal)
        packet = channel.read()
        now = time.monotonic_ns()
        if packet:
            model.update(packet, now)
        image, status = model.render(now)
        ok, encoded = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY,80])
        if not ok:
            raise RuntimeError('HUD JPEG encoder unavailable')
        state['jpeg'], state['status'] = encoded.tobytes(), status
        http_thread = threading.Thread(target=server.serve_forever, daemon=True)
        http_thread.start()
        ready.set()
        next_render = time.monotonic()
        with (output/'display.jsonl').open('x') as log:
            while not stop.is_set():
                started = time.perf_counter_ns()
                packet = channel.read()
                now = time.monotonic_ns()
                if packet:
                    model.update(packet, now)
                with cache_lock:
                    model.mode, model.raw_display = state['mode'], state['raw']
                image, status = model.render(now)
                ok, encoded = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY,80])
                if not ok:
                    raise RuntimeError('JPEG encoding failed')
                with cache_lock:
                    state['jpeg'], state['status'] = encoded.tobytes(), status
                log.write(json.dumps({'render_ns':now, **status}, allow_nan=False)+'\n')
                if frames == 0 or (status['valid'] and not (output/'preview.png').exists()):
                    cv2.imwrite(str(output/('preview.png' if status['valid'] else 'waiting.png')), image)
                frames += 1
                times.append((time.perf_counter_ns()-started)/1e6)
                next_render += 1/config['display_rate_hz']
                current = time.monotonic()
                if next_render < current:
                    skipped += 1
                    next_render = current+1/config['display_rate_hz']
                stop.wait(max(0,next_render-current))
        cv2.imwrite(str(output/'last_frame.png'), image)
    except Exception as exc:
        errors.append({'type':type(exc).__name__, 'message':str(exc)})
    finally:
        if server:
            if ready.is_set():
                server.shutdown()
            server.server_close()
        (output/'report.json').write_text(json.dumps({'frames':frames,'skipped_render_slots':skipped,
            'render_ms_p99':float(np.percentile(times,99)) if times else None,
            'render_ms_max':max(times) if times else None,
            'errors':errors,'commands_sent':False,'robot_connection_opened':False,
            'torch_imported':'torch' in sys.modules,'camera_library_imported':'pyrealsense2' in sys.modules,
            'robot_transport_imported':'force_aware_act.control.sdk_transport' in sys.modules,
            'show_high_residual_samples':True,'display_config':config},indent=2)+'\n')


class HudProcess:
    def __init__(self, channel, config, goal, output, port=8787):
        self.output = Path(output)
        self.output.mkdir(parents=True,exist_ok=False)
        context = mp.get_context('spawn')
        self.stop, self.ready = context.Event(), context.Event()
        self.process = context.Process(target=display_worker,
            args=(channel,self.stop,self.ready,config,goal,str(self.output),port))
        self.process.start()
        self.url = f'http://127.0.0.1:{port}'
        if not self.ready.wait(5):
            self.close()
            raise RuntimeError('HUD startup failed before device connection; inspect hud/report.json')

    def close(self):
        self.stop.set()
        self.process.join(3)
        if self.process.is_alive():
            self.process.terminate()  # This process has no robot connection.
            self.process.join(1)
        path = self.output/'report.json'
        return json.loads(path.read_text()) if path.exists() else {'errors':['HUD exited without report']}


def replay_packets(run):
    """Use recorded actual feedback and original per-run bias, never predictions."""
    run = Path(run)
    report = json.loads((run/'report.json').read_text())['controller']
    history = sorted(report['state_history'],key=lambda row:row['at_ns'])
    calibration = report.get('calibration')
    records = [json.loads(line)['value'] for line in (run/'controller/events.jsonl').read_text().splitlines()
               if json.loads(line)['kind']=='feedback']
    states = iter(history)
    next_state, state = next(states,None), 'WAITING'
    result = []
    for record in records:
        source = record['received_monotonic_ns']
        while next_state and next_state['at_ns'] <= source:
            state, next_state = next_state['state'], next(states,None)
        zeroed = calibration and source >= calibration['end_ns']
        context_matches = bool(zeroed and record['state']==calibration['mode']
                               and record['impedance_type']==calibration['impedance_type']
                               and [record['tool_kine'],record['tool_dyn']]==calibration['tool_context'])
        stop = report.get('stop') or {}
        result.append({'schema_version':1,'published_ns':source,'record':record,
            'source_mode':'saved_log_replay',
            'state':state,'bias_nm':calibration['bias_nm'] if zeroed else None,
            'bias_valid':context_matches,
            'calibration_status':'valid' if zeroed else 'collecting',
            'stop_reason':stop.get('reason') if source>=stop.get('requested_ns',math.inf) else None,
            'target_packets':None})
    return report['goal_selection'], result


def export_replay(run, output, config):
    """Fast, deterministic saved-log replay; no network/server/devices opened."""
    output = Path(output); output.mkdir(parents=True,exist_ok=False)
    goal, packets = replay_packets(run)
    model = DeploymentHud(config, goal)
    samples, valid, high, elapsed = 0, 0, 0, []
    next_render = 0
    run_start = next(p['record']['received_monotonic_ns'] for p in packets if p['state']=='RUNNING')
    screenshot_times = [run_start+int(t*1e9) for t in (.067,3.5,6.,9.)]
    with (output/'display.jsonl').open('x') as log:
        for packet in packets:
            now = packet['record']['received_monotonic_ns']
            model.update(packet, now)
            if now < next_render:
                continue
            next_render = now+int(1e9/config['display_rate_hz'])
            started = time.perf_counter_ns()
            image, status = model.render(now)
            elapsed.append((time.perf_counter_ns()-started)/1e6)
            samples += 1; valid += int(status['valid'])
            high += int(bool(status['estimate'] and status['estimate']['relative_residual']>.5))
            log.write(json.dumps({'render_ns':now,**status},allow_nan=False)+'\n')
            if screenshot_times and now >= screenshot_times[0]:
                selected = screenshot_times.pop(0)
                cv2.imwrite(str(output/f'frame_{(selected-run_start)/1e9:.3f}s.png'),image)
        cv2.imwrite(str(output/'last_frame.png'),image)
    result = {'source_run':str(Path(run).resolve()),'feedback_samples':len(packets),
              'display_frames':samples,'valid_frames':valid,'high_residual_display_frames':high,
              'render_ms_p99':float(np.percentile(elapsed,99)),
              'source_sha256':hashlib.sha256((Path(run)/'controller/events.jsonl').read_bytes()).hexdigest(),
              'commands_sent':False,'robot_connection_opened':False,'display_config':config}
    (output/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    return result
