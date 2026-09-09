"""Run four policies at ten paired, area-uniform XZ disk offsets."""

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_sampling_ablation_nominal_rollout_pilot import (
    build_specs, validate_spec_files, build_rollout_command, parse_args,
)


def points(seed=20260909):
    rng = np.random.default_rng(seed)
    radius = 0.002 * np.sqrt(rng.random(10))
    angle = rng.uniform(0, 2 * np.pi, 10)
    return [dict(point_id=f"point_{i+1:02d}", x=float(r*np.cos(a)),
                 z=float(r*np.sin(a)), rollout_seed=i)
            for i, (r, a) in enumerate(zip(radius, angle))]


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    options = parser.parse_args()
    root = options.output_dir.resolve()
    args = parse_args(['--execute', '--max-rollout-steps', '1200'])
    specs = build_specs(args.checkpoint_root)
    validate_spec_files(specs)
    grid = points()
    plan = dict(version='paired_xz_disk_2mm_hud_v1', point_seed=20260909,
                radius_m=0.002, max_steps=1200, points=grid,
                checkpoints={s.model_id: dict(path=str(s.checkpoint),
                             sha256=digest(s.checkpoint)) for s in specs})
    print(json.dumps(plan, indent=2), flush=True)
    if not options.execute:
        return
    root.mkdir(parents=True, exist_ok=True)
    plan_path = root/'plan.json'
    if plan_path.exists():
        assert json.loads(plan_path.read_text()) == plan, 'Existing plan differs'
    else:
        plan_path.write_text(json.dumps(plan, indent=2)+'\n')
    env = os.environ.copy()
    env['MUJOCO_GL'] = 'egl'
    env['PYTHONPATH'] = str(ROOT/'src')
    rows = []
    for point in grid:
        for spec in specs:
            args.output_dir = root/point['point_id']
            args.seed = point['rollout_seed']
            output = args.output_dir/spec.model_id
            command = [v for v in build_rollout_command(args, spec)
                       if not v.startswith(('--hole-offset-x=', '--hole-offset-z='))]
            command += [f"--hole-offset-x={point['x']}", f"--hole-offset-z={point['z']}",
                        '--save-force-hud-video', '--force-hud-camera', 'cctv_cam',
                        '--force-hud-width', '1280', '--force-hud-height', '720',
                        '--force-hud-primary-wrench', 'compensated']
            summary_path = output/'summary.json'
            print(f"[{len(rows)+1}/40] {point['point_id']} {spec.model_id}", flush=True)
            if not summary_path.exists():
                assert not output.exists(), f'Incomplete output: {output}'
                args.output_dir.mkdir(parents=True, exist_ok=True)
                with (args.output_dir/(spec.model_id+'.log')).open('x') as log:
                    result = subprocess.run(command, cwd=ROOT, env=env,
                                            stdout=log, stderr=subprocess.STDOUT)
                assert result.returncode == 0, f'Rollout error: {output}'
            s = json.loads(summary_path.read_text())
            expected = dict(task_profile='moving_hole_fixed_peg', rollout_mode='execute',
                            checkpoint=str(spec.checkpoint.resolve()), max_rollout_steps=1200,
                            terminal_hold_mode='collector', action_select_mode='temporal',
                            temporal_agg_decay=0.01, contact_latent_mode='zero',
                            policy_wrench_source='compensated', force_stop_threshold=40.0,
                            hole_offset_x=point['x'], hole_offset_y=0.0, hole_offset_z=point['z'],
                            seed=point['rollout_seed'], force_hud_video_saved=True)
            for k, v in expected.items():
                assert s.get(k) == v, (output, k, s.get(k), v)
            video = Path(s['force_hud_video_path'])
            assert video.is_file() and video.stat().st_size > 0
            row = dict(point_id=point['point_id'], model=spec.model_id,
                       x_mm=point['x']*1000, z_mm=point['z']*1000,
                       radius_mm=math.hypot(point['x'],point['z'])*1000,
                       seed=point['rollout_seed'], success=s['success'],
                       safe_success=s['safe_success'], success_time=s['success_time'],
                       stop_reason=s['stop_reason'], max_force_n=s['max_force_norm'],
                       final_distance_mm=s['final_peg_to_hole_dist']*1000,
                       hud_video=str(video), summary=str(summary_path))
            rows.append(row)
            with (root/'aggregate.csv').open('w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(row))
                writer.writeheader(); writer.writerows(rows)
            print(f"DONE success={row['success']} peak={row['max_force_n']:.3f}N "
                  f"stop={row['stop_reason']}", flush=True)
    print('ALL_40_COMPLETED', flush=True)


if __name__ == '__main__':
    main()
