#!/usr/bin/env python3
"""Prepare a reproducible 40/10/5 sensor-wrench experiment without modifying HDF5."""
import argparse
import json
from pathlib import Path
import random
import sys

import h5py
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
from force_aware_act.real_robot.config import RealRobotDataConfig
from force_aware_act.real_robot.reader import RawEpisode, discover_real_robot_episodes
from force_aware_act.real_robot.normalization import compute_real_robot_normalization_stats
from force_aware_act.real_robot.sampling import future_torque_slot_indices


def make_split(ids, source_groups, seed=0):
    """Fixed Oct20 collection sessions: 25 morning/early and 30 later recordings."""
    sessions = {}
    for key in sorted(ids):
        session = key.rsplit('_', 2)[1][:2]
        sessions.setdefault(session, []).append(key)
    if {k: len(v) for k, v in sessions.items()} != {'17': 25, '19': 30}:
        raise ValueError('this experiment requires the reviewed 25 + 30 recording sessions')
    if len(set(source_groups.values())) != 55:
        raise ValueError('recording source hashes must be unique before splitting')
    result = {'train_episode_ids': [], 'validation_episode_ids': [], 'test_episode_ids': []}
    rng = random.Random(seed)
    allocation = {'17': (18, 5, 2), '19': (22, 5, 3)}
    for session in sorted(sessions):
        items = sessions[session].copy();rng.shuffle(items)
        start = 0
        for role, count in zip(result, allocation[session]):
            result[role].extend(items[start:start + count]);start += count
    for role in result: result[role].sort()
    return {'version': 'real_robot_recording_split_v2', 'seed': seed, **result,
            'source_groups': source_groups,
            'condition_groups': {key: {'point': 'x14y08', 'session_hour': key.rsplit('_', 2)[1][:2]} for key in ids},
            'allocation_by_session': allocation, 'split_unit': 'complete original recording'}


def distribution(values):
    median = np.median(values, axis=0)
    return {'samples': len(values), 'mean': values.mean(0).tolist(), 'std': values.std(0).tolist(),
            'median': median.tolist(), 'mad': np.median(np.abs(values - median), axis=0).tolist(),
            'quantiles': {str(q): np.quantile(values, q, axis=0).tolist() for q in [0, .001, .01, .25, .5, .75, .99, .999, 1]},
            'absolute_p99': np.quantile(np.abs(values), .99, axis=0).tolist()}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--data-config', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--split-output', type=Path, required=True)
    args=parser.parse_args(argv)
    if args.split_output.exists() or (args.output_dir/'preparation.json').exists():
        raise ValueError('choose new preparation outputs; existing reviewed preparation is preserved')
    cfg=RealRobotDataConfig.from_dict(json.loads(args.data_config.read_text()))
    if cfg.force_dim != 6: raise ValueError('wrench6 preparation requires six sensor channels')
    records=discover_real_robot_episodes(args.data_root,cfg)
    source_groups={}
    identity={}
    for record in records:
        path=record.resolve(args.data_root);stat=path.stat()
        identity[record.episode_id]={'bytes':stat.st_size,'mtime_ns':stat.st_mtime_ns}
        with h5py.File(path,'r') as h: source_groups[record.episode_id]=str(h.attrs['source_sha256'])
    split=make_split([r.episode_id for r in records],source_groups)
    train=[r for r in records if r.episode_id in split['train_episode_ids']]
    stats=compute_real_robot_normalization_stats(args.data_root,train,cfg)
    per_episode={};training_values=[];training_quiet_candidates=[];phase={}
    for record in records:
        raw=RawEpisode(record.resolve(args.data_root),cfg)
        train_role=record.episode_id in split['train_episode_ids']
        if train_role: training_values.append(raw.force)
        # Low-activity candidates are diagnostics only, never certified contact-free calibration.
        windows=[]
        for start in range(0,len(raw.force)-299,300):
            values=raw.force[start:start+300]
            windows.append((float((values.std(0)/np.array(stats.force_std)).mean()),start,values))
        candidate=min(windows,key=lambda item:item[0]) if windows else None
        if candidate and train_role: training_quiet_candidates.append(candidate[2])
        indices=raw.valid_anchor_indices[::10]
        if indices[-1]!=raw.valid_anchor_indices[-1]: indices=np.r_[indices,raw.valid_anchor_indices[-1]]
        hits=slots=full_hits=full_slots=endpoint_hits=endpoints=0
        for grid in indices:
            anchor=int(raw.anchor_timestamps_ns[grid])
            match,_=future_torque_slot_indices(raw.torque_timestamps_ns,anchor,20,cfg,strict_horizon=True)
            hits+=int((match>=0).sum());slots+=match.size
            endpoint_hits+=int((match[:,-1]>=0).sum());endpoints+=20
            if anchor+200000000<=raw.torque_timestamps_ns[-1]:
                full_hits+=int((match>=0).sum());full_slots+=match.size
        normalized=(raw.force-np.asarray(stats.force_mean))/np.asarray(stats.force_std)
        per_episode[record.episode_id]={
            'role':next(role for role in ['train','validation','test'] if record.episode_id in split[role+'_episode_ids']),
            'diagnostics':raw.diagnostics,'wrench_distribution':distribution(raw.force),
            'normalized_abs_p99':np.quantile(np.abs(normalized),.99,axis=0).tolist(),
            'normalized_abs_max':np.abs(normalized).max(0).tolist(),
            'normalized_abs_gt5_fraction':(np.abs(normalized)>5).mean(0).tolist(),
            'audit_stride':10,'audit_windows':len(indices),'future_hits':hits,'future_slots':slots,
            'full_horizon_hits':full_hits,'full_horizon_slots':full_slots,
            'endpoint_hits':endpoint_hits,'endpoint_slots':endpoints,
            'low_activity_candidate':None if candidate is None else {'source_start_index':candidate[1],
                'source_end_index_exclusive':candidate[1]+300,'std':candidate[2].std(0).tolist(),
                'mean':candidate[2].mean(0).tolist(),'contact_free_confirmed':False}}
        phase[record.episode_id]=raw.diagnostics['action_grid']
    values=np.concatenate(training_values)
    normalized=(values-np.asarray(stats.force_mean))/np.asarray(stats.force_std)
    total={key:sum(r[key] for r in per_episode.values()) for key in ['future_hits','future_slots','full_horizon_hits','full_horizon_slots','endpoint_hits','endpoint_slots']}
    report={'version':'sensor_wrench6_training_preparation_v1','data_root':str(args.data_root.resolve()),
            'data_contract':cfg.contract_dict(),'split_counts':{role:len(split[role+'_episode_ids']) for role in ['train','validation','test']},
            'normalization_source':'40 training recordings; each native sensor sample once',
            'normalization':stats.to_dict(),'training_distribution':distribution(values),
            'training_normalized_abs_p99':np.quantile(np.abs(normalized),.99,axis=0).tolist(),
            'training_normalized_abs_max':np.abs(normalized).max(0).tolist(),
            'floor_active_axes':(values.std(0)<np.asarray(cfg.force_std_floor)).tolist(),
            'floor_policy':'0.01 N / 0.001 N*m numerical guards; not calibrated noise thresholds',
            'noise_calibration':'Low-activity one-second candidates saved for review; no contact-free labels inferred',
            'low_activity_candidate_pooled_distribution':distribution(np.concatenate(training_quiet_candidates)),
            'coverage_totals':total,'episodes':per_episode,'source_identity':identity,
            'training_started':False,'dataset_modified':False}
    for record in records:
        stat=record.resolve(args.data_root).stat()
        assert identity[record.episode_id]=={'bytes':stat.st_size,'mtime_ns':stat.st_mtime_ns},'source changed during preparation'
    write_json(args.split_output,split)
    write_json(args.output_dir/'normalization.json',stats.to_dict())
    write_json(args.output_dir/'action_grid_phases.json',phase)
    write_json(args.output_dir/'preparation.json',report)
    print(json.dumps({'split_counts':report['split_counts'],'force_mean':stats.force_mean,'force_std':stats.force_std,
                      'floor_active_axes':report['floor_active_axes'],'coverage_totals':total,'output':str(args.output_dir)},indent=2))
    return report


if __name__=='__main__':main()
