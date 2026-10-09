#!/usr/bin/env python3
"""Offline numeric references from recorded data; no hardware or fault injection.

These statistics are not approved physical safety limits. HDF5 files are opened
read-only. Candidate force rules are frozen from the 30 training episodes before
evaluating the other 20; the optional all-50 envelope is labelled separately.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import h5py
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.deployment.replay_controller_readonly import read_numeric
from scripts.analyze_real_robot_endpoints import flange_fk, load_model
from force_aware_act.control.contracts import Feedback, Limits
from force_aware_act.control.execution import ActionLimiter
from force_aware_act.real_robot.sampling import estimate_action_grid_phase_ns


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def stats(values):
    a = np.asarray(values, dtype=float)
    if a.ndim != 2 or a.shape[1] != 7 or not len(a) or not np.isfinite(a).all():
        raise ValueError("expected nonempty finite [N,7]")
    return {"samples": len(a), "min": a.min(0).tolist(), "max": a.max(0).tolist(),
            **{f"abs_p{p:g}": np.percentile(np.abs(a), p, axis=0).tolist()
               for p in (50, 95, 99, 99.9)}, "abs_max": np.abs(a).max(0).tolist()}


def window_rate(times, values, window_ns=0, max_gap_ns=20_000_000):
    """Backward secants, without crossing missing-data gaps.

    window=0 gives adjacent-sample derivatives. A positive window chooses the
    latest sample at least window_ns earlier, using actual elapsed time. A
    long-window derivative describes an average change, not an instantaneous cap.
    """
    t = np.asarray(times, dtype=np.int64)
    x = np.asarray(values, dtype=float)
    if len(t) != len(x) or np.any(np.diff(t) <= 0):
        raise ValueError("strictly increasing timestamps required")
    end = np.arange(1, len(t))
    begin = (end - 1 if window_ns == 0 else
             np.searchsorted(t, t[end] - window_ns, side="right") - 1)
    valid = begin >= 0
    end, begin = end[valid], begin[valid]
    segment = np.r_[0, np.cumsum(np.diff(t) > max_gap_ns)]
    valid = segment[end] == segment[begin]
    end, begin = end[valid], begin[valid]
    return end, (x[end] - x[begin]) / ((t[end] - t[begin]) / 1e9)[:, None]


def hysteresis_windows(times, magnitude, soft, release, max_gap_ns=20_000_000):
    """Observed intervals: start >=soft, reset <=release or at a >20ms gap.

    Ends are last observed samples before reset, so no unobserved dwell is added.
    Returns sample indices [start,end]; no gap, duplicate or future interpolation.
    """
    t = np.asarray(times, dtype=np.int64)
    a = np.asarray(magnitude, dtype=float)
    if len(t) != len(a) or np.any(np.diff(t) <= 0) or not 0 <= release < soft:
        raise ValueError("invalid hysteresis input")
    active = a > release
    gap = np.r_[True, np.diff(t) > max_gap_ns]
    starts = np.flatnonzero(active & (gap | ~np.r_[False, active[:-1]]))
    ends = np.flatnonzero(active & (np.r_[gap[1:], True] | ~np.r_[active[1:], False]))
    high = np.flatnonzero(a >= soft)
    result = []
    for begin, end in zip(starts, ends):
        k = np.searchsorted(high, begin)
        if k < len(high) and high[k] <= end:
            result.append((int(high[k]), int(end)))
    return result


def ceil_step(values, step):
    return np.round(np.ceil(np.asarray(values) / np.asarray(step) - 1e-12) * step, 8)


def derived(e):
    t, at = e["times"]["state"], e["times"]["action"]
    result = {"q": e["q"], "action": e["action"], "velocity": e["dq"],
              "external_torque": e["tau"], "internal_torque": e["internal"]}
    for window in (0, 50_000_000, 100_000_000):
        _, result[f"feedback_acceleration_{window//1_000_000}ms"] = window_rate(t, e["dq"], window)
    idx, cmdvel = window_rate(at, e["action"])
    result["command_velocity_native"] = cmdvel
    _, result["command_acceleration_native"] = window_rate(at[idx], cmdvel)
    # Equal 10ms grid with linear interpolation is an OFFLINE comparison only.
    # It is not used by inference/force rules, and excluded near >20ms gaps.
    relative = at - at[0]
    grid = np.arange(0, relative[-1] + 1, 10_000_000, dtype=np.int64)
    right = np.searchsorted(relative, grid, side="left").clip(1, len(at)-1)
    valid = relative[right] - relative[right-1] <= 20_000_000
    qgrid = np.column_stack([np.interp(grid, relative, e["action"][:,j]) for j in range(7)])
    gvalid = valid[1:] & valid[:-1]
    vgrid = np.diff(qgrid, axis=0) / .01
    result["command_velocity_grid10ms"] = vgrid[gvalid]
    result["command_acceleration_grid10ms"] = (np.diff(vgrid, axis=0) / .01)[gvalid[1:] & gvalid[:-1]]
    gaps = np.diff(at) <= 20_000_000
    result["command_step"] = np.diff(e["action"], axis=0)[gaps]
    # Published commands are not evidence of SDK acceptance; use causal joins.
    si = np.searchsorted(t, at, side="right") - 1
    ok = (si >= 0) & (at - t[si.clip(0)] <= 20_000_000)
    result["new_command_minus_latest_feedback"] = e["action"][ok] - e["q"][si[ok]]
    result["first_recorded_command_error"] = result["new_command_minus_latest_feedback"][:1]
    ai = np.searchsorted(at, t, side="right") - 1
    ok = (ai >= 0) & (t - at[ai.clip(0)] <= 30_000_000)
    result["latest_command_minus_feedback"] = e["action"][ai[ok]] - e["q"][ok]
    e["join_coverage"] = {"state_rows": len(t), "valid_latest_command_rows": int(ok.sum())}
    return result


def context(e, time_ns, joint, reason):
    t = e["times"]["state"]
    si = np.searchsorted(t, time_ns, side="right") - 1
    ai = np.searchsorted(e["times"]["action"], time_ns, side="right") - 1
    valid = si >= 0 and ai >= 0
    completion = e["completion_ns"]
    result = {"episode": e["id"], "split": e["split"], "goal": e["group"],
              "time_ns": int(time_ns), "elapsed_s": (time_ns-int(t[0]))/1e9,
              "joint": joint+1, "reason": reason,
              "relative_to_coarse_completion_s": (time_ns-completion)/1e9 if completion is not None else None}
    if valid:
        result.update(feedback_age_ms=(time_ns-int(t[si]))/1e6,
                      command_age_ms=(time_ns-int(e["times"]["action"][ai]))/1e6,
                      distance_mm=float(e["distance_mm"][si]),
                      velocity_rad_s=e["dq"][si].tolist(),
                      command_feedback_error_rad=(e["action"][ai]-e["q"][si]).tolist())
    return result


def evaluate_force(episodes, soft, release, hard, duration_ms):
    rows = []
    for e in episodes:
        t = e["times"]["disturbance"]; absolute = np.abs(e["tau"])
        first = []; maxima = []; all_soft = []; all_hard = []
        for j in range(7):
            hard_idx = np.flatnonzero(absolute[:,j] >= hard[j])
            h = int(t[hard_idx[0]]) if len(hard_idx) else None
            if h is not None: first.append((h, 0, j, "hard"))
            windows = hysteresis_windows(t, absolute[:,j], soft[j], release[j])
            maxima.append(max(((t[b]-t[a])/1e6 for a,b in windows), default=0.))
            candidates = [(a,b) for a,b in windows if t[b]-t[a] >= duration_ms*1e6]
            s = None
            if candidates:
                a,b = candidates[0]
                idx = np.searchsorted(t, t[a] + int(duration_ms*1e6), side="left")
                assert idx <= b
                s = int(t[idx]); first.append((s, 1, j, "soft"))
            all_soft.append(s);all_hard.append(h)
        first.sort()
        row = {"episode": e["id"], "split": e["split"], "group": e["group"],
               "max_hysteresis_duration_ms": [float(x) for x in maxima],
               "soft_first_ns_by_joint": all_soft, "hard_first_ns_by_joint": all_hard,
               "first_trigger": context(e, first[0][0], first[0][2], first[0][3]) if first else None}
        rows.append(row)
    groups = {"train": {"train"}, "validation": {"validation"}, "test": {"test"},
              "heldout": {"validation","test"}, "all": {"train","validation","test"}}
    summary = {}
    for name, selected in groups.items():
        rr = [r for r in rows if r["split"] in selected]
        triggers = [r["first_trigger"] for r in rr if r["first_trigger"]]
        summary[name] = {"episodes":len(rr), "triggered":len(triggers),
                         "first_reasons":dict(Counter(r["reason"] for r in triggers)),
                         "first_joints":dict(Counter(r["joint"] for r in triggers)),
                         "before_coarse_completion":sum(r["relative_to_coarse_completion_s"] is not None and
                                                        r["relative_to_coarse_completion_s"] < 0 for r in triggers),
                         "any_hard":sum(any(x is not None for x in r["hard_first_ns_by_joint"]) for r in rr),
                         "any_soft":sum(any(x is not None for x in r["soft_first_ns_by_joint"]) for r in rr)}
    return {"soft_nm":list(soft), "release_nm":list(release), "hard_nm":list(hard),
            "duration_ms":duration_ms, "summary":summary, "episodes":rows}


def motion_references(episodes, summary):
    """Train-only candidates and ordinary recorded-command limiter comparison."""
    train=summary['train']
    low=np.floor((np.minimum(train['q']['min'],train['action']['min'])-.05)/.01)*.01
    high=ceil_step(np.maximum(train['q']['max'],train['action']['max'])+.05,.01)
    velocity=ceil_step(np.asarray(train['velocity']['abs_max'])*1.2,.05)
    following=ceil_step(np.maximum(train['new_command_minus_latest_feedback']['abs_max'],
                                  train['latest_command_minus_feedback']['abs_max'])*1.5,.005)
    accel={name:ceil_step(np.asarray(train[key]['abs_p99.9'])*1.2,.5)
           for name,key in [('smooth50','feedback_acceleration_50ms'),
                            ('command_p999','command_acceleration_grid10ms')]}
    result={'authority':'dataset_diagnostic','approved_for_motion':False,'source':'train_only',
            'joint_min_rad':low.tolist(),'joint_max_rad':high.tolist(),
            'velocity_rad_s':velocity.tolist(),'following_error_rad':following.tolist(),
            'first_action_error_rad':[.02]*7,'command_step_reject_rad':.01,
            'acceleration_candidates_rad_s2':{k:v.tolist() for k,v in accel.items()},
            'max_consecutive_limited_ms':200.,'checks':{},'recorded_command_limiter':{}}
    for split, selected in [('train',{'train'}),('heldout',{'validation','test'})]:
        chosen=[e for e in episodes if e['split'] in selected]
        result['checks'][split]={}
        for name,key,limit in [('velocity','velocity',velocity),
                               ('following_proxy','new_command_minus_latest_feedback',following),
                               ('first_recorded_error','first_recorded_command_error',np.ones(7)*.02),
                               ('command_step','command_step',np.ones(7)*.01)]:
            bad=[e['id'] for e in chosen if np.any(np.abs(e['metrics'][key])>limit)]
            result['checks'][split][name]={'exceeding_episodes':bad}
        result['checks'][split]['joint_range']={'exceeding_episodes':[e['id'] for e in chosen if any(
            np.any((e['metrics'][key]<low)|(e['metrics'][key]>high)) for key in ('q','action'))]}
    for name,a in accel.items():
        records=[]
        for e in episodes:
            lim=ActionLimiter(Limits(joint_min_rad=tuple(low),joint_max_rad=tuple(high),
                velocity_rad_s=tuple(velocity),acceleration_rad_s2=tuple(a),
                following_error_rad=tuple(following),first_action_error_rad=(.02,)*7,
                authority='dataset_diagnostic'))
            t=e['times']['state'];at=e['times']['action']
            phase=estimate_action_grid_phase_ns(at,10_000_000,2_000_000_000)
            anchor=int(at[0])+int(phase['phase_ns'])
            earliest=max(int(t[0]),int(at[0]))+250_000_000
            origin=anchor+max(0,(earliest-anchor+9_999_999)//10_000_000)*10_000_000
            end=min(int(t[-1]),int(at[-1]))
            ticks=np.arange(origin,end+1,10_000_000,dtype=np.int64)
            si=np.searchsorted(t,ticks,side='right')-1;ai=np.searchsorted(at,ticks,side='right')-1
            counts=Counter();fault=None;max_run=0;run=0
            for tick,s,k in zip(ticks,si,ai):
                if tick-t[s]>20_000_000 or tick-at[k]>20_000_000:
                    fault={'reason':'recorded_stream_gap','elapsed_s':(int(tick)-int(t[0]))/1e9};break
                fb=Feedback(int(t[s]),int(s),(tuple(e['q'][s])),tuple(e['dq'][s]),(0.,)*7,int(t[s]),True,'offline_motion_only')
                row=lim.apply(e['action'][k],fb,int(tick));counts['ticks']+=1
                counts['limited']+=bool(row['limited']);run=run+1 if row['limited'] else 0;max_run=max(run,max_run)
                if row['fault']:
                    fault={'reason':row['fault'],'elapsed_s':(int(tick)-int(t[0]))/1e9};break
            records.append({'episode':e['id'],'split':e['split'],'counts':dict(counts),
                            'max_consecutive_limited_ticks':max_run,'first_fault':fault})
        result['recorded_command_limiter'][name]={'scope':'ordinary recorded command causal replay at stable 100Hz grid; synthetic force ignored; no model/hardware/fault injection; stop at first limiter rejection',
            'summary':{part:{'episodes':len([e for e in records if e['split'] in selected]),
                            'first_faults':dict(Counter(e['first_fault']['reason'] if e['first_fault'] else 'none'
                                                 for e in records if e['split'] in selected)),
                            'limited_fraction_until_first_rejection':sum(e['counts'].get('limited',0) for e in records if e['split'] in selected)/sum(e['counts']['ticks'] for e in records if e['split'] in selected)}
                       for part,selected in [('train',{'train'}),('heldout',{'validation','test'})]},'episodes':records}
    # Numerical smoothing diagnostic only: observe correction sizes without the
    # 200ms small-correction latch truncating the trace. No production flag changes.
    correction_rows=[]
    a=accel['smooth50']
    for e in episodes:
        lim=ActionLimiter(Limits(joint_min_rad=tuple(low),joint_max_rad=tuple(high),
            velocity_rad_s=tuple(velocity),acceleration_rad_s2=tuple(a),
            following_error_rad=tuple(following),first_action_error_rad=(.02,)*7,
            max_consecutive_limited_ms=1e9,authority='dataset_diagnostic'))
        t=e['times']['state'];at=e['times']['action']
        phase=estimate_action_grid_phase_ns(at,10_000_000,2_000_000_000)
        anchor=int(at[0])+int(phase['phase_ns']);earliest=max(int(t[0]),int(at[0]))+250_000_000
        origin=anchor+max(0,(earliest-anchor+9_999_999)//10_000_000)*10_000_000
        ticks=np.arange(origin,min(int(t[-1]),int(at[-1]))+1,10_000_000,dtype=np.int64)
        si=np.searchsorted(t,ticks,side='right')-1;ai=np.searchsorted(at,ticks,side='right')-1
        residuals=[];used_times=[];fault=None
        for tick,s,k in zip(ticks,si,ai):
            if tick-t[s]>20_000_000 or tick-at[k]>20_000_000:
                fault='recorded_stream_gap';break
            fb=Feedback(int(t[s]),int(s),tuple(e['q'][s]),tuple(e['dq'][s]),(0.,)*7,int(t[s]),True,'offline_motion_only')
            row=lim.apply(e['action'][k],fb,int(tick))
            if row['fault']:fault=row['fault'];break
            residuals.append(np.abs(np.asarray(row['action_rad'])-e['action'][k]));used_times.append(int(tick))
        residuals=np.asarray(residuals);tt=np.asarray(used_times,dtype=np.int64)
        maximum=np.max(residuals,axis=1)
        correction_rows.append({'episode':e['id'],'split':e['split'],'first_other_rejection':fault,
            'residual_rad':stats(residuals),
            'max_observed_duration_ms':{str(level):float(max(((tt[b]-tt[a])/1e6 for a,b in
                hysteresis_windows(tt,maximum,level,level*(1.-1e-8))),default=0.)) for level in (.002,.005,.01)}})
    result['smoothing_correction_diagnostic']={'scope':'recorded commands with smooth50 acceleration; 200ms tiny-correction latch bypassed ONLY in offline numeric diagnostic; all other limiter rejection rules retained; original robot feedback is fixed, not simulated response',
        'episodes':correction_rows}
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',type=Path,default=Path('/home/stw/Downloads/1mm-gap-10points_hdf5_raw_v2'))
    p.add_argument('--split',type=Path,default=ROOT/'configs/datasets/1mm_gap_50_10points_v2/split.json')
    p.add_argument('--bridge-config',type=Path,default=Path('/home/stw/tianji_ws/src/tianji_io_teleop_bridge/config/bridge_both_test.yaml'))
    p.add_argument('--joint-model',type=Path,default=Path('/home/stw/tianji_ws/src/tianji_io_teleop_bridge/include/tianji_io_teleop_bridge/joint_model.hpp'))
    p.add_argument('--fk-model',type=Path,default=Path('/home/stw/TJ_FX_ROBOT_CONTRL_SDK/CommonConfig/ccs_m6_40.MvKDCfg'))
    p.add_argument('--goal-results',type=Path,default=ROOT/'outputs/spherical_goals_20261007/episodes.json')
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    split=json.loads(args.split.read_text());goals={e['episode']:e for e in json.loads(args.goal_results.read_text())}
    sets={key:set(split[key+'_episode_ids']) for key in ('train','validation','test')}
    episodes=[read_numeric(path) for path in sorted(args.data_root.rglob('episode.hdf5'))]
    assert {e['id'] for e in episodes}==set.union(*sets.values())
    dh=load_model(args.fk_model)
    for e in episodes:
        e['split']=next(k for k,v in sets.items() if e['id'] in v)
        e['group']=split['condition_groups'][e['id']]
        goal_path=ROOT/'configs/deployment/goals'/f"{e['group']}.json"
        center=json.loads(goal_path.read_text())['goal']['target_position_m']
        e['distance_mm']=np.linalg.norm(flange_fk(e['q'],dh)[:,:3,3]-np.array(center)*1000,axis=1)
        c=goals[e['id']]['first_completion']
        e['completion_ns']=int(e['times']['state'][0])+int(round(c['elapsed_s']*1e9)) if c else None
        e['metrics']=derived(e)
        print(e['id'],flush=True)
    summary={}
    for name, selected in {'train':{'train'},'heldout':{'validation','test'},'all':set(sets)}.items():
        chosen=[e for e in episodes if e['split'] in selected]
        summary[name]={key:stats(np.concatenate([e['metrics'][key] for e in chosen])) for key in chosen[0]['metrics']}
    write(args.output/'numeric_statistics.json',summary)
    params=yaml.safe_load(args.bridge_config.read_text())['tianji_marvin_teleop_bridge']['ros__parameters']
    write(args.output/'collection_parameters.json',{'path':str(args.bridge_config),'sha256':sha(args.bridge_config),
          'parameters':params,'qualification':'current user-identified collection config snapshot; runtime overrides and history not fully recorded'})
    force=summary['train']['external_torque'];hard=np.asarray(force['abs_max'])*1.1
    grid=[]
    for quantile in ('abs_p99','abs_p99.9'):
        for multiplier in (1.,1.1,1.2):
            for duration in (50.,100.,200.):
                soft=np.asarray(force[quantile])*multiplier
                result=evaluate_force(episodes,soft,soft*.8,hard,duration)
                result.update(name=f'{quantile}_x{multiplier:g}_{duration:g}ms',source='train_only')
                grid.append(result)
    steps=np.array([.5,.25,.25,.5,.05,.25,.1])
    # Fixed formula, including hard margin; not selected to erase held-out triggers.
    for source in ('train','all'):
        f=summary[source]['external_torque']
        soft=ceil_step(np.asarray(f['abs_p99.9'])*1.2,steps)
        hard=ceil_step(np.asarray(f['abs_max'])*1.15,steps)
        r=evaluate_force(episodes,soft,soft*.8,hard,100.)
        r.update(name=f'{source}_rounded_reference',source='train_only' if source=='train' else 'all50_descriptive_calibration_not_independent_validation')
        grid.append(r)
    write(args.output/'force_candidates.json',grid)
    write(args.output/'candidate_summary.json',[{k:v for k,v in g.items() if k!='episodes'} for g in grid])
    write(args.output/'motion_references.json',motion_references(episodes,summary))
    per_episode=[]
    for e in episodes:
        record={'episode':e['id'],'split':e['split'],'goal':e['group'],'source_mcap_sha256':e['source_sha256'],
                'completion_ns':e['completion_ns'],'join_coverage':e['join_coverage'],
                'metrics':{k:stats(v) for k,v in e['metrics'].items()}}
        record['torque_peak_context']=[context(e,int(e['times']['disturbance'][np.argmax(np.abs(e['tau'][:,j]))]),j,'observed_peak') for j in range(7)]
        record['stream_intervals_ms']={k:{'p50':float(np.median(np.diff(t))/1e6),'max':float(np.max(np.diff(t))/1e6),
                                         'gaps_over20ms':int(np.sum(np.diff(t)>20_000_000))}
                                       for k,t in e['times'].items()}
        per_episode.append(record)
    write(args.output/'episodes.json',per_episode)
    provenance={'schema':'protection_reference_statistics_v1','commands_sent':False,'hardware_connected':False,
                'fault_injection_performed':False,'deployment_configuration_modified':False,
                'sources':{str(path):sha(path) for path in [Path(__file__),args.split,args.bridge_config,args.joint_model,args.fk_model,args.goal_results]},
                'source_mcap_hashes':{e['id']:e['source_sha256'] for e in episodes},
                'counts':dict(Counter(e['split'] for e in episodes)),
                'methods':{'timestamps':'MCAP log_ns; forces evaluated on native disturbance samples',
                           'time_gaps':'hysteresis reset and derivative windows excluded across >20ms gaps',
                           'percentiles':'sample-weighted pooled absolute values; per-episode summaries retained',
                           'command_error':'causal latest published command/feedback proxy, not confirmed SDK following error',
                           'first_error':'first recorded command, not true activation/model first action',
                           'command_grid':'OFFLINE linear interpolation to 10ms for derivative comparison only',
                           'completion':'existing 3mm/500ms coarse all-50 sphere, not true success timestamp; context only',
                           'force_comparison':'independent force-rule trigger counts over complete records, not previous combined controller replay; does not stop at coarse completion'}}
    write(args.output/'manifest.json',provenance)
    print('complete',args.output,flush=True)


if __name__=='__main__':main()
