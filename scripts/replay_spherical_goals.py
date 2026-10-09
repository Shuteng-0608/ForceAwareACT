#!/usr/bin/env python3
"""Prepare accepted coarse goal configs and replay actual feedback, file-only.

Goal-only replay has no force/policy/camera gate and cannot certify the whole
safety controller. Full controller priority/latching is tested separately.
Joint-space comparison uses per-joint boxes, no angle wrapping or velocity gate.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import gzip
import json
from pathlib import Path
import sys
import time

import h5py
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from force_aware_act.control.contracts import Feedback, Goal, Pose, ShadowConfig
from force_aware_act.control.kinematics import FRAME_ID, POSE_SOURCE, MODEL_SHA256, right_flange_pose
from force_aware_act.control.supervision import GoalMonitor
from scripts.analyze_real_robot_endpoints import (
    continuous_intervals, flange_fk, load_model, read_episode, sha256, write_json,
)


def prepare_configs(statistics, output, radius_mm=3., dwell_ms=500.):
    if radius_mm <= 0 or dwell_ms <= 0 or not np.isfinite([radius_mm,dwell_ms]).all():
        raise ValueError("positive finite radius and dwell required")
    groups=json.loads(Path(statistics).read_text())["last_0.5s"]
    expected={f"x{x}y{y:02}" for x in range(13,18) for y in (7,8)}
    if set(groups)!=expected or any(g["n_episodes"]!=5 for g in groups.values()):
        raise ValueError("expected reviewed ten groups with five demonstrations each")
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    files={}
    for name,group in groups.items():
        path=output/f"{name}.json"
        base=ShadowConfig.read(path) if path.exists() else ShadowConfig()
        goal=Goal(goal_id=name,kind="sphere",verified=False,frame_id=FRAME_ID,
                  pose_source=POSE_SOURCE,target_position_m=tuple(np.asarray(group["center_mm"])/1000),
                  radius_m=radius_mm/1000.,dwell_ms=dwell_ms,coarse_completion_accepted=True,
                  provenance=f"operator accepted coarse completion region 2026-10-07; all-50 last-0.5s equal-episode medians; {Path(statistics).name} sha256={sha256(statistics)}; flange, not calibrated peg-tip TCP")
        write_json(path,replace(base,goal=goal).to_dict())
        files[name]={"path":str(path),"sha256":sha256(path),"center_mm":group["center_mm"]}
    return {"scope":"operator accepted coarse completion; NO physical TCP calibration or motion authorization",
            "statistics":str(Path(statistics).resolve()),"statistics_sha256":sha256(statistics),
            "model_sha256":MODEL_SHA256,"frame_id":FRAME_ID,"pose_source":POSE_SOURCE,
            "radius_mm":radius_mm,"dwell_ms":dwell_ms,"joint_velocity_gate":False,
            "orientation_gate":False,"completion_latched_in_shadow_controller":True,"allow_motion":False,"goals":files}


def occupancy(times, inside, dwell_s=.5):
    runs=continuous_intervals(times,inside)
    eligible=[r for r in runs if r["duration_s"]>=dwell_s]
    return {"reached":bool(eligible),"max_duration_s":max((r["duration_s"] for r in runs),default=0.),
            "first_interval":eligible[0] if eligible else None}


def goal_only_replay(e, goal, dh):
    with h5py.File(e["path"],"r") as f:
        ts=np.asarray(f["timestamps/state/log_ns"],dtype=np.int64)
    ts=ts-ts[0]
    transforms=flange_fk(e["q"],dh)
    quat=Rotation.from_matrix(transforms[:,:3,:3]).as_quat()
    monitor=GoalMonitor(goal)
    zero=(0.,)*7
    first=None
    for i,t in enumerate(ts):
        # GoalMonitor uses q, dq and pose only. Synthetic force/context are not
        # fed to SafetySupervisor and do not claim historical calibration validity.
        f=Feedback(int(t),i+1,tuple(e["q"][i]),tuple(e["dq"][i]),zero,int(t),True,"goal_only_offline")
        p=Pose(int(t),FRAME_ID,tuple(transforms[i,:3,3]/1000.),tuple(quat[i]))
        result=monitor.update(f,int(t),p)
        if result["status"]=="COARSE_REACHED" and first is None:
            first={"elapsed_s":float(t/1e9),"seconds_before_record_end":float((ts[-1]-t)/1e9),
                   "result":result}
    return {"episode":e["episode"],"goal_id":goal.goal_id,"samples":len(ts),
            "first_completion":first,"last_status":monitor.result,
            "reached_before_final_5s":bool(first and first["seconds_before_record_end"]>5),
            "motor_control_api_calls":0,"source_sha256":e["source_sha256"]}


def space_comparison(e, records, dh):
    """Same four reference episodes for both spaces, no self-center leakage."""
    tail=e["t"]>=-5
    times=e["t"][tail]
    q=e["q"][tail]
    p=flange_fk(q,dh)[:,:3,3]
    all_names=sorted({r["condition"] for r in records})
    rows=[]
    for name in all_names:
        refs=[r["last_0.5s"] for r in records if r["condition"]==name and r["episode"]!=e["episode"]]
        qref=np.median([r["q_median_rad"] for r in refs],axis=0)
        pref=np.median([r["center_mm"] for r in refs],axis=0)
        qerr=np.rad2deg(np.max(np.abs(q-qref),axis=1))
        perr=np.linalg.norm(p-pref,axis=1)
        rows.append({"goal_id":name,"is_selected_goal":name==e["condition"],
                     "joint":{f"{r}deg":occupancy(times,qerr<=r) for r in (1,2,3,5)},
                     "sphere":{f"{r}mm":occupancy(times,perr<=r) for r in (1,2,3,5)}})
    return {"episode":e["episode"],"condition":e["condition"],"goals":rows}


def summarize_comparison(rows):
    result={}
    for space,unit in (("sphere","mm"),("joint","deg")):
        result[space]={}
        for radius in (1,2,3,5):
            key=f"{radius}{unit}"
            own=sum(next(g for g in e["goals"] if g["is_selected_goal"])[space][key]["reached"] for e in rows)
            ambiguous=sum(any(g[space][key]["reached"] for g in e["goals"] if not g["is_selected_goal"]) for e in rows)
            result[space][key]={"own_goal_reached":own,"episodes_also_matching_other_hole":ambiguous}
    return result


def replay_sample_fixture(path, goals_dir):
    """Replay portable recorded q/dq samples through online FK + goal logic.

    Only goal decisions are checked. Force, cameras and predictions are absent;
    synthetic Feedback fields below are NOT safety/calibration evidence.
    """
    with gzip.open(path, "rt") as handle:
        payload=json.load(handle)
    if payload.get("schema_version")!="recorded_goal_samples_v1":
        raise ValueError("unknown recorded sample schema")
    reports=[]
    for episode in payload["episodes"]:
        goal=ShadowConfig.read(Path(goals_dir)/f"{episode['goal_id']}.json").goal
        monitor=GoalMonitor(goal);counts={};first=None;previous=None
        for i,row in enumerate(episode["samples"]):
            t=row["timestamp_ns"]
            if previous is not None and t<=previous:raise ValueError("unordered recorded fixture")
            previous=t
            feedback=Feedback(t,i+1,tuple(row["q_rad"]),tuple(row["dq_rad_s"]),(0.,)*7,t,True,"offline_goal_only_not_safety")
            feedback.validate()
            pose=right_flange_pose(feedback.q_rad,t)
            result=monitor.update(feedback,t,pose)
            counts[result["status"]]=counts.get(result["status"],0)+1
            if first is None and result["status"]=="COARSE_REACHED":first=t
        reports.append({"episode":episode["episode"],"goal_id":goal.goal_id,
                        "samples":len(episode["samples"]),"first_coarse_reached_ns":first,
                        "status_counts":counts,"last_status":monitor.result["status"]})
    return {"scope":"recorded q/dq -> online FK -> GoalMonitor; no force or model/safety validation",
            "fixture_sha256":sha256(path),"episodes":reports,"commands_sent":False}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root",type=Path,required=True)
    parser.add_argument("--statistics-dir",type=Path,required=True)
    parser.add_argument("--model",type=Path,required=True)
    parser.add_argument("--goals-dir",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--prepare-configs",action="store_true")
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    if args.prepare_configs:
        manifest=prepare_configs(args.statistics_dir/"group_statistics.json",args.goals_dir)
        write_json(args.goals_dir.parent/"spherical_goals_manifest.json",manifest)
    dh=load_model(args.model)
    records=json.loads((args.statistics_dir/"episodes.json").read_text())
    original={r["episode"]:r for r in records}
    reports=[];comparison=[];validation=[]
    for path in sorted(args.data_root.glob("*/episode.hdf5")):
        e=read_episode(path)
        if e["numeric_sha256"]!=original[e["episode"]]["numeric_sha256"]:
            raise ValueError("recorded numeric input changed")
        cfg=ShadowConfig.read(args.goals_dir/f"{e['condition']}.json")
        reference=flange_fk(e["q"][[0,len(e["q"])//2,-1]],dh)
        for q,t in zip(e["q"][[0,len(e["q"])//2,-1]],reference):
            live=right_flange_pose(q,0)
            error=np.linalg.norm(np.asarray(live.position_m)*1000-t[:3,3])
            if error>1e-6:raise ValueError("live FK differs from audited offline FK")
            validation.append(float(error))
        reports.append(goal_only_replay(e,cfg.goal,dh))
        comparison.append(space_comparison(e,records,dh))
        first=reports[-1]["first_completion"]
        print(e["episode"],"completion before end",first["seconds_before_record_end"] if first else None,flush=True)
    write_json(args.output/"episodes.json",reports)
    write_json(args.output/"space_comparison.json",comparison)
    sample=np.asarray(records[0]["last_0.5s"]["q_median_rad"])
    start=time.perf_counter()
    for _ in range(2000):right_flange_pose(sample,0)
    fk_us=(time.perf_counter()-start)/2000*1e6
    summary={"scope":"goal-only full-trajectory replay; centers use all 50; not held-out success accuracy",
             "sphere_rule":"selected goal only; 3mm; 500ms; no orientation or joint velocity gate; reset on >20ms gaps",
             "episodes":len(reports),"completed":sum(r["first_completion"] is not None for r in reports),
             "completed_before_final_5s":sum(r["reached_before_final_5s"] for r in reports),
             "not_completed":[r["episode"] for r in reports if not r["first_completion"]],
             "space_comparison_scope":"last 5s; 500ms dwell; same-hole other-four reference; pure geometry; other-hole matches are ambiguity, not failure labels",
             "space_comparison":summarize_comparison(comparison),
             "runtime_fk_offline_max_error_mm":max(validation),"runtime_fk_checked_poses":len(validation),
             "local_fk_pose_mean_us":fk_us,"commands_sent":False,"motor_control_api_calls":0,
             "physical_success_verified":False,"script_sha256":sha256(__file__)}
    write_json(args.output/"summary.json",summary)
    print(json.dumps(summary,indent=2))


if __name__=="__main__":main()
