#!/usr/bin/env python3
"""Audit numeric HDF5 streams and replay shadow decisions without a robot.

The all-episode action source is explicitly a causal held-demonstration baseline,
NOT model inference: only commands already recorded at the query time are used.
Real model chunks can separately be replayed from a live monitor's events.jsonl.
No HDF5 data or labels are modified. All estimated limits/goals are diagnostic.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
import hashlib
import heapq
import json
from pathlib import Path
import sys

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from force_aware_act.control.contracts import Feedback, Goal, Limits, PERIOD_NS, ShadowConfig
from force_aware_act.control.runtime import ShadowController
from force_aware_act.real_robot.sampling import estimate_action_grid_phase_ns


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def sha256(path):
    with Path(path).open("rb") as f:
        digest=hashlib.sha256()
        for block in iter(lambda:f.read(1024*1024),b""):digest.update(block)
    return digest.hexdigest()


def read_numeric(path):
    with h5py.File(path, "r") as f:
        if f.attrs.get("schema_version") != "real_robot_right_arm_raw_v2" or f.attrs.get("arm") != "right":
            raise ValueError("right arm raw_v2 HDF5 required")
        force = f["observations/joint_external_torque"]
        if force.attrs.get("source_topic") != "/tianji_marvin_teleop_bridge/estimated_joint_disturbance":
            raise ValueError("external torque source mismatch")
        names = [x.decode() if isinstance(x, bytes) else str(x) for x in f["metadata/joint_names"][...]]
        if names != [f"Joint{i}_R" for i in range(1,8)]:
            raise ValueError("joint order mismatch")
        times={name:np.asarray(f[f"timestamps/{name}/log_ns"],dtype=np.int64)
               for name in ("state","disturbance","action","d455","d405")}
        for name, t in times.items():
            if len(t)<2 or np.any(np.diff(t)<=0):raise ValueError(f"bad {name} timestamps")
        data={key:np.asarray(f[path_],dtype=np.float64) for key,path_ in {
              "q":"observations/joint_pos","dq":"observations/joint_vel",
              "tau":"observations/joint_external_torque","internal":"observations/joint_internal_torque",
              "action":"action"}.items()}
        for key,values in data.items():
            stream="disturbance" if key=="tau" else "action" if key=="action" else "state"
            if values.shape!=(len(times[stream]),7) or not np.isfinite(values).all():
                raise ValueError(f"invalid numeric stream {key}")
        return {"id":path.parent.name,"path":str(path),"times":times,**data,
                "source_sha256":str(f.attrs["source_sha256"]),
                "calibration_validity":"not_recorded_in_historical_dataset"}


def numeric_stats(episodes):
    result={}
    for name in ("q","dq","tau","internal","action"):
        arr=np.concatenate([e[name] for e in episodes])
        result[name]={"count":len(arr),"min":arr.min(0).tolist(),"max":arr.max(0).tolist(),
                      "abs_p99":np.percentile(np.abs(arr),99,axis=0).tolist(),
                      "abs_p99_9":np.percentile(np.abs(arr),99.9,axis=0).tolist(),
                      "abs_max":np.max(np.abs(arr),axis=0).tolist()}
    return result


def diagnostic_limits(training):
    """Transparent candidate formula; NEVER approved for physical deployment."""
    s=numeric_stats(training)
    soft=np.maximum(s["tau"]["abs_p99"],.01)
    hard=np.maximum(np.asarray(s["tau"]["abs_max"])*1.10,soft+.01)
    low=np.minimum(s["q"]["min"],s["action"]["min"])-.05
    high=np.maximum(s["q"]["max"],s["action"]["max"])+.05
    velocity=np.maximum(np.asarray(s["dq"]["abs_max"])*1.20,.01)
    accel=[]; following=[]; first=[]
    for e in training:
        dt=np.diff(e["times"]["state"])/1e9
        accel.append(np.max(np.abs(np.diff(e["dq"],axis=0)/dt[:,None]),axis=0))
        idx=np.searchsorted(e["times"]["state"],e["times"]["action"],side="right")-1
        valid=idx>=0
        following.append(np.max(np.abs(e["action"][valid]-e["q"][idx[valid]]),axis=0))
        first.append(np.abs(e["action"][np.flatnonzero(valid)[0]]-e["q"][idx[valid][0]]))
    return Limits(torque_hard_nm=tuple(hard),torque_soft_nm=tuple(soft),torque_release_nm=tuple(soft*.8),
                  torque_soft_duration_ms=50.,joint_min_rad=tuple(low),joint_max_rad=tuple(high),
                  velocity_rad_s=tuple(velocity),acceleration_rad_s2=tuple(np.maximum(np.max(accel,axis=0)*1.20,.1)),
                  following_error_rad=tuple(np.maximum(np.max(following,axis=0)*1.20,.01)),
                  first_action_error_rad=tuple(np.maximum(np.max(following,axis=0)*1.20,.01)),
                  authority="dataset_diagnostic")


def goal_candidates(training, groups):
    result={}
    for group in sorted({groups[e["id"]] for e in training}):
        selected=[e for e in training if groups[e["id"]]==group]
        endpoints=[]; records=[]
        for e in selected:
            t=e["times"]["state"];mask=t>=t[-1]-500_000_000
            endpoints.append(np.median(e["q"][mask],axis=0))
            records.append({"episode":e["id"],"window":"last 0.5 s, task completion NOT verified",
                            "max_velocity_rad_s":float(np.max(np.abs(e["dq"][mask]))),
                            "position_range_rad":np.ptp(e["q"][mask],axis=0).tolist()})
        med=np.median(endpoints,axis=0)
        tol=np.maximum(np.max(np.abs(np.asarray(endpoints)-med),axis=0)*1.20,.005)
        goal=Goal(goal_id=group,kind="joint",verified=False,target_q_rad=tuple(med),tolerance_q_rad=tuple(tol),
                  provenance="training-only endpoint proxy; may contain withdrawal; requires physical calibration")
        result[group]={"goal":goal.__dict__,"evidence":records,"physical_target_approved":False}
    return result


def _events(times, kind, rank):
    for i,t in enumerate(times):yield (int(t),rank,kind,i)


def replay_episode(episode, config):
    times=episode["times"]
    phase=estimate_action_grid_phase_ns(times["action"],PERIOD_NS,2_000_000_000)
    first=int(times["action"][0])+int(phase["phase_ns"])
    earliest=max(int(t[0]) for t in times.values())+250_000_000
    origin=first+max(0,(earliest-first+PERIOD_NS-1)//PERIOD_NS)*PERIOD_NS
    end=min(int(t[-1]) for t in times.values())
    c=ShadowController(replace(config,mode="replay"),origin,episode["id"])
    ticks=np.arange(origin,end+1,PERIOD_NS,dtype=np.int64)
    queries=ticks[::2]
    # Synthetic 12 ms availability exercises delayed chunk indexing; this is
    # explicitly not a claim about model latency in these recorded episodes.
    completions=queries+12_000_000
    streams=[_events(times["state"],"feedback",0),_events(times["d455"],"d455",1),
             _events(times["d405"],"d405",1),_events(completions,"prediction",2),_events(ticks,"tick",3)]
    first_stop=None; proposal_steps=[]; weighted_ages=[]; previous=None; goal_status=Counter(); observations=0
    for t,_,kind,i in heapq.merge(*streams):
        if t<origin-20_000_000 or t>end:continue
        if kind=="feedback":
            force_i=np.searchsorted(times["disturbance"],t,side="right")-1
            if force_i<0:continue
            # SDK frame id/context/raw bias were not recorded. The indices below
            # are a replay sequence, not reconstructed physical controller frames.
            value=Feedback(t,i%1_000_000,tuple(episode["q"][i]),tuple(episode["dq"][i]),
                           tuple(episode["tau"][force_i]),int(times["disturbance"][force_i]),True,
                           "offline_assumed_calibrated_recorded_disturbance",internal_nm=tuple(episode["internal"][i]))
            c.feedback(value,t);observations+=1
        elif kind in ("d455","d405"):
            c.camera(kind,t,t)
        elif kind=="prediction":
            query=int(queries[i]);ai=np.searchsorted(times["action"],query,side="right")-1
            if ai>=0:
                assert int(times["action"][ai])<=query
                c.prediction(np.repeat(episode["action"][ai][None,:],100,axis=0),query_id=i,
                             start_ns=query,observed_ns=query,available_ns=t,now_ns=t,run_id=episode["id"])
        else:
            row=c.tick(t)
            if row.get("goal"):goal_status[row["goal"]["status"]]+=1
            if row["proposal"]:
                p=np.asarray(row["proposal"]["action_rad"])
                if previous is not None:proposal_steps.append(float(np.max(np.abs(p-previous))))
                previous=p;weighted_ages.append(row["proposal"]["weighted_age_ms"])
        if c.stop_request and first_stop is None:first_stop=dict(c.stop_request)
    report=c.report()
    report.update(episode=episode["id"],source_sha256=episode["source_sha256"],duration_s=(end-origin)/1e9,
                  observed_state_messages=observations,first_stop=first_stop,goal_status_counts=dict(goal_status),
                  action_source="causal_hold_of_last_recorded_command_NOT_model_predictions",
                  simulated_inference_delay_ms=12.,grid_phase=phase,
                  calibration_assumption=episode["calibration_validity"],
                  max_proposed_step_rad=max(proposal_steps) if proposal_steps else None,
                  weighted_prediction_age_ms_mean=float(np.mean(weighted_ages)) if weighted_ages else None)
    return report


def audit_dataset(data_root, split_path, output):
    output=Path(output)
    if output.exists():raise FileExistsError("use a fresh report directory")
    output.mkdir(parents=True)
    split=json.loads(Path(split_path).read_text());groups=split["condition_groups"]
    episodes=[read_numeric(p) for p in sorted(Path(data_root).rglob("episode.hdf5"))]
    ids={e["id"] for e in episodes}
    split_ids=set(split["train_episode_ids"]+split["validation_episode_ids"]+split["test_episode_ids"])
    if ids!=split_ids or ids!=set(groups):raise ValueError("dataset and split/group membership mismatch")
    training=[e for e in episodes if e["id"] in set(split["train_episode_ids"])]
    lim=diagnostic_limits(training);goals=goal_candidates(training,groups)
    write_json(output/"numeric_statistics.json",{"all":numeric_stats(episodes),"training_only":numeric_stats(training)})
    write_json(output/"diagnostic_limits.json",{"limits":lim.__dict__,"approved_for_motion":False,
        "source_episode_ids":[e["id"] for e in training],
        "formula":"soft=max(train abs P99,0.01 Nm); release=0.8*soft; hard=max(1.1*train abs max,soft+0.01 Nm); joint bounds=train q/action extrema +/-0.05 rad; velocity/acceleration/following envelopes=1.2*train extrema with documented numerical floors",
        "purpose":"software replay sensitivity only; successful demonstrations are not physical safety limits"})
    write_json(output/"goal_candidates.json",goals)
    for name in goals:
        write_json(output/"goals"/(name+".json"),ShadowConfig(goal=Goal(goal_id=name,provenance="different hole; actual TCP target pending calibration")).to_dict())
    reports=[]
    for e in episodes:
        cfg=ShadowConfig(limits=lim,goal=Goal(**goals[groups[e["id"]]]["goal"]))
        result=replay_episode(e,cfg);reports.append(result)
        write_json(output/"episodes"/(e["id"]+".json"),result)
        print(json.dumps({"episode":e["id"],"state":result["state"],"first_stop":result["first_stop"]}),flush=True)
    reasons=Counter(r["first_stop"]["reason"] if r["first_stop"] else "no_stop" for r in reports)
    summary={"schema_version":"controller_dataset_replay_v1","episodes":len(episodes),"holes":len(goals),
             "training_episodes_for_candidates":len(training),"split_sha256":sha256(split_path),
             "data_root":str(Path(data_root).resolve()),"commands_sent":False,"robot_connection_opened":False,
             "source":"recorded corrected external torque; raw/bias/SDK context not available",
             "action_source":"causal held recorded command baseline; no model execution in this dataset audit",
             "first_stop_counts":dict(reasons),"physical_safety_validated":False,"physical_goal_validated":False,
             "goal_layout":"10 distinct holes, operator confirmed 2026-10-07",
             "states":dict(Counter(r["state"] for r in reports)),
             "total_state_messages":sum(r["observed_state_messages"] for r in reports),
             "episodes_index":[{"episode":r["episode"],"source_sha256":r["source_sha256"],"goal_id":groups[r["episode"]]} for r in reports]}
    write_json(output/"report.json",summary)
    return summary


def replay_journal(journal, config, output):
    """Reapply all three aggregators to real model chunks, preserving arrival times."""
    events=[]
    with Path(journal).open() as f:
        for line in f:
            event=json.loads(line)
            if event["kind"]=="prediction":events.append(event)
    if not events:raise ValueError("journal contains no model predictions")
    # The original origin can be reconstructed from the first aligned query:
    # it changes only the integer grid label, not any absolute execution time.
    origin=events[0]["metadata"]["start_ns"]
    run_id=events[0]["metadata"]["run_id"]
    from force_aware_act.control.execution import TemporalExecutor,ActionUnavailable
    results={}
    for strategy in ("newest","official","recency"):
        ex=TemporalExecutor(origin,run_id,replace(config,aggregation=strategy));cursor=0;rows=[];missing=0;last=None;steps=[]
        end=events[-1]["metadata"]["start_ns"]+20_000_000
        for tick in range(origin,end+1,PERIOD_NS):
            while cursor<len(events) and events[cursor]["monitor_received_ns"]<=tick:
                event=events[cursor]
                # Replaying receiver availability must not backdate a delayed IPC delivery.
                meta={**event["metadata"],"available_ns":max(event["metadata"]["available_ns"],event["monitor_received_ns"])}
                ex.submit(event["actions"],now_ns=tick,**meta);cursor+=1
            try:row=ex.sample(tick)
            except ActionUnavailable:missing+=1;continue
            value=np.asarray(row["action_rad"])
            if last is not None:steps.append(float(np.max(np.abs(value-last))))
            last=value;rows.append(row)
        results[strategy]={"ticks":len(rows),"unavailable_ticks":missing,
                           "max_step_rad":max(steps) if steps else None,
                           "p99_step_rad":float(np.percentile(steps,99)) if steps else None,
                           "mean_prediction_age_ms":float(np.mean([r["weighted_age_ms"] for r in rows])) if rows else None}
    result={"source_journal":str(journal),"source_sha256":sha256(journal),"real_model_predictions":len(events),
            "comparison":results,"commands_sent":False,"purpose":"open-loop proposal comparison; smoothness is not task success"}
    if (Path(journal).parent/"execution.jsonl").exists() and (Path(journal).parent/"report.json").exists():
        result["full_controller_replay"]=replay_control_events(journal,config)
    write_json(Path(output),result)
    return result


def replay_control_events(journal,config):
    """Reproduce safety, goals and proposals using the recorded receive/tick times.

    Original execution rows are comparison evidence, never replay inputs beyond
    their clock schedule. Actual feedback, cameras and model chunks come from
    the event journal. This does not simulate physical robot dynamics.
    """
    directory=Path(journal).parent
    original=json.loads((directory/"report.json").read_text())
    def entries(path,kind):
        with path.open() as f:
            for line in f:
                row=json.loads(line)
                yield (row["monitor_received_ns"] if kind=="event" else row["evaluated_ns"],
                       0 if kind=="event" else 1,kind,row)
    with (directory/"execution.jsonl").open() as f:first_tick=json.loads(next(f))["tick_ns"]
    origin=original.get("origin_ns",first_tick)
    c=ShadowController(config,origin,original["run_id"])
    counts=Counter();examples=[];previous_tick=None;last_time=origin
    for now,_,kind,row in heapq.merge(entries(Path(journal),"event"),entries(directory/"execution.jsonl","tick")):
        last_time=now
        if kind=="event":
            event_kind=row["kind"]
            if event_kind=="feedback":c.feedback(Feedback(**row["value"]),now)
            elif event_kind=="camera":c.camera(row["name"],row["received_ns"],now)
            elif event_kind=="prediction":c.prediction(row["actions"],now_ns=now,**row["metadata"])
            elif event_kind=="fault":c.stop(row["reason"],now)
            elif event_kind=="pose":
                from force_aware_act.control.contracts import Pose
                c.pose=Pose(**row["value"])
            elif event_kind!="end":raise ValueError("unknown recorded event")
        else:
            counts["ticks"]+=1
            if previous_tick is not None and now-previous_tick-PERIOD_NS>config.execution_lateness_ms*1e6 and c.state=="RUNNING_SHADOW":
                c.stop("execution_deadline",now)
            actual=c.tick(row["tick_ns"],now);previous_tick=row["tick_ns"]
            mismatch=[]
            if actual["state"]!=row["state"]:mismatch.append("state")
            for key in ("proposal","constrained_proposal","goal"):
                if actual.get(key)!=row.get(key):mismatch.append(key)
            if mismatch:
                counts["mismatched_ticks"]+=1
                if len(examples)<5:examples.append({"tick_ns":row["tick_ns"],"fields":mismatch})
    if c.stop_request is None:c.stop("normal_readonly_end",last_time,outcome="ENDED_SHADOW")
    state_matches=c.state==original["state"]
    return {"ticks":counts["ticks"],"mismatched_ticks":counts["mismatched_ticks"],
            # JSON stores validated tuple-valued joint/goal fields as lists.
            "examples":examples,"same_config":config==ShadowConfig.from_dict(original["config"]),
            "final_state_matches":state_matches,"replayed_state":c.state,
            "replayed_stop":c.stop_request,"commands_sent":False,
            "exact_decisions_reproduced":counts["mismatched_ticks"]==0 and state_matches}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="mode",required=True)
    ds=sub.add_parser("dataset");ds.add_argument("--data-root",type=Path,required=True);ds.add_argument("--split",type=Path,required=True);ds.add_argument("--output",type=Path,required=True)
    journal=sub.add_parser("journal");journal.add_argument("--events",type=Path,required=True);journal.add_argument("--config",type=Path,required=True);journal.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    result=(audit_dataset(args.data_root,args.split,args.output) if args.mode=="dataset" else
            replay_journal(args.events,ShadowConfig.read(args.config),args.output))
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=="__main__":main()
