#!/usr/bin/env python3
"""Hardware-free FK/goal timing. No camera, robot SDK, model or GPU calls."""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from force_aware_act.control.contracts import Feedback, ShadowConfig
from force_aware_act.control.kinematics import right_flange_pose
from force_aware_act.control.supervision import GoalMonitor


def timing_summary(ns):
    ns=np.asarray(ns,dtype=float)
    if ns.ndim!=1 or not len(ns) or not np.isfinite(ns).all() or np.any(ns<0):
        raise ValueError("finite nonnegative timing samples required")
    return {"samples":len(ns),"mean_ms":float(ns.mean()/1e6),
            "p50_ms":float(np.percentile(ns,50)/1e6),"p99_ms":float(np.percentile(ns,99)/1e6),
            "p99_9_ms":float(np.percentile(ns,99.9)/1e6),"max_ms":float(ns.max()/1e6)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,required=True)
    parser.add_argument("--episodes",type=Path,required=True)
    parser.add_argument("--iterations",type=int,default=20000)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if args.iterations<1000:parser.error("use at least 1000 iterations")
    records=json.loads(args.episodes.read_text())
    joint_samples=np.asarray([r["last_0.5s"]["q_median_rad"] for r in records])
    cfg=ShadowConfig.read(args.config)
    # Continue measuring after candidates; do not shortcut with a completion latch.
    goal=replace(cfg.goal,coarse_completion_accepted=False)
    zero=(0.,)*7
    for i in range(500):right_flange_pose(joint_samples[i%len(joint_samples)],i)
    fk=[]
    for i in range(args.iterations):
        q=joint_samples[i%len(joint_samples)]
        start=time.perf_counter_ns();right_flange_pose(q,i);fk.append(time.perf_counter_ns()-start)
    monitor=GoalMonitor(goal);combined=[]
    for i in range(args.iterations+500):
        q=tuple(joint_samples[i%len(joint_samples)])
        t=i*3_333_333
        f=Feedback(t,i+1,q,zero,zero,t,True,"pure_compute_benchmark")
        start=time.perf_counter_ns()
        p=right_flange_pose(q,t)
        monitor.update(f,t,p)
        elapsed=time.perf_counter_ns()-start
        if i>=500:combined.append(elapsed)
    cpu=next((line.split(":",1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
              if line.startswith("model name")),"unknown")
    result={"host":platform.node(),"cpu":cpu,"python":platform.python_version(),"numpy":np.__version__,
            "scope":"local warmed pure computation; 50 recorded joint configurations; this program launches no model inference or hardware I/O; system load not controlled; not a hard-real-time bound",
            "fk_including_pose_and_quaternion_validation":timing_summary(fk),
            "fk_plus_goal_update":timing_summary(combined),
            "estimated_one_core_compute_percent_at_hz":{str(hz):float(np.mean(combined)/1e9*hz*100) for hz in (50,100,300)},
            "commands_sent":False,"gpu_calls":0,
            "source_sha256":{str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest() for path in
                              [Path(__file__).resolve(),ROOT/"src/force_aware_act/control/kinematics.py",
                               ROOT/"src/force_aware_act/control/supervision.py"]}}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))


if __name__=="__main__":main()
