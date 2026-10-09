"""Recorded-stream and real-prediction journal replay, without hardware."""
import importlib.util
import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from force_aware_act.control.contracts import Goal, Limits, ShadowConfig

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("controller_replay",ROOT/"scripts/deployment/replay_controller_readonly.py")
replay=importlib.util.module_from_spec(spec);spec.loader.exec_module(replay)


def episode(name="a",offset=0.):
    times={key:np.arange(0,1_000_000_001,10_000_000,dtype=np.int64)
           for key in ("state","action","disturbance","d455","d405")}
    z=np.zeros((101,7))
    return dict(id=name,path=name,times=times,q=z+offset,dq=z.copy(),tau=z.copy(),internal=z.copy(),
                action=z+offset,source_sha256="test",calibration_validity="not_recorded")


def complete_limits():
    return Limits(torque_hard_nm=(4.,)*7,torque_soft_nm=(2.,)*7,torque_release_nm=(1.,)*7,
                  joint_min_rad=(-2.,)*7,joint_max_rad=(2.,)*7,velocity_rad_s=(1.,)*7,
                  acceleration_rad_s2=(2.,)*7,following_error_rad=(.3,)*7,first_action_error_rad=(.2,)*7,
                  authority="synthetic_test")


def test_replay_uses_only_commands_available_at_query_time():
    e=episode();e["action"][60:]=1.
    result=replay.replay_episode(e,ShadowConfig(limits=complete_limits()))
    assert result["first_stop"]["reason"]=="target_feedback_error"
    assert result["first_stop"]["requested_ns"]>=600_000_000
    assert "NOT_model" in result["action_source"]
    assert not result["physical_stop_confirmed"]


def test_future_torque_spike_does_not_leak_backwards():
    e=episode();e["tau"][70,0]=8.
    result=replay.replay_episode(e,ShadowConfig(limits=complete_limits()))
    assert result["first_stop"]["reason"]=="external_torque_hard"
    assert result["first_stop"]["requested_ns"]==700_000_000


def test_separate_hole_candidates_are_unverified_and_diagnostic():
    records=[episode("a",.1),episode("b",.8)]
    goals=replay.goal_candidates(records,{"a":"holeA","b":"holeB"})
    assert set(goals)=={"holeA","holeB"}
    assert goals["holeA"]["goal"]["target_q_rad"]==pytest.approx([.1]*7)
    assert goals["holeB"]["goal"]["target_q_rad"]==pytest.approx([.8]*7)
    assert all(not value["goal"]["verified"] for value in goals.values())
    lim=replay.diagnostic_limits(records)
    assert lim.authority=="dataset_diagnostic"
    assert not ShadowConfig(limits=lim,goal=Goal(**goals["holeA"]["goal"])).readiness()["motion_possible"]


def test_hdf5_numeric_source_checks(tmp_path):
    path=tmp_path/"episode.hdf5";e=episode()
    with h5py.File(path,"w") as f:
        f.attrs.update(schema_version="real_robot_right_arm_raw_v2",arm="right",source_sha256="test")
        f.create_dataset("metadata/joint_names",data=np.array([f"Joint{i}_R" for i in range(1,8)],dtype=h5py.string_dtype()))
        for name,t in e["times"].items():f.create_dataset(f"timestamps/{name}/log_ns",data=t)
        for key,name in {"q":"observations/joint_pos","dq":"observations/joint_vel","tau":"observations/joint_external_torque",
                         "internal":"observations/joint_internal_torque","action":"action"}.items():f.create_dataset(name,data=e[key])
        f["observations/joint_external_torque"].attrs["source_topic"]="/tianji_marvin_teleop_bridge/estimated_joint_disturbance"
    assert replay.read_numeric(path)["tau"].shape==(101,7)
    with h5py.File(path,"r+") as f:f["observations/joint_external_torque"].attrs["source_topic"]="joint_states.effort"
    with pytest.raises(ValueError,match="source"):replay.read_numeric(path)


def test_real_journal_replay_respects_ipc_availability(tmp_path):
    path=tmp_path/"events.jsonl"
    events=[]
    for i,start in enumerate((0,20_000_000,40_000_000)):
        events.append({"kind":"prediction","monitor_received_ns":start+15_000_000,
                       "actions":np.full((100,7),i).tolist(),"metadata":dict(query_id=i,start_ns=start,
                       observed_ns=start,available_ns=start+12_000_000,run_id="test")})
    path.write_text("".join(json.dumps(row)+"\n" for row in events))
    report=replay.replay_journal(path,ShadowConfig(),tmp_path/"comparison.json")
    assert report["real_model_predictions"]==3
    assert report["comparison"]["newest"]["unavailable_ticks"]==2
    assert report["comparison"]["newest"]["max_step_rad"]==1
    assert not report["commands_sent"]


@pytest.mark.parametrize("with_sphere", [False, True])
def test_full_event_replay_reproduces_state_and_all_proposals(tmp_path, with_sphere):
    from force_aware_act.control.contracts import Feedback
    from force_aware_act.control.runtime import ShadowController
    goal=(Goal(goal_id="test",kind="sphere",frame_id="marvin_m6_ccs40_right_base",
               target_position_m=(.5,-.3,.25),radius_m=.003,
               coarse_completion_accepted=True,pose_source="m6_ccs40_right_flange_fk")
          if with_sphere else Goal())
    config=ShadowConfig(goal=goal);controller=ShadowController(config,0,"r")
    events=[];ticks=[]
    for t in (0,10_000_000,20_000_000):
        f=Feedback(t,t//1_000_000,(0.,)*7,(0.,)*7,(0.,)*7,t,True,"test")
        events.append(dict(kind="feedback",monitor_received_ns=t,value=f.__dict__))
        controller.feedback(f,t)
        if t==0:
            for camera in ("d455","d405"):
                controller.camera(camera,t,t)
                events.append(dict(kind="camera",monitor_received_ns=t,name=camera,received_ns=t))
            meta=dict(query_id=1,start_ns=0,observed_ns=0,available_ns=0,run_id="r")
            controller.prediction(np.zeros((100,7)),now_ns=0,**meta)
            events.append(dict(kind="prediction",monitor_received_ns=t,actions=np.zeros((100,7)).tolist(),metadata=meta))
        ticks.append(controller.tick(t))
    controller.stop("normal_readonly_end",20_000_000,outcome="ENDED_SHADOW")
    (tmp_path/"report.json").write_text(json.dumps(controller.report()))
    (tmp_path/"events.jsonl").write_text("".join(json.dumps(e)+"\n" for e in events))
    (tmp_path/"execution.jsonl").write_text("".join(json.dumps(e)+"\n" for e in ticks))
    result=replay.replay_control_events(tmp_path/"events.jsonl",config)
    assert result["exact_decisions_reproduced"] and result["ticks"]==3
    assert result["same_config"]
    from dataclasses import replace
    changed=replay.replay_control_events(tmp_path/"events.jsonl",replace(config,task_timeout_s=181.))
    assert not changed["same_config"]
