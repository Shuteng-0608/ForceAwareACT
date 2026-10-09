"""Coarse completion, FK geometry and runtime integration; no device access."""
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from force_aware_act.control.contracts import Feedback, Goal, Limits, Pose, ShadowConfig
from force_aware_act.control.kinematics import FRAME_ID, POSE_SOURCE, quaternion_xyzw, right_flange_pose, right_flange_transform
from force_aware_act.control.runtime import ShadowController
from force_aware_act.control.supervision import GoalMonitor

MS = 1_000_000


def feedback(ms, q=None, external=0.):
    return Feedback(ms*MS, ms+1, tuple(np.zeros(7) if q is None else q), (.1,)*7,
                    (external,)*7, ms*MS, True, "synthetic")


def sphere(**kwargs):
    return Goal(**{**dict(goal_id="selected", kind="sphere", frame_id=FRAME_ID,
                         target_position_m=(0.,0.,.8705), radius_m=.003, dwell_ms=30,
                         coarse_completion_accepted=True), **kwargs})


def pose(ms, distance=0., frame=FRAME_ID):
    return Pose(ms*MS, frame, (distance,0.,.8705), (0.,0.,0.,1.))


@pytest.mark.parametrize("kwargs", [dict(radius_m=-1),dict(radius_m=float("nan")),dict(radius_m=True),
                                   dict(target_position_m=(0,0)),dict(frame_id=""),
                                   dict(coarse_completion_accepted=1),
                                   dict(pose_source=POSE_SOURCE,frame_id="wrong")])
def test_bad_spheres_fail_configuration(kwargs):
    with pytest.raises(ValueError):
        sphere(**kwargs)


def test_sphere_has_explicit_coarse_acceptance_not_fake_physical_verification():
    monitor=GoalMonitor(sphere())
    for t in (0,10,20,30):
        result=monitor.update(feedback(t),t*MS,pose(t))
    assert result["status"]=="COARSE_REACHED" and not result["verified"]
    assert not result["slow"] and not result["joint_velocity_gate_applied"]
    monitor=GoalMonitor(sphere(coarse_completion_accepted=False,verified=True))
    for t in (0,10,20,30):
        result=monitor.update(feedback(t),t*MS,pose(t))
    assert result["status"]=="UNVERIFIED_CANDIDATE"


def test_radius_dwell_exit_and_sample_gap():
    monitor=GoalMonitor(sphere())
    for t in (0,10,20):
        assert monitor.update(feedback(t),t*MS,pose(t,.003))["status"]=="GOAL_CANDIDATE"
    assert monitor.update(feedback(30),30*MS,pose(30,.00301))["status"]=="RUNNING"
    for t in (40,50):
        monitor.update(feedback(t),t*MS,pose(t))
    assert monitor.update(feedback(80),80*MS,pose(80))["dwell_ms"]==0
    assert monitor.update(feedback(80),90*MS,pose(90))["dwell_ms"]==0
    assert monitor.update(feedback(79),90*MS,pose(90))["status"]=="UNCONFIRMED"


@pytest.mark.parametrize("p", [None,pose(10,frame="wrong"),pose(100),pose(0)])
def test_missing_stale_future_or_wrong_frame_pose_cannot_complete(p):
    monitor=GoalMonitor(sphere())
    monitor.update(feedback(0),0,pose(0))
    assert monitor.update(feedback(30),30*MS,p)["status"]=="UNCONFIRMED"
    assert monitor.inside_since is None


def test_fk_zero_measured_pose_units_and_source_timestamp():
    p=right_flange_pose(np.zeros(7),123)
    assert p.position_m==pytest.approx((0,0,.8705))
    assert p.frame_id==FRAME_ID and p.timestamp_ns==123
    assert abs(p.quaternion_xyzw[3])==pytest.approx(1.)
    with pytest.raises(ValueError):
        right_flange_pose([float("nan")]*7,123)


def test_runtime_fk_matches_independent_offline_chain_for_random_poses():
    from scripts.analyze_real_robot_endpoints import flange_fk
    dh=np.zeros((8,4))
    dh[0,2],dh[2,2],dh[4,2],dh[3,1],dh[7,2]=174.5,287,314,18,95
    q=np.random.default_rng(7).uniform(-2,2,(30,7))
    actual=np.asarray([right_flange_transform(x) for x in q])
    np.testing.assert_allclose(actual,flange_fk(q,dh),atol=1e-10)
    for t in actual:
        quat=quaternion_xyzw(t[:3,:3])
        np.testing.assert_allclose(Rotation.from_quat(quat).as_matrix(),t[:3,:3],atol=1e-10)


@pytest.mark.parametrize("axis", ["x","y","z"])
def test_quaternion_at_180_degrees(axis):
    r=Rotation.from_euler(axis,180,degrees=True).as_matrix()
    np.testing.assert_allclose(Rotation.from_quat(quaternion_xyzw(r)).as_matrix(),r,atol=1e-10)


def running_controller():
    c=ShadowController(ShadowConfig(goal=sphere(pose_source=POSE_SOURCE),
                       limits=Limits(torque_hard_nm=(4.,)*7)),0,"test")
    c.feedback(feedback(0),0)
    for cam in ("d455","d405"):c.camera(cam,0,0)
    c.prediction(np.ones((100,7)),query_id=0,start_ns=0,observed_ns=0,available_ns=0,now_ns=0,run_id="test")
    assert c.tick(0)["state"]=="RUNNING_SHADOW"
    return c


def test_runtime_finishes_only_shadow_and_fault_overrides_completion():
    c=running_controller()
    # External pose messages / policy predictions cannot replace actual FK.
    c.pose=pose(0,100)
    for t in (10,20,30):
        c.feedback(feedback(t),t*MS)
        row=c.tick(t*MS)
    assert row["state"]=="COMPLETED_SHADOW" and row["proposal"] is None
    assert c.stop_request["reason"]=="coarse_goal_reached"
    assert c.report()["motor_control_api_calls"]==0
    assert not c.executor.chunks
    timing=c.report()["compute_timing"]
    assert timing["fk"]["total_calls"]==4
    assert timing["fk"]["p99_ms"]>=0
    assert timing["goal_update"]["total_calls"]>=4
    c.feedback(feedback(40,external=5),40*MS)
    assert c.state=="ABORTED_SHADOW"
    c=running_controller()
    c.feedback(feedback(10),10*MS);c.tick(10*MS)
    c.feedback(feedback(20),20*MS);c.tick(20*MS)
    c.feedback(feedback(30,external=5),30*MS)
    assert c.tick(30*MS)["state"]=="ABORTED_SHADOW"


def test_between_tick_exit_resets_sphere_dwell():
    c=running_controller()
    c.feedback(feedback(10),10*MS);c.tick(10*MS)
    q=np.zeros(7);q[1]=.1
    c.feedback(feedback(15,q=q),15*MS)  # outside for a frame between ticks
    c.feedback(feedback(18),18*MS)
    assert c.tick(20*MS)["goal"]["dwell_ms"]==0
    c.feedback(feedback(30),30*MS)
    assert c.tick(30*MS)["state"]=="RUNNING_SHADOW"


def test_model_predictions_alone_cannot_trigger_goal():
    c=running_controller()
    q=np.zeros(7);q[1]=.1
    for t in (10,20,30,40):
        c.feedback(feedback(t,q=q),t*MS)
        assert c.tick(t*MS)["state"]=="RUNNING_SHADOW"


def test_replay_helpers_reset_gaps_and_summarize_target_ambiguity():
    from scripts.replay_spherical_goals import occupancy, summarize_comparison
    assert occupancy(np.arange(0.,1.,.01),np.ones(100,dtype=bool))["reached"]
    times=np.r_[np.arange(0.,.4,.01),np.arange(.5,.9,.01)]
    assert not occupancy(times,np.ones(len(times),dtype=bool))["reached"]
    def row(selected,match):
        return {"is_selected_goal":selected,"sphere":{f"{r}mm":{"reached":match} for r in (1,2,3,5)},
                "joint":{f"{r}deg":{"reached":match} for r in (1,2,3,5)}}
    result=summarize_comparison([{"goals":[row(True,True),row(False,True)]}])
    assert result["joint"]["5deg"]=={"own_goal_reached":1,"episodes_also_matching_other_hole":1}


def test_benchmark_reports_milliseconds_and_rejects_bad_samples():
    from scripts.benchmark_goal_monitor import timing_summary
    s=timing_summary([1_000_000,2_000_000,3_000_000])
    assert s["mean_ms"]==2 and s["max_ms"]==3 and s["samples"]==3
    with pytest.raises(ValueError):timing_summary([float("nan")])


def test_ten_selected_goal_configs_are_ready_only_for_coarse_readonly_completion():
    root=Path(__file__).resolve().parents[1]
    goals=[ShadowConfig.read(p) for p in sorted((root/"configs/deployment/goals").glob("x*.json"))]
    assert len(goals)==10 and len({c.goal.goal_id for c in goals})==10
    for c in goals:
        assert c.goal.kind=="sphere" and c.goal.coarse_completion_accepted and not c.goal.verified
        assert c.goal.pose_source==POSE_SOURCE and c.goal.radius_m==.003 and c.goal.dwell_ms==500
        assert not c.allow_motion and c.readiness()["hardware_command_backend"]=="absent"


def test_recorded_fixture_replays_online_fk_without_devices(tmp_path):
    import gzip
    from scripts.replay_spherical_goals import replay_sample_fixture
    (tmp_path/"selected.json").write_text(json.dumps(ShadowConfig(goal=sphere(pose_source=POSE_SOURCE)).to_dict()))
    payload={"schema_version":"recorded_goal_samples_v1","episodes":[{"episode":"synthetic","goal_id":"selected",
             "samples":[{"timestamp_ns":t*MS,"q_rad":[0.]*7,"dq_rad_s":[0.]*7} for t in (0,10,20,30)]}]}
    path=tmp_path/"sample.json.gz"
    with gzip.open(path,"wt") as handle:json.dump(payload,handle)
    result=replay_sample_fixture(path,tmp_path)
    assert result["episodes"][0]["first_coarse_reached_ns"]==30*MS
    assert not result["commands_sent"]
