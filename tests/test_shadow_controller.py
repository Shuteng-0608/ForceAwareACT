"""Virtual-clock fault injections. These tests never connect to a device."""
from dataclasses import replace
import json
from pathlib import Path
import time

import numpy as np
import pytest

from force_aware_act.control.contracts import Feedback, Goal, Limits, PERIOD_NS, Pose, ShadowConfig
from force_aware_act.control.execution import ActionLimiter, ActionUnavailable, TemporalExecutor
from force_aware_act.control.runtime import ShadowController, ShadowProcess
from force_aware_act.control.supervision import GoalMonitor, SafetySupervisor

MS = 1_000_000


def limits(**kw):
    values = dict(torque_hard_nm=(4.,)*7, torque_soft_nm=(2.,)*7, torque_release_nm=(1.,)*7,
                  torque_soft_duration_ms=30., joint_min_rad=(-2.,)*7, joint_max_rad=(2.,)*7,
                  velocity_rad_s=(1.,)*7, acceleration_rad_s2=(2.,)*7,
                  following_error_rad=(.3,)*7, first_action_error_rad=(.2,)*7,
                  max_consecutive_limited_ms=200., authority="synthetic_test")
    return Limits(**{**values, **kw})


def fb(t=0, frame=None, **kw):
    return Feedback(**{**dict(timestamp_ns=t, frame_serial=t//MS+1 if frame is None else frame,
                             q_rad=(0.,)*7, dq_rad_s=(0.,)*7, external_nm=(0.,)*7,
                             force_timestamp_ns=t, bias_valid=True, context="right:mode0:toolA",
                             raw_external_nm=(0.,)*7, internal_nm=(0.,)*7), **kw})


def joint_goal(**kw):
    return Goal(**{**dict(goal_id="hole1", kind="joint", verified=True, target_q_rad=(0.,)*7,
                        tolerance_q_rad=(.01,)*7, dwell_ms=30.), **kw})


def controller(config=None):
    cfg = config or ShadowConfig(limits=limits())
    c = ShadowController(cfg, 0, "run1")
    c.feedback(fb(), 0)
    for name in ("d455", "d405"):
        c.camera(name, 0, 0)
    c.prediction(np.zeros((100, 7)), query_id=1, start_ns=0, observed_ns=0,
                 available_ns=0, now_ns=0, run_id="run1")
    return c


@pytest.mark.parametrize("payload", [dict(allow_motion=True), dict(allow_motion=0), dict(mode="execute"),
                                      dict(query_hz=75), dict(execute_hz=50), dict(decay_ms=float("nan")),
                                      dict(aggregation="unknown"), dict(schema_version="bad")])
def test_config_cannot_enable_motion_or_accept_bad_contract(payload):
    with pytest.raises(ValueError):
        ShadowConfig(**payload)


def test_configuration_roundtrip_and_unknown_fields():
    config = ShadowConfig(limits=limits(), goal=joint_goal())
    assert ShadowConfig.from_dict(json.loads(json.dumps(config.to_dict()))) == config
    assert not config.readiness()["motion_possible"]
    assert ShadowConfig().readiness()["missing_limits"]
    with pytest.raises(TypeError):
        ShadowConfig.from_dict({"typo": 1})


@pytest.mark.parametrize("kw", [dict(torque_hard_nm=(0.,)*7), dict(torque_release_nm=(3.,)*7),
                               dict(joint_min_rad=(3.,)*7), dict(velocity_rad_s=(1.,)*6)])
def test_invalid_limits_rejected(kw):
    with pytest.raises(ValueError):
        limits(**kw)


def test_late_chunk_indices_and_expiry():
    ex = TemporalExecutor(0, "r", ShadowConfig())
    actions = np.arange(100)[:, None] * np.ones((1, 7))
    ex.submit(actions, query_id=1, start_ns=0, observed_ns=0, available_ns=13*MS, now_ns=13*MS, run_id="r")
    with pytest.raises(ActionUnavailable):
        ex.sample(10*MS)
    selected = ex.sample(20*MS)
    assert selected["chunk_indices"] == [2]
    assert selected["action_rad"] == [2.]*7
    with pytest.raises(ActionUnavailable):
        ex.sample(210*MS)


@pytest.mark.parametrize("mode,expected", [("official", "old"), ("recency", "new"), ("newest", "newest")])
def test_temporal_weights_and_completion_order(mode, expected):
    ex = TemporalExecutor(0, "r", ShadowConfig(aggregation=mode))
    # New observation completes first; old completion must not become 'newest'.
    ex.submit(np.ones((100,7))*2, query_id=2, start_ns=20*MS, observed_ns=20*MS, available_ns=30*MS, now_ns=30*MS, run_id="r")
    ex.submit(np.zeros((100,7)), query_id=1, start_ns=0, observed_ns=0, available_ns=35*MS, now_ns=35*MS, run_id="r")
    result=ex.sample(40*MS)
    assert result["chunk_indices"] == [4,2]
    assert result["query_ids"] == [1,2]
    assert sum(result["weights"]) == pytest.approx(1)
    if expected == "old":
        assert result["action_rad"][0] < 1
    elif expected == "new":
        assert result["action_rad"][0] > 1
    else:
        assert result["action_rad"] == [2.]*7


def test_no_future_availability_duplicates_or_wrong_run():
    ex=TemporalExecutor(0,"r",ShadowConfig())
    meta=dict(query_id=1,start_ns=0,observed_ns=0,available_ns=10*MS,now_ns=10*MS,run_id="r")
    ex.submit(np.zeros((100,7)),**meta)
    with pytest.raises(ActionUnavailable): ex.sample(0)
    with pytest.raises(ValueError,match="duplicate"): ex.submit(np.zeros((100,7)),**meta)
    with pytest.raises(ValueError,match="another run"): ex.submit(np.zeros((100,7)),**{**meta,"query_id":2,"run_id":"old"})
    with pytest.raises(ValueError,match="timestamps"): ex.submit(np.zeros((100,7)),**{**meta,"query_id":2,"available_ns":20*MS})
    assert ex.sample(10*MS)["action_rad"] == [0.]*7  # zero is a valid angle
    with pytest.raises(ValueError): ex.sample(10*MS)
    with pytest.raises(ValueError): ex.sample(15*MS)
    ex.close()
    with pytest.raises(ValueError,match="closed"): ex.submit(np.zeros((100,7)),**meta)


@pytest.mark.parametrize("bad", [np.zeros((99,7)), np.zeros((100,6)), np.full((100,7), np.nan)])
def test_prediction_shape_and_finiteness(bad):
    ex=TemporalExecutor(0,"r",ShadowConfig())
    with pytest.raises(ValueError): ex.submit(bad,query_id=1,start_ns=0,observed_ns=0,available_ns=0,now_ns=0,run_id="r")


def test_limiter_acceleration_velocity_and_joint_bounds():
    limiter=ActionLimiter(limits())
    first=limiter.apply([.1]*7,fb(),0)
    assert first["action_rad"] == pytest.approx([.0002]*7)
    assert first["velocity_rad_s"] == pytest.approx([.02]*7)
    second=limiter.apply([.1]*7,fb(10*MS),10*MS)
    assert second["action_rad"] == pytest.approx([.0006]*7)
    assert np.max(np.abs(np.asarray(second["velocity_rad_s"])-first["velocity_rad_s"])/.01) <= 2.0000001
    assert ActionLimiter(limits()).apply([3]*7,fb(),0)["fault"] == "target_joint_limit"
    assert ActionLimiter(limits()).apply([.25]*7,fb(),0)["fault"] == "target_feedback_error"
    assert ActionLimiter(Limits()).apply([.1]*7,fb(),0)["action_rad"] is None


def test_limiter_rejects_sustained_clipping_and_infeasible_braking():
    limiter=ActionLimiter(limits(max_consecutive_limited_ms=20))
    for t in (0,10*MS): assert limiter.apply([.1]*7,fb(t),t)["fault"] is None
    assert limiter.apply([.1]*7,fb(20*MS),20*MS)["fault"] == "sustained_action_limiting"
    edge=fb(q_rad=(1.999,)*7,dq_rad_s=(1.,)*7)
    assert ActionLimiter(limits()).apply([1.999]*7,edge,0)["fault"] == "infeasible_motion_constraints"


@pytest.mark.parametrize("replacement,reason", [
    (dict(external_nm=(4.,0,0,0,0,0,0)),"external_torque_hard"),
    (dict(q_rad=(float("nan"),)*7),"invalid_feedback"),
    (dict(bias_valid=False),"calibration_invalid"),
    (dict(error_code=99),"controller_error"),
    (dict(q_rad=(3.,)*7),"measured_joint_limit"),
    (dict(dq_rad_s=(2.,)*7),"measured_velocity_limit"),
])
def test_fast_safety_faults_are_latched(replacement,reason):
    s=SafetySupervisor(ShadowConfig(limits=limits()))
    s.ingest(fb(**replacement),0)
    assert s.fault["reason"] == reason
    s.ingest(fb(10*MS),10*MS)
    assert s.fault["reason"] == reason


def test_soft_threshold_hysteresis_and_unique_force_time():
    s=SafetySupervisor(ShadowConfig(limits=limits()))
    for t,tau in [(0,2.5),(10*MS,1.5),(20*MS,1.5)]:
        s.ingest(fb(t,external_nm=(tau,)*7),t)
        assert s.fault is None
    s.ingest(fb(30*MS,external_nm=(1.5,)*7),30*MS)
    assert s.fault["reason"] == "external_torque_sustained"
    s=SafetySupervisor(ShadowConfig(limits=limits()))
    for t,tau in [(0,2.5),(10*MS,.5),(20*MS,2.5),(30*MS,2.5)]: s.ingest(fb(t,external_nm=(tau,)*7),t)
    assert s.fault is None


def test_duplicate_feedback_does_not_refresh_watchdog():
    s=SafetySupervisor(ShadowConfig(limits=limits()))
    s.ingest(fb(0,frame=1),0)
    s.ingest(fb(10*MS,frame=1),10*MS)
    assert s.latest.timestamp_ns == 0
    s.ingest(fb(30*MS,frame=1),30*MS)
    assert s.fault["reason"] == "feedback_timeout"


@pytest.mark.parametrize("update,now,reason", [
    (fb(10*MS,context="other"),10*MS,"calibration_context_changed"),
    (fb(10*MS,frame=2),10*MS,"controller_frame_reset"),
    (fb(0,frame=101),10*MS,"feedback_time_reversed"),
    (fb(30*MS,frame=101),30*MS,"feedback_gap"),
    (fb(10*MS,frame=101),31*MS,"feedback_stale_or_future"),
    (fb(10*MS,frame=101,force_timestamp_ns=20*MS),10*MS,"force_stale_or_future"),
])
def test_feedback_context_time_and_counter_faults(update,now,reason):
    s=SafetySupervisor(ShadowConfig(limits=limits()));s.ingest(fb(frame=100),0)
    s.ingest(update,now)
    assert s.fault["reason"] == reason


def test_frame_wrap_and_zero_are_valid():
    s=SafetySupervisor(ShadowConfig(limits=limits()));s.ingest(fb(frame=999999),0)
    s.ingest(fb(10*MS,frame=0),10*MS)
    assert s.fault is None and s.new_frames == 2


def test_internal_and_raw_torque_have_separate_limits():
    for field,limit_field,reason in [("raw_external_nm","raw_torque_hard_nm","raw_torque_hard"),
                                     ("internal_nm","internal_torque_hard_nm","internal_torque_hard")]:
        s=SafetySupervisor(ShadowConfig(limits=limits(**{limit_field:(10.,)*7})))
        s.ingest(fb(**{field:(11.,)*7}),0)
        assert s.fault["reason"] == reason


def test_goal_needs_actual_dwell_low_velocity_and_verification():
    g=GoalMonitor(joint_goal())
    assert g.update(fb(),0)["status"] == "GOAL_CANDIDATE"
    assert g.update(fb(),15*MS)["dwell_ms"] == 0  # same frame does not advance dwell
    g.update(fb(10*MS,dq_rad_s=(.02,)*7),10*MS)
    assert g.inside_since is None
    for t in (20,30,40): assert g.update(fb(t*MS),t*MS)["status"] == "GOAL_CANDIDATE"
    assert g.update(fb(50*MS),50*MS)["status"] == "REACHED"
    g=GoalMonitor(joint_goal(verified=False))
    for t in (0,10,20,30): result=g.update(fb(t*MS),t*MS)
    assert result["status"] == "UNVERIFIED_CANDIDATE"


def test_goal_gap_resets_dwell_and_wrong_target_stalls():
    g=GoalMonitor(joint_goal(progress_timeout_s=.04))
    g.update(fb(),0)
    assert g.update(fb(30*MS),30*MS)["dwell_ms"] == 0
    g=GoalMonitor(joint_goal(target_q_rad=(1.,)*7,progress_timeout_s=.04))
    for t in (0,10,20,30,40): result=g.update(fb(t*MS),t*MS)
    assert result["status"] == "STALLED"


def insertion_goal(**kw):
    return Goal(**{**dict(goal_id="hole2",kind="insertion",verified=True,frame_id="fixture",
                         entrance_m=(0.,0.,0.),insertion_axis=(0.,0.,1.),
                         target_quaternion_xyzw=(0.,0.,0.,1.),depth_min_m=.02,depth_max_m=.025,
                         lateral_tolerance_m=.001,angle_tolerance_rad=.05,dwell_ms=30),**kw})


def test_insertion_geometry_frames_depth_and_orientation():
    g=GoalMonitor(insertion_goal())
    for t in (0,10,20,30):
        p=Pose(t*MS,"fixture",(0.,0.,.022),(0.,0.,0.,-1.))
        result=g.update(fb(t*MS),t*MS,p)
    assert result["status"] == "REACHED"  # quaternion sign equivalence
    assert g.update(fb(40*MS),40*MS,Pose(40*MS,"fixture",(0.,0.,.03),(0,0,0,1)))["status"] == "DEPTH_EXCEEDED"
    g=GoalMonitor(insertion_goal())
    assert g.update(fb(),0,Pose(0,"base",(0,0,.022),(0,0,0,1)))["status"] == "UNCONFIRMED"
    assert g.update(fb(10*MS),10*MS,Pose(10*MS,"fixture",(.003,0,.022),(0,0,0,1)))["status"] == "RUNNING"
    assert g.update(fb(20*MS),20*MS,None)["status"] == "UNCONFIRMED"


def test_safety_overrides_simultaneous_goal_and_clears_pending_actions():
    c=controller(ShadowConfig(limits=limits(),goal=joint_goal(dwell_ms=10)))
    assert c.tick(0)["state"] == "RUNNING_SHADOW"
    c.feedback(fb(10*MS,external_nm=(4.,)*7),10*MS)
    row=c.tick(10*MS)
    assert row["state"] == "ABORTED_SHADOW" and row["proposal"] is None
    assert c.stop_request["reason"] == "external_torque_hard"
    assert not c.executor.chunks
    assert not c.prediction(np.zeros((100,7)),query_id=2,start_ns=20*MS,observed_ns=20*MS,
                            available_ns=20*MS,now_ns=20*MS,run_id="run1")
    assert not c.report()["physical_stop_confirmed"]


def test_normal_goal_is_only_shadow_completion_and_safety_can_override():
    c=controller(ShadowConfig(limits=limits(),goal=joint_goal(dwell_ms=10)))
    c.tick(0);c.feedback(fb(10*MS),10*MS)
    row=c.tick(10*MS)
    assert row["state"] == "COMPLETED_SHADOW" and row["proposal"] is None
    c.feedback(fb(20*MS,external_nm=(4.,)*7),20*MS)
    assert c.state == "ABORTED_SHADOW"
    assert c.report()["motor_control_api_calls"] == 0


@pytest.mark.parametrize("case,reason", [("camera","camera_timeout"),("policy","action_unavailable_or_invalid"),
                                      ("timeout","task_timeout"),("late","execution_deadline")])
def test_runtime_watchdogs(case,reason):
    cfg=ShadowConfig(limits=limits(),task_timeout_s=.02 if case=="timeout" else 180)
    c=controller(cfg);c.tick(0)
    if case=="late":
        c.feedback(fb(20*MS),20*MS);c.tick(10*MS,21*MS)
    else:
        for t in range(10,221,10):
            c.feedback(fb(t*MS),t*MS)
            if case!="camera":
                for camera in ("d455","d405"):c.camera(camera,t*MS,t*MS)
            c.tick(t*MS)
            if c.stop_request:break
    assert c.stop_request["reason"] == reason


def test_unknown_limits_never_produce_approved_action():
    c=controller(ShadowConfig());row=c.tick(0)
    assert row["proposal"] is not None
    assert row["constrained_proposal"]["action_rad"] is None
    assert not c.report()["readiness"]["motion_possible"]


def test_startup_stale_data_cannot_arm_and_running_fault_never_recovers():
    c=ShadowController(ShadowConfig(),0,"run1")
    c.feedback(fb(),0)
    c.feedback(fb(10*MS),90*MS)  # logged/drop only before shadow is started
    assert c.state == "WAITING" and c.safety.startup_stale_dropped == 1
    c.feedback(fb(100*MS),100*MS)
    for name in ("d455","d405"):c.camera(name,100*MS,100*MS)
    c.prediction(np.zeros((100,7)),query_id=1,start_ns=100*MS,observed_ns=100*MS,
                 available_ns=100*MS,now_ns=100*MS,run_id="run1")
    assert c.tick(100*MS)["state"] == "RUNNING_SHADOW"
    c.feedback(fb(110*MS),140*MS)
    assert c.state == "ABORTED_SHADOW"
    c.feedback(fb(150*MS),150*MS)
    assert c.state == "ABORTED_SHADOW"
    assert c.safety.startup_gaps == 1


def test_startup_context_and_hard_faults_still_latch():
    c=ShadowController(ShadowConfig(limits=limits()),0,"run1")
    c.feedback(fb(),0)
    c.feedback(fb(10*MS,context="changed"),10*MS)
    assert c.state == "ABORTED_SHADOW"


def test_latched_soft_failure_still_records_hard_escalation_and_manual_stop():
    c=controller();c.tick(0)
    c.safety.trip("policy_timeout",0);c._sync_fault(0)
    assert c.stop_request["priority"]==2
    c.feedback(fb(10*MS,external_nm=(5.,)*7),10*MS)
    assert c.stop_request["reason"]=="external_torque_hard"
    assert c.stop_request["first_reason"]=="policy_timeout"
    assert c.report()["first_safety_fault"]["reason"]=="policy_timeout"
    assert c.executor.closed
    c.stop("operator_stop",20*MS)
    assert c.stop_request["priority"]==4
    c.feedback(fb(30*MS,error_code=7),30*MS)
    assert c.stop_request["reason"]=="operator_stop"
    c=ShadowController(ShadowConfig(limits=limits()),0,"run1")
    c.feedback(fb(external_nm=(4.,)*7),0)
    assert c.state == "ABORTED_SHADOW"


def test_independent_process_watches_when_parent_stops_producing(tmp_path):
    worker=ShadowProcess(ShadowConfig(),tmp_path/"monitor")
    now=time.monotonic_ns();anchor=worker.origin_ns+(now-worker.origin_ns)//PERIOD_NS*PERIOD_NS
    value=fb(now,frame=1)
    worker.publish("feedback",value=value.__dict__)
    for camera in ("d455","d405"):worker.publish("camera",name=camera,received_ns=now)
    worker.publish("prediction",actions=np.zeros((100,7)).tolist(),metadata=dict(query_id=1,start_ns=anchor,
                    observed_ns=anchor,available_ns=time.monotonic_ns(),run_id=worker.run_id))
    time.sleep(.12)
    worker.close()
    report=json.loads((tmp_path/"monitor/report.json").read_text())
    assert report["state"] == "ABORTED_SHADOW"
    assert report["stop_request"]["reason"] in ("feedback_timeout","feedback_stale_or_future","execution_deadline")
    assert report["motor_control_api_calls"] == 0
    assert report["monitor_pid"] != __import__("os").getpid()


def test_monitor_failure_with_queued_data_does_not_hang_cleanup(tmp_path):
    worker=ShadowProcess(ShadowConfig(),tmp_path/"failed_monitor")
    # Unknown input faults the consumer, leaving enough queued data to fill its
    # pipe. Cleanup must remain bounded after the consumer has exited.
    for _ in range(100):
        worker.inbox.put_nowait({"kind":"invalid_test_message","payload":"x"*10000})
    worker.process.join(5)
    assert not worker.process.is_alive()
    begin=time.monotonic();worker.close()
    assert time.monotonic()-begin<1
    report=json.loads((tmp_path/"failed_monitor/report.json").read_text())
    assert report["state"]=="ABORTED_SHADOW"
    assert report["stop_request"]["reason"]=="monitor_exception"
