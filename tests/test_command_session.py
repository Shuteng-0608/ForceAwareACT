"""Nominal startup/send/stop and configuration boundaries; no robot/fault injection."""
from dataclasses import replace
import json
from pathlib import Path
import queue
import subprocess
import sys
import threading

import numpy as np
import pytest

from force_aware_act.control.command_session import CommandSession, SessionOptions, SubmittedActionLimiter
from force_aware_act.control.contracts import Feedback
from force_aware_act.control.execution import TemporalExecutor
from force_aware_act.control.goal_selection import select_goal
from force_aware_act.control.kinematics import right_flange_pose
from force_aware_act.control.sdk_transport import RightArmTransport
from scripts.deployment.run_contact_controller import parse_arguments

ROOT = Path(__file__).resolve().parents[1]


class JournalTransport:
    hardware = False

    def __init__(self, q):
        self.q = np.array(q)
        self.calls = []
        self.revoked = False
        self.closed = False
        self.parameters = None
        self.mode = 0

    def configure(self, baseline):
        self.parameters = baseline
        self.calls.append(("parameters", baseline))
        return "submitted"

    def enter_impedance(self):
        self.mode = 3
        self.calls.append(("mode", 3))
        return "submitted"

    def target(self, q, *, deadline_ns):
        assert not self.revoked
        self.calls.append(("target", list(q)))
        return "submitted"

    def revoke_targets(self):
        self.revoked = True

    def servo_off(self):
        assert self.revoked
        self.calls.append(("servo_off", 0))
        self.mode = 0
        return "submitted"

    def close(self):
        self.closed = True

    def row(self, t):
        b = self.parameters or json.loads(select_goal("x13y07").controller_baseline_json)
        return {"received_monotonic_ns": t, "frame_serial": t//1_000_000+1,
                "state": self.mode, "impedance_type": 1, "error_code": 0,
                "qpos_rad": self.q.tolist(), "qvel_rad_s": [0.]*7,
                "external_raw_nm": [.3]*7, "internal_torque_nm": [2.]*7,
                "tool_kine": [0.]*6, "tool_dyn": [0.]*10,
                "velocity_ratio_percent": b["velocity_ratio_percent"],
                "acceleration_ratio_percent": b["acceleration_ratio_percent"],
                "joint_stiffness_nm_per_degree": b["joint_stiffness_nm_per_degree"],
                "joint_damping_ratio": b["joint_damping_ratio"]}


def prepared_session(*, servo_off=False, centered_goal=False):
    selected = select_goal("x13y07")
    q = (np.array(selected.config.limits.joint_min_rad)+selected.config.limits.joint_max_rad)/2
    if centered_goal:
        pose = right_flange_pose(q, 0)
        goal = replace(selected.config.goal, target_position_m=pose.position_m)
        selected = replace(selected, config=replace(selected.config, goal=goal))
    transport = JournalTransport(q)
    session = CommandSession(selected, SessionOptions(servo_off_after_standstill=servo_off), transport,
                             run_id="nominal", origin_ns=0, enable_control=True, operator_confirmed_no_contact=True)
    for t in range(0, 3_510_000_000, 10_000_000):
        session.feedback(transport.row(t), t)
        session.poll(t)
        if session.state == "READY":
            return session, transport, t
    raise AssertionError(session.report())


def nominal_tick(session, transport, t, query):
    session.feedback(transport.row(t), t)
    for name in ("d455", "d405"):
        session.camera(name, t, t)
    session.prediction(np.tile(transport.q+.004, (100,1)), query_id=query, start_ns=t,
                       observed_ns=t, available_ns=t, now_ns=t, run_id="nominal")
    return session.tick(t, t)


def test_startup_orders_packets_then_zero_in_final_mode_and_sends_only_when_ready():
    session, transport, t = prepared_session()
    assert [x[0] for x in transport.calls] == ["parameters", "mode"]
    assert session.calibration["mode"] == 3
    np.testing.assert_allclose(session.bias, .3)
    assert session.calibration["start_ns"] > session.history[3]["at_ns"]
    for query in range(1, 6):
        row = nominal_tick(session, transport, t+query*10_000_000, query)
        assert row["submission"] == "submitted"
        assert row["commands_sent"] is False  # Journal transport cannot send to hardware.
        if query > 1:
            assert row["constrained_proposal"]["step_reference"] == "previous_successfully_submitted_target"
    assert session.state == "RUNNING" and session.packet_counts["target"] == 5
    np.testing.assert_allclose(session.core.limiter.committed.previous_q, transport.q+.004)


@pytest.mark.parametrize("servo_off", [False, True])
def test_explicit_stop_latches_then_observes_standstill_and_optional_idle(servo_off):
    session, transport, t = prepared_session(servo_off=servo_off)
    nominal_tick(session, transport, t+10_000_000, 1)
    stop_at = t+20_000_000
    session.stop("operator_stop", stop_at)
    assert transport.revoked and session.core.executor.closed and not session.core.executor.chunks
    for now in range(stop_at+10_000_000, stop_at+900_000_000, 10_000_000):
        session.feedback(transport.row(now), now)
        assert session.tick(now, now) is None
    assert len([c for c in transport.calls if c[0] == "target"]) == 1
    assert session.state == ("SERVO_OFF_CONFIRMED" if servo_off else "STOPPED_HOLDING_LAST_TARGET")
    assert session.packet_counts["servo_off"] == int(servo_off)
    assert session.bias is None and session.stop_request["targets_revoked"]
    assert session.report()["physical_stop_verified_by_this_delivery"] is False


def test_actual_fk_sphere_dwell_uses_same_stop_path():
    session, transport, start = prepared_session(centered_goal=True)
    for i in range(1, 55):
        nominal_tick(session, transport, start+i*10_000_000, i)
        if session.stop_request:
            break
    assert session.stop_request["reason"] == "coarse_goal_reached"
    assert transport.revoked and session.core.goal.result["goal_id"] == "x13y07"


def test_preview_and_discard_do_not_advance_submitted_reference():
    config = select_goal("x13y07").config
    q = (np.array(config.limits.joint_min_rad)+config.limits.joint_max_rad)/2
    f = Feedback(0, 1, tuple(q), (0.,)*7, (0.,)*7, 0, True, "nominal")
    limiter = SubmittedActionLimiter(config.limits)
    limiter.apply(q+.004, f, 0)
    assert limiter.committed.previous_q is None
    limiter.commit()
    limiter.apply(q+.02, f, 10_000_000)
    limiter.discard()  # A valid preview is not automatically a transmitted target.
    np.testing.assert_allclose(limiter.committed.previous_q, q+.004)
    row = limiter.apply(q+.01, f, 20_000_000)
    np.testing.assert_allclose(row["step_rad"], .006)


def test_real_run_200ms_boundary_uses_fresh_chunks_on_the_same_tick():
    """Reproduce the 10 contributing query ages in the failed real run."""
    executor = TemporalExecutor(0, "boundary", select_goal("x13y07").config)
    for i in range(10):
        start = i*20_000_000
        executor.submit(np.tile(np.arange(100)[:, None], (1, 7)), query_id=i+1,
                        start_ns=start, observed_ns=start, available_ns=start+10_000_000,
                        now_ns=start+10_000_000, run_id="boundary")
    result = executor.sample(200_000_000, now_ns=200_421_834)
    assert result["query_ids"] == list(range(2, 11))
    assert result["chunk_indices"] == list(range(18, 0, -2))
    assert result["prediction_age_ms"] == list(range(180, 0, -20))
    assert result["tick_ns"] == 200_000_000
    assert sum(result["weights"]) == pytest.approx(1)


def test_expired_target_discards_preview_and_next_tick_can_submit():
    session, transport, start = prepared_session()
    nominal_tick(session, transport, start+10_000_000, 1)
    committed = session.core.limiter.committed
    target = transport.target
    transport.target = lambda *args, **kwargs: "expired"
    row = nominal_tick(session, transport, start+20_000_000, 2)
    assert row["submission"] == "expired" and not session.stop_request
    assert session.core.limiter.committed is committed
    assert session.core.limiter.pending is None
    assert session.packet_counts["target"] == 1 and session.expired_target_ticks == 1
    transport.target = target
    row = nominal_tick(session, transport, start+30_000_000, 3)
    assert row["submission"] == "submitted" and session.packet_counts["target"] == 2


def test_mode_transition_transient_waits_for_a_complete_stable_calibration_window():
    """Regression for the measured 2026-10-08 J3 startup peak, not a hardware fault test."""
    selected = select_goal("x13y07")
    q = (np.array(selected.config.limits.joint_min_rad)+selected.config.limits.joint_max_rad)/2
    transport = JournalTransport(q)
    options = SessionOptions()
    session = CommandSession(selected, options, transport, run_id="nominal", origin_ns=0,
                             enable_control=True, operator_confirmed_no_contact=True)
    for t in range(0, 2_000_000_000, 10_000_000):
        session.feedback(transport.row(t), t)
        session.poll(t)
        if session.state == "CALIBRATING":
            break
    assert session.state == "CALIBRATING"
    start = t
    for index in range(1, 201):
        t = start+index*10_000_000
        row = transport.row(t)
        velocity = .01931209989822374*np.sin(np.pi*index/16) if index < 16 else 0.
        transport.q[2] += velocity*.01
        row["qpos_rad"] = transport.q.tolist()
        row["qvel_rad_s"][2] = velocity
        session.feedback(row, t); session.poll(t)
    assert session.state == "CALIBRATING" and session.bias is None and not session.stop_request
    for index in range(201, 235):
        t = start+index*10_000_000
        session.feedback(transport.row(t), t); session.poll(t)
        if session.state == "READY":
            break
    assert session.state == "READY" and not session.stop_request
    assert session.calibration["end_ns"]-session.calibration["start_ns"] >= 2_000_000_000
    assert session.calibration_status["windows_checked"] > 1
    assert not session.calibration_status["exceeded_joints"]
    assert options.stationary_velocity_rad_s == .01 and options.stationary_range_rad == .005
    assert session.packet_counts["target"] == 0


def test_default_preview_never_loads_devices_and_records_operator_values():
    result = subprocess.run([sys.executable, "scripts/deployment/run_contact_controller.py", "--goal-id", "x13y07"],
                            cwd=ROOT, capture_output=True, text=True, check=True)
    plan = json.loads(result.stdout)
    assert not plan["hardware_connected"] and not plan["session"]["allow_hardware"]
    assert not plan["session"]["servo_off_after_standstill"]
    baseline = plan["goal_selection"]["controller_baseline"]
    assert baseline["velocity_ratio_percent"] == baseline["acceleration_ratio_percent"] == 10
    assert baseline["joint_stiffness_nm_per_degree"] == [2.5,2.5,2.5,2,1.5,1.2,1]
    assert baseline["joint_damping_ratio"] == [.3,.3,.3,.2,.2,.2,.2]


def test_explicit_flag_cannot_override_disabled_config_or_load_sdk(monkeypatch):
    args, _, options = parse_arguments(["--goal-id", "x13y07", "--prepare-only"])
    assert args.prepare_only and not args.enable_control and not options.allow_hardware
    with pytest.raises(SystemExit) as error:
        parse_arguments(["--goal-id", "x13y07", "--enable-control", "--operator-confirmed-no-contact"])
    assert error.value.code == 2
    import ctypes
    monkeypatch.setattr(ctypes, "CDLL", lambda *a, **k: pytest.fail("must not load SDK"))
    with pytest.raises(ValueError, match="disabled"):
        RightArmTransport("unused.so", "unused_sdk.so", SessionOptions(), enable_control=True)


def test_worker_nominal_stop_and_release_writes_report(tmp_path, monkeypatch):
    from force_aware_act.control import command_worker
    session, transport, t = prepared_session()
    now = [t+10_000_000]
    monkeypatch.setattr(command_worker.time, "monotonic_ns", lambda: now[0])
    monkeypatch.setattr(command_worker.time, "sleep", lambda seconds: now.__setitem__(0, now[0]+10_000_000))
    transport.read = lambda: transport.row(now[0])
    stop = threading.Event(); stop.set()
    report = command_worker.run_loop(session, transport, queue.Queue(), queue.Queue(), stop, threading.Event(), tmp_path)
    assert transport.closed and transport.revoked and report["state"] == "STOPPED_HOLDING_LAST_TARGET"
    assert report["commands_sent"] is False and not report["errors"]
    assert json.loads((tmp_path/"report.json").read_text())["stop"]["targets_revoked"]


def test_worker_optional_hud_reports_terminal_state_without_new_commands(tmp_path, monkeypatch):
    from force_aware_act.control import command_worker
    from force_aware_act.control.hud_telemetry import HudTelemetry
    session,transport,t=prepared_session()
    original=list(transport.calls)
    now=[t+10_000_000]
    monkeypatch.setattr(command_worker.time,'monotonic_ns',lambda:now[0])
    monkeypatch.setattr(command_worker.time,'sleep',lambda seconds:now.__setitem__(0,now[0]+10_000_000))
    transport.read=lambda:transport.row(now[0])
    stop=threading.Event();stop.set();channel=HudTelemetry()
    report=command_worker.run_loop(session,transport,queue.Queue(),queue.Queue(),stop,threading.Event(),tmp_path,channel)
    assert report['state']=='STOPPED_HOLDING_LAST_TARGET' and transport.calls==original
    assert channel.read()['state']==report['state']
    assert report['hud_telemetry']['publish_errors']==0


def test_process_constructor_is_locked_before_creating_files(tmp_path):
    from force_aware_act.control.command_worker import CommandProcess
    with pytest.raises(ValueError):
        CommandProcess(select_goal("x13y07"), SessionOptions(), "unused", "unused", "127.0.0.1", tmp_path/"no_create")
    assert not (tmp_path/"no_create").exists()
