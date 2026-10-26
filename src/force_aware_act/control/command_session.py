"""Single-owner right-arm execution lifecycle; no SDK import or device access.

The existing ShadowController supplies causal aggregation, force supervision and
FK goals. Its limiter is replaced locally by a transactional limiter: a proposed
target becomes the reference only after the transport reports SUBMITTED.
"""
from __future__ import annotations

from collections import deque
from copy import deepcopy
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path

import numpy as np

from .contracts import Feedback, timestamp, vector
from .execution import ActionLimiter
from .runtime import ShadowController


@dataclass(frozen=True)
class SessionOptions:
    schema_version: str = "right_command_session_v1"
    allow_hardware: bool = False
    stop_policy: str = "cease_targets"
    # An IDLE packet disables the servo, not the robot's main electrical supply.
    servo_off_after_standstill: bool = False
    startup_timeout_s: float = 15.
    packet_settle_ms: float = 250.
    transition_timeout_s: float = 2.
    calibration_window_s: float = 2.
    stationary_dwell_ms: float = 500.
    stationary_velocity_rad_s: float = .01
    stationary_range_rad: float = .005
    calibration_std_nm: float = .15
    stop_observe_timeout_s: float = 2.
    sdk_busy_timeout_ms: float = 100.
    expected_sdk_version: int = 100343013
    expected_sdk_sha256: str = "47ab2ab6e35cf899eae3978dfb50ee219e05b10f72aedd6bc9ba6a338a8b4011"
    ft_calibration_file: str | None = None
    ft_compensation_library: str | None = None
    ft_tare_window_s: float = 5.
    ft_force_hard_n: tuple[float, ...] | None = None
    ft_torque_hard_nm: tuple[float, ...] | None = None

    def __post_init__(self):
        if self.schema_version != "right_command_session_v1" or self.stop_policy != "cease_targets":
            raise ValueError("unsupported session schema/stop policy")
        if type(self.allow_hardware) is not bool or type(self.servo_off_after_standstill) is not bool:
            raise ValueError("hardware and servo-off switches must be JSON booleans")
        for name in self.__dataclass_fields__:
            if name.endswith(("_s", "_ms", "_rad", "_nm")) and name != "ft_torque_hard_nm":
                value = getattr(self, name)
                if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                    raise ValueError(f"invalid {name}")
        if self.calibration_window_s < 2 or self.stationary_dwell_ms < 500:
            raise ValueError("calibration >=2s and standstill observation >=500ms required")
        if bool(self.ft_calibration_file) != bool(self.ft_compensation_library):
            raise ValueError("FT requires both calibration file and compensation library")
        if self.ft_tare_window_s < 5:
            raise ValueError("sensor tare requires at least 5 seconds")
        for name in ("ft_force_hard_n", "ft_torque_hard_nm"):
            if getattr(self, name) is not None:
                object.__setattr__(self, name, tuple(vector(getattr(self, name), name, 3, positive=True)))

    @classmethod
    def read(cls, path):
        return cls(**json.loads(Path(path).read_text()))


class SubmittedActionLimiter:
    """A fresh proposal per tick. Busy/dropped proposals never advance history."""
    def __init__(self, limits):
        self.committed = ActionLimiter(limits)
        self.pending = None

    def apply(self, target, feedback, tick_ns):
        candidate = deepcopy(self.committed)
        result = candidate.apply(target, feedback, tick_ns)
        self.pending = candidate if result.get("action_rad") is not None else None
        if result.get("step_reference") == "previous_accepted_shadow_target":
            result["step_reference"] = "previous_successfully_submitted_target"
        return result

    def commit(self):
        if self.pending is None:
            raise RuntimeError("no checked target to commit")
        self.committed, self.pending = self.pending, None

    def discard(self):
        self.pending = None


class CommandSession:
    """Caller serializes all events, polling and transport calls in one CPU loop.

Transport: configure(baseline), enter_impedance(), target(q_rad), servo_off()
return 'submitted', 'busy' or 'error'; target may also return 'expired' without
sending or latching an SDK failure. `hardware` identifies real side effects.
Start authorization is per session, never persisted by importing this module.
"""
    def __init__(self, selection, options, transport, *, run_id, origin_ns,
                 enable_control=False, operator_confirmed_no_contact=False, sensor_ft=None, output=None):
        if not enable_control or not operator_confirmed_no_contact:
            raise ValueError("explicit startup control and no-contact confirmation required")
        if transport.hardware and not options.allow_hardware:
            raise ValueError("session configuration disables hardware control")
        self.selection, self.options, self.transport = selection, options, transport
        if bool(options.ft_calibration_file) != (sensor_ft is not None):
            raise ValueError("sensor session must have a loaded compensation core")
        self.sensor_ft, self.output = sensor_ft, Path(output) if output is not None else None
        if sensor_ft is not None and self.output is None:
            raise ValueError("sensor session requires an output directory for tare")
        self.baseline = json.loads(selection.controller_baseline_json)
        if self.baseline["arm"] != "right" or self.baseline["control_mode"] != "joint_impedance":
            raise ValueError("right joint impedance baseline required")
        for key in ("velocity_ratio_percent", "acceleration_ratio_percent"):
            value = self.baseline[key]
            if type(value) is not int or not 1 <= value <= 100:
                raise ValueError(f"invalid controller ratio: {key}")
        vector(self.baseline["joint_stiffness_nm_per_degree"], "K", positive=True)
        vector(self.baseline["joint_damping_ratio"], "D", positive=True)
        if selection.config.limits.motion_guard != "step_reject" or selection.config.limits.missing():
            raise ValueError("complete reviewed step-only protection required")
        self.core = ShadowController(selection.config, origin_ns, run_id)
        self.core.limiter = SubmittedActionLimiter(selection.config.limits)
        self.run_id, self.origin_ns = run_id, timestamp(origin_ns)
        self.state, self.phase_ns = "WAIT_STATIONARY", origin_ns
        self.last = None
        self.tool_context = None
        self.initial_mode = None
        self.rows = deque(maxlen=3000)
        self.bias = None
        self.stop_bias = None
        self.calibration = None
        self.calibration_status = {"status": "not_started", "windows_checked": 0}
        self.busy_since = None
        self.stop_request = None
        self.stop_rows = deque(maxlen=1000)
        self.stop_stage = None
        self.packet_counts = {k: 0 for k in ("parameters", "mode", "target", "servo_off")}
        self.expired_target_ticks = 0
        self.history = [{"state": self.state, "at_ns": origin_ns}]
        self.last_execution = None

    @property
    def terminal(self):
        return self.state in {"STOPPED_HOLDING_LAST_TARGET", "SERVO_OFF_CONFIRMED", "STOP_UNCONFIRMED"}

    def _transition(self, state, now):
        self.state, self.phase_ns = state, now
        self.history.append({"state": state, "at_ns": now})

    def _packet(self, kind, callback, now):
        result = callback()
        allowed = ("submitted", "busy", "error", "expired") if kind == "target" else ("submitted", "busy", "error")
        if result not in allowed:
            result = "error"
        if result == "submitted":
            self.packet_counts[kind] += 1
            self.busy_since = None
        elif result == "busy":
            if self.busy_since is None:
                self.busy_since = now
            if now-self.busy_since >= self.options.sdk_busy_timeout_ms*1e6:
                self.stop("sdk_busy_timeout", now)
        elif result == "expired":
            self.expired_target_ticks += 1
            self.busy_since = None
        else:
            self.stop("sdk_submission_error", now)
            self.stop_request["submission_failure"] = {"packet": kind, "at_ns": now,
                "transport": deepcopy(getattr(self.transport, "last_submission_detail", None))}
        return result

    def _parameters_match(self, row):
        b = self.baseline
        return (row["velocity_ratio_percent"] == b["velocity_ratio_percent"]
                and row["acceleration_ratio_percent"] == b["acceleration_ratio_percent"]
                and np.allclose(row["joint_stiffness_nm_per_degree"], b["joint_stiffness_nm_per_degree"], rtol=0, atol=1e-4)
                and np.allclose(row["joint_damping_ratio"], b["joint_damping_ratio"], rtol=0, atol=1e-5))

    def _stationary(self, rows, duration_ms):
        if len(rows) < 2 or rows[-1]["received_monotonic_ns"]-rows[0]["received_monotonic_ns"] < duration_ms*1e6:
            return False
        return (np.max(np.abs([r["qvel_rad_s"] for r in rows])) <= self.options.stationary_velocity_rad_s
                and np.max(np.ptp([r["qpos_rad"] for r in rows], axis=0)) <= self.options.stationary_range_rad)

    def _calibration_window(self, row):
        """Wait for a complete recent stable window; never filter bad samples.

Mode/parameter/tool and feedback checks still abort in feedback(). The existing
startup deadline bounds the wait. Thresholds apply to every sample retained in
the two-second window, including any transition transient still in that window.
"""
        self.rows.append(row)
        end_ns = row["received_monotonic_ns"]
        duration_ns = int(self.options.calibration_window_s*1e9)
        while len(self.rows) > 2 and end_ns-self.rows[1]["received_monotonic_ns"] >= duration_ns:
            self.rows.popleft()
        if len(self.rows) < 2 or end_ns-self.rows[0]["received_monotonic_ns"] < duration_ns:
            return
        velocity = np.max(np.abs([r["qvel_rad_s"] for r in self.rows]), axis=0)
        position_range = np.ptp([r["qpos_rad"] for r in self.rows], axis=0)
        raw = np.asarray([r["external_raw_nm"] for r in self.rows])
        std = raw.std(axis=0)
        pending = {
            "velocity": (velocity, self.options.stationary_velocity_rad_s),
            "position_range": (position_range, self.options.stationary_range_rad),
            "torque_std": (std, self.options.calibration_std_nm),
        }
        failed = {name: (np.flatnonzero(values > limit)+1).tolist()
                  for name, (values, limit) in pending.items() if np.any(values > limit)}
        self.calibration_status = {
            "status": "waiting_for_stable_window" if failed else "valid",
            "windows_checked": self.calibration_status["windows_checked"]+1,
            "start_ns": self.rows[0]["received_monotonic_ns"], "end_ns": end_ns,
            "samples": len(self.rows), "span_s": (end_ns-self.rows[0]["received_monotonic_ns"])/1e9,
            "max_abs_velocity_per_joint_rad_s": velocity.tolist(),
            "position_range_per_joint_rad": position_range.tolist(),
            "torque_std_per_joint_nm": std.tolist(), "exceeded_joints": failed,
        }
        if failed:
            return
        if self.sensor_ft is not None:
            if not self.sensor_ft.capture(end_ns, self.output / "ft_tare"):
                self.calibration_status["status"] = "waiting_for_sensor_tare"
                self.calibration_status["sensor_tare_reason"] = self.sensor_ft.waiting_reason
                return
            self.last["sensor_ft"] = self.sensor_ft.process(row)  # First freshly tared frame.
        self.bias = raw.mean(axis=0)
        self.calibration = {"bias_nm": self.bias.tolist(), "samples": len(self.rows),
                            "start_ns": self.rows[0]["received_monotonic_ns"], "end_ns": end_ns,
                            "mode": 3, "impedance_type": 1, "tool_context": self.tool_context,
                            "std_nm": std.tolist(), "source": "m_EST_Joint_Force",
                            "method": "recent_complete_stable_window_v2"}
        self._transition("READY", end_ns)

    def feedback(self, row, now_ns):
        """Each record is one NEW SDK frame, with monotonic receive time."""
        now_ns = timestamp(now_ns)
        received = timestamp(row["received_monotonic_ns"])
        fresh_limit = self.selection.config.feedback_max_age_ms*1e6
        if not 0 <= now_ns-received <= fresh_limit:
            self.stop("feedback_stale_or_future", now_ns); return
        for key in ("qpos_rad", "qvel_rad_s", "external_raw_nm", "internal_torque_nm",
                    "joint_stiffness_nm_per_degree", "joint_damping_ratio"):
            vector(row[key], key)
        vector(row["tool_kine"], "tool_kine", 6); vector(row["tool_dyn"], "tool_dyn", 10)
        frame = timestamp(row["frame_serial"])
        if frame > 1_000_000:
            self.stop("invalid_frame_serial", now_ns); return
        if self.last is not None:
            previous = self.last
            if frame == previous["frame_serial"]:
                return  # A cached frame cannot refresh safety or confirm stopping.
            if (received <= previous["received_monotonic_ns"] or
                (frame < previous["frame_serial"] and not (previous["frame_serial"] > 900000 and frame < 100000))):
                self.stop("feedback_sequence_reset", now_ns); return
            if received-previous["received_monotonic_ns"] > fresh_limit:
                self.rows.clear(); self.stop_rows.clear()
                if self.state != "WAIT_STATIONARY":
                    self.stop("feedback_gap", now_ns)
        self.last = dict(row)
        if self.stop_request:
            self._observe_stop(row, now_ns)
            return
        if row["error_code"] != 0:
            self.stop("controller_error", now_ns); return
        q = np.asarray(row["qpos_rad"])
        limits = self.selection.config.limits
        if np.any(q < limits.joint_min_rad) or np.any(q > limits.joint_max_rad):
            self.stop("measured_joint_limit", now_ns); return
        context = (tuple(row["tool_kine"]), tuple(row["tool_dyn"]))
        if self.tool_context is not None and context != self.tool_context:
            self.stop("tool_context_changed", now_ns); return
        self.tool_context = context
        if self.sensor_ft is not None:
            try:
                sensor = self.sensor_ft.process(row, collect=self.state == "CALIBRATING")
                self.last["sensor_ft"] = sensor
                if sensor["valid"]:
                    wrench = np.abs(sensor["wrench_si"])
                    for values, bound, reason in (
                        (wrench[:3], self.options.ft_force_hard_n, "sensor_force_hard"),
                        (wrench[3:], self.options.ft_torque_hard_nm, "sensor_torque_hard")):
                        if bound is not None and np.any(values >= bound):
                            self.stop(reason, now_ns); return
            except (ValueError, RuntimeError, KeyError) as exc:
                self.stop("sensor_ft_invalid", now_ns)
                self.stop_request["sensor_detail"] = str(exc)
                return
        if self.state == "WAIT_STATIONARY":
            if row["state"] not in (0, 3) or (row["state"] == 3 and row["impedance_type"] != 1):
                self.stop("unsupported_initial_mode", now_ns); return
            if self.initial_mode is not None and self.initial_mode != row["state"]:
                self.stop("initial_mode_changed", now_ns); return
            self.initial_mode = row["state"]
            self.rows.append(row)
            # Sliding window, independent of the eventual model execution rate.
            while len(self.rows) > 2 and received-self.rows[1]["received_monotonic_ns"] >= self.options.stationary_dwell_ms*1e6:
                self.rows.popleft()
        elif self.state == "PARAMETERS_SENT":
            if row["state"] != self.initial_mode or np.max(np.abs(row["qvel_rad_s"])) > self.options.stationary_velocity_rad_s:
                self.stop("arm_changed_during_parameter_preparation", now_ns); return
            if self._settled(now_ns) and self._parameters_match(row):
                self._transition("PARAMETERS_VERIFIED", now_ns)
        elif self.state == "MODE_SENT":
            if self._settled(now_ns) and row["state"] == 3 and row["impedance_type"] == 1 and self._parameters_match(row):
                self.rows.clear()
                self.calibration_status["status"] = "collecting"
                self._transition("CALIBRATING", now_ns)
        elif self.state in ("CALIBRATING", "READY", "RUNNING"):
            if row["state"] != 3 or row["impedance_type"] != 1 or not self._parameters_match(row):
                self.stop("mode_or_parameter_readback_changed", now_ns); return
            if self.state == "CALIBRATING":
                self._calibration_window(row)
            if self.bias is not None:
                value = self.as_feedback(row)
                self.core.feedback(value, now_ns)
                if self.core.stop_request:
                    self.stop(self.core.stop_request["reason"], now_ns)

    def as_feedback(self, row):
        if self.bias is None:
            raise RuntimeError("software bias not established")
        ns = row["received_monotonic_ns"]
        return Feedback(ns, row["frame_serial"], tuple(row["qpos_rad"]), tuple(row["qvel_rad_s"]),
                        tuple(np.asarray(row["external_raw_nm"])-self.bias), ns, True,
                        json.dumps((3, 1, self.tool_context)), row["error_code"],
                        tuple(row["external_raw_nm"]), tuple(row["internal_torque_nm"]))

    def camera(self, name, received_ns, now_ns):
        if self.stop_request is None:
            self.core.camera(name, received_ns, now_ns)
            if self.core.stop_request:
                self.stop(self.core.stop_request["reason"], now_ns)

    def prediction(self, actions, **metadata):
        if self.stop_request or self.state not in ("READY", "RUNNING"):
            return False
        result = self.core.prediction(actions, **metadata)
        if self.core.stop_request:
            self.stop(self.core.stop_request["reason"], metadata["now_ns"])
        return result

    def _settled(self, now):
        return now-self.phase_ns >= self.options.packet_settle_ms*1e6

    def poll(self, now_ns):
        """Always poll independently of camera/GPU activity, including shutdown."""
        if self.terminal:
            return
        if self.stop_request:
            # No new targets or automatic reconnection on any exit path.
            deadline = self.stop_request["requested_ns"] + self.options.stop_observe_timeout_s*1e9
            if now_ns >= deadline:
                self.stop_request["physical_status"] = "unconfirmed"
                self._transition("STOP_UNCONFIRMED", now_ns)
            return
        if self.last is not None and now_ns-self.last["received_monotonic_ns"] > self.selection.config.feedback_max_age_ms*1e6:
            self.stop("feedback_timeout", now_ns); return
        if self.state not in ("RUNNING",) and now_ns-self.origin_ns > self.options.startup_timeout_s*1e9:
            self.stop("calibration_timeout" if self.state == "CALIBRATING" else "startup_timeout", now_ns); return
        if self.state in ("PARAMETERS_SENT", "MODE_SENT") and now_ns-self.phase_ns > self.options.transition_timeout_s*1e9:
            self.stop("parameter_or_mode_confirmation_timeout", now_ns); return
        if self.state == "WAIT_STATIONARY" and self._stationary(self.rows, self.options.stationary_dwell_ms):
            if self._packet("parameters", lambda: self.transport.configure(self.baseline), now_ns) == "submitted":
                self.rows.clear(); self._transition("PARAMETERS_SENT", now_ns)
        elif self.state == "PARAMETERS_VERIFIED":
            if self._packet("mode", self.transport.enter_impedance, now_ns) == "submitted":
                self._transition("MODE_SENT", now_ns)

    def tick(self, tick_ns, now_ns):
        self.poll(now_ns)
        if self.stop_request or self.state not in ("READY", "RUNNING"):
            return None
        row = self.core.tick(tick_ns, now_ns)
        row["kind"] = "command_tick"
        row["motion_possible"] = self.transport.hardware
        row["submission"] = "none"
        if self.core.stop_request:
            self.stop(self.core.stop_request["reason"], now_ns)
        else:
            candidate = row.get("constrained_proposal")
            if candidate and candidate.get("action_rad") is not None:
                # Immediate same-process feedback/goal/force check has just run.
                cfg, fb = self.selection.config, self.core.safety.latest
                sources = {"execution": tick_ns+int(cfg.execution_lateness_ms*1e6),
                           "feedback": fb.timestamp_ns+int(cfg.feedback_max_age_ms*1e6),
                           "force": fb.force_timestamp_ns+int(cfg.force_max_age_ms*1e6),
                           "camera": min(self.core.camera_ns.values())+int(cfg.camera_max_age_ms*1e6),
                           "prediction": tick_ns+int((cfg.max_prediction_age_ms-max(row["proposal"]["prediction_age_ms"]))*1e6)}
                deadline_ns = min(sources.values())
                if now_ns > deadline_ns:
                    result = self._packet("target", lambda: "expired", now_ns)
                    row["submission_detail"] = {"result": "expired", "detail": "deadline_elapsed_before_transport"}
                else:
                    result = self._packet("target", lambda: self.transport.target(
                        candidate["action_rad"], deadline_ns=deadline_ns), now_ns)
                    row["submission_detail"] = deepcopy(getattr(self.transport, "last_submission_detail", None))
                row["submission"] = result
                row["submit_deadline_ns"] = deadline_ns
                row["submit_deadline_sources_ns"] = sources
                row["sdk_submit_finished_ns"] = (None if now_ns > deadline_ns else
                                                 getattr(self.transport, "last_submit_finished_ns", None))
                row["commands_sent"] = result == "submitted" and self.transport.hardware
                if result == "submitted":
                    self.core.limiter.commit()
                    if self.state == "READY":
                        self._transition("RUNNING", now_ns)
                else:
                    self.core.limiter.discard()
                # Busy/expired means DROP this tick; do not advance the submitted
                # reference or retry this target. Next tick aggregates afresh.
        row["state"] = self.state
        self.last_execution = row
        return row

    def stop(self, reason, now_ns):
        if self.stop_request is not None:
            return
        self.stop_request = {"reason": reason, "requested_ns": now_ns,
                             "targets_revoked": True, "physical_status": "unconfirmed",
                             "servo_off_requested": False, "servo_off_confirmed": False}
        self.core.stop(reason, now_ns, outcome="COMPLETED_SHADOW" if reason == "coarse_goal_reached" else "ABORTED_SHADOW")
        self.core.executor.close()
        self.core.limiter.discard()
        self.transport.revoke_targets()
        self.stop_bias = None if self.bias is None else self.bias.copy()
        self.bias = None  # Never reusable for inference or in another session.
        self.stop_rows.clear()
        self._transition("STOPPING", now_ns)

    def _observe_stop(self, row, now_ns):
        if self.terminal or row["received_monotonic_ns"] <= self.stop_request["requested_ns"]:
            return
        self._observe_sensor_after_stop(row, now_ns)
        if row["error_code"] != 0:
            self.stop_request["controller_error_after_stop"] = row["error_code"]
            self.stop_rows.clear(); return
        if (self.stop_bias is not None and row["state"] == 3 and row["impedance_type"] == 1
                and (tuple(row["tool_kine"]), tuple(row["tool_dyn"])) == self.tool_context):
            torque = np.abs(np.asarray(row["external_raw_nm"])-self.stop_bias)
            if np.any(torque >= self.selection.config.limits.torque_hard_nm):
                self.stop_request["hard_torque_after_stop"] = {"at_ns": now_ns, "absolute_nm": torque.tolist()}
        self.stop_rows.append(row)
        while len(self.stop_rows) > 2 and row["received_monotonic_ns"]-self.stop_rows[1]["received_monotonic_ns"] >= self.options.stationary_dwell_ms*1e6:
            self.stop_rows.popleft()
        if self.stop_stage == "IDLE_SENT":
            if row["state"] == 0 and now_ns-self.phase_ns >= self.options.packet_settle_ms*1e6:
                self.stop_request.update(servo_off_confirmed=True, physical_status="servo_off_feedback")
                self._transition("SERVO_OFF_CONFIRMED", now_ns)
            return
        if self._stationary(self.stop_rows, self.options.stationary_dwell_ms):
            self.stop_request["physical_status"] = "standstill_observed_servo_may_hold_last_target"
            if not self.options.servo_off_after_standstill:
                self._transition("STOPPED_HOLDING_LAST_TARGET", now_ns)
            elif row["state"] == 0:
                self.stop_request.update(servo_off_confirmed=True, physical_status="already_servo_off_feedback")
                self._transition("SERVO_OFF_CONFIRMED", now_ns)
            elif row["state"] == 3 and row["impedance_type"] == 1:
                result = self.transport.servo_off()
                self.stop_request["servo_off_submission"] = {
                    "result": result, "at_ns": now_ns,
                    "transport": deepcopy(getattr(self.transport, "last_submission_detail", None))}
                if result == "submitted":
                    self.packet_counts["servo_off"] += 1
                    self.stop_request["servo_off_requested"] = True
                    self.stop_stage, self.phase_ns = "IDLE_SENT", now_ns
                elif result != "busy":
                    self.stop_request["physical_status"] = "servo_off_submission_failed"
                    self._transition("STOP_UNCONFIRMED", now_ns)

    def _observe_sensor_after_stop(self, row, now_ns):
        """Keep the fixed run tare visible after revocation; never resume targets.

        This monitors and records escalation only. It does not implement contact
        unloading or change the separately configured standstill/servo-off path.
        """
        if self.sensor_ft is None:
            return
        monitor = self.stop_request.setdefault("sensor_monitor_after_stop", {
            "samples": 0, "valid_samples": 0, "invalid_samples": 0,
            "error_samples": 0, "peak_abs_N_Nm": None, "last_wrench_N_Nm": None})
        monitor["samples"] += 1
        try:
            if self.tool_context is not None and (
                    tuple(row["tool_kine"]), tuple(row["tool_dyn"])) != self.tool_context:
                raise ValueError("tool context changed after stop; fixed tare is no longer usable")
            sensor = self.sensor_ft.process(row)
            self.last["sensor_ft"] = sensor
            if not sensor["valid"]:
                monitor["invalid_samples"] += 1
                if self.sensor_ft.tare is not None:
                    raise ValueError("previously tared sensor became invalid after stop")
                return  # A preparation failure may have stopped before task tare.
            if (sensor["source_ns"] != row["received_monotonic_ns"]
                    or sensor["frame_serial"] != row["frame_serial"]):
                raise ValueError("sensor/actual-joint frame mismatch after stop")
            wrench = np.asarray(vector(sensor["wrench_si"], "sensor wrench after stop", 6))
            absolute = np.abs(wrench)
            monitor["valid_samples"] += 1
            previous = monitor["peak_abs_N_Nm"]
            monitor["peak_abs_N_Nm"] = np.maximum(previous if previous is not None else absolute, absolute).tolist()
            monitor["last_wrench_N_Nm"] = wrench.tolist()
            monitor["last_received_ns"] = row["received_monotonic_ns"]
            violations = []
            for offset, bounds, reason, units, names in (
                    (0, self.options.ft_force_hard_n, "sensor_force_hard_after_stop", "N", ("Fx", "Fy", "Fz")),
                    (3, self.options.ft_torque_hard_nm, "sensor_torque_hard_after_stop", "N*m", ("Tx", "Ty", "Tz"))):
                if bounds is None:
                    continue
                for axis in np.flatnonzero(absolute[offset:offset+3] >= bounds):
                    violations.append({"reason": reason, "axis": names[axis],
                        "value": float(wrench[offset+axis]), "limit": float(bounds[axis]), "units": units})
            if violations:
                event = {"detected_ns": now_ns, "received_ns": row["received_monotonic_ns"],
                         "frame_serial": row["frame_serial"], "violations": violations}
                self.stop_request.setdefault("sensor_hard_after_stop", event)
                self.core.safety.trip(violations[0]["reason"], now_ns, violations=violations)
        except (ValueError, RuntimeError, KeyError, TypeError) as exc:
            monitor["error_samples"] += 1
            row.pop("sensor_ft", None)
            self.last.pop("sensor_ft", None)
            event = {"detected_ns": now_ns, "received_ns": row["received_monotonic_ns"], "detail": str(exc)}
            self.stop_request.setdefault("sensor_ft_error_after_stop", event)
            self.core.safety.trip("sensor_ft_error_after_stop", now_ns, detail=str(exc))

    def report(self):
        reason = self.stop_request["reason"] if self.stop_request else None
        outcome = ("completed" if reason == "coarse_goal_reached" else
                   "ended" if reason in ("operator_stop", "operator_or_parent_stop") else "aborted")
        if self.stop_request and any(k in self.stop_request for k in (
                "controller_error_after_stop", "hard_torque_after_stop",
                "sensor_hard_after_stop", "sensor_ft_error_after_stop")):
            outcome = "aborted"
        return {"schema_version": "right_command_session_report_v1", "run_id": self.run_id,
                "state": self.state, "outcome": outcome, "hardware_backend": self.transport.hardware,
                "commands_sent": self.transport.hardware and self.packet_counts["target"] > 0,
                "any_control_packets_submitted": self.transport.hardware and any(self.packet_counts.values()),
                "control_packets_submitted": dict(self.packet_counts), "stop": self.stop_request,
                "expired_target_ticks": self.expired_target_ticks,
                "state_history": self.history, "calibration": self.calibration,
                "sensor_ft": self.sensor_ft.report() if self.sensor_ft is not None else None,
                "calibration_status": self.calibration_status,
                "calibration_valid_at_end": False, "options": asdict(self.options),
                "goal_selection": self.selection.to_dict(),
                "goal": self.core.goal.result, "first_safety_fault": self.core.safety.first_fault,
                "physical_stop_verified_by_this_delivery": False,
                "note": "SDK submission is not controller acknowledgement or physical arrival"}
