"""Stateful shadow controller and a CPU process independent of CUDA inference.

The only sink is an append-only diagnostic journal. There is no motion backend,
enable flag, SDK setter, or network robot connection in this module.
"""
from __future__ import annotations

from collections import Counter, deque
import json
import multiprocessing as mp
from pathlib import Path
import queue
import time
import uuid

from .contracts import Feedback, PERIOD_NS, Pose, ShadowConfig, timestamp
from .execution import ActionLimiter, ActionUnavailable, TemporalExecutor
from .supervision import GoalMonitor, SafetySupervisor
from .kinematics import POSE_SOURCE, right_flange_pose


class ShadowController:
    def __init__(self, config: ShadowConfig, origin_ns: int, run_id: str):
        self.config, self.origin_ns, self.run_id = config, timestamp(origin_ns), run_id
        self.executor = TemporalExecutor(origin_ns, run_id, config)
        self.safety = SafetySupervisor(config)
        self.limiter = ActionLimiter(config.limits)
        self.goal = GoalMonitor(config.goal)
        self.camera_ns = {}
        self.pose = None
        self._feedback_fk_pose = None
        self._coarse_reached_pending = False
        self.state = "WAITING"
        self.started_ns = None
        self.stop_request = None
        self.counters = Counter()
        # Bounded diagnostic timings only; never used for control decisions.
        self._compute_ns = {name: deque(maxlen=100_000) for name in ("fk", "goal_update")}
        self._compute_counts = Counter()

    def _record_compute(self, name, started_ns):
        self._compute_ns[name].append(time.perf_counter_ns()-started_ns)
        self._compute_counts[name] += 1

    def _goal_update(self, feedback, now_ns, pose):
        started = time.perf_counter_ns()
        try:
            return self.goal.update(feedback, now_ns, pose)
        finally:
            self._record_compute("goal_update", started)

    def stop(self, reason, now_ns, *, outcome="ABORTED_SHADOW", detail=None):
        # A late safety fault must override a previously requested normal finish.
        details=detail if isinstance(detail,dict) else {}
        priority=(4 if reason=="operator_stop" else details.get("severity",2 if outcome=="ABORTED_SHADOW" else 0))
        if self.stop_request is None or priority>self.stop_request.get("priority",0):
            first=self.stop_request["first_reason"] if self.stop_request else reason
            self.stop_request = {"reason": reason, "requested_ns": int(now_ns),
                                 "first_reason":first,"priority":priority,
                                 "physical_stop_status": "not_attempted_readonly", "detail": detail}
            self.state = outcome
            self.executor.close()

    def _sync_fault(self, now_ns):
        if self.safety.fault:
            self.stop(self.safety.fault["reason"], now_ns, detail=self.safety.fault)

    def feedback(self, value: Feedback, now_ns):
        fresh = self.safety.ingest(value, now_ns, preparing=self.state == "WAITING")
        if fresh and self.safety.fault is None and self.config.goal.pose_source == POSE_SOURCE:
            # FK uses only this actual feedback sample, with its original timestamp.
            try:
                started = time.perf_counter_ns()
                try:
                    self._feedback_fk_pose = right_flange_pose(value.q_rad, value.timestamp_ns)
                finally:
                    self._record_compute("fk", started)
                if self.state == "RUNNING_SHADOW" and not self._coarse_reached_pending:
                    # Observe every accepted actual frame, including a brief exit
                    # and re-entry between 100 Hz ticks. Safety wins on the tick.
                    result = self._goal_update(value, now_ns, self._feedback_fk_pose)
                    self._coarse_reached_pending = result["status"] == "COARSE_REACHED"
            except (ValueError, TypeError) as exc:
                self.safety.trip("goal_fk_invalid", now_ns, detail=str(exc))
        self._sync_fault(now_ns)
        return fresh

    def camera(self, name, received_ns, now_ns):
        timestamp(received_ns); timestamp(now_ns)
        if name not in ("d455", "d405") or received_ns > now_ns:
            self.safety.trip("invalid_camera_timestamp", now_ns)
        elif name in self.camera_ns and received_ns < self.camera_ns[name]:
            self.safety.trip("camera_time_reversed", now_ns)
        else:
            self.camera_ns[name] = received_ns
        self._sync_fault(now_ns)

    def prediction(self, actions, **metadata):
        if self.stop_request:
            self.counters["predictions_rejected_after_stop"] += 1
            return False
        try:
            self.executor.submit(actions, **metadata)
        except (ValueError, TypeError) as exc:
            self.safety.trip("invalid_prediction", metadata["now_ns"], detail=str(exc))
            self._sync_fault(metadata["now_ns"])
            return False
        self.counters["predictions_accepted"] += 1
        return True

    def tick(self, tick_ns, now_ns=None):
        now_ns = tick_ns if now_ns is None else now_ns
        self.executor.grid_index(tick_ns)
        timestamp(now_ns)
        row = {"kind": "execution_tick", "run_id": self.run_id, "tick_ns": tick_ns,
               "evaluated_ns": now_ns, "commands_sent": False,
               "motion_possible": False, "proposal": None, "constrained_proposal": None}
        if self.state == "WAITING":
            # Only shadow execution auto-starts; no robot mode/enable call exists.
            latest=self.safety.latest
            ready=(latest is not None and len(self.camera_ns)==2
                   and 0 <= now_ns-latest.timestamp_ns <= self.config.feedback_max_age_ms*1e6
                   and 0 <= now_ns-latest.force_timestamp_ns <= self.config.force_max_age_ms*1e6
                   and all(0 <= now_ns-t <= self.config.camera_max_age_ms*1e6 for t in self.camera_ns.values())
                   and any(c.available_ns <= tick_ns and 0 <= tick_ns-c.observed_ns <= self.config.max_prediction_age_ms*1e6
                           for c in self.executor.chunks))
            if ready:
                self.started_ns, self.state = now_ns, "RUNNING_SHADOW"
            elif now_ns - self.origin_ns > 30_000_000_000:
                self.stop("startup_timeout", now_ns)
        if self.state == "RUNNING_SHADOW":
            self.safety.poll(now_ns)
            if now_ns - self.started_ns >= self.config.task_timeout_s * 1e9:
                self.safety.trip("task_timeout", now_ns)
            if now_ns < tick_ns or now_ns - tick_ns > self.config.execution_lateness_ms * 1e6:
                self.safety.trip("execution_deadline", now_ns)
            if any(not 0 <= now_ns - t <= self.config.camera_max_age_ms * 1e6 for t in self.camera_ns.values()):
                self.safety.trip("camera_timeout", now_ns)
            self._sync_fault(now_ns)
        if self.state == "RUNNING_SHADOW":
            goal_pose = self._feedback_fk_pose if self.config.goal.pose_source == POSE_SOURCE else self.pose
            goal = dict(self.goal.result) if self._coarse_reached_pending else self._goal_update(self.safety.latest, now_ns, goal_pose)
            row["goal"] = goal
            if goal["status"] in ("DEPTH_EXCEEDED", "STALLED"):
                self.safety.trip("goal_" + goal["status"].lower(), now_ns)
                self._sync_fault(now_ns)
            elif goal["status"] == "REACHED":
                self.stop("goal_reached", now_ns, outcome="COMPLETED_SHADOW")
            elif goal["status"] == "COARSE_REACHED":
                self.stop("coarse_goal_reached", now_ns, outcome="COMPLETED_SHADOW", detail=goal)
        if self.state == "RUNNING_SHADOW":
            try:
                proposal = self.executor.sample(tick_ns, now_ns=now_ns)
                constrained = self.limiter.apply(proposal["action_rad"], self.safety.latest, tick_ns)
                row["proposal"], row["constrained_proposal"] = proposal, constrained
                if constrained.get("fault"):
                    self.safety.trip(constrained["fault"], now_ns)
                self._sync_fault(now_ns)
                if self.stop_request:
                    row["constrained_proposal"] = None
                else:
                    self.counters["proposals"] += 1
                    self.counters["limited"] += bool(constrained["limited"])
                    self.counters["fully_constrained_proposals"] += constrained["action_rad"] is not None
            except (ActionUnavailable, ValueError) as exc:
                self.safety.trip("action_unavailable_or_invalid", now_ns, detail=str(exc))
                self._sync_fault(now_ns)
        self.counters["ticks"] += 1
        row.update(state=self.state, stop_request=self.stop_request)
        return row

    def report(self):
        import numpy as np
        timings = {}
        for name, values in self._compute_ns.items():
            a = np.asarray(values, dtype=float)/1e6
            timings[name] = {"total_calls": self._compute_counts[name], "retained_samples": len(a),
                             "mean_ms": float(a.mean()) if len(a) else None,
                             "p99_ms": float(np.percentile(a, 99)) if len(a) else None,
                             "max_ms": float(a.max()) if len(a) else None}
        return {"schema_version": "readonly_controller_report_v1", "run_id": self.run_id,
                "origin_ns": self.origin_ns,
                "state": self.state, "commands_sent": False, "motor_control_api_calls": 0,
                "physical_stop_confirmed": False, "stop_request": self.stop_request,
                "first_safety_fault":self.safety.first_fault,
                "counters": dict(self.counters), "fresh_feedback": self.safety.new_frames,
                "duplicate_feedback": self.safety.duplicate_frames,
                "startup_stale_feedback_dropped": self.safety.startup_stale_dropped,
                "startup_feedback_gaps": self.safety.startup_gaps,
                "readiness": self.config.readiness(), "goal": self.goal.result,
                "config": self.config.to_dict(), "compute_timing": timings}


def _shadow_worker(config_dict, origin_ns, run_id, inbox, overflow, ready, output):
    config = ShadowConfig.from_dict(config_dict)
    controller = ShadowController(config, origin_ns, run_id)
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    stats = Counter(); lateness = []; feedback_delay = []; start = time.monotonic_ns()
    next_tick = origin_ns + ((start - origin_ns) // PERIOD_NS + 1) * PERIOD_NS
    end_reason = "normal_readonly_end"
    last_message_ns=start
    # Buffered journal is the only sink. No control SDK is imported here.
    with (output / "events.jsonl").open("w", buffering=262144) as events, (output / "execution.jsonl").open("w", buffering=262144) as ticks:
        ready.set()
        try:
            done = False
            while not done:
                now = time.monotonic_ns()
                if overflow.is_set():
                    controller.stop("ipc_overflow", now)
                try:
                    message = inbox.get(timeout=max(0., min(.003, (next_tick-now)/1e9)))
                except queue.Empty:
                    message = None
                now = time.monotonic_ns()
                if message is not None:
                    last_message_ns=now
                    kind = message["kind"]; stats[kind] += 1
                    message["monitor_received_ns"] = now
                    events.write(json.dumps(message, allow_nan=False) + "\n")
                    if kind == "feedback":
                        value = Feedback(**message["value"])
                        feedback_delay.append((now-value.timestamp_ns)/1e6)
                        controller.feedback(value, now)
                    elif kind == "camera":
                        controller.camera(message["name"], message["received_ns"], now)
                    elif kind == "prediction":
                        controller.prediction(message["actions"], now_ns=now, **message["metadata"])
                    elif kind == "pose":
                        controller.pose = Pose(**message["value"])
                    elif kind == "fault":
                        controller.stop(message["reason"], now)
                    elif kind == "end":
                        done = True
                    else:
                        raise ValueError("unknown IPC message")
                if now >= next_tick:
                    # Skip missed grid points; never issue a catch-up burst.
                    tick = origin_ns + (now-origin_ns)//PERIOD_NS * PERIOD_NS
                    stats["skipped_ticks"] += max(0, (tick-next_tick)//PERIOD_NS)
                    lateness.append((now-next_tick)/1e6)
                    if now-next_tick > config.execution_lateness_ms*1e6 and controller.state == "RUNNING_SHADOW":
                        controller.stop("execution_deadline", now)
                    ticks.write(json.dumps(controller.tick(tick, now), allow_nan=False) + "\n")
                    next_tick = tick + PERIOD_NS
                if controller.started_ns is not None and now-last_message_ns>2_000_000_000:
                    # Parent crash: record the already-latched watchdog and exit
                    # cleanly without waiting forever for an IPC end marker.
                    end_reason="upstream_disappeared"
                    controller.stop(end_reason,now)
                    done=True
        except Exception as exc:
            end_reason = "monitor_exception"
            controller.stop(end_reason, time.monotonic_ns(), detail=repr(exc))
        finally:
            if controller.stop_request is None:
                controller.stop(end_reason, time.monotonic_ns(), outcome="ENDED_SHADOW")
    import numpy as np
    def quantiles(values):
        return {"p50": float(np.percentile(values, 50)), "p99": float(np.percentile(values, 99)),
                "max": float(max(values))} if values else None
    report = controller.report()
    report.update(elapsed_s=(time.monotonic_ns()-start)/1e9, monitor_pid=__import__("os").getpid(),
                  messages=dict(stats), ipc_overflow=overflow.is_set(),
                  scheduled_tick_lateness_ms=quantiles(lateness), feedback_delivery_ms=quantiles(feedback_delay))
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


class ShadowProcess:
    """Bounded IPC; overflow latches a fault rather than blocking acquisition."""
    def __init__(self, config: ShadowConfig, output: Path):
        if Path(output).exists():
            raise FileExistsError("use a fresh controller output directory")
        self.origin_ns, self.run_id = time.monotonic_ns(), str(uuid.uuid4())
        self.closed = False
        ctx = mp.get_context("spawn")
        self.inbox = ctx.Queue(maxsize=4096)
        self.overflow, self.ready = ctx.Event(), ctx.Event()
        self.process = ctx.Process(target=_shadow_worker, args=(config.to_dict(), self.origin_ns,
                                   self.run_id, self.inbox, self.overflow, self.ready, str(output)))
        self.process.start()
        if not self.ready.wait(10):
            self.process.terminate(); self.process.join(5)
            raise RuntimeError("shadow monitor failed to initialize")

    def publish(self, kind, **payload):
        if self.closed:
            return
        if not self.process.is_alive():
            raise RuntimeError("shadow monitor exited")
        try:
            self.inbox.put_nowait({"kind": kind, **payload})
        except queue.Full:
            self.overflow.set()
            raise RuntimeError("shadow IPC overflow")

    def close(self):
        self.closed = True
        if self.process.is_alive():
            try:
                self.inbox.put({"kind": "end"}, timeout=2)
            except queue.Full:
                self.overflow.set()
            self.process.join(10)
        if self.process.is_alive():
            self.process.terminate(); self.process.join(5)
            raise RuntimeError("shadow monitor shutdown timed out")
        # The child has exited. A crashed consumer may leave the feeder blocked
        # on a full pipe; joining that feeder would hang hardware cleanup.
        # Normal shutdown already waited for the child's end marker/report.
        self.inbox.cancel_join_thread()
        self.inbox.close()
        if self.process.exitcode:
            raise RuntimeError(f"shadow monitor exit code {self.process.exitcode}")
