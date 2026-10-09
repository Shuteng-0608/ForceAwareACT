"""Feedback-based safety and goal decisions, independent of policy predictions."""
from __future__ import annotations

import math
import numpy as np

from .contracts import Feedback, Goal, Pose, ShadowConfig, timestamp


class SafetySupervisor:
    def __init__(self, config: ShadowConfig):
        self.config = config
        self.latest = None
        self.fault = None
        self.first_fault = None
        self.soft_since = [None] * 7
        self.duplicate_frames = 0
        self.new_frames = 0
        self.startup_stale_dropped = 0
        self.startup_gaps = 0

    def trip(self, reason, now_ns, **details):
        severity=3 if reason in {"external_torque_hard","raw_torque_hard","internal_torque_hard",
                                 "controller_error","measured_joint_limit","measured_velocity_limit"} else 2
        if self.fault is None or severity > self.fault["severity"]:
            self.fault = {"reason": reason, "detected_ns": int(now_ns), "severity":severity, **details}
            if self.first_fault is None:self.first_fault=dict(self.fault)
        return self.fault

    def ingest(self, feedback: Feedback, now_ns, *, preparing=False):
        timestamp(now_ns)
        try:
            feedback.validate()
        except (ValueError, TypeError) as exc:
            self.trip("invalid_feedback", now_ns, detail=str(exc)); return False
        if feedback.error_code:
            self.trip("controller_error", now_ns, code=feedback.error_code); return False
        if not feedback.bias_valid:
            self.trip("calibration_invalid", now_ns); return False
        if not 0 <= now_ns - feedback.timestamp_ns <= self.config.feedback_max_age_ms * 1e6:
            if preparing and now_ns >= feedback.timestamp_ns:
                self.startup_stale_dropped += 1
                return False
            self.trip("feedback_stale_or_future", now_ns); return False
        if not 0 <= now_ns - feedback.force_timestamp_ns <= self.config.force_max_age_ms * 1e6:
            if preparing and now_ns >= feedback.force_timestamp_ns:
                self.startup_stale_dropped += 1
                return False
            self.trip("force_stale_or_future", now_ns); return False
        old = self.latest
        # Keep the hard protection path active after an earlier soft/data fault.
        # Latching prohibits new actions; it must not hide escalation evidence.
        if self.fault is not None:
            if old is None or old.context==feedback.context:
                self._hard_checks(feedback,now_ns)
            return False
        if old is not None:
            if old.context != feedback.context:
                self.trip("calibration_context_changed", now_ns); return False
            if feedback.frame_serial == old.frame_serial:
                self.duplicate_frames += 1
                if not preparing:self.poll(now_ns)
                return False
            if feedback.timestamp_ns <= old.timestamp_ns or feedback.force_timestamp_ns < old.force_timestamp_ns:
                self.trip("feedback_time_reversed", now_ns); return False
            if feedback.frame_serial < old.frame_serial and not (old.frame_serial > 900000 and feedback.frame_serial < 100000):
                self.trip("controller_frame_reset", now_ns); return False
            if feedback.timestamp_ns - old.timestamp_ns > self.config.feedback_max_age_ms * 1e6:
                if not preparing:
                    self.trip("feedback_gap", now_ns); return False
                self.startup_gaps += 1
                self.soft_since = [None] * 7
        self.latest = feedback
        self.new_frames += 1
        limits = self.config.limits
        if self._hard_checks(feedback,now_ns):return True
        # A repeated force timestamp cannot advance the dwell timer.
        new_force = old is None or feedback.force_timestamp_ns > old.force_timestamp_ns
        if limits.torque_soft_nm is not None and new_force:
            for j, absolute in enumerate(np.abs(feedback.external_nm)):
                if absolute >= limits.torque_soft_nm[j] and self.soft_since[j] is None:
                    self.soft_since[j] = feedback.force_timestamp_ns
                elif absolute <= limits.torque_release_nm[j]:
                    self.soft_since[j] = None
                if self.soft_since[j] is not None and feedback.force_timestamp_ns - self.soft_since[j] >= limits.torque_soft_duration_ms * 1e6:
                    self.trip("external_torque_sustained", now_ns, joint=j + 1,
                              value_nm=float(feedback.external_nm[j]),
                              threshold_nm=float(limits.torque_soft_nm[j]),
                              duration_ms=(feedback.force_timestamp_ns-self.soft_since[j])/1e6)
                    return True
        return True

    def _hard_checks(self,feedback,now_ns):
        limits=self.config.limits
        for name, values, threshold in (
            ("external_torque_hard", feedback.external_nm, limits.torque_hard_nm),
            ("raw_torque_hard", feedback.raw_external_nm, limits.raw_torque_hard_nm),
            ("internal_torque_hard", feedback.internal_nm, limits.internal_torque_hard_nm),
        ):
            if threshold is None:
                continue
            if values is None:
                self.trip("required_torque_missing", now_ns, channel=name); return True
            tripped = np.flatnonzero(np.abs(values) >= threshold)
            if len(tripped):
                j = int(tripped[0])
                self.trip(name, now_ns, joint=j + 1, value_nm=float(values[j]),
                          threshold_nm=float(threshold[j]), frame_serial=feedback.frame_serial)
                return True
        q = np.asarray(feedback.q_rad)
        if limits.joint_min_rad is not None and (np.any(q < limits.joint_min_rad) or np.any(q > limits.joint_max_rad)):
            self.trip("measured_joint_limit", now_ns); return True
        if limits.velocity_rad_s is not None and np.any(np.abs(feedback.dq_rad_s) > limits.velocity_rad_s):
            self.trip("measured_velocity_limit", now_ns); return True
        return False

    def poll(self, now_ns):
        timestamp(now_ns)
        if self.latest is None:
            return self.trip("feedback_missing", now_ns)
        if not 0 <= now_ns - self.latest.timestamp_ns <= self.config.feedback_max_age_ms * 1e6:
            self.trip("feedback_timeout", now_ns)
        if not 0 <= now_ns - self.latest.force_timestamp_ns <= self.config.force_max_age_ms * 1e6:
            self.trip("force_timeout", now_ns)
        return self.fault


class GoalMonitor:
    """Dwell requires fresh actual samples. Endpoint candidates stay unverified."""
    def __init__(self, goal: Goal):
        self.goal = goal
        self.inside_since = None
        self.last_sample_ns = None
        self.last_pose_ns = None
        self.best_error = math.inf
        self.last_progress_ns = None
        self.result = {"status": "UNCONFIGURED" if goal.kind == "disabled" else "RUNNING",
                       "goal_id": goal.goal_id, "verified": goal.verified}

    def update(self, feedback: Feedback, now_ns: int, pose: Pose | None = None):
        goal = self.goal
        if goal.kind == "disabled":
            return dict(self.result)
        sample_ns = feedback.timestamp_ns
        if now_ns < sample_ns or now_ns - sample_ns > goal.max_sample_gap_ms * 1e6:
            self.inside_since = None
            return {**self.result, "status": "UNCONFIRMED", "reason": "stale_goal_feedback"}
        if self.last_sample_ns is not None and sample_ns < self.last_sample_ns:
            self.inside_since = None
            self.result = {**self.result, "status": "UNCONFIRMED", "dwell_ms": 0,
                           "reason": "goal_feedback_time_reversed"}
            return dict(self.result)
        if self.last_sample_ns is not None and sample_ns == self.last_sample_ns:
            return dict(self.result)
        if self.last_sample_ns is not None and sample_ns - self.last_sample_ns > goal.max_sample_gap_ms * 1e6:
            self.inside_since = None
        self.last_sample_ns = sample_ns
        metrics = {}
        if goal.kind == "joint":
            error = np.abs(np.asarray(feedback.q_rad) - goal.target_q_rad)
            normalized = float(np.max(error / goal.tolerance_q_rad))
            inside = normalized <= 1
            metrics["joint_error_rad"] = error.tolist()
        else:
            try:
                if pose is None:
                    raise ValueError("actual TCP pose unavailable")
                pose.validate()
                if pose.frame_id != goal.frame_id or not 0 <= now_ns - pose.timestamp_ns <= goal.pose_max_age_ms * 1e6:
                    raise ValueError("TCP pose frame/age mismatch")
            except ValueError as exc:
                self.inside_since = None
                self.result = {"goal_id": goal.goal_id, "verified": goal.verified,
                               "status": "UNCONFIRMED", "reason": str(exc)}
                return dict(self.result)
            if self.last_pose_ns is not None and pose.timestamp_ns <= self.last_pose_ns:
                if pose.timestamp_ns < self.last_pose_ns:
                    self.inside_since = None
                return {**self.result, "status": "UNCONFIRMED", "reason": "TCP pose not advancing"}
            if self.last_pose_ns is not None and pose.timestamp_ns - self.last_pose_ns > goal.max_sample_gap_ms * 1e6:
                self.inside_since = None
            self.last_pose_ns = pose.timestamp_ns
            sample_ns = min(sample_ns, pose.timestamp_ns)
            if goal.kind == "sphere":
                distance = float(np.linalg.norm(np.asarray(pose.position_m)-goal.target_position_m))
                normalized = distance / goal.radius_m
                inside = distance <= goal.radius_m
                metrics.update(distance_m=distance, radius_m=goal.radius_m, inside=inside,
                               measured_position_m=list(pose.position_m), target_position_m=list(goal.target_position_m),
                               completion_basis="coarse_position_sphere", pose_source=goal.pose_source,
                               coarse_completion_accepted=goal.coarse_completion_accepted,
                               joint_velocity_gate_applied=False)
            else:
                delta = np.asarray(pose.position_m) - goal.entrance_m
                depth = float(delta @ goal.insertion_axis)
                lateral = float(np.linalg.norm(delta - depth * np.asarray(goal.insertion_axis)))
                angle = float(2 * math.acos(np.clip(abs(np.dot(pose.quaternion_xyzw, goal.target_quaternion_xyzw)), 0, 1)))
                metrics.update(depth_m=depth, lateral_error_m=lateral, angle_error_rad=angle)
                if depth > goal.depth_max_m:
                    self.result = {"status": "DEPTH_EXCEEDED", "goal_id": goal.goal_id,
                                   "verified": goal.verified, **metrics}
                    return dict(self.result)
                normalized = max(lateral / goal.lateral_tolerance_m, angle / goal.angle_tolerance_rad,
                                 max(0., goal.depth_min_m - depth) / goal.depth_min_m)
                inside = (goal.depth_min_m <= depth <= goal.depth_max_m
                          and lateral <= goal.lateral_tolerance_m and angle <= goal.angle_tolerance_rad)
        if self.last_progress_ns is None or normalized <= self.best_error - goal.progress_epsilon:
            self.best_error, self.last_progress_ns = normalized, now_ns
        slow = bool(np.max(np.abs(feedback.dq_rad_s)) <= goal.max_joint_velocity_rad_s)
        # A coarse sphere defines position occupancy. Redundant joint motion must
        # not introduce an undocumented extra gate into this accepted rule.
        if inside and (slow or goal.kind == "sphere"):
            self.inside_since = sample_ns if self.inside_since is None else self.inside_since
        else:
            self.inside_since = None
        dwell = 0 if self.inside_since is None else (sample_ns - self.inside_since) / 1e6
        status = "RUNNING" if self.inside_since is None else "GOAL_CANDIDATE"
        if dwell >= goal.dwell_ms:
            if goal.kind == "sphere":
                status = "COARSE_REACHED" if goal.coarse_completion_accepted else "UNVERIFIED_CANDIDATE"
            else:
                status = "REACHED" if goal.verified else "UNVERIFIED_CANDIDATE"
        if goal.progress_timeout_s is not None and not inside and now_ns - self.last_progress_ns >= goal.progress_timeout_s * 1e9:
            status = "STALLED"
        self.result = {"status": status, "goal_id": goal.goal_id, "verified": goal.verified,
                       "dwell_ms": dwell, "normalized_error": normalized, "slow": slow, **metrics}
        return dict(self.result)
