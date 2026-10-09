"""Strict, versioned contracts for read-only execution experiments.

Numeric diagnostic limits are deliberately separate from physical authorization.
No configuration accepted here can enable robot motion.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
from typing import Optional

import numpy as np

PERIOD_NS = 10_000_000


def timestamp(value, name="timestamp") -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer nanosecond timestamp")
    return int(value)


def vector(value, name, size=7, *, positive=False):
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must have {size} finite values")
    if positive and np.any(result <= 0):
        raise ValueError(f"{name} must be positive")
    return result


@dataclass(frozen=True)
class Limits:
    # None means not assessed; it never means an infinite, approved limit.
    torque_hard_nm: Optional[tuple] = None
    torque_soft_nm: Optional[tuple] = None
    torque_release_nm: Optional[tuple] = None
    torque_soft_duration_ms: float = 50.0
    raw_torque_hard_nm: Optional[tuple] = None
    internal_torque_hard_nm: Optional[tuple] = None
    joint_min_rad: Optional[tuple] = None
    joint_max_rad: Optional[tuple] = None
    velocity_rad_s: Optional[tuple] = None
    acceleration_rad_s2: Optional[tuple] = None
    following_error_rad: Optional[tuple] = None
    first_action_error_rad: Optional[tuple] = None
    max_consecutive_limited_ms: Optional[float] = 200.0
    authority: str = "unconfigured"
    # The default preserves historical shadow/replay behavior. Step-only profiles
    # delegate speed/acceleration to the controller; no hardware setup occurs here.
    motion_guard: str = "velocity_acceleration"
    max_command_step_rad: Optional[tuple] = None

    def __post_init__(self):
        if self.authority not in ("unconfigured", "synthetic_test", "dataset_diagnostic", "operator_reviewed"):
            raise ValueError("unknown limit authority")
        if self.motion_guard not in ("velocity_acceleration", "step_reject"):
            raise ValueError("unknown motion guard")
        positive = {"torque_hard_nm", "torque_soft_nm", "raw_torque_hard_nm",
                    "internal_torque_hard_nm", "velocity_rad_s", "acceleration_rad_s2",
                    "following_error_rad", "first_action_error_rad", "max_command_step_rad"}
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if name.endswith(("_nm", "_rad", "_rad_s", "_rad_s2")) and value is not None:
                arr = vector(value, name, positive=name in positive)
                object.__setattr__(self, name, tuple(arr.tolist()))
        if (self.joint_min_rad is None) != (self.joint_max_rad is None):
            raise ValueError("joint bounds require both minimum and maximum")
        if self.joint_min_rad is not None and np.any(np.asarray(self.joint_min_rad) >= self.joint_max_rad):
            raise ValueError("joint minimum must be below maximum")
        trio = [self.torque_release_nm, self.torque_soft_nm, self.torque_hard_nm]
        if any(v is not None for v in trio[:2]):
            if any(v is None for v in trio):
                raise ValueError("soft torque protection requires release, soft and hard limits")
            release, soft, hard = map(np.asarray, trio)
            if np.any(release < 0) or np.any(release >= soft) or np.any(soft >= hard):
                raise ValueError("require 0 <= release < soft < hard for each joint")
        if self.motion_guard == "step_reject":
            if any(v is not None for v in (self.velocity_rad_s, self.acceleration_rad_s2,
                                           self.max_consecutive_limited_ms)):
                raise ValueError("step_reject requires host velocity/acceleration/dwell limits to be null")
        elif self.max_command_step_rad is not None:
            raise ValueError("max_command_step_rad requires step_reject")
        checked = ("torque_soft_duration_ms",) if self.motion_guard == "step_reject" else (
            "torque_soft_duration_ms", "max_consecutive_limited_ms")
        for name in checked:
            if getattr(self, name) is None:
                raise ValueError(f"{name} must be positive")
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")

    def missing_motion(self):
        required = ("joint_min_rad", "joint_max_rad")
        required += (("max_command_step_rad",) if self.motion_guard == "step_reject" else
                     ("velocity_rad_s", "acceleration_rad_s2"))
        required += ("following_error_rad", "first_action_error_rad")
        return [name for name in required if getattr(self, name) is None]

    def missing(self):
        required = ("torque_hard_nm", "torque_soft_nm")
        return [name for name in required if getattr(self, name) is None] + self.missing_motion()


@dataclass(frozen=True)
class Goal:
    goal_id: str = "unselected"
    kind: str = "disabled"  # joint, insertion, or coarse sphere
    verified: bool = False
    target_q_rad: Optional[tuple] = None
    tolerance_q_rad: Optional[tuple] = None
    frame_id: str = ""
    entrance_m: Optional[tuple] = None
    insertion_axis: Optional[tuple] = None
    target_quaternion_xyzw: Optional[tuple] = None
    depth_min_m: Optional[float] = None
    depth_max_m: Optional[float] = None
    lateral_tolerance_m: Optional[float] = None
    angle_tolerance_rad: Optional[float] = None
    max_joint_velocity_rad_s: float = 0.01
    dwell_ms: float = 300.0
    max_sample_gap_ms: float = 20.0
    pose_max_age_ms: float = 20.0
    progress_timeout_s: Optional[float] = None
    progress_epsilon: float = 0.01  # reduction in normalized goal error
    provenance: str = "unconfigured"
    # Coarse completion is explicitly accepted separately from physical verification.
    target_position_m: Optional[tuple] = None
    radius_m: Optional[float] = None
    coarse_completion_accepted: bool = False
    pose_source: str = "external"

    def __post_init__(self):
        if self.kind not in ("disabled", "joint", "insertion", "sphere") or not self.goal_id:
            raise ValueError("invalid goal kind/id")
        if not isinstance(self.verified, bool):
            raise ValueError("goal verified must be boolean")
        if not isinstance(self.coarse_completion_accepted, bool):
            raise ValueError("coarse completion acceptance must be boolean")
        if self.pose_source not in ("external", "m6_ccs40_right_flange_fk"):
            raise ValueError("unknown pose source")
        if self.kind != "sphere" and (self.coarse_completion_accepted or self.pose_source != "external"):
            raise ValueError("coarse completion / automatic flange FK require a sphere goal")
        if self.kind == "sphere":
            if not self.frame_id:
                raise ValueError("sphere requires an explicit coordinate frame")
            object.__setattr__(self, "target_position_m", tuple(vector(self.target_position_m, "target_position_m", 3)))
            if self.radius_m is None or isinstance(self.radius_m, bool) or not math.isfinite(self.radius_m) or self.radius_m <= 0:
                raise ValueError("sphere radius_m must be finite and positive")
            if self.pose_source == "m6_ccs40_right_flange_fk" and self.frame_id != "marvin_m6_ccs40_right_base":
                raise ValueError("flange FK sphere must use the model base frame")
        if self.kind == "joint":
            for name in ("target_q_rad", "tolerance_q_rad"):
                object.__setattr__(self, name, tuple(vector(getattr(self, name), name,
                                                          positive=name == "tolerance_q_rad")))
        if self.kind == "insertion":
            if not self.frame_id:
                raise ValueError("insertion goal requires an explicit coordinate frame")
            for name, size in (("entrance_m", 3), ("insertion_axis", 3), ("target_quaternion_xyzw", 4)):
                arr = vector(getattr(self, name), name, size)
                if name != "entrance_m":
                    if not np.isclose(np.linalg.norm(arr), 1.0, atol=1e-6):
                        raise ValueError(f"{name} must have unit norm")
                object.__setattr__(self, name, tuple(arr))
            for name in ("depth_min_m", "depth_max_m", "lateral_tolerance_m", "angle_tolerance_rad"):
                value = getattr(self, name)
                if value is None or not math.isfinite(value) or value <= 0:
                    raise ValueError(f"{name} must be positive")
            if self.depth_min_m >= self.depth_max_m or self.angle_tolerance_rad > math.pi:
                raise ValueError("invalid insertion depth/angle limits")
        for name in ("max_joint_velocity_rad_s", "dwell_ms", "max_sample_gap_ms", "pose_max_age_ms", "progress_epsilon"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.progress_timeout_s is not None and (not math.isfinite(self.progress_timeout_s) or self.progress_timeout_s <= 0):
            raise ValueError("progress timeout must be positive")


@dataclass(frozen=True)
class ShadowConfig:
    schema_version: str = "readonly_controller_v1"
    mode: str = "shadow"
    allow_motion: bool = False
    query_hz: int = 50
    execute_hz: int = 100
    aggregation: str = "recency"
    decay_ms: float = 80.0
    official_decay: float = 0.01
    max_prediction_age_ms: float = 200.0
    feedback_max_age_ms: float = 20.0
    force_max_age_ms: float = 20.0
    camera_max_age_ms: float = 100.0
    execution_lateness_ms: float = 10.0
    task_timeout_s: float = 180.0
    limits: Limits = field(default_factory=Limits)
    goal: Goal = field(default_factory=Goal)

    def __post_init__(self):
        if self.schema_version != "readonly_controller_v1" or self.mode not in ("shadow", "replay"):
            raise ValueError("only the versioned replay/shadow modes are supported")
        if self.allow_motion is not False:
            raise ValueError("robot motion is prohibited in this implementation")
        if self.query_hz != 50 or self.execute_hz != 100:
            raise ValueError("v1 contract is 50 Hz queries / 100 Hz execution")
        if self.aggregation not in ("newest", "official", "recency"):
            raise ValueError("unknown aggregation strategy")
        for name in ("decay_ms", "max_prediction_age_ms", "feedback_max_age_ms", "force_max_age_ms",
                     "camera_max_age_ms", "execution_lateness_ms", "task_timeout_s"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.official_decay) or self.official_decay < 0:
            raise ValueError("invalid official decay")
        if not isinstance(self.limits, Limits) or not isinstance(self.goal, Goal):
            raise ValueError("limits/goal must be validated contracts")

    @classmethod
    def from_dict(cls, payload):
        values = dict(payload)
        values["limits"] = Limits(**values.get("limits", {}))
        values["goal"] = Goal(**values.get("goal", {}))
        return cls(**values)

    @classmethod
    def read(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text()))

    def to_dict(self):
        return asdict(self)

    def readiness(self):
        return {"motion_possible": False, "hardware_command_backend": "absent",
                "limits_authority": self.limits.authority, "missing_limits": self.limits.missing(),
                "motion_guard": self.limits.motion_guard,
                "host_velocity_guard_enabled": self.limits.velocity_rad_s is not None,
                "controller_limits_readback_verified": False,
                "goal_id": self.goal.goal_id, "goal_configured": self.goal.kind != "disabled",
                "goal_verified": self.goal.verified,
                "coarse_completion_accepted": self.goal.coarse_completion_accepted,
                "goal_pose_source": self.goal.pose_source,
                "physical_stop_verified": False}


@dataclass(frozen=True)
class Feedback:
    timestamp_ns: int
    frame_serial: int
    q_rad: tuple
    dq_rad_s: tuple
    external_nm: tuple
    force_timestamp_ns: int
    bias_valid: bool
    context: str
    error_code: int = 0
    raw_external_nm: Optional[tuple] = None
    internal_nm: Optional[tuple] = None

    def validate(self):
        timestamp(self.timestamp_ns)
        timestamp(self.force_timestamp_ns)
        timestamp(self.frame_serial, "frame_serial")
        if self.frame_serial > 1_000_000:
            raise ValueError("frame serial out of SDK range")
        if not isinstance(self.bias_valid, bool) or not self.context:
            raise ValueError("invalid calibration/context")
        for name in ("q_rad", "dq_rad_s", "external_nm", "raw_external_nm", "internal_nm"):
            if getattr(self, name) is not None:
                vector(getattr(self, name), name)


@dataclass(frozen=True)
class Pose:
    timestamp_ns: int
    frame_id: str
    position_m: tuple
    quaternion_xyzw: tuple

    def validate(self):
        timestamp(self.timestamp_ns)
        vector(self.position_m, "position_m", 3)
        quat = vector(self.quaternion_xyzw, "quaternion_xyzw", 4)
        if not self.frame_id or not np.isclose(np.linalg.norm(quat), 1., atol=1e-6):
            raise ValueError("pose requires frame and unit quaternion")
