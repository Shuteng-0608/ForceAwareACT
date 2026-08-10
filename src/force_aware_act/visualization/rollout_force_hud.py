"""Causal force-feedback snapshots for rollout HUD rendering.

This module adapts the vendored ``arm_teleop`` force-feedback helpers to a
single policy-rate rollout state. It deliberately does not render frames or
write videos; those side effects belong to the rollout integration layer.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, Optional, Tuple

import mujoco
import numpy as np

from .force_feedback_overlay import (
    ForceFeedbackConfig,
    ForceFeedbackSmoother,
    compute_force_feedback,
    draw_force_feedback_overlay,
    resize_with_aspect_padding,
    trend_label,
)
from .ft_wrench_utils import (
    body_ids,
    compensated_ft_wrench,
    ft_sensor_pose_world,
    ft_sensor_site_id,
    gravity_wrench_sensor_frame,
)


PRIMARY_WRENCH_CHOICES = ("raw", "compensated")
COMPENSATION_MODE_CHOICES = ("none", "gravity")


@dataclass(frozen=True)
class RolloutForceHUDConfig:
    """MuJoCo and wrench semantics needed to build one HUD snapshot."""

    camera_name: str = "cctv_cam"
    force_sensor_name: str = "peg_ft_force"
    torque_sensor_name: str = "peg_ft_torque"
    primary_wrench: str = "compensated"
    compensation_mode: str = "gravity"
    gravity_tool_body_names: Tuple[str, ...] = ("peg_tool",)
    gravity_world: Tuple[float, float, float] = (0.0, 0.0, -9.81)
    gravity_sensor_sign: float = -1.0

    def __post_init__(self) -> None:
        if not self.camera_name:
            raise ValueError("camera_name must not be empty")
        if not self.force_sensor_name or not self.torque_sensor_name:
            raise ValueError("force and torque sensor names must not be empty")
        if self.primary_wrench not in PRIMARY_WRENCH_CHOICES:
            raise ValueError(
                f"primary_wrench must be one of {PRIMARY_WRENCH_CHOICES}, "
                f"got {self.primary_wrench!r}"
            )
        if self.compensation_mode not in COMPENSATION_MODE_CHOICES:
            raise ValueError(
                f"compensation_mode must be one of {COMPENSATION_MODE_CHOICES}, "
                f"got {self.compensation_mode!r}"
            )
        if not self.gravity_tool_body_names:
            raise ValueError("gravity_tool_body_names must not be empty")
        if len(set(self.gravity_tool_body_names)) != len(
            self.gravity_tool_body_names
        ):
            raise ValueError("gravity_tool_body_names must be unique")
        gravity = np.asarray(self.gravity_world, dtype=np.float64)
        if gravity.shape != (3,) or not np.isfinite(gravity).all():
            raise ValueError("gravity_world must be a finite 3-vector")
        if not np.isfinite(self.gravity_sensor_sign):
            raise ValueError("gravity_sensor_sign must be finite")


@dataclass(frozen=True)
class RolloutForceHUDSnapshot:
    """All force values and camera-projected feedback for one policy frame."""

    timestamp: float
    camera_name: str
    raw_wrench: np.ndarray
    gravity_wrench: np.ndarray
    compensated_wrench: np.ndarray
    primary_wrench: np.ndarray
    primary_source_label: str
    smoothed_primary_force: np.ndarray
    raw_force_norm: float
    compensated_force_norm: float
    guidance_right_world: np.ndarray
    guidance_up_world: np.ndarray
    feedback: Dict[str, Any]


@dataclass(frozen=True)
class RolloutForceHUDWrenches:
    """Raw, gravity, compensated, and selected wrench at one instant."""

    raw: np.ndarray
    gravity: np.ndarray
    compensated: np.ndarray
    primary: np.ndarray
    primary_source_label: str


@dataclass(frozen=True)
class RolloutForceHUDIntervalPeak:
    """Force-norm peaks observed directly over one physics interval."""

    start_timestamp: float
    end_timestamp: float
    sample_count: int
    raw_force_norm: float
    raw_timestamp: float
    compensated_force_norm: float
    compensated_timestamp: float
    primary_force_norm: float
    primary_timestamp: float
    primary_source_label: str
    threshold: float
    raw_above_threshold_duration: float
    compensated_above_threshold_duration: float
    raw_excess_force_exposure: float
    compensated_excess_force_exposure: float


class RolloutForceHUDAdapter:
    """Build one causal HUD snapshot per strictly increasing rollout time."""

    def __init__(
        self,
        model,
        force_feedback_config: ForceFeedbackConfig,
        rollout_config: Optional[RolloutForceHUDConfig] = None,
    ) -> None:
        self.model = model
        self.force_feedback_config = force_feedback_config
        self.rollout_config = rollout_config or RolloutForceHUDConfig()

        config = self.rollout_config
        self._force_sensor_site_id = _validated_ft_sensor_site_id(
            model,
            force_sensor_name=config.force_sensor_name,
            torque_sensor_name=config.torque_sensor_name,
        )

        self._gravity_tool_body_ids = body_ids(
            model,
            config.gravity_tool_body_names,
        )
        if len(self._gravity_tool_body_ids) != len(
            config.gravity_tool_body_names
        ):
            missing = [
                name
                for name in config.gravity_tool_body_names
                if mujoco.mj_name2id(
                    model,
                    mujoco.mjtObj.mjOBJ_BODY,
                    name,
                )
                == -1
            ]
            raise ValueError(f"gravity tool bodies are unavailable: {missing}")

        self._camera_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_CAMERA,
            config.camera_name,
        )
        if self._camera_id == -1:
            raise ValueError(f"HUD camera is unavailable: {config.camera_name!r}")

        self._force_smoother = ForceFeedbackSmoother(
            alpha=force_feedback_config.smoothing_alpha
        )
        self._force_norm_history: Deque[Tuple[float, float]] = deque()
        self._last_timestamp: Optional[float] = None
        self._update_count = 0

    @property
    def update_count(self) -> int:
        return self._update_count

    def reset(self) -> None:
        """Clear episode-local smoothing, trend, and timestamp state."""

        self._force_smoother = ForceFeedbackSmoother(
            alpha=self.force_feedback_config.smoothing_alpha
        )
        self._force_norm_history.clear()
        self._last_timestamp = None
        self._update_count = 0

    def update(
        self,
        data,
        raw_wrench,
        timestamp: float,
    ) -> RolloutForceHUDSnapshot:
        """Create one snapshot from the exact state used for a policy frame."""

        timestamp = float(timestamp)
        if not np.isfinite(timestamp):
            raise ValueError("HUD timestamp must be finite")
        if self._last_timestamp is not None and timestamp <= self._last_timestamp:
            raise ValueError(
                "HUD timestamps must be strictly increasing: "
                f"previous={self._last_timestamp} current={timestamp}"
            )

        wrenches = self.resolve_wrenches(data, raw_wrench)
        raw = wrenches.raw
        gravity = wrenches.gravity
        compensated = wrenches.compensated
        primary = wrenches.primary
        source_label = wrenches.primary_source_label
        config = self.rollout_config

        _, sensor_rotation_world = ft_sensor_pose_world(
            data,
            self._force_sensor_site_id,
        )
        if not np.isfinite(sensor_rotation_world).all():
            raise ValueError("FT sensor world rotation is unavailable")
        guidance_right, guidance_up = self._camera_screen_basis_world(data)

        # Resolve every model/data-dependent value before mutating episode-local
        # smoothing and trend state. A failed update must be retryable.
        smoothed_force = self._force_smoother.update(primary[:3])
        smoothed_force_norm = float(np.linalg.norm(smoothed_force))
        trend = self._update_trend(timestamp, smoothed_force_norm)
        feedback = compute_force_feedback(
            force_sensor=smoothed_force,
            config=self.force_feedback_config,
            source_label=source_label,
            R_ws=sensor_rotation_world,
            trend=trend,
            torque_sensor=primary[3:],
            guidance_right_world=guidance_right,
            guidance_up_world=guidance_up,
            guidance_basis_label=f"{config.camera_name}_screen",
        )

        self._last_timestamp = timestamp
        self._update_count += 1
        return RolloutForceHUDSnapshot(
            timestamp=timestamp,
            camera_name=config.camera_name,
            raw_wrench=raw.copy(),
            gravity_wrench=gravity.copy(),
            compensated_wrench=compensated.copy(),
            primary_wrench=primary.copy(),
            primary_source_label=source_label,
            smoothed_primary_force=smoothed_force.copy(),
            raw_force_norm=float(np.linalg.norm(raw[:3])),
            compensated_force_norm=float(np.linalg.norm(compensated[:3])),
            guidance_right_world=guidance_right.copy(),
            guidance_up_world=guidance_up.copy(),
            feedback=feedback,
        )

    def resolve_wrenches(
        self,
        data,
        raw_wrench,
    ) -> RolloutForceHUDWrenches:
        """Resolve display wrenches without changing episode-local HUD state."""

        raw = np.asarray(raw_wrench, dtype=np.float64)
        if raw.shape != (6,) or not np.isfinite(raw).all():
            raise ValueError("raw_wrench must be a finite 6-vector")
        raw = raw.copy()

        config = self.rollout_config
        gravity = gravity_wrench_sensor_frame(
            model=self.model,
            data=data,
            ft_site_id=self._force_sensor_site_id,
            tool_body_ids=self._gravity_tool_body_ids,
            gravity_world=config.gravity_world,
            sensor_sign=config.gravity_sensor_sign,
        )
        if gravity is None:
            raise ValueError("gravity wrench is unavailable")
        gravity = np.asarray(gravity, dtype=np.float64)
        if gravity.shape != (6,) or not np.isfinite(gravity).all():
            raise ValueError("gravity wrench must be a finite 6-vector")

        compensated = compensated_ft_wrench(
            raw_wrench=raw,
            gravity_wrench=gravity,
            compensation_mode=config.compensation_mode,
        )
        if compensated is None:
            raise ValueError(
                "wrench compensation failed for mode "
                f"{config.compensation_mode!r}"
            )
        compensated = np.asarray(compensated, dtype=np.float64)
        if compensated.shape != (6,) or not np.isfinite(compensated).all():
            raise ValueError("compensated wrench must be a finite 6-vector")

        if config.primary_wrench == "compensated":
            primary = compensated
            source_label = "comp"
        else:
            primary = raw
            source_label = "raw"
        return RolloutForceHUDWrenches(
            raw=raw.copy(),
            gravity=gravity.copy(),
            compensated=compensated.copy(),
            primary=primary.copy(),
            primary_source_label=source_label,
        )

    def _update_trend(self, timestamp: float, force_norm: float) -> str:
        window_sec = max(0.0, float(self.force_feedback_config.trend_window_sec))
        self._force_norm_history.append((timestamp, force_norm))
        cutoff = timestamp - window_sec
        while (
            len(self._force_norm_history) > 1
            and self._force_norm_history[0][0] < cutoff
        ):
            self._force_norm_history.popleft()

        if len(self._force_norm_history) < 2:
            return "STABLE"
        old_force = self._force_norm_history[0][1]
        return trend_label(
            force_norm - old_force,
            self.force_feedback_config,
        )

    def _camera_screen_basis_world(self, data) -> Tuple[np.ndarray, np.ndarray]:
        rotation_world_camera = np.asarray(
            data.cam_xmat[self._camera_id],
            dtype=np.float64,
        ).reshape(3, 3)
        if not np.isfinite(rotation_world_camera).all():
            raise ValueError("HUD camera world rotation is unavailable")

        config = self.force_feedback_config
        right_world = (
            rotation_world_camera[:, 0]
            * config.force_guidance_screen_right_sign
        )
        up_world = (
            rotation_world_camera[:, 1]
            * config.force_guidance_screen_up_sign
        )
        return _orthonormalize_task_plane_basis(
            right_world=right_world,
            up_world=up_world,
            insertion_axis_world=config.insertion_axis_world,
        )


class RolloutForceHUDIntervalPeakTracker:
    """Accumulate display-only peaks from direct physics-step observations."""

    def __init__(
        self,
        adapter: RolloutForceHUDAdapter,
        snapshot: RolloutForceHUDSnapshot,
        threshold: float = 40.0,
    ) -> None:
        threshold = float(threshold)
        if not np.isfinite(threshold) or threshold <= 0.0:
            raise ValueError("interval force threshold must be positive and finite")
        self._adapter = adapter
        self._threshold = threshold
        self._start_timestamp = float(snapshot.timestamp)
        self._end_timestamp = float(snapshot.timestamp)
        self._sample_count = 0
        self._raw_peak = (-np.inf, self._start_timestamp)
        self._compensated_peak = (-np.inf, self._start_timestamp)
        self._primary_peak = (-np.inf, self._start_timestamp)
        self._source_label = snapshot.primary_source_label
        self._raw_above_threshold_duration = 0.0
        self._compensated_above_threshold_duration = 0.0
        self._raw_excess_force_exposure = 0.0
        self._compensated_excess_force_exposure = 0.0
        self._previous_raw_norm: Optional[float] = None
        self._previous_compensated_norm: Optional[float] = None
        self._observe_wrenches(
            RolloutForceHUDWrenches(
                raw=snapshot.raw_wrench,
                gravity=snapshot.gravity_wrench,
                compensated=snapshot.compensated_wrench,
                primary=snapshot.primary_wrench,
                primary_source_label=snapshot.primary_source_label,
            ),
            snapshot.timestamp,
        )

    def observe(self, data, raw_wrench, timestamp: float) -> None:
        timestamp = float(timestamp)
        if not np.isfinite(timestamp) or timestamp <= self._end_timestamp:
            raise ValueError(
                "interval peak timestamps must be strictly increasing: "
                f"previous={self._end_timestamp} current={timestamp}"
            )
        self._observe_wrenches(
            self._adapter.resolve_wrenches(data, raw_wrench),
            timestamp,
        )

    def result(self) -> RolloutForceHUDIntervalPeak:
        return RolloutForceHUDIntervalPeak(
            start_timestamp=self._start_timestamp,
            end_timestamp=self._end_timestamp,
            sample_count=self._sample_count,
            raw_force_norm=self._raw_peak[0],
            raw_timestamp=self._raw_peak[1],
            compensated_force_norm=self._compensated_peak[0],
            compensated_timestamp=self._compensated_peak[1],
            primary_force_norm=self._primary_peak[0],
            primary_timestamp=self._primary_peak[1],
            primary_source_label=self._source_label,
            threshold=self._threshold,
            raw_above_threshold_duration=self._raw_above_threshold_duration,
            compensated_above_threshold_duration=(
                self._compensated_above_threshold_duration
            ),
            raw_excess_force_exposure=self._raw_excess_force_exposure,
            compensated_excess_force_exposure=(
                self._compensated_excess_force_exposure
            ),
        )

    def _observe_wrenches(
        self,
        wrenches: RolloutForceHUDWrenches,
        timestamp: float,
    ) -> None:
        if wrenches.primary_source_label != self._source_label:
            raise ValueError("primary wrench source changed within an interval")
        raw_norm = float(np.linalg.norm(wrenches.raw[:3]))
        compensated_norm = float(np.linalg.norm(wrenches.compensated[:3]))
        primary_norm = float(np.linalg.norm(wrenches.primary[:3]))
        elapsed = float(timestamp) - self._end_timestamp
        if self._previous_raw_norm is not None and elapsed > 0.0:
            if self._previous_raw_norm > self._threshold:
                self._raw_above_threshold_duration += elapsed
                self._raw_excess_force_exposure += (
                    self._previous_raw_norm - self._threshold
                ) * elapsed
            if self._previous_compensated_norm > self._threshold:
                self._compensated_above_threshold_duration += elapsed
                self._compensated_excess_force_exposure += (
                    self._previous_compensated_norm - self._threshold
                ) * elapsed
        self._raw_peak = _updated_peak(self._raw_peak, raw_norm, timestamp)
        self._compensated_peak = _updated_peak(
            self._compensated_peak,
            compensated_norm,
            timestamp,
        )
        self._primary_peak = _updated_peak(
            self._primary_peak,
            primary_norm,
            timestamp,
        )
        self._end_timestamp = float(timestamp)
        self._previous_raw_norm = raw_norm
        self._previous_compensated_norm = compensated_norm
        self._sample_count += 1


def draw_rollout_force_hud_rgb(
    frame_rgb: np.ndarray,
    snapshot: RolloutForceHUDSnapshot,
    force_feedback_config: ForceFeedbackConfig,
    interval_peak: RolloutForceHUDIntervalPeak,
    output_width: Optional[int] = None,
    output_height: Optional[int] = None,
) -> np.ndarray:
    """Draw the vendored CCTV HUD and interval audit onto an RGB video frame."""

    import cv2

    frame_rgb = np.asarray(frame_rgb)
    if (
        frame_rgb.ndim != 3
        or frame_rgb.shape[2] != 3
        or frame_rgb.dtype != np.uint8
    ):
        raise ValueError("frame_rgb must be an HxWx3 uint8 array")
    if interval_peak.start_timestamp != snapshot.timestamp:
        raise ValueError("interval peak and HUD snapshot start times must match")
    if interval_peak.primary_source_label != snapshot.primary_source_label:
        raise ValueError("interval peak and HUD snapshot sources must match")
    if (output_width is None) != (output_height is None):
        raise ValueError("output_width and output_height must be provided together")
    if output_width is not None and (output_width <= 0 or output_height <= 0):
        raise ValueError("HUD output dimensions must be positive")

    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    if output_width is not None:
        frame_bgr = resize_with_aspect_padding(
            frame_bgr=frame_bgr,
            target_width=int(output_width),
            target_height=int(output_height),
            padding_color=(0, 0, 0),
        )
    draw_force_feedback_overlay(
        frame_bgr=frame_bgr,
        feedback=snapshot.feedback,
        config=force_feedback_config,
        camera_name=snapshot.camera_name,
    )
    cv2.putText(
        frame_bgr,
        snapshot.camera_name,
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    peak_text = (
        f"interval peak |F| {interval_peak.primary_force_norm:5.1f} N "
        f"({interval_peak.primary_source_label}, n={interval_peak.sample_count})"
    )
    cv2.putText(
        frame_bgr,
        peak_text,
        (15, frame_bgr.shape[0] - 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (235, 235, 235),
        2,
        cv2.LINE_AA,
    )
    return np.ascontiguousarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))


def _orthonormalize_task_plane_basis(
    right_world,
    up_world,
    insertion_axis_world,
) -> Tuple[np.ndarray, np.ndarray]:
    axis = _unit_vector(insertion_axis_world, "insertion axis")
    right = _project_to_task_plane(right_world, axis)
    if right is None:
        raise ValueError("camera right axis is parallel to the insertion axis")

    up = _project_to_task_plane(up_world, axis)
    if up is None or abs(float(np.dot(right, up))) > 0.95:
        up = np.cross(axis, right)
        norm = float(np.linalg.norm(up))
        if norm < 1e-9:
            raise ValueError("could not construct camera up axis in task plane")
        up = up / norm
    else:
        up = up - float(np.dot(up, right)) * right
        norm = float(np.linalg.norm(up))
        if norm < 1e-9:
            raise ValueError("camera screen basis is degenerate")
        up = up / norm
    return right, up


def _updated_peak(
    current: Tuple[float, float],
    value: float,
    timestamp: float,
) -> Tuple[float, float]:
    if not np.isfinite(value):
        raise ValueError("force peak observation must be finite")
    if value > current[0]:
        return float(value), float(timestamp)
    return current


def _project_to_task_plane(vector, axis: np.ndarray) -> Optional[np.ndarray]:
    vector = np.asarray(vector, dtype=np.float64).reshape(3)
    projected = vector - float(np.dot(vector, axis)) * axis
    norm = float(np.linalg.norm(projected))
    if norm < 1e-9:
        return None
    return projected / norm


def _unit_vector(vector, label: str) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64).reshape(3)
    if not np.isfinite(vector).all():
        raise ValueError(f"{label} must be finite")
    norm = float(np.linalg.norm(vector))
    if norm < 1e-9:
        raise ValueError(f"{label} must be non-zero")
    return vector / norm


def _validated_ft_sensor_site_id(
    model,
    force_sensor_name: str,
    torque_sensor_name: str,
) -> int:
    sensor_ids = []
    for label, sensor_name in (
        ("force", force_sensor_name),
        ("torque", torque_sensor_name),
    ):
        sensor_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_SENSOR,
            sensor_name,
        )
        if sensor_id == -1:
            raise ValueError(f"{label} sensor is unavailable: {sensor_name!r}")
        if int(model.sensor_dim[sensor_id]) != 3:
            raise ValueError(f"{label} sensor must have dimension 3: {sensor_name!r}")
        if int(model.sensor_objtype[sensor_id]) != int(mujoco.mjtObj.mjOBJ_SITE):
            raise ValueError(f"{label} sensor must reference a site: {sensor_name!r}")
        sensor_ids.append(int(sensor_id))

    site_ids = [int(model.sensor_objid[sensor_id]) for sensor_id in sensor_ids]
    if site_ids[0] != site_ids[1]:
        raise ValueError(
            "force and torque sensors must reference the same site: "
            f"{force_sensor_name!r}/{torque_sensor_name!r}"
        )

    # Preserve the vendored resolver as the canonical site lookup and audit its
    # result against the stricter rollout contract above.
    site_id = ft_sensor_site_id(
        model,
        force_sensor_name=force_sensor_name,
        torque_sensor_name=torque_sensor_name,
    )
    if site_id != site_ids[0]:
        raise ValueError("force/torque sensor site resolution is inconsistent")
    return site_id
