"""Pure, hardware-independent dual-arm HUD model and OpenCV renderer.

Uses the simulation HUD's operator-only panel concept, but deliberately does not
copy its task directions, contact thresholds or SAFE/JAM labels onto real data.
"""

import math
from collections import deque
from dataclasses import dataclass
from types import SimpleNamespace as Obj

import cv2
import numpy as np


AXES = ("Fx", "Fy", "Fz", "Mx", "My", "Mz")
FT_AXES = ("Fx", "Fy", "Fz", "Tx", "Ty", "Tz")


def ft_sensor_report(message, expected_frame="right_ft_measurement", *,
                     display_rotation=None, display_frame=None, joint_stamp_ns=0, error=None):
    """Adapt the bridge's already compensated/tared JointState, without scaling or zeroing.

    Mx/My/Mz are internal legacy HUD labels for Tx/Ty/Tz, not a unit conversion.
    Left has no sensor and is explicitly unavailable, never a fabricated zero.
    """
    fields = {"schema_version": "1", "source_kind": "right_ft_sensor",
              "reference_point": "sensor", "wrench_frame": message.header.frame_id,
              "convention_confirmed": "true"}
    level, reason = 0, "FT SENSOR / gravity compensated + software tared"
    try:
        if (len(message.name) != 6 or len(message.effort) != 6
                or set(message.name) != set(FT_AXES)):
            raise ValueError("expected six unique Fx,Fy,Fz,Tx,Ty,Tz names and efforts")
        if not expected_frame or message.header.frame_id != expected_frame:
            raise ValueError("unexpected FT measurement frame")
        values = dict(zip(message.name, message.effort))
        ordered = [float(values[key]) for key in FT_AXES]
        if not all(math.isfinite(x) for x in ordered):
            raise ValueError("non-finite FT measurement")
        fields["measurement_frame"] = message.header.frame_id
        fields.update({"sensor_" + key: str(x) for key, x in zip(FT_AXES, ordered)})
        if error:
            raise ValueError(error)
        if display_rotation is not None:
            r = np.asarray(display_rotation, dtype=float)
            if (r.shape != (3, 3) or not np.isfinite(r).all()
                    or not np.allclose(r.T @ r, np.eye(3), atol=1e-6, rtol=0)
                    or not np.isclose(np.linalg.det(r), 1., atol=1e-6, rtol=0)
                    or not display_frame):
                raise ValueError("invalid display rotation/frame")
            # Rotate axes only; torque remains about the sensor origin (no translation).
            ordered = np.concatenate((r @ ordered[:3], r @ ordered[3:]))
            fields["wrench_frame"] = display_frame
            fields["sensor_to_display_rotation"] = ",".join(str(x) for x in r.flat)
            fields["joint_source_stamp_ns"] = str(joint_stamp_ns)
        elif display_frame and display_frame != expected_frame:
            raise ValueError("changing display frame requires a rotation")
        fields.update({"raw_" + key: str(x) for key, x in zip(AXES, ordered)})
    except (ValueError, TypeError, OverflowError) as error:
        level, reason = 2, "INVALID FT: " + str(error)
    left = {**fields, "wrench_frame": "", "convention_confirmed": "false"}
    statuses = []
    for side, status_level, text, data in (
            ("left", 2, "NO LEFT FT SENSOR", left), ("right", level, reason, fields)):
        statuses.append(Obj(name=side, level=status_level, message=text,
                            values=[Obj(key=k, value=v) for k, v in data.items()]))
    return Obj(header=message.header, status=statuses)


@dataclass
class ArmView:
    valid: bool = False
    message: str = "WAITING for estimate"
    confirmed: bool = False
    frame: str = ""
    raw: object = None
    filtered: object = None
    condition: float = math.nan
    residual: float = math.nan
    trend_delta: object = None
    force_bias: object = None  # Display-only baseline in the source wrench frame.
    zero_stamp_ns: int = 0
    source_kind: str = "joint_estimate"
    reference_point: str = "flange"
    measurement_frame: str = ""
    measurement_raw: object = None
    sensor_to_display_rotation: object = None
    joint_stamp_ns: int = 0


@dataclass(frozen=True)
class RingMapping:
    """Fixed display/task axes expressed in the explicitly named wrench frame.

    Defaults are placeholders; preview is an explicit uncalibrated diagnostic opt-in.
    Camera-relative axes moving in this frame require a future time-synced TF path.
    """
    confirmed: bool = False
    frame: str = ""
    right: tuple = (1.0, 0.0, 0.0)
    up: tuple = (0.0, 1.0, 0.0)
    axial: tuple = (0.0, 0.0, 1.0)
    preview: bool = False  # Explicit diagnostic display, never calibration confirmation.
    yaw_deg: float = 0.0
    force_sign: float = 1.0
    semantics: str = "force"
    lateral_deadzone: float = 0.0
    axial_deadzone: float = 0.0

    def __post_init__(self):
        matrix = np.asarray((self.right, self.up, self.axial), dtype=float)
        if (matrix.shape != (3, 3) or not np.isfinite(matrix).all()
                or not np.allclose(matrix @ matrix.T, np.eye(3), atol=1e-6, rtol=0)):
            raise ValueError("ring axes must be finite, unit length and mutually orthogonal")
        if (self.confirmed or self.preview) and not self.frame.strip():
            raise ValueError("confirmed or preview ring mapping requires an explicit wrench frame")
        if self.semantics not in ("force", "contact_side"):
            raise ValueError("ring semantics must be force or contact_side")
        if self.semantics == "contact_side" and np.linalg.det(matrix) < 0:
            raise ValueError("insertion task axes must form a right-handed rotation, not a reflection")
        if (not math.isfinite(self.yaw_deg) or self.force_sign not in (-1.0, 1.0)
                or not all(math.isfinite(x) and x >= 0 for x in
                           (self.lateral_deadzone, self.axial_deadzone))):
            raise ValueError("invalid task yaw, force sign or display deadzone")

    def rotation(self):
        # Task axes rotate +yaw about aligned +Z; vector components rotate -yaw.
        angle = math.radians(self.yaw_deg)
        c, s = math.cos(angle), math.sin(angle)
        return np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]]) @ np.asarray(
            (self.right, self.up, self.axial))

    def project(self, view):
        if not (view.valid and self.frame == view.frame
                and (self.preview or (self.confirmed and view.confirmed))):
            return None
        bias = np.zeros(3) if view.force_bias is None else view.force_bias
        return self.force_sign * (self.rotation() @ (view.filtered[:3] - bias))

    def arrow(self, projected):
        xy = np.asarray(projected[:2], dtype=float)
        if np.linalg.norm(xy) <= self.lateral_deadzone:
            return np.zeros(2)
        return -xy if self.semantics == "contact_side" else xy.copy()


class ForceBaseline:
    """Manual display tare from a recent stable window, never automatic force removal.

    Stability is NOT proof of no contact. Operator must ensure no external load.
    No raw estimate, controller state or model filter is changed.
    """
    def __init__(self, window_sec=1.0, max_std=0.3, max_gap_sec=0.2):
        if not all(math.isfinite(x) and x > 0 for x in (window_sec, max_std, max_gap_sec)):
            raise ValueError("invalid force zero window/stability parameters")
        self.window_sec, self.max_std, self.max_gap_sec = window_sec, max_std, max_gap_sec
        self.samples = deque()
        self.basis = None
        self.bias = None
        self.stamp_ns = 0

    def clear(self):
        self.bias, self.stamp_ns = None, 0

    def observe(self, view, stamp_ns):
        if stamp_ns < self.stamp_ns:
            self.clear()
        basis = (view.frame, view.confirmed)
        if view.valid and basis != self.basis:
            self.samples.clear()
            self.clear()
            self.basis = basis
        if not view.valid:
            self.samples.clear()
            return
        if self.samples:
            delta = (stamp_ns - self.samples[-1][0]) / 1e9
            if delta == 0:
                return
            if delta < 0 or delta > self.max_gap_sec:
                self.samples.clear()
                self.clear()
        self.samples.append((stamp_ns, view.raw[:3].copy()))
        cutoff = stamp_ns - int(self.window_sec * 1e9)
        while len(self.samples) > 1 and self.samples[0][0] < cutoff:
            self.samples.popleft()

    def candidate(self, view, stamp_ns):
        """Validate and compute without changing the current display baseline."""
        if (not view.valid or (view.frame, view.confirmed) != self.basis
                or len(self.samples) < 20 or self.samples[-1][0] != stamp_ns
                or (self.samples[-1][0] - self.samples[0][0]) / 1e9 < 0.8 * self.window_sec):
            raise ValueError("need a fresh valid stable window; keep unloaded and wait at least 1 s")
        forces = np.array([sample[1] for sample in self.samples])
        if not np.isfinite(forces).all() or np.max(np.std(forces, axis=0)) > self.max_std:
            raise ValueError("force window is unstable; display zero unchanged")
        return np.mean(forces, axis=0)

    def capture(self, view, stamp_ns):
        self.bias = self.candidate(view, stamp_ns)
        self.stamp_ns = stamp_ns

    def attach(self, view):
        if self.bias is not None and (view.frame, view.confirmed) == self.basis:
            view.force_bias = self.bias.copy()
            view.zero_stamp_ns = self.stamp_ns
        else:
            view.force_bias, view.zero_stamp_ns = None, 0


def capture_force_baselines(baselines, views, stamp_ns, sides):
    """All-or-nothing update within the HUD's single-threaded executor callback."""
    candidates = {}
    for side in sides:
        try:
            candidates[side] = baselines[side].candidate(views[side], stamp_ns)
        except ValueError as error:
            raise ValueError(f"{side}: {error}; requested display zeros unchanged") from error
    for side, bias in candidates.items():
        baselines[side].bias = bias
        baselines[side].stamp_ns = stamp_ns


class HudModel:
    def __init__(self, timeout=0.2, smoothing_sec=0.08, future_tolerance=0.05,
                 trend_window_sec=0.8, source_kind="joint_estimate"):
        if source_kind not in ("joint_estimate", "right_ft_sensor"):
            raise ValueError("unsupported HUD source")
        self.source_kind = source_kind
        if (not all(math.isfinite(x) for x in
                    (timeout, smoothing_sec, future_tolerance, trend_window_sec))
                or timeout <= 0 or smoothing_sec < 0 or future_tolerance < 0
                or trend_window_sec <= 0):
            raise ValueError("invalid HUD timing parameters")
        self.timeout = timeout
        self.smoothing_sec = smoothing_sec
        self.future_tolerance = future_tolerance
        self.arms = {}
        self.stamp_ns = 0
        self.received_at = None
        self.sample_seen_at = None
        self.trend_window_sec = trend_window_sec
        self.history = {side: deque() for side in ("left", "right")}
        self.invalidate("WAITING for FT data / bridge software tare" if source_kind == "right_ft_sensor"
                        else "WAITING for estimate")

    def invalidate(self, message):
        self.arms = {side: ArmView(message=message, source_kind=self.source_kind,
                                  reference_point="sensor" if self.source_kind == "right_ft_sensor" else "flange")
                     for side in ("left", "right")}
        for history in self.history.values():
            history.clear()

    def ingest(self, report, now_ros_ns, now_steady):
        stamp = report.header.stamp
        stamp_ns = stamp.sec * 1_000_000_000 + stamp.nanosec
        age = (now_ros_ns - stamp_ns) / 1e9
        self.received_at = now_steady
        if stamp_ns <= 0 or age < -self.future_tolerance or age > self.timeout:
            self.invalidate("STALE / invalid source timestamp")
            return
        if stamp_ns < self.stamp_ns:
            # Allow a later clean sample to recover after a ROS clock reset.
            self.stamp_ns = 0
            self.sample_seen_at = None
            self.invalidate("source clock moved backwards")
            return
        changed = stamp_ns != self.stamp_ns
        dt = (stamp_ns - self.stamp_ns) / 1e9 if self.stamp_ns else 0.0
        if changed:
            self.sample_seen_at = now_steady
        alpha = 1.0 if self.smoothing_sec == 0 else -math.expm1(-dt / self.smoothing_sec)
        statuses = {item.name: item for item in report.status}
        if len(report.status) != 2 or set(statuses) != {"left", "right"}:
            self.invalidate("invalid dual-arm estimate schema")
            return
        for side, status in statuses.items():
            old = self.arms[side]
            values = {item.key: item.value for item in status.values}
            view = ArmView(message=status.message)
            self.arms[side] = view
            try:
                if values.get("schema_version") != "1":
                    raise ValueError("unsupported estimate schema")
                view.confirmed = values.get("convention_confirmed") == "true"
                view.frame = values.get("wrench_frame", "")
                view.source_kind = values.get("source_kind", "joint_estimate")
                view.reference_point = values.get("reference_point", "flange")
                if view.source_kind != self.source_kind:
                    raise ValueError("unexpected HUD source kind")
                # diagnostic_msgs uses ROS 'byte', represented as bytes in Humble Python.
                level = (int.from_bytes(status.level, "little")
                         if isinstance(status.level, bytes) else int(status.level))
                if level >= 2:
                    self.history[side].clear()
                    continue
                view.raw = np.array([float(values["raw_" + key]) for key in AXES])
                if view.source_kind == "joint_estimate":
                    view.condition = float(values["scaled_condition"])
                    view.residual = float(values["relative_residual"])
                    if not all(math.isfinite(v) for v in (view.condition, view.residual)):
                        raise ValueError("non-finite estimator quality")
                if not np.isfinite(view.raw).all():
                    raise ValueError("non-finite estimate")
                if view.source_kind == "right_ft_sensor":
                    view.measurement_frame = values["measurement_frame"]
                    view.measurement_raw = np.array([float(values["sensor_" + key]) for key in FT_AXES])
                    if "sensor_to_display_rotation" in values:
                        view.sensor_to_display_rotation = np.array([
                            float(x) for x in values["sensor_to_display_rotation"].split(",")]).reshape(3, 3)
                        view.joint_stamp_ns = int(values["joint_source_stamp_ns"])
                if view.confirmed and not view.frame:
                    raise ValueError("confirmed convention needs a frame")
                view.valid = True
                same_basis = (old.confirmed == view.confirmed and old.frame == view.frame
                              and old.source_kind == view.source_kind)
                if old.valid and same_basis and 0 <= dt <= self.timeout:
                    view.filtered = (old.filtered + alpha * (view.raw - old.filtered)
                                     if changed else old.filtered.copy())
                else:
                    view.filtered = view.raw.copy()
                    self.history[side].clear()
                history = self.history[side]
                if changed:
                    history.append((stamp_ns / 1e9, float(np.linalg.norm(view.filtered[:3]))))
                    cutoff = stamp_ns / 1e9 - self.trend_window_sec
                    while len(history) > 1 and history[0][0] < cutoff:
                        history.popleft()
                if len(history) > 1 and history[-1][0] - history[0][0] >= self.trend_window_sec * 0.8:
                    view.trend_delta = history[-1][1] - history[0][1]
            except (KeyError, TypeError, ValueError, OverflowError) as error:
                self.arms[side] = ArmView(message="INVALID: " + str(error), source_kind=self.source_kind,
                                          reference_point="sensor" if self.source_kind == "right_ft_sensor" else "flange")
                self.history[side].clear()
        self.stamp_ns = stamp_ns

    def refresh(self, now_ros_ns, now_steady):
        if self.received_at is None:
            return
        age = (now_ros_ns - self.stamp_ns) / 1e9
        if (self.sample_seen_at is None
                or now_steady - self.received_at > self.timeout
                or now_steady - self.sample_seen_at > self.timeout
                or age > self.timeout or age < -self.future_tolerance):
            self.invalidate("STALE - no fresh source sample")


def render_hud(model, source_age, width=1280, height=720, force_scale=20.0, torque_scale=2.0,
               display_mode="ring", mappings=None, show_torque=False, display_arms="both"):
    if display_arms not in ("both", "left", "right"):
        raise ValueError("display_arms must be both, left or right")
    if display_mode not in ("ring", "bars"):
        raise ValueError("display_mode must be ring or bars")
    if (width < 640 or height < 360 or width > 1920 or height > 1080
            or not all(math.isfinite(v) and v > 0 for v in (force_scale, torque_scale))):
        raise ValueError("unsupported HUD size or display range")
    if display_mode == "bars":
        return render_bars(model, source_age, width, height, force_scale, torque_scale,
                           display_arms)
    return render_rings(model, source_age, width, height, force_scale,
                        mappings or {side: RingMapping() for side in model.arms}, show_torque,
                        display_arms)


def render_rings(model, source_age, width, height, force_scale, mappings, show_torque,
                 display_arms="both"):
    canvas = np.full((720, 1280, 3), (24, 19, 15), np.uint8)
    muted, bright = (176, 168, 155), (241, 237, 228)
    cyan, amber, grid = (222, 201, 83), (75, 189, 244), (90, 77, 62)

    def text(value, x, y, scale=0.6, color=bright, thickness=1, centered=False):
        value = str(value)
        if centered:
            x -= cv2.getTextSize(value, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0][0] // 2
        cv2.putText(canvas, value, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, color, thickness, cv2.LINE_AA)

    sides = ("left", "right") if display_arms == "both" else (display_arms,)
    insertion = any(mappings[side].semantics == "contact_side" for side in sides)
    title = "DUAL ARM" if display_arms == "both" else display_arms.upper() + " ARM"
    text("PEG INSERTION / CONTACT SIDE" if insertion else title + " / FORCE RINGS",
         32, 43, 0.85, bright, 2)
    text("FT SENSOR  |  Gravity compensated + software tared" if model.source_kind == "right_ft_sensor" else
         "ESTIMATED  |  Operator aid only - not a force sensor or safety system", 32, 76,
         0.56, amber)
    age = f"{source_age * 1000:.0f} ms" if math.isfinite(source_age) else "--"
    text("SOURCE AGE " + age, 965, 43, 0.55, muted)
    for index, side in enumerate(sides):
        view, mapping = model.arms[side], mappings[side]
        x, cy, radius = (24 + index * 632 if len(sides) == 2 else 336), 380, 130
        cx = x + 304
        cv2.rectangle(canvas, (x, 104), (x + 608, 636), (44, 37, 29), -1)
        color = (cyan if view.confirmed else amber) if view.valid else grid
        cv2.rectangle(canvas, (x, 104), (x + 608, 110), color, -1)
        text(side.upper() + " ARM", x + 20, 147, 0.8, bright, 2)
        badge = ("FT SENSOR" if view.source_kind == "right_ft_sensor" else
                 ("ESTIMATED" if view.confirmed else "UNVERIFIED")) if view.valid else "NO DATA"
        text(badge, x + 388, 146, 0.57, color, 2)
        text("Point: " + view.reference_point + " | Frame: " + (view.frame or "UNCONFIRMED")[:35],
             x + 20, 179, 0.48, muted)
        for r in (radius // 2, radius * 3 // 4, radius):
            cv2.circle(canvas, (cx, cy), r, grid, 1, cv2.LINE_AA)
        if not view.valid:
            # No numeric values, colored fill or force arrows survive invalidation.
            cv2.line(canvas, (cx - 22, cy - 22), (cx + 22, cy + 22), muted, 2, cv2.LINE_AA)
            cv2.line(canvas, (cx - 22, cy + 22), (cx + 22, cy - 22), muted, 2, cv2.LINE_AA)
            text("NO VALID FORCE DATA", cx, 224, 0.66, muted, 2, True)
            for line in range(3):
                text(view.message[line * 55:(line + 1) * 55], cx, 555 + line * 24,
                     0.46, amber, centered=True)
            continue
        corrected = view.filtered[:3] - (0 if view.force_bias is None else view.force_bias)
        force = float(np.linalg.norm(corrected))
        projected = mapping.project(view)
        unit = "N" if view.confirmed else "N*"
        if projected is None:
            # A circular magnitude gauge has NO spatial direction semantics.
            fraction = min(force / force_scale, 1.0)
            # Blend only the gauge's square, not the entire 1280x720 canvas.
            gauge = canvas[cy - radius:cy + radius + 1, cx - radius:cx + radius + 1]
            overlay = gauge.copy()
            cv2.circle(overlay, (radius, radius),
                       max(1, int(radius * math.sqrt(fraction))), cyan, -1)
            cv2.addWeighted(overlay, 0.16, gauge, 0.84, 0, gauge)
            if fraction > 0:
                cv2.ellipse(canvas, (cx, cy), (radius, radius), -90, 0,
                            360 * fraction, cyan, 6, cv2.LINE_AA)
            text("TOTAL FORCE", cx, 224, 0.65, cyan, 2, True)
            text(f"{force:.2f}", cx, cy + 10, 1.35, bright, 2, True)
            text(unit, cx, cy + 46, 0.65, muted, centered=True)
            text("MAGNITUDE ONLY / DIRECTION LOCKED", cx, 551, 0.51, amber, centered=True)
            overflow = force > force_scale
        else:
            right, up, axial = projected
            lateral = float(np.hypot(right, up))
            arrow_right, arrow_up = mapping.arrow(projected)
            arrow_norm = float(np.hypot(arrow_right, arrow_up))
            text(f"TOTAL {force:.2f} {unit}", cx, 224, 0.65, cyan, 2, True)
            cv2.line(canvas, (cx - radius, cy), (cx + radius, cy), grid, 1)
            cv2.line(canvas, (cx, cy - radius), (cx, cy + radius), grid, 1)
            axial_display = 0 if abs(axial) <= mapping.axial_deadzone else abs(axial)
            core_r = 8 + int(25 * math.sqrt(min(axial_display / force_scale, 1.0)))
            cv2.circle(canvas, (cx, cy), core_r, cyan if axial >= 0 else amber, -1, cv2.LINE_AA)
            if arrow_norm > 1e-9:
                scale = radius / max(force_scale, arrow_norm)
                # OpenCV image y grows DOWN. Mapping.up means operator screen UP.
                end = (cx + int(round(arrow_right * scale)), cy - int(round(arrow_up * scale)))
                if end != (cx, cy):
                    cv2.arrowedLine(canvas, (cx, cy), end, bright, 3, cv2.LINE_AA, tipLength=0.2)
                    cv2.circle(canvas, end, 5, bright, -1, cv2.LINE_AA)
                    if mapping.semantics == "contact_side":
                        theta = math.degrees(math.atan2(-arrow_up, arrow_right))
                        cv2.ellipse(canvas, (cx, cy), (radius, radius), 0,
                                    theta - 12, theta + 12, amber, 5, cv2.LINE_AA)
            text(f"Lateral {lateral:.2f} {unit}  |  Axial {axial:+.2f} {unit}", cx, 551,
                 0.59, bright, centered=True)
            if mapping.semantics == "contact_side":
                text("SIDE ESTIMATE / UNCALIBRATED PREVIEW" if mapping.preview else
                     "CONTACT SIDE ESTIMATE - not a motion instruction",
                     cx, 610, 0.46, amber, centered=True)
                text(f"Task yaw {mapping.yaw_deg:+g} deg | sign {mapping.force_sign:+g}",
                     cx, 244, 0.43, muted, centered=True)
            else:
                text("UNCALIBRATED PREVIEW - axes/sign under test" if mapping.preview else
                     "CONTACT FORCE - not a motion instruction", cx, 610, 0.48, amber, centered=True)
            overflow = lateral > force_scale or abs(axial) > force_scale
        if overflow:
            text("DISPLAY RANGE EXCEEDED", cx, 530, 0.42, amber, centered=True)
        if mapping.semantics == "contact_side":
            text(("BRIDGE SOFTWARE TARE" if view.source_kind == "right_ft_sensor" else
                  ("DISPLAY ZERO ACTIVE" if view.zero_stamp_ns else "DISPLAY ZERO NOT SET")) +
                 f" | XY deadzone {mapping.lateral_deadzone:g}", cx, 579,
                 0.48, muted, centered=True)
        else:
            delta = "--" if view.trend_delta is None else f"{view.trend_delta:+.2f} {unit}"
            text(f"Raw Delta |F| / ~{model.trend_window_sec:g}s: {delta}", cx, 579,
                 0.51, muted, centered=True)
        if show_torque and mapping.semantics != "contact_side":
            text(f"|M| {np.linalg.norm(view.filtered[3:]):.3f} " +
                 ("Nm" if view.confirmed else "Nm*"), cx, 244, 0.42, muted, centered=True)
    text(f"Ring full scale: {force_scale:g} force units | display scale only, NOT a safe limit",
         32, 670, 0.54, muted)
    rotated_ft = any(model.arms[side].sensor_to_display_rotation is not None for side in sides)
    text(("Fixed hole directions | FT rotated using same-frame joint feedback"
          if rotated_ft else "Sensor axes move with the wrist. Camera/task directions require a matching mapping.")
         if model.source_kind == "right_ft_sensor" else
         "* Verify units/frame/sign. Preview arrows are NOT calibrated physical directions.",
         32, 701, 0.49, amber)
    return (canvas if (width, height) == (1280, 720)
            else cv2.resize(canvas, (width, height), interpolation=cv2.INTER_AREA))


def render_bars(model, source_age, width=1280, height=720, force_scale=20.0, torque_scale=2.0,
                display_arms="both"):
    """Display ranges control bar lengths, NOT risk or safety thresholds."""
    if (width < 640 or height < 360 or width > 1920 or height > 1080
            or not all(math.isfinite(v) and v > 0 for v in (force_scale, torque_scale))):
        raise ValueError("unsupported HUD size or display range")
    canvas = np.full((720, 1280, 3), (24, 19, 15), np.uint8)
    muted, bright = (176, 168, 155), (241, 237, 228)
    cyan, amber = (222, 201, 83), (75, 189, 244)

    def text(value, x, y, scale=0.6, color=bright, thickness=1):
        cv2.putText(canvas, str(value), (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, color, thickness, cv2.LINE_AA)

    sides = ("left", "right") if display_arms == "both" else (display_arms,)
    title = "DUAL ARM" if display_arms == "both" else display_arms.upper() + " ARM"
    text(title + " / FORCE FEEDBACK", 32, 43, 0.85, bright, 2)
    text("FT SENSOR  |  Gravity compensated + software tared" if model.source_kind == "right_ft_sensor" else
         "ESTIMATED  |  Operator aid only - not a force sensor or safety system", 32, 76,
         0.56, amber)
    age_text = f"{source_age * 1000:.0f} ms" if math.isfinite(source_age) else "--"
    text("SOURCE AGE " + age_text, 965, 43, 0.55, muted)
    for index, side in enumerate(sides):
        view = model.arms[side]
        x = 24 + index * 632 if len(sides) == 2 else 336
        color = cyan if view.valid and view.confirmed else amber
        cv2.rectangle(canvas, (x, 104), (x + 608, 636), (44, 37, 29), -1)
        cv2.rectangle(canvas, (x, 104), (x + 608, 110), color, -1)
        text(side.upper() + " ARM", x + 20, 147, 0.8, bright, 2)
        text(("FT SENSOR" if view.source_kind == "right_ft_sensor" else
              ("ESTIMATED" if view.confirmed else "UNVERIFIED")) if view.valid else "NO DATA",
             x + 388, 146, 0.57, color, 2)
        text("Point: " + view.reference_point + " | Frame: " + (view.frame or "UNCONFIRMED")[:35],
             x + 20, 179, 0.48, muted)
        if not view.valid:
            text("NO VALID FORCE DATA", x + 40, 300, 0.78, amber, 2)
            # Wrap reason text so a long SDK/error message cannot overlap the other arm.
            message = view.message[:150]
            for line in range(3):
                text(message[line * 45:(line + 1) * 45], x + 30, 346 + line * 26, 0.5, amber)
            text("Values hidden; never substituted with zero", x + 30, 520, 0.51, muted)
            continue
        data = view.filtered
        force, torque = np.linalg.norm(data[:3]), np.linalg.norm(data[3:])
        text(f"|F| {force:.2f}", x + 20, 230, 0.9, cyan, 2)
        text(f"|M| {torque:.3f}", x + 310, 230, 0.9, cyan, 2)
        text("N / Nm" if view.confirmed else "N* / Nm*  (SDK unit convention unverified)",
             x + 20, 260, 0.48, muted)
        for i, key in enumerate(FT_AXES if view.source_kind == "right_ft_sensor" else AXES):
            y = 303 + i * 43
            limit = force_scale if i < 3 else torque_scale
            text(key, x + 22, y + 6, 0.62, bright)
            cv2.line(canvas, (x + 95, y), (x + 355, y), (78, 67, 53), 8)
            cv2.line(canvas, (x + 225, y - 11), (x + 225, y + 11), muted, 1)
            end = x + 225 + int(130 * np.clip(data[i] / limit, -1, 1))
            cv2.line(canvas, (x + 225, y), (end, y), cyan, 7)
            text(f"{data[i]:+.3f}", x + 384, y + 6, 0.64, bright)
            if abs(data[i]) > limit:
                text("RANGE", x + 527, y + 5, 0.38, amber)
        text("Bridge compensated + tared input" if view.source_kind == "right_ft_sensor" else
             f"cond {view.condition:.1f} | residual {view.residual:.3f}",
             x + 20, 585, 0.49, muted)
        text("Filtered display | no direction guidance", x + 20, 616, 0.49, muted)
    text("Bars: +/- %.1f force, +/- %.1f torque | display scale only, NOT safe limits" %
         (force_scale, torque_scale), 32, 670, 0.54, muted)
    text("HUD smoothing is display-only. Recording uses the compensated and tared input."
         if model.source_kind == "right_ft_sensor" else
         "* Verify units, frame, sign and tool load before interpreting contact forces.",
         32, 701, 0.52, amber)
    return (canvas if (width, height) == (1280, 720)
            else cv2.resize(canvas, (width, height), interpolation=cv2.INTER_AREA))
