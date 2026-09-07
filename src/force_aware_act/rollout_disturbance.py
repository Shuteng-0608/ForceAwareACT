"""One-shot, phase-triggered physical XZ force pulses (no policy/controller aid)."""

from __future__ import annotations

import csv
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

DISTURBANCE_VERSION = "fixed_peg_world_force_pulse_v1"
DIRECTIONS = {"xp": (1., 0., 0.), "xm": (-1., 0., 0.),
              "zp": (0., 0., 1.), "zm": (0., 0., -1.)}


@dataclass(frozen=True)
class PulseConfig:
    peak_n: float = 0.0
    duration_s: float = 0.5
    remaining_m: float = 0.025
    direction: str = "xp"

    def __post_init__(self):
        if not math.isfinite(self.peak_n) or not 0 <= self.peak_n <= 10:
            raise ValueError("pulse peak must be finite and in [0, 10] N")
        if not math.isfinite(self.duration_s) or not 0 < self.duration_s <= 2:
            raise ValueError("pulse duration must be finite and in (0, 2] seconds")
        if not math.isfinite(self.remaining_m) or not 0.005 <= self.remaining_m <= 0.05:
            raise ValueError("pulse remaining distance must be in [0.005, 0.05] m")
        if self.direction not in DIRECTIONS:
            raise ValueError("pulse direction must be xp/xm/zp/zm")

    @classmethod
    def from_args(cls, args):
        return cls(args.disturbance_peak_n, args.disturbance_duration_s,
                   args.disturbance_remaining_m, args.disturbance_direction)


class LateralPulse:
    """Apply at the moving socket COM in world coordinates; retain true sensor data.

    Trigger geometry: remaining = -dot(socket_goal - fixed_tip, insertion_axis).
    A trigger is only allowed on the approach side, before terminal hold. Records
    at physics-step resolution include actual peg/socket contact forces, which
    must not be confused with the deliberately applied external force.
    """

    def __init__(self, config: PulseConfig):
        self.config = config
        self.trigger_time = None
        self.release_time = None
        self.trigger_remaining_m = None
        self.trigger_lateral_m = None
        self.applied = np.zeros(3)
        self.impulse_ns = 0.0
        self.trace = []

    def force(self, timestamp, remaining_m, lateral_m, *, allowed=True):
        if (self.config.peak_n > 0 and self.trigger_time is None and allowed
                and math.isfinite(remaining_m) and math.isfinite(lateral_m)
                and 0 <= remaining_m <= self.config.remaining_m):
            self.trigger_time = float(timestamp)
            self.trigger_remaining_m = float(remaining_m)
            self.trigger_lateral_m = float(lateral_m)
        scale = 0.0
        if self.trigger_time is not None:
            elapsed = timestamp - self.trigger_time
            if elapsed >= self.config.duration_s - 1e-10:
                if self.release_time is None:
                    self.release_time = float(timestamp)
            elif elapsed >= 0:
                scale = self.config.peak_n * math.sin(math.pi * elapsed / self.config.duration_s)**2
        self.applied = scale * np.asarray(DIRECTIONS[self.config.direction])
        return self.applied.copy()

    def observe(self, *, timestamp, timestep, remaining_m, lateral_m, force_n,
                contact_force_n, contact_count):
        if self.trigger_time is None:
            return
        self.impulse_ns += float(np.linalg.norm(self.applied)) * timestep
        self.trace.append({
            "time": float(timestamp), "remaining_m": float(remaining_m),
            "lateral_m": float(lateral_m), "sensor_force_n": float(force_n),
            "peg_socket_contact_force_n": float(contact_force_n),
            "peg_socket_contact_count": int(contact_count),
            "applied_fx_n": float(self.applied[0]),
            "applied_fy_n": float(self.applied[1]),
            "applied_fz_n": float(self.applied[2]),
            "released": self.release_time is not None,
        })

    def summary(self, success, success_time):
        triggered = self.trigger_time is not None
        released = self.release_time is not None
        recovered = bool(released and success and success_time is not None
                         and success_time > self.release_time)
        during = [r for r in self.trace if not r["released"]]
        after = [r for r in self.trace if r["released"]]
        def maximum(rows, key):
            return max((r[key] for r in rows), default=None)
        contact_during = any(r["peg_socket_contact_force_n"] > 0.1 for r in during)
        return {
            "version": DISTURBANCE_VERSION, "config": asdict(self.config),
            "enabled": self.config.peak_n > 0,
            "application": "hole_tool_COM_world_xfrc_applied_sin_squared",
            "triggered": triggered, "fully_released": released,
            "trigger_time": self.trigger_time, "release_time": self.release_time,
            "trigger_remaining_m": self.trigger_remaining_m,
            "trigger_lateral_m": self.trigger_lateral_m,
            "delivered_impulse_ns": self.impulse_ns,
            "expected_full_impulse_ns": self.config.peak_n * self.config.duration_s / 2,
            "recovered": recovered,
            "recovery_time_after_release_s": success_time - self.release_time if recovered else None,
            "physical_contact_during_pulse": contact_during,
            "contact_recovery": bool(recovered and contact_during),
            "peak_sensor_force_during_n": maximum(during, "sensor_force_n"),
            "peak_contact_force_during_n": maximum(during, "peg_socket_contact_force_n"),
            "max_lateral_during_m": maximum(during, "lateral_m"),
            "max_lateral_after_m": maximum(after, "lateral_m"),
            "peak_sensor_force_after_n": maximum(after, "sensor_force_n"),
            "trace_rows": len(self.trace),
        }

    def write_trace(self, path: Path):
        if self.trace:
            with path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(self.trace[0]))
                writer.writeheader()
                writer.writerows(self.trace)


class MujocoPulse:
    """Small binding that restores an existing external wrench on every exit."""

    def __init__(self, model, data, config):
        import mujoco
        self.mujoco = mujoco
        self.model, self.data = model, data
        self.pulse = LateralPulse(config)
        self.body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hole_tool")
        self.peg_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "fixed_cylindrical_peg")
        if min(self.body_id, self.peg_id) < 0:
            raise ValueError("physical pulse requires moving socket and fixed peg model")
        self.socket_geoms = set(np.flatnonzero(model.geom_bodyid == self.body_id))
        self.original = data.xfrc_applied[self.body_id].copy()

    def apply(self, task, allowed=True):
        force = self.pulse.force(float(self.data.time),
                                 -task["peg_to_hole_axial_error"],
                                 task["peg_to_hole_lateral_error"], allowed=allowed)
        self.data.xfrc_applied[self.body_id] = self.original
        self.data.xfrc_applied[self.body_id, :3] += force

    def observe(self, task, force_n):
        if self.pulse.trigger_time is None:
            return
        count, force = 0, 0.0
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            if ((c.geom1 == self.peg_id and c.geom2 in self.socket_geoms)
                    or (c.geom2 == self.peg_id and c.geom1 in self.socket_geoms)):
                wrench = np.zeros(6)
                self.mujoco.mj_contactForce(self.model, self.data, i, wrench)
                force += float(np.linalg.norm(wrench[:3]))
                count += 1
        self.pulse.observe(timestamp=self.data.time, timestep=self.model.opt.timestep,
                           remaining_m=-task["peg_to_hole_axial_error"],
                           lateral_m=task["peg_to_hole_lateral_error"], force_n=force_n,
                           contact_force_n=force, contact_count=count)

    def close(self):
        self.data.xfrc_applied[self.body_id] = self.original
