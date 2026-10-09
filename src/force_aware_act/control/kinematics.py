"""Pure mathematical right-arm M6 CCS 4.0 FK; no SDK or hardware imports.

Actual q [7] radians -> flange transform [4,4] (translation mm), then Pose
(meters, quaternion xyzw) in the SDK right-arm base, with the SAME timestamp.
The fixed geometry matches the checksum-validated acquisition model, not a
calibrated peg-tip TCP. Joint order is Joint1_R ... Joint7_R.
"""
from __future__ import annotations

import numpy as np

from .contracts import Pose, vector

MODEL_SHA256 = "ce20b1a80974c3ed58cdac4973152625096c8ad93a779e5fd4108c26ce19bd52"
FRAME_ID = "marvin_m6_ccs40_right_base"
POSE_SOURCE = "m6_ccs40_right_flange_fk"


def _chain():
    base = np.tile(np.eye(4), (7, 1, 1))
    rx = np.array([[1., 0, 0], [0, 0, -1], [0, 1, 0]])
    base[0, 2, 3] = 174.5
    base[1, :3, :3] = rx
    base[2, :3, :3] = rx.T
    base[3, :3, :3] = rx
    base[3, :3, 3] = [18., 0, 287.]
    base[4, :3, :3] = rx.T
    base[4, :3, 3] = [-18., 314., 0]
    base[5, :3, :3] = rx
    base[6, :3, :3] = [[0, 0, 1], [1, 0, 0], [0, 1, 0]]
    flange = np.eye(4)
    flange[:3, :3] = [[0, 0, 1], [0, -1, 0], [1, 0, 0]]
    flange[0, 3] = 95.
    base.setflags(write=False)
    flange.setflags(write=False)
    return base, flange


_BASE, _FLANGE = _chain()


def right_flange_transform(q_rad):
    """Single measured sample, no timestamps, filtering, interpolation or future data."""
    q = vector(q_rad, "right measured q_rad")
    result = np.eye(4)
    for base, angle in zip(_BASE, q):
        c, s = np.cos(angle), np.sin(angle)
        rz = np.array([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        result = result @ base @ rz
    return result @ _FLANGE


def quaternion_xyzw(rotation):
    """SO(3) to normalized xyzw, including rotations near 180 degrees."""
    r = np.asarray(rotation, dtype=np.float64)
    if r.shape != (3, 3) or not np.isfinite(r).all():
        raise ValueError("finite [3,3] rotation required")
    if not np.allclose(r.T @ r, np.eye(3), atol=1e-8) or not np.isclose(np.linalg.det(r), 1., atol=1e-8):
        raise ValueError("proper rotation required")
    trace = np.trace(r)
    if trace > 0:
        s = np.sqrt(trace + 1.) * 2
        q = np.array([(r[2, 1]-r[1, 2])/s, (r[0, 2]-r[2, 0])/s, (r[1, 0]-r[0, 1])/s, .25*s])
    else:
        i = int(np.argmax(np.diag(r)))
        j, k = (i+1) % 3, (i+2) % 3
        s = np.sqrt(1 + r[i, i]-r[j, j]-r[k, k]) * 2
        q = np.empty(4)
        q[i], q[j], q[k], q[3] = .25*s, (r[j, i]+r[i, j])/s, (r[k, i]+r[i, k])/s, (r[k, j]-r[j, k])/s
    return q / np.linalg.norm(q)


def right_flange_pose(q_rad, timestamp_ns):
    transform = right_flange_transform(q_rad)
    pose = Pose(timestamp_ns, FRAME_ID, tuple(transform[:3, 3]/1000.),
                tuple(quaternion_xyzw(transform[:3, :3])))
    pose.validate()
    return pose
