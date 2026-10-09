"""Shared scalar/batched SI conversion for training and causal live receivers.

No SDK/device imports. Full six-dimensional least squares precedes force-only
selection; discarded moments are not assumed zero during inversion.
"""
from dataclasses import dataclass

import numpy as np

from force_aware_act.control.kinematics import MODEL_SHA256, _BASE, _FLANGE


def conversion_contract():
    return {"version": "m6_ccs40_flange_force3_svd_v1", "model_sha256": MODEL_SHA256,
            "reference_point": "right_flange_origin", "component_frame": "moving_right_flange",
            "component_order": ["Fx", "Fy", "Fz"], "unit": "N",
            "solver": "six_dimensional_unweighted_joint_torque_least_squares_then_take_force",
            "length_scale_m": 0.1, "svd_rcond": 1e-10, "max_scaled_condition": 200.0,
            "residual_filter": False, "smoothing": False, "additional_bias_subtraction": False,
            "sign": "inherited_from_estimated_joint_disturbance"}


def _geometry(q):
    """q [N,7] rad -> base geometric J [N,6,7] (meters), flange R [N,3,3]."""
    transform = np.broadcast_to(np.eye(4), (len(q), 4, 4)).copy()
    origins, axes = [], []
    for k, base in enumerate(_BASE):
        transform = transform @ base
        origins.append(transform[:, :3, 3].copy() / 1000.0)
        axes.append(transform[:, :3, 2].copy())
        c, s = np.cos(q[:, k]), np.sin(q[:, k])
        rz = np.broadcast_to(np.eye(4), transform.shape).copy()
        rz[:, 0, 0], rz[:, 0, 1], rz[:, 1, 0], rz[:, 1, 1] = c, -s, s, c
        transform = transform @ rz
    transform = transform @ _FLANGE
    a, o = np.stack(axes, axis=1), np.stack(origins, axis=1)
    linear = np.cross(a, transform[:, None, :3, 3] / 1000.0 - o)
    return np.concatenate((linear.transpose(0, 2, 1), a.transpose(0, 2, 1)), axis=1), transform[:, :3, :3]


def _scaled_svd(jacobian):
    scaled = jacobian.copy()
    scaled[:, :3] /= 0.1
    u, singular, vh = np.linalg.svd(scaled.transpose(0, 2, 1), full_matrices=False)
    keep = singular > singular[:, :1] * 1e-10
    condition = np.divide(singular[:, 0], singular[:, -1],
                          out=np.full(len(singular), np.inf), where=singular[:, -1] > 0)
    valid = keep.all(axis=1) & (condition <= 200.0)
    return u, singular, vh, keep, condition, valid


@dataclass(frozen=True)
class FlangeForceResult:
    force: np.ndarray
    valid: np.ndarray
    wrench_base: np.ndarray
    wrench_flange: np.ndarray
    scaled_condition: np.ndarray
    relative_residual: np.ndarray


def joint_torque_to_flange_force(q_rad, external_torque_nm):
    """Same algorithm for [7] live samples and [N,7] training batches.

    Invalid numeric inversions return masked force placeholders, never measured
    zeros. High reconstruction residual is diagnostic, not a contact label.
    """
    q, tau = np.asarray(q_rad, dtype=np.float64), np.asarray(external_torque_nm, dtype=np.float64)
    if q.shape != tau.shape or q.ndim not in (1, 2) or q.shape[-1] != 7:
        raise ValueError("q and external torque require matching [7] or [N,7] shapes")
    if not np.isfinite(q).all() or not np.isfinite(tau).all():
        raise ValueError("q and external torque must be finite")
    scalar = q.ndim == 1
    q, tau = q.reshape(-1, 7), tau.reshape(-1, 7)
    j, rotation = _geometry(q)
    u, s, vh, keep, condition, valid = _scaled_svd(j)
    inverse = np.divide(1., s, out=np.zeros_like(s), where=keep)
    projected = np.einsum("nji,nj->ni", u, tau) * inverse
    wrench = np.einsum("nji,nj->ni", vh, projected)
    wrench[:, :3] /= 0.1
    local = np.concatenate((np.einsum("nji,nj->ni", rotation, wrench[:, :3]),
                            np.einsum("nji,nj->ni", rotation, wrench[:, 3:])), axis=1)
    residual = np.linalg.norm(tau - np.einsum("nji,nj->ni", j, wrench), axis=1) / np.maximum(np.linalg.norm(tau, axis=1), .1)
    valid &= np.isfinite(local).all(axis=1)
    force = local[:, :3].copy()
    force[~valid] = 0.
    fields = (force, valid, wrench, local, condition, residual)
    return FlangeForceResult(*(field[0] if scalar else field for field in fields))


class FlangeForceStream:
    """Keep source arrays and pairing; calculate requested forces without caching.

    Header identity pairs feedback frames. Availability is the later source log
    time. Initialization audits pose conditioning only, not precomputed forces.
    """
    def __init__(self, qpos, torque, state_headers, disturbance_headers, state_logs, disturbance_logs):
        if np.any(np.diff(state_headers) <= 0):
            raise ValueError("force3 requires unique increasing state headers for exact frame pairing")
        self.qpos, self.torque = qpos, torque
        self.shape = (len(torque), 3)
        insertion = np.searchsorted(state_headers, disturbance_headers)
        safe = np.minimum(insertion, len(state_headers) - 1)
        paired = (insertion < len(state_headers)) & (state_headers[safe] == disturbance_headers)
        self.state_indices = np.where(paired, safe, -1)
        self.timestamps_ns = disturbance_logs.copy()
        self.timestamps_ns[paired] = np.maximum(disturbance_logs[paired], state_logs[safe[paired]])
        if np.any(np.diff(self.timestamps_ns) <= 0):
            raise ValueError("derived force availability must be strictly increasing")
        self.valid_samples = paired.copy()
        conditions = []
        for start in range(0, len(torque), 2048):
            rows = np.arange(start, min(start + 2048, len(torque)))
            rows = rows[paired[rows]]
            if len(rows):
                j, _ = _geometry(qpos[self.state_indices[rows]])
                _, _, _, _, condition, valid = _scaled_svd(j)
                self.valid_samples[rows] &= valid
                conditions.extend(condition[np.isfinite(condition)].tolist())
        self.diagnostics = {"conversion": conversion_contract(), "on_demand": True,
            "samples": len(torque), "exact_frame_pairs": int(paired.sum()),
            "unpaired_samples": int((~paired).sum()), "numeric_invalid_samples": int((paired & ~self.valid_samples).sum()),
            "availability_delayed_samples": int((self.timestamps_ns > disturbance_logs).sum()),
            "max_added_availability_ms": float(np.max(self.timestamps_ns - disturbance_logs) / 1e6),
            "max_finite_scaled_condition": max(conditions) if conditions else None}

    def read(self, indices):
        rows = np.atleast_1d(np.arange(len(self.torque))[indices])
        valid = self.valid_samples[rows].copy()
        force = np.zeros((len(rows), 3), dtype=np.float64)
        if valid.any():
            selected = rows[valid]
            result = joint_torque_to_flange_force(self.qpos[self.state_indices[selected]], self.torque[selected])
            force[valid] = result.force
            valid[valid] = result.valid
        return force, valid
