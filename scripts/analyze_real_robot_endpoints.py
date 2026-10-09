#!/usr/bin/env python3
"""Retrospective, file-only endpoint statistics for Marvin M6 CCS 4.0 data.

q: [N,7] radians -> flange: [N,4,4], translation in mm, SDK arm base.
This is an offline analysis, not a completion detector or robot interface.
Only libKine (pure local mathematics) is used to verify the NumPy FK.
"""
from __future__ import annotations

import argparse
import csv
import ctypes
import hashlib
import json
from pathlib import Path
import re

import h5py
import numpy as np
from scipy.spatial.transform import Rotation


MODEL_SHA = "ce20b1a80974c3ed58cdac4973152625096c8ad93a779e5fd4108c26ce19bd52"
LIBRARY_SHA = "f1512f8e440b53614b8d61adc27b217cab1b97d8bb9a80e32e00c880eee73397"
WINDOWS = (0.2, 0.5, 1.0, 2.0)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def condition_name(name):
    match = re.search(r"_(x\d+y\d+)$", name)
    if not match:
        raise ValueError(f"episode has no terminal hole label: {name}")
    return match.group(1)


def condition_xy(name):
    match = re.fullmatch(r"x(\d+)y(\d+)", name)
    if not match:
        raise ValueError(name)
    return tuple(map(int, match.groups()))


def load_model(path):
    """Restrict the specialized FK to the bridge's checksum-validated model."""
    if sha256(path) != MODEL_SHA:
        raise ValueError("expected the validated M6 CCS 4.0 configuration")
    rows = [[float(v) for v in line.rstrip(",\n").split(",")]
            for line in Path(path).read_text().splitlines() if line.strip()]
    if len(rows) != 26 or int(rows[13][0]) != 1017:
        raise ValueError("unexpected right-arm model layout")
    return np.asarray([r[:4] for r in rows[14:22]], dtype=np.float64)


def flange_fk(q, dh):
    """Vectorized vendor CCS chain: product B_i Rz(q_i), then fixed flange.

    Derived from FX_Init_Robot_Kine_Pilot_CCS / FX_Robot_Kine_Piolt in
    kinematicsSDK/FxRobot.cpp. This is NOT a generic DH interpretation.
    Input q is actual feedback radians; returned translation is millimeters.
    """
    q = np.asarray(q, dtype=np.float64)
    if q.ndim != 2 or q.shape[1] != 7 or not np.isfinite(q).all():
        raise ValueError("q must be finite [N,7] radians")
    dh = np.asarray(dh, dtype=np.float64)
    if dh.shape != (8, 4) or not np.isfinite(dh).all():
        raise ValueError("dh must be finite [8,4]")
    l1, l2, l3, d, flange = dh[0, 2], dh[2, 2], dh[4, 2], dh[3, 1], dh[7, 2]
    base = np.tile(np.eye(4), (7, 1, 1))
    rx = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    base[0, 2, 3] = l1
    base[1, :3, :3] = rx
    base[2, :3, :3] = rx.T
    base[3, :3, :3] = rx
    base[3, :3, 3] = [d, 0, l2]
    base[4, :3, :3] = rx.T
    base[4, :3, 3] = [-d, l3, 0]
    base[5, :3, :3] = rx
    base[6, :3, :3] = [[0, 0, 1], [1, 0, 0], [0, 1, 0]]
    end = np.eye(4)
    end[:3, :3] = [[0, 0, 1], [0, -1, 0], [1, 0, 0]]
    end[0, 3] = flange
    result = np.tile(np.eye(4), (len(q), 1, 1))
    rz = np.tile(np.eye(4), (len(q), 1, 1))
    for j in range(7):
        c, s = np.cos(q[:, j]), np.sin(q[:, j])
        rz[:, 0, 0] = c
        rz[:, 1, 1] = c
        rz[:, 0, 1] = -s
        rz[:, 1, 0] = s
        result = result @ base[j] @ rz
    return result @ end


class VendorFK:
    """Single-process reference wrapper for the pure kinematics shared library.

    API types match FXKineCommon.h: int32_t, unsigned char, double.
    No robot/control library is loaded, and no network functions are called.
    The library has global mutable math state; this wrapper is not thread-safe.
    """
    def __init__(self, library, dh):
        if sha256(library) != LIBRARY_SHA:
            raise ValueError("expected the bridge-validated pure kinematics library")
        self.lib = ctypes.CDLL(str(library))
        self.ptr = ctypes.POINTER(ctypes.c_double)
        specs = {
            "FX_LOG_SWITCH": ([ctypes.c_int32], None),
            "FX_Robot_Init_Type": ([ctypes.c_int32, ctypes.c_int32], ctypes.c_ubyte),
            "FX_Robot_Init_Kine": ([ctypes.c_int32, self.ptr], ctypes.c_ubyte),
            "FX_Robot_Tool_Rmv": ([ctypes.c_int32], ctypes.c_ubyte),
            "FX_Robot_Kine_FK": ([ctypes.c_int32, self.ptr, self.ptr], ctypes.c_ubyte),
        }
        for name, (args, ret) in specs.items():
            fn = getattr(self.lib, name)
            fn.argtypes, fn.restype = args, ret
        self.lib.FX_LOG_SWITCH(0)
        dh = np.ascontiguousarray(dh, dtype=np.float64)
        for ok in (self.lib.FX_Robot_Init_Type(1, 1017),
                   self.lib.FX_Robot_Init_Kine(1, dh.ctypes.data_as(self.ptr)),
                   self.lib.FX_Robot_Tool_Rmv(1)):
            if ok != 1:
                raise RuntimeError("kinematics initialization failed")

    def __call__(self, q):
        q = np.ascontiguousarray(np.rad2deg(q), dtype=np.float64)
        out = np.zeros((len(q), 4, 4))
        for row, matrix in zip(q, out):
            if self.lib.FX_Robot_Kine_FK(1, row.ctypes.data_as(self.ptr),
                                       matrix.ctypes.data_as(self.ptr)) != 1:
                raise RuntimeError("FK failed")
        return out


def rotation_distance_deg(rotations, reference):
    return np.rad2deg(Rotation.from_matrix(rotations @ np.asarray(reference).T).magnitude())


def pose_summary(transforms):
    """Median XYZ and chordal SO(3) mean; no Euler-angle averaging."""
    p = transforms[:, :3, 3]
    r = transforms[:, :3, :3]
    center = np.median(p, axis=0)
    mean_rotation = Rotation.from_matrix(r).mean().as_matrix()
    radius = np.linalg.norm(p - center, axis=1)
    angle = rotation_distance_deg(r, mean_rotation)
    return {
        "center_mm": center.tolist(), "rotation_matrix": mean_rotation.tolist(),
        "quat_xyzw": Rotation.from_matrix(mean_rotation).as_quat().tolist(),
        "radius_p95_mm": float(np.percentile(radius, 95)),
        "radius_max_mm": float(radius.max()), "range_xyz_mm": np.ptp(p, axis=0).tolist(),
        "orientation_p95_deg": float(np.percentile(angle, 95)),
        "orientation_max_deg": float(angle.max()),
        "first_to_last_mm": float(np.linalg.norm(p[-1] - p[0])),
    }


def lag_speed(times, transforms, lag_s=0.05, max_gap_s=0.02):
    """50 ms trailing secant speed on actual time; invalid across stream gaps.

    This damps timestamp/encoder differencing noise, is not instantaneous speed,
    and is used only for descriptive statistics, never safety certification.
    """
    idx = np.searchsorted(times, times - lag_s, side="right") - 1
    safe = np.maximum(idx, 0)
    dt = times - times[safe]
    gap_count = np.r_[0, np.cumsum(np.diff(times) > max_gap_s)]
    valid = (idx >= 0) & (dt > 0) & ((gap_count - gap_count[safe]) == 0)
    speed = np.full(len(times), np.nan)
    angular = speed.copy()
    speed[valid] = np.linalg.norm(transforms[valid, :3, 3] - transforms[safe[valid], :3, 3], axis=1) / dt[valid]
    rel = transforms[valid, :3, :3] @ transforms[safe[valid], :3, :3].transpose(0, 2, 1)
    angular[valid] = np.rad2deg(Rotation.from_matrix(rel).magnitude()) / dt[valid]
    return speed, angular


def finite_percentile(values, percent):
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    return float(np.percentile(values, percent)) if len(values) else None


def read_episode(path):
    with h5py.File(path, "r") as f:
        if f.attrs.get("schema_version") != "real_robot_right_arm_raw_v2" or f.attrs.get("arm") != "right":
            raise ValueError(f"wrong schema/arm: {path}")
        names = [v.decode() if isinstance(v, bytes) else str(v) for v in f["metadata/joint_names"][:]]
        if names != [f"Joint{i}_R" for i in range(1, 8)]:
            raise ValueError("joint order mismatch")
        if f.attrs.get("force_source_topic") != "/tianji_marvin_teleop_bridge/estimated_joint_disturbance":
            raise ValueError("external torque provenance mismatch")
        arrays = {key: np.asarray(f[name], dtype=np.float64) for key, name in {
            "q": "observations/joint_pos", "dq": "observations/joint_vel",
            "tau": "observations/joint_external_torque"}.items()}
        ts = np.asarray(f["timestamps/state/log_ns"], dtype=np.int64)
        tf = np.asarray(f["timestamps/disturbance/log_ns"], dtype=np.int64)
        for key, arr in arrays.items():
            n = len(tf) if key == "tau" else len(ts)
            if arr.shape != (n, 7) or not np.isfinite(arr).all():
                raise ValueError(f"invalid {key}: {path}")
        if len(ts) < 2 or len(tf) < 2 or np.any(np.diff(ts) <= 0) or np.any(np.diff(tf) <= 0):
            raise ValueError("non-increasing/empty timestamps")
        return {**arrays, "t": (ts - ts[-1]) / 1e9, "tf": (tf - ts[-1]) / 1e9,
                "state_end_log_ns": int(ts[-1]), "path": str(path), "episode": path.parent.name,
                "condition": condition_name(path.parent.name), "source_sha256": str(f.attrs["source_sha256"]),
                "hdf5_size_bytes": path.stat().st_size,
                "numeric_sha256": hashlib.sha256(ts.tobytes() + tf.tobytes() + b"".join(
                    arrays[k].tobytes() for k in ("q", "dq", "tau"))).hexdigest()}


def window_summary(e, transforms, start, end, speed, angular):
    mask = (e["t"] >= start) & (e["t"] <= end)
    t = e["t"][mask]
    if len(t) < 2:
        raise ValueError("insufficient window samples")
    force_mask = (e["tf"] >= start) & (e["tf"] <= end)
    tau = e["tau"][force_mask]
    return {**pose_summary(transforms[mask]), "start_s_from_end": float(t[0]),
            "end_s_from_end": float(t[-1]), "n": len(t), "max_gap_ms": float(np.diff(t).max()*1000),
            "speed_p50_mm_s": finite_percentile(speed[mask], 50),
            "speed_p95_mm_s": finite_percentile(speed[mask], 95),
            "angular_speed_p95_deg_s": finite_percentile(angular[mask], 95),
            "max_joint_speed_rad_s": float(np.abs(e["dq"][mask]).max()),
            "q_median_rad": np.median(e["q"][mask], axis=0).tolist(),
            "external_torque_n": len(tau),
            "external_torque_median_nm": np.median(tau, axis=0).tolist() if len(tau) else None,
            "external_torque_abs_p95_nm": np.percentile(np.abs(tau), 95, axis=0).tolist() if len(tau) else None}


def latest_stable_window(e, transforms, speed, angular):
    """Fixed rule, no group center/25 mm layout used to select a window.

    Search latest-first over last 5 seconds, step 0.1 s, window 0.5 s.
    Requirements: p95 radius <=1 mm, p95 angle <=1 deg, median trailing
    speed <=5 mm/s; >=120 samples; no gap >20 ms. Exploratory only.
    """
    for end in np.arange(0., -4.5001, -0.1):
        if end - 0.5 < e["t"][0]:
            continue
        s = window_summary(e, transforms, end - 0.5, end, speed, angular)
        if (s["n"] >= 120 and s["max_gap_ms"] <= 20 and s["radius_p95_mm"] <= 1
                and s["orientation_p95_deg"] <= 1 and s["speed_p50_mm_s"] is not None
                and s["speed_p50_mm_s"] <= 5):
            return s
    return None


def group_stats(records, window_key):
    result = {}
    for group in sorted({e["condition"] for e in records}):
        members = [e for e in records if e["condition"] == group]
        windows = [e[window_key] for e in members if e[window_key] is not None]
        if not windows:
            continue
        centers = np.asarray([w["center_mm"] for w in windows])
        rotations = np.asarray([w["rotation_matrix"] for w in windows])
        matrices = np.tile(np.eye(4), (len(centers), 1, 1))
        matrices[:, :3, 3] = centers
        matrices[:, :3, :3] = rotations
        summary = pose_summary(matrices)
        summary.pop("first_to_last_mm")
        # Flange Z is a diagnostic axis, NOT an assumed calibrated insertion axis.
        axis = np.asarray(summary["rotation_matrix"])[:, 2]
        delta = centers - np.asarray(summary["center_mm"])
        axial = delta @ axis
        perpendicular = np.linalg.norm(delta - axial[:, None] * axis, axis=1)
        pair_dist = np.linalg.norm(centers[:, None] - centers[None, :], axis=-1)
        loo = []
        if len(centers) > 1:
            for i in range(len(centers)):
                loo.append(float(np.linalg.norm(centers[i] - np.median(np.delete(centers, i, axis=0), axis=0))))
        result[group] = {**summary, "n_episodes": len(windows),
                         "episode_centers_mm": centers.tolist(),
                         "max_pairwise_mm": float(pair_dist.max()),
                         "diagnostic_flange_z_axis": axis.tolist(),
                         "flange_z_deviation_range_mm": [float(axial.min()), float(axial.max())],
                         "flange_z_perpendicular_max_mm": float(perpendicular.max()),
                         "leave_one_out_errors_mm": loo,
                         "joint_center_range_rad": np.ptp(np.asarray([w["q_median_rad"] for w in windows]), axis=0).tolist()}
    return result


def grid_stats(groups, expected_spacing=25.):
    names = sorted(groups)
    xy = np.asarray([condition_xy(name) for name in names], dtype=float)
    p = np.asarray([groups[name]["center_mm"] for name in names])
    xy -= xy.min(axis=0)
    design = np.column_stack([np.ones(len(xy)), xy])
    coefficients = np.linalg.lstsq(design, p, rcond=None)[0]
    fit = design @ coefficients
    vx, vy = coefficients[1:]
    center = p.mean(axis=0)
    _, singular, vt = np.linalg.svd(p - center, full_matrices=False)
    plane_distance = (p - center) @ vt[-1]
    # Known 25 mm, perpendicular lattice: rigid least-squares fit, no rescaling.
    ideal = np.column_stack([xy * expected_spacing, np.zeros(len(xy))])
    x = ideal - ideal.mean(axis=0)
    y = p - p.mean(axis=0)
    u, _, vh = np.linalg.svd(x.T @ y)
    correction = np.eye(3)
    correction[-1, -1] = np.linalg.det(u @ vh)
    rotation_row = u @ correction @ vh
    rigid_fit = x @ rotation_row + p.mean(axis=0)
    rigid_error = np.linalg.norm(p - rigid_fit, axis=1)
    edges = []
    for i, a in enumerate(names):
        for j, b in enumerate(names[i+1:], i+1):
            delta = xy[j] - xy[i]
            if np.abs(delta).sum() == 1:
                distance = float(np.linalg.norm(p[j] - p[i]))
                edges.append({"a": a, "b": b, "axis": "x" if delta[0] else "y",
                              "distance_mm": distance, "error_from_25_mm": distance-expected_spacing})
    return {"names": names, "affine_origin_mm": coefficients[0].tolist(),
            "x_step_vector_mm": vx.tolist(), "y_step_vector_mm": vy.tolist(),
            "x_step_mm": float(np.linalg.norm(vx)), "y_step_mm": float(np.linalg.norm(vy)),
            "xy_angle_deg": float(np.rad2deg(np.arccos(np.clip(vx@vy/(np.linalg.norm(vx)*np.linalg.norm(vy)), -1, 1)))),
            "affine_residual_mm": np.linalg.norm(p-fit, axis=1).tolist(),
            "rigid_25mm_residual_mm": rigid_error.tolist(),
            "rigid_25mm_rms_mm": float(np.sqrt(np.mean(rigid_error**2))),
            "rigid_fit_mm": rigid_fit.tolist(), "rigid_basis_row": rotation_row.tolist(),
            "plane_normal": vt[-1].tolist(), "plane_center_mm": center.tolist(),
            "plane_distance_mm": plane_distance.tolist(), "plane_singular_values_mm": singular.tolist(),
            "adjacent_edges": edges}


def endpoint_checks(records, groups, grid):
    """Repeatability and held-out endpoint assignment, NOT task detection."""
    names = sorted(groups)
    basis = np.asarray(grid["rigid_basis_row"])
    rows = []
    for e in records:
        p = np.asarray(e["last_0.5s"]["center_mm"])
        centers = []
        for name in names:
            others = [x["last_0.5s"]["center_mm"] for x in records
                      if x["condition"] == name and x["episode"] != e["episode"]]
            centers.append(np.median(others, axis=0))
        distances = np.linalg.norm(np.asarray(centers) - p, axis=1)
        own = names.index(e["condition"])
        delta = p - np.asarray(groups[e["condition"]]["center_mm"])
        axis = np.asarray(groups[e["condition"]]["rotation_matrix"])[:, 2]
        axial = float(delta @ axis)
        rows.append({"episode": e["episode"], "condition": e["condition"],
                     "distance_to_all5_median_mm": float(np.linalg.norm(delta)),
                     "delta_grid_xyz_mm": (delta @ basis.T).tolist(),
                     "delta_flange_z_mm": axial,
                     "delta_perpendicular_to_flange_z_mm": float(np.linalg.norm(delta-axial*axis)),
                     "loo_nearest_condition": names[int(np.argmin(distances))],
                     "loo_own_distance_mm": float(distances[own]),
                     "loo_nearest_other_distance_mm": float(np.min(np.delete(distances, own)))})
    train_centers = {name: np.median([e["last_0.5s"]["center_mm"] for e in records
                                    if e["condition"] == name and e["split"] == "train"], axis=0) for name in names}
    heldout = []
    for e in records:
        if e["split"] == "train":
            continue
        distances = {name: float(np.linalg.norm(np.asarray(e["last_0.5s"]["center_mm"])-p)) for name, p in train_centers.items()}
        heldout.append({"episode": e["episode"], "condition": e["condition"], "split": e["split"],
                        "own_error_mm": distances[e["condition"]], "nearest_condition": min(distances, key=distances.get)})
    return {"interpretation": "Only classifies terminal poses among holes; does NOT measure task-success accuracy or false stops during approach",
            "leave_one_episode_out": rows, "train_only_centers_mm": {k:v.tolist() for k,v in train_centers.items()},
            "held_out_endpoints": heldout,
            "largest_deviation_episode": max(rows, key=lambda x:x["distance_to_all5_median_mm"])["episode"]}


def causal_frame_index(times_ns, target_ns):
    index = int(np.searchsorted(times_ns, target_ns, side="right"))-1
    if index < 0:
        raise ValueError("no camera frame at or before target")
    return index


def continuous_intervals(times, valid, max_gap_s=.02):
    """Return sample-supported contiguous intervals; never bridge missing data."""
    indices = np.flatnonzero(valid)
    if not len(indices):
        return []
    split_at = np.flatnonzero((np.diff(indices) != 1) | (np.diff(times[indices]) > max_gap_s)) + 1
    result = []
    for run in np.split(indices, split_at):
        result.append({"start_s_from_end": float(times[run[0]]),
                       "end_s_from_end": float(times[run[-1]]),
                       "duration_s": float(times[run[-1]]-times[run[0]]), "n": len(run)})
    return result


def analyze_lookback(records, dh):
    """Test each final 5 s against OTHER demonstrations' terminal reference.

    This is retrospective evidence of reaching a neighborhood, not a relabeling
    of success events. The reference is not fitted to the inspected trajectory.
    """
    results = []
    for record in records:
        others = [x["last_0.5s"] for x in records if x["condition"] == record["condition"]
                  and x["episode"] != record["episode"]]
        reference = np.median([x["center_mm"] for x in others], axis=0)
        reference_rotation = Rotation.from_matrix(np.asarray([x["rotation_matrix"] for x in others])).mean().as_matrix()
        e = read_episode(Path(record["path"]))
        transform = flange_fk(e["q"], dh)
        speed, angular = lag_speed(e["t"], transform)
        dist = np.linalg.norm(transform[:, :3, 3]-reference, axis=1)
        angle = rotation_distance_deg(transform[:, :3, :3], reference_rotation)
        tail = e["t"] >= -5
        ids = np.flatnonzero(tail)
        closest = ids[np.argmin(dist[tail])]
        windows = []
        for end in np.arange(0., -4.5001, -.1):
            w = window_summary(e, transform, end-.5, end, speed, angular)
            mask = (e["t"] >= end-.5) & (e["t"] <= end)
            if w["max_gap_ms"] > 20 or w["n"] < 120:
                continue
            w["distance_to_reference_mm"] = float(np.linalg.norm(np.asarray(w["center_mm"])-reference))
            w["sample_distance_to_reference_p95_mm"] = float(np.percentile(dist[mask],95))
            w["sample_distance_to_reference_max_mm"] = float(np.max(dist[mask]))
            w["orientation_to_reference_max_deg"] = float(angle[mask].max())
            windows.append(w)
        threshold_stats = {}
        for radius in (1., 2., 3., 5.):
            runs = continuous_intervals(e["t"], tail & (dist <= radius) & (angle <= 1.))
            threshold_stats[f"{radius:g}mm"] = {
                "position_radius_mm":radius, "orientation_radius_deg":1., "intervals":runs,
                "max_continuous_duration_s":max((r["duration_s"] for r in runs),default=0.),
                "has_0.2s_hold":any(r["duration_s"]>=.2 for r in runs),
                "has_0.5s_hold":any(r["duration_s"]>=.5 for r in runs)}
        results.append({"episode":record["episode"],"condition":record["condition"],
                        "reference_source":"other four demonstrations' last-0.5s centers; held-out inspected episode",
                        "reference_center_mm":reference.tolist(),"reference_rotation":reference_rotation.tolist(),
                        "last_0.5s_center_distance_mm":float(np.linalg.norm(np.asarray(record["last_0.5s"]["center_mm"])-reference)),
                        "closest_sample_distance_mm":float(dist[closest]),
                        "closest_sample_time_s_from_end":float(e["t"][closest]),
                        "closest_0.5s_window":min(windows,key=lambda x:x["sample_distance_to_reference_p95_mm"]) if windows else None,
                        "thresholds":threshold_stats})
    summary = {key:{"has_0.2s_hold":sum(e["thresholds"][key]["has_0.2s_hold"] for e in results),
                     "has_0.5s_hold":sum(e["thresholds"][key]["has_0.5s_hold"] for e in results)}
               for key in ("1mm","2mm","3mm","5mm")}
    return {"scope":"last 5 seconds; same-hole leave-one-episode-out target; log timestamps",
            "interpretation":"Target-neighborhood occupancy only; no confirmed first-success labels and no speed/force gate; retrospective window search cannot be used directly online",
            "summary":summary,"episodes":results}


def plot_lookback(out, records, dh, lookback, outlier_id):
    import matplotlib.pyplot as plt
    by_id = {e["episode"]: e for e in lookback["episodes"]}
    names = sorted({e["condition"] for e in records})
    fig, axs = plt.subplots(5,2,figsize=(14,15),constrained_layout=True)
    for ax, name in zip(axs.flat, names):
        for record in records:
            if record["condition"]!=name:
                continue
            e=read_episode(Path(record["path"]))
            mask=e["t"]>=-5
            p=flange_fk(e["q"][mask],dh)[:,:3,3]
            d=np.linalg.norm(p-np.asarray(by_id[e["episode"]]["reference_center_mm"]),axis=1)
            ax.plot(e["t"][mask][::5],d[::5],label=e["episode"].split("_")[4],linewidth=1)
        ax.axhline(3,color="black",linestyle="--",linewidth=.8,label="3 mm")
        ax.set(title=name,xlabel="Seconds before state stream ends",ylabel="Distance to other-four target (mm)")
        ax.legend(fontsize=7,ncol=2);ax.grid(alpha=.2)
    fig.suptitle("Last 5 seconds vs leave-one-demonstration-out reference; NOT completion labels")
    fig.savefig(out/"lookback_5s.png",dpi=145);plt.close(fig)
    record=next(e for e in records if e["episode"]==outlier_id)
    e=read_episode(Path(record["path"]))
    mask=e["t"]>=-5
    p=flange_fk(e["q"][mask],dh)[:,:3,3]
    d=np.linalg.norm(p-np.asarray(by_id[outlier_id]["reference_center_mm"]),axis=1)
    fig,ax=plt.subplots(figsize=(10,4.5),constrained_layout=True)
    ax.plot(e["t"][mask],d,label="Distance to other-four target")
    ax.axhline(3,color="black",linestyle="--",label="3 mm neighborhood")
    for run in by_id[outlier_id]["thresholds"]["3mm"]["intervals"]:
        if run["duration_s"]>=.2:
            ax.axvspan(run["start_s_from_end"],run["end_s_from_end"],color="green",alpha=.15)
    ax.set(xlabel="Seconds before state stream ends",ylabel="Flange distance (mm)",title=outlier_id)
    ax.legend();ax.grid(alpha=.2)
    fig.savefig(out/"outlier_lookback_5s.png",dpi=160);plt.close(fig)


def make_outlier_evidence(out, records, groups, checks, dh):
    """Two raw camera-frame contact sheets and measured motion of largest outlier."""
    import matplotlib.pyplot as plt
    by_name = {e["episode"]:e for e in records}
    worst = by_name[checks["largest_deviation_episode"]]
    same = [e for e in checks["leave_one_episode_out"] if e["condition"] == worst["condition"]]
    reference = by_name[min(same, key=lambda x:x["distance_to_all5_median_mm"])["episode"]]
    frame_provenance = []
    for e in (reference, worst):
        fig, axs = plt.subplots(2, 3, figsize=(12, 6.5), constrained_layout=True)
        with h5py.File(e["path"], "r") as f:
            for row, cam in enumerate(("d455", "d405")):
                times = np.asarray(f[f"timestamps/{cam}/log_ns"], dtype=np.int64)
                for col, age in enumerate((5., 2., .05)):
                    target = e["state_end_log_ns"]-round(age*1e9)
                    idx = causal_frame_index(times, target)
                    actual_age = (e["state_end_log_ns"]-int(times[idx]))/1e9
                    axs[row, col].imshow(f[f"observations/images/{cam}/rgb"][idx])
                    axs[row, col].set_title(f"{cam} / {actual_age:.3f} s before end")
                    axs[row, col].axis("off")
                    frame_provenance.append({"episode":e["episode"], "camera":cam, "index":idx,
                                             "frame_log_ns":int(times[idx]), "age_before_state_end_s":actual_age})
        fig.suptitle(e["episode"])
        fig.savefig(out / f"frames_{e['episode'].split('_')[4]}.jpg", dpi=140)
        plt.close(fig)
    e = read_episode(Path(worst["path"]))
    transform = flange_fk(e["q"], dh)
    center = np.asarray(groups[e["condition"]]["center_mm"])
    delta = transform[:, :3, 3] - center
    mask = e["t"] >= -8
    fig, axs = plt.subplots(3, 1, figsize=(11, 8), sharex=True, constrained_layout=True)
    for i in range(3):
        axs[0].plot(e["t"][mask], delta[mask,i], label="XYZ"[i])
    axs[0].set(ylabel="Flange delta to group (mm)", title=worst["episode"])
    axs[0].legend(); axs[0].grid(alpha=.2)
    angles=rotation_distance_deg(transform[:, :3,:3], np.asarray(groups[e["condition"]]["rotation_matrix"]))
    axs[1].plot(e["t"][mask],angles[mask]); axs[1].set(ylabel="Rotation error (deg)"); axs[1].grid(alpha=.2)
    mask=e["tf"]>=-8
    for i in range(7):
        axs[2].plot(e["tf"][mask],e["tau"][mask,i],label=f"J{i+1}",linewidth=.8)
    axs[2].set(xlabel="Seconds before state stream ends",ylabel="External joint torque (Nm)")
    axs[2].legend(ncol=7,fontsize=8); axs[2].grid(alpha=.2)
    for ax in axs:
        ax.axvspan(-.5,0,color="grey",alpha=.12)
    fig.savefig(out/"outlier_terminal_detail.png",dpi=160);plt.close(fig)
    write_json(out/"frame_provenance.json",frame_provenance)


def make_plots(out, records, trajectories, groups, grid):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = sorted(groups)
    colors = dict(zip(names, plt.get_cmap("tab10").colors))
    p = np.asarray([groups[n]["center_mm"] for n in names])
    origin = p.mean(axis=0)
    basis = np.asarray(grid["rigid_basis_row"])
    projected = (p-origin) @ basis.T
    fig, axs = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)
    for i, name in enumerate(names):
        samples = (np.asarray(groups[name]["episode_centers_mm"])-origin) @ basis.T
        axs[0].scatter(samples[:, 0], samples[:, 1], color=colors[name], s=30, alpha=.7)
        axs[0].scatter(*projected[i, :2], marker="x", color=colors[name], s=90)
        axs[0].annotate(name, projected[i, :2], xytext=(4, 5), textcoords="offset points", fontsize=9)
    ideal = (np.asarray(grid["rigid_fit_mm"])-origin) @ basis.T
    axs[0].scatter(ideal[:, 0], ideal[:, 1], s=130, facecolors="none", edgecolors="black", label="25 mm rigid fit")
    axs[0].set(xlabel="Grid X projection (mm)", ylabel="Grid Y projection (mm)", title="Flange endpoints: five demos per hole")
    axs[0].axis("equal"); axs[0].legend(loc="upper left", fontsize=8); axs[0].grid(alpha=.2)
    axs[1].bar(np.arange(len(names)), [groups[n]["radius_max_mm"] for n in names], color=[colors[n] for n in names])
    axs[1].set_xticks(np.arange(len(names)), names, rotation=45)
    axs[1].set(ylabel="Max distance to group median (mm)", title="Between-demonstration endpoint spread")
    axs[1].grid(axis="y", alpha=.2)
    fig.suptitle("Actual joint feedback -> SDK-validated FK; last 0.5 s; flange, NOT calibrated peg tip")
    fig.savefig(out / "endpoint_grid.png", dpi=170); plt.close(fig)
    fig, axs = plt.subplots(5, 2, figsize=(14, 15), constrained_layout=True)
    for ax, name in zip(axs.flat, names):
        for e in records:
            if e["condition"] != name:
                continue
            t, pos, speed = trajectories[e["episode"]]
            distance = np.linalg.norm(pos - np.asarray(e["last_0.5s"]["center_mm"]), axis=1)
            mask = t >= -5
            ax.plot(t[mask], distance[mask], label=e["episode"].split("_")[4], linewidth=1)
        ax.axvspan(-.5, 0, color="grey", alpha=.12)
        ax.set(title=name, xlabel="Seconds before state stream ends", ylabel="Distance to own final center (mm)")
        ax.grid(alpha=.2); ax.legend(fontsize=7)
    fig.suptitle("Terminal motion: each trajectory relative to its own last-0.5-s center")
    fig.savefig(out / "terminal_motion.png", dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    values = [[e[f"last_{w:g}s"]["radius_p95_mm"] for e in records] for w in WINDOWS]
    ax.boxplot(values, tick_labels=[str(w) for w in WINDOWS], showfliers=True)
    ax.set(xlabel="Terminal window duration (s)", ylabel="Within-window position radius P95 (mm)", title="Window sensitivity across 50 demonstrations")
    ax.grid(axis="y", alpha=.2)
    fig.savefig(out / "window_sensitivity.png", dpi=170); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--kinematics-library", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    dh = load_model(args.model)
    vendor = VendorFK(args.kinematics_library, dh)
    paths = sorted(args.data_root.glob("*/episode.hdf5"))
    if not paths:
        raise ValueError("no episodes")
    split = json.loads(args.split.read_text())
    split_by_episode = {e: k for k, key in (("train", "train_episode_ids"),
                        ("val", "validation_episode_ids"), ("test", "test_episode_ids")) for e in split[key]}
    records, trajectories, validation = [], {}, []
    for path in paths:
        e = read_episode(path)
        transforms = flange_fk(e["q"], dh)
        # Include start/end, full trajectory and zero pose; do not just test endpoints.
        ids = np.linspace(0, len(e["q"])-1, 25).astype(int)
        qcheck = np.vstack([np.zeros((1, 7)), e["q"][ids]])
        reference = vendor(qcheck)
        actual = flange_fk(qcheck, dh)
        p_error = float(np.max(np.linalg.norm(reference[:, :3, 3]-actual[:, :3, 3], axis=1)))
        r_error = float(np.max(np.abs(reference[:, :3, :3]-actual[:, :3, :3])))
        if p_error > 1e-6 or r_error > 1e-9:
            raise ValueError(f"FK reference disagreement {p_error}, {r_error}")
        validation.append({"episode": e["episode"], "poses": len(qcheck), "max_position_error_mm": p_error, "max_rotation_element_error": r_error})
        speed, angular = lag_speed(e["t"], transforms)
        record = {k: e[k] for k in ("episode", "condition", "path", "source_sha256", "numeric_sha256", "hdf5_size_bytes", "state_end_log_ns")}
        record.update({"split": split_by_episode[e["episode"]], "state_samples": len(e["t"]),
                       "duration_s": float(-e["t"][0]), "max_state_gap_ms": float(np.diff(e["t"]).max()*1000),
                       "success_label": "user_confirmed_success", "last_sample_mm": transforms[-1, :3, 3].tolist()})
        for w in WINDOWS:
            record[f"last_{w:g}s"] = window_summary(e, transforms, -w, 0., speed, angular)
        record["stable"] = latest_stable_window(e, transforms, speed, angular)
        # Full-trajectory display cache only, every ~30 ms (no model resampling).
        stride = max(1, round(.03 / np.median(np.diff(e["t"]))))
        ids = np.unique(np.r_[np.arange(0, len(e["t"]), stride), len(e["t"])-1])
        trajectories[e["episode"]] = (e["t"][ids], transforms[ids, :3, 3], speed[ids])
        records.append(record)
        print(f"{e['condition']} {e['episode']} radius95={record['last_0.5s']['radius_p95_mm']:.3f}mm speed50={record['last_0.5s']['speed_p50_mm_s']:.3f}mm/s", flush=True)
    grouped = {f"last_{w:g}s": group_stats(records, f"last_{w:g}s") for w in WINDOWS}
    grouped["stable"] = group_stats(records, "stable")
    grids = {k: grid_stats(v) for k, v in grouped.items() if len(v) == 10}
    groups = grouped["last_0.5s"]
    checks = endpoint_checks(records, groups, grids["last_0.5s"])
    write_json(args.output / "endpoint_checks.json", checks)
    lookback = analyze_lookback(records, dh)
    write_json(args.output / "lookback_5s.json", lookback)
    write_json(args.output / "episodes.json", records)
    write_json(args.output / "group_statistics.json", grouped)
    write_json(args.output / "grid_statistics.json", grids)
    write_json(args.output / "fk_validation.json", validation)
    write_json(args.output / "manifest.json", {
        "script": str(Path(__file__).resolve()), "script_sha256": sha256(__file__),
        "data_root": str(args.data_root.resolve()), "episodes": len(records),
        "model": str(args.model.resolve()), "model_sha256": sha256(args.model),
        "library": str(args.kinematics_library.resolve()), "library_sha256": sha256(args.kinematics_library),
        "split": str(args.split.resolve()), "split_sha256": sha256(args.split),
        "scope": "all 50 demonstrations; equal episode weights; descriptive, NOT held-out detector validation",
        "frame": "right_arm_SDK_base_to_flange; not calibrated peg-tip TCP; no world transform",
        "timestamps": "MCAP log_ns; subtract integer state end before conversion to seconds",
        "external_torque": "estimated_joint_disturbance.effort, never joint_states.effort",
        "expected_grid_mm": 25, "hardware_connection": False, "goal_configs_modified": False,
        "numpy": np.__version__, "method": "median XYZ per window; median episode centers per group; chordal SO(3) means",
        "stable_window_rule": "latest 0.5s in last 5s at 0.1s step; radius P95 <=1mm; orientation P95 <=1deg; median 50ms secant speed <=5mm/s; N>=120; gap<=20ms"})
    with (args.output / "endpoint_summary.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["condition", "n", "flange_x_mm", "flange_y_mm", "flange_z_mm", "max_radius_mm", "max_pairwise_mm", "orientation_max_deg", "loo_max_error_mm"])
        for name, group in groups.items():
            writer.writerow([name, group["n_episodes"], *group["center_mm"], group["radius_max_mm"], group["max_pairwise_mm"], group["orientation_max_deg"], max(group["leave_one_out_errors_mm"])])
    make_plots(args.output, records, trajectories, groups, grids["last_0.5s"])
    make_outlier_evidence(args.output, records, groups, checks, dh)
    plot_lookback(args.output, records, dh, lookback, checks["largest_deviation_episode"])
    print(json.dumps({"episodes": len(records), "stable_found": sum(e["stable"] is not None for e in records),
                      "grid": grids["last_0.5s"]}, indent=2))


if __name__ == "__main__":
    main()
