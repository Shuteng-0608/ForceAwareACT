#!/usr/bin/env python3
"""Replay HDF5 joint states in MuJoCo to audit peg-tip task error.

This script is read-only with respect to HDF5 data. It reconstructs per-state
peg-tip and hole-center positions by setting recorded internal MuJoCo qpos and
calling mj_forward without stepping simulation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from force_aware_act.utils import resolve_episode_paths  # noqa: E402


CAMERA_NAMES = ("ee_cam", "base_top_cam")
SENSOR_NAMES = ("peg_ft_force", "peg_ft_torque")
SITE_NAMES = ("peg_tip_site", "hole_center_site")
GEOM_NAME = "cylindrical_peg"
DISTANCE_BANDS = (
    ("gt_0p05", 0.05, math.inf),
    ("0p03_0p05", 0.03, 0.05),
    ("0p02_0p03", 0.02, 0.03),
    ("0p01_0p02", 0.01, 0.02),
    ("lt_0p01", -math.inf, 0.01),
)
THRESHOLDS = (0.05, 0.03, 0.02, 0.01, 0.005)


def _load_mujoco():
    try:
        import mujoco
    except ImportError as error:
        raise RuntimeError("the 'mujoco' Python package is required for replay audit") from error
    return mujoco


def _discover_episodes(data_dir: Path, episode_list: Optional[Path]) -> list[Path]:
    if episode_list is not None:
        return resolve_episode_paths([], episode_list, project_root=REPO_ROOT)
    data_dir = data_dir.expanduser()
    if not data_dir.is_absolute():
        data_dir = REPO_ROOT / data_dir
    if not data_dir.is_dir():
        raise FileNotFoundError(f"data directory does not exist: {data_dir}")
    episodes = sorted(data_dir.glob("*/episode.hdf5"))
    print(f"discovered_episode_count={len(episodes)}")
    for episode in episodes:
        print(f"discovered_episode={episode}")
    return episodes


def _read_array(handle: h5py.File, keys: Sequence[str]) -> tuple[Optional[np.ndarray], str]:
    for key in keys:
        if key in handle:
            return np.asarray(handle[key]), key
    return None, ""


def _nearest_indices(source_timestamps: np.ndarray, target_timestamps: np.ndarray) -> np.ndarray:
    indices = np.searchsorted(source_timestamps, target_timestamps, side="left")
    indices = np.clip(indices, 0, len(source_timestamps) - 1)
    previous = np.clip(indices - 1, 0, len(source_timestamps) - 1)
    choose_previous = (
        np.abs(source_timestamps[previous] - target_timestamps)
        <= np.abs(source_timestamps[indices] - target_timestamps)
    )
    return np.where(choose_previous, previous, indices).astype(np.int64)


def _ratio_indices(source_len: int, target_len: int) -> np.ndarray:
    if source_len <= 0:
        return np.zeros(target_len, dtype=np.int64)
    if target_len <= 1:
        return np.zeros(target_len, dtype=np.int64)
    return np.round(np.linspace(0, source_len - 1, target_len)).astype(np.int64)


def _state_times(handle: h5py.File, n_state: int) -> tuple[np.ndarray, str, bool]:
    values, key = _read_array(
        handle,
        ("timestamps/state_episode", "timestamps/state/state_episode", "timestamps/state"),
    )
    if values is None or len(values) != n_state or not np.isfinite(values).all():
        return np.arange(n_state, dtype=np.float64), "state_index", False
    return values.astype(np.float64), key, True


def _force_norm_for_states(
    handle: h5py.File,
    state_times: np.ndarray,
    timestamps_are_real: bool,
) -> tuple[np.ndarray, str]:
    force, force_key = _read_array(handle, ("observations/ft_wrench",))
    if force is None:
        return np.full(len(state_times), np.nan, dtype=np.float64), "missing"
    force = np.asarray(force, dtype=np.float64)
    if force.ndim != 2 or force.shape[1] < 3:
        return np.full(len(state_times), np.nan, dtype=np.float64), f"invalid:{force_key}"
    force_norm_all = np.linalg.norm(force[:, :3], axis=1)
    force_times, force_time_key = _read_array(
        handle,
        ("timestamps/force_episode", "timestamps/force/force_episode", "timestamps/force"),
    )
    if (
        timestamps_are_real
        and force_times is not None
        and len(force_times) == len(force)
        and np.isfinite(force_times).all()
    ):
        indices = _nearest_indices(force_times.astype(np.float64), state_times)
        return force_norm_all[indices], f"nearest:{force_time_key}"
    indices = _ratio_indices(len(force), len(state_times))
    return force_norm_all[indices], "nearest_ratio"


def _duration_and_frame_dt(state_times: np.ndarray, timestamps_are_real: bool) -> tuple[float, np.ndarray]:
    if len(state_times) == 0:
        return float("nan"), np.asarray([], dtype=np.float64)
    if not timestamps_are_real:
        return float(max(0, len(state_times) - 1)), np.ones(len(state_times), dtype=np.float64)
    duration = float(state_times[-1] - state_times[0]) if len(state_times) > 1 else 0.0
    if len(state_times) <= 1:
        return duration, np.zeros(len(state_times), dtype=np.float64)
    deltas = np.diff(state_times)
    median_dt = float(np.median(deltas[np.isfinite(deltas)])) if np.isfinite(deltas).any() else 0.0
    frame_dt = np.concatenate([deltas, np.asarray([median_dt], dtype=np.float64)])
    frame_dt = np.where(np.isfinite(frame_dt) & (frame_dt >= 0), frame_dt, 0.0)
    return duration, frame_dt


def _resolve_required_ids(mujoco, model, object_type, names: Sequence[str], kind: str) -> list[int]:
    ids = [mujoco.mj_name2id(model, object_type, name) for name in names]
    missing = [name for name, object_id in zip(names, ids) if object_id < 0]
    if missing:
        raise ValueError(f"missing required {kind}: {', '.join(missing)}")
    return ids


def _print_model_probe(mujoco, model, site_ids: Sequence[int]) -> None:
    print(f"model_nq={model.nq} model_nv={model.nv} model_nu={model.nu}")
    for name, site_id in zip(SITE_NAMES, site_ids):
        print(f"{name}_id={site_id}")
    for camera_name in CAMERA_NAMES:
        camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        print(f"{camera_name}_camera_id={camera_id}")
    for sensor_name in SENSOR_NAMES:
        sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name)
        print(f"{sensor_name}_sensor_id={sensor_id}")
    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, GEOM_NAME)
    print(f"{GEOM_NAME}_geom_id={geom_id}")
    if geom_id >= 0:
        print(
            f"{GEOM_NAME}_geom_size="
            f"{np.array2string(model.geom_size[geom_id], precision=9, separator=',')}"
        )


def _set_replay_qpos(model, data, joint_qposadr: np.ndarray, qpos: np.ndarray) -> None:
    data.qpos[joint_qposadr] = qpos
    if model.nu >= len(qpos):
        data.ctrl[: len(qpos)] = qpos


def _replay_episode(
    path: Path,
    episode_index: int,
    mujoco,
    model,
    data,
    joint_qposadr: np.ndarray,
    site_ids: Sequence[int],
    hole_axis: np.ndarray,
    save_frame_rows: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    frame_rows: list[dict[str, Any]] = []
    summary = _empty_episode_summary(path, episode_index)
    try:
        with h5py.File(path, "r") as handle:
            if "observations/joint_pos" not in handle:
                raise KeyError("missing observations/joint_pos")
            joint_pos = np.asarray(handle["observations/joint_pos"], dtype=np.float64)
            if joint_pos.ndim != 2 or joint_pos.shape[1] != len(joint_qposadr):
                raise ValueError(
                    f"observations/joint_pos must have shape [N, {len(joint_qposadr)}], "
                    f"got {joint_pos.shape}"
                )
            if not np.isfinite(joint_pos).all():
                print(f"warning: {path} contains non-finite joint_pos", file=sys.stderr)

            n_state = len(joint_pos)
            state_times, state_time_key, timestamps_are_real = _state_times(handle, n_state)
            duration, frame_dt = _duration_and_frame_dt(state_times, timestamps_are_real)
            force_norm, force_alignment = _force_norm_for_states(handle, state_times, timestamps_are_real)
            if not np.isfinite(force_norm).all():
                print(f"warning: {path} contains unavailable or non-finite force norm", file=sys.stderr)

            peg_tip_pos = np.empty((n_state, 3), dtype=np.float64)
            hole_center_pos = np.empty((n_state, 3), dtype=np.float64)
            for state_index, qpos in enumerate(joint_pos):
                _set_replay_qpos(model, data, joint_qposadr, qpos)
                mujoco.mj_forward(model, data)
                peg_tip_pos[state_index] = data.site_xpos[site_ids[0]]
                hole_center_pos[state_index] = data.site_xpos[site_ids[1]]

            error_vec = hole_center_pos - peg_tip_pos
            distance = np.linalg.norm(error_vec, axis=1)
            axial_error = error_vec @ hole_axis
            lateral_vec = error_vec - axial_error[:, None] * hole_axis[None, :]
            lateral_error = np.linalg.norm(lateral_vec, axis=1)
            qpos_delta_next = np.full(n_state, np.nan, dtype=np.float64)
            qpos_delta_prev = np.full(n_state, np.nan, dtype=np.float64)
            if n_state > 1:
                qpos_step = np.linalg.norm(np.diff(joint_pos, axis=0), axis=1)
                qpos_delta_next[:-1] = qpos_step
                qpos_delta_prev[1:] = qpos_step

            summary.update(
                _episode_metrics(
                    path=path,
                    n_state=n_state,
                    duration=duration,
                    state_times=state_times,
                    timestamps_are_real=timestamps_are_real,
                    state_time_key=state_time_key,
                    force_alignment=force_alignment,
                    distance=distance,
                    axial_error=axial_error,
                    lateral_error=lateral_error,
                    lateral_vec=lateral_vec,
                    error_vec=error_vec,
                    peg_tip_pos=peg_tip_pos,
                    hole_center_pos=hole_center_pos,
                    force_norm=force_norm,
                    qpos_delta_next=qpos_delta_next,
                    qpos_delta_prev=qpos_delta_prev,
                    frame_dt=frame_dt,
                )
            )
            summary["ok"] = True

            if episode_index == 0:
                print(f"first_episode_path={path}")
                print(f"first_episode_initial_peg_tip={_vector_text(peg_tip_pos[0])}")
                print(f"first_episode_final_peg_tip={_vector_text(peg_tip_pos[-1])}")
                print(f"first_episode_initial_hole_center={_vector_text(hole_center_pos[0])}")
                print(f"first_episode_final_hole_center={_vector_text(hole_center_pos[-1])}")

            if save_frame_rows:
                frame_rows = _frame_rows(
                    episode_index=episode_index,
                    path=path,
                    state_times=state_times,
                    distance=distance,
                    axial_error=axial_error,
                    lateral_error=lateral_error,
                    lateral_vec=lateral_vec,
                    error_vec=error_vec,
                    peg_tip_pos=peg_tip_pos,
                    hole_center_pos=hole_center_pos,
                    force_norm=force_norm,
                    qpos_delta_next=qpos_delta_next,
                    qpos_delta_prev=qpos_delta_prev,
                )
    except Exception as error:
        summary["ok"] = False
        summary["failure"] = str(error)
        print(f"error: failed episode {path}: {error}", file=sys.stderr)
    return summary, frame_rows


def _empty_episode_summary(path: Path, episode_index: int) -> dict[str, Any]:
    row: dict[str, Any] = {
        "episode_index": episode_index,
        "episode_path": str(path),
        "ok": False,
        "failure": "",
    }
    metric_keys = (
        "num_state_frames",
        "duration_sec",
        "initial_peg_to_hole_dist",
        "final_peg_to_hole_dist",
        "min_peg_to_hole_dist",
        "min_peg_to_hole_dist_step",
        "min_peg_to_hole_dist_time",
        "initial_axial_error",
        "final_axial_error",
        "min_axial_error",
        "initial_lateral_error",
        "final_lateral_error",
        "min_lateral_error",
        "min_lateral_error_axial_error",
        "min_lateral_error_time",
        "min_axial_error_lateral_error",
        "min_axial_error_time",
        "final_peg_tip_x",
        "final_peg_tip_y",
        "final_peg_tip_z",
        "final_hole_center_x",
        "final_hole_center_y",
        "final_hole_center_z",
        "final_error_x",
        "final_error_y",
        "final_error_z",
        "max_force_norm",
        "mean_force_norm",
        "final_force_norm",
        "final_qpos_delta_prev",
    )
    for key in metric_keys:
        row[key] = ""
    for threshold in THRESHOLDS:
        row[f"reached_dist_lt_{_threshold_label(threshold)}"] = False
    for band_name, _, _ in DISTANCE_BANDS:
        row[f"mean_qpos_delta_next_dist_{band_name}"] = ""
        row[f"frame_count_dist_{band_name}"] = 0
        row[f"time_sec_dist_{band_name}"] = ""
    return row


def _episode_metrics(
    path: Path,
    n_state: int,
    duration: float,
    state_times: np.ndarray,
    timestamps_are_real: bool,
    state_time_key: str,
    force_alignment: str,
    distance: np.ndarray,
    axial_error: np.ndarray,
    lateral_error: np.ndarray,
    lateral_vec: np.ndarray,
    error_vec: np.ndarray,
    peg_tip_pos: np.ndarray,
    hole_center_pos: np.ndarray,
    force_norm: np.ndarray,
    qpos_delta_next: np.ndarray,
    qpos_delta_prev: np.ndarray,
    frame_dt: np.ndarray,
) -> dict[str, Any]:
    min_dist_index = int(np.nanargmin(distance))
    min_lateral_index = int(np.nanargmin(lateral_error))
    min_axial_index = int(np.nanargmin(axial_error))
    row: dict[str, Any] = {
        "failure": "",
        "num_state_frames": n_state,
        "duration_sec": duration,
        "state_time_key": state_time_key,
        "timestamps_are_real": timestamps_are_real,
        "force_alignment": force_alignment,
        "initial_peg_to_hole_dist": float(distance[0]),
        "final_peg_to_hole_dist": float(distance[-1]),
        "min_peg_to_hole_dist": float(distance[min_dist_index]),
        "min_peg_to_hole_dist_step": min_dist_index,
        "min_peg_to_hole_dist_time": float(state_times[min_dist_index]),
        "initial_axial_error": float(axial_error[0]),
        "final_axial_error": float(axial_error[-1]),
        "min_axial_error": float(axial_error[min_axial_index]),
        "initial_lateral_error": float(lateral_error[0]),
        "final_lateral_error": float(lateral_error[-1]),
        "min_lateral_error": float(lateral_error[min_lateral_index]),
        "min_lateral_error_axial_error": float(axial_error[min_lateral_index]),
        "min_lateral_error_time": float(state_times[min_lateral_index]),
        "min_axial_error_lateral_error": float(lateral_error[min_axial_index]),
        "min_axial_error_time": float(state_times[min_axial_index]),
        "final_peg_tip_x": float(peg_tip_pos[-1, 0]),
        "final_peg_tip_y": float(peg_tip_pos[-1, 1]),
        "final_peg_tip_z": float(peg_tip_pos[-1, 2]),
        "final_hole_center_x": float(hole_center_pos[-1, 0]),
        "final_hole_center_y": float(hole_center_pos[-1, 1]),
        "final_hole_center_z": float(hole_center_pos[-1, 2]),
        "final_error_x": float(error_vec[-1, 0]),
        "final_error_y": float(error_vec[-1, 1]),
        "final_error_z": float(error_vec[-1, 2]),
        "max_force_norm": _nan_float(np.nanmax(force_norm)),
        "mean_force_norm": _nan_float(np.nanmean(force_norm)),
        "final_force_norm": _nan_float(force_norm[-1]),
        "final_qpos_delta_prev": _nan_float(qpos_delta_prev[-1]),
    }
    for threshold in THRESHOLDS:
        row[f"reached_dist_lt_{_threshold_label(threshold)}"] = bool(
            np.nanmin(distance) < threshold
        )
    for band_name, low, high in DISTANCE_BANDS:
        mask = (distance > low) & (distance <= high)
        row[f"mean_qpos_delta_next_dist_{band_name}"] = _nanmean_or_nan(
            qpos_delta_next[mask]
        )
        row[f"frame_count_dist_{band_name}"] = int(mask.sum())
        row[f"time_sec_dist_{band_name}"] = _nan_float(np.nansum(frame_dt[mask]))
    return row


def _frame_rows(
    episode_index: int,
    path: Path,
    state_times: np.ndarray,
    distance: np.ndarray,
    axial_error: np.ndarray,
    lateral_error: np.ndarray,
    lateral_vec: np.ndarray,
    error_vec: np.ndarray,
    peg_tip_pos: np.ndarray,
    hole_center_pos: np.ndarray,
    force_norm: np.ndarray,
    qpos_delta_next: np.ndarray,
    qpos_delta_prev: np.ndarray,
) -> list[dict[str, Any]]:
    rows = []
    for index in range(len(distance)):
        rows.append(
            {
                "episode_index": episode_index,
                "episode_path": str(path),
                "state_index": index,
                "t_state": float(state_times[index]),
                "peg_to_hole_dist": float(distance[index]),
                "axial_error": float(axial_error[index]),
                "lateral_error": float(lateral_error[index]),
                "error_x": float(error_vec[index, 0]),
                "error_y": float(error_vec[index, 1]),
                "error_z": float(error_vec[index, 2]),
                "lateral_x": float(lateral_vec[index, 0]),
                "lateral_y": float(lateral_vec[index, 1]),
                "lateral_z": float(lateral_vec[index, 2]),
                "peg_tip_x": float(peg_tip_pos[index, 0]),
                "peg_tip_y": float(peg_tip_pos[index, 1]),
                "peg_tip_z": float(peg_tip_pos[index, 2]),
                "hole_center_x": float(hole_center_pos[index, 0]),
                "hole_center_y": float(hole_center_pos[index, 1]),
                "hole_center_z": float(hole_center_pos[index, 2]),
                "force_norm": _nan_float(force_norm[index]),
                "qpos_delta_next": _nan_float(qpos_delta_next[index]),
                "qpos_delta_prev": _nan_float(qpos_delta_prev[index]),
            }
        )
    return rows


def _threshold_label(value: float) -> str:
    return f"{value:g}".replace(".", "p").replace("-", "m")


def _nan_float(value: Any) -> float:
    value = float(value)
    return value if math.isfinite(value) else float("nan")


def _nanmean_or_nan(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return float("nan")
    return float(np.mean(finite))


def _vector_text(values: np.ndarray) -> str:
    return np.array2string(np.asarray(values), precision=9, separator=",")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _finite_values(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    values = []
    for row in rows:
        if not row.get("ok", False):
            continue
        try:
            value = float(row[key])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(value):
            values.append(value)
    return np.asarray(values, dtype=np.float64)


def _stats(values: np.ndarray) -> dict[str, float]:
    if len(values) == 0:
        return {"min": float("nan"), "mean": float("nan"), "median": float("nan"), "max": float("nan")}
    return {
        "min": float(np.min(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "max": float(np.max(values)),
    }


def _dataset_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ok_rows = [row for row in rows if row.get("ok", False)]
    summary: dict[str, Any] = {
        "episode_count": len(rows),
        "ok_episode_count": len(ok_rows),
        "failed_episode_count": len(rows) - len(ok_rows),
        "final_distance": _stats(_finite_values(rows, "final_peg_to_hole_dist")),
        "min_distance": _stats(_finite_values(rows, "min_peg_to_hole_dist")),
        "final_axial_error": _stats(_finite_values(rows, "final_axial_error")),
        "final_lateral_error": _stats(_finite_values(rows, "final_lateral_error")),
        "mean_time_spent_in_distance_bands": {},
    }
    for threshold in THRESHOLDS:
        key = f"reached_dist_lt_{_threshold_label(threshold)}"
        reached = [bool(row.get(key, False)) for row in ok_rows]
        summary[f"fraction_reaching_lt_{_threshold_label(threshold)}"] = (
            float(sum(reached) / len(reached)) if reached else float("nan")
        )
    for band_name, _, _ in DISTANCE_BANDS:
        summary["mean_time_spent_in_distance_bands"][band_name] = _nanmean_or_nan(
            _finite_values(rows, f"time_sec_dist_{band_name}")
        )
    for band_name in ("0p03_0p05", "0p02_0p03", "0p01_0p02"):
        summary[f"mean_qpos_delta_next_dist_{band_name}"] = _nanmean_or_nan(
            _finite_values(rows, f"mean_qpos_delta_next_dist_{band_name}")
        )
    failures = [
        {"episode_path": row["episode_path"], "failure": row.get("failure", "")}
        for row in rows
        if not row.get("ok", False)
    ]
    summary["failures"] = failures
    return summary


def _print_dataset_summary(summary: dict[str, Any]) -> None:
    print("\nDataset Summary")
    print("---------------")
    print(f"episode_count={summary['episode_count']}")
    print(f"ok_episode_count={summary['ok_episode_count']}")
    print(f"failed_episode_count={summary['failed_episode_count']}")
    for key in ("final_distance", "min_distance", "final_axial_error", "final_lateral_error"):
        stats = summary[key]
        print(
            f"{key}: min={stats['min']:.9g} mean={stats['mean']:.9g} "
            f"median={stats['median']:.9g} max={stats['max']:.9g}"
        )
    for threshold in THRESHOLDS:
        label = _threshold_label(threshold)
        print(f"fraction_reaching_lt_{label}={summary[f'fraction_reaching_lt_{label}']:.9g}")
    for band_name, value in summary["mean_time_spent_in_distance_bands"].items():
        print(f"mean_time_sec_dist_{band_name}={value:.9g}")
    for band_name in ("0p03_0p05", "0p02_0p03", "0p01_0p02"):
        print(
            f"mean_qpos_delta_next_dist_{band_name}="
            f"{summary[f'mean_qpos_delta_next_dist_{band_name}']:.9g}"
        )


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as json_file:
        json.dump(payload, json_file, indent=2, allow_nan=True)


def _plot_outputs(output_dir: Path, summary_rows: list[dict[str, Any]], frame_rows: list[dict[str, Any]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("warning: matplotlib is unavailable; skipping plots", file=sys.stderr)
        return

    ok_rows = [row for row in summary_rows if row.get("ok", False)]
    output_dir.mkdir(parents=True, exist_ok=True)

    def hist(key: str, filename: str, title: str) -> None:
        values = _finite_values(ok_rows, key)
        plt.figure(figsize=(7, 4))
        plt.hist(values, bins=30)
        plt.xlabel(key)
        plt.ylabel("episodes")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(output_dir / filename)
        plt.close()

    hist("final_peg_to_hole_dist", "final_distance_hist.png", "Final peg-to-hole distance")
    hist("min_peg_to_hole_dist", "min_distance_hist.png", "Minimum peg-to-hole distance")

    plt.figure(figsize=(8, 5))
    for row in ok_rows[:100]:
        episode_index = int(row["episode_index"])
        trajectory = [
            frame
            for frame in frame_rows
            if int(frame["episode_index"]) == episode_index
        ]
        if not trajectory:
            continue
        plt.plot(
            [frame["t_state"] for frame in trajectory],
            [frame["peg_to_hole_dist"] for frame in trajectory],
            alpha=0.25,
        )
    plt.xlabel("episode time or state index")
    plt.ylabel("peg-to-hole distance [m]")
    plt.title("Replay distance trajectories")
    plt.tight_layout()
    plt.savefig(output_dir / "distance_trajectories.png")
    plt.close()

    plt.figure(figsize=(6, 5))
    final_axial = _finite_values(ok_rows, "final_axial_error")
    final_lateral = _finite_values(ok_rows, "final_lateral_error")
    plt.scatter(final_axial, final_lateral, s=20)
    plt.xlabel("final axial error [m]")
    plt.ylabel("final lateral error [m]")
    plt.title("Final axial/lateral replay error")
    plt.tight_layout()
    plt.savefig(output_dir / "axial_lateral_scatter.png")
    plt.close()

    band_names = [band[0] for band in DISTANCE_BANDS]
    means = [
        _nanmean_or_nan(_finite_values(ok_rows, f"time_sec_dist_{band_name}"))
        for band_name in band_names
    ]
    plt.figure(figsize=(8, 4))
    plt.bar(band_names, means)
    plt.ylabel("mean time [s]")
    plt.title("Mean time spent in distance bands")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / "band_time_bar.png")
    plt.close()


def audit(args: argparse.Namespace) -> int:
    mujoco = _load_mujoco()
    model = mujoco.MjModel.from_xml_path(str(args.model_xml))
    data = mujoco.MjData(model)
    joint_ids = _resolve_required_ids(
        mujoco, model, mujoco.mjtObj.mjOBJ_JOINT, args.joint_names, "joints"
    )
    site_ids = _resolve_required_ids(mujoco, model, mujoco.mjtObj.mjOBJ_SITE, SITE_NAMES, "sites")
    joint_qposadr = np.asarray(model.jnt_qposadr[joint_ids], dtype=np.int64)
    _print_model_probe(mujoco, model, site_ids)

    episodes = _discover_episodes(args.data_dir, args.episode_list)
    if args.max_episodes is not None:
        episodes = episodes[: args.max_episodes]
    if not episodes:
        print("error: no episodes found", file=sys.stderr)
        return 2

    summary_rows: list[dict[str, Any]] = []
    frame_rows: list[dict[str, Any]] = []
    for episode_index, episode_path in enumerate(episodes):
        if episode_index % 10 == 0:
            print(f"auditing_episode={episode_index + 1}/{len(episodes)} path={episode_path}")
        summary, frames = _replay_episode(
            path=episode_path,
            episode_index=episode_index,
            mujoco=mujoco,
            model=model,
            data=data,
            joint_qposadr=joint_qposadr,
            site_ids=site_ids,
            hole_axis=args.hole_axis_world,
            save_frame_rows=args.save_frame_csv or args.plot,
        )
        summary_rows.append(summary)
        frame_rows.extend(frames)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "episode_summary.csv"
    dataset_summary_path = args.output_dir / "dataset_summary.json"
    _write_csv(summary_path, summary_rows)
    dataset_summary = _dataset_summary(summary_rows)
    _save_json(dataset_summary_path, dataset_summary)
    if args.save_frame_csv:
        _write_csv(args.output_dir / "frame_metrics.csv", frame_rows)
    if args.plot:
        _plot_outputs(args.output_dir, summary_rows, frame_rows)
    _print_dataset_summary(dataset_summary)
    print(f"saved_episode_summary={summary_path}")
    print(f"saved_dataset_summary={dataset_summary_path}")
    if args.save_frame_csv:
        print(f"saved_frame_metrics={args.output_dir / 'frame_metrics.csv'}")
    return 0 if dataset_summary["failed_episode_count"] == 0 else 1


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay HDF5 joint_pos in MuJoCo and reconstruct peg-tip task error.",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("mujoco_data/peg_hole_fixed_insertion"))
    parser.add_argument("--episode-list", type=Path, default=None)
    parser.add_argument("--model-xml", type=Path, default=Path("../arm_teleop/model/pangu_all_right.xml"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--hole-axis-world", type=float, nargs=3, default=(0.0, -1.0, 0.0))
    parser.add_argument("--joint-names", nargs="+", default=tuple(f"joint_{index}" for index in range(1, 8)))
    parser.add_argument("--save-frame-csv", action="store_true")
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--plot", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    args.model_xml = args.model_xml.expanduser()
    if not args.model_xml.is_absolute():
        args.model_xml = (REPO_ROOT / args.model_xml).resolve()
    if not args.model_xml.is_file():
        print(f"error: model XML does not exist: {args.model_xml}", file=sys.stderr)
        return 2
    args.output_dir = args.output_dir.expanduser()
    if args.max_episodes is not None and args.max_episodes <= 0:
        print("error: --max-episodes must be positive", file=sys.stderr)
        return 2
    if len(args.joint_names) != 7:
        print("error: --joint-names must provide exactly 7 names", file=sys.stderr)
        return 2
    args.hole_axis_world = np.asarray(args.hole_axis_world, dtype=np.float64)
    axis_norm = float(np.linalg.norm(args.hole_axis_world))
    if not np.isfinite(args.hole_axis_world).all() or axis_norm <= 0:
        print("error: --hole-axis-world must be a finite nonzero vector", file=sys.stderr)
        return 2
    args.hole_axis_world = args.hole_axis_world / axis_norm
    print(f"hole_axis_world={_vector_text(args.hole_axis_world)}")
    try:
        return audit(args)
    except Exception as error:
        print(f"error: replay audit failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
