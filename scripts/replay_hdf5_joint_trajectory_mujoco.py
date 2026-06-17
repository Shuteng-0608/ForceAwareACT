#!/usr/bin/env python3
"""Dynamically replay HDF5 joint trajectories or command labels in MuJoCo.

This is a read-only audit tool for demonstrations. It does not modify HDF5
files, training code, rollout code, or the arm_teleop project.
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


JOINT_NAMES = tuple(f"joint_{index}" for index in range(1, 8))
ACTUATOR_NAMES = tuple(f"motor_joint_{index}" for index in range(1, 8))
CAMERA_NAMES = ("ee_cam", "base_top_cam")
SITE_NAMES = ("peg_tip_site", "hole_center_site", "ft_sensor_site")
SENSOR_NAMES = ("peg_ft_force", "peg_ft_torque")
GEOM_NAME = "cylindrical_peg"
THRESHOLDS = (0.05, 0.03, 0.02, 0.01, 0.005)


def _load_mujoco():
    try:
        import mujoco
    except ImportError as error:
        raise RuntimeError("the 'mujoco' Python package is required for dynamic replay") from error
    return mujoco


def _read_array(handle: h5py.File, keys: Sequence[str]) -> tuple[Optional[np.ndarray], str]:
    for key in keys:
        if key in handle:
            return np.asarray(handle[key]), key
    return None, ""


def _discover_episodes(
    data_dir: Path,
    episode_list: Optional[Path],
    episode_path: Optional[Path],
) -> list[Path]:
    if episode_path is not None:
        return resolve_episode_paths([episode_path], None, project_root=REPO_ROOT)
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


def _resolve_ids(mujoco, model, object_type, names: Sequence[str], kind: str) -> np.ndarray:
    ids = np.asarray(
        [mujoco.mj_name2id(model, object_type, name) for name in names],
        dtype=np.int64,
    )
    missing = [name for name, object_id in zip(names, ids) if object_id < 0]
    if missing:
        raise ValueError(f"missing required {kind}: {', '.join(missing)}")
    return ids


def _resolve_optional_ids(mujoco, model, object_type, names: Sequence[str], kind: str) -> np.ndarray:
    ids = np.asarray(
        [mujoco.mj_name2id(model, object_type, name) for name in names],
        dtype=np.int64,
    )
    missing = [name for name, object_id in zip(names, ids) if object_id < 0]
    if missing:
        print(f"warning: missing optional {kind}: {', '.join(missing)}", file=sys.stderr)
    return ids


def _sensor_slice(model, sensor_id: int, expected_dim: int, name: str) -> Optional[slice]:
    if sensor_id < 0:
        return None
    dimension = int(model.sensor_dim[sensor_id])
    if dimension != expected_dim:
        print(
            f"warning: sensor {name} has dimension {dimension}, expected {expected_dim}; ignoring it",
            file=sys.stderr,
        )
        return None
    start = int(model.sensor_adr[sensor_id])
    return slice(start, start + dimension)


def _read_mujoco_force_norm(data, force_slice: Optional[slice], torque_slice: Optional[slice]) -> float:
    if force_slice is None or torque_slice is None:
        return float("nan")
    force = np.asarray(data.sensordata[force_slice], dtype=np.float64)
    torque = np.asarray(data.sensordata[torque_slice], dtype=np.float64)
    if not np.isfinite(force).all() or not np.isfinite(torque).all():
        return float("nan")
    return float(np.linalg.norm(force))


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


def _recorded_force_norm_for_targets(
    handle: h5py.File,
    state_times: np.ndarray,
    timestamps_are_real: bool,
) -> tuple[np.ndarray, str]:
    force, force_key = _read_array(handle, ("observations/ft_wrench",))
    if force is None:
        return np.full(len(state_times), np.nan, dtype=np.float64), "missing"
    force = np.asarray(force, dtype=np.float64)
    if not np.isfinite(force).all():
        print("warning: HDF5 ft_wrench contains non-finite values", file=sys.stderr)
    if force.ndim != 2 or force.shape[1] < 3:
        return np.full(len(state_times), np.nan, dtype=np.float64), f"invalid:{force_key}"
    force_norm = np.linalg.norm(force[:, :3], axis=1)
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
        return force_norm[indices], f"nearest:{force_time_key}"
    indices = _ratio_indices(len(force), len(state_times))
    return force_norm[indices], "nearest_ratio"


def _hold_dt(
    index: int,
    state_times: np.ndarray,
    timestamps_are_real: bool,
    use_recorded_timestamps: bool,
    control_rate_hz: float,
) -> float:
    fallback = 1.0 / control_rate_hz
    if not use_recorded_timestamps or not timestamps_are_real or index >= len(state_times) - 1:
        return fallback
    dt = float(state_times[index + 1] - state_times[index])
    if not math.isfinite(dt) or dt <= 0:
        return fallback
    return dt


def _threshold_label(value: float) -> str:
    return f"{value:g}".replace(".", "p").replace("-", "m")


def _nan_float(value: Any) -> float:
    value = float(value)
    return value if math.isfinite(value) else float("nan")


def _nanmean_or_nan(values: Sequence[float] | np.ndarray) -> float:
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    if len(finite) == 0:
        return float("nan")
    return float(np.mean(finite))


def _nanmin_or_nan(values: Sequence[float] | np.ndarray) -> float:
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    if len(finite) == 0:
        return float("nan")
    return float(np.min(finite))


def _nanmax_or_nan(values: Sequence[float] | np.ndarray) -> float:
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    if len(finite) == 0:
        return float("nan")
    return float(np.max(finite))


def _stats(values: np.ndarray) -> dict[str, float]:
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return {"min": float("nan"), "mean": float("nan"), "median": float("nan"), "max": float("nan")}
    return {
        "min": float(np.min(finite)),
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "max": float(np.max(finite)),
    }


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


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as json_file:
        json.dump(payload, json_file, indent=2, allow_nan=True)


def _init_video_writers(output_dir: Path, episode_name: str, fps: int):
    try:
        import imageio.v2 as imageio
    except ImportError as error:
        raise RuntimeError(
            "video rendering requires imageio and imageio-ffmpeg; install them with: "
            ".venv/bin/python -m pip install imageio imageio-ffmpeg"
        ) from error
    video_dir = output_dir / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    return {
        camera_name: imageio.get_writer(video_dir / f"{episode_name}_{camera_name}.mp4", fps=fps)
        for camera_name in CAMERA_NAMES
    }


def _close_video_writers(writers: dict[str, Any]) -> None:
    for writer in writers.values():
        writer.close()


def _write_video_frames(renderer, data, camera_ids: np.ndarray, writers: dict[str, Any]) -> None:
    for camera_name, camera_id in zip(CAMERA_NAMES, camera_ids):
        if camera_id < 0:
            continue
        renderer.update_scene(data, camera=int(camera_id))
        writers[camera_name].append_data(np.asarray(renderer.render(), dtype=np.uint8))


def _print_model_probe(
    mujoco,
    model,
    joint_ids: np.ndarray,
    actuator_ids: np.ndarray,
    site_ids: np.ndarray,
    sensor_ids: np.ndarray,
) -> None:
    print(f"model_timestep={model.opt.timestep}")
    print(f"model_nq={model.nq} model_nv={model.nv} model_nu={model.nu}")
    print(f"joint_ids={joint_ids.tolist()}")
    print(f"actuator_ids={actuator_ids.tolist()}")
    for name, site_id in zip(SITE_NAMES, site_ids):
        print(f"{name}_site_id={int(site_id)}")
    for name, sensor_id in zip(SENSOR_NAMES, sensor_ids):
        print(f"{name}_sensor_id={int(sensor_id)}")
    for camera_name in CAMERA_NAMES:
        camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        print(f"{camera_name}_camera_id={camera_id}")
    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, GEOM_NAME)
    print(f"{GEOM_NAME}_geom_id={geom_id}")
    if geom_id >= 0:
        print(
            f"{GEOM_NAME}_geom_size="
            f"{np.array2string(model.geom_size[geom_id], precision=9, separator=',')}"
        )


def _empty_summary(path: Path, episode_index: int, command_field: str) -> dict[str, Any]:
    row: dict[str, Any] = {
        "episode_index": episode_index,
        "episode_path": str(path),
        "command_field": command_field,
        "ok": False,
        "failure": "",
        "stop_reason": "failed",
    }
    for key in (
        "playback_max_joint_velocity",
        "num_targets",
        "replay_duration_sec",
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
        "max_force_norm",
        "mean_force_norm",
        "final_force_norm",
        "max_qpos_tracking_error_norm",
        "mean_qpos_tracking_error_norm",
        "final_qpos_tracking_error_norm",
        "mean_applied_ctrl_to_actual_error_norm",
        "max_applied_ctrl_to_actual_error_norm",
        "mean_target_to_applied_ctrl_error_norm",
        "max_target_to_applied_ctrl_error_norm",
        "ctrl_clipped_any",
        "ctrl_clipped_count",
    ):
        row[key] = ""
    for threshold in THRESHOLDS:
        label = _threshold_label(threshold)
        row[f"reached_dist_lt_{label}"] = False
        row[f"count_dist_lt_{label}"] = 0
        row[f"fraction_dist_lt_{label}"] = ""
    return row


def _episode_metrics(
    summary: dict[str, Any],
    frame_rows: list[dict[str, Any]],
    stop_reason: str,
) -> dict[str, Any]:
    if not frame_rows:
        raise ValueError("episode produced no replay frames")
    distance = np.asarray([row["peg_to_hole_dist"] for row in frame_rows], dtype=np.float64)
    axial = np.asarray([row["peg_to_hole_axial_error"] for row in frame_rows], dtype=np.float64)
    lateral = np.asarray([row["peg_to_hole_lateral_error"] for row in frame_rows], dtype=np.float64)
    force = np.asarray([row["force_norm"] for row in frame_rows], dtype=np.float64)
    tracking = np.asarray([row["qpos_tracking_error_norm"] for row in frame_rows], dtype=np.float64)
    applied_error = np.asarray(
        [row["applied_ctrl_to_actual_error_norm"] for row in frame_rows],
        dtype=np.float64,
    )
    target_to_applied = np.asarray(
        [row["target_to_applied_ctrl_error_norm"] for row in frame_rows],
        dtype=np.float64,
    )
    times = np.asarray([row["time"] for row in frame_rows], dtype=np.float64)
    clipped = np.asarray([bool(row["ctrl_clipped"]) for row in frame_rows], dtype=bool)
    min_dist_index = int(np.nanargmin(distance))
    summary.update(
        {
            "ok": stop_reason != "failed",
            "failure": "",
            "stop_reason": stop_reason,
            "num_targets": len(frame_rows),
            "replay_duration_sec": _nan_float(times[-1]),
            "initial_peg_to_hole_dist": _nan_float(distance[0]),
            "final_peg_to_hole_dist": _nan_float(distance[-1]),
            "min_peg_to_hole_dist": _nan_float(distance[min_dist_index]),
            "min_peg_to_hole_dist_step": min_dist_index,
            "min_peg_to_hole_dist_time": _nan_float(times[min_dist_index]),
            "initial_axial_error": _nan_float(axial[0]),
            "final_axial_error": _nan_float(axial[-1]),
            "min_axial_error": _nanmin_or_nan(axial),
            "initial_lateral_error": _nan_float(lateral[0]),
            "final_lateral_error": _nan_float(lateral[-1]),
            "min_lateral_error": _nanmin_or_nan(lateral),
            "max_force_norm": _nanmax_or_nan(force),
            "mean_force_norm": _nanmean_or_nan(force),
            "final_force_norm": _nan_float(force[-1]),
            "max_qpos_tracking_error_norm": _nanmax_or_nan(tracking),
            "mean_qpos_tracking_error_norm": _nanmean_or_nan(tracking),
            "final_qpos_tracking_error_norm": _nan_float(tracking[-1]),
            "mean_applied_ctrl_to_actual_error_norm": _nanmean_or_nan(applied_error),
            "max_applied_ctrl_to_actual_error_norm": _nanmax_or_nan(applied_error),
            "mean_target_to_applied_ctrl_error_norm": _nanmean_or_nan(target_to_applied),
            "max_target_to_applied_ctrl_error_norm": _nanmax_or_nan(target_to_applied),
            "ctrl_clipped_any": bool(clipped.any()),
            "ctrl_clipped_count": int(clipped.sum()),
        }
    )
    for threshold in THRESHOLDS:
        label = _threshold_label(threshold)
        mask = distance < threshold
        summary[f"reached_dist_lt_{label}"] = bool(mask.any())
        summary[f"count_dist_lt_{label}"] = int(mask.sum())
        summary[f"fraction_dist_lt_{label}"] = float(mask.sum() / len(mask))
    return summary


def _make_frame_row(
    episode_index: int,
    episode_path: Path,
    command_field: str,
    target_index: int,
    sim_time: float,
    replay_target: np.ndarray,
    applied_ctrl: np.ndarray,
    qpos: np.ndarray,
    peg_tip_pos: np.ndarray,
    hole_center_pos: np.ndarray,
    hole_axis: np.ndarray,
    force_norm: float,
    recorded_force_norm: float,
    ctrl_clipped: bool,
) -> dict[str, Any]:
    error_vec = hole_center_pos - peg_tip_pos
    distance = float(np.linalg.norm(error_vec))
    axial = float(error_vec @ hole_axis)
    lateral_vec = error_vec - axial * hole_axis
    row: dict[str, Any] = {
        "episode_index": episode_index,
        "episode_path": str(episode_path),
        "command_field": command_field,
        "time": sim_time,
        "target_index": target_index,
        "qpos_tracking_error_norm": float(np.linalg.norm(qpos - applied_ctrl)),
        "target_to_actual_error_norm": float(np.linalg.norm(replay_target - qpos)),
        "applied_ctrl_to_actual_error_norm": float(np.linalg.norm(applied_ctrl - qpos)),
        "target_to_applied_ctrl_error_norm": float(np.linalg.norm(replay_target - applied_ctrl)),
        "peg_tip_x": float(peg_tip_pos[0]),
        "peg_tip_y": float(peg_tip_pos[1]),
        "peg_tip_z": float(peg_tip_pos[2]),
        "hole_center_x": float(hole_center_pos[0]),
        "hole_center_y": float(hole_center_pos[1]),
        "hole_center_z": float(hole_center_pos[2]),
        "peg_to_hole_x": float(error_vec[0]),
        "peg_to_hole_y": float(error_vec[1]),
        "peg_to_hole_z": float(error_vec[2]),
        "peg_to_hole_dist": distance,
        "peg_to_hole_axial_error": axial,
        "peg_to_hole_lateral_error": float(np.linalg.norm(lateral_vec)),
        "force_norm": force_norm,
        "recorded_force_norm": recorded_force_norm,
        "ctrl_clipped": ctrl_clipped,
    }
    for index in range(7):
        row[f"replay_target_{index}"] = float(replay_target[index])
        row[f"applied_ctrl_{index}"] = float(applied_ctrl[index])
        row[f"actual_qpos_{index}"] = float(qpos[index])
        row[f"qcmd_{index}"] = float(applied_ctrl[index])
        row[f"qpos_{index}"] = float(qpos[index])
    return row


def _clip_ctrl(
    target: np.ndarray,
    model,
    actuator_ids: np.ndarray,
) -> tuple[np.ndarray, bool]:
    clipped = np.asarray(target, dtype=np.float64).copy()
    was_clipped = False
    ctrl_limited = getattr(model, "actuator_ctrllimited", None)
    ctrl_ranges = np.asarray(model.actuator_ctrlrange[actuator_ids], dtype=np.float64)
    for index, actuator_id in enumerate(actuator_ids):
        limited = bool(ctrl_limited[actuator_id]) if ctrl_limited is not None else True
        if not limited:
            continue
        low, high = ctrl_ranges[index]
        if not np.isfinite(low) or not np.isfinite(high) or low >= high:
            continue
        before = clipped[index]
        clipped[index] = np.clip(before, low, high)
        was_clipped = was_clipped or bool(clipped[index] != before)
    return clipped, was_clipped


def _velocity_limited_ctrl(
    previous_ctrl: np.ndarray,
    target: np.ndarray,
    dt_hold: float,
    max_joint_velocity: Optional[float],
) -> np.ndarray:
    if max_joint_velocity is None:
        return np.asarray(target, dtype=np.float64).copy()
    max_delta = float(max_joint_velocity) * float(dt_hold)
    delta = np.clip(target - previous_ctrl, -max_delta, max_delta)
    return previous_ctrl + delta


def _load_episode_inputs(
    path: Path,
    command_field: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool, str]:
    with h5py.File(path, "r") as handle:
        if "observations/joint_pos" not in handle:
            raise KeyError("missing observations/joint_pos")
        joint_pos = np.asarray(handle["observations/joint_pos"], dtype=np.float64)
        if joint_pos.ndim != 2 or joint_pos.shape[1] != 7:
            raise ValueError(f"observations/joint_pos must have shape [T, 7], got {joint_pos.shape}")
        if not np.isfinite(joint_pos).all():
            print(f"warning: {path} contains non-finite joint_pos", file=sys.stderr)
        if command_field not in handle:
            raise KeyError(f"missing command field: {command_field}")
        replay_targets = np.asarray(handle[command_field], dtype=np.float64)
        if replay_targets.ndim != 2 or replay_targets.shape[1] != 7:
            raise ValueError(
                f"{command_field} must have shape [T, 7], got {replay_targets.shape}"
            )
        if len(replay_targets) != len(joint_pos):
            raise ValueError(
                f"{command_field} length {len(replay_targets)} does not match "
                f"observations/joint_pos length {len(joint_pos)}"
            )
        if not np.isfinite(replay_targets).all():
            print(f"warning: {path} contains non-finite {command_field}", file=sys.stderr)
        state_times, state_time_key, timestamps_are_real = _state_times(handle, len(joint_pos))
        recorded_force_norm, force_alignment = _recorded_force_norm_for_targets(
            handle, state_times, timestamps_are_real
        )
    return joint_pos, replay_targets, state_times, recorded_force_norm, timestamps_are_real, (
        f"{state_time_key};{force_alignment}"
    )


def _replay_episode(
    args: argparse.Namespace,
    episode_path: Path,
    episode_index: int,
    mujoco,
    model,
    joint_qposadr: np.ndarray,
    joint_dofadr: np.ndarray,
    actuator_ids: np.ndarray,
    site_ids: np.ndarray,
    sensor_slices: tuple[Optional[slice], Optional[slice]],
    camera_ids: np.ndarray,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    summary = _empty_summary(episode_path, episode_index, args.command_field)
    summary["playback_max_joint_velocity"] = (
        float(args.playback_max_joint_velocity)
        if args.playback_max_joint_velocity is not None
        else ""
    )
    frame_rows: list[dict[str, Any]] = []
    writers: dict[str, Any] = {}
    renderer = None
    try:
        joint_pos, replay_targets, state_times, recorded_force_norm, timestamps_are_real, alignment_info = (
            _load_episode_inputs(episode_path, args.command_field)
        )
        data = mujoco.MjData(model)
        data.qpos[joint_qposadr] = joint_pos[0]
        data.qvel[joint_dofadr] = 0.0
        initial_ctrl, _ = _clip_ctrl(replay_targets[0], model, actuator_ids)
        data.ctrl[actuator_ids] = initial_ctrl
        mujoco.mj_forward(model, data)
        previous_ctrl = initial_ctrl.copy()

        if args.render_videos:
            if np.any(camera_ids < 0):
                print("warning: missing replay camera; skipping videos for this episode", file=sys.stderr)
            else:
                renderer = mujoco.Renderer(model, height=480, width=640)
                writers = _init_video_writers(args.output_dir, episode_path.parent.name, args.video_fps)

        stop_reason = "completed"
        for target_index, replay_target in enumerate(replay_targets):
            dt_hold = _hold_dt(
                target_index,
                state_times,
                timestamps_are_real,
                args.use_recorded_state_timestamps,
                args.control_rate_hz,
            )
            limited_ctrl = _velocity_limited_ctrl(
                previous_ctrl=previous_ctrl,
                target=replay_target,
                dt_hold=dt_hold,
                max_joint_velocity=args.playback_max_joint_velocity,
            )
            applied_ctrl, ctrl_clipped = _clip_ctrl(limited_ctrl, model, actuator_ids)
            data.ctrl[actuator_ids] = applied_ctrl
            previous_ctrl = applied_ctrl.copy()
            n_steps = max(1, int(round(dt_hold / float(model.opt.timestep))))
            for _ in range(n_steps):
                mujoco.mj_step(model, data)

            qpos_actual = np.asarray(data.qpos[joint_qposadr], dtype=np.float64).copy()
            peg_tip = np.asarray(data.site_xpos[site_ids[0]], dtype=np.float64).copy()
            hole_center = np.asarray(data.site_xpos[site_ids[1]], dtype=np.float64).copy()
            force_norm = _read_mujoco_force_norm(data, sensor_slices[0], sensor_slices[1])
            row = _make_frame_row(
                episode_index=episode_index,
                episode_path=episode_path,
                command_field=args.command_field,
                target_index=target_index,
                sim_time=float(data.time),
                replay_target=replay_target,
                applied_ctrl=applied_ctrl,
                qpos=qpos_actual,
                peg_tip_pos=peg_tip,
                hole_center_pos=hole_center,
                hole_axis=args.hole_axis_world,
                force_norm=force_norm,
                recorded_force_norm=_nan_float(recorded_force_norm[target_index]),
                ctrl_clipped=ctrl_clipped,
            )
            row["alignment_info"] = alignment_info
            frame_rows.append(row)

            if args.render_videos and writers and target_index % args.video_every == 0:
                _write_video_frames(renderer, data, camera_ids, writers)

            if args.stop_on_force is not None and math.isfinite(force_norm) and force_norm > args.stop_on_force:
                stop_reason = "force_stop"
                break

        if episode_index == 0 and frame_rows:
            print(f"first_episode_path={episode_path}")
            print(
                "first_episode_initial_replay_peg_tip="
                f"{_vector_text(np.asarray([frame_rows[0]['peg_tip_x'], frame_rows[0]['peg_tip_y'], frame_rows[0]['peg_tip_z']]))}"
            )
            print(
                "first_episode_final_replay_peg_tip="
                f"{_vector_text(np.asarray([frame_rows[-1]['peg_tip_x'], frame_rows[-1]['peg_tip_y'], frame_rows[-1]['peg_tip_z']]))}"
            )
            print(
                "first_episode_initial_replay_hole_center="
                f"{_vector_text(np.asarray([frame_rows[0]['hole_center_x'], frame_rows[0]['hole_center_y'], frame_rows[0]['hole_center_z']]))}"
            )
            print(
                "first_episode_final_replay_hole_center="
                f"{_vector_text(np.asarray([frame_rows[-1]['hole_center_x'], frame_rows[-1]['hole_center_y'], frame_rows[-1]['hole_center_z']]))}"
            )
        summary = _episode_metrics(summary, frame_rows, stop_reason)
    except Exception as error:
        summary["ok"] = False
        summary["failure"] = str(error)
        summary["stop_reason"] = "failed"
        print(f"error: failed episode {episode_path}: {error}", file=sys.stderr)
    finally:
        if writers:
            _close_video_writers(writers)
        if renderer is not None:
            renderer.close()
    return summary, frame_rows


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


def _dataset_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ok_rows = [row for row in rows if row.get("ok", False)]
    summary: dict[str, Any] = {
        "episode_count": len(rows),
        "ok_episode_count": len(ok_rows),
        "failed_episode_count": len(rows) - len(ok_rows),
        "final_distance": _stats(_finite_values(rows, "final_peg_to_hole_dist")),
        "min_distance": _stats(_finite_values(rows, "min_peg_to_hole_dist")),
        "max_force_norm": _stats(_finite_values(rows, "max_force_norm")),
        "mean_qpos_tracking_error_norm": _stats(
            _finite_values(rows, "mean_qpos_tracking_error_norm")
        ),
        "max_qpos_tracking_error_norm": _stats(_finite_values(rows, "max_qpos_tracking_error_norm")),
        "mean_applied_ctrl_to_actual_error_norm": _stats(
            _finite_values(rows, "mean_applied_ctrl_to_actual_error_norm")
        ),
        "max_applied_ctrl_to_actual_error_norm": _stats(
            _finite_values(rows, "max_applied_ctrl_to_actual_error_norm")
        ),
        "mean_target_to_applied_ctrl_error_norm": _stats(
            _finite_values(rows, "mean_target_to_applied_ctrl_error_norm")
        ),
        "max_target_to_applied_ctrl_error_norm": _stats(
            _finite_values(rows, "max_target_to_applied_ctrl_error_norm")
        ),
    }
    for threshold in THRESHOLDS:
        label = _threshold_label(threshold)
        reached = [bool(row.get(f"reached_dist_lt_{label}", False)) for row in ok_rows]
        summary[f"fraction_reaching_lt_{label}"] = (
            float(sum(reached) / len(reached)) if reached else float("nan")
        )
    summary["failures"] = [
        {"episode_path": row["episode_path"], "failure": row.get("failure", "")}
        for row in rows
        if not row.get("ok", False)
    ]
    return summary


def _print_dataset_summary(summary: dict[str, Any]) -> None:
    print("\nDynamic Replay Dataset Summary")
    print("------------------------------")
    print(f"episode_count={summary['episode_count']}")
    print(f"ok_episode_count={summary['ok_episode_count']}")
    print(f"failed_episode_count={summary['failed_episode_count']}")
    for key in (
        "final_distance",
        "min_distance",
        "max_force_norm",
        "mean_qpos_tracking_error_norm",
        "max_qpos_tracking_error_norm",
        "mean_applied_ctrl_to_actual_error_norm",
        "max_applied_ctrl_to_actual_error_norm",
        "mean_target_to_applied_ctrl_error_norm",
        "max_target_to_applied_ctrl_error_norm",
    ):
        stats = summary[key]
        print(
            f"{key}: min={stats['min']:.9g} mean={stats['mean']:.9g} "
            f"median={stats['median']:.9g} max={stats['max']:.9g}"
        )
    for threshold in THRESHOLDS:
        label = _threshold_label(threshold)
        print(f"fraction_reaching_lt_{label}={summary[f'fraction_reaching_lt_{label}']:.9g}")


def audit(args: argparse.Namespace) -> int:
    mujoco = _load_mujoco()
    model = mujoco.MjModel.from_xml_path(str(args.model_xml))
    joint_ids = _resolve_ids(mujoco, model, mujoco.mjtObj.mjOBJ_JOINT, args.joint_names, "joints")
    actuator_ids = _resolve_ids(
        mujoco, model, mujoco.mjtObj.mjOBJ_ACTUATOR, args.actuator_names, "actuators"
    )
    site_ids = _resolve_optional_ids(mujoco, model, mujoco.mjtObj.mjOBJ_SITE, SITE_NAMES, "sites")
    if site_ids[0] < 0 or site_ids[1] < 0:
        raise ValueError("peg_tip_site and hole_center_site are required for dynamic replay")
    sensor_ids = _resolve_optional_ids(
        mujoco, model, mujoco.mjtObj.mjOBJ_SENSOR, SENSOR_NAMES, "sensors"
    )
    camera_ids = _resolve_optional_ids(
        mujoco, model, mujoco.mjtObj.mjOBJ_CAMERA, CAMERA_NAMES, "cameras"
    )
    joint_qposadr = np.asarray(model.jnt_qposadr[joint_ids], dtype=np.int64)
    joint_dofadr = np.asarray(model.jnt_dofadr[joint_ids], dtype=np.int64)
    sensor_slices = (
        _sensor_slice(model, int(sensor_ids[0]), 3, SENSOR_NAMES[0]) if sensor_ids[0] >= 0 else None,
        _sensor_slice(model, int(sensor_ids[1]), 3, SENSOR_NAMES[1]) if sensor_ids[1] >= 0 else None,
    )
    _print_model_probe(mujoco, model, joint_ids, actuator_ids, site_ids, sensor_ids)
    print(f"command_field={args.command_field}")
    print(f"actuator_names={list(args.actuator_names)}")
    print(f"playback_max_joint_velocity={args.playback_max_joint_velocity}")

    episodes = _discover_episodes(args.data_dir, args.episode_list, args.episode_path)
    if args.max_episodes is not None:
        episodes = episodes[: args.max_episodes]
    if not episodes:
        print("error: no episodes found", file=sys.stderr)
        return 2

    summary_rows: list[dict[str, Any]] = []
    frame_rows: list[dict[str, Any]] = []
    for episode_index, episode_path in enumerate(episodes):
        if episode_index % 10 == 0:
            print(f"replaying_episode={episode_index + 1}/{len(episodes)} path={episode_path}")
        summary, frames = _replay_episode(
            args=args,
            episode_path=episode_path,
            episode_index=episode_index,
            mujoco=mujoco,
            model=model,
            joint_qposadr=joint_qposadr,
            joint_dofadr=joint_dofadr,
            actuator_ids=actuator_ids,
            site_ids=site_ids,
            sensor_slices=sensor_slices,
            camera_ids=camera_ids,
        )
        summary_rows.append(summary)
        if args.save_frame_csv:
            frame_rows.extend(frames)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "replay_episode_summary.csv"
    dataset_summary_path = args.output_dir / "replay_dataset_summary.json"
    _write_csv(summary_path, summary_rows)
    dataset_summary = _dataset_summary(summary_rows)
    _save_json(dataset_summary_path, dataset_summary)
    if args.save_frame_csv:
        _write_csv(args.output_dir / "replay_frame_metrics.csv", frame_rows)
    _print_dataset_summary(dataset_summary)
    print(f"saved_replay_episode_summary={summary_path}")
    print(f"saved_replay_dataset_summary={dataset_summary_path}")
    if args.save_frame_csv:
        print(f"saved_replay_frame_metrics={args.output_dir / 'replay_frame_metrics.csv'}")
    if args.render_videos:
        print(f"saved_videos_dir={args.output_dir / 'videos'}")
    return 0 if dataset_summary["failed_episode_count"] == 0 else 1


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dynamically replay HDF5 qpos or command-label trajectories as MuJoCo actuator commands.",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("mujoco_data/peg_hole_fixed_insertion"))
    parser.add_argument("--episode-list", type=Path, default=None)
    parser.add_argument("--episode-path", type=Path, default=None)
    parser.add_argument("--model-xml", type=Path, default=Path("../arm_teleop/model/pangu_all_right.xml"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--command-field", type=str, default="observations/joint_pos")
    parser.add_argument("--joint-names", nargs="+", default=JOINT_NAMES)
    parser.add_argument("--actuator-names", nargs="+", default=ACTUATOR_NAMES)
    parser.add_argument("--hole-axis-world", type=float, nargs=3, default=(0.0, -1.0, 0.0))
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--control-rate-hz", type=float, default=30.0)
    parser.add_argument("--playback-max-joint-velocity", type=float, default=None)
    parser.add_argument("--use-recorded-state-timestamps", action="store_true")
    parser.add_argument("--render-videos", action="store_true")
    parser.add_argument("--video-fps", type=int, default=30)
    parser.add_argument("--video-every", type=int, default=1)
    parser.add_argument("--save-frame-csv", action="store_true")
    parser.add_argument("--stop-on-force", type=float, default=None)
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
    if len(args.actuator_names) != 7:
        print("error: --actuator-names must provide exactly 7 names", file=sys.stderr)
        return 2
    if args.control_rate_hz <= 0 or not math.isfinite(args.control_rate_hz):
        print("error: --control-rate-hz must be finite and positive", file=sys.stderr)
        return 2
    if (
        args.playback_max_joint_velocity is not None
        and (args.playback_max_joint_velocity <= 0 or not math.isfinite(args.playback_max_joint_velocity))
    ):
        print("error: --playback-max-joint-velocity must be finite and positive", file=sys.stderr)
        return 2
    args.command_field = args.command_field.strip().strip("/")
    if not args.command_field:
        print("error: --command-field must be non-empty", file=sys.stderr)
        return 2
    if args.video_fps <= 0 or args.video_every <= 0:
        print("error: --video-fps and --video-every must be positive", file=sys.stderr)
        return 2
    if args.episode_path is not None and args.episode_list is not None:
        print("error: use only one of --episode-path or --episode-list", file=sys.stderr)
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
        print(f"error: dynamic replay audit failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
