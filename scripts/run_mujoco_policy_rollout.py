#!/usr/bin/env python3
"""Run a dry-run or guarded ForceAwareACT rollout in the arm_teleop MuJoCo model."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import deque
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as functional

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from force_aware_act.data import denormalize_tensor, normalize_tensor  # noqa: E402
from force_aware_act.models import ForceAwareACTPolicy  # noqa: E402


JOINT_NAMES = tuple(f"joint_{index}" for index in range(1, 8))
ACTUATOR_NAMES = tuple(f"motor_joint_{index}" for index in range(1, 8))
CAMERA_NAMES = ("ee_cam", "base_top_cam")
SENSOR_NAMES = ("peg_ft_force", "peg_ft_torque")
TASK_SITE_NAMES = ("peg_tip_site", "hole_center_site")
TASK_BODY_NAMES = ("peg_tool", "wall_task")
PUBLIC_INITIAL = np.asarray([-0.046, -0.2, 0.0, 1.6, -1.32, 0.005, 0.005])
ARM_SIGN = np.asarray([-1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0])
ACTION_CHUNK_DIAGNOSTIC_NAMES = (
    "action_chunk_delta_norm_0",
    "action_chunk_delta_norm_mid",
    "action_chunk_delta_norm_last",
    "action_chunk_path_length",
    "action_chunk_first_to_last_delta",
)


def _load_mujoco():
    try:
        import mujoco
    except ImportError as error:
        raise RuntimeError("the 'mujoco' Python package is required for rollout") from error
    return mujoco


def _load_stats(path: Path) -> Dict[str, torch.Tensor]:
    stats = torch.load(path, map_location="cpu")
    if not isinstance(stats, dict):
        raise ValueError("normalization stats file must contain a dict")
    for key in ("qpos_mean", "qpos_std", "action_mean", "action_std", "force_mean", "force_std"):
        if key not in stats or not torch.is_tensor(stats[key]):
            raise KeyError(f"normalization stats missing tensor: {key}")
    return stats


def _model_kwargs(checkpoint: dict, force_window_len: int, chunk_len: int) -> dict:
    model_config = dict(checkpoint.get("config", {}).get("model", {}))
    if not model_config:
        raise KeyError("checkpoint config is missing model settings")
    if "pretrained_vision" in model_config and "pretrained_resnet18" not in model_config:
        model_config["pretrained_resnet18"] = model_config.pop("pretrained_vision")
    model_config.setdefault("pretrained_resnet18", False)
    model_config.setdefault("dropout", 0.0)
    model_config.setdefault("max_force_window_len", max(force_window_len, 20))
    if int(model_config.get("chunk_len", chunk_len)) != chunk_len:
        raise ValueError(
            f"--chunk-len={chunk_len} does not match checkpoint chunk_len="
            f"{model_config.get('chunk_len')}"
        )
    return model_config


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


def _sensor_slice(model, sensor_id: int, expected_dim: int, name: str) -> slice:
    dimension = int(model.sensor_dim[sensor_id])
    if dimension != expected_dim:
        raise ValueError(f"sensor {name} has dimension {dimension}, expected {expected_dim}")
    start = int(model.sensor_adr[sensor_id])
    return slice(start, start + dimension)


def _read_wrench(data, force_slice: slice, torque_slice: slice) -> np.ndarray:
    return np.concatenate(
        [
            np.asarray(data.sensordata[force_slice], dtype=np.float64),
            np.asarray(data.sensordata[torque_slice], dtype=np.float64),
        ]
    )


def _render_images(
    renderer,
    data,
    camera_ids: np.ndarray,
    image_size: int,
) -> tuple[torch.Tensor, np.ndarray]:
    frames = []
    for camera_id in camera_ids:
        renderer.update_scene(data, camera=int(camera_id))
        rgb = np.asarray(renderer.render(), dtype=np.uint8)
        frames.append(rgb)
    images = np.stack(frames, axis=0).astype(np.float32) / 255.0
    tensor = torch.from_numpy(images).permute(0, 3, 1, 2)
    resized = functional.interpolate(
        tensor,
        size=(image_size, image_size),
        mode="bilinear",
        align_corners=False,
    )
    return resized, np.asarray(frames, dtype=np.uint8)


def _position_or_nan(positions: np.ndarray, object_id: int) -> np.ndarray:
    if object_id < 0:
        return np.full(3, np.nan, dtype=np.float64)
    return np.asarray(positions[object_id], dtype=np.float64).copy()


def _task_diagnostics(
    data,
    site_ids: np.ndarray,
    body_ids: np.ndarray,
    hole_axis_world: np.ndarray,
) -> dict[str, np.ndarray | float]:
    peg_tip = _position_or_nan(data.site_xpos, int(site_ids[0]))
    hole_center = _position_or_nan(data.site_xpos, int(site_ids[1]))
    peg_to_hole = hole_center - peg_tip
    if np.isfinite(peg_to_hole).all():
        peg_to_hole_dist = float(np.linalg.norm(peg_to_hole))
        axial_error = float(np.dot(peg_to_hole, hole_axis_world))
        lateral = peg_to_hole - axial_error * hole_axis_world
        lateral_error = float(np.linalg.norm(lateral))
    else:
        peg_to_hole_dist = float("nan")
        axial_error = float("nan")
        lateral = np.full(3, np.nan, dtype=np.float64)
        lateral_error = float("nan")
    return {
        "peg_tip": peg_tip,
        "hole_center": hole_center,
        "peg_to_hole": peg_to_hole,
        "peg_to_hole_dist": peg_to_hole_dist,
        "peg_to_hole_axial_error": axial_error,
        "peg_to_hole_lateral": lateral,
        "peg_to_hole_lateral_error": lateral_error,
        "peg_tool": _position_or_nan(data.xpos, int(body_ids[0])),
        "wall_task": _position_or_nan(data.xpos, int(body_ids[1])),
    }


def _save_snapshots(snapshot_dir: Path, step: int, frames: np.ndarray) -> None:
    from PIL import Image

    snapshot_dir.mkdir(parents=True, exist_ok=True)
    for camera_name, frame in zip(CAMERA_NAMES, frames):
        Image.fromarray(frame).save(
            snapshot_dir / f"step_{step:04d}_{camera_name}.png"
        )


def _open_video_writers(video_dir: Path, fps: int) -> dict[str, object]:
    try:
        import imageio.v2 as imageio
    except ImportError as error:
        raise RuntimeError(
            "video recording requires imageio and imageio-ffmpeg; install them with: "
            ".venv/bin/python -m pip install imageio imageio-ffmpeg"
        ) from error

    video_dir.mkdir(parents=True, exist_ok=True)
    writers = {}
    try:
        for camera_name in CAMERA_NAMES:
            writers[camera_name] = imageio.get_writer(
                video_dir / f"{camera_name}.mp4",
                fps=fps,
            )
    except Exception as error:
        for writer in writers.values():
            writer.close()
        raise RuntimeError(
            "could not initialize MP4 video writers; ensure imageio and imageio-ffmpeg "
            "are installed with: .venv/bin/python -m pip install imageio imageio-ffmpeg"
        ) from error
    return writers


def _append_video_frames(
    video_writers: dict[str, object],
    frames: np.ndarray,
    frame_counts: dict[str, int],
) -> None:
    try:
        for camera_name, frame in zip(CAMERA_NAMES, frames):
            video_writers[camera_name].append_data(frame)
            frame_counts[camera_name] += 1
    except Exception as error:
        raise RuntimeError(
            "could not write MP4 video frames; ensure imageio and imageio-ffmpeg "
            "are installed with: .venv/bin/python -m pip install imageio imageio-ffmpeg"
        ) from error


def _resample_force_window(
    force_history: deque[tuple[float, np.ndarray]],
    current_time: float,
    window_duration: float,
    window_len: int,
) -> np.ndarray:
    history_times = np.asarray([item[0] for item in force_history], dtype=np.float64)
    history_values = np.asarray([item[1] for item in force_history], dtype=np.float64)
    target_times = np.linspace(current_time - window_duration, current_time, window_len)
    return np.stack(
        [
            np.interp(
                target_times,
                history_times,
                history_values[:, component],
                left=history_values[0, component],
                right=history_values[-1, component],
            )
            for component in range(6)
        ],
        axis=1,
    )


def _run_mode(
    model: ForceAwareACTPolicy,
    images: torch.Tensor,
    qpos: torch.Tensor,
    force_window: torch.Tensor,
    mode: str,
) -> dict:
    with torch.no_grad():
        return model(
            images=images,
            qpos=qpos,
            force_window=force_window,
            action_chunk=None,
            future_force_chunk=None,
            is_training=False,
            contact_latent_mode=mode,
            deterministic_prior=True,
        )


def _denormalize_predictions(
    output: dict,
    stats: Dict[str, torch.Tensor],
) -> tuple[np.ndarray, np.ndarray]:
    action = denormalize_tensor(output["pred_action"], stats["action_mean"], stats["action_std"])
    force = denormalize_tensor(output["pred_force"], stats["force_mean"], stats["force_std"])
    return action.squeeze(0).cpu().numpy(), force.squeeze(0).cpu().numpy()


def _action_chunk_diagnostics(action_chunk: np.ndarray, qpos: np.ndarray) -> dict[str, float]:
    if action_chunk.ndim != 2 or action_chunk.shape[0] == 0:
        raise ValueError(f"predicted action chunk must be non-empty [K, action_dim], got {action_chunk.shape}")
    middle_index = action_chunk.shape[0] // 2
    path_length = (
        float(np.linalg.norm(np.diff(action_chunk, axis=0), axis=1).sum())
        if action_chunk.shape[0] > 1
        else 0.0
    )
    return {
        "action_chunk_delta_norm_0": float(np.linalg.norm(action_chunk[0] - qpos)),
        "action_chunk_delta_norm_mid": float(np.linalg.norm(action_chunk[middle_index] - qpos)),
        "action_chunk_delta_norm_last": float(np.linalg.norm(action_chunk[-1] - qpos)),
        "action_chunk_path_length": path_length,
        "action_chunk_first_to_last_delta": float(
            np.linalg.norm(action_chunk[-1] - action_chunk[0])
        ),
    }


def _selected_action_index(action_chunk_len: int, mode: str) -> int:
    if mode == "first":
        return 0
    if mode == "mid":
        return action_chunk_len // 2
    if mode == "last":
        return action_chunk_len - 1
    if mode == "temporal":
        return -1
    raise ValueError(f"unknown action selection mode: {mode}")


def _temporal_aggregate_action(
    predicted_chunks: deque[tuple[int, np.ndarray]],
    current_step: int,
    decay: float,
) -> tuple[np.ndarray, int, float]:
    valid_actions = []
    ages = []
    for prediction_step, action_chunk in predicted_chunks:
        age = current_step - prediction_step
        if 0 <= age < action_chunk.shape[0]:
            valid_actions.append(action_chunk[age])
            ages.append(age)
    if not valid_actions:
        raise ValueError(f"no valid temporally aligned actions at rollout step {current_step}")
    weights = np.exp(-decay * np.asarray(ages, dtype=np.float64))
    selected_action = np.average(np.asarray(valid_actions, dtype=np.float64), axis=0, weights=weights)
    mean_age = float(np.average(np.asarray(ages, dtype=np.float64), weights=weights))
    return selected_action, len(valid_actions), mean_age


def _axial_push_joint_bias(
    mujoco,
    model,
    data,
    peg_tip_site_id: int,
    joint_dofadr: np.ndarray,
    desired_dx_world: np.ndarray,
    damping: float = 1.0e-3,
) -> np.ndarray:
    jacp = np.zeros((3, model.nv), dtype=np.float64)
    jacr = np.zeros((3, model.nv), dtype=np.float64)
    mujoco.mj_jacSite(model, data, jacp, jacr, peg_tip_site_id)
    arm_jacp = jacp[:, joint_dofadr]
    regularized = arm_jacp @ arm_jacp.T + damping**2 * np.eye(3, dtype=np.float64)
    return arm_jacp.T @ np.linalg.solve(regularized, desired_dx_world)


def _fieldnames() -> list[str]:
    fields = ["step", "time", "mode", "dry_run", "action_select_mode", "selected_action_index"]
    fields.extend(f"qpos_{index}" for index in range(7))
    fields.extend(f"qvel_{index}" for index in range(7))
    fields.extend(f"ft_{index}" for index in range(6))
    fields.extend(f"qcmd_{index}" for index in range(7))
    fields.extend(
        [
            "peg_tip_x",
            "peg_tip_y",
            "peg_tip_z",
            "hole_center_x",
            "hole_center_y",
            "hole_center_z",
            "peg_to_hole_dx",
            "peg_to_hole_dy",
            "peg_to_hole_dz",
            "peg_to_hole_dist",
            "hole_axis_x",
            "hole_axis_y",
            "hole_axis_z",
            "peg_to_hole_axial_error",
            "peg_to_hole_lateral_error",
            "peg_to_hole_lateral_x",
            "peg_to_hole_lateral_y",
            "peg_to_hole_lateral_z",
            "peg_tool_x",
            "peg_tool_y",
            "peg_tool_z",
            "wall_task_x",
            "wall_task_y",
            "wall_task_z",
        ]
    )
    fields.extend(
        [
            "force_norm",
            *(f"pred_action0_{index}" for index in range(7)),
            *(f"raw_pred_action0_{index}" for index in range(7)),
            *(f"delta_clipped_action0_{index}" for index in range(7)),
            *(f"ema_action0_{index}" for index in range(7)),
            *(f"ctrl_clipped_action0_{index}" for index in range(7)),
            *(f"selected_raw_action_{index}" for index in range(7)),
            *(f"selected_delta_clipped_action_{index}" for index in range(7)),
            *(f"selected_ema_action_{index}" for index in range(7)),
            *(f"selected_ctrl_clipped_action_{index}" for index in range(7)),
            *(f"selected_raw_action_with_bias_{index}" for index in range(7)),
            "axial_push_enabled",
            "axial_push_active",
            "axial_push_speed",
            "axial_push_start_dist",
            "axial_push_stop_force",
            "axial_push_dx_x",
            "axial_push_dx_y",
            "axial_push_dx_z",
            *(f"axial_push_dq_{index}" for index in range(7)),
            "axial_push_dq_norm",
            "action_delta_norm_raw_to_current",
            "action_delta_norm_after_clip",
            "action_delta_norm_after_ema",
            "selected_action_delta_norm_raw_to_current",
            "selected_action_delta_norm_after_clip",
            "selected_action_delta_norm_after_ema",
            "temporal_num_predictions",
            "temporal_mean_age",
            *ACTION_CHUNK_DIAGNOSTIC_NAMES,
            "pred_action_min",
            "pred_action_max",
            "pred_force_norm_0",
            "pred_force_norm_mean",
            "pred_force_norm_max",
            "prior_vs_zero_action_mean_abs_diff",
            "prior_vs_zero_force_mean_abs_diff",
            "stop_reason",
        ]
    )
    return fields


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=_fieldnames())
        writer.writeheader()
        writer.writerows(rows)


def _finite_max(values: Sequence[float]) -> float:
    finite_values = np.asarray(values, dtype=np.float64)
    finite_values = finite_values[np.isfinite(finite_values)]
    return float(finite_values.max()) if len(finite_values) else float("nan")


def run_rollout(args: argparse.Namespace) -> int:
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    mujoco = _load_mujoco()
    stats = _load_stats(args.normalization_stats)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint must contain a dict")
    model = ForceAwareACTPolicy(
        **_model_kwargs(checkpoint, args.force_window_len, args.chunk_len)
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    mj_model = mujoco.MjModel.from_xml_path(str(args.model_xml))
    data = mujoco.MjData(mj_model)
    joint_ids = _resolve_ids(mujoco, mj_model, mujoco.mjtObj.mjOBJ_JOINT, JOINT_NAMES, "joints")
    actuator_ids = _resolve_ids(
        mujoco, mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, ACTUATOR_NAMES, "actuators"
    )
    camera_ids = _resolve_ids(
        mujoco, mj_model, mujoco.mjtObj.mjOBJ_CAMERA, CAMERA_NAMES, "cameras"
    )
    sensor_ids = _resolve_ids(
        mujoco, mj_model, mujoco.mjtObj.mjOBJ_SENSOR, SENSOR_NAMES, "sensors"
    )
    site_ids = _resolve_optional_ids(
        mujoco, mj_model, mujoco.mjtObj.mjOBJ_SITE, TASK_SITE_NAMES, "sites"
    )
    body_ids = _resolve_optional_ids(
        mujoco, mj_model, mujoco.mjtObj.mjOBJ_BODY, TASK_BODY_NAMES, "bodies"
    )
    joint_qposadr = np.asarray(mj_model.jnt_qposadr[joint_ids], dtype=np.int64)
    joint_dofadr = np.asarray(mj_model.jnt_dofadr[joint_ids], dtype=np.int64)
    force_slice = _sensor_slice(mj_model, int(sensor_ids[0]), 3, SENSOR_NAMES[0])
    torque_slice = _sensor_slice(mj_model, int(sensor_ids[1]), 3, SENSOR_NAMES[1])

    internal_initial = PUBLIC_INITIAL * ARM_SIGN
    mujoco.mj_resetData(mj_model, data)
    data.qpos[joint_qposadr] = internal_initial
    data.qvel[joint_dofadr] = 0.0
    data.ctrl[actuator_ids] = internal_initial
    mujoco.mj_forward(mj_model, data)

    control_ranges = np.asarray(mj_model.actuator_ctrlrange[actuator_ids], dtype=np.float64)
    physics_steps_per_policy = max(
        1, int(round(1.0 / (args.policy_rate_hz * float(mj_model.opt.timestep))))
    )
    current_wrench = _read_wrench(data, force_slice, torque_slice)
    force_history: deque[tuple[float, np.ndarray]] = deque(
        [(float(data.time), current_wrench.copy())],
    )
    predicted_action_chunks: deque[tuple[int, np.ndarray]] = deque()
    previous_command = internal_initial.copy()
    renderer = mujoco.Renderer(
        mj_model,
        height=args.image_height,
        width=args.image_width,
    )

    rows: list[dict[str, object]] = []
    force_norm_history: list[float] = []
    lateral_error_history: list[tuple[int, float]] = []
    first_shapes: Optional[dict[str, tuple[int, ...]]] = None
    first_action: Optional[np.ndarray] = None
    first_force_norms: Optional[np.ndarray] = None
    first_qcmd: Optional[np.ndarray] = None
    final_qcmd: Optional[np.ndarray] = None
    first_selected_raw_action: Optional[np.ndarray] = None
    final_selected_raw_action: Optional[np.ndarray] = None
    first_action_chunk_diagnostics: Optional[dict[str, float]] = None
    final_action_chunk_diagnostics: Optional[dict[str, float]] = None
    first_temporal_num_predictions: Optional[int] = None
    final_temporal_num_predictions: Optional[int] = None
    first_temporal_mean_age: Optional[float] = None
    final_temporal_mean_age: Optional[float] = None
    axial_push_active_steps = 0
    axial_push_dq_norms: list[float] = []
    raw_delta_norms: list[float] = []
    clipped_delta_norms: list[float] = []
    ema_delta_norms: list[float] = []
    stop_reason = "max_rollout_steps"
    initial_task: Optional[dict[str, np.ndarray | float]] = None
    final_task: Optional[dict[str, np.ndarray | float]] = None
    snapshots_saved = False
    snapshot_dir = args.output_dir / "snapshots"
    video_dir = args.output_dir / "videos"
    video_writers: dict[str, object] = {}
    video_frame_counts = {camera_name: 0 for camera_name in CAMERA_NAMES}

    try:
        if args.save_videos:
            video_writers = _open_video_writers(video_dir, args.video_fps)
        for step in range(args.max_rollout_steps):
            qpos = np.asarray(data.qpos[joint_qposadr], dtype=np.float32).copy()
            qvel = np.asarray(data.qvel[joint_dofadr], dtype=np.float32).copy()
            wrench = _read_wrench(data, force_slice, torque_slice).astype(np.float32)
            force_norm = float(np.linalg.norm(wrench[:3]))
            force_norm_history.append(force_norm)
            force_window_np = _resample_force_window(
                force_history,
                float(data.time),
                args.force_window_duration,
                args.force_window_len,
            ).astype(np.float32)
            images_chw, raw_frames = _render_images(renderer, data, camera_ids, args.image_size)
            images = images_chw.unsqueeze(0)
            if args.save_camera_snapshots and step % args.snapshot_every == 0:
                _save_snapshots(snapshot_dir, step, raw_frames)
                snapshots_saved = True
            if args.save_videos and step % args.video_every == 0:
                _append_video_frames(video_writers, raw_frames, video_frame_counts)
            task = _task_diagnostics(data, site_ids, body_ids, args.hole_axis_world)
            lateral_error_history.append((step, float(task["peg_to_hole_lateral_error"])))
            if initial_task is None:
                initial_task = task
            final_task = task
            qpos_tensor = normalize_tensor(
                torch.from_numpy(qpos).unsqueeze(0), stats["qpos_mean"], stats["qpos_std"]
            )
            force_window_tensor = normalize_tensor(
                torch.from_numpy(force_window_np).unsqueeze(0),
                stats["force_mean"],
                stats["force_std"],
            )

            selected_output = _run_mode(
                model, images, qpos_tensor, force_window_tensor, args.contact_latent_mode
            )
            selected_action, selected_force = _denormalize_predictions(selected_output, stats)
            zero_action = zero_force = None
            if args.contact_latent_mode == "prior":
                zero_output = _run_mode(model, images, qpos_tensor, force_window_tensor, "zero")
                zero_action, zero_force = _denormalize_predictions(zero_output, stats)

            predicted_force_norms = np.linalg.norm(selected_force[:, :3], axis=1)
            finite = bool(
                np.isfinite(selected_action).all()
                and np.isfinite(selected_force).all()
                and np.isfinite(qpos).all()
                and np.isfinite(wrench).all()
            )
            row_stop_reason = ""
            if not finite:
                row_stop_reason = "nonfinite_value"
            elif force_norm > args.force_stop_threshold:
                row_stop_reason = "force_stop_threshold"

            if first_shapes is None:
                first_shapes = {
                    "images": tuple(images.shape),
                    "qpos": tuple(qpos_tensor.shape),
                    "force_window": tuple(force_window_tensor.shape),
                }
                first_action = selected_action[0].copy()
                first_force_norms = predicted_force_norms.copy()

            action0 = selected_action[0].astype(np.float64, copy=True)
            selected_action_index = _selected_action_index(
                selected_action.shape[0], args.action_select_mode
            )
            temporal_num_predictions: int | str = ""
            temporal_mean_age: float | str = ""
            if args.action_select_mode == "temporal":
                predicted_action_chunks.append(
                    (step, selected_action.astype(np.float64, copy=True))
                )
                while (
                    predicted_action_chunks
                    and step - predicted_action_chunks[0][0] >= selected_action.shape[0]
                ):
                    predicted_action_chunks.popleft()
                (
                    selected_raw_action,
                    temporal_num_predictions,
                    temporal_mean_age,
                ) = _temporal_aggregate_action(
                    predicted_action_chunks,
                    step,
                    args.temporal_agg_decay,
                )
                if first_temporal_num_predictions is None:
                    first_temporal_num_predictions = temporal_num_predictions
                    first_temporal_mean_age = temporal_mean_age
                final_temporal_num_predictions = temporal_num_predictions
                final_temporal_mean_age = temporal_mean_age
            else:
                selected_raw_action = selected_action[selected_action_index].astype(
                    np.float64, copy=True
                )
            axial_push_active = bool(
                args.enable_axial_push
                and int(site_ids[0]) >= 0
                and int(site_ids[1]) >= 0
                and np.isfinite(float(task["peg_to_hole_dist"]))
                and float(task["peg_to_hole_dist"]) <= args.axial_push_start_dist
                and force_norm < args.axial_push_stop_force
            )
            axial_push_dx = np.zeros(3, dtype=np.float64)
            axial_push_dq = np.zeros(7, dtype=np.float64)
            if axial_push_active:
                axial_push_dx = (
                    args.axial_push_speed / args.policy_rate_hz * args.hole_axis_world
                )
                axial_push_dq = _axial_push_joint_bias(
                    mujoco,
                    mj_model,
                    data,
                    int(site_ids[0]),
                    joint_dofadr,
                    axial_push_dx,
                )
                axial_push_active_steps += 1
            axial_push_dq_norm = float(np.linalg.norm(axial_push_dq))
            axial_push_dq_norms.append(axial_push_dq_norm)
            selected_raw_action_with_bias = selected_raw_action + axial_push_dq
            if not np.isfinite(selected_raw_action_with_bias).all():
                row_stop_reason = "nonfinite_value"
            if first_selected_raw_action is None:
                first_selected_raw_action = selected_raw_action.copy()
            final_selected_raw_action = selected_raw_action.copy()
            action_chunk_diagnostics = _action_chunk_diagnostics(selected_action, qpos)
            if first_action_chunk_diagnostics is None:
                first_action_chunk_diagnostics = action_chunk_diagnostics.copy()
            final_action_chunk_diagnostics = action_chunk_diagnostics.copy()
            nan_action = np.full(7, np.nan, dtype=np.float64)
            delta_clipped_action = nan_action.copy()
            ema_action = nan_action.copy()
            ctrl_clipped_action = nan_action.copy()
            raw_delta_norm = float(np.linalg.norm(selected_raw_action_with_bias - qpos))
            clipped_delta_norm = float("nan")
            ema_delta_norm = float("nan")
            if args.execute_actions:
                delta_clipped_action = qpos + np.clip(
                    selected_raw_action_with_bias - qpos,
                    -args.max_delta_q,
                    args.max_delta_q,
                )
                ema_action = (
                    args.ema_alpha * delta_clipped_action
                    + (1.0 - args.ema_alpha) * previous_command
                )
                ctrl_clipped_action = np.clip(
                    ema_action,
                    control_ranges[:, 0],
                    control_ranges[:, 1],
                )
                clipped_delta_norm = float(np.linalg.norm(delta_clipped_action - qpos))
                ema_delta_norm = float(np.linalg.norm(ema_action - qpos))

            if args.execute_actions and not row_stop_reason:
                data.ctrl[actuator_ids] = ctrl_clipped_action
                previous_command = ctrl_clipped_action.copy()
            qcmd = np.asarray(data.ctrl[actuator_ids], dtype=np.float64).copy()
            raw_delta_norms.append(raw_delta_norm)
            clipped_delta_norms.append(clipped_delta_norm)
            ema_delta_norms.append(ema_delta_norm)
            if first_qcmd is None:
                first_qcmd = qcmd.copy()
            final_qcmd = qcmd.copy()

            row: dict[str, object] = {
                "step": step,
                "time": float(data.time),
                "mode": args.contact_latent_mode,
                "dry_run": not args.execute_actions,
                "action_select_mode": args.action_select_mode,
                "selected_action_index": selected_action_index,
                "force_norm": force_norm,
                "action_delta_norm_raw_to_current": raw_delta_norm,
                "action_delta_norm_after_clip": clipped_delta_norm,
                "action_delta_norm_after_ema": ema_delta_norm,
                "selected_action_delta_norm_raw_to_current": raw_delta_norm,
                "selected_action_delta_norm_after_clip": clipped_delta_norm,
                "selected_action_delta_norm_after_ema": ema_delta_norm,
                "temporal_num_predictions": temporal_num_predictions,
                "temporal_mean_age": temporal_mean_age,
                "axial_push_enabled": args.enable_axial_push,
                "axial_push_active": axial_push_active,
                "axial_push_speed": args.axial_push_speed,
                "axial_push_start_dist": args.axial_push_start_dist,
                "axial_push_stop_force": args.axial_push_stop_force,
                "axial_push_dx_x": float(axial_push_dx[0]),
                "axial_push_dx_y": float(axial_push_dx[1]),
                "axial_push_dx_z": float(axial_push_dx[2]),
                "axial_push_dq_norm": axial_push_dq_norm,
                **action_chunk_diagnostics,
                "pred_action_min": float(selected_action.min()),
                "pred_action_max": float(selected_action.max()),
                "pred_force_norm_0": float(predicted_force_norms[0]),
                "pred_force_norm_mean": float(predicted_force_norms.mean()),
                "pred_force_norm_max": float(predicted_force_norms.max()),
                "prior_vs_zero_action_mean_abs_diff": (
                    float(np.abs(selected_action - zero_action).mean())
                    if zero_action is not None
                    else ""
                ),
                "prior_vs_zero_force_mean_abs_diff": (
                    float(np.abs(selected_force - zero_force).mean())
                    if zero_force is not None
                    else ""
                ),
                "stop_reason": row_stop_reason,
            }
            row.update({f"qpos_{index}": float(value) for index, value in enumerate(qpos)})
            row.update({f"qvel_{index}": float(value) for index, value in enumerate(qvel)})
            row.update({f"ft_{index}": float(value) for index, value in enumerate(wrench)})
            row.update({f"qcmd_{index}": float(value) for index, value in enumerate(qcmd)})
            row.update(
                {
                    **{
                        f"peg_tip_{axis}": float(task["peg_tip"][index])
                        for index, axis in enumerate(("x", "y", "z"))
                    },
                    **{
                        f"hole_center_{axis}": float(task["hole_center"][index])
                        for index, axis in enumerate(("x", "y", "z"))
                    },
                    **{
                        f"peg_to_hole_d{axis}": float(task["peg_to_hole"][index])
                        for index, axis in enumerate(("x", "y", "z"))
                    },
                    "peg_to_hole_dist": float(task["peg_to_hole_dist"]),
                    **{
                        f"hole_axis_{axis}": float(args.hole_axis_world[index])
                        for index, axis in enumerate(("x", "y", "z"))
                    },
                    "peg_to_hole_axial_error": float(task["peg_to_hole_axial_error"]),
                    "peg_to_hole_lateral_error": float(task["peg_to_hole_lateral_error"]),
                    **{
                        f"peg_to_hole_lateral_{axis}": float(
                            task["peg_to_hole_lateral"][index]
                        )
                        for index, axis in enumerate(("x", "y", "z"))
                    },
                    **{
                        f"peg_tool_{axis}": float(task["peg_tool"][index])
                        for index, axis in enumerate(("x", "y", "z"))
                    },
                    **{
                        f"wall_task_{axis}": float(task["wall_task"][index])
                        for index, axis in enumerate(("x", "y", "z"))
                    },
                }
            )
            row.update(
                {f"pred_action0_{index}": float(value) for index, value in enumerate(selected_action[0])}
            )
            row.update(
                {f"raw_pred_action0_{index}": float(value) for index, value in enumerate(action0)}
            )
            row.update(
                {
                    f"delta_clipped_action0_{index}": float(value)
                    for index, value in enumerate(delta_clipped_action)
                }
            )
            row.update(
                {f"ema_action0_{index}": float(value) for index, value in enumerate(ema_action)}
            )
            row.update(
                {
                    f"ctrl_clipped_action0_{index}": float(value)
                    for index, value in enumerate(ctrl_clipped_action)
                }
            )
            row.update(
                {
                    f"selected_raw_action_{index}": float(value)
                    for index, value in enumerate(selected_raw_action)
                }
            )
            row.update(
                {
                    f"selected_delta_clipped_action_{index}": float(value)
                    for index, value in enumerate(delta_clipped_action)
                }
            )
            row.update(
                {
                    f"selected_ema_action_{index}": float(value)
                    for index, value in enumerate(ema_action)
                }
            )
            row.update(
                {
                    f"selected_ctrl_clipped_action_{index}": float(value)
                    for index, value in enumerate(ctrl_clipped_action)
                }
            )
            row.update(
                {
                    f"selected_raw_action_with_bias_{index}": float(value)
                    for index, value in enumerate(selected_raw_action_with_bias)
                }
            )
            row.update(
                {
                    f"axial_push_dq_{index}": float(value)
                    for index, value in enumerate(axial_push_dq)
                }
            )
            rows.append(row)

            if row_stop_reason:
                stop_reason = row_stop_reason
                break

            for _ in range(physics_steps_per_policy):
                mujoco.mj_step(mj_model, data)
                sampled_wrench = _read_wrench(data, force_slice, torque_slice)
                force_history.append((float(data.time), sampled_wrench.copy()))
            oldest_needed = float(data.time) - args.force_window_duration - float(mj_model.opt.timestep)
            while len(force_history) > 1 and force_history[1][0] < oldest_needed:
                force_history.popleft()
    finally:
        renderer.close()
        for video_writer in video_writers.values():
            video_writer.close()

    if rows and not rows[-1]["stop_reason"]:
        rows[-1]["stop_reason"] = stop_reason
    log_path = args.output_dir / "rollout_log.csv"
    _write_csv(log_path, rows)

    print(f"output_dir={args.output_dir}")
    print(f"rollout_mode={'execute' if args.execute_actions else 'dry_run'}")
    print(f"steps_executed={len(rows)}")
    print(f"final_time={float(data.time):.9g}")
    print(f"first_input_shapes={first_shapes}")
    print(f"first_predicted_action={np.array2string(first_action, precision=6, separator=',')}")
    print(
        "first_predicted_force_norm_trend="
        f"{np.array2string(first_force_norms, precision=6, separator=',')}"
    )
    print(f"first_qcmd={np.array2string(first_qcmd, precision=6, separator=',')}")
    print(f"final_qcmd={np.array2string(final_qcmd, precision=6, separator=',')}")
    print(f"action_select_mode={args.action_select_mode}")
    print(f"selected_action_index={_selected_action_index(args.chunk_len, args.action_select_mode)}")
    if args.action_select_mode == "temporal":
        print(f"temporal_agg_decay={args.temporal_agg_decay:.9g}")
        print(f"first_temporal_num_predictions={first_temporal_num_predictions}")
        print(f"final_temporal_num_predictions={final_temporal_num_predictions}")
        print(f"first_temporal_mean_age={first_temporal_mean_age:.9g}")
        print(f"final_temporal_mean_age={final_temporal_mean_age:.9g}")
    print(
        "first_selected_raw_action="
        f"{np.array2string(first_selected_raw_action, precision=6, separator=',')}"
    )
    print(
        "final_selected_raw_action="
        f"{np.array2string(final_selected_raw_action, precision=6, separator=',')}"
    )
    print(f"max_action_delta_norm_raw_to_current={_finite_max(raw_delta_norms):.9g}")
    print(f"max_action_delta_norm_after_clip={_finite_max(clipped_delta_norms):.9g}")
    print(f"max_action_delta_norm_after_ema={_finite_max(ema_delta_norms):.9g}")
    print(f"max_selected_action_delta_norm_raw_to_current={_finite_max(raw_delta_norms):.9g}")
    print(f"max_selected_action_delta_norm_after_clip={_finite_max(clipped_delta_norms):.9g}")
    print(f"max_selected_action_delta_norm_after_ema={_finite_max(ema_delta_norms):.9g}")
    print(f"axial_push_enabled={args.enable_axial_push}")
    print(f"axial_push_active_steps={axial_push_active_steps}")
    print(f"max_axial_push_dq_norm={_finite_max(axial_push_dq_norms):.9g}")
    for diagnostic_name in ACTION_CHUNK_DIAGNOSTIC_NAMES:
        print(
            f"first_{diagnostic_name}="
            f"{first_action_chunk_diagnostics[diagnostic_name]:.9g}"
        )
        print(
            f"final_{diagnostic_name}="
            f"{final_action_chunk_diagnostics[diagnostic_name]:.9g}"
        )
    print(f"max_force_norm={max(force_norm_history):.9g}")
    print(f"mean_force_norm={float(np.mean(force_norm_history)):.9g}")
    print(
        "initial_peg_tip_position="
        f"{np.array2string(initial_task['peg_tip'], precision=6, separator=',')}"
    )
    print(
        "final_peg_tip_position="
        f"{np.array2string(final_task['peg_tip'], precision=6, separator=',')}"
    )
    print(
        "initial_hole_center_position="
        f"{np.array2string(initial_task['hole_center'], precision=6, separator=',')}"
    )
    print(
        "final_hole_center_position="
        f"{np.array2string(final_task['hole_center'], precision=6, separator=',')}"
    )
    print(
        "initial_peg_to_hole="
        f"{np.array2string(initial_task['peg_to_hole'], precision=6, separator=',')} "
        f"distance={float(initial_task['peg_to_hole_dist']):.9g}"
    )
    print(
        "final_peg_to_hole="
        f"{np.array2string(final_task['peg_to_hole'], precision=6, separator=',')} "
        f"distance={float(final_task['peg_to_hole_dist']):.9g}"
    )
    finite_lateral_errors = [
        (step, error) for step, error in lateral_error_history if np.isfinite(error)
    ]
    min_lateral_step, min_lateral_error = (
        min(finite_lateral_errors, key=lambda item: item[1])
        if finite_lateral_errors
        else (-1, float("nan"))
    )
    print(f"initial_peg_to_hole_axial_error={float(initial_task['peg_to_hole_axial_error']):.9g}")
    print(f"final_peg_to_hole_axial_error={float(final_task['peg_to_hole_axial_error']):.9g}")
    print(
        "initial_peg_to_hole_lateral_error="
        f"{float(initial_task['peg_to_hole_lateral_error']):.9g}"
    )
    print(
        "final_peg_to_hole_lateral_error="
        f"{float(final_task['peg_to_hole_lateral_error']):.9g}"
    )
    print(f"min_peg_to_hole_lateral_error={min_lateral_error:.9g}")
    print(f"min_peg_to_hole_lateral_error_step={min_lateral_step}")
    print(f"final_peg_to_hole_distance={float(final_task['peg_to_hole_dist']):.9g}")
    print(f"snapshots_saved={snapshots_saved}")
    if snapshots_saved:
        print(f"snapshot_dir={snapshot_dir}")
    videos_saved = args.save_videos and any(video_frame_counts.values())
    print(f"videos_saved={videos_saved}")
    if args.save_videos:
        print(f"video_dir={video_dir}")
        for camera_name in CAMERA_NAMES:
            print(f"video_frames_{camera_name}={video_frame_counts[camera_name]}")
        print(f"video_fps={args.video_fps}")
    print(f"stop_reason={stop_reason}")
    print(f"rollout_log_csv={log_path}")
    return 0


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ForceAwareACT in a local MuJoCo environment.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--normalization-stats", type=Path, required=True)
    parser.add_argument(
        "--model-xml",
        type=Path,
        default=Path("../arm_teleop/model/pangu_all_right.xml"),
    )
    parser.add_argument("--contact-latent-mode", choices=("zero", "prior"), default="prior")
    parser.add_argument(
        "--action-select-mode",
        choices=("first", "mid", "last", "temporal"),
        default="first",
    )
    parser.add_argument("--temporal-agg-decay", type=float, default=0.3)
    parser.add_argument("--chunk-len", type=int, default=10)
    parser.add_argument("--force-window-len", type=int, default=20)
    parser.add_argument("--force-window-duration", type=float, default=0.25)
    parser.add_argument("--policy-rate-hz", type=float, default=30.0)
    parser.add_argument("--max-rollout-steps", type=int, default=100)
    parser.add_argument("--image-width", type=int, default=640)
    parser.add_argument("--image-height", type=int, default=480)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--execute-actions", action="store_true")
    parser.add_argument("--ema-alpha", type=float, default=0.3)
    parser.add_argument("--max-delta-q", type=float, default=0.05)
    parser.add_argument("--force-stop-threshold", type=float, default=300.0)
    parser.add_argument("--enable-axial-push", action="store_true")
    parser.add_argument("--axial-push-speed", type=float, default=0.0)
    parser.add_argument("--axial-push-start-dist", type=float, default=0.05)
    parser.add_argument("--axial-push-stop-force", type=float, default=5.0)
    parser.add_argument(
        "--hole-axis-world",
        type=float,
        nargs=3,
        default=(0.0, -1.0, 0.0),
    )
    parser.add_argument("--save-camera-snapshots", action="store_true")
    parser.add_argument("--snapshot-every", type=int, default=10)
    parser.add_argument("--save-videos", action="store_true")
    parser.add_argument("--video-fps", type=int, default=30)
    parser.add_argument("--video-every", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    for key in ("checkpoint", "normalization_stats", "model_xml"):
        path = getattr(args, key).expanduser().resolve()
        setattr(args, key, path)
        if not path.is_file():
            print(f"error: {key.replace('_', ' ')} does not exist: {path}", file=sys.stderr)
            return 2
    args.output_dir = args.output_dir.expanduser()
    if args.chunk_len <= 0 or args.force_window_len <= 0 or args.max_rollout_steps <= 0:
        print("error: chunk/window/rollout lengths must be positive", file=sys.stderr)
        return 2
    if args.force_window_duration < 0 or args.policy_rate_hz <= 0:
        print("error: force window duration must be non-negative and policy rate positive", file=sys.stderr)
        return 2
    if args.image_width <= 0 or args.image_height <= 0 or args.image_size <= 0:
        print("error: image dimensions must be positive", file=sys.stderr)
        return 2
    if not 0.0 <= args.ema_alpha <= 1.0:
        print("error: --ema-alpha must be in [0, 1]", file=sys.stderr)
        return 2
    if args.max_delta_q <= 0 or args.force_stop_threshold <= 0:
        print("error: --max-delta-q and --force-stop-threshold must be positive", file=sys.stderr)
        return 2
    if not np.isfinite(args.axial_push_speed):
        print("error: --axial-push-speed must be finite", file=sys.stderr)
        return 2
    if args.axial_push_start_dist < 0 or args.axial_push_stop_force < 0:
        print(
            "error: --axial-push-start-dist and --axial-push-stop-force must be non-negative",
            file=sys.stderr,
        )
        return 2
    args.hole_axis_world = np.asarray(args.hole_axis_world, dtype=np.float64)
    hole_axis_norm = float(np.linalg.norm(args.hole_axis_world))
    if not np.isfinite(args.hole_axis_world).all() or hole_axis_norm <= 0:
        print("error: --hole-axis-world must be a finite nonzero vector", file=sys.stderr)
        return 2
    args.hole_axis_world = args.hole_axis_world / hole_axis_norm
    if args.temporal_agg_decay < 0:
        print("error: --temporal-agg-decay must be non-negative", file=sys.stderr)
        return 2
    if args.snapshot_every <= 0:
        print("error: --snapshot-every must be positive", file=sys.stderr)
        return 2
    if args.video_fps <= 0 or args.video_every <= 0:
        print("error: --video-fps and --video-every must be positive", file=sys.stderr)
        return 2
    args.output_dir.mkdir(parents=True, exist_ok=True)
    try:
        return run_rollout(args)
    except Exception as error:
        print(f"error: MuJoCo policy rollout failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
