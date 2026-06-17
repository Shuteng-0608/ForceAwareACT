#!/usr/bin/env python3
"""Read-only probe for the arm_teleop MuJoCo deployment environment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np


EXPECTED_JOINTS = tuple(f"joint_{index}" for index in range(1, 8))
EXPECTED_ACTUATORS = tuple(f"motor_joint_{index}" for index in range(1, 8))
EXPECTED_CAMERAS = ("ee_cam", "base_top_cam")
EXPECTED_SENSORS = ("peg_ft_force", "peg_ft_torque")
CAMERA_RESOLUTION_ASSUMPTION = {
    "width": 640,
    "height": 480,
    "color_format": "raw RGB",
}


def _load_mujoco():
    try:
        import mujoco
    except ImportError as error:
        raise RuntimeError(
            "the 'mujoco' Python package is required to probe the MuJoCo environment"
        ) from error
    return mujoco


def _object_id(mujoco, model, object_type, name: str) -> int:
    return int(mujoco.mj_name2id(model, object_type, name))


def _require_ids(kind: str, names: Sequence[str], ids: dict[str, int], errors: list[str]) -> None:
    missing = [name for name in names if ids[name] < 0]
    if missing:
        errors.append(f"missing required {kind}: {', '.join(missing)}")


def _round_list(values: np.ndarray) -> list[float]:
    return [float(value) for value in np.asarray(values, dtype=np.float64).reshape(-1)]


def _int_list(values: np.ndarray) -> list[int]:
    return [int(value) for value in np.asarray(values).reshape(-1)]


def _site_summary(mujoco, model, data, name: str) -> dict[str, Any]:
    site_id = _object_id(mujoco, model, mujoco.mjtObj.mjOBJ_SITE, name)
    result: dict[str, Any] = {"id": site_id, "exists": site_id >= 0}
    if site_id >= 0:
        result.update(
            {
                "model_pos": _round_list(model.site_pos[site_id]),
                "model_quat": _round_list(model.site_quat[site_id]),
                "world_pos_after_forward": _round_list(data.site_xpos[site_id]),
                "world_xmat_after_forward": np.asarray(
                    data.site_xmat[site_id], dtype=np.float64
                ).reshape(3, 3).tolist(),
            }
        )
    return result


def _body_summary(mujoco, model, data, name: str) -> dict[str, Any]:
    body_id = _object_id(mujoco, model, mujoco.mjtObj.mjOBJ_BODY, name)
    result: dict[str, Any] = {"id": body_id, "exists": body_id >= 0}
    if body_id >= 0:
        result.update(
            {
                "model_pos": _round_list(model.body_pos[body_id]),
                "world_pos_after_forward": _round_list(data.xpos[body_id]),
            }
        )
    return result


def probe_environment(model_xml: Path) -> tuple[dict[str, Any], list[str]]:
    mujoco = _load_mujoco()
    model = mujoco.MjModel.from_xml_path(str(model_xml))
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    joint_ids = {
        name: _object_id(mujoco, model, mujoco.mjtObj.mjOBJ_JOINT, name)
        for name in EXPECTED_JOINTS
    }
    actuator_ids = {
        name: _object_id(mujoco, model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        for name in EXPECTED_ACTUATORS
    }
    camera_ids = {
        name: _object_id(mujoco, model, mujoco.mjtObj.mjOBJ_CAMERA, name)
        for name in EXPECTED_CAMERAS
    }
    sensor_ids = {
        name: _object_id(mujoco, model, mujoco.mjtObj.mjOBJ_SENSOR, name)
        for name in EXPECTED_SENSORS
    }

    joint_qposadr = {
        name: int(model.jnt_qposadr[joint_id]) if joint_id >= 0 else None
        for name, joint_id in joint_ids.items()
    }
    joint_dofadr = {
        name: int(model.jnt_dofadr[joint_id]) if joint_id >= 0 else None
        for name, joint_id in joint_ids.items()
    }
    initial_qpos = {
        name: float(data.qpos[address]) if address is not None else None
        for name, address in joint_qposadr.items()
    }
    initial_qvel = {
        name: float(data.qvel[address]) if address is not None else None
        for name, address in joint_dofadr.items()
    }

    actuator_trnid = {
        name: _int_list(model.actuator_trnid[actuator_id])
        if actuator_id >= 0
        else None
        for name, actuator_id in actuator_ids.items()
    }
    actuator_ctrlrange = {
        name: _round_list(model.actuator_ctrlrange[actuator_id])
        if actuator_id >= 0
        else None
        for name, actuator_id in actuator_ids.items()
    }

    sensor_dimensions = {
        name: int(model.sensor_dim[sensor_id]) if sensor_id >= 0 else None
        for name, sensor_id in sensor_ids.items()
    }
    sensor_address_ranges = {
        name: [
            int(model.sensor_adr[sensor_id]),
            int(model.sensor_adr[sensor_id] + model.sensor_dim[sensor_id]),
        ]
        if sensor_id >= 0
        else None
        for name, sensor_id in sensor_ids.items()
    }
    initial_sensordata: dict[str, Optional[list[float]]] = {}
    for name, address_range in sensor_address_ranges.items():
        if address_range is None:
            initial_sensordata[name] = None
        else:
            start, stop = address_range
            initial_sensordata[name] = _round_list(data.sensordata[start:stop])
    initial_force = initial_sensordata["peg_ft_force"]
    initial_force_norm = (
        float(np.linalg.norm(np.asarray(initial_force, dtype=np.float64)))
        if initial_force is not None
        else None
    )

    summary: dict[str, Any] = {
        "model_xml": str(model_xml),
        "timestep": float(model.opt.timestep),
        "nq": int(model.nq),
        "nv": int(model.nv),
        "nu": int(model.nu),
        "nsensor": int(model.nsensor),
        "ncam": int(model.ncam),
        "joint_names_found": joint_ids,
        "joint_qposadr": joint_qposadr,
        "joint_dofadr": joint_dofadr,
        "actuator_names_found": actuator_ids,
        "actuator_trnid": actuator_trnid,
        "actuator_ctrlrange": actuator_ctrlrange,
        "camera_names_found": camera_ids,
        "camera_resolution_assumption": CAMERA_RESOLUTION_ASSUMPTION,
        "sensor_names_found": sensor_ids,
        "sensor_dimensions": sensor_dimensions,
        "sensor_address_ranges": sensor_address_ranges,
        "ft_sensor_site": _site_summary(mujoco, model, data, "ft_sensor_site"),
        "peg_tool_body": _body_summary(mujoco, model, data, "peg_tool"),
        "wall_task_body": _body_summary(mujoco, model, data, "wall_task"),
        "peg_tip_site_exists": (
            _object_id(mujoco, model, mujoco.mjtObj.mjOBJ_SITE, "peg_tip_site") >= 0
        ),
        "hole_center_site_exists": (
            _object_id(mujoco, model, mujoco.mjtObj.mjOBJ_SITE, "hole_center_site") >= 0
        ),
        "initial_qpos": initial_qpos,
        "initial_qvel": initial_qvel,
        "initial_sensordata": initial_sensordata,
        "initial_force_norm": initial_force_norm,
    }

    errors: list[str] = []
    _require_ids("joints", EXPECTED_JOINTS, joint_ids, errors)
    _require_ids("actuators", EXPECTED_ACTUATORS, actuator_ids, errors)
    _require_ids("cameras", EXPECTED_CAMERAS, camera_ids, errors)
    _require_ids("sensors", EXPECTED_SENSORS, sensor_ids, errors)
    for sensor_name in EXPECTED_SENSORS:
        dimension = sensor_dimensions[sensor_name]
        if dimension is not None and dimension != 3:
            errors.append(
                f"sensor {sensor_name} has dimension {dimension}, expected 3"
            )
    if not np.isclose(float(model.opt.timestep), 0.001, rtol=0.0, atol=1e-9):
        errors.append(
            f"model timestep is {float(model.opt.timestep):.9g}, expected approximately 0.001"
        )

    summary["validation"] = {"passed": not errors, "errors": errors}
    return summary, errors


def _print_summary(summary: dict[str, Any]) -> None:
    print(f"model_xml={summary['model_xml']}")
    for key in ("timestep", "nq", "nv", "nu", "nsensor", "ncam"):
        print(f"{key}={summary[key]}")
    for key in (
        "joint_names_found",
        "joint_qposadr",
        "joint_dofadr",
        "actuator_names_found",
        "actuator_trnid",
        "actuator_ctrlrange",
        "camera_names_found",
        "camera_resolution_assumption",
        "sensor_names_found",
        "sensor_dimensions",
        "sensor_address_ranges",
        "ft_sensor_site",
        "peg_tool_body",
        "wall_task_body",
        "peg_tip_site_exists",
        "hole_center_site_exists",
        "initial_qpos",
        "initial_qvel",
        "initial_sensordata",
        "initial_force_norm",
    ):
        print(f"{key}={json.dumps(summary[key], sort_keys=True)}")
    print(f"validation_passed={summary['validation']['passed']}")


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe the arm_teleop MuJoCo XML without commanding the robot.",
    )
    parser.add_argument(
        "--model-xml",
        type=Path,
        default=Path("../arm_teleop/model/pangu_all_right.xml"),
    )
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    model_xml = args.model_xml.expanduser().resolve()
    if not model_xml.is_file():
        print(f"error: model XML does not exist: {model_xml}", file=sys.stderr)
        return 2

    try:
        summary, errors = probe_environment(model_xml)
        _print_summary(summary)
        if args.output_json is not None:
            output_json = args.output_json.expanduser()
            output_json.parent.mkdir(parents=True, exist_ok=True)
            output_json.write_text(
                json.dumps(summary, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            print(f"saved_json={output_json}")
        if errors:
            for error in errors:
                print(f"validation_error: {error}", file=sys.stderr)
            return 1
        return 0
    except Exception as error:
        print(f"error: MuJoCo environment probe failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
