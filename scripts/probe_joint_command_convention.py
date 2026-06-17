#!/usr/bin/env python3
"""Probe internal versus public joint-command conventions in arm_teleop MuJoCo."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np


JOINT_NAMES = tuple(f"joint_{index}" for index in range(1, 8))
ACTUATOR_NAMES = tuple(f"motor_joint_{index}" for index in range(1, 8))
PUBLIC_INITIAL = np.asarray([-0.046, -0.2, 0.0, 1.6, -1.32, 0.005, 0.005])
ARM_SIGN = np.asarray([-1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0])
INTERNAL_TARGET_DELTA = np.asarray([0.02, -0.01, 0.015, -0.02, 0.01, 0.0, 0.0])
CONCLUSION = (
    "ForceAwareACT rollout should command internal MuJoCo qpos targets directly. "
    "Do not pass policy-predicted internal qpos into set_arm_positions() unless "
    "converting back to public convention first."
)


def _load_mujoco():
    try:
        import mujoco
    except ImportError as error:
        raise RuntimeError(
            "the 'mujoco' Python package is required to probe joint command conventions"
        ) from error
    return mujoco


def _resolve_ids(mujoco, model, object_type, names: Sequence[str], kind: str) -> np.ndarray:
    ids = np.asarray(
        [mujoco.mj_name2id(model, object_type, name) for name in names],
        dtype=np.int64,
    )
    missing = [name for name, object_id in zip(names, ids) if object_id < 0]
    if missing:
        raise ValueError(f"missing required {kind}: {', '.join(missing)}")
    return ids


def _as_list(values: np.ndarray) -> list[float]:
    return [float(value) for value in np.asarray(values, dtype=np.float64).reshape(-1)]


def _reset_to_internal_initial(
    mujoco,
    model,
    data,
    joint_qposadr: np.ndarray,
    joint_dofadr: np.ndarray,
    actuator_ids: np.ndarray,
    internal_initial: np.ndarray,
) -> None:
    mujoco.mj_resetData(model, data)
    data.qpos[joint_qposadr] = internal_initial
    data.qvel[joint_dofadr] = 0.0
    data.ctrl[actuator_ids] = internal_initial
    mujoco.mj_forward(model, data)


def _run_path(
    mujoco,
    model,
    data,
    joint_qposadr: np.ndarray,
    joint_dofadr: np.ndarray,
    actuator_ids: np.ndarray,
    internal_initial: np.ndarray,
    command_target: np.ndarray,
    steps: int,
) -> np.ndarray:
    _reset_to_internal_initial(
        mujoco,
        model,
        data,
        joint_qposadr,
        joint_dofadr,
        actuator_ids,
        internal_initial,
    )
    data.ctrl[actuator_ids] = command_target
    for _ in range(steps):
        mujoco.mj_step(model, data)
    return np.asarray(data.qpos[joint_qposadr], dtype=np.float64).copy()


def probe_convention(model_xml: Path, steps: int) -> dict[str, Any]:
    mujoco = _load_mujoco()
    model = mujoco.MjModel.from_xml_path(str(model_xml))
    data = mujoco.MjData(model)

    joint_ids = _resolve_ids(
        mujoco,
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        JOINT_NAMES,
        "joints",
    )
    actuator_ids = _resolve_ids(
        mujoco,
        model,
        mujoco.mjtObj.mjOBJ_ACTUATOR,
        ACTUATOR_NAMES,
        "actuators",
    )
    joint_qposadr = np.asarray(model.jnt_qposadr[joint_ids], dtype=np.int64)
    joint_dofadr = np.asarray(model.jnt_dofadr[joint_ids], dtype=np.int64)

    internal_initial = PUBLIC_INITIAL * ARM_SIGN
    internal_target = internal_initial + INTERNAL_TARGET_DELTA
    wrong_internal_target = internal_target * ARM_SIGN

    correct_final = _run_path(
        mujoco,
        model,
        data,
        joint_qposadr,
        joint_dofadr,
        actuator_ids,
        internal_initial,
        internal_target,
        steps,
    )
    wrong_final = _run_path(
        mujoco,
        model,
        data,
        joint_qposadr,
        joint_dofadr,
        actuator_ids,
        internal_initial,
        wrong_internal_target,
        steps,
    )

    correct_error = correct_final - internal_target
    wrong_error = wrong_final - internal_target
    target_difference = wrong_internal_target - internal_target
    non_flipped_indices = np.asarray([1, 2, 4, 5, 6], dtype=np.int64)
    wrong_path_mainly_flips_joint_1_and_joint_4 = bool(
        np.allclose(target_difference[non_flipped_indices], 0.0, atol=1e-12)
        and abs(target_difference[0]) > 0.0
        and abs(target_difference[3]) > 0.0
    )

    return {
        "model_xml": str(model_xml),
        "steps": int(steps),
        "timestep": float(model.opt.timestep),
        "joint_names": list(JOINT_NAMES),
        "joint_ids": [int(value) for value in joint_ids],
        "joint_qposadr": [int(value) for value in joint_qposadr],
        "joint_dofadr": [int(value) for value in joint_dofadr],
        "actuator_names": list(ACTUATOR_NAMES),
        "actuator_ids": [int(value) for value in actuator_ids],
        "public_initial": _as_list(PUBLIC_INITIAL),
        "arm_sign": _as_list(ARM_SIGN),
        "internal_initial_expected": _as_list(internal_initial),
        "internal_target": _as_list(internal_target),
        "wrong_internal_target": _as_list(wrong_internal_target),
        "correct_path_final_qpos": _as_list(correct_final),
        "wrong_path_final_qpos": _as_list(wrong_final),
        "correct_path_error_norm": float(np.linalg.norm(correct_error)),
        "wrong_path_error_norm": float(np.linalg.norm(wrong_error)),
        "correct_path_per_joint_error": _as_list(correct_error),
        "wrong_path_per_joint_error": _as_list(wrong_error),
        "wrong_path_mainly_flips_joint_1_and_joint_4": (
            wrong_path_mainly_flips_joint_1_and_joint_4
        ),
        "conclusion": CONCLUSION,
    }


def _print_summary(summary: dict[str, Any]) -> None:
    for key in (
        "model_xml",
        "steps",
        "timestep",
        "joint_ids",
        "joint_qposadr",
        "joint_dofadr",
        "actuator_ids",
        "public_initial",
        "arm_sign",
        "internal_initial_expected",
        "internal_target",
        "wrong_internal_target",
        "correct_path_final_qpos",
        "wrong_path_final_qpos",
        "correct_path_error_norm",
        "wrong_path_error_norm",
        "correct_path_per_joint_error",
        "wrong_path_per_joint_error",
        "wrong_path_mainly_flips_joint_1_and_joint_4",
    ):
        print(f"{key}={json.dumps(summary[key])}")
    print(f"conclusion={summary['conclusion']}")


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe internal and public-style MuJoCo joint command conventions.",
    )
    parser.add_argument(
        "--model-xml",
        type=Path,
        default=Path("../arm_teleop/model/pangu_all_right.xml"),
    )
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--steps", type=int, default=1000)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    model_xml = args.model_xml.expanduser().resolve()
    if not model_xml.is_file():
        print(f"error: model XML does not exist: {model_xml}", file=sys.stderr)
        return 2
    if args.steps <= 0:
        print("error: --steps must be positive", file=sys.stderr)
        return 2

    try:
        summary = probe_convention(model_xml, args.steps)
        _print_summary(summary)
        if args.output_json is not None:
            output_json = args.output_json.expanduser()
            output_json.parent.mkdir(parents=True, exist_ok=True)
            output_json.write_text(
                json.dumps(summary, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            print(f"saved_json={output_json}")
        return 0
    except Exception as error:
        print(f"error: joint command convention probe failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
