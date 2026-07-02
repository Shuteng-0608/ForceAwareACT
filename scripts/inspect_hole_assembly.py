#!/usr/bin/env python3
"""Inspect and validate the MuJoCo hole assembly offset without loading a policy."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from run_mujoco_policy_rollout import (  # noqa: E402
    DEFAULT_HOLE_BODY_NAME,
    DEFAULT_HOLE_SITE_NAME,
    EXPECTED_HOLE_GEOM_NAMES,
    _body_name,
    _load_mujoco,
    apply_hole_body_offset,
    body_subtree_ids,
    resolve_hole_body,
    resolve_named_site,
    validate_hole_assembly_structure,
)


def _geom_name(model, geom_id: int) -> str:
    mujoco = _load_mujoco()
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(geom_id))
    return name or f"geom_{geom_id}"


def _geom_id(model, geom_name: str) -> int:
    mujoco = _load_mujoco()
    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
    if geom_id < 0:
        raise ValueError(f"missing geom: {geom_name}")
    return int(geom_id)


def inspect_hole_assembly(args: argparse.Namespace) -> dict:
    mujoco = _load_mujoco()
    model = mujoco.MjModel.from_xml_path(str(args.model_xml))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    site_id = resolve_named_site(model, args.hole_site_name)
    body_id = resolve_hole_body(model, site_id, args.hole_body_name)
    structure = validate_hole_assembly_structure(
        model,
        body_id,
        site_id,
        EXPECTED_HOLE_GEOM_NAMES,
    )
    subtree = body_subtree_ids(model, body_id)
    hole_geom_ids = [_geom_id(model, name) for name in structure["hole_geom_names"]]
    nominal_site = np.asarray(data.site_xpos[site_id], dtype=np.float64).copy()
    nominal_geoms = {
        _geom_name(model, geom_id): np.asarray(data.geom_xpos[geom_id], dtype=np.float64).copy()
        for geom_id in hole_geom_ids
    }
    reference_body_names = [
        _body_name(model, body_index)
        for body_index in range(model.nbody)
        if body_index not in subtree and _body_name(model, body_index) in {"peg_tool", "visual_room_background"}
    ]
    nominal_references = {
        name: np.asarray(data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)], dtype=np.float64).copy()
        for name in reference_body_names
    }

    requested = np.asarray(
        [args.test_offset_x, args.test_offset_y, args.test_offset_z],
        dtype=np.float64,
    )
    offset_metadata = apply_hole_body_offset(
        model,
        data,
        body_id,
        site_id,
        requested,
        args.offset_frame,
    )
    actual_site = np.asarray(data.site_xpos[site_id], dtype=np.float64).copy()
    actual_site_offset = actual_site - nominal_site
    if args.offset_frame == "world":
        expected_displacement = requested
    else:
        expected_displacement = actual_site_offset
    if not np.allclose(actual_site_offset, expected_displacement, atol=1.0e-7):
        raise ValueError(
            f"hole_goal_site displacement mismatch: requested={requested.tolist()} "
            f"actual={actual_site_offset.tolist()}"
        )

    geom_displacements = {}
    for geom_id in hole_geom_ids:
        name = _geom_name(model, geom_id)
        displacement = np.asarray(data.geom_xpos[geom_id], dtype=np.float64) - nominal_geoms[name]
        geom_displacements[name] = displacement
        if not np.allclose(displacement, expected_displacement, atol=1.0e-7):
            raise ValueError(
                f"hole geom {name!r} displacement mismatch: expected="
                f"{expected_displacement.tolist()} actual={displacement.tolist()}"
            )

    reference_displacements = {}
    for name, nominal_position in nominal_references.items():
        body_id_ref = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        displacement = np.asarray(data.xpos[body_id_ref], dtype=np.float64) - nominal_position
        reference_displacements[name] = displacement
        if not np.allclose(displacement, np.zeros(3), atol=1.0e-10):
            raise ValueError(f"unrelated reference body {name!r} moved by {displacement.tolist()}")

    print(f"selected_hole_assembly_body={structure['hole_body_name']}")
    print(f"site_owner_body={structure['site_owner_body_name']}")
    print(f"selected_body_parent={structure['selected_body_parent_name']}")
    print(f"hole_site_name={structure['hole_site_name']}")
    print(f"hole_geom_names={structure['hole_geom_names']}")
    print(f"selected_body_subtree={structure['selected_body_subtree_names']}")
    print(f"nominal_hole_goal_position={nominal_site.tolist()}")
    print(f"actual_hole_goal_position={actual_site.tolist()}")
    print(f"requested_displacement={requested.tolist()}")
    print(f"actual_displacement={actual_site_offset.tolist()}")
    for name, displacement in geom_displacements.items():
        print(f"geom_displacement[{name}]={displacement.tolist()}")
    print(f"unrelated_reference_bodies_checked={list(reference_displacements)}")
    print("validation=ok")

    return {
        "structure": structure,
        "offset": offset_metadata,
        "geom_displacements": geom_displacements,
        "reference_displacements": reference_displacements,
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect hole assembly body/site/geoms.")
    parser.add_argument("--model-xml", type=Path, required=True)
    parser.add_argument("--hole-site-name", default=DEFAULT_HOLE_SITE_NAME)
    parser.add_argument("--hole-body-name", default=DEFAULT_HOLE_BODY_NAME)
    parser.add_argument("--test-offset-x", type=float, default=0.002)
    parser.add_argument("--test-offset-y", type=float, default=0.0)
    parser.add_argument("--test-offset-z", type=float, default=0.002)
    parser.add_argument("--offset-frame", choices=("world", "body"), default="world")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        inspect_hole_assembly(args)
    except Exception as error:
        print(f"error: hole assembly inspection failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
