#!/usr/bin/env python3
"""Frozen paired random-target and physical pulse evaluation; plan-only by default."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
from dataclasses import asdict, dataclass

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from force_aware_act.rollout_disturbance import DISTURBANCE_VERSION, PulseConfig
from scripts.run_sampling_ablation_nominal_rollout_pilot import (
    DEFAULT_CHECKPOINT_ROOT, build_specs, validate_spec_files,
)
from scripts.run_sampling_ablation_xz_cardinal_pilot import (
    OffsetPoint, build_rollout_command as offset_command,
    validate_completed_summary as validate_offset_summary, result_row,
)

SUITE_VERSION = "sampling_random_targets_physical_recovery_v1"
MODEL_XML = Path("/home/stw/ForceAwareACT_workspace/arm_teleop/model/pangu_moving_hole_fixed_peg.xml")
DEFAULT_OUTPUT = REPO_ROOT / "runs/moving_hole_fixed_peg_random_recovery_v1_seed0"
PILOT_CASES = ("control", "random_01", "rim_2n_xp", "mid_6n_xp")


@dataclass(frozen=True)
class Case:
    case_id: str
    kind: str
    x_m: float = 0.
    z_m: float = 0.
    peak_n: float = 0.
    remaining_m: float = .025
    direction: str = "xp"

    def point(self):
        return OffsetPoint(self.case_id, self.x_m, self.z_m)

    def pulse(self):
        return PulseConfig(self.peak_n, .5, self.remaining_m, self.direction)


def build_cases(point_seed=20260907):
    rng = random.Random(point_seed)
    cases = [Case("control", "control")]
    # Four uniform-area points per annulus, independent uniform azimuth. These
    # strata are equally weighted in reports, NOT a uniform sample of the disk.
    for lo, hi in ((0., 1.5), (1.5, 3.)):
        for _ in range(4):
            radius = math.sqrt(rng.uniform(lo**2, hi**2)) / 1000
            angle = rng.uniform(0., 2 * math.pi)
            cases.append(Case(f"random_{len(cases):02d}", "random_target",
                              radius * math.cos(angle), radius * math.sin(angle)))
    for phase, remaining in (("rim", .045), ("mid", .025)):
        for peak in (2., 6.):
            for direction in ("xp", "xm", "zp", "zm"):
                cases.append(Case(f"{phase}_{peak:g}n_{direction}", "physical_pulse",
                                  peak_n=peak, remaining_m=remaining, direction=direction))
    return tuple(cases)


def command(args, case, spec):
    cmd = offset_command(args, case.point(), spec)
    if case.peak_n:
        cmd += ["--disturbance-peak-n", str(case.peak_n),
                "--disturbance-duration-s", ".5", "--disturbance-remaining-m", str(case.remaining_m),
                "--disturbance-direction", case.direction]
    return cmd


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def make_plan(args, cases, specs):
    sources = sorted([*(REPO_ROOT / "src").rglob("*.py"), *(REPO_ROOT / "scripts").rglob("*.py")])
    return {
        "suite_version": SUITE_VERSION, "disturbance_version": DISTURBANCE_VERSION,
        "point_seed": args.point_seed, "policy_seed": args.seed,
        "max_rollout_steps": args.max_rollout_steps, "device": args.device,
        "save_videos": args.save_videos,
        "model_xml_sha256": sha256(MODEL_XML),
        "checkpoint_sha256": {s.model_id: sha256(s.checkpoint) for s in specs},
        "source_sha256": {str(p.relative_to(REPO_ROOT)): sha256(p) for p in sources},
        "cases": [asdict(c) for c in cases],
        "runs": [{"case_id": c.case_id, "model_id": s.model_id,
                  "command": command(args, c, s)} for c in cases for s in specs],
    }


def validate_summary(summary, args, case, spec):
    errors = validate_offset_summary(summary, args, case.point(), spec)
    if summary.get("architecture_version") != spec.architecture_version:
        errors.append("architecture_version mismatch")
    pulse = summary.get("disturbance")
    if case.peak_n:
        if not pulse or pulse.get("version") != DISTURBANCE_VERSION or pulse.get("config") != asdict(case.pulse()):
            errors.append("disturbance contract mismatch")
        elif pulse["recovered"] and not (pulse["fully_released"] and summary["success"]):
            errors.append("invalid recovery classification")
    elif pulse is not None:
        errors.append("unexpected physical pulse")
    return errors


def validate_artifacts(directory, summary, videos):
    errors = []
    with (directory / "rollout_log.csv").open() as handle:
        count = sum(1 for _ in csv.DictReader(handle))
    if count != summary["steps_executed"]:
        errors.append("rollout CSV row count mismatch")
    pulse = summary.get("disturbance")
    if pulse and pulse["triggered"]:
        with (directory / "disturbance_trace.csv").open() as handle:
            count = sum(1 for _ in csv.DictReader(handle))
        if count != pulse["trace_rows"] or not count:
            errors.append("physics disturbance trace row count mismatch")
    if videos:
        import cv2
        for camera in ("ee_cam", "base_top_cam"):
            cap = cv2.VideoCapture(str(directory / "videos" / f"{camera}.mp4"))
            if not cap.isOpened() or int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) != summary["steps_executed"]:
                errors.append(f"{camera} video frame count mismatch")
            cap.release()
    return errors


def aggregate_row(case, spec, summary):
    row = result_row(case.point(), spec, summary)
    pulse = summary.get("disturbance", {})
    row.update(case_kind=case.kind, pulse_peak_n=case.peak_n,
               pulse_direction=case.direction if case.peak_n else "",
               pulse_remaining_m=case.remaining_m if case.peak_n else None)
    for key in ("triggered", "fully_released", "recovered", "contact_recovery",
                "physical_contact_during_pulse", "trigger_time", "release_time",
                "delivered_impulse_ns", "trigger_lateral_m", "max_lateral_during_m",
                "peak_sensor_force_during_n", "peak_contact_force_during_n",
                "recovery_time_after_release_s"):
        row[key] = pulse.get(key)
    return row


def write_results(root, rows, total):
    with (root / "aggregate.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    grouped = {}
    for row in rows:
        group = grouped.setdefault(row["model_id"], {})
        result = group.setdefault(row["case_kind"], dict(completed=0, success=0,
            force_stops=0, triggered=0, released=0, recovered=0, contact_exposed=0, contact_recovered=0))
        result["completed"] += 1
        result["success"] += row["success"] is True
        result["force_stops"] += row["stop_reason"] == "force_stop_threshold"
        for key, column in (("triggered", "triggered"), ("released", "fully_released"),
                            ("recovered", "recovered"), ("contact_exposed", "physical_contact_during_pulse"),
                            ("contact_recovered", "contact_recovery")):
            result[key] += row[column] is True
    payload = dict(suite_version=SUITE_VERSION, completed=len(rows), planned=total, by_model=grouped)
    (root / "suite_summary.json").write_text(json.dumps(payload, indent=2) + "\n")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--point-seed", type=int, default=20260907)
    parser.add_argument("--max-rollout-steps", type=int, default=1200)
    parser.add_argument("--save-videos", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--pilot-only", action="store_true", help="run 16 preregistered engineering checks from the full frozen plan")
    args = parser.parse_args(argv)
    args.checkpoint_root = args.checkpoint_root.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    if args.max_rollout_steps <= 0:
        parser.error("max-rollout-steps must be positive")
    return args


def main(argv=None):
    args = parse_args(argv)
    specs, cases = build_specs(args.checkpoint_root), build_cases(args.point_seed)
    validate_spec_files(specs)
    plan = make_plan(args, cases, specs)
    if not args.execute:
        print(json.dumps(plan, indent=2))
        print("plan_only=true; no files written or simulation executed")
        return 0
    root = args.output_dir
    plan_path = root / "suite_plan.json"
    if root.exists():
        if not plan_path.is_file() or json.loads(plan_path.read_text()) != plan:
            raise ValueError("refusing output reuse: plan, code, checkpoint, or XML mismatch")
    else:
        root.mkdir(parents=True)
        plan_path.write_text(json.dumps(plan, indent=2) + "\n")
    environment = os.environ.copy()
    environment["MUJOCO_GL"] = "egl"
    environment["PYTHONPATH"] = str(REPO_ROOT / "src")
    rows = []
    for case in cases:
        for spec in specs:
            directory = root / case.case_id / spec.model_id
            summary_path = directory / "summary.json"
            selected = not args.pilot_only or case.case_id in PILOT_CASES
            if directory.exists() and not summary_path.is_file():
                raise ValueError(f"refusing to overwrite incomplete run: {directory}")
            if not directory.exists() and selected:
                directory.parent.mkdir(parents=True, exist_ok=True)
                print(f"starting {case.case_id}/{spec.model_id}", flush=True)
                with (directory.parent / f"{spec.model_id}.process.log").open("w") as log:
                    result = subprocess.run(command(args, case, spec), cwd=REPO_ROOT,
                                            env=environment, stdout=log, stderr=subprocess.STDOUT)
                if result.returncode:
                    raise RuntimeError(f"rollout failed ({result.returncode}): {directory}; see process log")
            if summary_path.exists():
                summary = json.loads(summary_path.read_text())
                errors = validate_summary(summary, args, case, spec)
                if summary.get("model_xml_sha256") != plan["model_xml_sha256"]:
                    errors.append("XML SHA256 mismatch")
                errors += validate_artifacts(directory, summary, args.save_videos)
                if errors:
                    raise ValueError(f"{directory}: {errors}")
                row = aggregate_row(case, spec, summary)
                rows.append(row)
                write_results(root, rows, len(cases) * len(specs))
                print(f"validated {len(rows)}/{len(cases)*len(specs)}: {case.case_id}/{spec.model_id} "
                      f"success={summary['success']} stop={summary['stop_reason']} "
                      f"triggered={row['triggered']} recovered={row['recovered']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
