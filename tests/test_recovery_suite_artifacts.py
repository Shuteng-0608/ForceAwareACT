"""Opt-in acceptance test for actual completed MuJoCo suite artifacts.

FORCEACT_RECOVERY_ARTIFACT_DIR=/absolute/run/root PYTHONPATH=src python -m pytest
    -q tests/test_recovery_suite_artifacts.py
This test only reads artifacts; it never launches or changes a simulation.
"""
import csv
import json
import os
from pathlib import Path

import numpy as np
import pytest

from force_aware_act.rollout_disturbance import DIRECTIONS
from scripts.run_sampling_ablation_recovery_suite import (
    aggregate_row, build_cases, build_specs, make_plan, parse_args,
    validate_artifacts, validate_summary,
)


@pytest.mark.skipif(not os.environ.get("FORCEACT_RECOVERY_ARTIFACT_DIR"),
                    reason="requires an explicitly selected, completed simulation suite")
def test_completed_real_rollout_suite_is_reproducible_and_internally_consistent():
    root = Path(os.environ["FORCEACT_RECOVERY_ARTIFACT_DIR"]).resolve()
    args = parse_args(["--execute", "--output-dir", str(root)])
    cases, specs = build_cases(args.point_seed), build_specs(args.checkpoint_root)
    plan = json.loads((root / "suite_plan.json").read_text())
    assert plan == make_plan(args, cases, specs), "execution source/checkpoint/plan fingerprint changed"
    assert len(list(root.glob("*/*/summary.json"))) == len(cases) * len(specs) == 100
    expected_rows, expected_groups = [], {}
    for case in cases:
        for spec in specs:
            directory = root / case.case_id / spec.model_id
            s = json.loads((directory / "summary.json").read_text())
            assert not validate_summary(s, args, case, spec), str(directory)
            assert not validate_artifacts(directory, s, True), str(directory)
            assert s["model_xml_sha256"] == plan["model_xml_sha256"]
            assert s["actual_target_offset"] == pytest.approx([case.x_m, 0, case.z_m], abs=1e-12)
            assert 0 < s["steps_executed"] <= 1200
            assert np.isfinite([s[k] for k in ("final_time", "min_peg_to_hole_dist",
                               "final_peg_to_hole_dist", "force_metrics_compensated_max_force_norm")]).all()
            with (directory / "rollout_log.csv").open() as handle:
                log = list(csv.DictReader(handle))
            keys = [f"{prefix}_{i}" for prefix in ("qpos", "qvel", "qcmd") for i in range(7)]
            keys += [f"ft_{i}" for i in range(6)]
            assert np.isfinite([[float(r[k]) for k in keys] for r in log]).all(), str(directory)
            assert (np.diff([float(r["time"]) for r in log]) > 0).all()
            for axis in "xyz":
                coordinates = [float(r[f"peg_tip_{axis}"]) for r in log]
                assert max(coordinates) - min(coordinates) < 1e-12
            d = s.get("disturbance", {})
            if d.get("triggered"):
                with (directory / "disturbance_trace.csv").open() as handle:
                    trace = list(csv.DictReader(handle))
                values = {k: np.array([float(r[k]) for r in trace]) for k in trace[0] if k != "released"}
                assert all(np.isfinite(v).all() for v in values.values())
                timestamps = values["time"]
                assert np.diff(timestamps) == pytest.approx(np.full(len(trace) - 1, .001), abs=1e-10)
                assert timestamps[0] == pytest.approx(d["trigger_time"] + .001, abs=1e-10)
                elapsed = timestamps - .001 - d["trigger_time"]
                expected_scale = np.where((elapsed >= -1e-10) & (elapsed < .5 - 1e-10),
                                          case.peak_n * np.sin(np.pi * elapsed / .5)**2, 0.)
                expected_force = expected_scale[:, None] * np.array(DIRECTIONS[case.direction])[None, :]
                actual_force = np.stack([values[f"applied_f{axis}_n"] for axis in "xyz"], axis=1)
                assert actual_force == pytest.approx(expected_force, abs=1e-8)
                impulse = float(np.linalg.norm(actual_force, axis=1).sum() * .001)
                assert impulse == pytest.approx(d["delivered_impulse_ns"], abs=1e-8)
                released = np.array([r["released"] == "True" for r in trace])
                if d["fully_released"]:
                    assert impulse == pytest.approx(case.peak_n * .5 / 2, abs=1e-8)
                    assert released.any()
                    assert d["release_time"] - d["trigger_time"] == pytest.approx(.5, abs=1e-9)
                    assert not actual_force[released].any()
                assert d["fully_released"] == bool(released.any())
                actual_contact = bool((values["peg_socket_contact_force_n"][~released] > .1).any())
                assert d["physical_contact_during_pulse"] == actual_contact
                assert d["peak_sensor_force_during_n"] == pytest.approx(values["sensor_force_n"][~released].max())
                assert d["peak_contact_force_during_n"] == pytest.approx(values["peg_socket_contact_force_n"][~released].max())
                assert d["max_lateral_during_m"] == pytest.approx(values["lateral_m"][~released].max())
                recovered = bool(d["fully_released"] and s["success"] and s["success_time"] > d["release_time"])
                assert d["recovered"] == recovered
                assert d["contact_recovery"] == (actual_contact and recovered)
            elif d:
                assert not d["recovered"] and not d["contact_recovery"]
                assert not (directory / "disturbance_trace.csv").exists()
            expected_rows.append(aggregate_row(case, spec, s))
            group = expected_groups.setdefault(spec.model_id, {}).setdefault(case.kind, [])
            group.append(s)

    with (root / "aggregate.csv").open() as handle:
        actual_rows = list(csv.DictReader(handle))
    assert len(actual_rows) == len(expected_rows)
    for actual, expected in zip(actual_rows, expected_rows):
        assert actual == {k: "" if v is None else str(v) for k, v in expected.items()}
    summary = json.loads((root / "suite_summary.json").read_text())
    assert summary["completed"] == summary["planned"] == 100
    for model, kinds in expected_groups.items():
        for kind, runs in kinds.items():
            expected = dict(completed=len(runs), success=sum(s["success"] for s in runs),
                            force_stops=sum(s["stop_reason"] == "force_stop_threshold" for s in runs))
            for key, field in (("triggered", "triggered"), ("released", "fully_released"),
                               ("recovered", "recovered"), ("contact_exposed", "physical_contact_during_pulse"),
                               ("contact_recovered", "contact_recovery")):
                expected[key] = sum(s.get("disturbance", {}).get(field, False) for s in runs)
            assert summary["by_model"][model][kind] == expected
