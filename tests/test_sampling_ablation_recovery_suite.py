from dataclasses import asdict
import math

import pytest

from force_aware_act.rollout_disturbance import DISTURBANCE_VERSION
from scripts.run_sampling_ablation_recovery_suite import (
    PILOT_CASES, build_cases, command, parse_args, validate_summary,
)
from scripts.run_sampling_ablation_nominal_rollout_pilot import build_specs


def test_frozen_paired_matrix_and_random_strata():
    cases = build_cases()
    assert len(cases) == 25
    assert len({c.case_id for c in cases}) == 25
    assert cases == build_cases()
    assert cases[1:9] != build_cases(1)[1:9]
    assert cases[9:] == build_cases(1)[9:]
    assert set(PILOT_CASES) <= {c.case_id for c in cases}
    for i, c in enumerate(cases[1:9]):
        radius = math.hypot(c.x_m, c.z_m)
        assert (0 <= radius <= .0015) if i < 4 else (.0015 <= radius <= .003)
        angle = math.atan2(c.z_m, c.x_m) % (2 * math.pi)
        assert int(angle / (math.pi / 2)) == i % 4
    assert {c.direction for c in cases[9:]} == {"xp", "xm", "zp", "zm"}
    assert {c.peak_n for c in cases[9:]} == {2, 6}
    assert {c.remaining_m for c in cases[9:]} == {.025, .045}
    assert all(c.x_m == c.z_m == 0 for c in cases[9:])


def test_defaults_are_plan_only_and_commands_preserve_controls(tmp_path):
    args = parse_args(["--output-dir", str(tmp_path)])
    assert not args.execute
    spec = build_specs(args.checkpoint_root)[0]
    for case in build_cases():
        cmd = command(args, case, spec)
        assert "--execute-actions" not in cmd
        assert "--enable-axial-push" not in cmd
        assert cmd[cmd.index("--max-rollout-steps") + 1] == "1200"
        assert cmd[cmd.index("--contact-latent-mode") + 1] == "zero"
        assert ("--disturbance-peak-n" in cmd) == bool(case.peak_n)
    args.execute = True
    assert "--execute-actions" in command(args, build_cases()[0], spec)


def test_resume_rejects_changed_pulse_contract(tmp_path, monkeypatch):
    import scripts.run_sampling_ablation_recovery_suite as suite
    args = parse_args(["--output-dir", str(tmp_path)])
    case = build_cases()[9]
    spec = build_specs(args.checkpoint_root)[0]
    monkeypatch.setattr(suite, "validate_offset_summary", lambda *a: [])
    summary = {"architecture_version": spec.architecture_version,
               "disturbance": {"version": DISTURBANCE_VERSION,
                               "config": asdict(case.pulse()), "recovered": False}}
    assert not validate_summary(summary, args, case, spec)
    summary["disturbance"]["config"]["peak_n"] = 9
    assert validate_summary(summary, args, case, spec)
    assert validate_summary(summary, args, build_cases()[0], spec)


def test_nonpositive_horizon_rejected():
    with pytest.raises(SystemExit):
        parse_args(["--max-rollout-steps", "0"])
