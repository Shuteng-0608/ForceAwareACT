"""Offline goal mapping, CLI validation and normal occupancy; no SDK or fault injection."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from force_aware_act.control.contracts import Feedback, Pose, ShadowConfig
from force_aware_act.control.goal_selection import DEFAULT_PROFILE, select_goal
from force_aware_act.control.supervision import GoalMonitor
from scripts.deployment.shadow_contact_readonly import parse_arguments


ROOT = Path(__file__).resolve().parents[1]
HOLES = [f"x{x}y{y:02d}" for x in range(13, 18) for y in (7, 8)]


@pytest.mark.parametrize("hole", HOLES)
def test_each_hole_maps_to_its_original_fk_goal_and_common_limits(hole):
    selected = select_goal(hole)
    original = ShadowConfig.read(ROOT / f"configs/deployment/goals/{hole}.json")
    base = ShadowConfig.read(DEFAULT_PROFILE / "base.json")
    assert selected.config.goal == original.goal
    assert selected.config.limits == base.limits
    snapshot = json.loads(json.dumps(selected.to_dict()))
    assert snapshot["goal_id"] == hole and snapshot["locked_for_run"]
    assert snapshot["config_sha256"] == hashlib.sha256(selected.config_path.read_bytes()).hexdigest()
    assert ShadowConfig.from_dict(snapshot["config"]) == selected.config
    assert not snapshot["model_conditioned_on_goal_id"] and not selected.config.allow_motion
    assert snapshot["controller_baseline"]["velocity_ratio_percent"] == 10
    assert snapshot["controller_baseline"]["acceleration_ratio_percent"] == 10
    assert snapshot["controller_baseline"]["apply_on_load"] is False
    assert snapshot["controller_baseline_sha256"] == hashlib.sha256(
        (DEFAULT_PROFILE/"controller_baseline.json").read_bytes()).hexdigest()


@pytest.mark.parametrize("hole", ["", "x14y8", "X14Y08", "../x14y08", "x18y08"])
def test_invalid_or_unknown_labels_have_no_default(hole):
    with pytest.raises(ValueError):
        select_goal(hole)


def test_valid_hole_uses_only_its_own_completion_region_and_new_run_state():
    selected = select_goal("x14y08")
    other = select_goal("x14y07")
    monitor = GoalMonitor(selected.config.goal)
    other_monitor = GoalMonitor(other.config.goal)
    for ms in range(0, 501, 10):
        ns = ms * 1_000_000
        feedback = Feedback(ns, ms+1, (0.,)*7, (0.,)*7, (0.,)*7, ns, True, "normal_goal_occupancy")
        pose = Pose(ns, selected.config.goal.frame_id, selected.config.goal.target_position_m, (0.,0.,0.,1.))
        result = monitor.update(feedback, ns, pose)
        other_result = other_monitor.update(feedback, ns, pose)
    assert result["status"] == "COARSE_REACHED" and result["goal_id"] == "x14y08"
    assert other_result["status"] == "RUNNING" and not other_result["inside"]
    fresh = GoalMonitor(select_goal("x14y08").config.goal)
    assert fresh.inside_since is None and fresh.result["status"] == "RUNNING"


def run_args(tmp_path):
    return ["--checkpoint", "unused.pt", "--reader", "unused_reader", "--output", str(tmp_path/"run"),
            "--query-hz", "50", "--operator-confirmed-no-contact"]


def test_launcher_passes_selected_config_to_the_existing_controller(tmp_path):
    args, config, selection = parse_arguments(run_args(tmp_path)+["--goal-id", "x14y08"])
    assert config is selection.config and config.goal.goal_id == args.goal_id == "x14y08"
    assert not args.output.exists()


@pytest.mark.parametrize("options", [
    ["--controller-profile", str(DEFAULT_PROFILE)],
    ["--describe-goal"],
    ["--goal-id", "x14y08", "--controller-config", str(DEFAULT_PROFILE/"base.json")],
])
def test_ambiguous_or_missing_selection_is_rejected_before_device_loading(tmp_path, options):
    with pytest.raises(SystemExit) as result:
        parse_arguments(run_args(tmp_path)+options)
    assert result.value.code == 2


def test_existing_prediction_only_and_explicit_config_arguments_remain_supported(tmp_path):
    args, config, selection = parse_arguments(run_args(tmp_path))
    assert config is selection is None
    path = DEFAULT_PROFILE/"goals/x13y07.json"
    args, config, selection = parse_arguments(run_args(tmp_path)+["--controller-config", str(path)])
    assert config == ShadowConfig.read(path) and selection is None


def test_describe_cli_works_without_checkpoint_reader_output_or_hardware():
    result = subprocess.run(
        [sys.executable, "scripts/deployment/shadow_contact_readonly.py", "--goal-id", "x14y08", "--describe-goal"],
        cwd=ROOT, text=True, capture_output=True, check=True, timeout=20,
    )
    snapshot = json.loads(result.stdout)
    assert snapshot["goal_id"] == "x14y08"
    assert snapshot["radius_m"] == .003 and snapshot["dwell_ms"] == 500
    assert snapshot["config"]["allow_motion"] is False
