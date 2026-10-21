"""Validation-only replay and synthetic-state tests; no experiment is launched."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from force_aware_act.real_robot.multi_metric_control import (
    MultiMetricController, resolve_metric_mode, validation_objectives,
)
from force_aware_act.real_robot.training_control import EarlyStoppingConfig, EarlyStoppingController


def values(action=1., force=1., moment=1.):
    return {"episode_macro_action_mae_rad": action,
            "episode_macro_force_endpoint_mae_n": force,
            "episode_macro_force_mae_n": force,
            "episode_macro_moment_endpoint_mae_nm": moment,
            "episode_macro_moment_mae_nm": moment}


def controller(kind="contact", representation="end_effector_wrench6", config=None):
    return MultiMetricController(config or EarlyStoppingConfig(1, 2, .001),
                                 validation_objectives(kind, representation, "episode_macro_action_mae_rad"))


@pytest.mark.parametrize("kind,representation,names", [
    ("act", "end_effector_wrench6", ["action"]),
    ("contact", "end_effector_wrench6", ["action", "force", "moment"]),
    ("motion", "end_effector_wrench6", ["action", "force", "moment"]),
    ("contact", "end_effector_force3", ["action", "force"]),
    ("motion", "flange_force3", ["action", "force"]),
    ("motion", "joint_torque7", ["action", "torque"]),
])
def test_objective_mapping_and_units(kind, representation, names):
    c = controller(kind, representation)
    assert list(c.controls) == names
    assert c.objectives["metrics"]["action"]["checkpoint"] == "best.pt"
    if "force" in names:
        v = values()
        v["episode_macro_force_endpoint_mae_n"] = 3
        v["episode_macro_force_mae_n"] = 6
        c.update(v, completed_epochs=1, step=5)
        assert c.values["force"] == 4
        assert c.best["force"]["unit"] == "N"
    if "moment" in names:
        assert c.best["moment"]["unit"] == "N*m"
    if "torque" in names:
        c.update({"episode_macro_action_mae_rad": 1., "episode_macro_torque_endpoint_mae_nm": 3.,
                  "episode_macro_torque_mae_nm": 6.}, completed_epochs=1, step=5)
        assert c.values["torque"] == 4 and c.best["torque"]["checkpoint"] == "best_torque.pt"


def test_independent_patience_can_recover_and_only_all_stalled_stops():
    c = controller()
    sequence = [values(), values(), values(force=.8), values(force=.8, moment=.8),
                values(force=.8, moment=.8), values(force=.8, moment=.8)]
    for step, v in enumerate(sequence, 1):
        stopped = c.update(v, completed_epochs=step, step=step * 5)
        assert stopped == (step == 6)
        if step == 3:
            assert c.controls["action"].stopped and c.controls["moment"].stopped
        if step == 4:
            assert not c.controls["moment"].stopped and c.controls["moment"].bad_checks == 0
    assert c.best["action"]["epoch"] == 1
    assert c.best["force"]["epoch"] == 3
    assert c.best["moment"]["epoch"] == 4


def test_checkpoint_minimum_is_exact_but_patience_uses_relative_threshold():
    c = controller(config=EarlyStoppingConfig(1, 2, .01))
    c.update(values(), completed_epochs=1, step=5)
    c.update(values(force=.995), completed_epochs=2, step=10)
    assert c.improved_metrics == ["force"]
    assert c.best["force"]["value"] == pytest.approx(.995)
    assert c.controls["force"].reference_metric == 1
    assert c.controls["force"].bad_checks == 1
    assert c.update(values(force=.995), completed_epochs=3, step=15)


@pytest.mark.parametrize("invalid", [None, float("nan"), float("inf"), -1., True, "1"])
def test_invalid_last_component_does_not_partially_mutate_state(invalid):
    c = controller()
    c.update(values(), completed_epochs=1, step=5)
    before = c.state_dict()
    v = values(action=.5, force=.5)
    v["episode_macro_moment_mae_nm"] = invalid
    with pytest.raises(ValueError, match="requires finite"):
        c.update(v, completed_epochs=2, step=10)
    assert before == c.state_dict()


def test_absent_metric_and_step_reversal_are_rejected():
    c = controller()
    v = values(); del v["episode_macro_force_mae_n"]
    with pytest.raises(ValueError, match="requires finite"):
        c.update(v, completed_epochs=1, step=5)
    c.update(values(), completed_epochs=1, step=5)
    with pytest.raises(ValueError, match="steps"):
        c.update(values(), completed_epochs=1, step=5)
    with pytest.raises(ValueError, match="epochs"):
        c.update(values(), completed_epochs=0, step=10)


def test_disabled_stopping_and_act_match_legacy_exactly():
    for patience in (0, 2):
        config = EarlyStoppingConfig(1, patience, .001)
        c = controller(kind="act", config=config)
        legacy = EarlyStoppingController(config)
        for epoch, metric in enumerate([1., .9995, .9, .95, .96], 1):
            assert c.update(values(action=metric), completed_epochs=epoch, step=epoch * 5) == legacy.update(
                metric, completed_epochs=epoch, step=epoch * 5)
            assert c.controls["action"].state_dict() == legacy.state_dict()


def test_serialized_resume_matches_uninterrupted_control_and_best_selection():
    c = controller()
    c.update(values(), completed_epochs=1, step=5)
    c.update(values(force=.8), completed_epochs=2, step=10)
    restored = controller()
    restored.load_state_dict(json.loads(json.dumps(c.state_dict())))
    for epoch in range(3, 6):
        for instance in (c, restored):
            instance.update(values(force=.8, moment=.8), completed_epochs=epoch, step=epoch * 5)
        assert c.state_dict() == restored.state_dict()
        assert c.improved_metrics == restored.improved_metrics


@pytest.mark.parametrize("damage", ["objectives", "missing_metric", "counter", "step", "best", "flag", "unit", "components"])
def test_corrupt_resume_is_rejected_without_partial_load(damage):
    c = controller()
    c.update(values(), completed_epochs=1, step=5)
    state = deepcopy(c.state_dict())
    if damage == "objectives": state["objectives"]["metrics"]["force"]["components"]["episode_macro_force_mae_n"] = .5
    if damage == "missing_metric": del state["metrics"]["moment"]
    if damage == "counter": state["metrics"]["moment"]["controller"]["bad_checks"] = -1
    if damage == "step": state["metrics"]["force"]["controller"]["last_validation_step"] = 4
    if damage == "best": state["metrics"]["force"]["best"]["value"] = float("nan")
    if damage == "flag": state["stopped"] = True
    if damage == "unit": state["metrics"]["force"]["best"]["unit"] = "N*m"
    if damage == "components": state["metrics"]["force"]["best"]["components"]["episode_macro_force_mae_n"] = 2
    target = controller()
    before = target.state_dict()
    with pytest.raises(ValueError): target.load_state_dict(state)
    assert target.state_dict() == before


def test_mode_resolution_preserves_legacy_and_rejects_silent_migration():
    assert resolve_metric_mode(None) == "action"
    assert resolve_metric_mode(None, {}) == "action"
    saved = {"validation_objectives": controller().objectives}
    assert resolve_metric_mode(None, saved) == "multi"
    assert resolve_metric_mode("multi", saved) == "multi"
    with pytest.raises(ValueError, match="cannot change"): resolve_metric_mode("multi", {})
    with pytest.raises(ValueError, match="cannot change"): resolve_metric_mode("action", saved)
    with pytest.raises(ValueError, match="episode_macro"):
        validation_objectives("contact", "end_effector_wrench6", "action_mae_rad")


@pytest.mark.parametrize("kind,expected", [("contact", [20, 10, 21]), ("motion", [20, 7, 13])])
def test_completed_4090_validation_replay(kind, expected):
    fixture = json.loads((Path(__file__).parent / "fixtures/x14y08_multi_metric_replay.json").read_text())
    c = controller(kind=kind, config=EarlyStoppingConfig(3000, 20, .001))
    for row in fixture["models"][kind]:
        c.update(row["metrics"], completed_epochs=row["epoch"], step=row["step"])
    assert [control.bad_checks for control in c.controls.values()] == expected
    assert not c.stopped
