"""Synthetic CLI integration: scripted validation and tiny checkpoint states.

No real dataset, pretrained network, gradient update, or GPU is used here.
"""
from copy import deepcopy
from dataclasses import replace
import importlib.util
import json
from pathlib import Path

import pytest
import torch

from force_aware_act.real_robot.checkpoint import read_checkpoint, restore_checkpoint
from force_aware_act.real_robot.training_sampling import RealRobotTrainingSampler


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def setup(tmp_path, monkeypatch, model_kind, dimension):
    repo = Path(__file__).parents[1]
    runner = module(repo / "scripts/train_real_robot.py", "multi_metric_cli")
    fixture = module(Path(__file__).with_name("test_real_robot_wrench6.py"), "multi_metric_sensor_fixture")
    root = tmp_path / "data"
    for name in ("recording_a", "recording_b", "recording_c"):
        fixture.episode(root, name)
    config = fixture.config()
    if dimension == 3:
        config = replace(config, force_representation="end_effector_force3",
                         wrench_component_units=("N",) * 3, force_std_floor=(.01,) * 3)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config.to_dict()))
    sequence = [(1., 1., 1.), (.8, .9, .8), (.9, .7, .9), (1., .8, .6), (1., .8, .7), (1., .8, .7)]

    def tiny_model(*args, **kwargs):
        result = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad(): result.weight.zero_()
        return result

    def advance_marker(kind, model, *args):
        with torch.no_grad(): model.weight.add_(1)
        return {"loss_total": .1, "loss_action": .1}

    def evaluate(kind, model, *args, **kwargs):
        index = int(model.weight.item()) - 1
        action, force, moment = sequence[index]
        result = {"episode_macro_action_mae_rad": action}
        if kind != "act":
            result.update(episode_macro_force_endpoint_mae_n=force, episode_macro_force_mae_n=force)
            if dimension == 6:
                result.update(episode_macro_moment_endpoint_mae_nm=moment, episode_macro_moment_mae_nm=moment)
        return result

    monkeypatch.setattr(runner, "build_model", tiny_model)
    monkeypatch.setattr(runner, "build_optimizer", lambda model, _: torch.optim.AdamW(model.parameters()))
    monkeypatch.setattr(runner, "build_criterion", lambda *_: None)
    monkeypatch.setattr(runner, "train_step", advance_marker)
    monkeypatch.setattr(runner, "evaluate_episodes", evaluate)

    def args(output, steps, explicit=True):
        return ["--data-root", str(root), "--data-config", str(path), "--output-dir", str(output),
                "--mode", "train", "--model", model_kind, "--compact", "--device", "cpu", "--threads", "1",
                "--seed", "0", "--chunk-len", "3", "--batch-size", "2", "--steps", str(steps),
                "--train-sampling-mode", "episodic_uniform", "--validation-interval", "1", "--checkpoint-interval", "1",
                "--selection-metric", "episode_macro_action_mae_rad", "--validation-anchor-stride", "10",
                "--minimum-sampling-epochs", "1", "--early-stop-patience", "2", "--early-stop-min-relative-improvement", ".001",
                *(["--early-stop-metrics", "multi"] if explicit else [])]
    return runner, args


@pytest.mark.parametrize("model_kind,dimension,stop_step", [
    ("act", 6, 4), ("contact", 6, 6), ("motion", 6, 6), ("contact", 3, 5), ("motion", 3, 5),
])
def test_distinct_best_checkpoints_all_metric_stopping_and_exact_resume(tmp_path, monkeypatch, capsys,
                                                                      model_kind, dimension, stop_step):
    runner, args = setup(tmp_path, monkeypatch, model_kind, dimension)
    resumed_dir, full_dir = tmp_path / "resumed", tmp_path / "full"
    runner.main(args(resumed_dir, 3))
    old_action = (resumed_dir / "best.pt").read_bytes()
    resumed = runner.main(args(resumed_dir, 8, explicit=False) + ["--resume", str(resumed_dir / "last.pt")])
    full = runner.main(args(full_dir, 8))
    assert resumed["completed_steps"] == full["completed_steps"] == stop_step
    assert resumed["termination"] == full["termination"] == "early_stopping"
    assert (resumed_dir / "best.pt").read_bytes() == old_action
    assert (resumed_dir / "metrics.jsonl").read_text() == (full_dir / "metrics.jsonl").read_text()
    expected = {"best.pt": 2, "last.pt": stop_step}
    if model_kind != "act": expected["best_force.pt"] = 3
    if model_kind != "act" and dimension == 6: expected["best_moment.pt"] = 4
    assert {p.name for p in resumed_dir.glob("*.pt")} == set(expected)
    for name, step in expected.items():
        left, right = read_checkpoint(resumed_dir / name), read_checkpoint(full_dir / name)
        assert left["global_step"] == right["global_step"] == step
        assert left["model_state"]["weight"].item() == step
        torch.testing.assert_close(left["model_state"]["weight"], right["model_state"]["weight"], rtol=0, atol=0)
        assert left["multi_metric_state"] == right["multi_metric_state"]
        if name != "last.pt": assert left["checkpoint_selection"]["step"] == step
    assert resumed["checkpoint_round_trip"] == "passed"
    with pytest.raises(ValueError, match="already satisfied"):
        runner.main(args(resumed_dir, 9, explicit=False) + ["--resume", str(resumed_dir / "last.pt")])
    with pytest.raises(ValueError, match="cannot change"):
        runner.main(args(resumed_dir, 9, explicit=False) + ["--resume", str(resumed_dir / "last.pt"), "--early-stop-metrics", "action"])
    capsys.readouterr()


def test_corrupt_control_and_changed_metric_contract_rejected_before_parameter_restore(tmp_path, monkeypatch, capsys):
    runner, args = setup(tmp_path, monkeypatch, "contact", 6)
    output = tmp_path / "run"
    runner.main(args(output, 2))
    payload = read_checkpoint(output / "last.pt")
    expected = deepcopy(payload)
    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad(): model.weight.fill_(99)
    optimizer = torch.optim.AdamW(model.parameters())
    sampler = RealRobotTrainingSampler(list(payload["run_config"]["train_episode_valid_anchors"].values()),
                                       mode="episodic_uniform", seed=0)
    damaged = deepcopy(payload)
    damaged["multi_metric_state"]["metrics"]["moment"]["best"]["step"] = 999
    with pytest.raises(ValueError, match="identity"):
        restore_checkpoint(damaged, model=model, optimizer=optimizer, sampler=sampler, expected=expected)
    assert model.weight.item() == 99
    changed = deepcopy(expected)
    changed["run_config"]["validation_objectives"]["metrics"]["force"]["components"]["episode_macro_force_mae_n"] = .5
    with pytest.raises(ValueError, match="validation_objectives"):
        restore_checkpoint(payload, model=model, optimizer=optimizer, sampler=sampler, expected=changed)
    with pytest.raises(ValueError, match="validation_objectives"):
        runner.validate_output_identity(output, changed, 2)
    assert model.weight.item() == 99
    capsys.readouterr()
