"""Exercise prior-driven selection/resume with scripted metrics and tiny states."""
from copy import deepcopy
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


@pytest.mark.parametrize("dimension,stop_step", [(3, 5), (6, 6)])
def test_prior_drives_all_best_checkpoints_and_stop_while_zero_keeps_improving(tmp_path, monkeypatch, capsys,
                                                                           dimension, stop_step):
    helper = module(Path(__file__).with_name("test_real_robot_multi_metric_cli.py"), "prior_cli_helper")
    runner, args = helper.setup(tmp_path, monkeypatch, "contact", dimension)
    prior_evaluation = runner.evaluate_episodes
    calls = []

    def evaluation(kind, model, *values, contact_latent_mode="zero", **kwargs):
        row = prior_evaluation(kind, model, *values, **kwargs)
        step = int(model.weight.item())
        calls.append((step, contact_latent_mode))
        if contact_latent_mode == "zero":
            # Every zero metric improves every check; it must never prevent stopping.
            row = {key: .0001 / step for key in row}
        return row

    monkeypatch.setattr(runner, "evaluate_episodes", evaluation)
    resumed_dir, full_dir = tmp_path / "resumed", tmp_path / "full"
    runner.main(args(resumed_dir, 3) + ["--contact-latent-mode", "prior_mean"])
    resumed = runner.main(args(resumed_dir, 8, explicit=False) + ["--resume", str(resumed_dir / "last.pt")])
    full = runner.main(args(full_dir, 8) + ["--contact-latent-mode", "prior_mean"])
    assert resumed["completed_steps"] == full["completed_steps"] == stop_step
    assert resumed["termination"] == "early_stopping"
    assert (resumed_dir / "metrics.jsonl").read_text() == (full_dir / "metrics.jsonl").read_text()
    for i in range(0, len(calls), 2):
        assert calls[i][1] == "prior_mean" and calls[i+1] == (calls[i][0], "zero")
    for row in map(json.loads, (full_dir / "metrics.jsonl").read_text().splitlines()):
        assert row["validation_contact_latent_mode"] == "prior_mean"
        assert row["zero_latent_reference"]["episode_macro_action_mae_rad"] < row["held_out_validation"]["episode_macro_action_mae_rad"]
    best_steps = {"best.pt": 2, "best_force.pt": 3}
    if dimension == 6:
        best_steps["best_moment.pt"] = 4
    for name, step in best_steps.items():
        saved = read_checkpoint(full_dir / name)
        assert saved["global_step"] == step
        assert saved["checkpoint_selection"]["contact_latent_mode"] == "prior_mean"
        assert saved["run_config"]["contact_latent_config"]["training"] == "posterior_sample"
    with pytest.raises(ValueError, match="cannot change contact latent mode"):
        runner.main(args(full_dir, 10) + ["--resume", str(full_dir / "last.pt"), "--contact-latent-mode", "zero"])

    saved = read_checkpoint(full_dir / "last.pt")
    expected = deepcopy(saved)
    expected["run_config"].pop("contact_latent_config")
    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad(): model.weight.fill_(99.)
    sampler = RealRobotTrainingSampler(list(saved["run_config"]["train_episode_valid_anchors"].values()),
                                       mode="episodic_uniform", seed=0)
    with pytest.raises(ValueError, match="contact_latent_config"):
        restore_checkpoint(saved, model=model, optimizer=torch.optim.AdamW(model.parameters()),
                           sampler=sampler, expected=expected)
    assert model.weight.item() == 99.
    with pytest.raises(ValueError, match="contact_latent_config"):
        runner.validate_output_identity(full_dir, expected, stop_step)

    monitor = module(Path(__file__).parents[1] / "scripts/monitor_real_robot_training.py", "prior_monitor")
    reader = monitor.IncrementalMetrics(full_dir / "metrics.jsonl")
    report = monitor.build_report(full_dir, reader, monitor.WallRate(), processes=[], include_gpu=False)
    assert report["contact_latent_mode"] == "prior_mean"
    rendered = monitor.render_report(report)
    assert "posterior 采样" in rendered and "仅记录，不参与早停" in rendered
    assert "zero 对照力" in rendered
    assert ("zero 对照力矩" in rendered) == (dimension == 6)
    capsys.readouterr()


def test_frozen_evaluator_inherits_saved_mode_and_records_explicit_override(tmp_path, monkeypatch, capsys):
    helper = module(Path(__file__).with_name("test_real_robot_multi_metric_cli.py"), "prior_eval_helper")
    runner, args = helper.setup(tmp_path, monkeypatch, "contact", 6)
    output = tmp_path / "run"
    runner.main(args(output, 1) + ["--contact-latent-mode", "prior_mean"])
    evaluator = module(Path(__file__).parents[1] / "scripts/evaluate_real_robot.py", "prior_frozen_evaluator")
    original = runner.build_model

    def build(kind, config, **kwargs):
        from force_aware_act.models.act_aligned import ACTAlignedHighRateConfig
        result = original()
        result.config = ACTAlignedHighRateConfig(**config)
        return result

    calls = []
    monkeypatch.setattr(evaluator, "build_model", build)
    monkeypatch.setattr(evaluator, "evaluate_episodes", lambda *a, **kw: calls.append(kw["contact_latent_mode"]) or {"action_mae_rad": .1})
    common = ["--checkpoint", str(output / "last.pt"), "--data-root", str(tmp_path / "data"),
              "--split-role", "validation", "--device", "cpu", "--threads", "1"]
    report = evaluator.main(common + ["--output", str(tmp_path / "prior.json")])
    assert report["latent_mode"] == "prior_mean" and not report["latent_mode_override"]
    report = evaluator.main(common + ["--output", str(tmp_path / "zero.json"), "--contact-latent-mode", "zero"])
    assert report["latent_mode"] == "zero" and report["latent_mode_override"]
    assert calls == ["prior_mean", "zero"]
    capsys.readouterr()
