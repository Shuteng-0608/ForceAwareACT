"""KL ablation contracts, loss/gradient equivalence and resumable update clock."""
from copy import deepcopy
from dataclasses import replace
import importlib.util
import json
import os
from pathlib import Path
import random
import subprocess

import numpy as np
import pytest
import torch

from force_aware_act.real_robot.kl_schedule import (
    PosteriorKLSchedule, resolve_kl_schedule, schedule_from_metadata, validate_kl_schedule_identity,
)
from force_aware_act.act_aligned_training.high_rate_losses import ACTAlignedHighRateCriterion
from force_aware_act.act_aligned_training.high_rate_config import ACTAlignedHighRateTrainingConfig
from force_aware_act.real_robot.checkpoint import read_checkpoint, restore_checkpoint, sampler_states_equal
from force_aware_act.real_robot.training_sampling import RealRobotTrainingSampler

ROOT = Path(__file__).parents[1]


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_schedule_boundary_resume_and_legacy():
    schedule = PosteriorKLSchedule("linear_warmup", 10., 1000, 5)
    assert [schedule.weight(s) for s in (0, 1, 2500, 4999, 5000, 8000)] == pytest.approx([0., .002, 5., 9.998, 10., 10.])
    assert schedule.state(5000) == pytest.approx(dict(completed_updates=5000, last_update_weight=9.998, next_update_weight=10.))
    saved = dict(run_config={"posterior_kl_schedule": schedule.to_dict()}, training_config={"posterior_kl_weight": 10.})
    assert resolve_kl_schedule(None, None, target_weight=10., steps_per_epoch=5, episodic_uniform=True, saved_metadata=saved) == schedule
    with pytest.raises(ValueError, match="cannot change"):
        resolve_kl_schedule("constant", 0, target_weight=10., steps_per_epoch=5, episodic_uniform=True, saved_metadata=saved)
    with pytest.raises(ValueError, match="episodic_uniform"):
        resolve_kl_schedule("linear_warmup", 1000, target_weight=10., steps_per_epoch=5, episodic_uniform=False)
    for weight in (1., 10.):
        legacy = dict(run_config={}, training_config={"posterior_kl_weight": weight})
        explicit = deepcopy(legacy)
        explicit["run_config"]["posterior_kl_schedule"] = PosteriorKLSchedule(target_weight=weight).to_dict()
        validate_kl_schedule_identity(legacy, explicit)
        assert schedule_from_metadata(legacy).weight(5000) == weight
    with pytest.raises(ValueError, match="target differs"):
        schedule_from_metadata(dict(run_config=saved["run_config"], training_config={"posterior_kl_weight": 1.}))


@pytest.mark.parametrize("kwargs", [
    dict(kind="bad"), dict(version="bad"), dict(target_weight=float("nan")),
    dict(target_weight=float("inf")), dict(target_weight=-1), dict(target_weight=True),
    dict(warmup_epochs=1), dict(kind="linear_warmup"), dict(kind="linear_warmup", warmup_epochs=1.5, steps_per_epoch=5),
    dict(kind="linear_warmup", warmup_epochs=1000, steps_per_epoch=0),
])
def test_invalid_schedule(kwargs):
    with pytest.raises(ValueError): PosteriorKLSchedule(**kwargs)


def loss_inputs():
    torch.manual_seed(14)
    outputs = {"pred_action": torch.randn(2, 3, 7, requires_grad=True),
               "pred_force": torch.randn(2, 3, 6, requires_grad=True),
               "pred_force_highrate": torch.randn(2, 3, 4, 6, requires_grad=True)}
    for key in ("mu_contact", "logvar_contact", "mu_contact_prior", "logvar_contact_prior"):
        outputs[key] = torch.randn(2, 32, requires_grad=True)
    masks = (torch.zeros(2, 3, dtype=torch.bool), torch.zeros(2, 3, dtype=torch.bool), torch.zeros(2, 3, 4, dtype=torch.bool))
    targets = tuple(torch.zeros_like(outputs[key]) for key in ("pred_action", "pred_force", "pred_force_highrate"))
    return outputs, (*targets, *masks)


def test_dynamic_beta_loss_and_gradient_equivalence_and_zero_is_exact():
    outputs, args = loss_inputs()
    criterion = ACTAlignedHighRateCriterion(ACTAlignedHighRateTrainingConfig())
    legacy = criterion(outputs, *args)
    explicit = criterion(outputs, *args, posterior_kl_weight=10.)
    for key in legacy: torch.testing.assert_close(legacy[key], explicit[key], rtol=0, atol=0)
    old_grads = torch.autograd.grad(legacy["loss_total"], tuple(outputs.values()), retain_graph=True)
    new_grads = torch.autograd.grad(explicit["loss_total"], tuple(outputs.values()), retain_graph=True)
    for left, right in zip(old_grads, new_grads): torch.testing.assert_close(left, right, rtol=0, atol=0)
    low = criterion(outputs, *args, posterior_kl_weight=1.)
    assert torch.equal(low["loss_prior_match"], legacy["loss_prior_match"])
    direct = ACTAlignedHighRateCriterion(replace(criterion.config, posterior_kl_weight=1.))(outputs, *args)
    torch.testing.assert_close(low["loss_total"], direct["loss_total"], rtol=0, atol=0)
    zero = criterion(outputs, *args, posterior_kl_weight=0.)
    grads = torch.autograd.grad(zero["loss_total"], (outputs["mu_contact"], outputs["logvar_contact"], outputs["mu_contact_prior"]))
    assert grads[0].count_nonzero() == grads[1].count_nonzero() == 0  # No residual q KL gradient.
    assert grads[2].abs().sum() > 0  # Detached-q prior matching still learns.
    for bad in (float("nan"), float("inf"), -1., True):
        with pytest.raises(ValueError): criterion(outputs, *args, posterior_kl_weight=bad)


def test_cli_resume_crosses_warmup_exactly_and_rejects_change_before_mutation(tmp_path, monkeypatch, capsys):
    helper = module(Path(__file__).with_name("test_real_robot_multi_metric_cli.py"), "kl_cli_helper")
    runner, args = helper.setup(tmp_path, monkeypatch, "contact", 6)

    def toy_step(kind, model, criterion, optimizer, batch, settings, *, posterior_kl_weight):
        optimizer.zero_grad(set_to_none=True)
        x = torch.randn(1, 1) + random.random() + np.random.random()
        loss = (model(x) - 1).square().sum() + posterior_kl_weight * model.weight.square().sum()
        loss.backward(); optimizer.step()
        return {"loss_total": float(loss.detach()), "posterior_kl_beta": posterior_kl_weight}

    monkeypatch.setattr(runner, "train_step", toy_step)
    monkeypatch.setattr(runner, "evaluate_episodes", lambda *a, **kw: dict(
        episode_macro_action_mae_rad=.1, episode_macro_force_endpoint_mae_n=.2,
        episode_macro_force_mae_n=.3, episode_macro_moment_endpoint_mae_nm=.4, episode_macro_moment_mae_nm=.5))
    def command(output, steps, schedule=True):
        return args(output, steps) + ["--minimum-sampling-epochs", "100", "--contact-latent-mode", "prior_mean"] + (
            ["--posterior-kl-schedule", "linear_warmup", "--posterior-kl-warmup-epochs", "4"] if schedule else [])

    resumed_dir, full_dir = tmp_path / "resumed", tmp_path / "full"
    runner.main(command(resumed_dir, 3))
    runner.main(command(resumed_dir, 6, False) + ["--resume", str(resumed_dir / "last.pt")])
    runner.main(command(full_dir, 6))
    assert (resumed_dir / "metrics.jsonl").read_text() == (full_dir / "metrics.jsonl").read_text()
    rows = list(map(json.loads, (full_dir / "metrics.jsonl").read_text().splitlines()))
    assert [r["train"]["posterior_kl_beta"] for r in rows] == [0., 2.5, 5., 7.5, 10., 10.]
    left, right = read_checkpoint(resumed_dir / "last.pt"), read_checkpoint(full_dir / "last.pt")
    for key in ("model_state", "optimizer_state", "sampler_state", "rng_state", "multi_metric_state", "posterior_kl_schedule_state"):
        if key == "rng_state":
            assert left[key]["python"] == right[key]["python"]
            np.testing.assert_array_equal(left[key]["numpy"][1], right[key]["numpy"][1])
            assert torch.equal(left[key]["torch"], right[key]["torch"])
        else:
            assert sampler_states_equal(left[key], right[key]), key
    assert left["posterior_kl_schedule_state"] == dict(completed_updates=6, last_update_weight=10., next_update_weight=10.)
    assert read_checkpoint(full_dir / "best.pt")["posterior_kl_schedule_state"]["last_update_weight"] == 0.
    assert all("zero_latent_reference" in row for row in rows)
    before = (resumed_dir / "run_manifest.json").read_bytes()
    with pytest.raises(ValueError, match="cannot change"):
        runner.main(command(resumed_dir, 7, False) + ["--resume", str(resumed_dir / "last.pt"), "--posterior-kl-schedule", "constant", "--posterior-kl-warmup-epochs", "0"])
    assert (resumed_dir / "run_manifest.json").read_bytes() == before
    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad(): model.weight.fill_(99.)
    optimizer = torch.optim.AdamW(model.parameters())
    sampler = RealRobotTrainingSampler([2, 2], mode="episodic_uniform", seed=0)
    changed = deepcopy(left)
    changed["run_config"]["posterior_kl_schedule"] = PosteriorKLSchedule().to_dict()
    for damaged, expected in ((left, changed), ({**left, "posterior_kl_schedule_state": {}}, left)):
        with pytest.raises(ValueError, match="posterior_kl_schedule"):
            restore_checkpoint(damaged, model=model, optimizer=optimizer, sampler=sampler, expected=expected)
        assert model.weight.item() == 99. and not optimizer.state
    with pytest.raises(ValueError, match="posterior_kl_schedule"):
        runner.validate_output_identity(full_dir, changed, 6)
    monitor = module(ROOT / "scripts/monitor_real_robot_training.py", "kl_monitor")
    report = monitor.build_report(full_dir, monitor.IncrementalMetrics(full_dir / "metrics.jsonl"), monitor.WallRate(), processes=[], include_gpu=False)
    text = monitor.render_report(report)
    assert "最近更新 β=10.000000" in text and "仅记录，不参与早停" in text and "训练态先验" in text
    capsys.readouterr()


def test_train_diagnostics_preserve_optimizer_and_rng_and_zero_still_reconstructs(tmp_path):
    from dataclasses import asdict
    from force_aware_act.act_aligned_training.high_rate_data import collate_high_rate_samples
    from force_aware_act.real_robot.training import (RealRobotOptimizationConfig, build_model,
        build_optimizer, build_criterion, forward_training, train_step)
    torch.set_num_threads(1)
    fixture = module(Path(__file__).with_name("test_real_robot_wrench6.py"), "kl_sensor_fixture")
    fixture.episode(tmp_path)
    dataset, stats, config = fixture.dataset(tmp_path, fixture.config())
    batch = collate_high_rate_samples([dataset[10], dataset[-1]])
    model = build_model("contact", asdict(config))
    copied = deepcopy(model)
    settings = RealRobotOptimizationConfig()
    optimizer, copied_optimizer = build_optimizer(model, settings), build_optimizer(copied, settings)
    criterion = build_criterion("contact", settings)
    rng = torch.get_rng_state()
    model.train(); optimizer.zero_grad(set_to_none=True)
    _, losses = forward_training("contact", model, criterion, batch)
    losses["loss_total"].backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), settings.gradient_clip_norm, error_if_nonfinite=True)
    optimizer.step()
    reference_rng = torch.get_rng_state()
    torch.set_rng_state(rng)
    metrics = train_step("contact", copied, criterion, copied_optimizer, batch, settings, posterior_kl_weight=10.)
    assert torch.equal(reference_rng, torch.get_rng_state())
    assert sampler_states_equal(model.state_dict(), copied.state_dict())
    assert sampler_states_equal(optimizer.state_dict(), copied_optimizer.state_dict())
    assert metrics["loss_posterior_kl_weighted"] == 10 * metrics["loss_posterior_kl"]
    assert metrics["loss_total"] == pytest.approx(metrics["loss_reconstruction"] + metrics["loss_posterior_kl_weighted"] + metrics["loss_prior_match"])
    assert np.isfinite(list(metrics.values())).all()
    for label in ("prior", "posterior"):
        assert metrics[f"train_{label}_mu_rms"] >= metrics[f"train_{label}_mu_centered_rms"]
        assert metrics[f"train_{label}_sigma_mean"] > 0
    model.zero_grad(set_to_none=True)
    _, losses = forward_training("contact", model, criterion, batch, posterior_kl_weight=0.)
    losses["loss_total"].backward()
    # Removing only standard-normal KL does not disconnect posterior reconstruction.
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.contact_posterior.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.contact_prior.parameters())
    dataset.close()


def test_launch_profiles_preserve_other_settings_and_isolate_results(tmp_path):
    script = ROOT / "scripts/training/train_x14y08_contact_prior_kl.sh"
    subprocess.run(["bash", "-n", str(script)], check=True)
    stub = tmp_path / "bash"
    stub.write_text("#!/usr/bin/env python3\nimport json,sys,os\nprint(json.dumps([sys.argv[1:],os.environ['FORCEACT_OUTPUT_ROOT']]))\n")
    stub.chmod(0o755)
    env = {**os.environ, "PATH": str(tmp_path) + os.pathsep + os.environ["PATH"], "FORCEACT_OUTPUT_ROOT": str(tmp_path / "out")}
    for profile, kind, epochs in (("beta1", "constant", "0"), ("warmup1000", "linear_warmup", "1000")):
        argv, output = json.loads(subprocess.check_output(["/bin/bash", str(script), profile, "test"], env=env, text=True))
        assert argv[1:4] == ["wrench6", "10", "test"]
        assert output.endswith("/" + profile)
        assert argv[-4:] == ["--posterior-kl-schedule", kind, "--posterior-kl-warmup-epochs", epochs]
        actual = json.loads(Path(argv[argv.index("--optimization-config") + 1]).read_text())
        original = json.loads((ROOT / "configs/experiments/1mm_x14y08_55_wrench6_v1/optimization.json").read_text())
        assert actual.pop("posterior_kl_weight") == (1. if profile == "beta1" else 10.)
        original.pop("posterior_kl_weight")
        assert actual == original


def test_formal_launcher_expands_exact_agreed_budget_without_starting_training(tmp_path):
    # Run every shell layer, substituting ONLY conda activation and the final Python.
    conda = tmp_path / "conda.sh"
    conda.write_text("conda() { :; }\n")
    python = tmp_path / "python"
    python.write_text("#!/usr/bin/env python3\nimport json,sys\nprint(json.dumps(sys.argv[1:]))\n")
    python.chmod(0o755)
    env = {**os.environ, "PATH": str(tmp_path) + os.pathsep + os.environ["PATH"],
           "FORCEACT_CONDA_SH": str(conda), "FORCEACT_OUTPUT_ROOT": str(tmp_path / "out")}
    for profile in ("beta1", "warmup1000"):
        command = json.loads(subprocess.check_output(["/bin/bash", str(ROOT / "scripts/training/train_x14y08_contact_prior_kl.sh"),
                             profile, "train"], env=env, text=True))
        def argument(flag):
            i = max(i for i, value in enumerate(command) if value == flag)
            return command[i+1]
        expected = {"--model": "contact", "--mode": "train", "--batch-size": "8", "--seed": "0",
                    "--chunk-len": "20", "--force-prediction-steps": "20", "--contact-latent-mode": "prior_mean",
                    "--max-sampling-epochs": "20000", "--minimum-sampling-epochs": "3000",
                    "--early-stop-patience": "20", "--early-stop-min-relative-improvement": "0.001",
                    "--early-stop-metrics": "multi", "--validation-interval": "250", "--checkpoint-interval": "250",
                    "--validation-anchor-stride": "10", "--train-sampling-mode": "episodic_uniform"}
        assert {flag: argument(flag) for flag in expected} == expected
        assert argument("--data-config") == "configs/real_robot/right_arm_wrench6.json"
        assert "/" + profile + "/runs/" in argument("--output-dir")
        assert "--strict-force-horizon" in command
