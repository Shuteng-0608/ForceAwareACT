"""Prior-mean inference, unchanged posterior gradients, and causal sensor inputs."""
from copy import deepcopy
from dataclasses import asdict, replace
import importlib.util
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest
import torch

from force_aware_act.act_aligned_training.high_rate_data import collate_high_rate_samples
from force_aware_act.inference.real_robot import RealRobotPolicyAdapter, RealRobotObservation
from force_aware_act.real_robot.contact_latent_mode import (
    contact_latent_contract, resolve_contact_latent_mode, validate_contact_latent_identity,
)
from force_aware_act.real_robot.evaluation import evaluate_episodes
from force_aware_act.real_robot.training import (
    build_model, build_criterion, forward_training, deployment_predictions, RealRobotOptimizationConfig,
)

ROOT = Path(__file__).parents[1]


def helper(name):
    spec = importlib.util.spec_from_file_location("prior_" + name, Path(__file__).with_name(name + ".py"))
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_mode_contract_legacy_defaults_and_resume_rejection():
    prior = {"contact_latent_config": contact_latent_contract("prior_mean")}
    assert resolve_contact_latent_mode("contact") == "zero"
    assert resolve_contact_latent_mode("contact", run_config=prior, resume=True) == "prior_mean"
    assert resolve_contact_latent_mode("contact", "zero", prior) == "zero"  # offline comparison
    validate_contact_latent_identity({}, {"contact_latent_config": contact_latent_contract("zero")})
    for kind in ("act", "motion", "dual-zero"):
        with pytest.raises(ValueError, match="only supported"):
            resolve_contact_latent_mode(kind, "prior_mean")
    for saved, requested in ((prior, "zero"), ({}, "prior_mean")):
        with pytest.raises(ValueError, match="cannot change"):
            resolve_contact_latent_mode("contact", requested, saved, resume=True)
    damaged = deepcopy(prior)
    damaged["contact_latent_config"]["training"] = "prior_sample"
    with pytest.raises(ValueError, match="invalid"):
        resolve_contact_latent_mode("contact", run_config=damaged)


@pytest.mark.parametrize("dimension", [3, 6])
@pytest.mark.parametrize("history_ms", [10, 50])
def test_prior_is_deterministic_causal_and_training_gradients_stay_isolated(tmp_path, dimension, history_ms):
    torch.set_num_threads(1)
    torch.manual_seed(17)
    fixture = helper("test_real_robot_wrench6")
    fixture.episode(tmp_path)
    cfg = fixture.config()
    if dimension == 3:
        cfg = replace(cfg, force_representation="end_effector_force3", wrench_component_units=("N",)*3,
                      force_std_floor=(.01,)*3)
    if history_ms == 50:
        cfg = replace(cfg, history_interval_ms=50, history_max_samples_per_interval=20)
    ds, stats, config = fixture.dataset(tmp_path, cfg)
    batch = collate_high_rate_samples([ds[10], ds[-1]])
    model = build_model("contact", asdict(config))
    outputs, losses = forward_training("contact", model,
        build_criterion("contact", RealRobotOptimizationConfig()), batch)
    assert outputs["contact_latent_source"] == "posterior"
    torch.testing.assert_close(outputs["z_contact"], outputs["z_contact_posterior"], rtol=0, atol=0)
    losses["loss_prior_match"].backward(retain_graph=True)
    for name, parameter in model.named_parameters():
        if not name.startswith("contact_prior."):
            assert parameter.grad is None, name
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.contact_prior.parameters())
    model.zero_grad(set_to_none=True)
    losses["loss_total"].backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.contact_posterior.parameters())
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    model.eval()
    rng = torch.get_rng_state().clone()
    predicted = deployment_predictions("contact", model, batch, contact_latent_mode="prior_mean")
    assert torch.equal(rng, torch.get_rng_state())
    assert predicted["contact_latent_source"] == "prior_mean"
    torch.testing.assert_close(predicted["z_contact"], predicted["mu_contact_prior"], rtol=0, atol=0)
    corrupted = replace(batch, action_chunk=torch.full_like(batch.action_chunk, float("nan")),
                        future_force_intervals=torch.full_like(batch.future_force_intervals, float("nan")),
                        future_force_target=torch.full_like(batch.future_force_target, float("nan")))
    again = deployment_predictions("contact", model, corrupted, contact_latent_mode="prior_mean")
    for key in ("pred_action", "pred_force", "pred_force_highrate"):
        torch.testing.assert_close(predicted[key], again[key], rtol=0, atol=0)
    zero = deployment_predictions("contact", model, batch)
    assert zero["z_contact"].count_nonzero() == 0
    assert not torch.equal(predicted["pred_action"], zero["pred_action"])
    metrics = evaluate_episodes("contact", model, ds, {ds.episodes[0].episode_id: [10, len(ds)-1]},
                               stats, "cpu", 2, contact_latent_mode="prior_mean")
    assert np.isfinite(metrics["episode_macro_force_mae_n"])
    assert ("episode_macro_moment_mae_nm" in metrics) == (dimension == 6)

    # Reload the saved selection mode into the causal adapter, without future labels.
    payload = dict(format_version="real_robot_training_v1", model_kind="contact", model_config=asdict(config),
                   model_state=model.state_dict(), data_config=cfg.to_dict(), data_contract=cfg.contract_dict(),
                   normalization=stats.to_dict(), run_config={"contact_latent_config": contact_latent_contract("prior_mean")})
    adapter = RealRobotPolicyAdapter.from_checkpoint(payload)
    assert adapter.default_contact_latent_mode == "prior"
    anchor = ds.sample_trace(10)["observation"]["anchor_ns"]
    raw = ds.raw_episodes[0]
    state, _ = raw.observation_indices(anchor)
    observation = RealRobotObservation(anchor_ns=anchor, images=tuple(raw.images_at(anchor)),
        image_timestamps_ns=(anchor, anchor), qpos=raw.qpos[state], state_timestamp_ns=anchor,
        torque_timestamps_ns=raw.torque_timestamps_ns, torque_values=raw.force)
    prepared = adapter.prepare_snapshot(observation)
    actual = adapter.predict_prepared(prepared)
    explicit = adapter.predict_prepared(prepared, contact_latent_mode="prior_mean")
    for key in actual:
        np.testing.assert_array_equal(actual[key], explicit[key])
    single = collate_high_rate_samples([ds[10]])
    expected = deployment_predictions("contact", adapter.model, single, contact_latent_mode="prior_mean")
    np.testing.assert_allclose(actual["pred_action"], stats.denormalize_action(expected["pred_action"])[0].numpy(), atol=1e-6)
    payload.pop("run_config")
    assert RealRobotPolicyAdapter.from_checkpoint(payload).default_contact_latent_mode == "zero"


def test_launcher_delegates_unchanged_profiles_and_uses_independent_outputs(tmp_path):
    script = ROOT / "scripts/training/train_x14y08_contact_prior.sh"
    subprocess.run(["bash", "-n", str(script)], check=True)
    stub = tmp_path / "bash"
    stub.write_text("#!/usr/bin/env python3\nimport json,sys\nprint(json.dumps(sys.argv[1:]))\n")
    stub.chmod(0o755)
    env = {**os.environ, "PATH": str(tmp_path) + os.pathsep + os.environ["PATH"],
           "FORCEACT_OUTPUT_ROOT": str(tmp_path / "outputs")}
    for representation in ("wrench6", "sensor_force3"):
        for history in ("10", "50"):
            args = json.loads(subprocess.check_output(["/bin/bash", str(script), representation, history, "train"],
                                                     env=env, text=True))
            assert args[-2:] == ["--contact-latent-mode", "prior_mean"]
            assert "prior_mean" in args[args.index("--output-dir")+1]
            assert args[0].endswith("history50ms.sh" if history == "50" else f"{representation}.sh")
            assert "motion" not in args
