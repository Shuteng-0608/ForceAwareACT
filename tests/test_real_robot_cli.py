"""End-to-end CLI contracts with independent synthetic raw recordings."""

import importlib.util
import json
from pathlib import Path

import h5py
import pytest
import torch

from force_aware_act.real_robot.checkpoint import read_checkpoint, sampler_states_equal
from force_aware_act.real_robot.training_sampling import EPISODIC_UNIFORM_VERSION, UNIFORM_VALID_ANCHOR_SAMPLING


def _raw_fixture_helpers():
    # The suite uses pytest's importlib mode, so sibling tests are not top-level
    # importable modules. Reuse the existing fixture without changing sys.path.
    spec = importlib.util.spec_from_file_location(
        "real_robot_cli_raw_fixtures", Path(__file__).with_name("test_real_robot_data.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._config, module._write_raw


def _runner():
    path = Path(__file__).parents[1] / "scripts" / "train_real_robot.py"
    spec = importlib.util.spec_from_file_location("real_robot_cli_integration", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("budget", [[], ["--steps", "301"], ["--max-sampling-epochs", "10"]])
def test_short_test_mode_requires_bounded_update_budget(tmp_path, budget):
    with pytest.raises(ValueError, match="test mode"):
        _runner().main(["--data-root", str(tmp_path), "--output-dir", str(tmp_path / "out"),
                        "--mode", "test", *budget])


def test_short_test_mode_uses_held_out_data_and_records_its_role(tmp_path, capsys):
    root, config = _data(tmp_path)
    output = tmp_path / "check"
    args = _args(root, config, output, 1)
    args[args.index("--mode") + 1] = "test"
    result = _runner().main(args)
    assert result["completed_steps"] == 1
    assert result["checkpoint_round_trip"] == "passed"
    manifest = json.loads((output / "run_manifest.json").read_text())
    assert manifest["run_config"]["mode"] == "test"
    assert manifest["split_manifest"]["validation_episode_ids"]


def _data(tmp_path, count=2, *, unequal_anchors=False):
    _config, _write_raw = _raw_fixture_helpers()
    root = tmp_path / "data"
    for index in range(count):
        missing_actions = tuple(range(50, 60 + index * 40, 10)) if unequal_anchors else (50,)
        path = _write_raw(root, f"recording_{index}", offset=index * 3.0,
                          missing_action_ms=missing_actions, missing_state_ms=(50,))
        with h5py.File(path, "r+") as handle:
            handle.attrs["source_sha256"] = f"independent_recording_{index}"
    config_path = tmp_path / "data_config.json"
    config_path.write_text(json.dumps(_config().to_dict()))
    return root, config_path


def _args(root, data_config, output, steps, *, model="contact", batch_size=1, sampling_mode=None):
    result = [
        "--data-root", str(root), "--data-config", str(data_config),
        "--output-dir", str(output), "--mode", "train", "--model", model,
        "--compact", "--device", "cpu", "--threads", "2", "--seed", "17",
        "--chunk-len", "3", "--batch-size", str(batch_size), "--steps", str(steps),
        "--validation-interval", "1", "--checkpoint-interval", "1",
    ]
    if sampling_mode is not None:
        result.extend(["--train-sampling-mode", sampling_mode])
    return result


def _assert_identical(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_identical(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert type(left) is type(right) and len(left) == len(right)
        for lhs, rhs in zip(left, right):
            _assert_identical(lhs, rhs)
    else:
        assert left == right


def test_formal_cli_resume_matches_continuous_training_and_preserves_best(tmp_path, capsys):
    runner = _runner()
    root, config = _data(tmp_path)
    resumed_output = tmp_path / "resumed"
    first = runner.main(_args(root, config, resumed_output, 1))
    assert first["completed_steps"] == 1
    assert first["validation_anchors"] > 0
    assert first["checkpoint_round_trip"] == "passed"
    assert (resumed_output / "best.pt").is_file()
    assert (resumed_output / "last.pt").is_file()
    manifest = json.loads((resumed_output / "run_manifest.json").read_text())
    split = manifest["split_manifest"]
    groups = split["source_groups"]
    assert {groups[key] for key in split["train_episode_ids"]}.isdisjoint(
        groups[key] for key in split["validation_episode_ids"])
    assert manifest["model_config"]["image_height"] == 2
    assert manifest["model_config"]["image_width"] == 3
    assert manifest["model_config"]["force_dim"] == 7
    assert manifest["model_config"]["policy_sample_rate_hz"] == 100
    assert manifest["run_config"]["synthetic_terminal_hold"] is False

    resumed = runner.main(_args(root, config, resumed_output, 2)
                          + ["--resume", str(resumed_output / "last.pt")])
    assert resumed["completed_steps"] == 2
    assert resumed["checkpoint_round_trip"] == "passed"
    resumed_rows = [json.loads(line) for line in (resumed_output / "metrics.jsonl").read_text().splitlines()]
    assert [row["step"] for row in resumed_rows] == [1, 2]
    assert all("held_out_validation" in row and "same_recording_smoke_check" not in row
               for row in resumed_rows)
    best = read_checkpoint(resumed_output / "best.pt")
    scores = [row["held_out_validation"]["action_mae_rad"] for row in resumed_rows]
    assert best["best_metric"] == min(scores)
    assert best["global_step"] == scores.index(min(scores)) + 1
    del best

    continuous_output = tmp_path / "continuous"
    continuous = runner.main(_args(root, config, continuous_output, 2))
    assert continuous["completed_steps"] == 2
    continuous_rows = [json.loads(line) for line in (continuous_output / "metrics.jsonl").read_text().splitlines()]
    assert [row["sample_indices"] for row in continuous_rows] == [row["sample_indices"] for row in resumed_rows]
    assert continuous_rows == resumed_rows
    resumed_state = read_checkpoint(resumed_output / "last.pt")
    continuous_state = read_checkpoint(continuous_output / "last.pt")
    _assert_identical(resumed_state["model_state"], continuous_state["model_state"])
    _assert_identical(resumed_state["optimizer_state"], continuous_state["optimizer_state"])
    torch.testing.assert_close(resumed_state["sampler_state"], continuous_state["sampler_state"], rtol=0, atol=0)
    capsys.readouterr()


def test_formal_cli_refuses_a_single_recording_without_training(tmp_path, capsys):
    root, config = _data(tmp_path, count=1)
    output = tmp_path / "rejected"
    with pytest.raises(ValueError, match="independent recordings"):
        _runner().main(_args(root, config, output, 1))
    assert not (output / "last.pt").exists()
    assert not (output / "run_manifest.json").exists()
    capsys.readouterr()


@pytest.mark.parametrize("model", ["contact", "motion", "dual-zero", "act"])
def test_episodic_cli_balances_valid_windows_and_resumes_mid_epoch_exactly(tmp_path, capsys, model):
    runner = _runner()
    root, config = _data(tmp_path, count=4, unequal_anchors=True)
    resumed_output = tmp_path / "resumed"
    first = runner.main(_args(root, config, resumed_output, 1, model=model, batch_size=2,
                              sampling_mode="episodic_uniform"))
    assert first["completed_sampling_epochs"] == 0
    manifest = json.loads((resumed_output / "run_manifest.json").read_text())
    run_config = manifest["run_config"]
    assert run_config["train_sampling_mode"] == "episodic_uniform"
    assert run_config["sampling"] == EPISODIC_UNIFORM_VERSION
    assert run_config["sampling_epoch_windows"] == 3
    assert run_config["steps_per_sampling_epoch"] == 2
    counts = run_config["train_episode_valid_anchors"]
    assert len(set(counts.values())) > 1
    saved = read_checkpoint(resumed_output / "last.pt")
    assert saved["sampler_state"]["cursor"] == 2
    assert saved["sampler_state"]["epoch"] == 1
    del saved

    # Omitting the mode on resume must inherit episodic sampling, not the
    # legacy new-run default. Resume starts with the remaining one-item batch.
    resumed = runner.main(_args(root, config, resumed_output, 4, model=model, batch_size=2)
                          + ["--resume", str(resumed_output / "last.pt")])
    assert resumed["completed_sampling_epochs"] == 2
    assert resumed["checkpoint_round_trip"] == "passed"
    resumed_rows = [json.loads(line) for line in (resumed_output / "metrics.jsonl").read_text().splitlines()]
    assert [row["batch_size_actual"] for row in resumed_rows] == [2, 1, 2, 1]
    assert [row["sampling_epoch"] for row in resumed_rows] == [1, 1, 2, 2]
    assert [row["sampling_epoch_complete"] for row in resumed_rows] == [False, True, False, True]
    train_ids = manifest["split_manifest"]["train_episode_ids"]
    for start in (0, 2):
        epoch_ids = [episode for row in resumed_rows[start:start + 2] for episode in row["sample_episode_ids"]]
        assert sorted(epoch_ids) == sorted(train_ids)
    assert all(len(row["sample_anchor_grid_indices"]) == row["batch_size_actual"] for row in resumed_rows)

    continuous_output = tmp_path / "continuous"
    runner.main(_args(root, config, continuous_output, 4, model=model, batch_size=2,
                      sampling_mode="episodic_uniform"))
    continuous_rows = [json.loads(line) for line in (continuous_output / "metrics.jsonl").read_text().splitlines()]
    assert continuous_rows == resumed_rows
    resumed_state = read_checkpoint(resumed_output / "last.pt")
    continuous_state = read_checkpoint(continuous_output / "last.pt")
    _assert_identical(resumed_state["model_state"], continuous_state["model_state"])
    _assert_identical(resumed_state["optimizer_state"], continuous_state["optimizer_state"])
    assert sampler_states_equal(resumed_state["sampler_state"], continuous_state["sampler_state"])

    if model == "contact":
        # Switching modes may not mix two sampling contracts in one run.
        with pytest.raises(ValueError, match="sampling"):
            runner.main(_args(root, config, resumed_output, 5, model=model, batch_size=2,
                              sampling_mode=UNIFORM_VALID_ANCHOR_SAMPLING)
                        + ["--resume", str(resumed_output / "last.pt")])
    capsys.readouterr()


def test_three_way_split_fixed_macro_validation_early_stopping_and_test_evaluation(tmp_path, capsys):
    runner = _runner()
    root, config = _data(tmp_path, count=4, unequal_anchors=True)
    split_path = tmp_path / "split.json"
    split_path.write_text(json.dumps({"seed": 20260928, "train_episode_ids": ["recording_0", "recording_1"],
                                     "validation_episode_ids": ["recording_2"], "test_episode_ids": ["recording_3"]}))
    output = tmp_path / "train"
    args = _args(root, config, output, 5, batch_size=2, sampling_mode="episodic_uniform")
    args.extend(["--split", str(split_path), "--minimum-sampling-epochs", "1", "--early-stop-patience", "1",
                 "--early-stop-min-relative-improvement", ".99", "--validation-anchor-stride", "10",
                 "--selection-metric", "episode_macro_action_mae_rad"])
    result = runner.main(args)
    assert result["termination"] == "early_stopping"
    assert result["completed_steps"] == 2
    checkpoint = read_checkpoint(output / "last.pt")
    assert checkpoint["split_manifest"]["test_episode_ids"] == ["recording_3"]
    assert checkpoint["split_manifest"]["seed"] == 20260928
    assert checkpoint["early_stopping_state"]["bad_checks"] == 1
    assert result["train_episodes"] == 2
    indices = json.loads((output / "validation_indices.json").read_text())
    assert {row["episode_id"] for row in indices["entries"]} == {"recording_2"}
    last = result["final_metrics"]["held_out_validation"]
    assert last["episode_macro_action_mae_rad"] == last["action_mae_rad"]

    evaluation_path = Path(__file__).parents[1] / "scripts/evaluate_real_robot.py"
    spec = importlib.util.spec_from_file_location("real_robot_evaluation_test", evaluation_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    report = module.main(["--checkpoint", str(output / "best.pt"), "--data-root", str(root),
                          "--output", str(tmp_path / "test_report.json"), "--device", "cpu", "--threads", "2", "--stride", "10"])
    assert set(report["metrics"]["per_episode"]) == {"recording_3"}
    assert report["latent_mode"] == "zero"
    assert len(report["metrics"]["per_episode"]["recording_3"]["action_mae_per_joint_rad"]) == 7
    assert report["metrics"]["episode_macro_hold_current_qpos_mae_rad"] >= 0
    capsys.readouterr()


def test_epoch_budget_converts_to_steps_and_test_source_leakage_is_rejected(tmp_path, capsys):
    runner = _runner()
    root, config = _data(tmp_path, count=4)
    args = _args(root, config, tmp_path / "epochs", 4, batch_size=2, sampling_mode="episodic_uniform")
    position = args.index("--steps")
    args[position:position + 2] = ["--max-sampling-epochs", "2"]
    result = runner.main(args)
    assert result["completed_steps"] == 4
    assert result["completed_sampling_epochs"] == 2
    with h5py.File(root / "recording_3" / "episode.hdf5", "r+") as handle:
        handle.attrs["source_sha256"] = "independent_recording_0"
    split = tmp_path / "split.json"
    split.write_text(json.dumps({"train_episode_ids": ["recording_0", "recording_1"],
                                 "validation_episode_ids": ["recording_2"], "test_episode_ids": ["recording_3"]}))
    with pytest.raises(ValueError, match="test set"):
        runner.main(_args(root, config, tmp_path / "rejected_test", 1) + ["--split", str(split)])
    capsys.readouterr()


def test_early_stopping_resume_keeps_counters_and_matches_continuous_training(tmp_path, capsys):
    runner = _runner()
    root, config = _data(tmp_path, count=3)
    extra = ["--minimum-sampling-epochs", "0", "--early-stop-patience", "3",
             "--early-stop-min-relative-improvement", ".99", "--validation-anchor-stride", "10",
             "--selection-metric", "episode_macro_action_mae_rad"]
    resumed_dir = tmp_path / "resumed_stopping"
    first = runner.main(_args(root, config, resumed_dir, 2, batch_size=2, sampling_mode="episodic_uniform") + extra)
    assert first["termination"] == "step_budget"
    assert first["final_metrics"]["early_stopping"]["bad_checks"] == 1
    resumed = runner.main(_args(root, config, resumed_dir, 5, batch_size=2) + extra
                          + ["--resume", str(resumed_dir / "last.pt")])
    continuous_dir = tmp_path / "continuous_stopping"
    continuous = runner.main(_args(root, config, continuous_dir, 5, batch_size=2, sampling_mode="episodic_uniform") + extra)
    assert resumed["completed_steps"] == continuous["completed_steps"] == 4
    assert resumed["termination"] == continuous["termination"] == "early_stopping"
    assert (resumed_dir / "metrics.jsonl").read_text() == (continuous_dir / "metrics.jsonl").read_text()
    left = read_checkpoint(resumed_dir / "last.pt")
    right = read_checkpoint(continuous_dir / "last.pt")
    _assert_identical(left["model_state"], right["model_state"])
    assert left["early_stopping_state"] == right["early_stopping_state"]
    capsys.readouterr()
