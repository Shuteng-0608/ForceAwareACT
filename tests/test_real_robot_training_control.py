import copy
import importlib.util
import json
from pathlib import Path

import pytest

from force_aware_act.real_robot.training_control import EarlyStoppingConfig, EarlyStoppingController
from force_aware_act.real_robot.evaluation import aggregate_episode_metrics


def test_warmup_significant_improvement_and_exact_early_stopping_resume():
    config = EarlyStoppingConfig(minimum_epochs=2, patience=2, minimum_relative_improvement=.005)
    control = EarlyStoppingController(config)
    assert not control.update(1.0, completed_epochs=1, step=1)
    assert not control.update(1.1, completed_epochs=2, step=2)
    assert control.bad_checks == 0
    assert not control.update(.998, completed_epochs=3, step=3)
    state = control.state_dict()
    restored = EarlyStoppingController(config)
    restored.load_state_dict(state)
    for instance in (control, restored):
        assert not instance.update(.99, completed_epochs=4, step=4)
        assert instance.bad_checks == 0
        assert not instance.update(.989, completed_epochs=5, step=5)
        assert instance.update(.988, completed_epochs=6, step=6)
    assert control.state_dict() == restored.state_dict()


def test_disabled_stopping_and_invalid_restore():
    control = EarlyStoppingController(EarlyStoppingConfig(0, 0))
    for step in range(1, 5):
        assert not control.update(1.0, completed_epochs=step, step=step)
    with pytest.raises(ValueError, match="steps"):
        control.update(1.0, completed_epochs=5, step=4)
    with pytest.raises(ValueError, match="finite"):
        control.update(float("nan"), completed_epochs=5, step=5)
    changed = copy.deepcopy(control.state_dict())
    changed["config"]["patience"] = 2
    with pytest.raises(ValueError, match="configuration"):
        control.load_state_dict(changed)


def test_macro_metric_gives_unequal_episodes_equal_weight():
    rows = {}
    for name, error, count in (("short", 1.0, 7), ("long", 3.0, 63)):
        rows[name] = {"windows": count // 7, "action_mae_rad": error, "action_valid_scalars": count,
                      "torque_mae_nm": None, "torque_valid_scalars": 0,
                      "torque_endpoint_mae_nm": None, "torque_endpoint_valid_scalars": 0}
    report = aggregate_episode_metrics(rows)
    assert report["episode_macro_action_mae_rad"] == 2.0
    assert report["action_mae_rad"] == pytest.approx(2.8)
    assert report["episode_macro_torque_mae_nm"] is None


def test_dataset_preparation_split_is_balanced_and_independent_of_row_order():
    path = Path(__file__).parents[1] / "scripts/prepare_1mm_gap_dataset.py"
    spec = importlib.util.spec_from_file_location("prepare_real_robot_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rows = [{"dataset_name": f"{group}_{index}", "group": group} for group in ("a", "b", "c", "d") for index in range(5)]
    split = module.balanced_split(rows)
    assert split == module.balanced_split(list(reversed(rows)))
    for role, count in (("train", 3), ("validation", 1), ("test", 1)):
        ids = split[f"{role}_episode_ids"]
        assert all(sum(name.startswith(group + "_") for name in ids) == count for group in ("a", "b", "c", "d"))
    assert len(set(split["train_episode_ids"] + split["validation_episode_ids"] + split["test_episode_ids"])) == 20


def test_queue_excludes_dual_zero_and_preserves_epoch_budget():
    path = Path(__file__).parents[1] / "scripts/run_real_robot_experiment_queue.py"
    spec = importlib.util.spec_from_file_location("real_robot_queue_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = {"models": ["contact", "act", "motion"], "train_episodes": 12, "batch_size": 2,
              "data_root": "/tmp/data", "data_config": "/tmp/data.json", "optimization_config": "/tmp/opt.json",
              "split": "/tmp/split.json", "max_sampling_epochs": 20000, "minimum_sampling_epochs": 500,
              "early_stop_patience": 8, "minimum_relative_improvement": .005,
              "validation_interval_epochs": 50, "validation_anchor_stride": 10}
    jobs = module.build_jobs(config)
    assert [(job["model"], job["seed"]) for job in jobs] == [
        ("contact", 0), ("act", 0), ("motion", 0), ("contact", 1), ("act", 1), ("contact", 2), ("act", 2)]
    command = module.training_command(config, jobs[0], Path("/tmp/out"))
    assert command[command.index("--max-sampling-epochs") + 1] == "20000"
    assert command[command.index("--validation-interval") + 1] == "300"
    assert "--compact" not in command and "--no-pretrained" not in command


def test_prepared_queue_cannot_start_before_the_requested_test_report(tmp_path):
    path = Path(__file__).parents[1] / "scripts/run_real_robot_experiment_queue.py"
    spec = importlib.util.spec_from_file_location("real_robot_queue_guard_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = tmp_path / "queue.json"
    config.write_text(json.dumps({"execution_status": "prepared_not_started_per_user_request"}))
    with pytest.raises(ValueError, match="not authorized"):
        module.main(["--config", str(config)])
