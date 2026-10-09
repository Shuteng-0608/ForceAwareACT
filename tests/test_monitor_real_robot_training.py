"""Read-only monitor tests, including concurrent writes and resumed run status."""

import importlib.util
import json
from pathlib import Path

import pytest


def _module():
    path = Path(__file__).parents[1] / "scripts/monitor_real_robot_training.py"
    spec = importlib.util.spec_from_file_location("monitor_real_robot_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _row(step, metric=None, **values):
    result = {"step": step, "train": {"loss_action": 1 / step, "loss_total": 2 / step},
              "sampling_epoch": (step + 5) // 6, "sampling_epoch_complete": step % 6 == 0,
              "batch_size_actual": 2, "data_seconds": .02, "train_seconds": .08, **values}
    if metric is not None:
        result.update(held_out_validation={"episode_macro_action_mae_rad": metric},
                      early_stopping={"bad_checks": 2, "reference_metric": .08})
    return result


def _fixture(tmp_path, module):
    run = tmp_path / "run"
    run.mkdir()
    manifest = {"model_kind": "contact", "run_config": {"steps": 120000, "max_sampling_epochs": 20000,
                "steps_per_sampling_epoch": 6, "train_sampling_mode": "episodic_uniform",
                "selection_metric": "held_out_validation.episode_macro_action_mae_rad",
                "early_stopping_config": {"minimum_epochs": 500, "patience": 8,
                                          "minimum_relative_improvement": .005}},
                "data_contract": {"force_source_topic": "/bridge/estimated_joint_disturbance"}}
    (run / "run_manifest.json").write_text(json.dumps(manifest))
    (run / "metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in
        (_row(6, .2), _row(12, .1), _row(18, .12))))
    return run, module.IncrementalMetrics(run / "metrics.jsonl", recent_window=2)


def _process(run, *, resume=False, pid=100, parent_pid=1):
    argv = ["/env/bin/python", "scripts/train_real_robot.py", "--output-dir", str(run), "--validation-interval", "300"]
    if resume:
        argv += ["--resume", str(run / "last.pt")]
    return {"pid": pid, "parent_pid": parent_pid, "elapsed_s": 100., "output_dir": str(run.resolve()), "argv": argv}


def test_incremental_reader_keeps_partial_line_and_historical_best(tmp_path):
    m = _module()
    run, reader = _fixture(tmp_path, m)
    reader.poll("episode_macro_action_mae_rad")
    assert reader.rows == 3 and len(reader.recent) == 2
    assert reader.best == {"step": 12, "value": .1}
    offset = reader.offset
    reader.poll("episode_macro_action_mae_rad")
    assert reader.rows == 3 and reader.offset == offset
    line = json.dumps(_row(24, .09))
    with reader.path.open("a") as stream:
        stream.write(line[:20])
    reader.poll("episode_macro_action_mae_rad")
    assert reader.rows == 3 and reader.pending
    with reader.path.open("a") as stream:
        stream.write(line[20:] + "\ninvalid json\n")
    reader.poll("episode_macro_action_mae_rad")
    assert reader.rows == 4 and reader.invalid_lines == 1
    assert reader.best == {"step": 24, "value": .09}


def test_reader_resets_for_truncation_and_atomic_replacement(tmp_path):
    m = _module()
    run, reader = _fixture(tmp_path, m)
    reader.poll("episode_macro_action_mae_rad")
    reader.path.write_text(json.dumps(_row(1)) + "\n")
    reader.poll("episode_macro_action_mae_rad")
    assert reader.rows == 1 and reader.latest["step"] == 1 and reader.best is None
    replacement = run / "replacement.jsonl"
    replacement.write_text(json.dumps(_row(2)) + "\n")
    replacement.replace(reader.path)
    reader.poll("episode_macro_action_mae_rad")
    assert reader.rows == 1 and reader.latest["step"] == 2


def test_monitor_counts_epochs_best_early_stop_and_does_not_change_artifacts(tmp_path):
    m = _module()
    run, reader = _fixture(tmp_path, m)
    before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in run.iterdir()}
    report = m.build_report(run, reader, m.WallRate(), processes=[_process(run)], include_gpu=False)
    assert report["status"] == "running" and report["pid"] == 100
    assert report["completed_epochs"] == 3 and report["epoch_budget"] == 20000
    assert report["best_validation"] == {"step": 12, "value": .1}
    assert report["validation_value"] == .12
    assert report["early_stopping_state"]["bad_checks"] == 2
    assert report["mean_losses"]["loss_action"] == pytest.approx((1/12 + 1/18)/2)
    assert report["mean_read_train_seconds"] == pytest.approx(.1)
    assert report["next_validation_step"] == 300
    rendered = m.render_report(report)
    assert "完整采样epoch" in rendered and "早停可能提前结束" in rendered and "预热阶段" in rendered
    assert before == {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in run.iterdir()}


def test_force3_monitor_labels_are_force_not_joint_torque(tmp_path):
    m = _module()
    run, reader = _fixture(tmp_path, m)
    path = run / "run_manifest.json"
    manifest = json.loads(path.read_text())
    manifest["data_config"] = {"force_representation": "flange_force3"}
    path.write_text(json.dumps(manifest))
    report = m.build_report(run, reader, m.WallRate(), processes=[], include_gpu=False)
    rendered = m.render_report(report)
    assert report["force_representation"] == "flange_force3"
    assert "三维力端点L1" in rendered and "高频三维力L1" in rendered
    assert "物理指标单位 N" in rendered


def test_old_summary_is_ignored_during_resume_and_after_later_steps(tmp_path):
    m = _module()
    run, reader = _fixture(tmp_path, m)
    (run / "summary.json").write_text(json.dumps({"completed_steps": 18, "step_budget": 120000,
                                                 "termination": "early_stopping"}))
    assert m.build_report(run, reader, m.WallRate(), processes=[], include_gpu=False)["status"] == "early_stopped"
    active = m.build_report(run, reader, m.WallRate(), processes=[_process(run, resume=True)], include_gpu=False)
    assert active["status"] == "running" and active["estimated_remaining_to_budget_s"] is None
    with reader.path.open("a") as stream:
        stream.write(json.dumps(_row(19)) + "\n")
    assert m.build_report(run, reader, m.WallRate(), processes=[], include_gpu=False)["status"] == "process_not_found"


def test_wall_rate_includes_wait_time_and_resets_after_step_rewind():
    m = _module()
    rate = m.WallRate()
    assert rate.observe(0, 100) is None
    assert rate.observe(10, 110) is None
    assert rate.observe(20, 110) == pytest.approx(.5)  # validation took 10s
    assert rate.observe(30, 120) == pytest.approx(2/3)
    assert rate.observe(40, 1) is None


def test_process_matching_resolves_paths_and_excludes_children(tmp_path):
    m = _module()
    run = tmp_path / "run"
    assert m.process_output_dir(["python", "/repo/scripts/train_real_robot.py", "--output-dir=run"], tmp_path) == run
    assert m.process_output_dir(["python", "monitor_real_robot_training.py", "--run-dir", str(run)], tmp_path) is None
    assert m.process_output_dir(["bash", "-c", "python train_real_robot.py --output-dir run"], tmp_path) is None
    parent, child = _process(run), _process(run, pid=101, parent_pid=100)
    assert m.matching_processes(run, [parent, child]) == [parent]


def test_duplicate_processes_stale_metrics_and_error_are_visible(tmp_path):
    m = _module()
    run, reader = _fixture(tmp_path, m)
    duplicate = m.build_report(run, reader, m.WallRate(), processes=[_process(run), _process(run, pid=102)], include_gpu=False)
    assert duplicate["status"] == "multiple_processes" and duplicate["warnings"]
    stale = m.build_report(run, reader, m.WallRate(), now=reader.path.stat().st_mtime+200,
                           processes=[_process(run)], include_gpu=False)
    assert any("长时间未更新" in v for v in stale["warnings"])
    (run / "train.log").write_text("RuntimeError: CUDA out of memory\n")
    stopped = m.build_report(run, reader, m.WallRate(), processes=[], include_gpu=False)
    assert stopped["status"] == "error" and "CUDA out of memory" in stopped["warnings"][-1]


def test_nonfinite_metrics_have_warnings_and_json_remains_valid(tmp_path):
    m = _module()
    run, reader = _fixture(tmp_path, m)
    with reader.path.open("a") as stream:
        stream.write(json.dumps(_row(19, train={"loss_total": float("nan")})) + "\n")
    report = m.build_report(run, reader, m.WallRate(), processes=[], include_gpu=False)
    assert report["latest_losses"]["loss_total"] is None and reader.nonfinite_rows == 1
    assert any("非有限" in w for w in report["warnings"])
    json.dumps(report, allow_nan=False)


def test_gpu_query_is_optional_and_own_process_memory_is_distinguished(monkeypatch):
    m = _module()
    original_query = m._nvidia_query
    def query(fields, kind="gpu"):
        return [["GPU-1", "Test GPU", "80", "5000", "16000", "60"]] if kind == "gpu" else [["GPU-1", "100", "2600"]]
    monkeypatch.setattr(m, "_nvidia_query", query)
    gpu = m.gpu_snapshot(100)[0]
    assert gpu["used_mib"] == 5000 and gpu["training_process_mib"] == 2600
    monkeypatch.setattr(m, "_nvidia_query", original_query)
    def missing(*args, **kwargs):
        raise FileNotFoundError()
    monkeypatch.setattr(m.subprocess, "run", missing)
    assert m._nvidia_query("name") == []


def test_cli_json_is_readable_and_accepts_completed_run_without_torch(tmp_path, capsys):
    m = _module()
    run, _ = _fixture(tmp_path, m)
    (run / "summary.json").write_text(json.dumps({"completed_steps": 18, "step_budget": 120000,
                                                 "termination": "early_stopping"}))
    assert m.main(["--run-dir", str(run), "--json", "--watch", "--no-gpu"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "early_stopped"


@pytest.mark.parametrize("option,value", [("--interval", "0"), ("--interval", "nan"),
                                          ("--recent-window", "0"), ("--stale-after", "inf")])
def test_cli_rejects_invalid_polling_controls(option, value):
    with pytest.raises(SystemExit):
        _module().main([option, value])
