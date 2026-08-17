import json
from datetime import datetime, timezone

import pytest

from scripts.monitor_act_aligned_training import (
    ProcessInfo,
    estimate_eta,
    find_training_process,
    read_jsonl_records,
    render_report,
    resolve_monitor_control,
)


def test_read_jsonl_records_ignores_partial_final_line(tmp_path):
    path = tmp_path / "metrics.jsonl"
    first = {"record_type": "step", "global_step": 10}
    second = {"record_type": "step", "global_step": 20}
    path.write_text(
        json.dumps(first) + "\n" + json.dumps(second) + "\n{\"partial\":",
        encoding="utf-8",
    )

    assert read_jsonl_records(path) == [first, second]


def test_estimate_eta_uses_process_elapsed_time():
    now = datetime(2026, 7, 29, 12, 0, tzinfo=timezone.utc)

    estimate = estimate_eta(
        current_step=1200,
        start_step=0,
        target_steps=24000,
        elapsed_seconds=600.0,
        now=now,
    )

    assert estimate is not None
    assert estimate.completed_steps == 1200
    assert estimate.steps_per_second == pytest.approx(2.0)
    assert estimate.seconds_per_step == pytest.approx(0.5)
    assert estimate.remaining_seconds == pytest.approx(11400.0)
    assert estimate.estimated_total_seconds == pytest.approx(12000.0)
    assert estimate.estimated_finish == datetime(
        2026,
        7,
        29,
        15,
        10,
        tzinfo=timezone.utc,
    )


def test_estimate_eta_supports_resumed_training_start_step():
    estimate = estimate_eta(
        current_step=3000,
        start_step=2000,
        target_steps=24000,
        elapsed_seconds=500.0,
        now=datetime(2026, 7, 29, tzinfo=timezone.utc),
    )

    assert estimate is not None
    assert estimate.completed_steps == 1000
    assert estimate.steps_per_second == pytest.approx(2.0)


def test_find_training_process_excludes_dataloader_children(
    tmp_path,
    monkeypatch,
):
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    command = (
        "python scripts/train_act_aligned_contact_cvae.py data "
        f"--output-dir {output_dir}"
    )
    processes = [
        ProcessInfo(100, 10, 300.0, command),
        ProcessInfo(101, 100, 290.0, command),
        ProcessInfo(102, 100, 290.0, command),
    ]
    monkeypatch.setattr(
        "scripts.monitor_act_aligned_training._process_table",
        lambda: processes,
    )

    process = find_training_process(output_dir)

    assert process is not None
    assert process.pid == 100


def test_find_training_process_supports_motion_control_entry(
    tmp_path,
    monkeypatch,
):
    output_dir = tmp_path / "motion"
    output_dir.mkdir()
    command = (
        "python scripts/train_act_aligned_motion_cvae_control.py data "
        f"--output-dir {output_dir}"
    )
    monkeypatch.setattr(
        "scripts.monitor_act_aligned_training._process_table",
        lambda: [ProcessInfo(200, 10, 30.0, command)],
    )

    process = find_training_process(output_dir)

    assert process is not None
    assert process.pid == 200


def test_find_training_process_supports_official_act_entry(
    tmp_path,
    monkeypatch,
):
    output_dir = tmp_path / "official"
    output_dir.mkdir()
    command = (
        "python scripts/train_official_act.py data "
        f"--output-dir {output_dir}"
    )
    monkeypatch.setattr(
        "scripts.monitor_act_aligned_training._process_table",
        lambda: [ProcessInfo(300, 10, 30.0, command)],
    )

    process = find_training_process(output_dir)

    assert process is not None
    assert process.pid == 300


def test_find_training_process_supports_official_act_no_latent_entry(
    tmp_path,
    monkeypatch,
):
    output_dir = tmp_path / "official_no_latent"
    output_dir.mkdir()
    command = (
        "python scripts/train_official_act_no_latent.py data "
        f"--output-dir {output_dir}"
    )
    monkeypatch.setattr(
        "scripts.monitor_act_aligned_training._process_table",
        lambda: [ProcessInfo(301, 10, 30.0, command)],
    )

    process = find_training_process(output_dir)

    assert process is not None
    assert process.pid == 301


def test_monitor_control_defaults_to_persisted_extension_horizon(tmp_path):
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    (output_dir / "run_metadata.json").write_text(
        json.dumps(
            {
                "run_control": {
                    "start_global_step": 25_000,
                    "target_optimizer_steps": 50_000,
                },
                "training": {"checkpoint_interval_steps": 4_000},
            }
        ),
        encoding="utf-8",
    )

    assert resolve_monitor_control(
        output_dir,
        target_steps=None,
        start_step=None,
        checkpoint_interval=None,
    ) == (50_000, 25_000, 4_000)


def test_monitor_derives_official_checkpoint_interval_in_steps(tmp_path):
    output_dir = tmp_path / "official"
    output_dir.mkdir()
    (output_dir / "run_metadata.json").write_text(
        json.dumps(
            {
                "run_control": {
                    "start_global_step": 25_000,
                    "target_optimizer_steps": 100_000,
                },
                "steps_per_epoch": 5,
                "training": {"checkpoint_interval_epochs": 100},
            }
        ),
        encoding="utf-8",
    )

    assert resolve_monitor_control(
        output_dir,
        target_steps=None,
        start_step=None,
        checkpoint_interval=None,
    ) == (100_000, 25_000, 500)


def test_report_distinguishes_convergence_stop_from_budget_completion(
    tmp_path,
    monkeypatch,
):
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    convergence = {
        "metric_name": "deployment_zero_action_l1_physical",
        "minimum_optimizer_steps": 25_000,
        "patience_validations": 3,
        "min_relative_improvement": 0.01,
        "best_metric": 0.02,
        "best_step": 28_000,
        "validations_without_meaningful_improvement": 3,
    }
    (output_dir / "metrics.jsonl").write_text(
        json.dumps(
            {
                "record_type": "epoch_segment",
                "global_step": 31_000,
                "epoch": 10,
                "step_in_epoch": 0,
                "validation": {
                    "deployment_zero_action_l1_physical": 0.021,
                },
                "convergence": convergence,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    best = {
        "global_step": 28_000,
        "metric": 0.02,
        "path": "/runs/example/best.pt",
    }
    (output_dir / "training_summary.json").write_text(
        json.dumps(
            {
                "passed": True,
                "global_step": 31_000,
                "stop_reason": "early_stopping_plateau",
                "run_control": {"convergence": convergence},
                "best_artifact": best,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "scripts.monitor_act_aligned_training.find_training_process",
        lambda *args, **kwargs: None,
    )

    report, _ = render_report(
        output_dir=output_dir,
        target_steps=50_000,
        start_step=25_000,
        checkpoint_interval=2_000,
        recent_window=20,
        stale_after=120.0,
        requested_pid=None,
        now_monotonic=1.0,
        now=datetime(2026, 8, 17, tzinfo=timezone.utc),
    )

    assert "EARLY STOPPED (CONVERGED)" in report
    assert "31,000/50,000" in report
    assert "patience=3/3" in report
    assert "best policy: step=28000" in report
