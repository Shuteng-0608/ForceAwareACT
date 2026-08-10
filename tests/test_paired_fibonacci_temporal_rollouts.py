import argparse
import csv
import json
from pathlib import Path

import pytest

from scripts.run_paired50_q1_temporal_rollout_pilot import (
    build_signed_decay_specs,
)
from scripts.run_paired_fibonacci_temporal_rollouts import (
    DEFAULT_SIGNED_DECAYS,
    TaskPoint,
    _configuration_rows,
    _point_rows,
    _protocol_fingerprint,
    build_point_rollout_command,
    build_run_matrix,
    read_task_points,
    select_points,
    validate_run_summary,
    write_or_validate_plan,
    write_progress,
)


def _write_points(path: Path) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "point_index",
                "hole_offset_x",
                "hole_offset_y",
                "hole_offset_z",
            ],
        )
        writer.writeheader()
        writer.writerows(
            [
                {
                    "point_index": 1,
                    "hole_offset_x": 0.001,
                    "hole_offset_y": 0.0,
                    "hole_offset_z": -0.002,
                },
                {
                    "point_index": 2,
                    "hole_offset_x": -0.003,
                    "hole_offset_y": 0.0,
                    "hole_offset_z": 0.001,
                },
            ]
        )


def _args(tmp_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        output_dir=tmp_path / "experiment",
        model_xml=tmp_path / "model.xml",
        official_checkpoint=tmp_path / "official.pt",
        contact_checkpoint=tmp_path / "contact.pt",
        device="cuda",
        max_rollout_steps=600,
        selected_model_ids=("official_act", "highrate_contact_v3"),
        seed_base=100,
    )


def test_default_k_set_is_preregistered_typical_set():
    assert DEFAULT_SIGNED_DECAYS == (0.01, 0.05, 0.071, 0.0735, 0.075)


def test_read_points_requires_consecutive_unique_finite_rows(tmp_path):
    path = tmp_path / "points.csv"
    _write_points(path)

    points = read_task_points(path)

    assert points == (
        TaskPoint(1, 0.001, 0.0, -0.002),
        TaskPoint(2, -0.003, 0.0, 0.001),
    )
    assert points[0].radius_mm == pytest.approx(5**0.5)


def test_select_points_preserves_source_order_and_rejects_unknown(tmp_path):
    path = tmp_path / "points.csv"
    _write_points(path)
    points = read_task_points(path)

    assert [point.point_index for point in select_points(points, [2, 1])] == [1, 2]
    with pytest.raises(ValueError, match="unknown point indices"):
        select_points(points, [3])


def test_run_matrix_is_paired_with_same_point_seed(tmp_path):
    points = (TaskPoint(1, 0.001, 0.0, 0.002), TaskPoint(4, -0.003, 0.0, 0.0))
    specs = build_signed_decay_specs(
        tmp_path / "official.pt",
        tmp_path / "contact.pt",
        (0.01, 0.0735),
    )

    runs = build_run_matrix(points, specs, tmp_path / "out", seed_base=10)

    assert len(runs) == 8
    assert [run.seed for run in runs[:4]] == [10] * 4
    assert [run.seed for run in runs[4:]] == [13] * 4
    assert {run.point_index for run in runs[:4]} == {1}
    assert runs[0].output_dir == (
        tmp_path / "out" / "official_act__signed_kp0p01" / "point_001"
    )


def test_point_command_locks_current_fairness_protocol(tmp_path):
    args = _args(tmp_path)
    specs = build_signed_decay_specs(
        args.official_checkpoint,
        args.contact_checkpoint,
        (0.0735,),
        model_ids=("highrate_contact_v3",),
    )
    run = build_run_matrix(
        (TaskPoint(7, -0.00125, 0.0, 0.0035),),
        specs,
        args.output_dir,
        seed_base=100,
    )[0]

    command = build_point_rollout_command(args, specs[0], run)

    def value(option: str) -> str:
        return command[command.index(option) + 1]

    assert value("--output-dir") == str(run.output_dir)
    assert value("--hole-offset-x") == "-0.00125"
    assert value("--hole-offset-z") == "0.0035"
    assert value("--seed") == "106"
    assert value("--action-mode") == "joint_pos"
    assert value("--action-select-mode") == "signed_temporal"
    assert value("--temporal-agg-decay") == "0.0735"
    assert value("--policy-rate-hz") == "30"
    assert value("--force-stop-threshold") == "100"
    assert value("--safe-force-threshold") == "40"
    assert value("--contact-latent-mode") == "zero"
    assert "--save-videos" not in command
    assert "--save-force-hud-video" not in command
    assert "--record-force-metrics" in command
    assert "--execute-actions" in command


def test_existing_plan_is_immutable_by_protocol_fingerprint(tmp_path):
    path = tmp_path / "experiment_plan.json"
    protocol = {"point_sha256": "abc", "signed_decays": [0.01]}
    run = build_run_matrix(
        (TaskPoint(1, 0.0, 0.0, 0.0),),
        build_signed_decay_specs(
            tmp_path / "official.pt",
            tmp_path / "contact.pt",
            (0.01,),
            model_ids=("official_act",),
        ),
        tmp_path / "out",
        seed_base=0,
    )

    first = write_or_validate_plan(path, protocol, run)
    second = write_or_validate_plan(path, protocol, run)

    assert first == second
    assert first["protocol_fingerprint"] == _protocol_fingerprint(protocol)
    assert json.loads(path.read_text())["planned_runs"] == 1
    with pytest.raises(ValueError, match="different protocol"):
        write_or_validate_plan(path, {"signed_decays": [0.02]}, run)


def test_configuration_and_point_aggregates_preserve_pairing(tmp_path):
    specs = build_signed_decay_specs(
        tmp_path / "official.pt",
        tmp_path / "contact.pt",
        (0.01,),
    )
    points = (TaskPoint(1, 0.001, 0.0, 0.0), TaskPoint(2, 0.0, 0.0, 0.002))
    rows = []
    for point in points:
        for spec in specs:
            rows.append(
                {
                    "configuration_id": spec.configuration_id,
                    "model_id": spec.model_id,
                    "executor_id": spec.executor_id,
                    "signed_decay": spec.signed_decay,
                    "point_index": point.point_index,
                    "seed": 10 + point.point_index - 1,
                    "task_success": spec.model_id == "highrate_contact_v3",
                    "safe_success_raw": False,
                    "safe_success_compensated": False,
                    "stop_reason": "success" if spec.model_id == "highrate_contact_v3" else "max_rollout_steps",
                    "success_time": 10.0 if spec.model_id == "highrate_contact_v3" else None,
                    "compensated_max_force_norm": 50.0,
                    "compensated_above_40n_duration": 0.2,
                    "compensated_excess_force_exposure": 1.0,
                }
            )

    configurations = _configuration_rows(rows, specs, total_points=2)
    fields, paired = _point_rows(rows, points, specs)

    assert configurations[0]["task_success_rate"] == 0.0
    assert configurations[1]["task_success_rate"] == 1.0
    assert configurations[1]["mean_success_time"] == 10.0
    assert len(paired) == 2
    assert paired[0]["seed"] == 10
    assert "official_act__signed_kp0p01__task_success" in fields
    assert paired[0]["highrate_contact_v3__signed_kp0p01__task_success"] is True


def test_progress_eta_uses_only_newly_completed_rollouts(tmp_path, monkeypatch):
    times = iter((100.0, 120.0))
    monkeypatch.setattr(
        "scripts.run_paired_fibonacci_temporal_rollouts.time.monotonic",
        lambda: next(times),
    )

    first = write_progress(
        tmp_path,
        status="RUNNING",
        completed_rollouts=4,
        planned_rollouts=10,
        failed_run_ids=[],
        current_run_id=None,
        process_started_monotonic=100.0,
        process_start_completed=4,
    )
    second = write_progress(
        tmp_path,
        status="RUNNING",
        completed_rollouts=6,
        planned_rollouts=10,
        failed_run_ids=[],
        current_run_id="config/point_007",
        process_started_monotonic=100.0,
        process_start_completed=4,
    )

    assert first["eta_seconds"] is None
    assert second["average_seconds_per_new_rollout"] == 10.0
    assert second["eta_seconds"] == 40.0
    assert json.loads((tmp_path / "progress.json").read_text()) == second


def test_fibonacci_summary_contract_rejects_incomplete_force_metrics(tmp_path):
    args = _args(tmp_path)
    args.model_xml.touch()
    args.contact_checkpoint.touch()
    specs = build_signed_decay_specs(
        args.official_checkpoint,
        args.contact_checkpoint,
        (0.0735,),
        model_ids=("highrate_contact_v3",),
    )
    spec = specs[0]
    run = build_run_matrix(
        (TaskPoint(1, 0.0, 0.0, 0.0),), specs, args.output_dir, seed_base=0
    )[0]
    summary = {
        "rollout_protocol_version": "paired_action_executor_rollout_v3",
        "checkpoint": str(spec.checkpoint.resolve()),
        "model_xml": str(args.model_xml.resolve()),
        "rollout_mode": "execute",
        "seed": 0,
        "action_mode": "joint_pos",
        "action_select_mode": "signed_temporal",
        "policy_query_interval": 1,
        "contact_latent_mode": "zero",
        "policy_rate_hz": 30.0,
        "max_rollout_steps": 600,
        "ema_alpha": 1.0,
        "max_delta_q": 0.02,
        "force_stop_threshold": 100.0,
        "safe_force_threshold": 40.0,
        "hole_offset_x": 0.0,
        "hole_offset_y": 0.0,
        "hole_offset_z": 0.0,
        "temporal_signed_decay_equivalent": 0.0735,
        "temporal_endpoint": None,
        "force_metrics_recorded": True,
        "force_metrics_version": "raw_gravity_compensated_physics_interval_v1",
        "force_metrics_primary_wrench": "compensated",
        "force_metrics_threshold": 40.0,
    }

    errors = validate_run_summary(summary, args, spec, run)

    assert any("missing required force metric" in error for error in errors)
