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
    _protocol_fingerprint,
    build_point_rollout_command,
    build_run_matrix,
    read_task_points,
    select_points,
    write_or_validate_plan,
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
