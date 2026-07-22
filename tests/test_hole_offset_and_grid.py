import csv
import json

import numpy as np
import pandas as pd
import pytest
import torch

from scripts.plot_hole_grid_results import aggregate_grid_results, main as plot_grid_main
from scripts.generate_fibonacci_disk_points import fibonacci_disk_points, write_points_csv
from scripts.generate_random_disk_points import random_disk_points
from scripts.run_mujoco_hole_grid import (
    _build_rollout_command,
    _expected_rollout_contract,
    _summary_row_from_manifest_run,
    generate_task_points,
    latin_hypercube_points,
    parse_args as parse_grid_args,
    parse_offset_list,
    read_task_points_csv,
    resolve_point_set_seed,
    resolve_rollout_seed_base,
    run_grid,
    run_name,
    wilson_ci,
    write_position_summary_csv,
    write_random_position_summary,
)
from scripts.run_mujoco_policy_rollout import (
    SUMMARY_REQUIRED_KEYS,
    apply_hole_body_offset,
    resolve_hole_body,
    resolve_named_site,
    validate_hole_assembly_structure,
)
from scripts.summarize_rollouts import collect_rollouts


def _load_mujoco():
    mujoco = pytest.importorskip("mujoco")
    return mujoco


def _model_from_xml(xml: str):
    mujoco = _load_mujoco()
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return mujoco, model, data


def _minimal_xml(parent_euler: str = "0 0 0") -> str:
    return f"""
    <mujoco>
      <compiler angle="degree"/>
      <worldbody>
        <body name="parent" pos="0 0 0" euler="{parent_euler}">
          <body name="hole_body" pos="0.1 0.2 0.3">
            <geom name="hole_geom" type="box" size="0.01 0.01 0.01"/>
            <site name="hole_goal_site" pos="0 0 0" size="0.002"/>
          </body>
          <body name="wrong_body" pos="0.5 0 0">
            <geom name="wrong_geom" type="box" size="0.01 0.01 0.01"/>
          </body>
        </body>
      </worldbody>
    </mujoco>
    """


def test_world_frame_offset_with_world_aligned_parent_moves_site_by_requested_offset():
    _, model, data = _model_from_xml(_minimal_xml())
    site_id = resolve_named_site(model, "hole_goal_site")
    body_id = resolve_hole_body(model, site_id, "hole_body")

    metadata = apply_hole_body_offset(
        model,
        data,
        body_id,
        site_id,
        np.asarray([0.002, 0.0, -0.002]),
        "world",
    )

    np.testing.assert_allclose(metadata["actual_hole_offset"], [0.002, 0.0, -0.002], atol=1e-9)


def test_world_frame_offset_with_rotated_parent_uses_parent_local_conversion():
    _, model, data = _model_from_xml(_minimal_xml(parent_euler="0 0 90"))
    site_id = resolve_named_site(model, "hole_goal_site")
    body_id = resolve_hole_body(model, site_id, "hole_body")

    metadata = apply_hole_body_offset(
        model,
        data,
        body_id,
        site_id,
        np.asarray([0.002, 0.0, 0.002]),
        "world",
    )

    np.testing.assert_allclose(metadata["actual_hole_offset"], [0.002, 0.0, 0.002], atol=1e-9)
    assert not np.allclose(
        metadata["actual_hole_body_local_position"] - metadata["nominal_hole_body_local_position"],
        [0.002, 0.0, 0.002],
    )


def test_zero_offset_leaves_site_and_body_positions_unchanged():
    _, model, data = _model_from_xml(_minimal_xml())
    site_id = resolve_named_site(model, "hole_goal_site")
    body_id = resolve_hole_body(model, site_id, "hole_body")

    metadata = apply_hole_body_offset(model, data, body_id, site_id, np.zeros(3), "world")

    np.testing.assert_allclose(
        metadata["actual_hole_goal_position"],
        metadata["nominal_hole_goal_position"],
        atol=1e-12,
    )
    np.testing.assert_allclose(
        metadata["actual_hole_body_local_position"],
        metadata["nominal_hole_body_local_position"],
        atol=1e-12,
    )


def test_missing_site_error_contains_site_name():
    _, model, _ = _model_from_xml(_minimal_xml())

    with pytest.raises(ValueError, match="missing_site"):
        resolve_named_site(model, "missing_site")


def test_missing_body_error_contains_body_name():
    _, model, _ = _model_from_xml(_minimal_xml())
    site_id = resolve_named_site(model, "hole_goal_site")

    with pytest.raises(ValueError, match="missing_body"):
        resolve_hole_body(model, site_id, "missing_body")


def test_explicit_body_validation_accepts_site_and_expected_geom_subtree():
    _, model, _ = _model_from_xml(_minimal_xml())
    site_id = resolve_named_site(model, "hole_goal_site")
    body_id = resolve_hole_body(model, site_id, "hole_body")

    metadata = validate_hole_assembly_structure(
        model,
        body_id,
        site_id,
        expected_hole_geom_names=("hole_geom",),
    )

    assert metadata["hole_body_name"] == "hole_body"
    assert metadata["site_owner_body_name"] == "hole_body"
    assert metadata["hole_geom_names"] == ["hole_geom"]


def test_wrong_body_produces_clear_error_without_fallback():
    _, model, _ = _model_from_xml(_minimal_xml())
    site_id = resolve_named_site(model, "hole_goal_site")
    wrong_body_id = resolve_hole_body(model, site_id, "wrong_body")

    with pytest.raises(ValueError, match="not inside selected hole body"):
        validate_hole_assembly_structure(
            model,
            wrong_body_id,
            site_id,
            expected_hole_geom_names=("hole_geom",),
        )


def test_summary_required_keys_include_hole_offset_metadata():
    for key in (
        "hole_site_name",
        "hole_body_name",
        "hole_offset_frame",
        "requested_hole_offset",
        "actual_hole_offset",
        "nominal_hole_goal_position",
        "actual_hole_goal_position",
        "nominal_hole_body_local_position",
        "actual_hole_body_local_position",
    ):
        assert key in SUMMARY_REQUIRED_KEYS


def test_aggregator_fallback_old_log_uses_zero_offsets(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    with (run_dir / "rollout_log.csv").open("w", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "step",
                "time",
                "mode",
                "action_select_mode",
                "peg_to_hole_dist",
                "peg_to_hole_axial_error",
                "peg_to_hole_lateral_error",
                "force_norm",
                "stop_reason",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "step": 0,
                "time": 0.0,
                "mode": "zero",
                "action_select_mode": "mid",
                "peg_to_hole_dist": 0.01,
                "peg_to_hole_axial_error": 0.01,
                "peg_to_hole_lateral_error": 0.002,
                "force_norm": 1.0,
                "stop_reason": "max_rollout_steps",
            }
        )

    rows = collect_rollouts(tmp_path, "run")

    assert rows[0]["hole_offset_x"] == 0.0
    assert rows[0]["hole_offset_y"] == 0.0
    assert rows[0]["hole_offset_z"] == 0.0


def test_grid_offset_parsing_and_run_name_are_stable():
    assert parse_offset_list("-0.002,0,0.002") == [-0.002, 0.0, 0.002]
    assert run_name(-0.002, 0.002, 1) == "x_m002mm_z_p002mm_repeat_001"
    assert run_name(0.0, 0.0, 1) == "x_p000mm_z_p000mm_repeat_001"


def test_fibonacci_disk_points_are_deterministic_and_area_uniform():
    points = fibonacci_disk_points(100, 4.0)

    assert points == fibonacci_disk_points(100, 4.0)
    assert len(points) == 100
    radii_mm = np.asarray(
        [
            np.hypot(point["hole_offset_x"], point["hole_offset_z"]) * 1000.0
            for point in points
        ]
    )
    expected_squared_radii = 16.0 * (np.arange(100) + 0.5) / 100.0
    np.testing.assert_allclose(radii_mm**2, expected_squared_radii, atol=1e-12)
    assert radii_mm.max() < 4.0


def test_random_disk_points_are_seeded_and_stay_inside_radius():
    points = random_disk_points(100, 60.0, seed=20260714)

    assert points == random_disk_points(100, 60.0, seed=20260714)
    assert points != random_disk_points(100, 60.0, seed=20260715)
    assert len(points) == 100
    radii_mm = np.asarray(
        [
            np.hypot(point["hole_offset_x"], point["hole_offset_z"]) * 1000.0
            for point in points
        ]
    )
    assert bool((radii_mm <= 60.0).all())
    assert bool((radii_mm >= 0.0).all())
    assert {point["sampling_seed"] for point in points} == {20260714}


def test_fixed_task_points_csv_round_trips_in_metres(tmp_path):
    path = tmp_path / "fixed.csv"
    source = fibonacci_disk_points(5, 4.0)
    write_points_csv(path, source)

    loaded = read_task_points_csv(path)

    assert len(loaded) == 5
    assert loaded[0]["point_index"] == 1
    assert loaded[0]["hole_offset_x"] == pytest.approx(source[0]["hole_offset_x"])
    assert loaded[0]["hole_offset_y"] == 0.0
    assert loaded[0]["run_name"].startswith("point_001_")


def test_fixed_task_points_override_generated_sampling_in_dry_run(tmp_path):
    point_path = tmp_path / "fixed.csv"
    write_points_csv(point_path, fibonacci_disk_points(5, 4.0))
    args = parse_grid_args(
        [
            "--checkpoint",
            "checkpoint.pt",
            "--normalization-stats",
            "stats.pt",
            "--model-xml",
            "model.xml",
            "--output-root",
            str(tmp_path / "grid"),
            "--task-points-csv",
            str(point_path),
            "--dry-run",
            "--no-plot-results",
        ]
    )

    manifest = run_grid(args)

    assert manifest["sampling_mode"] == "file"
    assert manifest["num_points"] == 5
    assert len(manifest["runs"]) == 5
    assert manifest["task_points_csv"] == point_path


def test_grid_dry_run_creates_manifest_with_all_commands(tmp_path):
    args = parse_grid_args(
        [
            "--checkpoint",
            "checkpoint.pt",
            "--normalization-stats",
            "stats.pt",
            "--model-xml",
            "model.xml",
            "--output-root",
            str(tmp_path / "grid"),
            "--device",
            "cpu",
            "--x-offsets=-0.002,0,0.002",
            "--z-offsets=-0.002,0,0.002",
            "--dry-run",
            "--no-plot-results",
        ]
    )

    manifest = run_grid(args)
    manifest_path = tmp_path / "grid" / "grid_manifest.json"

    assert manifest_path.is_file()
    assert len(manifest["runs"]) == 9
    assert all(run["status"] == "dry_run" for run in manifest["runs"])
    commands = [run["command"] for run in manifest["runs"]]
    joined_commands = [" ".join(command) for command in commands]
    assert any("--hole-offset-x=-0.002" in command for command in commands)
    assert any("--hole-offset-y=0.0" in command for command in commands)
    assert any("--hole-offset-z=0.002" in command for command in commands)
    assert all("--hole-site-name hole_goal_site" in command for command in joined_commands)
    assert all("--hole-body-name wall_task" in command for command in joined_commands)
    assert all("--device cpu" in command for command in joined_commands)
    assert manifest["device"] == "cpu"
    assert manifest["policy_config"]["device"] == "cpu"
    assert all(run["device"] == "cpu" for run in manifest["runs"])
    assert manifest["policy_config"]["hole_body_name"] == "wall_task"
    assert manifest["policy_config"]["contact_enter_force_threshold"] == 5.0
    assert manifest["policy_config"]["contact_exit_force_threshold"] == 3.0
    assert manifest["policy_config"]["contact_min_steps"] == 2
    assert manifest["policy_config"]["safe_force_threshold"] is None
    assert manifest["policy_config"]["hard_force_threshold"] is None
    assert manifest["contact_recovery_config"] == {
        "contact_enter_force_n": 5.0,
        "contact_exit_force_n": 3.0,
        "contact_min_steps": 2,
        "success_force_n": 40.0,
        "safe_force_n": 40.0,
        "hard_force_n": 1000.0,
        "success_hold_steps": 15,
    }


def test_grid_rollout_command_forwards_requested_device(tmp_path):
    args = parse_grid_args(
        [
            "--checkpoint",
            "checkpoint.pt",
            "--normalization-stats",
            "stats.pt",
            "--model-xml",
            "model.xml",
            "--output-root",
            str(tmp_path / "grid"),
            "--device",
            "cuda",
        ]
    )

    command = _build_rollout_command(args, tmp_path / "run", 0.0, 0.0, 0.0, seed=0)

    device_index = command.index("--device")
    assert command[device_index + 1] == "cuda"


def test_grid_rollout_command_forwards_numeric_action_index_and_temporal_decay(tmp_path):
    args = parse_grid_args(
        [
            "--checkpoint",
            "checkpoint.pt",
            "--normalization-stats",
            "stats.pt",
            "--model-xml",
            "model.xml",
            "--output-root",
            str(tmp_path / "grid"),
            "--action-select-mode",
            "7",
            "--temporal-agg-decay",
            "0.25",
        ]
    )

    command = _build_rollout_command(args, tmp_path / "run", 0.0, 0.0, 0.0, seed=0)

    assert command[command.index("--action-select-mode") + 1] == "7"
    assert command[command.index("--temporal-agg-decay") + 1] == "0.25"


def test_grid_rollout_command_forwards_contact_recovery_thresholds(tmp_path):
    args = parse_grid_args(
        [
            "--checkpoint",
            "checkpoint.pt",
            "--normalization-stats",
            "stats.pt",
            "--model-xml",
            "model.xml",
            "--output-root",
            str(tmp_path / "grid"),
            "--contact-enter-force-threshold",
            "6.5",
            "--contact-exit-force-threshold",
            "2.5",
            "--contact-min-steps",
            "4",
            "--safe-force-threshold",
            "35",
            "--hard-force-threshold",
            "120",
        ]
    )

    command = _build_rollout_command(args, tmp_path / "run", 0.0, 0.0, 0.0, seed=0)

    expected = {
        "--contact-enter-force-threshold": "6.5",
        "--contact-exit-force-threshold": "2.5",
        "--contact-min-steps": "4",
        "--safe-force-threshold": "35.0",
        "--hard-force-threshold": "120.0",
    }
    for option, value in expected.items():
        assert command[command.index(option) + 1] == value


def test_grid_rollout_command_omits_optional_none_thresholds(tmp_path):
    args = parse_grid_args(
        [
            "--checkpoint",
            "checkpoint.pt",
            "--normalization-stats",
            "stats.pt",
            "--model-xml",
            "model.xml",
            "--output-root",
            str(tmp_path / "grid"),
        ]
    )

    command = _build_rollout_command(args, tmp_path / "run", 0.0, 0.0, 0.0, seed=0)

    assert "--safe-force-threshold" not in command
    assert "--hard-force-threshold" not in command
    assert "None" not in command


def test_grid_rejects_invalid_contact_recovery_thresholds_before_writing(tmp_path):
    output_root = tmp_path / "grid"
    args = parse_grid_args(
        [
            "--checkpoint",
            "checkpoint.pt",
            "--normalization-stats",
            "stats.pt",
            "--model-xml",
            "model.xml",
            "--output-root",
            str(output_root),
            "--safe-force-threshold",
            "4",
            "--dry-run",
        ]
    )

    with pytest.raises(ValueError, match="contact_enter_force_n"):
        run_grid(args)

    assert not output_root.exists()


def test_grid_position_summary_extracts_contact_recovery_metrics(tmp_path):
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    summary = {
        "success": True,
        "max_force_norm": 45.0,
        "recovery_success": True,
        "safe_recovery_success": False,
        "contact_recovery_metrics_valid": True,
        "contact_recovery_metrics": {
            "contact_event_count": 3,
            "contact_duration_s": 0.75,
            "force_excess_integral_n_s": 1.25,
            "hard_force_violation": False,
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary))
    run = {
        "point_index": 1,
        "x_offset": 0.001,
        "y_offset": 0.0,
        "z_offset": -0.001,
        "output_dir": output_dir,
        "status": "success",
    }

    row = _summary_row_from_manifest_run(run, success_force_threshold=40.0)

    assert row["success"] is True
    assert row["safe_success"] is False
    assert row["recovery_success"] is True
    assert row["safe_recovery_success"] is False
    assert row["contact_recovery_metrics_valid"] is True
    assert row["contact_event_count"] == 3
    assert row["contact_duration_s"] == pytest.approx(0.75)
    assert row["force_excess_integral_n_s"] == pytest.approx(1.25)
    assert row["hard_force_violation"] is False

    csv_path = tmp_path / "position_summary.csv"
    written_rows = write_position_summary_csv(
        csv_path,
        {
            "success_thresholds": {"success_force_threshold": 40.0},
            "runs": [run],
        },
    )
    with csv_path.open(newline="") as csv_file:
        csv_row = next(csv.DictReader(csv_file))

    assert written_rows[0]["contact_event_count"] == 3
    assert csv_row["recovery_success"] == "True"
    assert csv_row["safe_recovery_success"] == "False"
    assert csv_row["contact_recovery_metrics_valid"] == "True"
    assert csv_row["contact_event_count"] == "3"
    assert csv_row["contact_duration_s"] == "0.75"
    assert csv_row["force_excess_integral_n_s"] == "1.25"
    assert csv_row["hard_force_violation"] == "False"


def test_grid_position_summary_keeps_legacy_success_without_fabricating_new_metrics(
    tmp_path,
):
    output_dir = tmp_path / "legacy"
    output_dir.mkdir()
    (output_dir / "summary.json").write_text(
        json.dumps({"success": True, "max_force_norm": 20.0})
    )
    run = {
        "point_index": 1,
        "x_offset": 0.0,
        "y_offset": 0.0,
        "z_offset": 0.0,
        "output_dir": output_dir,
        "status": "success",
    }

    row = _summary_row_from_manifest_run(run, success_force_threshold=40.0)

    assert row["success"] is True
    assert row["safe_success"] is True
    assert row["recovery_success"] == ""
    assert row["safe_recovery_success"] == ""
    assert row["contact_recovery_metrics_valid"] == ""


def test_random_summary_recovery_rates_use_only_valid_metric_runs(tmp_path):
    rows = [
        {
            "success": False,
            "safe_success": False,
            "contact_recovery_metrics_valid": True,
            "recovery_success": True,
            "safe_recovery_success": True,
            "hard_force_violation": False,
            "hole_offset_x": 0.0,
            "hole_offset_z": 0.0,
            "radial_offset": 0.0,
            "quadrant": "center",
            "success_time": "",
            "max_force": 10.0,
            "contact_event_count": 1,
            "contact_duration_s": 0.1,
            "force_excess_integral_n_s": 0.0,
        },
        {
            "success": False,
            "safe_success": False,
            "contact_recovery_metrics_valid": True,
            "recovery_success": False,
            "safe_recovery_success": False,
            "hard_force_violation": False,
            "hole_offset_x": 0.001,
            "hole_offset_z": 0.0,
            "radial_offset": 0.001,
            "quadrant": "axis",
            "success_time": "",
            "max_force": 20.0,
            "contact_event_count": 1,
            "contact_duration_s": 0.1,
            "force_excess_integral_n_s": 0.0,
        },
        {
            "success": False,
            "safe_success": False,
            "contact_recovery_metrics_valid": False,
            # Invalid results must never increase either numerator.
            "recovery_success": True,
            "safe_recovery_success": True,
            "hard_force_violation": True,
            "hole_offset_x": -0.001,
            "hole_offset_z": 0.0,
            "radial_offset": 0.001,
            "quadrant": "axis",
            "success_time": "",
            "max_force": 30.0,
            "contact_event_count": "",
            "contact_duration_s": "",
            "force_excess_integral_n_s": "",
        },
    ]
    manifest = {
        "sampling_mode": "grid",
        "x_min": -0.002,
        "x_max": 0.002,
        "z_min": -0.002,
        "z_max": 0.002,
        "base_seed": 0,
        "point_set_seed": 0,
        "rollout_seed_base": 0,
        "total_planned_runs": 3,
        "runs": [{"status": "task_failed"} for _ in range(3)],
    }

    summary = write_random_position_summary(
        tmp_path / "random_position_summary.json", manifest, rows
    )

    assert summary["contact_metrics_reported_runs"] == 3
    assert summary["contact_metrics_valid_runs"] == 2
    assert summary["contact_metrics_invalid_runs"] == 1
    assert summary["recovery_rate_denominator_valid_runs"] == 2
    assert summary["recovery_successes"] == 1
    assert summary["safe_recovery_successes"] == 1
    assert summary["recovery_success_rate"] == pytest.approx(0.5)
    assert summary["safe_recovery_success_rate"] == pytest.approx(0.5)


def _write_legacy_rollout_artifacts(tmp_path):
    checkpoint = tmp_path / "checkpoint.pt"
    stats = tmp_path / "stats.pt"
    model_xml = tmp_path / "model.xml"
    torch.save({"config": {"policy_variant": "force_aware_act"}}, checkpoint)
    torch.save(
        {
            "qpos_mean": torch.zeros(7),
            "qpos_std": torch.ones(7),
            "action_mean": torch.zeros(7),
            "action_std": torch.ones(7),
            "force_mean": torch.zeros(6),
            "force_std": torch.ones(6),
            "action_mode": "action",
        },
        stats,
    )
    model_xml.write_text("<mujoco/>")
    return checkpoint, stats, model_xml


def test_grid_skip_existing_requires_exact_contract_and_artifact_hashes(tmp_path):
    checkpoint, stats, model_xml = _write_legacy_rollout_artifacts(tmp_path)
    output_root = tmp_path / "grid"
    args = parse_grid_args(
        [
            "--checkpoint",
            str(checkpoint),
            "--normalization-stats",
            str(stats),
            "--model-xml",
            str(model_xml),
            "--output-root",
            str(output_root),
            "--x-offsets=0",
            "--z-offsets=0",
            "--skip-existing",
            "--no-plot-results",
        ]
    )
    output_dir = output_root / run_name(0.0, 0.0, 1)
    output_dir.mkdir(parents=True)
    command = _build_rollout_command(args, output_dir, 0.0, 0.0, 0.0, seed=0)
    contract = _expected_rollout_contract(command)
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "rollout_contract": contract,
                "success": False,
                "max_force_norm": 0.0,
            }
        )
    )

    manifest = run_grid(args)

    assert manifest["runs"][0]["status"] == "skipped_existing"
    model_xml.write_text("<mujoco model='changed'/>")
    with pytest.raises(ValueError, match="does not exactly match"):
        run_grid(args)


def test_grid_skip_existing_rejects_legacy_summary_without_contract(tmp_path):
    checkpoint, stats, model_xml = _write_legacy_rollout_artifacts(tmp_path)
    output_root = tmp_path / "grid"
    args = parse_grid_args(
        [
            "--checkpoint",
            str(checkpoint),
            "--normalization-stats",
            str(stats),
            "--model-xml",
            str(model_xml),
            "--output-root",
            str(output_root),
            "--x-offsets=0",
            "--z-offsets=0",
            "--skip-existing",
            "--no-plot-results",
        ]
    )
    output_dir = output_root / run_name(0.0, 0.0, 1)
    output_dir.mkdir(parents=True)
    (output_dir / "summary.json").write_text(json.dumps({"success": True}))

    with pytest.raises(ValueError, match="without rollout_contract"):
        run_grid(args)


def test_rollout_command_keeps_negative_scientific_offsets_single_token(tmp_path):
    source_offsets = {
        "--hole-offset-x": -6.87416984117234e-05,
        "--hole-offset-y": 0.0,
        "--hole-offset-z": -2.525207122118121e-05,
    }
    args = parse_grid_args(
        [
            "--checkpoint",
            "checkpoint.pt",
            "--normalization-stats",
            "stats.pt",
            "--model-xml",
            "model.xml",
            "--output-root",
            str(tmp_path / "grid"),
            "--dry-run",
            "--no-plot-results",
        ]
    )

    command = _build_rollout_command(
        args,
        tmp_path / "grid" / "scientific_offsets",
        source_offsets["--hole-offset-x"],
        source_offsets["--hole-offset-y"],
        source_offsets["--hole-offset-z"],
        seed=0,
    )

    for option, source_value in source_offsets.items():
        matching_tokens = [token for token in command if token.startswith(f"{option}=")]
        assert len(matching_tokens) == 1
        token = matching_tokens[0]
        assert token == f"{option}={float(source_value)!r}"
        assert float(token.split("=", 1)[1]) == source_value
        assert option not in command

    assert "-6.87416984117234e-05" not in command
    assert "-2.525207122118121e-05" not in command


def test_latin_hypercube_sampling_is_deterministic_and_seeded():
    points_a = latin_hypercube_points(50, -0.002, 0.002, -0.002, 0.002, seed=20260702)
    points_b = latin_hypercube_points(50, -0.002, 0.002, -0.002, 0.002, seed=20260702)
    points_c = latin_hypercube_points(50, -0.002, 0.002, -0.002, 0.002, seed=20260703)

    assert points_a == points_b
    assert points_a != points_c
    assert len(points_a) == 50
    assert all(-0.002 <= x <= 0.002 and -0.002 <= z <= 0.002 for x, z in points_a)


def test_latin_hypercube_uses_each_one_dimensional_stratum_once():
    points = latin_hypercube_points(50, -0.002, 0.002, -0.002, 0.002, seed=7)
    x_strata = sorted(int((x - -0.002) / 0.004 * 50) for x, _ in points)
    z_strata = sorted(int((z - -0.002) / 0.004 * 50) for _, z in points)

    assert x_strata == list(range(50))
    assert z_strata == list(range(50))


def test_generate_lhs_task_points_count_is_paired_not_cartesian(tmp_path):
    args = parse_grid_args(
        [
            "--sampling-mode",
            "latin_hypercube",
            "--num-points",
            "50",
            "--checkpoint",
            "checkpoint.pt",
            "--normalization-stats",
            "stats.pt",
            "--model-xml",
            "model.xml",
            "--output-root",
            str(tmp_path / "grid"),
        ]
    )

    points = generate_task_points(args)

    assert len(points) == 50
    assert len({(point["hole_offset_x"], point["hole_offset_z"]) for point in points}) == 50
    assert points[0]["run_name"].startswith("point_001_x_")


def test_wilson_interval_edge_cases():
    low0, high0 = wilson_ci(0, 50)
    low25, high25 = wilson_ci(25, 50)
    low50, high50 = wilson_ci(50, 50)

    assert low0 == pytest.approx(0.0)
    assert 0.0 < high0 < 0.1
    assert low25 < 0.5 < high25
    assert 0.9 < low50 < 1.0
    assert high50 == pytest.approx(1.0)


def test_lhs_dry_run_creates_50_commands_no_video_and_reproducible_points(tmp_path):
    base_args = [
        "--sampling-mode",
        "latin_hypercube",
        "--num-points",
        "50",
        "--base-seed",
        "20260702",
        "--checkpoint",
        "checkpoint.pt",
        "--normalization-stats",
        "stats.pt",
        "--model-xml",
        "model.xml",
        "--dry-run",
        "--no-plot-results",
    ]
    args_a = parse_grid_args([*base_args, "--output-root", str(tmp_path / "grid_a")])
    args_b = parse_grid_args([*base_args, "--output-root", str(tmp_path / "grid_b")])

    manifest_a = run_grid(args_a)
    manifest_b = run_grid(args_b)

    assert len(manifest_a["runs"]) == 50
    assert len(manifest_b["runs"]) == 50
    assert all("--save-videos" not in run["command"] for run in manifest_a["runs"])
    assert [run["x_offset"] for run in manifest_a["runs"]] == [
        run["x_offset"] for run in manifest_b["runs"]
    ]
    task_points_a = (tmp_path / "grid_a" / "task_points.csv").read_text()
    task_points_b = (tmp_path / "grid_b" / "task_points.csv").read_text()
    normalized_a = task_points_a.replace(str(tmp_path / "grid_a"), "<ROOT>")
    normalized_b = task_points_b.replace(str(tmp_path / "grid_b"), "<ROOT>")
    assert normalized_a == normalized_b


def test_point_set_and_rollout_seeds_are_independent_and_recorded(tmp_path):
    common = [
        "--sampling-mode",
        "latin_hypercube",
        "--num-points",
        "3",
        "--point-set-seed",
        "101",
        "--checkpoint",
        "checkpoint.pt",
        "--normalization-stats",
        "stats.pt",
        "--model-xml",
        "model.xml",
        "--dry-run",
        "--no-plot-results",
    ]
    args_a = parse_grid_args(
        [
            *common,
            "--rollout-seed-base",
            "500",
            "--output-root",
            str(tmp_path / "a"),
        ]
    )
    args_b = parse_grid_args(
        [
            *common,
            "--rollout-seed-base",
            "900",
            "--output-root",
            str(tmp_path / "b"),
        ]
    )

    manifest_a = run_grid(args_a)
    manifest_b = run_grid(args_b)

    assert resolve_point_set_seed(args_a) == 101
    assert resolve_rollout_seed_base(args_a) == 500
    assert manifest_a["point_set_seed"] == manifest_b["point_set_seed"] == 101
    assert [run["x_offset"] for run in manifest_a["runs"]] == [
        run["x_offset"] for run in manifest_b["runs"]
    ]
    assert [run["z_offset"] for run in manifest_a["runs"]] == [
        run["z_offset"] for run in manifest_b["runs"]
    ]
    assert [run["rollout_seed"] for run in manifest_a["runs"]] == [500, 501, 502]
    assert [run["rollout_seed"] for run in manifest_b["runs"]] == [900, 901, 902]
    task_points = pd.read_csv(tmp_path / "a" / "task_points.csv")
    assert task_points["point_set_seed"].tolist() == [101, 101, 101]
    assert task_points["rollout_seed"].tolist() == [500, 501, 502]


def test_legacy_base_seed_remains_coupled(tmp_path):
    args = parse_grid_args(
        [
            "--sampling-mode",
            "latin_hypercube",
            "--num-points",
            "2",
            "--base-seed",
            "77",
            "--checkpoint",
            "checkpoint.pt",
            "--normalization-stats",
            "stats.pt",
            "--model-xml",
            "model.xml",
            "--output-root",
            str(tmp_path / "legacy"),
            "--dry-run",
            "--no-plot-results",
        ]
    )

    manifest = run_grid(args)

    assert manifest["base_seed"] == 77
    assert manifest["point_set_seed"] == 77
    assert manifest["rollout_seed_base"] == 77
    assert [run["rollout_seed"] for run in manifest["runs"]] == [77, 78]


def test_heatmap_aggregation_multiple_repeats():
    import pandas as pd

    df = pd.DataFrame(
        [
            {"hole_offset_x": -0.002, "hole_offset_z": 0.0, "success": True, "success_time": 1.0, "final_dist": 0.002, "final_lateral": 0.001, "max_force": 10.0, "force_gt_20_steps": 0, "force_gt_40_steps": 0},
            {"hole_offset_x": -0.002, "hole_offset_z": 0.0, "success": False, "success_time": "", "final_dist": 0.02, "final_lateral": 0.01, "max_force": 30.0, "force_gt_20_steps": 2, "force_gt_40_steps": 0},
            {"hole_offset_x": 0.0, "hole_offset_z": 0.0, "success": True, "success_time": 2.0, "final_dist": 0.003, "final_lateral": 0.002, "max_force": 20.0, "force_gt_20_steps": 1, "force_gt_40_steps": 0},
        ]
    )
    df["success_bool"] = df["success"]
    for column in ("success_time", "final_dist", "final_lateral", "max_force", "force_gt_20_steps", "force_gt_40_steps"):
        df[column] = pd.to_numeric(df[column], errors="coerce")

    table = aggregate_grid_results(df)
    cell = table[(table["hole_offset_x"] == -0.002) & (table["hole_offset_z"] == 0.0)].iloc[0]

    assert cell["runs"] == 2
    assert cell["successes"] == 1
    assert cell["success_rate"] == pytest.approx(0.5)
    assert cell["mean_success_time"] == pytest.approx(1.0)


def test_heatmap_outputs_are_created(tmp_path):
    summary_csv = tmp_path / "grid_summary.csv"
    output_dir = tmp_path / "plots"
    with summary_csv.open("w", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "run",
                "success",
                "success_time",
                "final_dist",
                "final_lateral",
                "max_force",
                "force_gt_20_steps",
                "force_gt_40_steps",
                "hole_offset_x",
                "hole_offset_z",
            ],
        )
        writer.writeheader()
        writer.writerow({"run": "a", "success": True, "success_time": 1.0, "final_dist": 0.002, "final_lateral": 0.001, "max_force": 10.0, "force_gt_20_steps": 0, "force_gt_40_steps": 0, "hole_offset_x": -0.002, "hole_offset_z": 0.0})
        writer.writerow({"run": "b", "success": False, "success_time": "", "final_dist": 0.02, "final_lateral": 0.01, "max_force": 30.0, "force_gt_20_steps": 2, "force_gt_40_steps": 0, "hole_offset_x": 0.0, "hole_offset_z": 0.0})

    exit_code = plot_grid_main(
        [
            "--summary-csv",
            str(summary_csv),
            "--output-dir",
            str(output_dir),
            "--formats",
            "png",
            "--dpi",
            "80",
            "--annotate",
        ]
    )

    assert exit_code == 0
    assert (output_dir / "hole_offset_success_rate_heatmap.png").is_file()
    assert (output_dir / "hole_grid_results_table.csv").is_file()
    assert (output_dir / "hole_position_success_scatter.png").is_file()
    assert (output_dir / "hole_position_safe_success_scatter.png").is_file()
    assert (output_dir / "success_rate_by_radial_bin.png").is_file()
    assert (output_dir / "success_rate_by_z_bin.png").is_file()
