import csv
import json

import numpy as np
import pytest

from scripts.plot_hole_grid_results import aggregate_grid_results, main as plot_grid_main
from scripts.run_mujoco_hole_grid import (
    _build_rollout_command,
    generate_task_points,
    latin_hypercube_points,
    parse_args as parse_grid_args,
    parse_offset_list,
    run_grid,
    run_name,
    wilson_ci,
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
    assert manifest["policy_config"]["hole_body_name"] == "wall_task"


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
