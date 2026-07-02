import csv
import json

import pandas as pd

from scripts.plot_rollout_sensor_analysis import (
    Thresholds,
    compute_markers,
    compute_retroactive_success,
    load_rollout_log,
    load_summary_json,
    main,
)


def _write_rollout(path, rows, include_pred=True):
    fieldnames = [
        "step",
        "time",
        "peg_to_hole_dist",
        "peg_to_hole_axial_error",
        "peg_to_hole_lateral_error",
        "force_norm",
        "ft_0",
        "ft_1",
        "ft_2",
        "selected_action_delta_norm_after_ema",
        "applied_ctrl_delta_from_qpos_norm",
        "success_condition",
        "success_hold_counter",
        "stop_reason",
    ]
    fieldnames.extend(f"qcmd_{index}" for index in range(7))
    fieldnames.extend(f"qpos_{index}" for index in range(7))
    if include_pred:
        fieldnames.extend(["pred_force_norm_0", "pred_force_norm_mean", "pred_force_norm_max"])
    with path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            full_row = {field: row.get(field, "") for field in fieldnames}
            for index in range(7):
                full_row[f"qcmd_{index}"] = row.get(f"qcmd_{index}", 0.1 * index)
                full_row[f"qpos_{index}"] = row.get(f"qpos_{index}", 0.1 * index + 0.01)
            writer.writerow(full_row)


def _synthetic_rows(count=24, success_start=8, stop_reason=""):
    rows = []
    hold = 0
    for step in range(count):
        success_condition = step >= success_start
        hold = hold + 1 if success_condition else 0
        rows.append(
            {
                "step": step,
                "time": step / 30.0,
                "peg_to_hole_dist": max(0.002, 0.05 - 0.002 * step),
                "peg_to_hole_axial_error": 0.04 - 0.0015 * step,
                "peg_to_hole_lateral_error": max(0.001, 0.02 - 0.001 * step),
                "force_norm": [0.0, 1.0, 6.0, 12.0, 25.0, 41.0][step]
                if step < 6
                else 15.0,
                "ft_0": 0.1 * step,
                "ft_1": 0.2 * step,
                "ft_2": 0.3 * step,
                "selected_action_delta_norm_after_ema": 0.01 + 0.001 * step,
                "applied_ctrl_delta_from_qpos_norm": 0.02 + 0.001 * step,
                "pred_force_norm_0": 2.0 + step,
                "pred_force_norm_mean": 3.0 + step,
                "pred_force_norm_max": 4.0 + step,
                "success_condition": success_condition,
                "success_hold_counter": hold,
                "stop_reason": stop_reason if step == count - 1 else "",
            }
        )
    return rows


def test_marker_extraction_from_synthetic_rollout(tmp_path):
    log_path = tmp_path / "rollout_log.csv"
    _write_rollout(log_path, _synthetic_rows())
    df = load_rollout_log(log_path)

    markers = compute_markers(df, {}, Thresholds())

    assert markers["first_contact_step"] == 2
    assert markers["first_high_force_step"] == 4
    assert markers["first_very_high_force_step"] == 5
    assert markers["max_force_step"] == 5
    assert markers["max_force_norm"] == 41.0
    assert markers["min_dist_step"] == 23
    assert markers["min_lateral_step"] == 19


def test_success_reconstruction_requires_full_hold():
    success_df = pd.DataFrame(_synthetic_rows(count=24, success_start=8))
    short_df = pd.DataFrame(_synthetic_rows(count=20, success_start=8))

    success = compute_retroactive_success(success_df, Thresholds(success_hold_steps=15))
    short = compute_retroactive_success(short_df, Thresholds(success_hold_steps=15))

    assert success["success"] is True
    assert success["success_step"] == 22
    assert short["success"] is False
    assert short["success_hold_steps_observed"] == 12


def test_missing_optional_predicted_force_columns_do_not_crash(tmp_path):
    rollout_dir = tmp_path / "rollout"
    rollout_dir.mkdir()
    _write_rollout(rollout_dir / "rollout_log.csv", _synthetic_rows(), include_pred=False)

    exit_code = main(
        [
            "--rollout-dir",
            str(rollout_dir),
            "--output-dir",
            str(tmp_path / "plots"),
            "--formats",
            "png",
            "--dpi",
            "80",
        ]
    )

    assert exit_code == 0
    assert (tmp_path / "plots" / "combined_analysis.png").is_file()
    assert (tmp_path / "plots" / "summary_markers.json").is_file()


def test_summary_json_success_step_overrides_retroactive_inference(tmp_path):
    log_path = tmp_path / "rollout_log.csv"
    summary_path = tmp_path / "summary.json"
    _write_rollout(log_path, _synthetic_rows())
    summary_path.write_text(json.dumps({"success": True, "success_step": 99, "success_time": 3.3}))
    df = load_rollout_log(log_path)
    summary = load_summary_json(summary_path)

    markers = compute_markers(df, summary, Thresholds())

    assert markers["success_step"] == 99
    assert markers["success_time"] == 3.3
    assert markers["success_source"] == "summary_json"


def test_single_rollout_mode_generates_combined_and_markers(tmp_path):
    rollout_dir = tmp_path / "single"
    output_dir = tmp_path / "single_plots"
    rollout_dir.mkdir()
    _write_rollout(rollout_dir / "rollout_log.csv", _synthetic_rows(), include_pred=True)
    (rollout_dir / "summary.json").write_text(json.dumps({"success": True, "success_step": 22}))

    exit_code = main(
        [
            "--rollout-dir",
            str(rollout_dir),
            "--output-dir",
            str(output_dir),
            "--formats",
            "png",
            "--dpi",
            "80",
        ]
    )

    assert exit_code == 0
    assert (output_dir / "combined_analysis.png").is_file()
    assert (output_dir / "summary_markers.json").is_file()


def test_compare_mode_generates_combined_and_summary(tmp_path):
    rollout_a = tmp_path / "success"
    rollout_b = tmp_path / "failed"
    output_dir = tmp_path / "compare_plots"
    rollout_a.mkdir()
    rollout_b.mkdir()
    _write_rollout(rollout_a / "rollout_log.csv", _synthetic_rows(), include_pred=True)
    _write_rollout(
        rollout_b / "rollout_log.csv",
        _synthetic_rows(count=20, success_start=30, stop_reason="max_rollout_steps"),
        include_pred=True,
    )
    (rollout_a / "summary.json").write_text(json.dumps({"success": True, "success_step": 22}))
    (rollout_b / "summary.json").write_text(json.dumps({"success": False}))

    exit_code = main(
        [
            "--compare-rollout-dir-a",
            str(rollout_a),
            "--compare-rollout-dir-b",
            str(rollout_b),
            "--label-a",
            "success_mid_dq002",
            "--label-b",
            "failed_mid_dq001",
            "--output-dir",
            str(output_dir),
            "--formats",
            "png",
            "--dpi",
            "80",
        ]
    )

    assert exit_code == 0
    assert (output_dir / "compare_combined_analysis.png").is_file()
    assert (output_dir / "compare_summary.json").is_file()
