import csv
import json

from scripts.summarize_rollouts import OUTPUT_COLUMNS, collect_rollouts, write_summary_csv


def test_collect_rollouts_prefers_summary_json_and_sorts(tmp_path):
    root = tmp_path / "outputs"
    root.mkdir()
    success_run = root / "rollout_success"
    fallback_run = root / "rollout_fallback"
    success_run.mkdir()
    fallback_run.mkdir()

    summary = {
        "success": True,
        "success_step": 3,
        "success_time": 0.1,
        "stop_reason": "success",
        "contact_latent_mode": "zero",
        "action_select_mode": "mid",
        "max_delta_q": 0.02,
        "steps_executed": 4,
        "final_time": 0.1,
        "initial_peg_to_hole_dist": 0.02,
        "final_peg_to_hole_dist": 0.003,
        "min_peg_to_hole_dist": 0.003,
        "min_peg_to_hole_dist_step": 3,
        "initial_peg_to_hole_axial_error": 0.02,
        "final_peg_to_hole_axial_error": 0.002,
        "min_abs_peg_to_hole_axial_error": 0.002,
        "initial_peg_to_hole_lateral_error": 0.01,
        "final_peg_to_hole_lateral_error": 0.002,
        "min_peg_to_hole_lateral_error": 0.002,
        "min_peg_to_hole_lateral_error_step": 3,
        "max_force_norm": 12.0,
        "mean_force_norm": 4.0,
        "force_gt_5_steps": 1,
        "force_gt_20_steps": 0,
        "force_gt_40_steps": 0,
        "checkpoint": "checkpoint.pt",
        "rollout_log_csv": str(success_run / "rollout_log.csv"),
    }
    (success_run / "summary.json").write_text(json.dumps(summary))

    with (fallback_run / "rollout_log.csv").open("w", newline="") as csv_file:
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
                "success_hold_counter",
                "stop_reason",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "step": 0,
                "time": 0.0,
                "mode": "zero",
                "action_select_mode": "first",
                "peg_to_hole_dist": 0.03,
                "peg_to_hole_axial_error": 0.03,
                "peg_to_hole_lateral_error": 0.02,
                "force_norm": 1.0,
                "success_hold_counter": 0,
                "stop_reason": "",
            }
        )
        writer.writerow(
            {
                "step": 1,
                "time": 0.033,
                "mode": "zero",
                "action_select_mode": "first",
                "peg_to_hole_dist": 0.02,
                "peg_to_hole_axial_error": 0.01,
                "peg_to_hole_lateral_error": 0.015,
                "force_norm": 6.0,
                "success_hold_counter": 0,
                "stop_reason": "max_rollout_steps",
            }
        )

    rows = collect_rollouts(root, "rollout_*")

    assert [row["run"] for row in rows] == ["rollout_success", "rollout_fallback"]
    assert rows[0]["summary_json"].endswith("summary.json")
    assert rows[1]["summary_json"] == ""
    assert rows[1]["steps_executed"] == 2
    assert rows[1]["force_gt_5_steps"] == 1


def test_write_summary_csv_uses_expected_columns(tmp_path):
    output = tmp_path / "summary.csv"
    write_summary_csv(output, [{"run": "a", "success": True, "final_dist": 0.1}])

    with output.open(newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        assert reader.fieldnames == OUTPUT_COLUMNS
        rows = list(reader)

    assert rows[0]["run"] == "a"
    assert rows[0]["success"] == "True"
