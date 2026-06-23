import csv

import numpy as np
import pytest

from scripts.analyze_contact_stage import (
    analyze_contact_arrays,
    analyze_rollout_csv,
)


def test_analyze_contact_arrays_detects_contact_and_lateral_improvement():
    steps = np.arange(5, dtype=np.float64)
    times = steps * 0.1
    force_norm = np.asarray([0.0, 2.0, 21.0, 30.0, 25.0])
    axial = np.asarray([0.06, 0.04, 0.025, 0.024, 0.023])
    lateral = np.asarray([0.02, 0.018, 0.012, 0.010, 0.009])
    dist = np.sqrt(axial**2 + lateral**2)
    target_delta = np.asarray([0.0, 0.01, 0.02, 0.03, 0.04])
    applied_delta = np.asarray([0.0, 0.008, 0.015, 0.02, 0.025])

    summary = analyze_contact_arrays(
        source="synthetic",
        mode="rollout",
        steps=steps,
        times=times,
        dist=dist,
        axial=axial,
        lateral=lateral,
        force_norm=force_norm,
        hole_entrance_offset=0.024,
        force_thresholds=(5.0, 10.0, 20.0, 50.0),
        target_delta_norm=target_delta,
        applied_delta_norm=applied_delta,
    )

    assert summary["first_force_gt_20_step"] == pytest.approx(2.0)
    assert summary["first_force_gt_20_time"] == pytest.approx(0.2)
    assert summary["dist_at_force20"] == pytest.approx(dist[2])
    assert summary["axial_at_force20"] == pytest.approx(0.025)
    assert summary["lateral_at_force20"] == pytest.approx(0.012)
    assert summary["max_force"] == pytest.approx(30.0)
    assert summary["force_rises_near_entrance"] is True
    assert summary["lateral_error_decreases_after_contact"] is True
    assert summary["mean_target_delta_after_contact"] == pytest.approx(np.mean(target_delta[2:]))
    assert summary["mean_applied_delta_after_contact"] == pytest.approx(np.mean(applied_delta[2:]))


def test_analyze_rollout_csv_computes_command_deltas(tmp_path):
    path = tmp_path / "rollout_log.csv"
    fieldnames = [
        "step",
        "time",
        "peg_to_hole_dist",
        "peg_to_hole_axial_error",
        "peg_to_hole_lateral_error",
        "force_norm",
        *(f"target_ctrl_{index}" for index in range(7)),
        *(f"applied_ctrl_{index}" for index in range(7)),
        *(f"current_qpos_{index}" for index in range(7)),
    ]
    rows = []
    for step, force in enumerate((0.0, 6.0, 22.0)):
        row = {
            "step": step,
            "time": step * 0.1,
            "peg_to_hole_dist": 0.05 - 0.01 * step,
            "peg_to_hole_axial_error": 0.04 - 0.008 * step,
            "peg_to_hole_lateral_error": 0.015 - 0.003 * step,
            "force_norm": force,
        }
        for index in range(7):
            row[f"current_qpos_{index}"] = 1.0
            row[f"target_ctrl_{index}"] = 1.1
            row[f"applied_ctrl_{index}"] = 1.05
        rows.append(row)

    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = analyze_rollout_csv(
        path,
        hole_entrance_offset=0.024,
        force_thresholds=(5.0, 10.0, 20.0, 50.0),
    )

    assert summary["first_force_gt_5_step"] == pytest.approx(1.0)
    assert summary["first_force_gt_20_step"] == pytest.approx(2.0)
    assert summary["mean_target_delta_after_contact"] == pytest.approx(np.sqrt(7) * 0.1)
    assert summary["mean_applied_delta_after_contact"] == pytest.approx(np.sqrt(7) * 0.05)
