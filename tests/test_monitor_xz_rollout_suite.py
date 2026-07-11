import json
from pathlib import Path

import pandas as pd

from scripts.monitor_xz_rollout_suite import (
    build_parser,
    collect_progress,
    load_plan,
    render_report,
)
from scripts.run_xz_rollout_suite import output_dir_for, selected_models


def _plan(tmp_path):
    output_base = tmp_path / "suite"
    configurations = [
        {
            "point_set_seed": 101,
            "rollout_seed_base": rollout_seed,
            "output_base": str(output_base / "pointset_101" / f"rollout_{rollout_seed}"),
        }
        for rollout_seed in (500, 600, 700)
    ]
    return {
        "schema_version": 1,
        "output_base": str(output_base),
        "point_set_seeds": [101],
        "rollout_seed_bases": [500, 600, 700],
        "seed_configurations": configurations,
        "models": ["contact_cvae", "act_baseline"],
        "action_select_modes": ["mid"],
        "num_points": 3,
        "offset_mm": 4.0,
        "max_rollout_steps": 900,
        "max_delta_q": 0.02,
    }


def _root(plan, rollout_seed, model_key):
    configuration = next(
        item
        for item in plan["seed_configurations"]
        if item["rollout_seed_base"] == rollout_seed
    )
    model = selected_models([model_key])[0]
    return output_dir_for(
        model,
        "mid",
        Path(configuration["output_base"]),
        num_points=3,
        offset_mm=4.0,
        max_rollout_steps=900,
        max_delta_q=0.02,
    )


def _write_complete(root):
    root.mkdir(parents=True)
    pd.DataFrame({"point_index": [1, 2, 3]}).to_csv(
        root / "grid_summary.csv", index=False
    )


def test_monitor_reports_complete_running_and_queued_configurations(tmp_path):
    plan = _plan(tmp_path)
    for model_key in ("contact_cvae", "act_baseline"):
        _write_complete(_root(plan, 500, model_key))
    _write_complete(_root(plan, 600, "contact_cvae"))
    active_root = _root(plan, 600, "act_baseline")
    active_root.mkdir(parents=True)
    (active_root / "grid_manifest.json").write_text(
        json.dumps({"runs": [{"status": "success"}]})
    )
    process_lines = [
        f"python run_mujoco_policy_rollout.py --output-dir "
        f"{active_root}/point_002_x_p000000mm_z_p000000mm_repeat_001"
    ]

    progress = collect_progress(plan, process_lines)
    report = render_report(plan, progress, None)

    running = [item for item in progress if item.status == "running"]
    assert len(running) == 1
    assert running[0].model_key == "act_baseline"
    assert running[0].rollout_seed_base == 600
    assert running[0].current_point == 2
    assert "completed: (101,500)" in report
    assert "queued: (101,700)" in report
    assert "model=act_baseline" in report
    assert "point=2/3" in report


def test_monitor_loads_suite_plan_without_repeated_cli_protocol(tmp_path):
    plan = _plan(tmp_path)
    output_base = tmp_path / "suite"
    output_base.mkdir()
    plan_path = output_base / "suite_plan.json"
    plan_path.write_text(json.dumps(plan))
    args = build_parser().parse_args(["--output-base", str(output_base)])

    loaded, loaded_path = load_plan(args)

    assert loaded == plan
    assert loaded_path == plan_path


def test_monitor_cli_fallback_builds_cross_product_plan(tmp_path):
    args = build_parser().parse_args(
        [
            "--output-base",
            str(tmp_path / "suite"),
            "--point-set-seeds",
            "101",
            "102",
            "--rollout-seed-bases",
            "500",
            "600",
            "--models",
            "contact_cvae",
            "act_baseline",
            "--action-select-modes",
            "mid",
            "--num-points",
            "100",
            "--offset-mm",
            "4",
        ]
    )

    plan, plan_path = load_plan(args)

    assert plan_path is None
    assert len(plan["seed_configurations"]) == 4
    assert plan["models"] == ["contact_cvae", "act_baseline"]
