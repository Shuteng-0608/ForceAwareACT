import json

import pandas as pd
import pytest

from scripts.run_xz_multiseed_rollout_suite import (
    aggregate_seed_summaries,
    build_parser,
    build_seed_command,
    build_suite_plan,
    collect_seed_summaries,
    configuration_output_base,
    seed_configurations,
    seed_suite_complete,
    seed_output_base,
    validate_args,
    wilson_interval,
    write_summaries,
    write_suite_plan,
)
from scripts.run_xz_rollout_suite import output_dir_for, selected_models


def _args(tmp_path, *extra):
    return build_parser().parse_args(
        [
            "--seeds",
            "101",
            "102",
            "--models",
            "contact_cvae_prior",
            "--action-select-modes",
            "mid",
            "--output-base",
            str(tmp_path / "multi"),
            *extra,
        ]
    )


def _write_result(
    args,
    seed,
    successes,
    safe_successes,
    rollout_seed_base=None,
):
    model = selected_models(["contact_cvae_prior"])[0]
    rollout_seed = seed if rollout_seed_base is None else rollout_seed_base
    root = output_dir_for(
        model,
        "mid",
        configuration_output_base(args, seed, rollout_seed),
        num_points=args.num_points,
        offset_mm=args.offset_mm,
        max_rollout_steps=args.max_rollout_steps,
        max_delta_q=args.max_delta_q,
    )
    root.mkdir(parents=True)
    rows = []
    for index in range(1, args.num_points + 1):
        rows.append(
            {
                "point_index": index,
                "hole_offset_x": index / 100000.0,
                "hole_offset_y": 0.0,
                "hole_offset_z": -index / 100000.0,
                "success": index <= successes,
                "safe_success": index <= safe_successes,
            }
        )
    pd.DataFrame(rows).to_csv(root / "grid_summary.csv", index=False)
    (root / "random_position_summary.json").write_text(
        json.dumps({"process_error_runs": 0})
    )


def test_seed_command_uses_isolated_output_and_protocol(tmp_path):
    args = _args(tmp_path, "--offset-mm", "6", "--max-rollout-steps", "700")

    command = build_seed_command(args, 101)

    assert command[command.index("--base-seed") + 1] == "101"
    assert command[command.index("--point-set-seed") + 1] == "101"
    assert command[command.index("--rollout-seed-base") + 1] == "101"
    assert command[command.index("--offset-mm") + 1] == "6.0"
    assert command[command.index("--max-rollout-steps") + 1] == "700"
    assert command[command.index("--output-base") + 1].endswith("multi/seed_101")


def test_separated_seed_cross_product_uses_isolated_output_paths(tmp_path):
    args = build_parser().parse_args(
        [
            "--point-set-seeds",
            "101",
            "102",
            "--rollout-seed-bases",
            "500",
            "600",
            "--models",
            "contact_cvae_prior",
            "--output-base",
            str(tmp_path / "multi"),
        ]
    )

    assert seed_configurations(args) == [
        (101, 500),
        (101, 600),
        (102, 500),
        (102, 600),
    ]
    output = configuration_output_base(args, 101, 500)
    assert output == tmp_path / "multi" / "pointset_101" / "rollout_500"
    command = build_seed_command(args, 101, 500)
    assert command[command.index("--point-set-seed") + 1] == "101"
    assert command[command.index("--rollout-seed-base") + 1] == "500"
    assert command[command.index("--output-base") + 1] == str(output)


def test_suite_plan_records_seed_dimensions_and_protocol(tmp_path):
    args = build_parser().parse_args(
        [
            "--point-set-seeds",
            "101",
            "102",
            "--rollout-seed-bases",
            "500",
            "600",
            "--models",
            "act_baseline",
            "contact_cvae",
            "--output-base",
            str(tmp_path / "multi"),
        ]
    )

    plan = build_suite_plan(args)
    path = write_suite_plan(args)
    written = json.loads(path.read_text())

    assert len(plan["seed_configurations"]) == 4
    assert plan["models"] == ["contact_cvae", "act_baseline"]
    assert written["point_set_seeds"] == [101, 102]
    assert written["rollout_seed_bases"] == [500, 600]
    assert path == tmp_path / "multi" / "suite_plan.json"


def test_default_action_select_mode_is_mid_only(tmp_path):
    args = build_parser().parse_args(
        ["--seeds", "101", "102", "--output-base", str(tmp_path)]
    )

    assert args.action_select_modes == ["mid"]


def test_duplicate_seeds_are_rejected(tmp_path):
    args = build_parser().parse_args(
        ["--seeds", "101", "101", "--output-base", str(tmp_path)]
    )

    with pytest.raises(ValueError, match="duplicates"):
        validate_args(args)


def test_dry_run_and_aggregate_only_are_mutually_exclusive(tmp_path):
    args = _args(tmp_path, "--dry-run", "--aggregate-only")

    with pytest.raises(ValueError, match="cannot"):
        validate_args(args)


def test_collect_and_aggregate_safe_success(tmp_path):
    args = _args(tmp_path)
    _write_result(args, 101, successes=30, safe_successes=20)
    _write_result(args, 102, successes=40, safe_successes=30)

    per_seed = collect_seed_summaries(args)
    aggregate = aggregate_seed_summaries(per_seed)
    row = aggregate.iloc[0]

    assert len(per_seed) == 2
    assert row["available_seeds"] == 2
    assert row["total_completed_points"] == 100
    assert row["task_successes"] == 70
    assert row["safe_successes"] == 50
    assert row["safe_success_rate_pooled"] == pytest.approx(0.5)
    assert row["safe_success_rate_seed_mean"] == pytest.approx(0.5)
    assert row["safe_success_rate_seed_std"] == pytest.approx(0.02**0.5)
    assert bool(row["all_point_sets_match_within_seed"])


def test_collect_and_aggregate_separated_seed_configurations(tmp_path):
    args = build_parser().parse_args(
        [
            "--point-set-seeds",
            "101",
            "102",
            "--rollout-seed-bases",
            "500",
            "600",
            "--models",
            "contact_cvae_prior",
            "--action-select-modes",
            "mid",
            "--output-base",
            str(tmp_path / "multi"),
        ]
    )
    for point_seed, rollout_seed in seed_configurations(args):
        _write_result(
            args,
            point_seed,
            successes=30,
            safe_successes=20,
            rollout_seed_base=rollout_seed,
        )

    per_seed = collect_seed_summaries(args)
    aggregate = aggregate_seed_summaries(per_seed)
    row = aggregate.iloc[0]

    assert len(per_seed) == 4
    assert set(per_seed["point_set_seed"]) == {101, 102}
    assert set(per_seed["rollout_seed_base"]) == {500, 600}
    assert row["available_point_set_seeds"] == 2
    assert row["available_rollout_seed_bases"] == 2
    assert row["available_seed_configurations"] == 4
    assert row["total_completed_points"] == 200


def test_missing_seed_is_retained_in_per_seed_inventory(tmp_path):
    args = _args(tmp_path)
    _write_result(args, 101, successes=10, safe_successes=8)

    per_seed = collect_seed_summaries(args)

    assert per_seed["status"].tolist() == ["complete", "missing"]


def test_complete_seed_suite_can_be_skipped(tmp_path):
    args = _args(tmp_path)
    _write_result(args, 101, successes=10, safe_successes=8)

    assert seed_suite_complete(args, 101)
    assert not seed_suite_complete(args, 102)


def test_incomplete_seed_suite_is_not_skipped(tmp_path):
    args = _args(tmp_path)
    _write_result(args, 101, successes=10, safe_successes=8)
    model = selected_models(["contact_cvae_prior"])[0]
    root = output_dir_for(
        model,
        "mid",
        seed_output_base(args.output_base, 101),
        num_points=args.num_points,
        offset_mm=args.offset_mm,
        max_rollout_steps=args.max_rollout_steps,
        max_delta_q=args.max_delta_q,
    )
    pd.read_csv(root / "grid_summary.csv").head(args.num_points - 1).to_csv(
        root / "grid_summary.csv",
        index=False,
    )

    assert not seed_suite_complete(args, 101)


def test_write_summaries_creates_csv_files(tmp_path):
    args = _args(tmp_path)
    _write_result(args, 101, successes=10, safe_successes=8)
    _write_result(args, 102, successes=12, safe_successes=9)
    per_seed = collect_seed_summaries(args)
    aggregate = aggregate_seed_summaries(per_seed)

    per_seed_path, aggregate_path = write_summaries(args.output_base, per_seed, aggregate)

    assert per_seed_path.is_file()
    assert aggregate_path.is_file()


def test_wilson_interval_contains_observed_rate():
    lower, upper = wilson_interval(25, 50)

    assert lower < 0.5 < upper
