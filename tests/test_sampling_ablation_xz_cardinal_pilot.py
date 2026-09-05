import argparse
import json
from pathlib import Path

from scripts.run_sampling_ablation_xz_cardinal_pilot import (
    build_points,
    build_rollout_command,
    main,
    parse_args,
    validate_completed_summary,
)
from scripts.run_sampling_ablation_nominal_rollout_pilot import build_specs


def _args(tmp_path: Path, execute: bool = False) -> argparse.Namespace:
    return argparse.Namespace(
        output_dir=tmp_path / "pilot",
        checkpoint_root=tmp_path / "checkpoints",
        radius_mm=4.0,
        device="cuda",
        max_rollout_steps=1200,
        seed=0,
        save_videos=True,
        execute=execute,
    )


def _make_checkpoint_sidecars(root: Path) -> None:
    for spec in build_specs(root):
        spec.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        spec.checkpoint.touch()
        (spec.checkpoint.parent / "run_metadata.json").write_text(
            json.dumps(
                {
                    "train_sampling_mode": spec.train_sampling_mode,
                    "model": {"architecture_version": spec.architecture_version},
                }
            )
        )


def test_cardinal_points_are_exactly_four_paired_directions():
    points = build_points(4.0)

    assert [(p.point_id, p.offset_x_m, p.offset_z_m) for p in points] == [
        ("x_p4mm", 0.004, 0.0),
        ("x_m4mm", -0.004, 0.0),
        ("z_p4mm", 0.0, 0.004),
        ("z_m4mm", 0.0, -0.004),
    ]


def test_default_contract_uses_forty_second_lower_radius_pilot():
    args = parse_args([])

    assert args.radius_mm == 4.0
    assert args.max_rollout_steps == 1200
    assert "r4mm_h1200" in args.output_dir.name


def test_plan_command_moves_fixed_target_and_never_executes_by_default(tmp_path):
    args = _args(tmp_path)
    point = build_points(4.0)[1]
    spec = build_specs(args.checkpoint_root)[0]
    command = build_rollout_command(args, point, spec)

    assert command[command.index("--task-profile") + 1] == "moving_hole_fixed_peg"
    assert command[command.index("--terminal-hold-mode") + 1] == "collector"
    assert "--hole-offset-x=-0.004" in command
    assert "--hole-offset-y=0.0" in command
    assert "--hole-offset-z=0.0" in command
    assert command[command.index("--max-rollout-steps") + 1] == "1200"
    assert command[command.index("--force-stop-threshold") + 1] == "40"
    assert "--record-force-metrics" in command
    assert "--enable-axial-push" not in command
    assert "--execute-actions" not in command


def test_execute_flag_adds_action_permission(tmp_path):
    args = _args(tmp_path, execute=True)
    command = build_rollout_command(
        args, build_points(4.0)[0], build_specs(args.checkpoint_root)[0]
    )

    assert command.count("--execute-actions") == 1


def test_summary_validator_binds_actual_offset_target(tmp_path):
    args = _args(tmp_path, execute=True)
    point = build_points(4.0)[2]
    spec = build_specs(args.checkpoint_root)[3]
    summary = {
        "rollout_protocol_version": "collector_terminal_hold_rollout_v5",
        "checkpoint": str(spec.checkpoint.resolve()),
        "rollout_mode": "execute",
        "task_profile": "moving_hole_fixed_peg",
        "offset_target_body_name": "fixed_peg_fixture",
        "policy_wrench_source": "compensated",
        "terminal_hold_mode": "collector",
        "seed": 0,
        "action_mode": "joint_pos",
        "action_select_mode": "temporal",
        "temporal_agg_decay": 0.01,
        "contact_latent_mode": "zero",
        "deployment_latent_source": "zero",
        "policy_rate_hz": 30.0,
        "max_rollout_steps": 1200,
        "ema_alpha": 1.0,
        "max_delta_q": 0.02,
        "force_stop_threshold": 40.0,
        "safe_force_threshold": 40.0,
        "hole_offset_x": 0.0,
        "hole_offset_y": 0.0,
        "hole_offset_z": 0.004,
        "force_metrics_recorded": True,
    }

    assert validate_completed_summary(summary, args, point, spec) == []
    summary["offset_target_body_name"] = "hole_tool"
    assert "offset_target_body_name" in validate_completed_summary(
        summary, args, point, spec
    )[0]


def test_main_plan_only_does_not_create_outputs(tmp_path, capsys):
    checkpoint_root = tmp_path / "checkpoints"
    _make_checkpoint_sidecars(checkpoint_root)
    output_dir = tmp_path / "output"

    result = main(
        [
            "--checkpoint-root",
            str(checkpoint_root),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert result == 0
    assert not output_dir.exists()
    assert "plan_only=true" in capsys.readouterr().out
