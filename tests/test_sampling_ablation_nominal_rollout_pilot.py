import argparse
import json
from pathlib import Path

from scripts.run_sampling_ablation_nominal_rollout_pilot import (
    MODEL_CONTRACTS,
    build_rollout_command,
    build_specs,
    main,
    validate_completed_summary,
    validate_spec_files,
)


def _args(tmp_path: Path, *, execute: bool = False) -> argparse.Namespace:
    return argparse.Namespace(
        output_dir=tmp_path / "pilot",
        checkpoint_root=tmp_path / "checkpoints",
        device="cuda",
        max_rollout_steps=900,
        seed=0,
        save_videos=True,
        execute=execute,
    )


def _make_specs(tmp_path: Path):
    specs = build_specs(tmp_path / "checkpoints")
    for spec in specs:
        spec.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        spec.checkpoint.touch()
        (spec.checkpoint.parent / "run_metadata.json").write_text(
            json.dumps(
                {
                    "train_sampling_mode": spec.train_sampling_mode,
                    "model": {"architecture_version": spec.architecture_version},
                }
            ),
            encoding="utf-8",
        )
    return specs


def test_four_model_order_and_contracts_are_frozen(tmp_path):
    specs = build_specs(tmp_path)

    assert [spec.model_id for spec in specs] == [row[0] for row in MODEL_CONTRACTS]
    assert [spec.train_sampling_mode for spec in specs] == [
        "episodic_uniform",
        "dense_windows",
        "dense_windows",
        "episodic_uniform",
    ]


def test_plan_command_freezes_center_point_and_does_not_execute(tmp_path):
    args = _args(tmp_path)
    spec = build_specs(args.checkpoint_root)[2]
    command = build_rollout_command(args, spec)

    assert command[command.index("--task-profile") + 1] == "moving_hole_fixed_peg"
    assert command[command.index("--terminal-hold-mode") + 1] == "collector"
    assert command[command.index("--action-select-mode") + 1] == "temporal"
    assert command[command.index("--temporal-agg-decay") + 1] == "0.01"
    assert command[command.index("--contact-latent-mode") + 1] == "zero"
    assert command[command.index("--max-rollout-steps") + 1] == "900"
    assert command[command.index("--force-stop-threshold") + 1] == "40"
    assert "--hole-offset-x=0.0" in command
    assert "--hole-offset-y=0.0" in command
    assert "--hole-offset-z=0.0" in command
    assert "--record-force-metrics" in command
    assert "--save-videos" in command
    assert "--enable-axial-push" not in command
    assert "--execute-actions" not in command


def test_execute_flag_is_the_only_way_to_apply_actions(tmp_path):
    args = _args(tmp_path, execute=True)
    command = build_rollout_command(args, build_specs(args.checkpoint_root)[0])

    assert command.count("--execute-actions") == 1


def test_checkpoint_sidecars_guard_sampling_semantics(tmp_path):
    specs = _make_specs(tmp_path)
    validate_spec_files(specs)

    metadata_path = specs[0].checkpoint.parent / "run_metadata.json"
    metadata_path.write_text(
        json.dumps({"train_sampling_mode": "dense_windows"}), encoding="utf-8"
    )
    try:
        validate_spec_files(specs)
    except ValueError as error:
        assert "official_episodic train_sampling_mode" in str(error)
    else:
        raise AssertionError("mismatched sampling semantics were accepted")


def test_summary_validator_requires_terminal_hold_and_force_metrics(tmp_path):
    args = _args(tmp_path, execute=True)
    spec = build_specs(args.checkpoint_root)[2]
    summary = {
        "rollout_protocol_version": "collector_terminal_hold_rollout_v5",
        "checkpoint": str(spec.checkpoint.resolve()),
        "rollout_mode": "execute",
        "task_profile": "moving_hole_fixed_peg",
        "policy_wrench_source": "compensated",
        "terminal_hold_mode": "collector",
        "seed": 0,
        "action_mode": "joint_pos",
        "action_select_mode": "temporal",
        "temporal_agg_decay": 0.01,
        "contact_latent_mode": "zero",
        "deployment_latent_source": "zero",
        "policy_rate_hz": 30.0,
        "max_rollout_steps": 900,
        "ema_alpha": 1.0,
        "max_delta_q": 0.02,
        "force_stop_threshold": 40.0,
        "safe_force_threshold": 40.0,
        "hole_offset_x": 0.0,
        "hole_offset_y": 0.0,
        "hole_offset_z": 0.0,
        "force_metrics_recorded": True,
    }

    assert validate_completed_summary(summary, args, spec) == []
    summary["terminal_hold_mode"] = "disabled"
    assert "terminal_hold_mode" in validate_completed_summary(summary, args, spec)[0]


def test_main_defaults_to_plan_only_without_creating_output(tmp_path, capsys):
    _make_specs(tmp_path)
    output_dir = tmp_path / "planned-output"

    result = main(
        [
            "--checkpoint-root",
            str(tmp_path / "checkpoints"),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert result == 0
    assert not output_dir.exists()
    assert "plan_only=true" in capsys.readouterr().out
