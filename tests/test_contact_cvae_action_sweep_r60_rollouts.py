import argparse
import json
from pathlib import Path

import pytest

from scripts.monitor_hole_random_contact_cvae_action_sweep_r60_rollouts import (
    Experiment,
    inspect,
)
from scripts.run_hole_random_contact_cvae_action_sweep_r60_rollouts import (
    DEFAULT_MODES,
    build_grid_command,
    experiment_specs,
    mode_token,
    validate_action_modes,
)


def runner_args(tmp_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        python_executable="python",
        training_root=Path("training"),
        normalization_stats=Path("stats.pt"),
        task_points_csv=Path("points.csv"),
        model_xml=Path("model.xml"),
        output_root=tmp_path,
        device="cuda",
        mujoco_gl="egl",
        rollout_seed_base=31000,
        chunk_len=10,
        force_window_len=20,
        force_window_duration=0.25,
        policy_rate_hz=30.0,
        max_rollout_steps=900,
        max_delta_q=0.02,
        force_stop_threshold=1000.0,
        success_distance_threshold=0.005,
        success_lateral_threshold=0.006,
        success_force_threshold=40.0,
        success_hold_steps=15,
        temporal_agg_decay=0.3,
        save_videos=False,
    )


def test_default_sweep_has_two_models_times_eleven_modes():
    modes = validate_action_modes(DEFAULT_MODES, chunk_len=10)
    specs = experiment_specs(modes)

    assert modes == tuple(str(index) for index in range(1, 11)) + ("temporal",)
    assert len(specs) == 22
    assert len({spec.key for spec in specs}) == 22
    assert [spec.model_key for spec in specs[:2]] == [
        "contact_cvae_zero",
        "contact_cvae_prior",
    ]
    assert specs[0].action_select_mode == "1"
    assert specs[-1].action_select_mode == "temporal"


@pytest.mark.parametrize("modes", [("0",), ("11",), ("first",), ("1", "01")])
def test_action_sweep_rejects_invalid_or_duplicate_numeric_modes(modes):
    with pytest.raises(ValueError):
        validate_action_modes(modes, chunk_len=10)


def test_grid_commands_forward_numeric_mode_temporal_decay_and_egl(tmp_path):
    args = runner_args(tmp_path)
    numeric, temporal = experiment_specs(("7", "temporal"))[0::3]

    numeric_command = build_grid_command(args, numeric)
    temporal_command = build_grid_command(args, temporal)

    assert numeric_command[numeric_command.index("--action-select-mode") + 1] == "7"
    assert temporal_command[temporal_command.index("--action-select-mode") + 1] == "temporal"
    for command in (numeric_command, temporal_command):
        assert command[command.index("--temporal-agg-decay") + 1] == "0.3"
        assert command[command.index("--mujoco-gl") + 1] == "egl"
        assert command[command.index("--rollout-seed-base") + 1] == "31000"


def test_mode_tokens_are_stable_and_do_not_alias_temporal():
    assert mode_token("1") == "action_01"
    assert mode_token("10") == "action_10"
    assert mode_token("temporal") == "temporal"


def test_monitor_inspects_nested_experiment_output(tmp_path):
    output_dir = tmp_path / "contact_cvae_zero" / "action_01"
    point_dir = output_dir / "point_001_test"
    pipeline = tmp_path / ".pipeline"
    point_dir.mkdir(parents=True)
    pipeline.mkdir()
    (point_dir / "summary.json").write_text(
        json.dumps({"success": True, "max_force_norm": 12.5, "stop_reason": "success"})
    )
    (output_dir / "grid_manifest.json").write_text(
        json.dumps(
            {
                "runs": [
                    {
                        "status": "success",
                        "start_time": "2026-07-18T10:00:00+08:00",
                        "end_time": "2026-07-18T10:00:05+08:00",
                    }
                ]
            }
        )
    )
    experiment = Experiment(
        key="contact_cvae_zero__action_01",
        model="contact_cvae_zero",
        mode="1",
        output_dir=output_dir,
    )

    result = inspect(experiment, pipeline, None, False, 40.0)

    assert result.attempted == 1
    assert result.valid == 1
    assert result.successes == 1
    assert result.safe_successes == 1
    assert result.maximum_force == 12.5
    assert result.elapsed_seconds == 5.0
