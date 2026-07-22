import argparse
import json
from pathlib import Path

from scripts.monitor_hole_random_5model_fibonacci_r60_rollouts import (
    inspect_model,
)
from scripts.run_hole_random_5model_fibonacci_r60_rollouts import (
    MODEL_SPECS,
    build_grid_command,
    read_fixed_points,
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
        action_select_mode="mid",
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
        save_videos=False,
    )


def test_fixed_fibonacci_file_has_expected_count_and_radius() -> None:
    points = read_fixed_points(
        Path("configs/experiments/fibonacci_disk_100_r60mm.csv")
    )
    assert len(points) == 100
    assert max((x * x + z * z) ** 0.5 for x, _, z in points) < 0.06


def test_grid_commands_use_best_checkpoints_and_deployment_modes(tmp_path) -> None:
    args = runner_args(tmp_path)
    commands = {spec.key: build_grid_command(args, spec) for spec in MODEL_SPECS}

    for spec in MODEL_SPECS:
        command = commands[spec.key]
        assert "checkpoint_best.pt" in command[command.index("--checkpoint") + 1]
        assert command[command.index("--task-points-csv") + 1] == "points.csv"
        assert command[command.index("--rollout-seed-base") + 1] == "31000"
        assert command[command.index("--action-select-mode") + 1] == "mid"
        assert command[command.index("--mujoco-gl") + 1] == "egl"

    prior = commands["contact_cvae_prior"]
    assert prior[prior.index("--contact-latent-mode") + 1] == "prior"
    for key, command in commands.items():
        if key != "contact_cvae_prior":
            assert command[command.index("--contact-latent-mode") + 1] == "zero"


def test_monitor_counts_success_safe_success_and_process_errors(tmp_path) -> None:
    output_root = tmp_path / "suite"
    model_root = output_root / "contact_cvae_zero"
    pipeline = output_root / ".pipeline"
    point_1 = model_root / "point_001_test"
    point_2 = model_root / "point_002_test"
    point_1.mkdir(parents=True)
    point_2.mkdir(parents=True)
    pipeline.mkdir(parents=True)
    (point_1 / "summary.json").write_text(
        json.dumps({"success": True, "max_force_norm": 39.0, "stop_reason": "success"})
    )
    (point_2 / "summary.json").write_text(
        json.dumps(
            {
                "success": False,
                "max_force_norm": 50.0,
                "stop_reason": "force_stop_threshold",
            }
        )
    )
    (model_root / "grid_manifest.json").write_text(
        json.dumps(
            {
                "runs": [
                    {
                        "status": "success",
                        "start_time": "2026-07-17T10:00:00+08:00",
                        "end_time": "2026-07-17T10:00:10+08:00",
                    },
                    {
                        "status": "process_error",
                        "start_time": "2026-07-17T10:00:10+08:00",
                        "end_time": "2026-07-17T10:00:15+08:00",
                    },
                ]
            }
        )
    )
    (pipeline / "contact_cvae_zero.status").write_text("running\t2026-07-17T10:00:00+08:00\n")

    result = inspect_model(
        output_root,
        pipeline,
        "contact_cvae_zero",
        "contact_cvae_zero",
        True,
        40.0,
    )

    assert result.status == "running"
    assert result.attempted == 2
    assert result.valid == 2
    assert result.successes == 1
    assert result.safe_successes == 1
    assert result.process_errors == 1
    assert result.force_stops == 1
    assert result.maximum_force == 50.0
    assert result.current_point == 3
    assert result.timed_attempts == 2
    assert result.elapsed_seconds == 15.0
