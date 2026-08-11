from pathlib import Path
from unittest.mock import patch

from scripts.build_scripted50_manifest import main


def _manifest():
    return {
        "dataset_fingerprint": "fingerprint",
        "all_episode_count": 50,
        "selected_episode_count": 50,
        "train_episode_count": 40,
        "validation_episode_count": 10,
        "holdout_episode_count": 0,
        "selection": {
            "algorithm": "chronological_stratified_validation_all50_v1",
            "success_status": "scripted_replay_success",
            "collection_method": "scripted_two_stage_replay",
            "audit": {
                "summaries": {
                    "train": {"total_timesteps": 30000},
                    "validation": {"total_timesteps": 7500},
                },
                "standardized_mean_rms": {
                    "selected_vs_all": 0.0,
                    "holdout_vs_all": None,
                    "train_vs_selected": 0.1,
                    "validation_vs_selected": 0.2,
                },
            },
        },
    }


def test_cli_builds_writes_and_reports_scripted50_manifest(tmp_path, capsys):
    data_root = tmp_path / "data"
    output = tmp_path / "manifest.json"
    manifest = _manifest()

    with (
        patch(
            "scripts.build_scripted50_manifest."
            "build_scripted50_train_validation_manifest",
            return_value=manifest,
        ) as build,
        patch(
            "scripts.build_scripted50_manifest."
            "write_paired_episode_subset_manifest"
        ) as write,
    ):
        exit_code = main(
            [
                str(data_root),
                "--output",
                str(output),
                "--seed",
                "7",
            ]
        )

    assert exit_code == 0
    build.assert_called_once_with(Path(data_root), seed=7)
    write.assert_called_once_with(manifest, output)
    report = capsys.readouterr().out
    assert '"train_episodes": 40' in report
    assert '"validation_episodes": 10' in report
    assert '"holdout_episodes": 0' in report
    assert '"success_status": "scripted_replay_success"' in report
