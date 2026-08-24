from pathlib import Path
from unittest.mock import patch

from scripts.build_scripted200_manifest import main


def _manifest():
    return {
        "dataset_fingerprint": "fingerprint200",
        "all_episode_count": 200,
        "selected_episode_count": 100,
        "train_episode_count": 80,
        "validation_episode_count": 20,
        "holdout_episode_count": 100,
        "selection": {
            "algorithm": "scripted200_balanced_paired_train80_val20_holdout100_v1",
            "success_status": "scripted_replay_success",
            "collection_method": "scripted_two_stage_replay",
            "audit": {
                "summaries": {
                    "train": {"total_timesteps": 50000},
                    "validation": {"total_timesteps": 12500},
                    "holdout": {"total_timesteps": 62500},
                },
                "standardized_mean_rms": {
                    "selected_vs_all": 0.01,
                    "holdout_vs_all": 0.01,
                    "train_vs_selected": 0.02,
                    "validation_vs_selected": 0.03,
                },
            },
        },
    }


def test_cli_builds_writes_and_reports_scripted200_manifest(tmp_path, capsys):
    data_root = tmp_path / "data"
    output = tmp_path / "manifest.json"
    manifest = _manifest()

    with (
        patch(
            "scripts.build_scripted200_manifest."
            "build_scripted200_balanced_paired_manifest",
            return_value=manifest,
        ) as build,
        patch(
            "scripts.build_scripted200_manifest."
            "write_paired_episode_subset_manifest"
        ) as write,
    ):
        exit_code = main(
            [
                str(data_root),
                "--output",
                str(output),
                "--seed",
                "11",
            ]
        )

    assert exit_code == 0
    build.assert_called_once_with(Path(data_root), seed=11)
    write.assert_called_once_with(manifest, output)
    report = capsys.readouterr().out
    assert '"train_episodes": 80' in report
    assert '"validation_episodes": 20' in report
    assert '"holdout_episodes": 100' in report
    assert '"holdout_timesteps": 62500' in report
