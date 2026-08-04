import json
from dataclasses import replace
from unittest.mock import patch

import pytest

from force_aware_act.act_aligned_training import EpisodeRecord
from force_aware_act.experiments.paired_episode_subset import (
    EpisodeSelectionFeatures,
    LoadedPairedEpisodeSubset,
    load_paired_episode_subset_manifest,
    select_paired_episode_subset,
    validate_checkpoint_experiment_provenance,
)


def _records_and_features(count: int = 20):
    records = []
    features = []
    for index in range(count):
        episode_id = f"20260701_{index:06d}_teleop"
        num_steps = 200 + index * 3
        records.append(
            EpisodeRecord(
                episode_id=episode_id,
                relative_hdf5_path=f"{episode_id}/episode.hdf5",
                num_steps=num_steps,
                camera_names=("ee_cam", "base_top_cam"),
            )
        )
        features.append(
            EpisodeSelectionFeatures(
                episode_id=episode_id,
                num_steps=num_steps,
                duration_sim=7.0 + index * 0.2,
                action_step_delta_mean=0.01 + (index % 4) * 0.002,
                action_range_norm=1.0 + (index % 5) * 0.1,
                force_norm_mean=1.0 + index * 0.1,
                force_norm_p95=5.0 + (index % 7),
                force_norm_max=10.0 + index,
                torque_norm_p95=0.2 + (index % 3) * 0.1,
                contact_onset_fraction_5n=0.1 + (index % 6) * 0.05,
                contact_fraction_5n=0.2 + (index % 4) * 0.1,
                contact_fraction_20n=(index % 5) * 0.02,
            )
        )
    return tuple(records), tuple(features)


def test_selection_is_deterministic_disjoint_and_stratified():
    records, features = _records_and_features()

    first = select_paired_episode_subset(
        records,
        features,
        temporal_strata=4,
        selected_per_stratum=2,
        validation_per_stratum=1,
        seed=7,
    )
    second = select_paired_episode_subset(
        records,
        features,
        temporal_strata=4,
        selected_per_stratum=2,
        validation_per_stratum=1,
        seed=7,
    )

    assert first == second
    groups = first["groups"]
    assert len(groups["selected"]) == 8
    assert len(groups["train"]) == 4
    assert len(groups["validation"]) == 4
    assert len(groups["holdout"]) == 12
    assert set(groups["train"]).isdisjoint(groups["validation"])
    assert set(groups["train"]) | set(groups["validation"]) == set(
        groups["selected"]
    )
    assert set(groups["selected"]).isdisjoint(groups["holdout"])
    for stratum in first["strata"]:
        assert len(stratum["all_episode_ids"]) == 5
        assert len(stratum["selected_episode_ids"]) == 2
        assert len(stratum["train_episode_ids"]) == 1
        assert len(stratum["validation_episode_ids"]) == 1


def test_selection_audit_records_counts_and_finite_balance():
    records, features = _records_and_features()

    selection = select_paired_episode_subset(
        records,
        features,
        temporal_strata=4,
        selected_per_stratum=2,
        validation_per_stratum=1,
    )

    audit = selection["audit"]
    assert audit["summaries"]["all"]["episode_count"] == 20
    assert audit["summaries"]["selected"]["episode_count"] == 8
    assert audit["summaries"]["train"]["total_timesteps"] > 0
    assert all(
        value >= 0.0
        for value in audit["standardized_mean_rms"].values()
    )


def test_selection_rejects_mismatched_feature_identity():
    records, features = _records_and_features()
    invalid = (replace(features[0], episode_id="missing"), *features[1:])

    with pytest.raises(ValueError, match="identifiers must match"):
        select_paired_episode_subset(
            records,
            invalid,
            temporal_strata=4,
            selected_per_stratum=2,
            validation_per_stratum=1,
        )


def test_checkpoint_provenance_locks_the_exact_partition(tmp_path):
    records, _features = _records_and_features(100)
    subset = LoadedPairedEpisodeSubset(
        path=tmp_path / "manifest.json",
        dataset_fingerprint="abc123",
        algorithm="test_algorithm",
        seed=0,
        train_episodes=records[:40],
        validation_episodes=records[40:50],
        holdout_episodes=records[50:],
    )
    provenance = subset.checkpoint_provenance()

    validate_checkpoint_experiment_provenance(provenance, subset)
    mismatched = dict(provenance)
    mismatched["dataset_fingerprint"] = "different"
    with pytest.raises(ValueError, match="dataset_fingerprint"):
        validate_checkpoint_experiment_provenance(mismatched, subset)
    with pytest.raises(ValueError, match="does not record"):
        validate_checkpoint_experiment_provenance(None, subset)


def test_manifest_loader_returns_typed_partition(tmp_path):
    records, _features = _records_and_features(100)
    serialized = [
        {
            "episode_id": item.episode_id,
            "relative_hdf5_path": item.relative_hdf5_path,
            "num_steps": item.num_steps,
            "camera_names": list(item.camera_names),
        }
        for item in records
    ]
    manifest = {
        "dataset_fingerprint": "abc123",
        "selection": {"algorithm": "test_algorithm", "seed": 0},
        "train_episodes": serialized[:40],
        "validation_episodes": serialized[40:50],
        "holdout_episodes": serialized[50:],
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    with patch(
        "force_aware_act.experiments.paired_episode_subset."
        "validate_paired_episode_subset_manifest"
    ):
        loaded = load_paired_episode_subset_manifest(
            path,
            data_root=tmp_path,
        )

    assert isinstance(loaded, LoadedPairedEpisodeSubset)
    assert len(loaded.train_episodes) == 40
    assert len(loaded.validation_episodes) == 10
    assert len(loaded.holdout_episodes) == 50
