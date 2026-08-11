import json
from dataclasses import asdict, replace
from unittest.mock import patch

import pytest

from force_aware_act.act_aligned_training import EpisodeRecord
from force_aware_act.experiments.paired_episode_subset import (
    PAIRED_EPISODE_SUBSET_VERSION,
    SCRIPTED_COLLECTION_METHOD,
    SCRIPTED_FIXED50_SELECTION_ALGORITHM,
    SCRIPTED_REPLAY_SUCCESS_STATUS,
    EpisodeSelectionFeatures,
    LoadedPairedEpisodeSubset,
    build_scripted50_train_validation_manifest,
    extract_episode_selection_features,
    load_paired_episode_subset_manifest,
    select_all_episode_train_validation_split,
    select_paired_episode_subset,
    validate_checkpoint_experiment_provenance,
    validate_paired_episode_subset_manifest,
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


def _manifest_from_selection(records, features, selection):
    records_by_id = {record.episode_id: record for record in records}
    groups = selection["groups"]

    def serialized(group):
        return [asdict(records_by_id[item]) for item in sorted(groups[group])]

    return {
        "format_version": PAIRED_EPISODE_SUBSET_VERSION,
        "data_root_hint": "synthetic",
        "dataset_fingerprint": "synthetic-fingerprint",
        "selection": {
            key: value
            for key, value in selection.items()
            if key not in {"format_version", "groups"}
        },
        "all_episode_count": len(groups["all"]),
        "selected_episode_count": len(groups["selected"]),
        "train_episode_count": len(groups["train"]),
        "validation_episode_count": len(groups["validation"]),
        "holdout_episode_count": len(groups["holdout"]),
        "train_episodes": serialized("train"),
        "validation_episodes": serialized("validation"),
        "holdout_episodes": serialized("holdout"),
        "selected_episodes": serialized("selected"),
        "episode_features": {
            item.episode_id: item.to_dict() for item in features
        },
    }


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


def test_all50_selection_is_deterministic_stratified_and_has_no_holdout():
    records, features = _records_and_features(50)

    first = select_all_episode_train_validation_split(
        records,
        features,
        seed=17,
    )
    second = select_all_episode_train_validation_split(
        records,
        features,
        seed=17,
    )

    assert first == second
    assert first["algorithm"] == SCRIPTED_FIXED50_SELECTION_ALGORITHM
    groups = first["groups"]
    assert len(groups["all"]) == 50
    assert len(groups["selected"]) == 50
    assert len(groups["train"]) == 40
    assert len(groups["validation"]) == 10
    assert groups["holdout"] == []
    assert first["audit"]["summaries"]["holdout"] == {
        "episode_count": 0,
        "total_timesteps": 0,
        "features": {},
    }
    assert first["audit"]["standardized_mean_rms"]["holdout_vs_all"] is None
    for stratum in first["strata"]:
        assert len(stratum["all_episode_ids"]) == 5
        assert len(stratum["selected_episode_ids"]) == 5
        assert len(stratum["train_episode_ids"]) == 4
        assert len(stratum["validation_episode_ids"]) == 1
        assert stratum["holdout_episode_ids"] == []


def test_scripted50_manifest_protocol_is_strictly_validated():
    records, features = _records_and_features(50)
    selection = select_all_episode_train_validation_split(records, features)
    selection["success_status"] = SCRIPTED_REPLAY_SUCCESS_STATUS
    selection["collection_method"] = SCRIPTED_COLLECTION_METHOD
    manifest = _manifest_from_selection(records, features, selection)

    validate_paired_episode_subset_manifest(manifest)

    invalid = json.loads(json.dumps(manifest))
    invalid["selection"]["success_status"] = "different"
    with pytest.raises(ValueError, match="success_status"):
        validate_paired_episode_subset_manifest(invalid)


def test_scripted50_manifest_builder_locks_collection_contract(tmp_path):
    records, features = _records_and_features(50)

    with (
        patch(
            "force_aware_act.experiments.paired_episode_subset.discover_episodes",
            return_value=records,
        ),
        patch(
            "force_aware_act.experiments.paired_episode_subset."
            "extract_episode_selection_features",
            return_value=features,
        ) as extract,
        patch(
            "force_aware_act.experiments.paired_episode_subset."
            "validate_paired_episode_subset_manifest"
        ) as validate,
    ):
        manifest = build_scripted50_train_validation_manifest(tmp_path, seed=9)

    extract.assert_called_once_with(
        tmp_path,
        records,
        expected_success_status=SCRIPTED_REPLAY_SUCCESS_STATUS,
        expected_collection_method=SCRIPTED_COLLECTION_METHOD,
    )
    assert manifest["train_episode_count"] == 40
    assert manifest["validation_episode_count"] == 10
    assert manifest["holdout_episode_count"] == 0
    assert manifest["selection"]["seed"] == 9
    assert manifest["selection"]["success_status"] == (
        SCRIPTED_REPLAY_SUCCESS_STATUS
    )
    validate.assert_called_once_with(manifest, data_root=tmp_path)


def test_legacy_paired50_manifest_protocol_remains_valid():
    records, features = _records_and_features(100)
    selection = select_paired_episode_subset(records, features)
    manifest = _manifest_from_selection(records, features, selection)

    validate_paired_episode_subset_manifest(manifest)


def test_feature_extraction_reports_success_status_mismatch_before_hdf5(tmp_path):
    episode_dir = tmp_path / "episode_001"
    episode_dir.mkdir()
    (episode_dir / "metadata.json").write_text(
        json.dumps({"status": SCRIPTED_REPLAY_SUCCESS_STATUS}),
        encoding="utf-8",
    )
    record = EpisodeRecord(
        episode_id="episode_001",
        relative_hdf5_path="episode_001/episode.hdf5",
        num_steps=10,
        camera_names=("ee_cam", "base_top_cam"),
    )

    with pytest.raises(ValueError, match="status mismatch"):
        extract_episode_selection_features(tmp_path, (record,))


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


def test_scripted_checkpoint_provenance_locks_collection_semantics(tmp_path):
    records, _features = _records_and_features(50)
    subset = LoadedPairedEpisodeSubset(
        path=tmp_path / "manifest.json",
        dataset_fingerprint="scripted123",
        algorithm=SCRIPTED_FIXED50_SELECTION_ALGORITHM,
        seed=0,
        train_episodes=records[:40],
        validation_episodes=records[40:],
        holdout_episodes=(),
        success_status=SCRIPTED_REPLAY_SUCCESS_STATUS,
        collection_method=SCRIPTED_COLLECTION_METHOD,
    )
    provenance = subset.checkpoint_provenance()

    validate_checkpoint_experiment_provenance(provenance, subset)
    mismatched = dict(provenance)
    mismatched["success_status"] = "different"
    with pytest.raises(ValueError, match="success_status"):
        validate_checkpoint_experiment_provenance(mismatched, subset)


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
