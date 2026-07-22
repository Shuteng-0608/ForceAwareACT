import json
import uuid
from pathlib import Path

import pytest

from force_aware_act.data.manifest import (
    MANIFEST_SCHEMA_VERSION,
    DatasetManifest,
    EpisodeIdentity,
    EpisodeManifestEntry,
    canonical_json_bytes,
    canonical_json_sha256,
    sha256_file,
    validate_disjoint_splits,
    validate_normalization_population,
    validate_stage_population,
)


def _uuid(index: int) -> str:
    return str(uuid.UUID(int=index + 1))


def _fake_identity(tmp_path: Path, index: int) -> EpisodeIdentity:
    return EpisodeIdentity(
        episode_uuid=_uuid(index),
        path=(tmp_path / f"episode_{index:03d}.hdf5").resolve(),
        file_sha256=f"{index + 1:064x}",
    )


def test_canonical_json_hash_is_independent_of_object_key_order():
    left = {"z": [3, 2, 1], "a": {"beta": "力", "alpha": 1}}
    right = {"a": {"alpha": 1, "beta": "力"}, "z": [3, 2, 1]}

    assert canonical_json_bytes(left) == canonical_json_bytes(right)
    assert canonical_json_sha256(left) == canonical_json_sha256(right)
    assert b" " not in canonical_json_bytes(left)


def test_canonical_json_rejects_nonfinite_values():
    with pytest.raises(ValueError, match="non-finite"):
        canonical_json_bytes({"bad": float("nan")})


def test_canonical_json_rejects_non_string_object_keys():
    with pytest.raises(TypeError, match="keys must be strings"):
        canonical_json_bytes({1: "ambiguous with the string key '1'"})


def test_episode_identity_hashes_and_verifies_an_existing_file(tmp_path):
    episode = tmp_path / "episode.hdf5"
    episode.write_bytes(b"auditable episode bytes")

    identity = EpisodeIdentity.from_path(episode, _uuid(0))

    assert identity.path == episode.resolve()
    assert identity.file_sha256 == sha256_file(episode, chunk_size=3)
    assert EpisodeIdentity.from_dict(identity.to_dict(), verify_file=True) == identity
    assert len(identity.content_sha256) == 64


def test_episode_identity_rejects_noncanonical_uuid_and_sha(tmp_path):
    path = (tmp_path / "episode.hdf5").resolve()
    with pytest.raises(ValueError, match="canonical lowercase"):
        EpisodeIdentity("AAAAAAAA-AAAA-4AAA-8AAA-AAAAAAAAAAAA", path, "a" * 64)
    with pytest.raises(ValueError, match="64 lowercase hexadecimal"):
        EpisodeIdentity(_uuid(0), path, "A" * 64)


def test_episode_identity_detects_file_tampering(tmp_path):
    episode = tmp_path / "episode.hdf5"
    episode.write_bytes(b"before")
    identity = EpisodeIdentity.from_path(episode, _uuid(0))
    episode.write_bytes(b"after")

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        EpisodeIdentity.from_dict(identity.to_dict(), verify_file=True)


@pytest.mark.parametrize("overlap_field", ["episode_uuid", "path", "file_sha256"])
def test_disjoint_splits_detect_every_identity_axis(tmp_path, overlap_field):
    train = _fake_identity(tmp_path, 0)
    val_values = _fake_identity(tmp_path, 1).to_dict()
    val_values[overlap_field] = train.to_dict()[overlap_field]
    val = EpisodeIdentity.from_dict(val_values)

    with pytest.raises(ValueError, match=f"identity_field='{overlap_field}'"):
        validate_disjoint_splits({"train": [train], "val": [val], "test": []})


def test_disjoint_splits_reject_duplicate_inside_one_population(tmp_path):
    identity = _fake_identity(tmp_path, 0)

    with pytest.raises(ValueError, match="train population contains duplicate episode_uuid"):
        validate_disjoint_splits({"train": [identity, identity]})


def test_normalization_population_requires_exact_train_union(tmp_path):
    train = [_fake_identity(tmp_path, 0), _fake_identity(tmp_path, 1)]
    val = [_fake_identity(tmp_path, 2)]
    test = [_fake_identity(tmp_path, 3)]
    splits = {"train": train, "val": val, "test": test}

    validate_normalization_population(reversed(train), splits)

    with pytest.raises(ValueError, match="exactly equal the train union"):
        validate_normalization_population(train[:1], splits)
    with pytest.raises(ValueError, match="exactly equal the train union"):
        validate_normalization_population([*train, val[0]], splits)


def test_stage_population_can_be_train_union_subset_but_not_outside(tmp_path):
    train = [_fake_identity(tmp_path, 0), _fake_identity(tmp_path, 1)]
    validation = _fake_identity(tmp_path, 2)

    validate_stage_population(
        [train[0]],
        train,
        evaluation_populations={"r60_val": [validation]},
    )

    with pytest.raises(ValueError, match="not a subset"):
        validate_stage_population([validation], train)


def test_stage_validation_detects_copied_normalization_file(tmp_path):
    train = [_fake_identity(tmp_path, 0), _fake_identity(tmp_path, 1)]
    copied_values = _fake_identity(tmp_path, 2).to_dict()
    copied_values["file_sha256"] = train[1].file_sha256
    copied = EpisodeIdentity.from_dict(copied_values)

    with pytest.raises(ValueError, match="identity_field='file_sha256'"):
        validate_stage_population(
            [train[0]],
            train,
            evaluation_populations={"r2_test": [copied]},
        )


def test_dataset_manifest_round_trip_and_content_hash(tmp_path):
    train = _fake_identity(tmp_path, 0)
    val = _fake_identity(tmp_path, 1)
    manifest = DatasetManifest(
        episodes=(
            EpisodeManifestEntry(
                train,
                domain="r60",
                split="train",
                metadata={"requested_offset_mm": [10.0, -5.0]},
            ),
            EpisodeManifestEntry(val, domain="r2_contact", split="val"),
        ),
        metadata={"protocol": "staged_visual_force_v1"},
    )

    restored = DatasetManifest.from_dict(manifest.to_dict())

    assert restored == manifest
    assert restored.content_sha256 == manifest.content_sha256
    assert restored.split_populations()["train"] == (train,)
    assert restored.split_populations()["test"] == ()


def test_dataset_manifest_load_resolves_relative_paths_and_verifies_sha(tmp_path):
    episode = tmp_path / "data" / "episode.hdf5"
    episode.parent.mkdir()
    episode.write_bytes(b"episode")
    payload = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "episodes": [
            {
                "episode_uuid": _uuid(0),
                "path": "data/episode.hdf5",
                "file_sha256": sha256_file(episode),
                "domain": "r60",
                "split": "train",
                "metadata": {},
            }
        ],
        "metadata": {},
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    manifest = DatasetManifest.load(manifest_path)

    assert manifest.episodes[0].identity.path == episode.resolve()


def test_dataset_manifest_rejects_unknown_fields_and_nonfinite_metadata(tmp_path):
    identity = _fake_identity(tmp_path, 0)
    payload = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "episodes": [
            {
                **identity.to_dict(),
                "domain": "r60",
                "split": "train",
                "metadata": {},
                "splti": "typo",
            }
        ],
    }
    with pytest.raises(ValueError, match="unexpected fields"):
        DatasetManifest.from_dict(payload)
    with pytest.raises(ValueError, match="finite JSON values"):
        EpisodeManifestEntry(identity, "r60", "train", {"bad": float("inf")})


def test_dataset_manifest_load_rejects_duplicate_json_keys(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        '{"schema_version":1,"schema_version":1,"episodes":[]}',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate JSON key"):
        DatasetManifest.load(manifest_path, verify_files=False)
