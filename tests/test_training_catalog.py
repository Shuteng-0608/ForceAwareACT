import json
from dataclasses import dataclass

import pytest

from force_aware_act.training.catalog import PhaseCatalog


@dataclass
class _Index:
    episode_path: object
    state_index: int


def _write_catalog(tmp_path, *, segments=None):
    episode = tmp_path / "episode.hdf5"
    episode.touch()
    document = {
        "schema_version": 2,
        "labeler": {
            "version": "manual-v2",
            "dataset_manifest_sha256": "a" * 64,
        },
        "episodes": [
            {
                "path": "episode.hdf5",
                "domain": "r2_contact",
                "episode_uuid": "12345678-1234-5678-1234-567812345678",
                "file_sha256": "b" * 64,
                "segments": segments
                or [
                    {"start": 0, "stop": 2, "phase": "free"},
                    {"start": 2, "stop": 5, "phase": "contact"},
                ],
            }
        ],
    }
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return path, episode


def test_phase_catalog_resolves_segments_and_hashes(tmp_path):
    path, episode = _write_catalog(tmp_path)
    catalog = PhaseCatalog.load(path)

    assert catalog.phase_for(episode, 0) == "free"
    assert catalog.phase_for(episode, 4) == "contact"
    assert len(catalog.content_sha256) == 64
    assert catalog.dataset_manifest_sha256 == "a" * 64
    assert catalog.episode_for(episode).domain == "r2_contact"
    catalog.validate_indices([_Index(episode, 0), _Index(episode, 4)])


def test_phase_catalog_rejects_gap_when_dataset_uses_it(tmp_path):
    path, episode = _write_catalog(
        tmp_path,
        segments=[
            {"start": 0, "stop": 2, "phase": "free"},
            {"start": 3, "stop": 5, "phase": "contact"},
        ],
    )
    catalog = PhaseCatalog.load(path)
    with pytest.raises(KeyError, match="not covered"):
        catalog.validate_indices([_Index(episode, 2)])


def test_phase_catalog_rejects_overlapping_segments(tmp_path):
    path, _ = _write_catalog(
        tmp_path,
        segments=[
            {"start": 0, "stop": 3, "phase": "free"},
            {"start": 2, "stop": 5, "phase": "contact"},
        ],
    )
    with pytest.raises(ValueError, match="overlapping"):
        PhaseCatalog.load(path)


def test_phase_catalog_rejects_duplicate_episode_paths(tmp_path):
    path, _ = _write_catalog(tmp_path)
    document = json.loads(path.read_text(encoding="utf-8"))
    document["episodes"].append(document["episodes"][0])
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate phase catalog episode"):
        PhaseCatalog.load(path)


def test_phase_catalog_rejects_nonfinite_or_duplicate_json_keys(tmp_path):
    path = tmp_path / "catalog.json"
    path.write_text('{"schema_version":2,"schema_version":2,"episodes":[]}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON key"):
        PhaseCatalog.load(path)

    path.write_text('{"schema_version":2,"episodes":[],"labeler":{"x":NaN}}', encoding="utf-8")
    with pytest.raises(ValueError, match="non-finite"):
        PhaseCatalog.load(path)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("domain", "", "domain"),
        ("episode_uuid", "not-a-uuid", "valid UUID"),
        ("file_sha256", "ABC", "64 lowercase"),
    ],
)
def test_phase_catalog_rejects_invalid_episode_identity(tmp_path, field, value, message):
    path, _ = _write_catalog(tmp_path)
    document = json.loads(path.read_text(encoding="utf-8"))
    document["episodes"][0][field] = value
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises((TypeError, ValueError), match=message):
        PhaseCatalog.load(path)


def test_phase_catalog_requires_manifest_binding(tmp_path):
    path, _ = _write_catalog(tmp_path)
    document = json.loads(path.read_text(encoding="utf-8"))
    document["labeler"].pop("dataset_manifest_sha256")
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError, match="dataset_manifest_sha256"):
        PhaseCatalog.load(path)
