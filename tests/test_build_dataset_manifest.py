import json
import shutil
import uuid
from pathlib import Path

import h5py
import numpy as np
import pytest

from force_aware_act.data.manifest import (
    DatasetManifest,
    canonical_json_bytes,
    sha256_file,
    validate_episode_uuid_provenance,
)
from scripts.build_dataset_manifest import (
    GroupSpec,
    build_dataset_manifest,
    main,
    parse_group_spec,
)


def _uuid(index: int) -> str:
    return str(uuid.UUID(int=index + 1))


def _write_episode(
    directory: Path,
    *,
    marker: int,
    episode_uuid=None,
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "episode.hdf5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("marker", data=np.array([marker], dtype=np.int64))
        if episode_uuid is not None:
            handle.attrs["episode_uuid"] = episode_uuid
    return path.resolve()


def _write_list(path: Path, *episodes: Path) -> Path:
    path.write_text("\n".join(str(episode) for episode in episodes) + "\n", encoding="utf-8")
    return path.resolve()


def test_parse_group_spec_accepts_only_domain_split_assignment():
    spec = parse_group_spec("r60:train=configs/r60_train.txt")

    assert spec == GroupSpec("r60", "train", Path("configs/r60_train.txt"))

    with pytest.raises(Exception, match="DOMAIN:SPLIT"):
        parse_group_spec("r60=train.txt")
    with pytest.raises(Exception, match="SPLIT must be one of"):
        parse_group_spec("r60:validation=train.txt")


def test_hdf5_root_uuid_has_priority_over_sibling_metadata(tmp_path):
    episode = _write_episode(tmp_path / "episode_dir", marker=1, episode_uuid=_uuid(0))
    (episode.parent / "metadata.json").write_text(
        json.dumps({"episode_uuid": _uuid(1)}),
        encoding="utf-8",
    )
    episode_list = _write_list(tmp_path / "train.txt", episode)

    manifest = build_dataset_manifest([GroupSpec("r60", "train", episode_list)])

    entry = manifest.episodes[0]
    assert entry.identity.episode_uuid == _uuid(0)
    assert entry.metadata["episode_uuid_source"] == "hdf5_root_attr:episode_uuid"
    assert entry.identity.file_sha256 == sha256_file(episode)


def test_sibling_metadata_uuid_is_used_when_root_attr_is_absent(tmp_path):
    episode = _write_episode(tmp_path / "episode_dir", marker=2)
    (episode.parent / "metadata.json").write_text(
        json.dumps({"episode_uuid": _uuid(2), "status": "success"}),
        encoding="utf-8",
    )
    episode_list = _write_list(tmp_path / "train.txt", episode)

    manifest = build_dataset_manifest([GroupSpec("r2", "train", episode_list)])

    entry = manifest.episodes[0]
    assert entry.identity.episode_uuid == _uuid(2)
    assert entry.metadata["episode_uuid_source"] == "sibling_metadata_json:episode_uuid"


def test_missing_uuid_fails_unless_historical_derivation_is_explicit(tmp_path):
    episode = _write_episode(tmp_path / "episode_dir", marker=3)
    episode_list = _write_list(tmp_path / "train.txt", episode)
    groups = [GroupSpec("r60", "train", episode_list)]

    with pytest.raises(ValueError, match="episode UUID is missing"):
        build_dataset_manifest(groups)

    first = build_dataset_manifest(groups, derive_uuid_from_sha256=True)
    second = build_dataset_manifest(groups, derive_uuid_from_sha256=True)

    assert first.episodes[0].identity.episode_uuid == second.episodes[0].identity.episode_uuid
    assert first.episodes[0].metadata["episode_uuid_source"] == (
        "derived:uuid5(file_sha256)"
    )
    assert first.metadata["derive_uuid_from_sha256"] is True
    with pytest.raises(ValueError, match="forbids SHA-derived"):
        validate_episode_uuid_provenance(first, allow_derived=False)
    validate_episode_uuid_provenance(first, allow_derived=True)


def test_invalid_explicit_uuid_is_not_silently_replaced_by_derivation(tmp_path):
    episode = _write_episode(
        tmp_path / "episode_dir",
        marker=4,
        episode_uuid="not-a-uuid",
    )
    episode_list = _write_list(tmp_path / "train.txt", episode)

    with pytest.raises(ValueError, match="invalid episode UUID from hdf5_root_attr"):
        build_dataset_manifest(
            [GroupSpec("r60", "train", episode_list)],
            derive_uuid_from_sha256=True,
        )


def test_same_episode_across_splits_is_rejected(tmp_path):
    episode = _write_episode(tmp_path / "episode_dir", marker=5, episode_uuid=_uuid(5))
    train_list = _write_list(tmp_path / "train.txt", episode)
    val_list = _write_list(tmp_path / "val.txt", episode)

    with pytest.raises(ValueError, match="episode leakage"):
        build_dataset_manifest(
            [
                GroupSpec("r60", "train", train_list),
                GroupSpec("r60", "val", val_list),
            ]
        )


def test_copied_file_with_different_metadata_uuid_is_rejected_by_sha(tmp_path):
    first = _write_episode(tmp_path / "first", marker=6)
    second_dir = tmp_path / "second"
    second_dir.mkdir()
    second = (second_dir / "episode.hdf5").resolve()
    shutil.copyfile(first, second)
    (first.parent / "metadata.json").write_text(
        json.dumps({"episode_uuid": _uuid(6)}),
        encoding="utf-8",
    )
    (second.parent / "metadata.json").write_text(
        json.dumps({"episode_uuid": _uuid(7)}),
        encoding="utf-8",
    )
    train_list = _write_list(tmp_path / "train.txt", first)
    val_list = _write_list(tmp_path / "val.txt", second)

    with pytest.raises(ValueError, match="identity_field='file_sha256'"):
        build_dataset_manifest(
            [
                GroupSpec("r60", "train", train_list),
                GroupSpec("r2", "val", val_list),
            ]
        )


def test_duplicate_path_inside_one_episode_list_is_rejected(tmp_path):
    episode = _write_episode(tmp_path / "episode_dir", marker=8, episode_uuid=_uuid(8))
    episode_list = _write_list(tmp_path / "train.txt", episode, episode)

    with pytest.raises(ValueError, match="duplicate episode_uuid"):
        build_dataset_manifest([GroupSpec("r60", "train", episode_list)])


def test_duplicate_domain_split_group_is_rejected(tmp_path):
    first = _write_episode(tmp_path / "first", marker=9, episode_uuid=_uuid(9))
    second = _write_episode(tmp_path / "second", marker=10, episode_uuid=_uuid(10))
    first_list = _write_list(tmp_path / "first.txt", first)
    second_list = _write_list(tmp_path / "second.txt", second)

    with pytest.raises(ValueError, match="duplicate domain/split"):
        build_dataset_manifest(
            [
                GroupSpec("r60", "train", first_list),
                GroupSpec("r60", "train", second_list),
            ]
        )


def test_malformed_sibling_metadata_is_rejected_even_in_derive_mode(tmp_path):
    episode = _write_episode(tmp_path / "episode_dir", marker=11)
    (episode.parent / "metadata.json").write_text(
        '{"episode_uuid":"x","episode_uuid":"y"}',
        encoding="utf-8",
    )
    episode_list = _write_list(tmp_path / "train.txt", episode)

    with pytest.raises(ValueError, match="duplicate JSON key"):
        build_dataset_manifest(
            [GroupSpec("r60", "train", episode_list)],
            derive_uuid_from_sha256=True,
        )


def test_cli_writes_canonical_manifest_and_refuses_overwrite(tmp_path, capsys):
    train = _write_episode(tmp_path / "train", marker=12, episode_uuid=_uuid(12))
    val = _write_episode(tmp_path / "val", marker=13, episode_uuid=_uuid(13))
    train_list = _write_list(tmp_path / "train.txt", train)
    val_list = _write_list(tmp_path / "val.txt", val)
    output = tmp_path / "manifests" / "dataset.json"
    argv = [
        "--group",
        f"r60:val={val_list}",
        "--group",
        f"r60:train={train_list}",
        "--output",
        str(output),
    ]

    assert main(argv) == 0
    original_bytes = output.read_bytes()
    parsed = json.loads(original_bytes)
    assert original_bytes == canonical_json_bytes(parsed) + b"\n"
    loaded = DatasetManifest.load(output, verify_files=True)
    assert [entry.split for entry in loaded.episodes] == ["train", "val"]
    assert loaded.metadata["manifest_builder_version"] == 1
    assert len(loaded.metadata["groups"]) == 2

    assert main(argv) == 2
    assert output.read_bytes() == original_bytes
    assert "refusing to overwrite" in capsys.readouterr().err


def test_manifest_is_independent_of_group_argument_order(tmp_path):
    r60 = _write_episode(tmp_path / "r60", marker=14, episode_uuid=_uuid(14))
    r2 = _write_episode(tmp_path / "r2", marker=15, episode_uuid=_uuid(15))
    r60_list = _write_list(tmp_path / "r60.txt", r60)
    r2_list = _write_list(tmp_path / "r2.txt", r2)
    groups = [
        GroupSpec("r60", "train", r60_list),
        GroupSpec("r2", "test", r2_list),
    ]

    first = build_dataset_manifest(groups)
    second = build_dataset_manifest(list(reversed(groups)))

    assert first.to_dict() == second.to_dict()
    assert first.content_sha256 == second.content_sha256
