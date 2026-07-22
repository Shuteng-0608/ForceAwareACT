import hashlib
import json
import uuid
from pathlib import Path

import h5py
import numpy as np
import pytest

from force_aware_act.data import ContactForceHDF5Dataset
from force_aware_act.data.manifest import (
    DERIVED_EPISODE_UUID_SOURCE,
    DatasetManifest,
    EpisodeIdentity,
    EpisodeManifestEntry,
)
from force_aware_act.training.catalog import PhaseCatalog
from scripts.build_phase_catalog import main


EPISODE_UUID = "12345678-1234-5678-1234-567812345678"


def _write_episode(
    path: Path, *, n_state: int = 8, episode_uuid: str = EPISODE_UUID
) -> None:
    n_image = n_state
    n_force = n_state * 2
    joint_values = np.arange(n_state * 7, dtype=np.float32).reshape(n_state, 7)
    with h5py.File(path, "w") as handle:
        handle.attrs["episode_uuid"] = episode_uuid
        timestamps = handle.create_group("timestamps")
        timestamps.create_dataset(
            "state_episode", data=np.arange(n_state, dtype=np.float64) * 0.1
        )
        timestamps.create_dataset(
            "image_episode", data=np.arange(n_image, dtype=np.float64) * 0.1
        )
        timestamps.create_dataset(
            "force_episode", data=np.arange(n_force, dtype=np.float64) * 0.05
        )

        observations = handle.create_group("observations")
        observations.create_dataset("joint_pos", data=joint_values)
        observations.create_dataset("joint_vel", data=joint_values + 1.0)
        observations.create_dataset("joint_torque", data=joint_values + 2.0)
        observations.create_dataset("ee_pose", data=joint_values + 3.0)
        observations.create_dataset(
            "ft_wrench", data=np.zeros((n_force, 6), dtype=np.float32)
        )
        images = observations.create_group("images")
        images.create_dataset(
            "camera", data=np.zeros((n_image, 2, 2, 3), dtype=np.uint8)
        )


def _write_inputs(tmp_path: Path, rows: str) -> tuple[Path, Path, Path, Path]:
    episode = tmp_path / "episode.hdf5"
    _write_episode(episode)
    episode_list = tmp_path / "episodes.txt"
    episode_list.write_text(f"{episode}\n", encoding="utf-8")
    annotations = tmp_path / "annotations.csv"
    annotations.write_text(
        "episode_path,start,stop,phase\n" + rows,
        encoding="utf-8",
    )
    return episode, episode_list, annotations, tmp_path / "catalog.json"


def _args(episode_list: Path, annotations: Path, output: Path) -> list[str]:
    episode = Path(episode_list.read_text(encoding="utf-8").strip()).resolve()
    entry = EpisodeManifestEntry(
        identity=EpisodeIdentity.from_path(episode, EPISODE_UUID),
        domain="r2_contact",
        split="train",
        metadata={"episode_uuid_source": "hdf5_root_attr:episode_uuid"},
    )
    manifest = DatasetManifest(
        (entry,), metadata={"derive_uuid_from_sha256": False}
    )
    manifest_path = episode_list.parent / "dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest.to_dict()), encoding="utf-8")
    return [
        "--annotation",
        str(annotations),
        "--episode-list",
        str(episode_list),
        "--dataset-manifest",
        str(manifest_path),
        "--dataset-manifest-sha256",
        manifest.content_sha256,
        "--source-domain",
        "r2_contact",
        "--output",
        str(output),
        "--action-mode",
        "joint_pos",
        "--chunk-len",
        "2",
        "--force-window-len",
        "3",
        "--force-window-duration",
        "0.1",
        "--camera-names",
        "camera",
        "--image-size",
        "8",
        "8",
        "--image-alignment",
        "latest_past",
        "--max-image-lag-seconds",
        "0.1",
        "--strict-lengths",
    ]


def test_builder_writes_loadable_catalog_with_exact_dataset_coverage(tmp_path, capsys):
    episode, episode_list, annotations, output = _write_inputs(
        tmp_path,
        f"{tmp_path / 'episode.hdf5'},0,2,free\n"
        f"{tmp_path / 'episode.hdf5'},2,6,contact\n",
    )

    assert main(_args(episode_list, annotations, output)) == 0

    document = json.loads(output.read_text(encoding="utf-8"))
    canonical = json.dumps(
        document,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    assert output.read_bytes() == canonical
    assert document["episodes"] == [
        {
            "domain": "r2_contact",
            "episode_uuid": EPISODE_UUID,
            "file_sha256": hashlib.sha256(episode.read_bytes()).hexdigest(),
            "path": str(episode.resolve()),
            "segments": [
                {"phase": "free", "start": 0, "stop": 2},
                {"phase": "contact", "start": 2, "stop": 6},
            ],
        }
    ]
    labeler = document["labeler"]
    assert labeler["phase_source"] == "manual_csv_only"
    assert labeler["coverage"] == "exact_complete_dataset_state_indices"
    assert labeler["annotation_sha256"] == hashlib.sha256(annotations.read_bytes()).hexdigest()
    assert labeler["episode_list_sha256"] == hashlib.sha256(episode_list.read_bytes()).hexdigest()
    manifest = DatasetManifest.load(tmp_path / "dataset_manifest.json")
    assert labeler["dataset_manifest_sha256"] == manifest.content_sha256
    assert labeler["dataset_semantics"] == {
        "action_dataset": "observations/joint_pos",
        "action_mode": "joint_pos",
        "action_offset": 1,
        "camera_names": ["camera"],
        "chunk_len": 2,
        "force_window_duration": 0.1,
        "force_window_len": 3,
        "image_size": [8, 8],
        "imagenet_normalize": False,
        "image_alignment": "latest_past",
        "include_force": True,
        "max_image_lag_seconds": 0.1,
        "max_length_mismatch": 0,
        "normalize_images": True,
        "tolerate_length_mismatch": False,
    }

    catalog = PhaseCatalog.load(output)
    dataset = ContactForceHDF5Dataset(
        episode,
        camera_names=("camera",),
        action_mode="joint_pos",
        chunk_len=2,
        force_window_len=3,
        force_window_duration=0.1,
        image_size=(8, 8),
        image_alignment="latest_past",
        max_image_lag_seconds=0.1,
        tolerate_length_mismatch=False,
        max_length_mismatch=0,
    )
    catalog.validate_indices(dataset.indices)
    assert catalog.phase_for(episode, 5) == "contact"
    assert "phase_source=manual_csv_only" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        ("{episode},0,2,free\n{episode},3,5,contact\n", "annotation gap"),
        ("{episode},0,3,free\n{episode},2,5,contact\n", "annotation overlap"),
    ],
)
def test_builder_rejects_annotation_gap_or_overlap(tmp_path, capsys, rows, message):
    episode = tmp_path / "episode.hdf5"
    _, episode_list, annotations, output = _write_inputs(
        tmp_path,
        rows.format(episode=episode),
    )

    assert main(_args(episode_list, annotations, output)) == 1
    assert not output.exists()
    assert message in capsys.readouterr().err


def test_builder_rejects_episode_outside_the_supplied_list(tmp_path, capsys):
    _, episode_list, annotations, output = _write_inputs(tmp_path, "")
    outside = tmp_path / "outside.hdf5"
    _write_episode(outside)
    annotations.write_text(
        "episode_path,start,stop,phase\n"
        f"{outside},0,5,contact\n",
        encoding="utf-8",
    )

    assert main(_args(episode_list, annotations, output)) == 1
    assert not output.exists()
    error = capsys.readouterr().err
    assert str(outside) in error
    assert "is not in the specified episode list" in error


def test_builder_rejects_uncovered_dataset_index(tmp_path, capsys):
    episode = tmp_path / "episode.hdf5"
    _, episode_list, annotations, output = _write_inputs(
        tmp_path,
        f"{episode},0,4,contact\n",
    )

    assert main(_args(episode_list, annotations, output)) == 1
    assert not output.exists()
    error = capsys.readouterr().err
    assert "coverage does not exactly match dataset indices" in error
    assert "uncovered suffix [4,6)" in error


@pytest.mark.parametrize("bad_integer", ["NaN", "1.5", "01", "-1"])
def test_builder_rejects_non_integer_or_noncanonical_bounds(tmp_path, capsys, bad_integer):
    episode = tmp_path / "episode.hdf5"
    _, episode_list, annotations, output = _write_inputs(
        tmp_path,
        f"{episode},{bad_integer},5,contact\n",
    )

    assert main(_args(episode_list, annotations, output)) == 1
    assert not output.exists()
    assert "must be a canonical non-negative integer" in capsys.readouterr().err


def test_builder_refuses_to_overwrite_existing_output(tmp_path, capsys):
    episode = tmp_path / "episode.hdf5"
    _, episode_list, annotations, output = _write_inputs(
        tmp_path,
        f"{episode},0,5,contact\n",
    )
    output.write_bytes(b"keep-me")

    assert main(_args(episode_list, annotations, output)) == 1
    assert output.read_bytes() == b"keep-me"
    assert "refusing to overwrite" in capsys.readouterr().err


def test_builder_output_is_deterministic(tmp_path):
    episode = tmp_path / "episode.hdf5"
    _, episode_list, annotations, first_output = _write_inputs(
        tmp_path,
        f"{episode},2,6,contact\n{episode},0,2,free\n",
    )
    second_output = tmp_path / "catalog-second.json"

    assert main(_args(episode_list, annotations, first_output)) == 0
    assert main(_args(episode_list, annotations, second_output)) == 0
    assert first_output.read_bytes() == second_output.read_bytes()


def test_builder_rejects_unpinned_or_changed_manifest(tmp_path, capsys):
    episode = tmp_path / "episode.hdf5"
    _, episode_list, annotations, output = _write_inputs(
        tmp_path,
        f"{episode},0,5,contact\n",
    )
    args = _args(episode_list, annotations, output)
    args[args.index("--dataset-manifest-sha256") + 1] = "0" * 64

    assert main(args) == 1
    assert not output.exists()
    assert "dataset manifest SHA256 mismatch" in capsys.readouterr().err


def test_builder_rejects_list_that_is_not_exact_domain_train_population(
    tmp_path, capsys
):
    episode = tmp_path / "episode.hdf5"
    _, episode_list, annotations, output = _write_inputs(
        tmp_path,
        f"{episode},0,5,contact\n",
    )
    args = _args(episode_list, annotations, output)
    manifest_path = tmp_path / "dataset_manifest.json"
    manifest = DatasetManifest.load(manifest_path)
    extra = tmp_path / "extra.hdf5"
    extra_uuid = str(uuid.uuid4())
    _write_episode(extra, episode_uuid=extra_uuid)
    extended = DatasetManifest(
        manifest.episodes
        + (
            EpisodeManifestEntry(
                identity=EpisodeIdentity.from_path(extra, extra_uuid),
                domain="r2_contact",
                split="train",
                metadata={"episode_uuid_source": "hdf5_root_attr:episode_uuid"},
            ),
        ),
        metadata={"derive_uuid_from_sha256": False},
    )
    manifest_path.write_text(json.dumps(extended.to_dict()), encoding="utf-8")
    args[args.index("--dataset-manifest-sha256") + 1] = extended.content_sha256

    assert main(args) == 1
    assert not output.exists()
    assert "must exactly equal dataset manifest train entries" in capsys.readouterr().err


def test_builder_rejects_derived_episode_uuid_provenance(tmp_path, capsys):
    episode = tmp_path / "episode.hdf5"
    _, episode_list, annotations, output = _write_inputs(
        tmp_path,
        f"{episode},0,5,contact\n",
    )
    args = _args(episode_list, annotations, output)
    manifest_path = tmp_path / "dataset_manifest.json"
    manifest = DatasetManifest.load(manifest_path)
    original = manifest.episodes[0]
    derived = DatasetManifest(
        (
            EpisodeManifestEntry(
                identity=original.identity,
                domain=original.domain,
                split=original.split,
                metadata={"episode_uuid_source": DERIVED_EPISODE_UUID_SOURCE},
            ),
        ),
        metadata={"derive_uuid_from_sha256": True},
    )
    manifest_path.write_text(json.dumps(derived.to_dict()), encoding="utf-8")
    args[args.index("--dataset-manifest-sha256") + 1] = derived.content_sha256

    assert main(args) == 1
    assert not output.exists()
    assert "forbids SHA-derived episode UUIDs" in capsys.readouterr().err
