"""Deterministic episode discovery and disjoint train/validation splits."""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Tuple

from force_aware_act.act_aligned_training.schema import (
    EXPECTED_SCHEMA_VERSION,
    inspect_episode,
)


SPLIT_FORMAT_VERSION = "act_aligned_episode_split_v1"


@dataclass(frozen=True)
class EpisodeRecord:
    episode_id: str
    relative_hdf5_path: str
    num_steps: int
    camera_names: Tuple[str, ...]

    def resolve(self, data_root: Path) -> Path:
        return Path(data_root) / self.relative_hdf5_path


@dataclass(frozen=True)
class EpisodeSplitManifest:
    format_version: str
    data_root: str
    seed: int
    validation_fraction: float
    train_episodes: Tuple[EpisodeRecord, ...]
    validation_episodes: Tuple[EpisodeRecord, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "EpisodeSplitManifest":
        if values.get("format_version") != SPLIT_FORMAT_VERSION:
            raise ValueError("unsupported split manifest format")
        return cls(
            format_version=values["format_version"],
            data_root=values["data_root"],
            seed=int(values["seed"]),
            validation_fraction=float(values["validation_fraction"]),
            train_episodes=tuple(
                EpisodeRecord(
                    episode_id=item["episode_id"],
                    relative_hdf5_path=item["relative_hdf5_path"],
                    num_steps=int(item["num_steps"]),
                    camera_names=tuple(item["camera_names"]),
                )
                for item in values["train_episodes"]
            ),
            validation_episodes=tuple(
                EpisodeRecord(
                    episode_id=item["episode_id"],
                    relative_hdf5_path=item["relative_hdf5_path"],
                    num_steps=int(item["num_steps"]),
                    camera_names=tuple(item["camera_names"]),
                )
                for item in values["validation_episodes"]
            ),
        )


def discover_episodes(data_root: Path) -> Tuple[EpisodeRecord, ...]:
    """Discover and validate every compact MuJoCo episode."""

    data_root = Path(data_root).resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(f"data root does not exist: {data_root}")
    records = []
    for episode_dir in sorted(path for path in data_root.iterdir() if path.is_dir()):
        hdf5_path = episode_dir / "episode.hdf5"
        metadata_path = episode_dir / "metadata.json"
        if not hdf5_path.is_file() or not metadata_path.is_file():
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("schema_version") != EXPECTED_SCHEMA_VERSION:
            raise ValueError(f"{episode_dir} has an unsupported metadata schema")
        schema = inspect_episode(hdf5_path)
        if int(metadata.get("n_state", -1)) != schema.num_steps:
            raise ValueError(f"{episode_dir} metadata n_state is inconsistent")
        records.append(
            EpisodeRecord(
                episode_id=episode_dir.name,
                relative_hdf5_path=str(hdf5_path.relative_to(data_root)),
                num_steps=schema.num_steps,
                camera_names=schema.camera_names,
            )
        )
    if len(records) < 2:
        raise ValueError("at least two valid episodes are required")
    return tuple(records)


def create_episode_split(
    data_root: Path,
    *,
    validation_fraction: float = 0.1,
    seed: int = 0,
) -> EpisodeSplitManifest:
    """Create a deterministic episode-disjoint split."""

    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in (0, 1)")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed must be an integer")
    records = list(discover_episodes(data_root))
    random.Random(seed).shuffle(records)
    validation_count = max(1, round(len(records) * validation_fraction))
    validation_count = min(validation_count, len(records) - 1)
    validation = tuple(sorted(records[:validation_count], key=lambda item: item.episode_id))
    train = tuple(sorted(records[validation_count:], key=lambda item: item.episode_id))
    return EpisodeSplitManifest(
        format_version=SPLIT_FORMAT_VERSION,
        data_root=str(Path(data_root).resolve()),
        seed=seed,
        validation_fraction=float(validation_fraction),
        train_episodes=train,
        validation_episodes=validation,
    )
