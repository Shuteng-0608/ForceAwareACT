"""Versioned phase labels used by the staged-training batch sampler."""

from __future__ import annotations

import bisect
import hashlib
import json
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple


PHASE_CATALOG_SCHEMA_VERSION = 2

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _nonempty_string(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context} must be a non-empty string")
    return value.strip()


def _nonnegative_int(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{context} must be a non-negative integer")
    return value


def _sha256(value: Any, context: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{context} must be exactly 64 lowercase hexadecimal characters")
    return value


def _canonical_uuid(value: Any, context: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{context} must be a string")
    try:
        canonical = str(uuid.UUID(value))
    except (AttributeError, ValueError) as error:
        raise ValueError(f"{context} is not a valid UUID: {value!r}") from error
    if value != canonical:
        raise ValueError(
            f"{context} must use canonical lowercase hyphenated form: "
            f"expected={canonical!r} actual={value!r}"
        )
    return canonical


def _reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"phase catalog contains duplicate JSON key {key!r}")
        result[key] = value
    return result


@dataclass(frozen=True)
class PhaseSegment:
    start: int
    stop: int
    phase: str

    def __post_init__(self) -> None:
        start = _nonnegative_int(self.start, "phase segment start")
        stop = _nonnegative_int(self.stop, "phase segment stop")
        if stop <= start:
            raise ValueError("phase segment stop must be greater than start")
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "stop", stop)
        object.__setattr__(self, "phase", _nonempty_string(self.phase, "phase segment phase"))


@dataclass(frozen=True)
class EpisodePhaseCatalog:
    episode_path: Path
    domain: str
    episode_uuid: str
    file_sha256: str
    segments: Tuple[PhaseSegment, ...]

    def __post_init__(self) -> None:
        path = Path(self.episode_path).expanduser().resolve(strict=False)
        segments = tuple(self.segments)
        if not segments:
            raise ValueError(f"phase catalog episode {path} has no segments")
        ordered = tuple(sorted(segments, key=lambda segment: (segment.start, segment.stop)))
        for previous, current in zip(ordered, ordered[1:]):
            if current.start < previous.stop:
                raise ValueError(
                    f"phase catalog episode {path} has overlapping segments "
                    f"[{previous.start},{previous.stop}) and [{current.start},{current.stop})"
                )
        object.__setattr__(self, "episode_path", path)
        object.__setattr__(self, "domain", _nonempty_string(self.domain, "phase catalog domain"))
        object.__setattr__(
            self,
            "episode_uuid",
            _canonical_uuid(self.episode_uuid, "phase catalog episode_uuid"),
        )
        object.__setattr__(
            self,
            "file_sha256",
            _sha256(self.file_sha256, "phase catalog file_sha256"),
        )
        object.__setattr__(self, "segments", ordered)

    def phase_for(self, state_index: int) -> str:
        """Return the unique phase for a state index or fail on a coverage gap."""

        index = _nonnegative_int(state_index, "state_index")
        starts = [segment.start for segment in self.segments]
        position = bisect.bisect_right(starts, index) - 1
        if position < 0:
            raise KeyError(f"state_index={index} is not covered for {self.episode_path}")
        segment = self.segments[position]
        if index >= segment.stop:
            raise KeyError(f"state_index={index} is not covered for {self.episode_path}")
        return segment.phase


@dataclass(frozen=True)
class PhaseCatalog:
    episodes: Tuple[EpisodePhaseCatalog, ...]
    labeler: Mapping[str, Any]
    source_path: Optional[Path]
    content_sha256: str
    dataset_manifest_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        episodes = tuple(self.episodes)
        by_path: Dict[Path, EpisodePhaseCatalog] = {}
        by_uuid: Dict[str, EpisodePhaseCatalog] = {}
        by_file_sha256: Dict[str, EpisodePhaseCatalog] = {}
        for episode in episodes:
            if episode.episode_path in by_path:
                raise ValueError(f"duplicate phase catalog episode: {episode.episode_path}")
            if episode.episode_uuid in by_uuid:
                raise ValueError(
                    "duplicate phase catalog episode_uuid: "
                    f"{episode.episode_uuid}"
                )
            if episode.file_sha256 in by_file_sha256:
                raise ValueError(
                    "duplicate phase catalog file_sha256: "
                    f"{episode.file_sha256}"
                )
            by_path[episode.episode_path] = episode
            by_uuid[episode.episode_uuid] = episode
            by_file_sha256[episode.file_sha256] = episode
        manifest_sha256 = _sha256(
            self.labeler.get("dataset_manifest_sha256"),
            "phase catalog labeler.dataset_manifest_sha256",
        )
        object.__setattr__(self, "episodes", episodes)
        object.__setattr__(self, "labeler", dict(self.labeler))
        object.__setattr__(self, "_by_path", by_path)
        object.__setattr__(self, "dataset_manifest_sha256", manifest_sha256)

    @classmethod
    def load(cls, path: Path) -> "PhaseCatalog":
        source_path = Path(path).expanduser().resolve()
        if not source_path.is_file():
            raise FileNotFoundError(f"phase catalog does not exist: {source_path}")
        try:
            document = json.loads(
                source_path.read_text(encoding="utf-8"),
                object_pairs_hook=_reject_duplicate_keys,
                parse_constant=lambda value: (_ for _ in ()).throw(
                    ValueError(f"phase catalog contains non-finite value {value}")
                ),
            )
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid phase catalog JSON: {source_path}: {error}") from error
        if not isinstance(document, Mapping):
            raise ValueError("phase catalog root must be a JSON object")
        unknown = sorted(set(document) - {"schema_version", "episodes", "labeler"})
        if unknown:
            raise ValueError(f"phase catalog contains unknown keys: {', '.join(unknown)}")
        if document.get("schema_version") != PHASE_CATALOG_SCHEMA_VERSION:
            raise ValueError(
                "unsupported phase catalog schema_version: "
                f"{document.get('schema_version')!r}"
            )
        raw_episodes = document.get("episodes")
        if not isinstance(raw_episodes, list) or not raw_episodes:
            raise ValueError("phase catalog episodes must be a non-empty array")
        episodes = []
        for episode_index, raw_episode in enumerate(raw_episodes):
            context = f"episodes[{episode_index}]"
            if not isinstance(raw_episode, Mapping):
                raise ValueError(f"{context} must be a JSON object")
            unknown_episode = sorted(
                set(raw_episode)
                - {"path", "domain", "episode_uuid", "file_sha256", "segments"}
            )
            if unknown_episode:
                raise ValueError(f"{context} contains unknown keys: {', '.join(unknown_episode)}")
            episode_text = _nonempty_string(raw_episode.get("path"), f"{context}.path")
            episode_path = Path(episode_text).expanduser()
            if not episode_path.is_absolute():
                episode_path = source_path.parent / episode_path
            raw_segments = raw_episode.get("segments")
            if not isinstance(raw_segments, list) or not raw_segments:
                raise ValueError(f"{context}.segments must be a non-empty array")
            segments = []
            for segment_index, raw_segment in enumerate(raw_segments):
                segment_context = f"{context}.segments[{segment_index}]"
                if not isinstance(raw_segment, Mapping):
                    raise ValueError(f"{segment_context} must be a JSON object")
                unknown_segment = sorted(set(raw_segment) - {"start", "stop", "phase"})
                if unknown_segment:
                    raise ValueError(
                        f"{segment_context} contains unknown keys: {', '.join(unknown_segment)}"
                    )
                segments.append(
                    PhaseSegment(
                        start=raw_segment.get("start"),
                        stop=raw_segment.get("stop"),
                        phase=raw_segment.get("phase"),
                    )
                )
            episodes.append(
                EpisodePhaseCatalog(
                    episode_path=episode_path,
                    domain=raw_episode.get("domain"),
                    episode_uuid=raw_episode.get("episode_uuid"),
                    file_sha256=raw_episode.get("file_sha256"),
                    segments=tuple(segments),
                )
            )
        labeler = document.get("labeler", {})
        if not isinstance(labeler, Mapping):
            raise ValueError("phase catalog labeler must be a JSON object")
        return cls(
            episodes=tuple(episodes),
            labeler=dict(labeler),
            source_path=source_path,
            content_sha256=hashlib.sha256(_canonical_json_bytes(document)).hexdigest(),
        )

    def phase_for(self, episode_path: Path, state_index: int) -> str:
        path = Path(episode_path).expanduser().resolve(strict=False)
        try:
            episode = self._by_path[path]
        except KeyError as error:
            raise KeyError(f"episode is absent from phase catalog: {path}") from error
        return episode.phase_for(state_index)

    def episode_for(self, episode_path: Path) -> EpisodePhaseCatalog:
        """Return the immutable catalog identity for one canonical episode path."""

        path = Path(episode_path).expanduser().resolve(strict=False)
        try:
            return self._by_path[path]
        except KeyError as error:
            raise KeyError(f"episode is absent from phase catalog: {path}") from error

    def validate_indices(self, indices: Sequence[Any]) -> None:
        """Ensure every dataset index has exactly one phase label."""

        for item in indices:
            if not hasattr(item, "episode_path") or not hasattr(item, "state_index"):
                raise TypeError("dataset indices must expose episode_path and state_index")
            self.phase_for(item.episode_path, item.state_index)
