"""Strict, auditable dataset-manifest primitives.

An episode is identified on three independent axes: a stable UUID, its
canonical filesystem path, and the SHA-256 digest of its bytes.  Treating all
three as identities lets callers detect both accidental list duplication and
less obvious leakage such as a copied file under a different name.

This module deliberately does not inspect or modify HDF5 contents.  It only
describes files and validates the populations used by a training protocol.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union


MANIFEST_SCHEMA_VERSION = 1
SPLIT_NAMES = ("train", "val", "test")
NATIVE_EPISODE_UUID_SOURCES = (
    "hdf5_root_attr:episode_uuid",
    "sibling_metadata_json:episode_uuid",
)
DERIVED_EPISODE_UUID_SOURCE = "derived:uuid5(file_sha256)"

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_IDENTITY_FIELDS = ("episode_uuid", "path", "file_sha256")


def canonical_json_bytes(value: Any) -> bytes:
    """Encode a JSON value using the repository's canonical representation.

    Object keys are sorted, insignificant whitespace is removed, UTF-8 is
    emitted directly, and non-finite floating point values are rejected.
    Arrays intentionally retain their order because episode order can affect
    deterministic sampling and is therefore part of the manifest identity.
    """

    _validate_json_value(value, path="$")
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def canonical_json_sha256(value: Any) -> str:
    """Return the SHA-256 digest of :func:`canonical_json_bytes`."""

    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: Union[str, Path], chunk_size: int = 1024 * 1024) -> str:
    """Hash an existing regular file without loading it all into memory."""

    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    resolved = _resolve_path(path, base_dir=None, require_file=True)
    digest = hashlib.sha256()
    with resolved.open("rb") as file_handle:
        while True:
            chunk = file_handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class EpisodeIdentity:
    """Immutable UUID/path/content identity for one episode file."""

    episode_uuid: str
    path: Path
    file_sha256: str

    def __post_init__(self) -> None:
        canonical_uuid = _validate_uuid(self.episode_uuid)
        canonical_path = _resolve_path(self.path, base_dir=None, require_file=False)
        canonical_sha256 = _validate_sha256(self.file_sha256)
        object.__setattr__(self, "episode_uuid", canonical_uuid)
        object.__setattr__(self, "path", canonical_path)
        object.__setattr__(self, "file_sha256", canonical_sha256)

    @classmethod
    def from_path(cls, path: Union[str, Path], episode_uuid: str) -> "EpisodeIdentity":
        """Build and hash an identity from an existing episode file."""

        resolved = _resolve_path(path, base_dir=None, require_file=True)
        return cls(
            episode_uuid=episode_uuid,
            path=resolved,
            file_sha256=sha256_file(resolved),
        )

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, Any],
        *,
        base_dir: Optional[Path] = None,
        verify_file: bool = False,
    ) -> "EpisodeIdentity":
        """Parse an identity and optionally verify its file and digest."""

        _require_mapping(value, "episode identity")
        for key in _IDENTITY_FIELDS:
            if key not in value:
                raise ValueError(f"episode identity is missing required field: {key}")
        resolved = _resolve_path(value["path"], base_dir=base_dir, require_file=verify_file)
        identity = cls(
            episode_uuid=value["episode_uuid"],
            path=resolved,
            file_sha256=value["file_sha256"],
        )
        if verify_file:
            actual_sha256 = sha256_file(identity.path)
            if actual_sha256 != identity.file_sha256:
                raise ValueError(
                    "episode file SHA-256 mismatch: "
                    f"path={identity.path}; expected={identity.file_sha256}; "
                    f"actual={actual_sha256}"
                )
        return identity

    def to_dict(self) -> Dict[str, str]:
        """Return a JSON-compatible canonical identity."""

        return {
            "episode_uuid": self.episode_uuid,
            "path": str(self.path),
            "file_sha256": self.file_sha256,
        }

    @property
    def content_sha256(self) -> str:
        """Hash the canonical JSON representation of this identity."""

        return canonical_json_sha256(self.to_dict())


@dataclass(frozen=True)
class EpisodeManifestEntry:
    """One episode identity plus its immutable protocol assignment."""

    identity: EpisodeIdentity
    domain: str
    split: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.identity, EpisodeIdentity):
            raise TypeError("identity must be an EpisodeIdentity")
        object.__setattr__(self, "domain", _validate_nonempty_string(self.domain, "domain"))
        if self.split not in SPLIT_NAMES:
            raise ValueError(
                f"split must be one of {SPLIT_NAMES}, got {self.split!r}"
            )
        object.__setattr__(self, "metadata", _canonical_json_copy(self.metadata, "metadata"))

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, Any],
        *,
        base_dir: Optional[Path] = None,
        verify_file: bool = False,
    ) -> "EpisodeManifestEntry":
        """Parse a strict, flat manifest entry."""

        _require_mapping(value, "episode manifest entry")
        allowed_fields = set(_IDENTITY_FIELDS) | {"domain", "split", "metadata"}
        unexpected = sorted(set(value) - allowed_fields)
        if unexpected:
            raise ValueError(f"episode manifest entry has unexpected fields: {unexpected}")
        for key in ("domain", "split"):
            if key not in value:
                raise ValueError(f"episode manifest entry is missing required field: {key}")
        identity = EpisodeIdentity.from_dict(
            value,
            base_dir=base_dir,
            verify_file=verify_file,
        )
        return cls(
            identity=identity,
            domain=value["domain"],
            split=value["split"],
            metadata=value.get("metadata", {}),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-compatible flat entry."""

        result: Dict[str, Any] = dict(self.identity.to_dict())
        result.update(
            {
                "domain": self.domain,
                "split": self.split,
                "metadata": _canonical_json_copy(self.metadata, "metadata"),
            }
        )
        return result


@dataclass(frozen=True)
class DatasetManifest:
    """Versioned episode collection with strict leakage checks."""

    episodes: Sequence[EpisodeManifestEntry]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = MANIFEST_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != MANIFEST_SCHEMA_VERSION:
            raise ValueError(
                "unsupported dataset manifest schema_version: "
                f"expected={MANIFEST_SCHEMA_VERSION}; actual={self.schema_version!r}"
            )
        episodes = tuple(self.episodes)
        if not episodes:
            raise ValueError("dataset manifest must contain at least one episode")
        if not all(isinstance(entry, EpisodeManifestEntry) for entry in episodes):
            raise TypeError("episodes must contain only EpisodeManifestEntry values")
        object.__setattr__(self, "episodes", episodes)
        object.__setattr__(self, "metadata", _canonical_json_copy(self.metadata, "metadata"))
        validate_disjoint_splits(self.split_populations())

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, Any],
        *,
        base_dir: Optional[Path] = None,
        verify_files: bool = False,
    ) -> "DatasetManifest":
        """Parse and validate a manifest mapping."""

        _require_mapping(value, "dataset manifest")
        allowed_fields = {"schema_version", "episodes", "metadata"}
        unexpected = sorted(set(value) - allowed_fields)
        if unexpected:
            raise ValueError(f"dataset manifest has unexpected fields: {unexpected}")
        for key in ("schema_version", "episodes"):
            if key not in value:
                raise ValueError(f"dataset manifest is missing required field: {key}")
        raw_episodes = value["episodes"]
        if not isinstance(raw_episodes, list):
            raise ValueError("dataset manifest episodes must be a JSON array")
        episodes = tuple(
            EpisodeManifestEntry.from_dict(
                entry,
                base_dir=base_dir,
                verify_file=verify_files,
            )
            for entry in raw_episodes
        )
        return cls(
            schema_version=value["schema_version"],
            episodes=episodes,
            metadata=value.get("metadata", {}),
        )

    @classmethod
    def load(cls, path: Union[str, Path], *, verify_files: bool = True) -> "DatasetManifest":
        """Read a UTF-8 JSON manifest and resolve paths relative to it."""

        manifest_path = _resolve_path(path, base_dir=None, require_file=True)
        try:
            raw_text = manifest_path.read_text(encoding="utf-8")
            value = json.loads(
                raw_text,
                object_pairs_hook=_reject_duplicate_json_keys,
                parse_constant=_reject_nonfinite_json_constant,
            )
        except UnicodeDecodeError as error:
            raise ValueError(f"dataset manifest is not valid UTF-8: {manifest_path}") from error
        except json.JSONDecodeError as error:
            raise ValueError(f"dataset manifest is not valid JSON: {manifest_path}: {error}") from error
        return cls.from_dict(
            value,
            base_dir=manifest_path.parent,
            verify_files=verify_files,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Return the canonical JSON-compatible manifest mapping."""

        return {
            "schema_version": self.schema_version,
            "episodes": [entry.to_dict() for entry in self.episodes],
            "metadata": _canonical_json_copy(self.metadata, "metadata"),
        }

    @property
    def content_sha256(self) -> str:
        """Hash the canonical JSON representation of the whole manifest."""

        return canonical_json_sha256(self.to_dict())

    def split_populations(self) -> Dict[str, Tuple[EpisodeIdentity, ...]]:
        """Return episode identities grouped by train/val/test split."""

        return {
            split: tuple(
                entry.identity for entry in self.episodes if entry.split == split
            )
            for split in SPLIT_NAMES
        }


def validate_episode_uuid_provenance(
    manifest: DatasetManifest, *, allow_derived: bool = False
) -> None:
    """Require explicit, internally consistent UUID-source provenance.

    Formal new-data protocols accept only UUIDs stored with the episode.  The
    deterministic SHA-derived source remains available solely for explicitly
    enabled historical compatibility checks.
    """

    if not isinstance(manifest, DatasetManifest):
        raise TypeError("manifest must be a DatasetManifest")
    if not isinstance(allow_derived, bool):
        raise TypeError("allow_derived must be a bool")
    known_sources = set(NATIVE_EPISODE_UUID_SOURCES) | {
        DERIVED_EPISODE_UUID_SOURCE
    }
    derived_count = 0
    for entry in manifest.episodes:
        source = entry.metadata.get("episode_uuid_source")
        if source not in known_sources:
            raise ValueError(
                "dataset manifest has missing or disallowed episode UUID provenance: "
                f"path={entry.identity.path} source={source!r}"
            )
        if source == DERIVED_EPISODE_UUID_SOURCE:
            derived_count += 1
    recorded_derive_mode = manifest.metadata.get("derive_uuid_from_sha256")
    if not isinstance(recorded_derive_mode, bool):
        raise ValueError(
            "dataset manifest must record boolean derive_uuid_from_sha256 provenance"
        )
    if recorded_derive_mode != bool(derived_count):
        raise ValueError(
            "dataset manifest derive_uuid_from_sha256 metadata disagrees with "
            "episode UUID sources"
        )
    if derived_count and not allow_derived:
        raise ValueError(
            "formal training forbids SHA-derived episode UUIDs; use native UUID "
            "metadata for newly collected episodes"
        )


IdentityLike = Union[EpisodeIdentity, EpisodeManifestEntry, Mapping[str, Any]]


def validate_disjoint_splits(
    split_populations: Mapping[str, Iterable[IdentityLike]],
) -> None:
    """Reject duplicates within or across train/val/test populations.

    Leakage is detected independently by UUID, canonical path, and file digest.
    Thus a copied or renamed episode cannot evade the validation.
    """

    _require_mapping(split_populations, "split populations")
    unexpected = sorted(set(split_populations) - set(SPLIT_NAMES))
    if unexpected:
        raise ValueError(f"unknown split population names: {unexpected}")
    prepared = {
        split: _prepare_population(split, split_populations.get(split, ()))
        for split in SPLIT_NAMES
    }
    for left_index, left_name in enumerate(SPLIT_NAMES):
        for right_name in SPLIT_NAMES[left_index + 1 :]:
            _raise_on_population_overlap(
                left_name,
                prepared[left_name],
                right_name,
                prepared[right_name],
            )


def validate_normalization_population(
    normalization_population: Iterable[IdentityLike],
    split_populations: Mapping[str, Iterable[IdentityLike]],
) -> None:
    """Require normalization data to equal the complete training union.

    Validation and test identities are checked for leakage on every identity
    axis.  A later training stage can use a subset of this union via
    :func:`validate_stage_population` while continuing to reuse the same stats.
    """

    if "train" not in split_populations:
        raise ValueError("split populations must include an explicit train population")
    validate_disjoint_splits(split_populations)
    normalization = _prepare_population("normalization", normalization_population)
    if not normalization:
        raise ValueError("normalization population must not be empty")
    train = _prepare_population("train", split_populations["train"])
    expected = {_identity_key(identity) for identity in train}
    actual = {_identity_key(identity) for identity in normalization}
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if missing or unexpected:
        raise ValueError(
            "normalization population must exactly equal the train union: "
            f"missing={_format_identity_keys(missing)}; "
            f"unexpected={_format_identity_keys(unexpected)}"
        )
    for split in ("val", "test"):
        evaluation = _prepare_population(split, split_populations.get(split, ()))
        _raise_on_population_overlap("normalization", normalization, split, evaluation)


def validate_stage_population(
    stage_population: Iterable[IdentityLike],
    normalization_population: Iterable[IdentityLike],
    *,
    evaluation_populations: Optional[Mapping[str, Iterable[IdentityLike]]] = None,
) -> None:
    """Require a stage's episodes to be a non-empty subset of stats data.

    ``evaluation_populations`` may contain named validation/test domains.  They
    are rejected if they overlap either the stage or normalization population
    by UUID, canonical path, or file SHA-256.
    """

    stage = _prepare_population("stage", stage_population)
    normalization = _prepare_population("normalization", normalization_population)
    if not stage:
        raise ValueError("stage population must not be empty")
    if not normalization:
        raise ValueError("normalization population must not be empty")
    normalization_keys = {_identity_key(identity) for identity in normalization}
    outside = sorted(
        _identity_key(identity)
        for identity in stage
        if _identity_key(identity) not in normalization_keys
    )
    if outside:
        raise ValueError(
            "stage population is not a subset of the normalization train union: "
            f"outside={_format_identity_keys(outside)}"
        )
    if evaluation_populations is None:
        return
    _require_mapping(evaluation_populations, "evaluation populations")
    for name, values in evaluation_populations.items():
        evaluation = _prepare_population(
            _validate_nonempty_string(name, "evaluation population name"),
            values,
        )
        _raise_on_population_overlap("stage", stage, name, evaluation)
        _raise_on_population_overlap("normalization", normalization, name, evaluation)


def _resolve_path(
    value: Union[str, Path],
    *,
    base_dir: Optional[Path],
    require_file: bool,
) -> Path:
    if not isinstance(value, (str, Path)):
        raise TypeError(f"path must be a string or Path, got {type(value).__name__}")
    if isinstance(value, str) and not value.strip():
        raise ValueError("path must not be empty")
    path = Path(value).expanduser()
    if not path.is_absolute():
        if base_dir is None:
            raise ValueError(f"relative path requires a base directory: {value!s}")
        path = base_dir.expanduser().resolve() / path
    resolved = path.resolve()
    if require_file:
        if not resolved.exists():
            raise FileNotFoundError(f"episode file does not exist: {resolved}")
        if not resolved.is_file():
            raise FileNotFoundError(f"episode path is not a file: {resolved}")
    return resolved


def _validate_uuid(value: Any) -> str:
    if not isinstance(value, str):
        raise TypeError("episode_uuid must be a string")
    try:
        parsed = uuid.UUID(value)
    except (AttributeError, ValueError) as error:
        raise ValueError(f"episode_uuid is not a valid UUID: {value!r}") from error
    canonical = str(parsed)
    if value != canonical:
        raise ValueError(
            "episode_uuid must use canonical lowercase hyphenated form: "
            f"expected={canonical!r}; actual={value!r}"
        )
    return canonical


def _validate_sha256(value: Any) -> str:
    if not isinstance(value, str):
        raise TypeError("file_sha256 must be a string")
    if _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError("file_sha256 must be exactly 64 lowercase hexadecimal characters")
    return value


def _validate_nonempty_string(value: Any, description: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{description} must be a string")
    if not value or value != value.strip():
        raise ValueError(f"{description} must be non-empty and have no surrounding whitespace")
    return value


def _canonical_json_copy(value: Any, description: str) -> Any:
    if not isinstance(value, Mapping):
        raise TypeError(f"{description} must be a mapping")
    try:
        encoded = canonical_json_bytes(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{description} must contain only finite JSON values") from error
    return json.loads(encoded.decode("utf-8"))


def _validate_json_value(value: Any, path: str) -> None:
    """Reject Python values that do not have an unambiguous JSON identity."""

    if value is None or isinstance(value, (bool, str, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"canonical JSON contains a non-finite number at {path}")
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise TypeError(
                    "canonical JSON object keys must be strings: "
                    f"path={path}; key={key!r}"
                )
            _validate_json_value(child, f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _validate_json_value(child, f"{path}[{index}]")
        return
    raise TypeError(
        "canonical JSON contains an unsupported value: "
        f"path={path}; type={type(value).__name__}"
    )


def _require_mapping(value: Any, description: str) -> None:
    if not isinstance(value, Mapping):
        raise ValueError(f"{description} must be a JSON object")


def _coerce_identity(value: IdentityLike) -> EpisodeIdentity:
    if isinstance(value, EpisodeIdentity):
        return value
    if isinstance(value, EpisodeManifestEntry):
        return value.identity
    if isinstance(value, Mapping):
        return EpisodeIdentity.from_dict(value)
    raise TypeError(
        "population entries must be EpisodeIdentity, EpisodeManifestEntry, or mappings"
    )


def _prepare_population(name: str, values: Iterable[IdentityLike]) -> Tuple[EpisodeIdentity, ...]:
    if isinstance(values, (str, bytes, Mapping)):
        raise TypeError(f"{name} population must be an iterable of episode identities")
    identities = tuple(_coerce_identity(value) for value in values)
    for field_name in _IDENTITY_FIELDS:
        seen: Dict[str, int] = {}
        for index, identity in enumerate(identities):
            field_value = _identity_field(identity, field_name)
            if field_value in seen:
                raise ValueError(
                    f"{name} population contains duplicate {field_name}: "
                    f"value={field_value!r}; first_index={seen[field_value]}; "
                    f"duplicate_index={index}"
                )
            seen[field_value] = index
    return identities


def _identity_field(identity: EpisodeIdentity, field_name: str) -> str:
    if field_name == "path":
        return str(identity.path)
    return str(getattr(identity, field_name))


def _identity_key(identity: EpisodeIdentity) -> Tuple[str, str, str]:
    return (
        identity.episode_uuid,
        str(identity.path),
        identity.file_sha256,
    )


def _raise_on_population_overlap(
    left_name: str,
    left: Sequence[EpisodeIdentity],
    right_name: str,
    right: Sequence[EpisodeIdentity],
) -> None:
    for field_name in _IDENTITY_FIELDS:
        left_values = {_identity_field(identity, field_name) for identity in left}
        right_values = {_identity_field(identity, field_name) for identity in right}
        overlap = sorted(left_values & right_values)
        if overlap:
            raise ValueError(
                "episode leakage between populations: "
                f"left={left_name!r}; right={right_name!r}; "
                f"identity_field={field_name!r}; overlap={overlap}"
            )


def _format_identity_keys(values: Sequence[Tuple[str, str, str]]) -> Sequence[str]:
    return [
        f"uuid={episode_uuid},path={path},sha256={file_sha256}"
        for episode_uuid, path, file_sha256 in values
    ]


def _reject_duplicate_json_keys(pairs: Sequence[Tuple[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"dataset manifest contains duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_nonfinite_json_constant(value: str) -> None:
    raise ValueError(f"dataset manifest contains non-finite JSON value: {value}")
