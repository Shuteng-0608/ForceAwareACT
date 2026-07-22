#!/usr/bin/env python3
"""Build a strict canonical dataset manifest from episode-list groups."""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import h5py


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from force_aware_act.data.manifest import (  # noqa: E402
    SPLIT_NAMES,
    DatasetManifest,
    EpisodeIdentity,
    EpisodeManifestEntry,
    canonical_json_bytes,
    sha256_file,
)
from force_aware_act.utils import resolve_episode_paths  # noqa: E402


MANIFEST_BUILDER_VERSION = 1
_DERIVED_UUID_NAMESPACE = uuid.UUID("b739db7e-5dd7-4fb8-8832-083078a7ec37")


@dataclass(frozen=True)
class GroupSpec:
    """One domain/split assignment backed by one episode-list file."""

    domain: str
    split: str
    episode_list: Path


def parse_group_spec(value: str) -> GroupSpec:
    """Parse ``DOMAIN:SPLIT=EPISODE_LIST`` without accepting ambiguity."""

    assignment, equals, raw_list = value.partition("=")
    domain, colon, split = assignment.partition(":")
    if (
        not equals
        or not colon
        or not domain
        or domain != domain.strip()
        or not split
        or split != split.strip()
        or not raw_list.strip()
    ):
        raise argparse.ArgumentTypeError(
            "--group must use DOMAIN:SPLIT=EPISODE_LIST"
        )
    if split not in SPLIT_NAMES:
        raise argparse.ArgumentTypeError(
            f"--group SPLIT must be one of {SPLIT_NAMES}, got {split!r}"
        )
    return GroupSpec(domain=domain, split=split, episode_list=Path(raw_list.strip()))


def build_dataset_manifest(
    groups: Sequence[GroupSpec],
    *,
    derive_uuid_from_sha256: bool = False,
) -> DatasetManifest:
    """Resolve, hash, identify, and validate all grouped episodes."""

    if not groups:
        raise ValueError("at least one --group is required")
    if not isinstance(derive_uuid_from_sha256, bool):
        raise TypeError("derive_uuid_from_sha256 must be a bool")
    for group in groups:
        if not isinstance(group, GroupSpec):
            raise TypeError("groups must contain only GroupSpec values")
        if not group.domain or group.domain != group.domain.strip():
            raise ValueError(
                "group domain must be non-empty and have no surrounding whitespace"
            )
        if group.split not in SPLIT_NAMES:
            raise ValueError(
                f"group split must be one of {SPLIT_NAMES}, got {group.split!r}"
            )
        if not isinstance(group.episode_list, Path):
            raise TypeError("group episode_list must be a Path")
    group_keys = [(group.domain, group.split) for group in groups]
    duplicate_group_keys = sorted(
        key for key in set(group_keys) if group_keys.count(key) > 1
    )
    if duplicate_group_keys:
        raise ValueError(
            "duplicate domain/split group assignments: "
            f"{duplicate_group_keys}"
        )

    resolved_groups = []
    entries = []
    split_order = {name: index for index, name in enumerate(SPLIT_NAMES)}
    for group in sorted(
        groups,
        key=lambda item: (item.domain, split_order[item.split], str(item.episode_list)),
    ):
        episode_list = _resolve_episode_list(group.episode_list)
        episode_paths = resolve_episode_paths(
            [],
            episode_list,
            project_root=REPO_ROOT,
            deduplicate=False,
        )
        if not episode_paths:
            raise ValueError(f"episode list is empty: {episode_list}")
        episode_paths = sorted((path.resolve() for path in episode_paths), key=str)
        resolved_groups.append(
            {
                "domain": group.domain,
                "split": group.split,
                "episode_list": str(episode_list),
                "episode_list_sha256": sha256_file(episode_list),
                "episode_count": len(episode_paths),
            }
        )
        for episode_path in episode_paths:
            file_sha256 = sha256_file(episode_path)
            raw_uuid, uuid_source = _read_episode_uuid(
                episode_path,
                file_sha256=file_sha256,
                derive_uuid_from_sha256=derive_uuid_from_sha256,
            )
            try:
                identity = EpisodeIdentity(
                    episode_uuid=raw_uuid,
                    path=episode_path,
                    file_sha256=file_sha256,
                )
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"{episode_path}: invalid episode UUID from {uuid_source}: {error}"
                ) from error
            entries.append(
                EpisodeManifestEntry(
                    identity=identity,
                    domain=group.domain,
                    split=group.split,
                    metadata={"episode_uuid_source": uuid_source},
                )
            )

    entries.sort(
        key=lambda entry: (
            entry.domain,
            split_order[entry.split],
            str(entry.identity.path),
        )
    )
    return DatasetManifest(
        episodes=tuple(entries),
        metadata={
            "manifest_builder": "scripts/build_dataset_manifest.py",
            "manifest_builder_version": MANIFEST_BUILDER_VERSION,
            "derive_uuid_from_sha256": bool(derive_uuid_from_sha256),
            "groups": resolved_groups,
        },
    )


def write_canonical_manifest(manifest: DatasetManifest, output: Path) -> Path:
    """Create, but never replace, a canonical UTF-8 JSON manifest."""

    output = output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_json_bytes(manifest.to_dict()) + b"\n"
    try:
        with output.open("xb") as output_file:
            output_file.write(payload)
    except FileExistsError as error:
        raise FileExistsError(f"refusing to overwrite existing output: {output}") from error
    return output


def _resolve_episode_list(path: Path) -> Path:
    candidate = path.expanduser()
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    resolved = candidate.resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"episode list does not exist: {resolved}")
    if not resolved.is_file():
        raise FileNotFoundError(f"episode list is not a file: {resolved}")
    return resolved


def _read_episode_uuid(
    episode_path: Path,
    *,
    file_sha256: str,
    derive_uuid_from_sha256: bool,
) -> Tuple[str, str]:
    with h5py.File(episode_path, "r") as handle:
        if "episode_uuid" in handle.attrs:
            return (
                _coerce_uuid_value(
                    handle.attrs["episode_uuid"],
                    description=f"{episode_path}: HDF5 root attr episode_uuid",
                ),
                "hdf5_root_attr:episode_uuid",
            )

    metadata_path = episode_path.parent / "metadata.json"
    if metadata_path.exists():
        metadata = _read_metadata_json(metadata_path)
        if "episode_uuid" in metadata:
            return (
                _coerce_uuid_value(
                    metadata["episode_uuid"],
                    description=f"{metadata_path}: episode_uuid",
                ),
                "sibling_metadata_json:episode_uuid",
            )

    if derive_uuid_from_sha256:
        return (
            str(
                uuid.uuid5(
                    _DERIVED_UUID_NAMESPACE,
                    f"force-aware-act:episode-sha256:{file_sha256}",
                )
            ),
            "derived:uuid5(file_sha256)",
        )
    raise ValueError(
        f"{episode_path}: episode UUID is missing from HDF5 root attr "
        "episode_uuid and sibling metadata.json; use --derive-uuid-from-sha256 "
        "only for explicitly approved historical data"
    )


def _coerce_uuid_value(value: Any, *, description: str) -> str:
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            value = value.item()
        except ValueError:
            pass
    if isinstance(value, bytes):
        try:
            value = value.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError(f"{description} is not valid UTF-8") from error
    if not isinstance(value, str) or not value:
        raise ValueError(f"{description} must be a non-empty UUID string")
    return value


def _read_metadata_json(path: Path) -> Dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_nonfinite_json_constant,
        )
    except UnicodeDecodeError as error:
        raise ValueError(f"metadata is not valid UTF-8: {path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"metadata is not valid JSON: {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"metadata must contain a JSON object: {path}")
    return value


def _reject_duplicate_json_keys(pairs: Sequence[Tuple[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"metadata contains duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_nonfinite_json_constant(value: str) -> None:
    raise ValueError(f"metadata contains non-finite JSON value: {value}")


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--group",
        action="append",
        type=parse_group_spec,
        required=True,
        metavar="DOMAIN:SPLIT=EPISODE_LIST",
        help="Repeat for each domain/split episode-list assignment.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--derive-uuid-from-sha256",
        action="store_true",
        help="Explicit compatibility mode for historical episodes without UUID metadata.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    output = args.output.expanduser().resolve()
    if output.exists():
        print(f"error: refusing to overwrite existing output: {output}", file=sys.stderr)
        return 2
    try:
        manifest = build_dataset_manifest(
            args.group,
            derive_uuid_from_sha256=args.derive_uuid_from_sha256,
        )
        written = write_canonical_manifest(manifest, output)
    except (FileNotFoundError, FileExistsError, KeyError, OSError, TypeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(f"manifest={written}")
    print(f"episode_count={len(manifest.episodes)}")
    print(f"manifest_content_sha256={manifest.content_sha256}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
