#!/usr/bin/env python3
"""Build a strict, auditable phase catalog from manual CSV annotations.

The builder deliberately does not infer phases from force or motion signals.  It
only validates human/acquisition-metadata annotations against the exact sample
indices produced by :class:`ContactForceHDF5Dataset` for the supplied dataset
semantics.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, DefaultDict, Mapping, Optional, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from force_aware_act.data.contact_force_hdf5_dataset import (  # noqa: E402
    ACTION_MODE_TO_DATASET,
    ContactForceHDF5Dataset,
)
from force_aware_act.data.manifest import (  # noqa: E402
    DatasetManifest,
    EpisodeManifestEntry,
    validate_episode_uuid_provenance,
)
from force_aware_act.training.catalog import PHASE_CATALOG_SCHEMA_VERSION  # noqa: E402
from force_aware_act.utils.episode_paths import resolve_episode_paths  # noqa: E402


BUILDER_VERSION = "2.0.0"
REQUIRED_COLUMNS = ("episode_path", "start", "stop", "phase")
_INTEGER_PATTERN = re.compile(r"(?:0|[1-9][0-9]*)\Z")
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")


@dataclass(frozen=True)
class _AnnotationSegment:
    episode_path: Path
    start: int
    stop: int
    phase: str
    line_number: int


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_repo_file(raw_path: Path, description: str) -> Path:
    candidate = raw_path.expanduser()
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    resolved = candidate.resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"{description} does not exist: {resolved}")
    if not resolved.is_file():
        raise FileNotFoundError(f"{description} is not a file: {resolved}")
    return resolved


def _parse_csv_integer(raw_value: Optional[str], *, field: str, line_number: int) -> int:
    if raw_value is None:
        raise ValueError(f"annotation line {line_number}: missing {field}")
    value = raw_value.strip()
    if not _INTEGER_PATTERN.fullmatch(value):
        raise ValueError(
            f"annotation line {line_number}: {field} must be a canonical "
            f"non-negative integer, got {raw_value!r}"
        )
    return int(value)


def _resolve_annotation_episode(
    raw_value: Optional[str],
    *,
    line_number: int,
    annotation_parent: Path,
    episode_list_parent: Path,
    allowed_paths: set[Path],
) -> Path:
    if raw_value is None or not raw_value.strip():
        raise ValueError(f"annotation line {line_number}: episode_path must be non-empty")
    raw_path = Path(raw_value.strip()).expanduser()
    if raw_path.is_absolute():
        candidates = [raw_path.resolve(strict=False)]
    else:
        candidates = [
            (REPO_ROOT / raw_path).resolve(strict=False),
            (annotation_parent / raw_path).resolve(strict=False),
            (episode_list_parent / raw_path).resolve(strict=False),
        ]
    matching = sorted(set(candidates) & allowed_paths, key=str)
    if not matching:
        attempted = ", ".join(str(path) for path in dict.fromkeys(candidates))
        raise ValueError(
            f"annotation line {line_number}: episode_path {raw_value!r} is not in "
            f"the specified episode list; attempted canonical paths: {attempted}"
        )
    if len(matching) > 1:
        raise ValueError(
            f"annotation line {line_number}: ambiguous episode_path {raw_value!r}; "
            f"it matches multiple listed episodes: {', '.join(str(path) for path in matching)}"
        )
    return matching[0]


def _read_manual_annotations(
    annotation_path: Path,
    *,
    episode_list_path: Path,
    episode_paths: Sequence[Path],
) -> Mapping[Path, tuple[_AnnotationSegment, ...]]:
    allowed_paths = set(episode_paths)
    grouped: DefaultDict[Path, list[_AnnotationSegment]] = defaultdict(list)
    seen_rows: set[tuple[Path, int, int, str]] = set()

    with annotation_path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source, strict=True)
        fieldnames = reader.fieldnames
        if fieldnames is None:
            raise ValueError("annotation CSV is empty or has no header")
        duplicates = sorted(name for name, count in Counter(fieldnames).items() if count > 1)
        if duplicates:
            raise ValueError(
                "annotation CSV contains duplicate columns: "
                + ", ".join(repr(name) for name in duplicates)
            )
        missing = [column for column in REQUIRED_COLUMNS if column not in fieldnames]
        if missing:
            raise ValueError(
                "annotation CSV is missing required columns: " + ", ".join(missing)
            )

        try:
            for row in reader:
                line_number = reader.line_num
                if None in row:
                    raise ValueError(
                        f"annotation line {line_number}: too many CSV fields for the header"
                    )
                episode_path = _resolve_annotation_episode(
                    row.get("episode_path"),
                    line_number=line_number,
                    annotation_parent=annotation_path.parent,
                    episode_list_parent=episode_list_path.parent,
                    allowed_paths=allowed_paths,
                )
                start = _parse_csv_integer(row.get("start"), field="start", line_number=line_number)
                stop = _parse_csv_integer(row.get("stop"), field="stop", line_number=line_number)
                if stop <= start:
                    raise ValueError(
                        f"annotation line {line_number}: stop must be greater than start; "
                        f"got [{start},{stop})"
                    )
                raw_phase = row.get("phase")
                if raw_phase is None or not raw_phase.strip():
                    raise ValueError(f"annotation line {line_number}: phase must be non-empty")
                phase = raw_phase.strip()
                identity = (episode_path, start, stop, phase)
                if identity in seen_rows:
                    raise ValueError(
                        f"annotation line {line_number}: duplicate segment for {episode_path}: "
                        f"[{start},{stop}) phase={phase!r}"
                    )
                seen_rows.add(identity)
                grouped[episode_path].append(
                    _AnnotationSegment(
                        episode_path=episode_path,
                        start=start,
                        stop=stop,
                        phase=phase,
                        line_number=line_number,
                    )
                )
        except csv.Error as error:
            raise ValueError(
                f"invalid annotation CSV near line {reader.line_num}: {error}"
            ) from error

    missing_episodes = sorted(set(episode_paths) - set(grouped), key=str)
    if missing_episodes:
        raise ValueError(
            "annotation CSV has no segments for listed episodes: "
            + ", ".join(str(path) for path in missing_episodes)
        )
    return {
        path: tuple(
            sorted(segments, key=lambda segment: (segment.start, segment.stop, segment.phase))
        )
        for path, segments in grouped.items()
    }


def _dataset_indices_by_episode(
    episode_paths: Sequence[Path],
    args: argparse.Namespace,
) -> Mapping[Path, tuple[int, ...]]:
    dataset = ContactForceHDF5Dataset(
        episode_paths,
        camera_names=tuple(args.camera_names),
        action_mode=args.action_mode,
        chunk_len=args.chunk_len,
        force_window_len=args.force_window_len,
        force_window_duration=args.force_window_duration,
        image_size=tuple(args.image_size),
        normalize_images=args.normalize_images,
        imagenet_normalize=args.imagenet_normalize,
        image_alignment=args.image_alignment,
        max_image_lag_seconds=args.max_image_lag_seconds,
        include_force=True,
        tolerate_length_mismatch=args.tolerate_length_mismatch,
        max_length_mismatch=args.max_length_mismatch,
    )
    grouped: dict[Path, list[int]] = {path: [] for path in episode_paths}
    for item in dataset.indices:
        canonical_path = Path(item.episode_path).expanduser().resolve()
        if canonical_path not in grouped:
            raise RuntimeError(
                f"dataset produced an index for an unexpected episode: {canonical_path}"
            )
        grouped[canonical_path].append(int(item.state_index))
    return {path: tuple(sorted(indices)) for path, indices in grouped.items()}


def _validate_exact_coverage(
    episode_path: Path,
    segments: Sequence[_AnnotationSegment],
    dataset_indices: Sequence[int],
) -> None:
    if not dataset_indices:
        raise ValueError(
            "listed episode has no usable dataset indices under the requested semantics: "
            f"{episode_path}"
        )
    if len(set(dataset_indices)) != len(dataset_indices):
        raise RuntimeError(f"dataset produced duplicate state indices for {episode_path}")
    for previous, current in zip(dataset_indices, dataset_indices[1:]):
        if current != previous + 1:
            raise RuntimeError(
                f"dataset produced non-contiguous state indices for {episode_path}: "
                f"{previous} followed by {current}"
            )

    previous_segment: Optional[_AnnotationSegment] = None
    for segment in segments:
        if previous_segment is not None:
            if segment.start < previous_segment.stop:
                raise ValueError(
                    f"annotation overlap for {episode_path}: lines "
                    f"{previous_segment.line_number} and {segment.line_number} cover "
                    f"[{previous_segment.start},{previous_segment.stop}) and "
                    f"[{segment.start},{segment.stop})"
                )
            if segment.start > previous_segment.stop:
                raise ValueError(
                    f"annotation gap for {episode_path}: no phase covers "
                    f"[{previous_segment.stop},{segment.start})"
                )
        previous_segment = segment

    expected_start = dataset_indices[0]
    expected_stop = dataset_indices[-1] + 1
    actual_start = segments[0].start
    actual_stop = segments[-1].stop
    if actual_start != expected_start or actual_stop != expected_stop:
        missing_prefix = (
            f" uncovered prefix [{expected_start},{actual_start})"
            if actual_start > expected_start
            else ""
        )
        missing_suffix = (
            f" uncovered suffix [{actual_stop},{expected_stop})"
            if actual_stop < expected_stop
            else ""
        )
        raise ValueError(
            f"annotation coverage does not exactly match dataset indices for {episode_path}: "
            f"expected [{expected_start},{expected_stop}), got [{actual_start},{actual_stop});"
            f"{missing_prefix}{missing_suffix}"
        )
    if len(dataset_indices) != actual_stop - actual_start:
        raise RuntimeError(
            f"dataset index coverage cardinality mismatch for {episode_path}: "
            f"indices={len(dataset_indices)}, annotation_span={actual_stop - actual_start}"
        )


def _validate_args(args: argparse.Namespace) -> None:
    if not isinstance(args.dataset_manifest_sha256, str) or not _SHA256_PATTERN.fullmatch(
        args.dataset_manifest_sha256
    ):
        raise ValueError(
            "--dataset-manifest-sha256 must be exactly 64 lowercase hexadecimal characters"
        )
    if not isinstance(args.source_domain, str) or not args.source_domain.strip():
        raise ValueError("--source-domain must be non-empty")
    if args.source_domain != args.source_domain.strip():
        raise ValueError("--source-domain must not have surrounding whitespace")
    if args.chunk_len <= 0:
        raise ValueError("--chunk-len must be positive")
    if args.force_window_len <= 0:
        raise ValueError("--force-window-len must be positive")
    if not math.isfinite(args.force_window_duration) or args.force_window_duration < 0:
        raise ValueError("--force-window-duration must be non-negative and finite")
    if len(args.image_size) != 2 or any(dimension <= 0 for dimension in args.image_size):
        raise ValueError("--image-size must contain two positive integers")
    if not args.camera_names or any(not name.strip() for name in args.camera_names):
        raise ValueError("--camera-names must contain non-empty names")
    if len(set(args.camera_names)) != len(args.camera_names):
        raise ValueError("--camera-names must not contain duplicates")
    if args.imagenet_normalize and not args.normalize_images:
        raise ValueError("--imagenet-normalize requires --normalize-images")
    if args.max_length_mismatch < 0:
        raise ValueError("--max-length-mismatch must be non-negative")
    if args.image_alignment != "latest_past":
        raise ValueError("--image-alignment must be latest_past")
    if (
        not math.isfinite(args.max_image_lag_seconds)
        or args.max_image_lag_seconds < 0.0
    ):
        raise ValueError("--max-image-lag-seconds must be finite and non-negative")


def build_phase_catalog(args: argparse.Namespace) -> tuple[Path, str]:
    """Validate inputs, exclusively create the catalog, and return path/hash."""

    _validate_args(args)
    annotation_path = _resolve_repo_file(args.annotation, "annotation CSV")
    episode_list_path = _resolve_repo_file(args.episode_list, "episode list")
    manifest_path = _resolve_repo_file(args.dataset_manifest, "dataset manifest")
    output_path = args.output.expanduser()
    if not output_path.is_absolute():
        output_path = REPO_ROOT / output_path
    output_path = output_path.resolve(strict=False)
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output_path}")

    raw_episode_paths = resolve_episode_paths(
        [],
        episode_list_path,
        project_root=REPO_ROOT,
        deduplicate=False,
    )
    if not raw_episode_paths:
        raise ValueError(f"episode list is empty: {episode_list_path}")
    episode_paths = tuple(path.expanduser().resolve() for path in raw_episode_paths)
    duplicate_paths = sorted(
        (path for path, count in Counter(episode_paths).items() if count > 1),
        key=str,
    )
    if duplicate_paths:
        raise ValueError(
            "episode list contains duplicate canonical paths: "
            + ", ".join(str(path) for path in duplicate_paths)
        )

    manifest = DatasetManifest.load(manifest_path, verify_files=True)
    validate_episode_uuid_provenance(manifest, allow_derived=False)
    if manifest.content_sha256 != args.dataset_manifest_sha256:
        raise ValueError(
            "dataset manifest SHA256 mismatch: "
            f"expected={args.dataset_manifest_sha256} actual={manifest.content_sha256}"
        )
    manifest_entries = tuple(
        entry
        for entry in manifest.episodes
        if entry.split == "train" and entry.domain == args.source_domain
    )
    if not manifest_entries:
        raise ValueError(
            "dataset manifest has no train entries for source domain "
            f"{args.source_domain!r}"
        )
    manifest_by_path: dict[Path, EpisodeManifestEntry] = {
        entry.identity.path: entry for entry in manifest_entries
    }
    listed = set(episode_paths)
    expected = set(manifest_by_path)
    if listed != expected:
        raise ValueError(
            "episode list must exactly equal dataset manifest train entries for "
            f"domain={args.source_domain!r}: "
            f"manifest_only={len(expected - listed)} list_only={len(listed - expected)}"
        )

    annotations = _read_manual_annotations(
        annotation_path,
        episode_list_path=episode_list_path,
        episode_paths=episode_paths,
    )
    indices_by_episode = _dataset_indices_by_episode(episode_paths, args)
    for episode_path in episode_paths:
        _validate_exact_coverage(
            episode_path,
            annotations[episode_path],
            indices_by_episode[episode_path],
        )

    dataset_semantics = {
        "action_dataset": ACTION_MODE_TO_DATASET[args.action_mode],
        "action_mode": args.action_mode,
        "action_offset": 1 if args.action_mode == "joint_pos" else 0,
        "camera_names": list(args.camera_names),
        "chunk_len": args.chunk_len,
        "force_window_duration": args.force_window_duration,
        "force_window_len": args.force_window_len,
        "image_size": list(args.image_size),
        "imagenet_normalize": bool(args.imagenet_normalize),
        "image_alignment": args.image_alignment,
        "max_image_lag_seconds": args.max_image_lag_seconds,
        "include_force": True,
        "max_length_mismatch": args.max_length_mismatch,
        "normalize_images": bool(args.normalize_images),
        "tolerate_length_mismatch": bool(args.tolerate_length_mismatch),
    }
    document = {
        "schema_version": PHASE_CATALOG_SCHEMA_VERSION,
        "episodes": [
            {
                "domain": manifest_by_path[episode_path].domain,
                "episode_uuid": manifest_by_path[episode_path].identity.episode_uuid,
                "file_sha256": manifest_by_path[episode_path].identity.file_sha256,
                "path": str(episode_path),
                "segments": [
                    {"start": segment.start, "stop": segment.stop, "phase": segment.phase}
                    for segment in annotations[episode_path]
                ],
            }
            for episode_path in episode_paths
        ],
        "labeler": {
            "annotation_path": str(annotation_path),
            "annotation_sha256": _sha256_file(annotation_path),
            "builder": "scripts/build_phase_catalog.py",
            "builder_version": BUILDER_VERSION,
            "coverage": "exact_complete_dataset_state_indices",
            "dataset_manifest_path": str(manifest_path),
            "dataset_manifest_sha256": manifest.content_sha256,
            "dataset_index_counts": {
                str(path): len(indices_by_episode[path]) for path in episode_paths
            },
            "dataset_semantics": dataset_semantics,
            "episode_list_path": str(episode_list_path),
            "episode_list_sha256": _sha256_file(episode_list_path),
            "phase_labels": sorted(
                {segment.phase for segments in annotations.values() for segment in segments}
            ),
            "phase_source": "manual_csv_only",
        },
    }
    encoded = _canonical_json_bytes(document)
    content_sha256 = hashlib.sha256(encoded).hexdigest()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("xb") as destination:
        destination.write(encoded)
    return output_path, content_sha256


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a phase catalog from manual CSV labels and verify exact coverage "
            "of ContactForceHDF5Dataset indices. No phase inference is performed."
        )
    )
    parser.add_argument("--annotation", type=Path, required=True, help="Manual annotation CSV.")
    parser.add_argument("--episode-list", type=Path, required=True)
    parser.add_argument(
        "--dataset-manifest",
        type=Path,
        required=True,
        help="Pinned dataset manifest containing this source's train population.",
    )
    parser.add_argument(
        "--dataset-manifest-sha256",
        required=True,
        help="Expected canonical dataset manifest content SHA-256.",
    )
    parser.add_argument("--source-domain", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--action-mode",
        choices=tuple(sorted(ACTION_MODE_TO_DATASET)),
        required=True,
    )
    parser.add_argument("--chunk-len", type=int, required=True)
    parser.add_argument("--force-window-len", type=int, required=True)
    parser.add_argument("--force-window-duration", type=float, required=True)
    parser.add_argument("--camera-names", nargs="+", required=True)
    parser.add_argument(
        "--image-size",
        nargs=2,
        type=int,
        required=True,
        metavar=("HEIGHT", "WIDTH"),
    )

    image_group = parser.add_mutually_exclusive_group()
    image_group.add_argument("--normalize-images", dest="normalize_images", action="store_true")
    image_group.add_argument("--no-normalize-images", dest="normalize_images", action="store_false")
    parser.set_defaults(normalize_images=True)
    parser.add_argument("--imagenet-normalize", action="store_true")
    parser.add_argument(
        "--image-alignment",
        choices=("latest_past",),
        required=True,
    )
    parser.add_argument("--max-image-lag-seconds", type=float, required=True)

    length_group = parser.add_mutually_exclusive_group()
    length_group.add_argument(
        "--strict-lengths",
        dest="tolerate_length_mismatch",
        action="store_false",
        help="Require exact synchronized and action lengths (default).",
    )
    length_group.add_argument(
        "--tolerate-length-mismatch",
        dest="tolerate_length_mismatch",
        action="store_true",
        help="Allow mismatches up to --max-length-mismatch.",
    )
    parser.set_defaults(tolerate_length_mismatch=False)
    parser.add_argument("--max-length-mismatch", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        output_path, content_sha256 = build_phase_catalog(args)
    except Exception as error:
        print(f"error: failed to build phase catalog: {error}", file=sys.stderr)
        return 1
    print(f"saved_phase_catalog={output_path}")
    print(f"phase_catalog_sha256={content_sha256}")
    print("phase_source=manual_csv_only")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
