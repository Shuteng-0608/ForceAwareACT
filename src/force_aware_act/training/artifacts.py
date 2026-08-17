"""Traceable checkpoint ancestry and best-policy artifact metadata."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, Mapping, Optional


CHECKPOINT_LINEAGE_VERSION = "checkpoint_lineage_v1"
BEST_ARTIFACT_VERSION = "best_policy_artifact_v1"
SELECTION_MIGRATION_VERSION = "selection_metric_migration_v1"


def checkpoint_identity(
    path: Path,
    *,
    payload: Optional[Mapping[str, Any]] = None,
    chunk_bytes: int = 8 * 1024 * 1024,
) -> dict[str, Any]:
    """Return a content-addressed identity for a resume checkpoint."""

    checkpoint = Path(path).resolve(strict=True)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint is not a file: {checkpoint}")
    if chunk_bytes <= 0:
        raise ValueError("chunk_bytes must be positive")

    digest = hashlib.sha256()
    with checkpoint.open("rb") as stream:
        while block := stream.read(chunk_bytes):
            digest.update(block)
    stat = checkpoint.stat()
    progress = {} if payload is None else payload.get("progress", {})
    return {
        "version": CHECKPOINT_LINEAGE_VERSION,
        "path": str(checkpoint),
        "size_bytes": int(stat.st_size),
        "sha256": digest.hexdigest(),
        "format_version": (
            None if payload is None else payload.get("format_version")
        ),
        "global_step": (
            None
            if not isinstance(progress, Mapping)
            or "global_step" not in progress
            else int(progress["global_step"])
        ),
    }


def best_policy_artifact(
    path: Path,
    *,
    metric_name: str,
    metric: float,
    global_step: int,
    storage: str,
) -> dict[str, Any]:
    """Describe the exact artifact selected by a validation metric."""

    if not metric_name:
        raise ValueError("metric_name must be non-empty")
    if global_step < 0:
        raise ValueError("global_step must be non-negative")
    if not storage:
        raise ValueError("storage must be non-empty")
    return {
        "version": BEST_ARTIFACT_VERSION,
        "path": str(Path(path).resolve()),
        "metric_name": metric_name,
        "metric": float(metric),
        "global_step": int(global_step),
        "storage": storage,
    }


def selection_metric_migration(
    *,
    source_metric_name: Optional[str],
    source_best_metric: Optional[float],
    target_metric_name: str,
    target_metric: float,
    global_step: int,
    reason: str,
) -> dict[str, Any]:
    """Record why a historical best value was re-based on resume."""

    if not target_metric_name:
        raise ValueError("target_metric_name must be non-empty")
    if not reason:
        raise ValueError("reason must be non-empty")
    return {
        "version": SELECTION_MIGRATION_VERSION,
        "source_metric_name": source_metric_name,
        "source_best_metric": (
            None
            if source_best_metric is None
            else float(source_best_metric)
        ),
        "target_metric_name": target_metric_name,
        "target_metric": float(target_metric),
        "global_step": int(global_step),
        "reason": reason,
    }


def materialize_best_artifact_reference(
    output_path: Path,
    artifact: Mapping[str, Any],
) -> bool:
    """Create a non-copying local reference to a prior best checkpoint.

    A later atomic checkpoint save safely replaces the symlink itself rather
    than overwriting the referenced parent artifact.
    """

    destination = Path(output_path)
    source = Path(str(artifact["path"])).resolve(strict=True)
    if destination.resolve() == source:
        return False
    if destination.exists() or destination.is_symlink():
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(source, destination)
    return True
