#!/usr/bin/env python3
"""Evaluate and retention-gate staged-training checkpoint candidates.

Candidate CSV schema (paths are relative to the CSV file when not absolute)::

    candidate_id,checkpoint_path,checkpoint_sha256,epoch,step
    stage2_0100,outputs/checkpoint_step_00000100.pt,<sha256>,3,100

``epoch`` and ``step`` are optional columns; when provided they are verified
against checkpoint metadata. The stage-1 reference is supplied separately and
defines the retention baseline. Outputs are created exclusively and never
replace an existing report.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from force_aware_act.data import (  # noqa: E402
    ContactForceHDF5Dataset,
    DatasetManifest,
    canonical_json_sha256,
    validate_balanced_normalization_contract,
    validate_episode_uuid_provenance,
    validate_normalization_provenance_hashes,
)
from force_aware_act.models import (  # noqa: E402
    ForceAwareACTContactCVAEPolicy,
    ForceAwareACTMotionCVAEPolicy,
    ForceAwareACTPolicy,
)
from force_aware_act.training.checkpointing import (  # noqa: E402
    INIT_COMPATIBILITY_KEYS,
    file_sha256,
    validate_checkpoint_compatibility,
    validate_checkpoint_v2_payload,
)
from force_aware_act.training.control import (  # noqa: E402
    RetentionGatedCheckpointSelector,
    evaluate_named_deployment_metrics,
    resolve_validation_deployment_mode,
)
from force_aware_act.training.engine import validate_normalization_stats  # noqa: E402
from force_aware_act.training.policies import resolved_model_config  # noqa: E402
from force_aware_act.training.protocol import ResolvedProtocol, load_protocol  # noqa: E402
from force_aware_act.utils import resolve_episode_paths  # noqa: E402


CANDIDATE_REQUIRED_COLUMNS = (
    "candidate_id",
    "checkpoint_path",
    "checkpoint_sha256",
)
CANDIDATE_OPTIONAL_COLUMNS = ("epoch", "step")
REPORT_FILENAMES = {
    "metrics": "metrics_long.csv",
    "decisions": "decisions.csv",
    "shortlist_csv": "shortlist.csv",
    "shortlist_json": "shortlist.json",
    "completion": "evaluation_completion.json",
}
METRICS_FIELDS = (
    "candidate_id",
    "role",
    "evaluation_order",
    "checkpoint_path",
    "checkpoint_sha256",
    "checkpoint_stage",
    "checkpoint_stage_index",
    "checkpoint_epoch",
    "checkpoint_step",
    "domain",
    "metric",
    "value",
    "deployment_mode",
    "protocol_sha256",
    "normalization_sha256",
    "episode_list",
    "episode_list_sha256",
    "num_episodes",
)
DECISION_FIELDS = (
    "candidate_id",
    "evaluation_order",
    "checkpoint_path",
    "checkpoint_sha256",
    "checkpoint_epoch",
    "checkpoint_step",
    "metric",
    "objective_domain",
    "objective_value",
    "retention_domain",
    "retention_value",
    "retention_baseline",
    "retention_limit",
    "retention_passed",
    "objective_improved",
    "selected",
    "final_best",
    "reason",
)
SHORTLIST_FIELDS = (
    "rank",
    "candidate_id",
    "checkpoint_path",
    "checkpoint_sha256",
    "checkpoint_epoch",
    "checkpoint_step",
    "metric",
    "objective_value",
    "retention_value",
    "retention_limit",
    "final_selected",
    "fallback_reference",
)
IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
REFERENCE_CANDIDATE_ID = "stage1_reference"
RUN_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")


def _enable_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _runtime_versions() -> Mapping[str, Any]:
    return {
        "python": sys.version.split()[0],
        "torch": str(torch.__version__),
        "numpy": str(np.__version__),
        "h5py": str(h5py.__version__),
        "cuda_runtime": str(torch.version.cuda),
        "cudnn": (
            None
            if not hasattr(torch.backends, "cudnn")
            else torch.backends.cudnn.version()
        ),
    }


def _capture_input_file_hashes(paths: Sequence[Path]) -> Mapping[str, str]:
    snapshots: Dict[str, str] = {}
    for raw_path in paths:
        path = Path(raw_path).expanduser()
        if path.is_symlink():
            raise ValueError(f"formal evaluation input must not be a symlink: {path}")
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"formal evaluation input is missing: {path}")
        snapshots[str(path)] = file_sha256(path)
    return dict(sorted(snapshots.items()))


@dataclass(frozen=True)
class CandidateSpec:
    candidate_id: str
    checkpoint_path: Path
    expected_sha256: str
    expected_epoch: Optional[int]
    expected_step: Optional[int]


@dataclass(frozen=True)
class ValidationDomain:
    name: str
    episode_list: Path
    episode_list_sha256: str
    episode_paths: Tuple[Path, ...]


@dataclass(frozen=True)
class LoadedCheckpoint:
    candidate_id: str
    role: str
    path: Path
    file_sha256: str
    payload: Mapping[str, Any]
    config: Mapping[str, Any]
    stage_name: str
    stage_index: int
    epoch: int
    step: int

    @property
    def stage_step(self) -> int:
        training_state = self.payload.get("training_state")
        if isinstance(training_state, Mapping) and "stage_step" in training_state:
            return int(training_state["stage_step"])
        # Manually constructed unit-test records predate the strict loader.
        return self.step

    @property
    def run_id(self) -> str:
        return str(self.config["run_id"])

    @property
    def run_manifest_sha256(self) -> str:
        return str(self.config["run_manifest_sha256"])


@dataclass
class EvaluatedCandidate:
    checkpoint: LoadedCheckpoint
    evaluation_order: int
    metrics: Mapping[str, Mapping[str, float]]
    decision: Optional[Any] = None


def _validate_sha256(value: str, context: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{context} must be a SHA256 hex digest")
    normalized = value.strip().lower()
    if len(normalized) != 64 or any(c not in "0123456789abcdef" for c in normalized):
        raise ValueError(f"{context} must be a 64-character SHA256 hex digest")
    return normalized


def _positive_optional_int(value: Optional[str], context: str) -> Optional[int]:
    if value is None or not value.strip():
        return None
    try:
        parsed = int(value)
    except ValueError as error:
        raise ValueError(f"{context} must be a positive integer") from error
    if parsed <= 0:
        raise ValueError(f"{context} must be a positive integer")
    return parsed


def _validate_identifier(value: str, context: str) -> str:
    if not isinstance(value, str) or not IDENTIFIER_PATTERN.fullmatch(value):
        raise ValueError(
            f"{context} must match {IDENTIFIER_PATTERN.pattern!r}; got {value!r}"
        )
    return value


def _validate_run_id(value: Any, context: str) -> str:
    if not isinstance(value, str) or RUN_ID_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{context} must be 32 lowercase hexadecimal characters")
    return value


def load_candidate_specs(path: Path) -> Tuple[CandidateSpec, ...]:
    """Load a strict, ordered candidate CSV and resolve checkpoint paths."""

    csv_path = Path(path).expanduser().resolve()
    if not csv_path.is_file():
        raise FileNotFoundError(f"candidate CSV does not exist: {csv_path}")
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("candidate CSV has no header")
        fields = tuple(reader.fieldnames)
        if len(set(fields)) != len(fields):
            raise ValueError("candidate CSV header contains duplicate columns")
        missing = sorted(set(CANDIDATE_REQUIRED_COLUMNS) - set(fields))
        unknown = sorted(
            set(fields) - set(CANDIDATE_REQUIRED_COLUMNS) - set(CANDIDATE_OPTIONAL_COLUMNS)
        )
        if missing:
            raise ValueError("candidate CSV is missing columns: " + ", ".join(missing))
        if unknown:
            raise ValueError("candidate CSV has unknown columns: " + ", ".join(unknown))

        specs = []
        seen_ids = set()
        seen_paths = set()
        for row_index, row in enumerate(reader, start=2):
            if None in row:
                raise ValueError(
                    f"candidate CSV row {row_index} has more values than the header"
                )
            candidate_id = _validate_identifier(
                (row.get("candidate_id") or "").strip(),
                f"candidate CSV row {row_index} candidate_id",
            )
            if candidate_id == REFERENCE_CANDIDATE_ID:
                raise ValueError(
                    f"candidate CSV row {row_index} candidate_id "
                    f"{REFERENCE_CANDIDATE_ID!r} is reserved for the stage-1 reference"
                )
            raw_path = (row.get("checkpoint_path") or "").strip()
            if not raw_path:
                raise ValueError(
                    f"candidate CSV row {row_index} checkpoint_path must not be empty"
                )
            checkpoint_path = Path(raw_path).expanduser()
            if not checkpoint_path.is_absolute():
                checkpoint_path = csv_path.parent / checkpoint_path
            if checkpoint_path.is_symlink():
                raise ValueError(
                    f"candidate checkpoint must not be a symlink at row {row_index}: "
                    f"{checkpoint_path}"
                )
            checkpoint_path = checkpoint_path.resolve()
            if not checkpoint_path.is_file():
                raise FileNotFoundError(
                    f"candidate checkpoint does not exist at row {row_index}: "
                    f"{checkpoint_path}"
                )
            if candidate_id in seen_ids:
                raise ValueError(f"duplicate candidate_id in candidate CSV: {candidate_id}")
            if checkpoint_path in seen_paths:
                raise ValueError(
                    f"duplicate checkpoint_path in candidate CSV: {checkpoint_path}"
                )
            seen_ids.add(candidate_id)
            seen_paths.add(checkpoint_path)
            specs.append(
                CandidateSpec(
                    candidate_id=candidate_id,
                    checkpoint_path=checkpoint_path,
                    expected_sha256=_validate_sha256(
                        row.get("checkpoint_sha256") or "",
                        f"candidate CSV row {row_index} checkpoint_sha256",
                    ),
                    expected_epoch=_positive_optional_int(
                        row.get("epoch"), f"candidate CSV row {row_index} epoch"
                    ),
                    expected_step=_positive_optional_int(
                        row.get("step"), f"candidate CSV row {row_index} step"
                    ),
                )
            )
    if not specs:
        raise ValueError("candidate CSV contains no candidates")
    return tuple(specs)


def parse_validation_domain(value: str) -> Tuple[str, Path]:
    """Parse one ``NAME=EPISODE_LIST`` command-line value."""

    if not isinstance(value, str) or value.count("=") != 1:
        raise ValueError("--val-domain must use NAME=EPISODE_LIST")
    raw_name, raw_path = value.split("=", 1)
    name = _validate_identifier(raw_name.strip(), "validation domain name")
    if not raw_path.strip():
        raise ValueError("validation episode-list path must not be empty")
    return name, Path(raw_path.strip()).expanduser()


def resolve_validation_domains(values: Sequence[str]) -> Tuple[ValidationDomain, ...]:
    """Resolve lists strictly and reject duplicate/leaking validation episodes."""

    if not values:
        raise ValueError("at least one --val-domain is required")
    domains = []
    seen_names = set()
    owner_by_episode: Dict[Path, str] = {}
    for value in values:
        name, list_path = parse_validation_domain(value)
        if name in seen_names:
            raise ValueError(f"duplicate validation domain name: {name}")
        seen_names.add(name)
        if not list_path.is_absolute():
            list_path = REPO_ROOT / list_path
        list_path = list_path.resolve()
        if not list_path.is_file():
            raise FileNotFoundError(f"validation episode list does not exist: {list_path}")
        paths = tuple(
            resolve_episode_paths(
                [], list_path, project_root=REPO_ROOT, deduplicate=False
            )
        )
        if not paths:
            raise ValueError(f"validation episode list is empty: {list_path}")
        canonical = tuple(Path(path).resolve() for path in paths)
        if len(set(canonical)) != len(canonical):
            raise ValueError(f"validation episode list contains duplicates: {list_path}")
        for episode_path in canonical:
            previous_owner = owner_by_episode.get(episode_path)
            if previous_owner is not None:
                raise ValueError(
                    "validation domains overlap: "
                    f"episode={episode_path} domains={previous_owner!r},{name!r}"
                )
            owner_by_episode[episode_path] = name
        domains.append(
            ValidationDomain(
                name=name,
                episode_list=list_path,
                episode_list_sha256=file_sha256(list_path),
                episode_paths=canonical,
            )
        )
    return tuple(domains)


def _torch_load_bytes(payload_bytes: bytes) -> Any:
    try:
        return torch.load(
            io.BytesIO(payload_bytes), map_location="cpu", weights_only=False
        )
    except TypeError:
        return torch.load(io.BytesIO(payload_bytes), map_location="cpu")


def _load_file_snapshot(path: Path, *, role: str) -> Tuple[Any, str, Path]:
    """Read, hash, and deserialize one immutable byte snapshot."""

    candidate = Path(path).expanduser()
    if candidate.is_symlink():
        raise ValueError(f"{role} path must not be a symlink: {candidate}")
    resolved = candidate.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{role} does not exist: {resolved}")
    payload_bytes = resolved.read_bytes()
    return (
        _torch_load_bytes(payload_bytes),
        hashlib.sha256(payload_bytes).hexdigest(),
        resolved,
    )


def load_checkpoint_strict(
    *,
    candidate_id: str,
    role: str,
    path: Path,
    expected_sha256: str,
    expected_epoch: Optional[int] = None,
    expected_step: Optional[int] = None,
) -> LoadedCheckpoint:
    """Load one v2 checkpoint and verify file, content, and CSV metadata."""

    payload, actual_sha256, resolved = _load_file_snapshot(
        path, role=f"{role} checkpoint"
    )
    expected_sha256 = _validate_sha256(expected_sha256, f"{role} checkpoint SHA256")
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"{role} checkpoint SHA256 mismatch: "
            f"expected={expected_sha256} actual={actual_sha256} path={resolved}"
        )
    if not isinstance(payload, Mapping):
        raise ValueError(f"{role} checkpoint must contain a mapping")
    validate_checkpoint_v2_payload(payload)
    config = payload["config"]
    stage = payload["stage"]
    training_state = payload["training_state"]
    epoch = int(training_state["epoch"])
    step = int(training_state["global_step"])
    stage_step = int(training_state["stage_step"])
    if role == "candidate" and (epoch <= 0 or step <= 0):
        raise ValueError("candidate checkpoints must have positive epoch and global step")
    if expected_epoch is not None and epoch != expected_epoch:
        raise ValueError(
            f"candidate {candidate_id!r} epoch mismatch: "
            f"CSV={expected_epoch} checkpoint={epoch}"
        )
    if expected_step is not None and step != expected_step:
        raise ValueError(
            f"candidate {candidate_id!r} step mismatch: "
            f"CSV={expected_step} checkpoint={step}"
        )
    if config.get("training_stage") != stage["name"]:
        raise ValueError(f"{role} checkpoint config training_stage disagrees with stage")
    if config.get("stage_index") != stage["index"]:
        raise ValueError(f"{role} checkpoint config stage_index disagrees with stage")
    _validate_run_id(config.get("run_id"), f"{role} checkpoint run_id")
    _validate_sha256(
        config.get("run_manifest_sha256"),
        f"{role} checkpoint run_manifest_sha256",
    )
    initial_global_step = config.get("stage_initial_global_step")
    if (
        isinstance(initial_global_step, bool)
        or not isinstance(initial_global_step, int)
        or initial_global_step < 0
        or step - stage_step != initial_global_step
    ):
        raise ValueError(f"{role} checkpoint has invalid stage_initial_global_step")
    cadence = config.get("checkpoint_every_steps")
    if isinstance(cadence, bool) or not isinstance(cadence, int) or cadence < 0:
        raise ValueError(f"{role} checkpoint has invalid checkpoint_every_steps")
    return LoadedCheckpoint(
        candidate_id=candidate_id,
        role=role,
        path=resolved,
        file_sha256=actual_sha256,
        payload=payload,
        config=config,
        stage_name=stage["name"],
        stage_index=int(stage["index"]),
        epoch=epoch,
        step=step,
    )


def _reject_duplicate_json_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"JSON document contains duplicate key {key!r}")
        result[key] = value
    return result


def _load_json_attestation(path: Path, *, role: str) -> Tuple[Mapping[str, Any], str]:
    candidate = Path(path).expanduser()
    if candidate.is_symlink():
        raise ValueError(f"{role} must not be a symlink: {candidate}")
    resolved = candidate.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{role} does not exist: {resolved}")
    payload_bytes = resolved.read_bytes()
    try:
        document = json.loads(
            payload_bytes.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"{role} contains non-finite value {value}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid {role}: {resolved}") from error
    if not isinstance(document, Mapping):
        raise ValueError(f"{role} root must be a JSON object")
    return document, hashlib.sha256(payload_bytes).hexdigest()


def _validate_run_manifest_artifact(
    checkpoint: LoadedCheckpoint,
) -> Tuple[Mapping[str, Any], Path, str]:
    path = checkpoint.path.parent / "run_manifest.json"
    document, file_digest = _load_json_attestation(path, role="run manifest")
    if document.get("schema_version") != 3:
        raise ValueError("unsupported run manifest schema_version")
    if document.get("run_id") != checkpoint.run_id:
        raise ValueError(f"{checkpoint.role} run manifest run_id mismatch")
    semantic_digest = canonical_json_sha256(document)
    if semantic_digest != checkpoint.run_manifest_sha256:
        raise ValueError(f"{checkpoint.role} run manifest SHA256 mismatch")
    manifest_config = document.get("config")
    if not isinstance(manifest_config, Mapping):
        raise ValueError("run manifest is missing config")
    expected_config = {
        key: value
        for key, value in checkpoint.config.items()
        if key != "run_manifest_sha256"
    }
    if canonical_json_sha256(manifest_config) != canonical_json_sha256(expected_config):
        raise ValueError(f"{checkpoint.role} run manifest config mismatch")
    if document.get("stage") != checkpoint.stage_name or document.get(
        "stage_index"
    ) != checkpoint.stage_index:
        raise ValueError(f"{checkpoint.role} run manifest stage mismatch")
    return document, path.resolve(), file_digest


def _validate_stage_completion_artifact(
    checkpoint: LoadedCheckpoint,
    *,
    require_selected_best: bool,
) -> Tuple[Mapping[str, Any], Path, str]:
    path = checkpoint.path.parent / "stage_completion.json"
    document, file_digest = _load_json_attestation(path, role="stage completion")
    if document.get("schema_version") != 2:
        raise ValueError("unsupported stage completion schema_version")
    expected = {
        "run_id": checkpoint.run_id,
        "run_manifest_sha256": checkpoint.run_manifest_sha256,
        "stage": {"name": checkpoint.stage_name, "index": checkpoint.stage_index},
        "protocol_sha256": checkpoint.payload["integrity"]["protocol_sha256"],
        "normalization_sha256": checkpoint.payload["integrity"][
            "normalization_sha256"
        ],
        "checkpoint_every_steps": checkpoint.config["checkpoint_every_steps"],
        "stage_initial_global_step": checkpoint.config["stage_initial_global_step"],
    }
    for key, expected_value in expected.items():
        if document.get(key) != expected_value:
            raise ValueError(f"stage completion {key} mismatch")
    validation_count = document.get("validation_count")
    minimum_validations = checkpoint.config.get("minimum_validations")
    if (
        isinstance(validation_count, bool)
        or not isinstance(validation_count, int)
        or isinstance(minimum_validations, bool)
        or not isinstance(minimum_validations, int)
        or validation_count < minimum_validations
        or document.get("minimum_validations") != minimum_validations
    ):
        raise ValueError("stage completion validation-count gate is invalid")
    initial_global_step = checkpoint.config["stage_initial_global_step"]
    final_stage_step = document.get("final_stage_step")
    final_global_step = document.get("final_global_step")
    if (
        isinstance(final_stage_step, bool)
        or not isinstance(final_stage_step, int)
        or final_stage_step < 0
        or final_global_step != initial_global_step + final_stage_step
    ):
        raise ValueError("stage completion final step chronology is invalid")

    final_path = checkpoint.path.parent / "checkpoint.pt"
    best_path = checkpoint.path.parent / "checkpoint_best.pt"
    for artifact_path, path_key, digest_key in (
        (final_path, "final_checkpoint", "final_checkpoint_sha256"),
        (best_path, "selected_best_checkpoint", "selected_best_checkpoint_sha256"),
    ):
        if artifact_path.is_symlink() or not artifact_path.is_file():
            raise ValueError(f"completed checkpoint artifact is missing or symlinked: {artifact_path}")
        if Path(str(document.get(path_key, ""))).expanduser().resolve() != artifact_path.resolve():
            raise ValueError(f"stage completion {path_key} mismatch")
        if document.get(digest_key) != file_sha256(artifact_path):
            raise ValueError(f"stage completion {digest_key} mismatch")
    if require_selected_best:
        if checkpoint.path != best_path.resolve() or (
            checkpoint.file_sha256 != document.get("selected_best_checkpoint_sha256")
        ):
            raise ValueError(
                "stage-1 reference must be the completed stage's selected-best alias"
            )
    return document, path.resolve(), file_digest


def validate_run_artifacts(
    reference: LoadedCheckpoint,
    candidates: Sequence[LoadedCheckpoint],
) -> Mapping[str, Any]:
    """Bind selection to completed runs and the full periodic candidate cadence."""

    _, reference_manifest_path, reference_manifest_file_sha256 = (
        _validate_run_manifest_artifact(reference)
    )
    _, reference_completion_path, reference_completion_sha256 = (
        _validate_stage_completion_artifact(reference, require_selected_best=True)
    )
    candidate_parents = {candidate.path.parent for candidate in candidates}
    if len(candidate_parents) != 1:
        raise ValueError("all candidate checkpoints must come from one output directory")
    run_ids = {candidate.run_id for candidate in candidates}
    manifest_hashes = {candidate.run_manifest_sha256 for candidate in candidates}
    if len(run_ids) != 1 or len(manifest_hashes) != 1:
        raise ValueError("candidate checkpoints must share one immutable run identity")
    for candidate in candidates:
        _validate_run_manifest_artifact(candidate)
    candidate_manifest, candidate_manifest_path, candidate_manifest_file_sha256 = (
        _validate_run_manifest_artifact(candidates[0])
    )
    completion, candidate_completion_path, candidate_completion_sha256 = (
        _validate_stage_completion_artifact(
            candidates[0], require_selected_best=False
        )
    )
    if candidate_manifest.get("run_id") != completion.get("run_id"):
        raise ValueError("candidate run manifest and completion run_id disagree")
    raw_universe = completion.get("candidate_checkpoints")
    if not isinstance(raw_universe, list) or not raw_universe:
        raise ValueError("completed candidate stage has no periodic candidate universe")
    cadence = candidates[0].config["checkpoint_every_steps"]
    if cadence <= 0:
        raise ValueError("formal checkpoint selection requires a positive checkpoint cadence")
    final_stage_step = completion.get("final_stage_step")
    if isinstance(final_stage_step, bool) or not isinstance(final_stage_step, int):
        raise ValueError("stage completion final_stage_step must be an integer")
    expected_stage_steps = list(range(cadence, final_stage_step + 1, cadence))
    observed_stage_steps = [row.get("stage_step") for row in raw_universe if isinstance(row, Mapping)]
    if len(observed_stage_steps) != len(raw_universe) or observed_stage_steps != expected_stage_steps:
        raise ValueError(
            "stage completion candidate universe does not match the full cadence"
        )
    if len(candidates) != len(raw_universe):
        raise ValueError(
            "candidate CSV must contain the complete periodic checkpoint universe"
        )
    initial_global_step = candidates[0].config["stage_initial_global_step"]
    for candidate, record, expected_stage_step in zip(
        candidates, raw_universe, expected_stage_steps
    ):
        expected_path = candidate.path.parent / (
            f"checkpoint_step_{expected_stage_step:08d}.pt"
        )
        if candidate.path != expected_path.resolve():
            raise ValueError("candidate CSV path/order differs from the completed cadence")
        expected_record = {
            "stage_step": expected_stage_step,
            "global_step": initial_global_step + expected_stage_step,
            "epoch": candidate.epoch,
            "checkpoint_path": str(expected_path.resolve()),
            "checkpoint_sha256": candidate.file_sha256,
            "model_sha256": candidate.payload["integrity"]["model_state_sha256"],
        }
        if dict(record) != expected_record:
            raise ValueError(
                f"candidate completion record mismatch at stage_step={expected_stage_step}"
            )
        if candidate.stage_step != expected_stage_step or candidate.step != (
            initial_global_step + expected_stage_step
        ):
            raise ValueError("candidate checkpoint chronology disagrees with completion")
    return {
        "reference_run_id": reference.run_id,
        "reference_run_manifest": str(reference_manifest_path),
        "reference_run_manifest_file_sha256": reference_manifest_file_sha256,
        "reference_stage_completion": str(reference_completion_path),
        "reference_stage_completion_sha256": reference_completion_sha256,
        "candidate_run_id": candidates[0].run_id,
        "candidate_run_manifest": str(candidate_manifest_path),
        "candidate_run_manifest_sha256": candidates[0].run_manifest_sha256,
        "candidate_run_manifest_file_sha256": candidate_manifest_file_sha256,
        "candidate_stage_completion": str(candidate_completion_path),
        "candidate_stage_completion_sha256": candidate_completion_sha256,
        "candidate_stage_steps": expected_stage_steps,
    }


def _normalization_semantic_sha256(stats: Mapping[str, Any]) -> str:
    recorded = stats.get("normalization_content_sha256")
    recorded = _validate_sha256(recorded, "normalization_content_sha256")
    descriptor = {
        "normalization_config_sha256": stats.get("normalization_config_sha256"),
        "population_sha256": stats.get("population_sha256"),
        "statistics": {
            key: {
                "dtype": str(stats[key].dtype),
                "shape": list(stats[key].shape),
                "values": stats[key].detach().cpu().tolist(),
            }
            for key in (
                "qpos_mean",
                "qpos_std",
                "action_mean",
                "action_std",
                "force_mean",
                "force_std",
            )
        },
    }
    actual = canonical_json_sha256(descriptor)
    if actual != recorded:
        raise ValueError(
            "normalization semantic SHA256 mismatch: "
            f"recorded={recorded} actual={actual}"
        )
    return actual


def load_normalization_strict(
    path: Path,
    *,
    expected_action_mode: str,
    expected_domain_weights: Mapping[str, float],
    strict_lengths: bool,
    expected_sha256: str,
    validation_domains: Sequence[ValidationDomain],
    expected_qpos_dim: int = 7,
    expected_action_dim: int = 7,
    expected_force_dim: int = 6,
) -> Tuple[Mapping[str, Any], str]:
    """Load hashed train-only stats and reject validation-population leakage."""

    stats, _stats_file_sha256, resolved = _load_file_snapshot(
        path, role="normalization stats"
    )
    if not isinstance(stats, Mapping):
        raise ValueError("normalization stats file must contain a mapping")
    validate_normalization_stats(stats)
    validate_normalization_provenance_hashes(stats, require_components=True)
    for label, expected_dimension in (
        ("qpos", expected_qpos_dim),
        ("action", expected_action_dim),
        ("force", expected_force_dim),
    ):
        if expected_dimension <= 0:
            raise ValueError(f"expected_{label}_dim must be positive")
        for suffix in ("mean", "std"):
            key = f"{label}_{suffix}"
            if int(stats[key].numel()) != expected_dimension:
                raise ValueError(
                    f"normalization {key} dimension mismatch: "
                    f"stats={stats[key].numel()} expected={expected_dimension}"
                )
    if "action_mode" not in stats:
        raise ValueError("normalization stats must record action_mode")
    if stats["action_mode"] != expected_action_mode:
        raise ValueError(
            "normalization action_mode mismatch: "
            f"stats={stats['action_mode']!r} expected={expected_action_mode!r}"
        )
    validate_balanced_normalization_contract(
        stats,
        expected_action_mode=expected_action_mode,
        expected_domain_weights=expected_domain_weights,
        strict_lengths=strict_lengths,
    )
    actual_semantic_hash = _normalization_semantic_sha256(stats)
    expected_sha256 = _validate_sha256(expected_sha256, "expected normalization SHA256")
    if actual_semantic_hash != expected_sha256:
        raise ValueError(
            "normalization SHA256 disagrees with checkpoint: "
            f"checkpoint={expected_sha256} stats={actual_semantic_hash}"
        )
    population_paths = stats.get("population_paths")
    if not isinstance(population_paths, (list, tuple)) or not population_paths:
        raise ValueError("normalization stats must record a non-empty population_paths")
    population = {Path(path).expanduser().resolve() for path in population_paths}
    validation = {
        path
        for domain in validation_domains
        for path in domain.episode_paths
    }
    overlap = sorted(population & validation)
    if overlap:
        raise ValueError(
            "normalization population overlaps validation episodes: "
            + ", ".join(str(path) for path in overlap[:5])
        )
    return stats, actual_semantic_hash


def validate_normalization_dataset_config(
    stats: Mapping[str, Any], protocol: ResolvedProtocol
) -> None:
    """Require stats to use the same temporal and visual data semantics."""

    dataset = protocol.dataset
    expected = {
        "chunk_len": dataset.chunk_len,
        "force_window_len": dataset.force_window_len,
        "force_window_duration": dataset.force_window_duration,
        "camera_names": tuple(dataset.camera_names),
        "image_size": tuple(dataset.image_size),
        "imagenet_normalize": dataset.imagenet_normalize,
    }
    for key, expected_value in expected.items():
        if key not in stats:
            raise ValueError(f"normalization stats is missing dataset config key: {key}")
        actual_value = stats[key]
        if key in {"camera_names", "image_size"}:
            if not isinstance(actual_value, (list, tuple)):
                raise ValueError(f"normalization stats {key} must be a list or tuple")
            actual_value = tuple(actual_value)
        if actual_value != expected_value:
            raise ValueError(
                f"normalization dataset config mismatch for {key}: "
                f"stats={actual_value!r} protocol={expected_value!r}"
            )


def _expected_config_from_protocol(protocol: ResolvedProtocol) -> Dict[str, Any]:
    dataset = protocol.dataset
    return {
        "policy_variant": protocol.model.policy_variant,
        "action_mode": dataset.action_mode,
        "chunk_len": dataset.chunk_len,
        "force_window_len": dataset.force_window_len,
        "force_window_duration": dataset.force_window_duration,
        "image_size": dataset.image_size,
        "camera_names": dataset.camera_names,
        "imagenet_normalize": dataset.imagenet_normalize,
        "image_alignment": dataset.image_alignment,
        "max_image_lag_seconds": dataset.max_image_lag_seconds,
        "model": resolved_model_config(protocol.model, dataset),
    }


def validate_checkpoint_family(
    reference: LoadedCheckpoint,
    candidates: Sequence[LoadedCheckpoint],
    protocol: ResolvedProtocol,
) -> Tuple[Any, str, float]:
    """Require one architecture/protocol/stats identity and a later candidate stage."""

    expected = _expected_config_from_protocol(protocol)
    protocol_hash = protocol.content_sha256
    all_checkpoints = (reference, *candidates)
    for checkpoint in all_checkpoints:
        validate_checkpoint_compatibility(
            checkpoint.config,
            expected,
            keys=INIT_COMPATIBILITY_KEYS,
            require_keys=True,
        )
        integrity = checkpoint.payload["integrity"]
        if checkpoint.config.get("protocol_sha256") != protocol_hash:
            raise ValueError(
                f"{checkpoint.role} checkpoint config protocol_sha256 mismatch"
            )
        if integrity.get("protocol_sha256") != protocol_hash:
            raise ValueError(
                f"{checkpoint.role} checkpoint integrity protocol_sha256 mismatch"
            )
        config_normalization = checkpoint.config.get("normalization_sha256")
        integrity_normalization = integrity.get("normalization_sha256")
        if config_normalization != integrity_normalization:
            raise ValueError(
                f"{checkpoint.role} checkpoint normalization SHA256 fields disagree"
            )
        if integrity_normalization != reference.payload["integrity"]["normalization_sha256"]:
            raise ValueError("checkpoints use different normalization SHA256 values")

    for candidate in candidates:
        if candidate.step <= reference.step:
            raise ValueError(
                "candidate global_step must be strictly greater than the supplied "
                "stage-1 reference: "
                f"candidate={candidate.candidate_id!r}:{candidate.step} "
                f"reference={reference.candidate_id!r}:{reference.step}"
            )
        parent_hash = candidate.payload["lineage"].get("parent_checkpoint_sha256")
        if parent_hash != reference.file_sha256:
            raise ValueError(
                "candidate lineage does not point to the supplied stage-1 reference: "
                f"candidate={candidate.candidate_id!r} "
                f"parent={parent_hash!r} reference={reference.file_sha256!r}"
            )

    candidate_stages = {(item.stage_name, item.stage_index) for item in candidates}
    if len(candidate_stages) != 1:
        raise ValueError("all candidates must belong to the same training stage")
    candidate_reference_config = candidates[0].config
    for candidate in candidates[1:]:
        validate_checkpoint_compatibility(
            candidate_reference_config,
            candidate.config,
            keys=(
                "training_stage",
                "stage_index",
                "optimizer_groups",
                "data_provenance",
                "validation_deployment_mode",
                "validation_aggregation",
                "run_id",
                "run_manifest_sha256",
                "stage_initial_global_step",
                "checkpoint_every_steps",
                "validation_every_steps",
                "minimum_validations",
                "freeze_vision_batch_norm",
                "training_device",
                "training_code_sha256",
                "runtime_versions",
            ),
            require_keys=True,
        )
    candidate_stage_name, candidate_stage_index = next(iter(candidate_stages))
    if candidate_reference_config.get("stage_initial_global_step") != reference.step:
        raise ValueError(
            "candidate stage_initial_global_step must equal the stage-1 reference step"
        )
    if candidate_stage_index != reference.stage_index + 1:
        raise ValueError(
            "candidate stage must immediately follow the reference stage: "
            f"reference_index={reference.stage_index} candidate_index={candidate_stage_index}"
        )
    protocol_stage_names = [stage.name for stage in protocol.stages]
    if reference.stage_name not in protocol_stage_names:
        raise ValueError("stage-1 reference stage is absent from the protocol")
    if protocol_stage_names.index(reference.stage_name) != reference.stage_index:
        raise ValueError("stage-1 reference stage index disagrees with the protocol")
    if candidate_stage_name not in protocol_stage_names:
        raise ValueError("candidate stage is absent from the protocol")
    if protocol_stage_names.index(candidate_stage_name) != candidate_stage_index:
        raise ValueError("candidate stage index disagrees with the protocol")
    stage = protocol.stage(candidate_stage_name)
    requested_mode = candidates[0].config.get("validation_deployment_mode")
    if not isinstance(requested_mode, str):
        raise ValueError("candidate config is missing validation_deployment_mode")
    resolved_mode = resolve_validation_deployment_mode(
        policy_variant=protocol.model.policy_variant,
        requested_mode=stage.objective.validation_deployment_mode,
        train_latent_mode=stage.objective.train_latent_mode,
        lambda_prior=stage.objective.lambda_prior,
    )
    if requested_mode != resolved_mode:
        raise ValueError(
            "candidate validation_deployment_mode disagrees with protocol: "
            f"checkpoint={requested_mode!r} protocol={resolved_mode!r}"
        )
    requested_aggregation = candidates[0].config.get("validation_aggregation")
    if requested_aggregation != stage.monitor.aggregation:
        raise ValueError(
            "candidate validation_aggregation disagrees with protocol: "
            f"checkpoint={requested_aggregation!r} "
            f"protocol={stage.monitor.aggregation!r}"
        )
    for candidate in candidates[1:]:
        if candidate.config.get("validation_deployment_mode") != resolved_mode:
            raise ValueError("candidate deployment modes are inconsistent")
    return stage, resolved_mode, float(stage.objective.lambda_force)


def validate_candidate_chronology(
    candidates: Sequence[LoadedCheckpoint],
    *,
    reference: Optional[LoadedCheckpoint] = None,
) -> None:
    """Require one stage offset and strictly increasing stage/global steps."""

    previous: Optional[LoadedCheckpoint] = None
    for candidate in candidates:
        if reference is not None and candidate.step - candidate.stage_step != reference.step:
            raise ValueError(
                "candidate global_step-stage_step must equal the stage-1 reference "
                f"global_step: candidate={candidate.candidate_id!r}"
            )
        if previous is not None and (
            candidate.step <= previous.step
            or candidate.stage_step <= previous.stage_step
        ):
            raise ValueError(
                "candidate checkpoints must be ordered by strictly increasing "
                "stage_step and global_step: "
                f"previous={previous.candidate_id!r}:"
                f"({previous.stage_step},{previous.step}) "
                f"current={candidate.candidate_id!r}:"
                f"({candidate.stage_step},{candidate.step})"
            )
        previous = candidate


def validate_selector_contract(args: argparse.Namespace, stage: Any) -> None:
    """Require formal shortlist gates to match the stage protocol monitor."""

    monitor = stage.monitor
    if monitor.retention_domain is None or monitor.max_retention_regression is None:
        raise ValueError(
            "candidate stage protocol monitor must define a retention domain and limit"
        )
    expected = {
        "objective_domain": monitor.primary_domain,
        "retention_domain": monitor.retention_domain,
        "metric": monitor.metric,
        "max_relative_degradation": monitor.max_retention_regression,
        "min_relative_improvement": monitor.min_delta,
    }
    for argument_name, expected_value in expected.items():
        actual_value = getattr(args, argument_name)
        if actual_value != expected_value:
            raise ValueError(
                f"--{argument_name.replace('_', '-')} disagrees with protocol monitor: "
                f"argument={actual_value!r} protocol={expected_value!r}"
            )
    if args.max_absolute_degradation != 0.0:
        raise ValueError(
            "--max-absolute-degradation must be 0 because the protocol defines only "
            "a relative retention limit"
        )


def validate_evaluation_data_contract(
    *,
    protocol: ResolvedProtocol,
    stage: Any,
    domains: Sequence[ValidationDomain],
    reference: LoadedCheckpoint,
    candidates: Sequence[LoadedCheckpoint],
) -> None:
    """Pin every evaluated checkpoint to val lists, manifest, and episode bytes."""

    expected_specs = {item.name: item for item in stage.validation_domains}
    supplied = {domain.name: domain for domain in domains}
    if set(supplied) != set(expected_specs):
        raise ValueError(
            "--val-domain names must exactly match the candidate stage protocol: "
            f"missing={sorted(set(expected_specs) - set(supplied))} "
            f"unexpected={sorted(set(supplied) - set(expected_specs))}"
        )

    expected_paths: Dict[str, Tuple[Path, ...]] = {}
    for name, spec in expected_specs.items():
        domain = supplied[name]
        if domain.episode_list != spec.episode_list.resolve():
            raise ValueError(
                f"validation domain {name!r} must use the protocol episode list: "
                f"supplied={domain.episode_list} protocol={spec.episode_list}"
            )
        paths = tuple(
            Path(path).resolve()
            for path in resolve_episode_paths(
                [], spec.episode_list, project_root=REPO_ROOT, deduplicate=False
            )
        )
        if len(set(paths)) != len(paths):
            raise ValueError(f"protocol validation list contains duplicates: {spec.episode_list}")
        if domain.episode_paths != paths:
            raise ValueError(
                f"validation domain {name!r} resolved episodes differ from protocol"
            )
        if len(paths) != spec.expected_episode_count:
            raise ValueError(
                f"validation domain {name!r} episode count mismatch: "
                f"protocol={spec.expected_episode_count} resolved={len(paths)}"
            )
        expected_paths[name] = paths

    validation_paths_hash = canonical_json_sha256(
        {
            name: [str(path) for path in expected_paths[name]]
            for name in expected_specs
        }
    )
    checkpoints = (reference, *candidates)
    for checkpoint in checkpoints:
        provenance = checkpoint.config.get("data_provenance")
        if not isinstance(provenance, Mapping):
            raise ValueError(
                f"{checkpoint.role} {checkpoint.candidate_id!r} is missing "
                "data_provenance"
            )
        if provenance.get("validation_paths_sha256") != validation_paths_hash:
            raise ValueError(
                f"{checkpoint.role} {checkpoint.candidate_id!r} validation path "
                "provenance mismatch"
            )
        list_hashes = provenance.get("episode_list_sha256")
        if not isinstance(list_hashes, Mapping):
            raise ValueError(
                f"{checkpoint.role} {checkpoint.candidate_id!r} is missing "
                "episode-list hashes"
            )
        for name, spec in expected_specs.items():
            actual_list_hash = supplied[name].episode_list_sha256
            if list_hashes.get(str(spec.episode_list.resolve())) != actual_list_hash:
                raise ValueError(
                    f"{checkpoint.role} {checkpoint.candidate_id!r} "
                    "episode-list hash mismatch "
                    f"for validation domain {name!r}"
                )
        episode_counts = provenance.get("episode_counts")
        validation_counts = (
            episode_counts.get("validation")
            if isinstance(episode_counts, Mapping)
            else None
        )
        if not isinstance(validation_counts, Mapping) or dict(
            validation_counts
        ) != {name: len(paths) for name, paths in expected_paths.items()}:
            raise ValueError(
                f"{checkpoint.role} {checkpoint.candidate_id!r} validation episode "
                "counts mismatch"
            )

    manifest_spec = protocol.dataset_manifest
    if manifest_spec is None:
        raise ValueError("formal checkpoint evaluation requires protocol.dataset_manifest")
    manifest = DatasetManifest.load(manifest_spec.path, verify_files=False)
    validate_episode_uuid_provenance(manifest, allow_derived=False)
    manifest_hash = manifest.content_sha256
    if (
        manifest_spec.expected_sha256 is not None
        and manifest_spec.expected_sha256 != manifest_hash
    ):
        raise ValueError("protocol dataset manifest SHA256 mismatch")
    for checkpoint in checkpoints:
        provenance = checkpoint.config["data_provenance"]
        if provenance.get("dataset_manifest_sha256") != manifest_hash:
            raise ValueError(
                f"{checkpoint.role} {checkpoint.candidate_id!r} dataset manifest "
                "mismatch"
            )

    entries = {entry.identity.path: entry for entry in manifest.episodes}
    for name, spec in expected_specs.items():
        for path in expected_paths[name]:
            entry = entries.get(path)
            if entry is None:
                raise ValueError(
                    f"validation episode is absent from dataset manifest: {path}"
                )
            if entry.split != "val" or entry.domain != spec.domain:
                raise ValueError(
                    f"validation manifest assignment mismatch for {path}: "
                    f"split={entry.split!r} domain={entry.domain!r} "
                    f"expected_domain={spec.domain!r}"
                )
            actual_file_hash = file_sha256(path)
            if actual_file_hash != entry.identity.file_sha256:
                raise ValueError(
                    f"validation episode file SHA256 mismatch: {path}"
                )


def build_validation_loaders(
    protocol: ResolvedProtocol,
    domains: Sequence[ValidationDomain],
    *,
    batch_size: int,
    num_workers: int,
) -> Mapping[str, DataLoader]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if num_workers < 0:
        raise ValueError("num_workers must be non-negative")
    spec = protocol.dataset
    loaders = {}
    for domain in domains:
        dataset = ContactForceHDF5Dataset(
            domain.episode_paths,
            camera_names=spec.camera_names,
            action_mode=spec.action_mode,
            chunk_len=spec.chunk_len,
            force_window_len=spec.force_window_len,
            force_window_duration=spec.force_window_duration,
            image_size=spec.image_size,
            imagenet_normalize=spec.imagenet_normalize,
            image_alignment=spec.image_alignment,
            max_image_lag_seconds=spec.max_image_lag_seconds,
            tolerate_length_mismatch=not spec.strict_lengths,
            max_length_mismatch=0 if spec.strict_lengths else 1,
        )
        if len(dataset) == 0:
            raise ValueError(f"validation dataset is empty: {domain.name}")
        indexed_paths = {
            Path(index.episode_path).resolve() for index in dataset.indices
        }
        missing = sorted(set(domain.episode_paths) - indexed_paths)
        if missing:
            raise ValueError(
                f"validation domain {domain.name!r} contains episodes with zero "
                "usable samples: "
                + ", ".join(str(path) for path in missing[:5])
            )
        loaders[domain.name] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
    return loaders


def _build_model_from_config(config: Mapping[str, Any]) -> torch.nn.Module:
    variant = config["policy_variant"]
    model_config = config.get("model")
    if not isinstance(model_config, Mapping):
        raise ValueError("checkpoint config model must be a mapping")
    kwargs = dict(model_config)
    # All parameters are loaded immediately; downloading initialization weights
    # would be both unnecessary and a source of external-state dependence.
    kwargs["pretrained_resnet18"] = False
    if variant == "force_aware_motion_cvae":
        return ForceAwareACTMotionCVAEPolicy(**kwargs)
    if variant == "force_aware_contact_cvae":
        return ForceAwareACTContactCVAEPolicy(**kwargs)
    if variant == "force_aware_act":
        return ForceAwareACTPolicy(**kwargs)
    raise ValueError(f"unsupported policy_variant={variant!r}")


def evaluate_checkpoint(
    checkpoint: LoadedCheckpoint,
    *,
    dataloaders: Mapping[str, DataLoader],
    device: torch.device,
    deployment_mode: str,
    normalization_stats: Mapping[str, Any],
    lambda_force: float,
    aggregation: str,
    expected_episode_counts: Mapping[str, int],
) -> Mapping[str, Mapping[str, float]]:
    model = _build_model_from_config(checkpoint.config).to(device)
    model.load_state_dict(checkpoint.payload["model_state_dict"], strict=True)
    model.eval()
    metrics = evaluate_named_deployment_metrics(
        model=model,
        dataloaders=dataloaders,
        device=device,
        policy_variant=str(checkpoint.config["policy_variant"]),
        deployment_mode=deployment_mode,
        normalization_stats=normalization_stats,
        lambda_force=lambda_force,
        aggregation=aggregation,
    )
    for domain, domain_metrics in metrics.items():
        for metric, value in domain_metrics.items():
            if not math.isfinite(float(value)):
                raise ValueError(
                    f"non-finite metric for {checkpoint.candidate_id}/{domain}/{metric}"
                )
    if set(metrics) != set(dataloaders):
        raise ValueError(
            f"metric domains differ from requested domains for {checkpoint.candidate_id}: "
            f"metrics={sorted(metrics)} requested={sorted(dataloaders)}"
        )
    if set(expected_episode_counts) != set(metrics):
        raise ValueError("expected validation episode-count domains are inconsistent")
    for domain, expected_count in expected_episode_counts.items():
        observed = metrics[domain].get("num_episodes")
        if float(observed) != float(expected_count):
            raise ValueError(
                f"validation domain {domain!r} observed episode count mismatch: "
                f"expected={expected_count} observed={observed!r}"
            )
    return metrics


def _report_paths(output_dir: Path) -> Dict[str, Path]:
    return {name: output_dir / filename for name, filename in REPORT_FILENAMES.items()}


def validate_new_outputs(output_dir: Path) -> Dict[str, Path]:
    candidate = Path(output_dir).expanduser()
    if candidate.is_symlink():
        raise ValueError(f"output directory must not be a symlink: {candidate}")
    resolved = candidate.resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    try:
        resolved.mkdir(exist_ok=False)
    except FileExistsError as error:
        raise FileExistsError(
            f"checkpoint evaluation requires a new output directory: {resolved}"
        ) from error
    return _report_paths(resolved)


def _csv_text(fields: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, extrasaction="raise")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def _write_new_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())


def _metric_rows(
    evaluations: Sequence[EvaluatedCandidate],
    domains: Sequence[ValidationDomain],
    *,
    deployment_mode: str,
    protocol_sha256: str,
    normalization_sha256: str,
) -> list[Dict[str, Any]]:
    domain_lookup = {domain.name: domain for domain in domains}
    rows = []
    for evaluation in evaluations:
        checkpoint = evaluation.checkpoint
        for domain_name, metrics in evaluation.metrics.items():
            domain = domain_lookup[domain_name]
            for metric_name in sorted(metrics):
                rows.append(
                    {
                        "candidate_id": checkpoint.candidate_id,
                        "role": checkpoint.role,
                        "evaluation_order": evaluation.evaluation_order,
                        "checkpoint_path": str(checkpoint.path),
                        "checkpoint_sha256": checkpoint.file_sha256,
                        "checkpoint_stage": checkpoint.stage_name,
                        "checkpoint_stage_index": checkpoint.stage_index,
                        "checkpoint_epoch": checkpoint.epoch,
                        "checkpoint_step": checkpoint.step,
                        "domain": domain_name,
                        "metric": metric_name,
                        "value": float(metrics[metric_name]),
                        "deployment_mode": deployment_mode,
                        "protocol_sha256": protocol_sha256,
                        "normalization_sha256": normalization_sha256,
                        "episode_list": str(domain.episode_list),
                        "episode_list_sha256": domain.episode_list_sha256,
                        "num_episodes": len(domain.episode_paths),
                    }
                )
    return rows


def _decision_rows(
    candidates: Sequence[EvaluatedCandidate], final_best_id: Optional[str]
) -> list[Dict[str, Any]]:
    rows = []
    for evaluation in candidates:
        checkpoint = evaluation.checkpoint
        decision = evaluation.decision
        rows.append(
            {
                "candidate_id": checkpoint.candidate_id,
                "evaluation_order": evaluation.evaluation_order,
                "checkpoint_path": str(checkpoint.path),
                "checkpoint_sha256": checkpoint.file_sha256,
                "checkpoint_epoch": checkpoint.epoch,
                "checkpoint_step": checkpoint.step,
                "metric": "",  # populated by caller from selector state below
                "objective_domain": "",
                "objective_value": decision.objective_value,
                "retention_domain": "",
                "retention_value": decision.retention_value,
                "retention_baseline": "",
                "retention_limit": decision.retention_limit,
                "retention_passed": decision.retention_passed,
                "objective_improved": decision.objective_improved,
                "selected": decision.selected,
                "final_best": checkpoint.candidate_id == final_best_id,
                "reason": decision.reason,
            }
        )
    return rows


def write_reports(
    *,
    output_paths: Mapping[str, Path],
    reference: EvaluatedCandidate,
    candidates: Sequence[EvaluatedCandidate],
    domains: Sequence[ValidationDomain],
    selector: RetentionGatedCheckpointSelector,
    shortlist_size: int,
    deployment_mode: str,
    protocol_sha256: str,
    normalization_sha256: str,
    candidates_csv: Path,
    protocol_path: Path,
    normalization_stats_path: Path,
    run_evidence: Optional[Mapping[str, Any]] = None,
    evaluation_contract: Optional[Mapping[str, Any]] = None,
) -> Mapping[str, Any]:
    """Render all artifacts in memory, then create each destination exclusively."""

    if shortlist_size <= 0:
        raise ValueError("shortlist_size must be positive")
    final_best_id = None
    for evaluation in candidates:
        if evaluation.decision.selected:
            final_best_id = evaluation.checkpoint.candidate_id

    reference_objective = float(
        reference.metrics[selector.objective_domain][selector.metric]
    )
    required_reference_improvement = (
        abs(reference_objective) * selector.min_relative_improvement
    )
    eligible = [
        item
        for item in candidates
        if item.decision.retention_passed
        and item.decision.objective_value
        < reference_objective - required_reference_improvement
    ]
    eligible.sort(
        key=lambda item: (
            item.decision.objective_value,
            item.decision.retention_value,
            item.checkpoint.candidate_id,
        )
    )
    if final_best_id is None:
        shortlisted = []
    else:
        final_best = next(
            item
            for item in eligible
            if item.checkpoint.candidate_id == final_best_id
        )
        shortlisted = [final_best]
        shortlisted.extend(
            item
            for item in eligible
            if item.checkpoint.candidate_id != final_best_id
        )
        shortlisted = shortlisted[:shortlist_size]
    fallback = not shortlisted
    shortlist_rows = []
    if fallback:
        objective_value = float(
            reference.metrics[selector.objective_domain][selector.metric]
        )
        retention_value = float(
            reference.metrics[selector.retention_domain][selector.metric]
        )
        shortlist_rows.append(
            {
                "rank": 1,
                "candidate_id": reference.checkpoint.candidate_id,
                "checkpoint_path": str(reference.checkpoint.path),
                "checkpoint_sha256": reference.checkpoint.file_sha256,
                "checkpoint_epoch": reference.checkpoint.epoch,
                "checkpoint_step": reference.checkpoint.step,
                "metric": selector.metric,
                "objective_value": objective_value,
                "retention_value": retention_value,
                "retention_limit": selector.retention_limit,
                "final_selected": True,
                "fallback_reference": True,
            }
        )
    else:
        for rank, evaluation in enumerate(shortlisted, start=1):
            checkpoint = evaluation.checkpoint
            decision = evaluation.decision
            shortlist_rows.append(
                {
                    "rank": rank,
                    "candidate_id": checkpoint.candidate_id,
                    "checkpoint_path": str(checkpoint.path),
                    "checkpoint_sha256": checkpoint.file_sha256,
                    "checkpoint_epoch": checkpoint.epoch,
                    "checkpoint_step": checkpoint.step,
                    "metric": selector.metric,
                    "objective_value": decision.objective_value,
                    "retention_value": decision.retention_value,
                    "retention_limit": decision.retention_limit,
                    "final_selected": checkpoint.candidate_id == final_best_id,
                    "fallback_reference": False,
                }
            )
    shortlist_json = [dict(row) for row in shortlist_rows]

    decisions = _decision_rows(candidates, final_best_id)
    for row in decisions:
        row["metric"] = selector.metric
        row["objective_domain"] = selector.objective_domain
        row["retention_domain"] = selector.retention_domain
        row["retention_baseline"] = selector.retention_baseline

    metrics = _metric_rows(
        (reference, *candidates),
        domains,
        deployment_mode=deployment_mode,
        protocol_sha256=protocol_sha256,
        normalization_sha256=normalization_sha256,
    )
    selected = shortlist_rows[0] if fallback else next(
        (
            row
            for row in shortlist_rows
            if row["candidate_id"] == final_best_id
        ),
        shortlist_rows[0],
    )
    document = {
        "schema_version": 2,
        "candidates_csv": str(Path(candidates_csv).resolve()),
        "candidates_csv_sha256": file_sha256(candidates_csv),
        "protocol_path": str(Path(protocol_path).resolve()),
        "protocol_file_sha256": file_sha256(protocol_path),
        "protocol_sha256": protocol_sha256,
        "normalization_stats_path": str(Path(normalization_stats_path).resolve()),
        "normalization_stats_file_sha256": file_sha256(normalization_stats_path),
        "normalization_sha256": normalization_sha256,
        "deployment_mode": deployment_mode,
        "validation_domains": [
            {
                "name": domain.name,
                "episode_list": str(domain.episode_list),
                "episode_list_sha256": domain.episode_list_sha256,
                "num_episodes": len(domain.episode_paths),
            }
            for domain in domains
        ],
        "selector": selector.checkpoint_metadata(),
        "reference": {
            "candidate_id": reference.checkpoint.candidate_id,
            "checkpoint_path": str(reference.checkpoint.path),
            "checkpoint_sha256": reference.checkpoint.file_sha256,
            "metrics": reference.metrics,
        },
        "selected": selected,
        "fallback_to_stage1_reference": fallback,
        "run_evidence": dict(run_evidence or {}),
        "evaluation_contract": dict(evaluation_contract or {}),
        "decisions": decisions,
        "shortlist": shortlist_json,
        "report_files": {name: str(path) for name, path in output_paths.items()},
    }
    rendered = {
        "metrics": _csv_text(METRICS_FIELDS, metrics),
        "decisions": _csv_text(DECISION_FIELDS, decisions),
        "shortlist_csv": _csv_text(SHORTLIST_FIELDS, shortlist_rows),
        "shortlist_json": json.dumps(
            document, indent=2, sort_keys=True, allow_nan=False
        )
        + "\n",
    }
    for name in ("metrics", "decisions", "shortlist_csv", "shortlist_json"):
        _write_new_text(output_paths[name], rendered[name])
    report_hashes = {
        name: {
            "path": str(output_paths[name].resolve()),
            "file_sha256": file_sha256(output_paths[name]),
        }
        for name in ("metrics", "decisions", "shortlist_csv", "shortlist_json")
    }
    completion = {
        "schema_version": 1,
        "status": "complete",
        "selection_report": str(output_paths["shortlist_json"].resolve()),
        "selection_report_sha256": report_hashes["shortlist_json"]["file_sha256"],
        "protocol_sha256": protocol_sha256,
        "normalization_sha256": normalization_sha256,
        "selected_checkpoint_path": selected["checkpoint_path"],
        "selected_checkpoint_sha256": selected["checkpoint_sha256"],
        "fallback_to_stage1_reference": fallback,
        "run_evidence": dict(run_evidence or {}),
        "evaluation_contract": dict(evaluation_contract or {}),
        "report_files": report_hashes,
    }
    _write_new_text(
        output_paths["completion"],
        json.dumps(completion, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )
    return document


def run(args: argparse.Namespace) -> Mapping[str, Any]:
    output_paths = validate_new_outputs(args.output_dir)
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative")
    if args.shortlist_size <= 0:
        raise ValueError("--shortlist-size must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but torch.cuda.is_available() is false")
    if device.type == "mps":
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is None or not mps_backend.is_available():
            raise RuntimeError("MPS device requested but the MPS backend is unavailable")
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())

    domains = resolve_validation_domains(args.val_domain)
    domain_names = {domain.name for domain in domains}
    if args.objective_domain not in domain_names:
        raise ValueError(f"objective domain is not defined: {args.objective_domain!r}")
    if args.retention_domain not in domain_names:
        raise ValueError(f"retention domain is not defined: {args.retention_domain!r}")
    if args.objective_domain == args.retention_domain:
        raise ValueError("objective and retention domains must differ")

    candidate_specs = load_candidate_specs(args.candidates_csv)
    reference = load_checkpoint_strict(
        candidate_id=REFERENCE_CANDIDATE_ID,
        role="stage1_reference",
        path=args.stage1_reference,
        expected_sha256=args.stage1_reference_sha256,
    )
    if any(spec.checkpoint_path == reference.path for spec in candidate_specs):
        raise ValueError("stage-1 reference must not also appear as a candidate")
    candidates = tuple(
        load_checkpoint_strict(
            candidate_id=spec.candidate_id,
            role="candidate",
            path=spec.checkpoint_path,
            expected_sha256=spec.expected_sha256,
            expected_epoch=spec.expected_epoch,
            expected_step=spec.expected_step,
        )
        for spec in candidate_specs
    )
    validate_candidate_chronology(candidates, reference=reference)
    run_evidence = validate_run_artifacts(reference, candidates)
    protocol_path = args.protocol
    if protocol_path is None:
        recorded_protocol = reference.config.get("protocol_path")
        if not isinstance(recorded_protocol, str) or not recorded_protocol:
            raise ValueError("--protocol is required when reference config has no protocol_path")
        protocol_path = Path(recorded_protocol)
    protocol = load_protocol(Path(protocol_path))
    if not protocol.deterministic:
        raise ValueError("formal checkpoint selection requires protocol.deterministic=true")
    _enable_determinism(protocol.seed)
    stage, deployment_mode, lambda_force = validate_checkpoint_family(
        reference, candidates, protocol
    )
    validate_selector_contract(args, stage)
    validate_evaluation_data_contract(
        protocol=protocol,
        stage=stage,
        domains=domains,
        reference=reference,
        candidates=candidates,
    )
    normalization_path = args.normalization_stats
    if normalization_path is None:
        normalization_path = protocol.normalization.stats_path
    expected_normalization_hash = reference.payload["integrity"]["normalization_sha256"]
    stats, normalization_hash = load_normalization_strict(
        Path(normalization_path),
        expected_action_mode=protocol.dataset.action_mode,
        expected_domain_weights=protocol.normalization.domain_weights,
        strict_lengths=protocol.dataset.strict_lengths,
        expected_sha256=expected_normalization_hash,
        validation_domains=domains,
        expected_qpos_dim=7,
        expected_action_dim=protocol.model.action_dim,
        expected_force_dim=protocol.model.force_dim,
    )
    validate_normalization_dataset_config(stats, protocol)
    if (
        protocol.normalization.expected_sha256 is not None
        and protocol.normalization.expected_sha256 != normalization_hash
    ):
        raise ValueError("protocol normalization SHA256 differs from loaded statistics")

    immutable_input_paths = [
        Path(args.candidates_csv),
        reference.path,
        *(candidate.path for candidate in candidates),
        Path(protocol_path),
        Path(normalization_path),
        *(domain.episode_list for domain in domains),
    ]
    for key in (
        "reference_run_manifest",
        "reference_stage_completion",
        "candidate_run_manifest",
        "candidate_stage_completion",
    ):
        value = run_evidence.get(key)
        if isinstance(value, str) and value:
            immutable_input_paths.append(Path(value))
    immutable_input_hashes = _capture_input_file_hashes(immutable_input_paths)

    dataloaders = build_validation_loaders(
        protocol,
        domains,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    expected_episode_counts = {
        domain.name: len(domain.episode_paths) for domain in domains
    }
    reference_metrics = evaluate_checkpoint(
        reference,
        dataloaders=dataloaders,
        device=device,
        deployment_mode=deployment_mode,
        normalization_stats=stats,
        lambda_force=lambda_force,
        aggregation=stage.monitor.aggregation,
        expected_episode_counts=expected_episode_counts,
    )
    reference_evaluation = EvaluatedCandidate(
        checkpoint=reference,
        evaluation_order=0,
        metrics=reference_metrics,
    )
    retention_baseline = float(
        reference_metrics[args.retention_domain][args.metric]
    )
    objective_baseline = float(
        reference_metrics[args.objective_domain][args.metric]
    )
    selector = RetentionGatedCheckpointSelector(
        objective_domain=args.objective_domain,
        retention_domain=args.retention_domain,
        retention_baseline=retention_baseline,
        metric=args.metric,
        max_relative_degradation=args.max_relative_degradation,
        max_absolute_degradation=args.max_absolute_degradation,
        min_relative_improvement=args.min_relative_improvement,
        best_objective_value=objective_baseline,
        best_retention_value=retention_baseline,
    )
    candidate_evaluations = []
    for evaluation_order, checkpoint in enumerate(candidates, start=1):
        metrics = evaluate_checkpoint(
            checkpoint,
            dataloaders=dataloaders,
            device=device,
            deployment_mode=deployment_mode,
            normalization_stats=stats,
            lambda_force=lambda_force,
            aggregation=stage.monitor.aggregation,
            expected_episode_counts=expected_episode_counts,
        )
        evaluation = EvaluatedCandidate(
            checkpoint=checkpoint,
            evaluation_order=evaluation_order,
            metrics=metrics,
        )
        evaluation.decision = selector.update(
            metrics,
            epoch=checkpoint.epoch,
            step=checkpoint.step,
        )
        candidate_evaluations.append(evaluation)

    # Re-resolve lists and re-hash every HDF5 through the manifest after the
    # potentially long candidate loop.  Reports are not written if any input
    # changed while evaluation was in progress.
    refreshed_domains = resolve_validation_domains(args.val_domain)
    if refreshed_domains != domains:
        raise ValueError("validation episode lists changed during checkpoint evaluation")
    validate_evaluation_data_contract(
        protocol=protocol,
        stage=stage,
        domains=refreshed_domains,
        reference=reference,
        candidates=candidates,
    )
    refreshed_input_hashes = _capture_input_file_hashes(immutable_input_paths)
    if refreshed_input_hashes != immutable_input_hashes:
        changed = sorted(
            path
            for path in set(immutable_input_hashes) | set(refreshed_input_hashes)
            if immutable_input_hashes.get(path) != refreshed_input_hashes.get(path)
        )
        raise ValueError(
            "formal evaluation inputs changed during checkpoint evaluation: "
            + ", ".join(changed[:5])
        )

    evaluation_contract = {
        "deployment_mode": deployment_mode,
        "aggregation": stage.monitor.aggregation,
        "seed": protocol.seed,
        "protocol_deterministic": True,
        "deterministic_algorithms": True,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "device": str(device),
        "runtime_versions": dict(_runtime_versions()),
    }

    document = write_reports(
        output_paths=output_paths,
        reference=reference_evaluation,
        candidates=candidate_evaluations,
        domains=domains,
        selector=selector,
        shortlist_size=args.shortlist_size,
        deployment_mode=deployment_mode,
        protocol_sha256=protocol.content_sha256,
        normalization_sha256=normalization_hash,
        candidates_csv=args.candidates_csv,
        protocol_path=Path(protocol_path),
        normalization_stats_path=Path(normalization_path),
        run_evidence=run_evidence,
        evaluation_contract=evaluation_contract,
    )
    print(
        json.dumps(
            {
                "candidate_stage": stage.name,
                "selected": document["selected"]["candidate_id"],
                "fallback_to_stage1_reference": document[
                    "fallback_to_stage1_reference"
                ],
                "output_dir": str(Path(args.output_dir).resolve()),
            },
            sort_keys=True,
        )
    )
    return document


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates-csv", type=Path, required=True)
    parser.add_argument("--stage1-reference", type=Path, required=True)
    parser.add_argument("--stage1-reference-sha256", required=True)
    parser.add_argument("--protocol", type=Path, default=None)
    parser.add_argument("--normalization-stats", type=Path, default=None)
    parser.add_argument(
        "--val-domain",
        action="append",
        default=[],
        metavar="NAME=EPISODE_LIST",
        help="Repeat for each independent validation domain.",
    )
    parser.add_argument("--objective-domain", required=True)
    parser.add_argument("--retention-domain", required=True)
    parser.add_argument(
        "--metric",
        choices=("deploy_loss", "action_l1", "force_l1"),
        default="deploy_loss",
    )
    parser.add_argument("--max-relative-degradation", type=float, default=0.05)
    parser.add_argument("--max-absolute-degradation", type=float, default=0.0)
    parser.add_argument("--min-relative-improvement", type=float, default=0.0)
    parser.add_argument("--shortlist-size", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cuda")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        run(args)
        return 0
    except (
        FileNotFoundError,
        FileExistsError,
        KeyError,
        TypeError,
        ValueError,
        RuntimeError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
