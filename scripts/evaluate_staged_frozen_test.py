#!/usr/bin/env python3
"""Run the one-shot, frozen test evaluation for a staged-training selection.

The selected checkpoint is read exclusively from a SHA-256-pinned shortlist
report produced by ``evaluate_staged_checkpoints.py``.  Test populations are
read exclusively from the pinned protocol.  The command intentionally has no
checkpoint or episode-list override flags: changing either after validation
selection would invalidate the frozen-test contract.
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
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import scripts.evaluate_staged_checkpoints as selection_evaluator  # noqa: E402
from force_aware_act.data import (  # noqa: E402
    ContactForceHDF5Dataset,
    DatasetManifest,
    normalize_tensor,
    validate_episode_uuid_provenance,
)
from force_aware_act.training.checkpointing import (  # noqa: E402
    INIT_COMPATIBILITY_KEYS,
    file_sha256,
    validate_checkpoint_compatibility,
)
from force_aware_act.training.control import (  # noqa: E402
    resolve_validation_deployment_mode,
)
from force_aware_act.training.protocol import ResolvedProtocol, load_protocol  # noqa: E402
from force_aware_act.utils import resolve_episode_paths  # noqa: E402


FROZEN_TEST_SCHEMA_VERSION = 1
SELECTION_REPORT_SCHEMA_VERSIONS = (2,)
FORMAL_TEST_DOMAIN_COUNT = 2
FORMAL_EPISODES_PER_DOMAIN = 5
DEPLOYMENT_MODE = "prior"
AGGREGATION = "episode_uniform"
MIN_BOOTSTRAP_REPLICATES = 1000
DEFAULT_BOOTSTRAP_SEED = 20260722
DEFAULT_BOOTSTRAP_REPLICATES = 10000
DEFAULT_BATCH_SIZE = 16
EPISODE_METRIC_FIELDS = (
    "test_name",
    "manifest_domain",
    "episode_order",
    "episode_uuid",
    "episode_path",
    "episode_file_sha256",
    "num_samples",
    "action_l1",
    "force_l1",
    "deploy_loss",
)
OUTPUT_FILENAMES = {
    "episode_metrics": "episode_metrics.csv",
    "domain_metrics": "domain_metrics.json",
    "report": "frozen_test_report.json",
    "completion": "completion.json",
}


@dataclass(frozen=True)
class FrozenTestDomain:
    name: str
    manifest_domain: str
    episode_list: Path
    episode_list_sha256: str
    episode_paths: Tuple[Path, ...]


@dataclass(frozen=True)
class SelectionArtifact:
    report_path: Path
    report_sha256: str
    document: Mapping[str, Any]
    selected_id: str
    checkpoint_path: Path
    checkpoint_sha256: str
    checkpoint_epoch: int
    checkpoint_step: int
    companion_sha256: Mapping[str, str]


def _validate_sha256(value: Any, context: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{context} must be a SHA256 hex digest")
    digest = value.strip().lower()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"{context} must be a 64-character SHA256 hex digest")
    return digest


def _positive_int(value: Any, context: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{context} must be a positive integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{context} must be a positive integer") from error
    if result <= 0 or str(result) != str(value).strip():
        raise ValueError(f"{context} must be a canonical positive integer")
    return result


def _strict_json_load(path: Path, *, context: str) -> Mapping[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite_constant,
        )
    except UnicodeDecodeError as error:
        raise ValueError(f"{context} is not valid UTF-8: {path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"{context} is not valid JSON: {path}: {error}") from error
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must contain a JSON object")
    return value


def _reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"JSON contains duplicate key {key!r}")
        result[key] = value
    return result


def _reject_nonfinite_constant(value: str):
    raise ValueError(f"JSON contains non-finite value {value}")


def _absolute_without_resolving(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(Path(path).expanduser())))


def _reject_symlink_components(path: Path, *, context: str) -> None:
    absolute = _absolute_without_resolving(path)
    current = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        current = current / component
        if current.is_symlink():
            raise ValueError(f"{context} must not traverse a symlink: {current}")


def _strict_regular_file(path: Path, *, context: str) -> Path:
    candidate = _absolute_without_resolving(path)
    _reject_symlink_components(candidate, context=context)
    if not candidate.is_file():
        raise FileNotFoundError(f"{context} does not exist: {candidate}")
    return candidate.resolve()


def _resolve_recorded_path(value: Any, *, base_dir: Path, context: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context} must be a non-empty path string")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return _strict_regular_file(path, context=context)


def _load_shortlist_csv(path: Path) -> Tuple[Mapping[str, str], ...]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("selection shortlist CSV has no header")
        if tuple(reader.fieldnames) != selection_evaluator.SHORTLIST_FIELDS:
            raise ValueError(
                "selection shortlist CSV header disagrees with the staged evaluator"
            )
        rows = tuple(dict(row) for row in reader)
    if not rows:
        raise ValueError("selection shortlist CSV contains no selected candidates")
    return rows


def load_selection_artifact(
    report_path: Path,
    *,
    expected_sha256: str,
    protocol: ResolvedProtocol,
    normalization_path: Path,
) -> SelectionArtifact:
    """Verify a staged-evaluator shortlist and return its sole selected model."""

    report_path = _strict_regular_file(report_path, context="selection report")
    expected_sha256 = _validate_sha256(expected_sha256, "selection report SHA256")
    actual_sha256 = file_sha256(report_path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            "selection report SHA256 mismatch: "
            f"expected={expected_sha256} actual={actual_sha256}"
        )
    document = _strict_json_load(report_path, context="selection report")
    if document.get("schema_version") not in SELECTION_REPORT_SCHEMA_VERSIONS:
        raise ValueError(
            "unsupported selection report schema_version: "
            f"{document.get('schema_version')!r}"
        )
    if document.get("protocol_sha256") != protocol.content_sha256:
        raise ValueError("selection report protocol SHA256 mismatch")
    if document.get("deployment_mode") != DEPLOYMENT_MODE:
        raise ValueError("formal frozen test requires a prior-mode selection report")
    evaluation_contract = document.get("evaluation_contract")
    required_contract_keys = {
        "deployment_mode",
        "aggregation",
        "seed",
        "protocol_deterministic",
        "deterministic_algorithms",
        "batch_size",
        "num_workers",
        "device",
        "runtime_versions",
    }
    if not isinstance(evaluation_contract, Mapping) or set(
        evaluation_contract
    ) != required_contract_keys:
        raise ValueError("selection report evaluation_contract is incomplete")
    if (
        evaluation_contract["deployment_mode"] != DEPLOYMENT_MODE
        or evaluation_contract["aggregation"] != AGGREGATION
        or evaluation_contract["seed"] != protocol.seed
        or evaluation_contract["protocol_deterministic"] is not True
        or evaluation_contract["deterministic_algorithms"] is not True
    ):
        raise ValueError("selection report evaluation_contract is not formal/deterministic")
    if (
        not isinstance(evaluation_contract["batch_size"], int)
        or isinstance(evaluation_contract["batch_size"], bool)
        or evaluation_contract["batch_size"] <= 0
        or not isinstance(evaluation_contract["num_workers"], int)
        or isinstance(evaluation_contract["num_workers"], bool)
        or evaluation_contract["num_workers"] < 0
        or not isinstance(evaluation_contract["device"], str)
        or not evaluation_contract["device"]
        or not isinstance(evaluation_contract["runtime_versions"], Mapping)
        or not evaluation_contract["runtime_versions"]
    ):
        raise ValueError("selection report evaluation_contract has invalid runtime fields")

    protocol_path = _strict_regular_file(protocol.source_path, context="protocol")
    recorded_protocol_path = _resolve_recorded_path(
        document.get("protocol_path"),
        base_dir=report_path.parent,
        context="selection report protocol",
    )
    if recorded_protocol_path != protocol_path:
        raise ValueError("selection report points to a different protocol file")
    if document.get("protocol_file_sha256") != file_sha256(protocol_path):
        raise ValueError("selection report protocol file SHA256 mismatch")

    normalization_path = _strict_regular_file(
        normalization_path, context="normalization statistics"
    )
    recorded_normalization_path = _resolve_recorded_path(
        document.get("normalization_stats_path"),
        base_dir=report_path.parent,
        context="selection report normalization statistics",
    )
    if recorded_normalization_path != normalization_path:
        raise ValueError("selection report points to different normalization statistics")
    if document.get("normalization_stats_file_sha256") != file_sha256(normalization_path):
        raise ValueError("selection report normalization file SHA256 mismatch")

    candidates_path = _resolve_recorded_path(
        document.get("candidates_csv"),
        base_dir=report_path.parent,
        context="selection candidate CSV",
    )
    if document.get("candidates_csv_sha256") != file_sha256(candidates_path):
        raise ValueError("selection candidate CSV SHA256 mismatch")

    report_files = document.get("report_files")
    if not isinstance(report_files, Mapping) or set(report_files) != set(
        selection_evaluator.REPORT_FILENAMES
    ):
        raise ValueError("selection report_files must name the complete evaluator output set")
    companions: Dict[str, Path] = {}
    companion_sha256: Dict[str, str] = {}
    for name in selection_evaluator.REPORT_FILENAMES:
        path = _resolve_recorded_path(
            report_files[name],
            base_dir=report_path.parent,
            context=f"selection report file {name}",
        )
        companions[name] = path
        companion_sha256[name] = file_sha256(path)
        expected_path = report_path.parent / selection_evaluator.REPORT_FILENAMES[name]
        if path != expected_path.resolve():
            raise ValueError(
                f"selection report file {name!r} is outside the canonical report layout"
            )
    if companions["shortlist_json"] != report_path:
        raise ValueError("selection report_files.shortlist_json does not identify this report")

    completion = _strict_json_load(
        companions["completion"], context="selection completion attestation"
    )
    if completion.get("schema_version") != 1 or completion.get("status") != "complete":
        raise ValueError("selection completion attestation is not complete schema version 1")
    completion_report = _resolve_recorded_path(
        completion.get("selection_report"),
        base_dir=companions["completion"].parent,
        context="selection completion report",
    )
    if completion_report != report_path:
        raise ValueError("selection completion points to a different shortlist report")
    if completion.get("selection_report_sha256") != actual_sha256:
        raise ValueError("selection completion report SHA256 mismatch")
    completion_reports = completion.get("report_files")
    expected_report_names = set(selection_evaluator.REPORT_FILENAMES) - {"completion"}
    if not isinstance(completion_reports, Mapping) or set(completion_reports) != expected_report_names:
        raise ValueError("selection completion report_files set is incomplete")
    for name in sorted(expected_report_names):
        record = completion_reports[name]
        if not isinstance(record, Mapping) or set(record) != {"path", "file_sha256"}:
            raise ValueError(f"selection completion report_files[{name!r}] is invalid")
        recorded_path = _resolve_recorded_path(
            record["path"],
            base_dir=companions["completion"].parent,
            context=f"selection completion report file {name}",
        )
        if recorded_path != companions[name]:
            raise ValueError(f"selection completion path mismatch for {name}")
        recorded_sha256 = _validate_sha256(
            record["file_sha256"],
            f"selection completion report file {name} SHA256",
        )
        if recorded_sha256 != companion_sha256[name]:
            raise ValueError(f"selection completion file SHA256 mismatch for {name}")
    completion_expected = {
        "protocol_sha256": protocol.content_sha256,
        "normalization_sha256": document.get("normalization_sha256"),
        "fallback_to_stage1_reference": document.get("fallback_to_stage1_reference"),
        "run_evidence": document.get("run_evidence"),
        "evaluation_contract": evaluation_contract,
    }
    for key, expected_value in completion_expected.items():
        if completion.get(key) != expected_value:
            raise ValueError(f"selection completion {key} disagrees with report")
    run_evidence = document.get("run_evidence")
    required_run_evidence = {
        "reference_run_id",
        "reference_run_manifest",
        "reference_run_manifest_file_sha256",
        "reference_stage_completion",
        "reference_stage_completion_sha256",
        "candidate_run_id",
        "candidate_run_manifest",
        "candidate_run_manifest_sha256",
        "candidate_run_manifest_file_sha256",
        "candidate_stage_completion",
        "candidate_stage_completion_sha256",
        "candidate_stage_steps",
    }
    if not isinstance(run_evidence, Mapping) or set(run_evidence) != required_run_evidence:
        raise ValueError("selection report run_evidence is incomplete")
    for run_key in ("reference_run_id", "candidate_run_id"):
        selection_evaluator._validate_run_id(run_evidence.get(run_key), f"run_evidence.{run_key}")
    _validate_sha256(
        run_evidence.get("candidate_run_manifest_sha256"),
        "run_evidence.candidate_run_manifest_sha256",
    )
    for prefix in ("reference", "candidate"):
        for suffix, hash_suffix in (
            ("run_manifest", "run_manifest_file_sha256"),
            ("stage_completion", "stage_completion_sha256"),
        ):
            path_key = f"{prefix}_{suffix}"
            hash_key = f"{prefix}_{hash_suffix}"
            evidence_path = _resolve_recorded_path(
                run_evidence[path_key],
                base_dir=report_path.parent,
                context=f"selection {path_key}",
            )
            evidence_sha256 = _validate_sha256(
                run_evidence[hash_key], f"selection {hash_key}"
            )
            if file_sha256(evidence_path) != evidence_sha256:
                raise ValueError(f"selection run evidence SHA256 mismatch for {path_key}")
    steps = run_evidence.get("candidate_stage_steps")
    if not isinstance(steps, list) or not steps or any(
        isinstance(step, bool) or not isinstance(step, int) or step <= 0 for step in steps
    ) or steps != sorted(set(steps)):
        raise ValueError("selection candidate_stage_steps must be unique and increasing")

    shortlist = document.get("shortlist")
    selected = document.get("selected")
    if not isinstance(shortlist, list) or not shortlist:
        raise ValueError("selection report shortlist must be a non-empty array")
    if not isinstance(selected, Mapping):
        raise ValueError("selection report selected must be an object")
    matches = [row for row in shortlist if isinstance(row, Mapping) and row == selected]
    if len(matches) != 1:
        raise ValueError("selection report selected row must occur exactly once in shortlist")
    final_rows = [
        row
        for row in shortlist
        if isinstance(row, Mapping) and row.get("final_selected") is True
    ]
    if len(final_rows) != 1 or final_rows[0] != selected:
        raise ValueError("selection report must mark exactly one final_selected row")
    fallback = document.get("fallback_to_stage1_reference")
    if not isinstance(fallback, bool):
        raise ValueError("selection report fallback flag must be boolean")
    if bool(selected.get("fallback_reference")) != fallback:
        raise ValueError("selection report fallback flags disagree")
    if fallback and (len(shortlist) != 1 or selected.get("candidate_id") != "stage1_reference"):
        raise ValueError("stage-1 fallback must be the sole shortlisted selection")

    selected_id = selected.get("candidate_id")
    if not isinstance(selected_id, str) or not selection_evaluator.IDENTIFIER_PATTERN.fullmatch(
        selected_id
    ):
        raise ValueError("selection report candidate_id is invalid")
    checkpoint_path = _resolve_recorded_path(
        selected.get("checkpoint_path"),
        base_dir=report_path.parent,
        context="selected checkpoint",
    )
    checkpoint_sha256 = _validate_sha256(
        selected.get("checkpoint_sha256"), "selected checkpoint SHA256"
    )
    if file_sha256(checkpoint_path) != checkpoint_sha256:
        raise ValueError("selected checkpoint file SHA256 mismatch")
    checkpoint_epoch = _positive_int(
        selected.get("checkpoint_epoch"), "selected checkpoint epoch"
    )
    checkpoint_step = _positive_int(
        selected.get("checkpoint_step"), "selected checkpoint step"
    )
    completion_checkpoint = _resolve_recorded_path(
        completion.get("selected_checkpoint_path"),
        base_dir=companions["completion"].parent,
        context="selection completion checkpoint",
    )
    if completion_checkpoint != checkpoint_path:
        raise ValueError("selection completion selected checkpoint path mismatch")
    if completion.get("selected_checkpoint_sha256") != checkpoint_sha256:
        raise ValueError("selection completion selected checkpoint SHA256 mismatch")

    csv_rows = _load_shortlist_csv(companions["shortlist_csv"])
    csv_selected = [row for row in csv_rows if row.get("final_selected") == "True"]
    if len(csv_selected) != 1:
        raise ValueError("selection shortlist CSV must contain one final_selected row")
    csv_row = csv_selected[0]
    expected_csv_values = {
        "candidate_id": selected_id,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_epoch": str(checkpoint_epoch),
        "checkpoint_step": str(checkpoint_step),
    }
    for key, expected_value in expected_csv_values.items():
        actual_value = csv_row.get(key)
        if key == "checkpoint_path" and isinstance(actual_value, str):
            actual_value = str(
                _resolve_recorded_path(
                    actual_value,
                    base_dir=companions["shortlist_csv"].parent,
                    context="shortlist CSV checkpoint",
                )
            )
        if actual_value != expected_value:
            raise ValueError(f"selection JSON/CSV mismatch for {key}")

    return SelectionArtifact(
        report_path=report_path,
        report_sha256=actual_sha256,
        document=document,
        selected_id=selected_id,
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=checkpoint_sha256,
        checkpoint_epoch=checkpoint_epoch,
        checkpoint_step=checkpoint_step,
        companion_sha256=dict(sorted(companion_sha256.items())),
    )


def resolve_frozen_test_domains(protocol: ResolvedProtocol) -> Tuple[FrozenTestDomain, ...]:
    """Resolve every protocol test list and enforce the registered 5+5 design."""

    specs = protocol.test_episode_lists
    if len(specs) != FORMAL_TEST_DOMAIN_COUNT:
        raise ValueError(
            "formal frozen test requires exactly two protocol test domains; "
            f"found={len(specs)}"
        )
    domains = []
    seen_paths: Dict[Path, str] = {}
    seen_manifest_domains = set()
    for name, spec in specs.items():
        if spec.expected_episode_count != FORMAL_EPISODES_PER_DOMAIN:
            raise ValueError(
                f"formal test domain {name!r} must preregister exactly "
                f"{FORMAL_EPISODES_PER_DOMAIN} episodes"
            )
        list_path = _strict_regular_file(spec.episode_list, context=f"test list {name}")
        resolved = resolve_episode_paths(
            [], list_path, project_root=REPO_ROOT, deduplicate=False
        )
        paths = tuple(Path(path).resolve() for path in resolved)
        if len(paths) != spec.expected_episode_count:
            raise ValueError(
                f"test domain {name!r} episode count mismatch: "
                f"protocol={spec.expected_episode_count} resolved={len(paths)}"
            )
        if len(set(paths)) != len(paths):
            raise ValueError(f"test domain {name!r} contains duplicate episodes")
        if spec.domain in seen_manifest_domains:
            raise ValueError(f"test domains reuse manifest domain {spec.domain!r}")
        seen_manifest_domains.add(spec.domain)
        for path in paths:
            previous = seen_paths.get(path)
            if previous is not None:
                raise ValueError(
                    f"test domains overlap: episode={path} domains={previous!r},{name!r}"
                )
            seen_paths[path] = name
        domains.append(
            FrozenTestDomain(
                name=name,
                manifest_domain=spec.domain,
                episode_list=list_path,
                episode_list_sha256=file_sha256(list_path),
                episode_paths=paths,
            )
        )
    return tuple(domains)


def load_and_validate_manifest(
    protocol: ResolvedProtocol,
    domains: Sequence[FrozenTestDomain],
) -> Tuple[DatasetManifest, str, Mapping[Path, Any]]:
    """Verify the pinned manifest, native UUIDs, all bytes, and test assignments."""

    spec = protocol.dataset_manifest
    if spec is None or spec.expected_sha256 is None:
        raise ValueError("formal frozen test requires a SHA256-pinned dataset manifest")
    manifest_path = _strict_regular_file(spec.path, context="dataset manifest")
    manifest = DatasetManifest.load(manifest_path, verify_files=True)
    validate_episode_uuid_provenance(manifest, allow_derived=False)
    manifest_sha256 = manifest.content_sha256
    if manifest_sha256 != spec.expected_sha256:
        raise ValueError(
            "dataset manifest SHA256 mismatch: "
            f"protocol={spec.expected_sha256} actual={manifest_sha256}"
        )
    entries = {entry.identity.path: entry for entry in manifest.episodes}
    for domain in domains:
        for episode_path in domain.episode_paths:
            entry = entries.get(episode_path)
            if entry is None:
                raise ValueError(f"test episode is absent from dataset manifest: {episode_path}")
            if entry.split != "test" or entry.domain != domain.manifest_domain:
                raise ValueError(
                    f"test manifest assignment mismatch for {episode_path}: "
                    f"split={entry.split!r} domain={entry.domain!r}"
                )
            if file_sha256(episode_path) != entry.identity.file_sha256:
                raise ValueError(f"test episode file SHA256 mismatch: {episode_path}")
    return manifest, manifest_sha256, entries


def validate_selected_checkpoint(
    selection: SelectionArtifact,
    *,
    protocol: ResolvedProtocol,
    normalization_sha256: str,
    manifest_sha256: str,
    domains: Sequence[FrozenTestDomain],
) -> Tuple[selection_evaluator.LoadedCheckpoint, Any]:
    """Load the selected v2 checkpoint and bind it to protocol/test provenance."""

    checkpoint = selection_evaluator.load_checkpoint_strict(
        candidate_id=selection.selected_id,
        role="frozen_test_selected",
        path=selection.checkpoint_path,
        expected_sha256=selection.checkpoint_sha256,
        expected_epoch=selection.checkpoint_epoch,
        expected_step=selection.checkpoint_step,
    )
    validate_checkpoint_compatibility(
        checkpoint.config,
        selection_evaluator._expected_config_from_protocol(protocol),
        keys=INIT_COMPATIBILITY_KEYS,
        require_keys=True,
    )
    integrity = checkpoint.payload["integrity"]
    for context, value in (
        ("checkpoint config protocol", checkpoint.config.get("protocol_sha256")),
        ("checkpoint integrity protocol", integrity.get("protocol_sha256")),
    ):
        if value != protocol.content_sha256:
            raise ValueError(f"{context} SHA256 mismatch")
    for context, value in (
        ("checkpoint config normalization", checkpoint.config.get("normalization_sha256")),
        ("checkpoint integrity normalization", integrity.get("normalization_sha256")),
        ("selection report normalization", selection.document.get("normalization_sha256")),
    ):
        if value != normalization_sha256:
            raise ValueError(f"{context} SHA256 mismatch")

    if len(protocol.stages) != 2:
        raise ValueError("formal frozen test requires the registered two-stage protocol")
    if checkpoint.stage_index < 0 or checkpoint.stage_index >= len(protocol.stages):
        raise ValueError("selected checkpoint stage index is absent from protocol")
    stage = protocol.stages[checkpoint.stage_index]
    if stage.name != checkpoint.stage_name:
        raise ValueError("selected checkpoint stage identity disagrees with protocol")
    fallback = selection.document.get("fallback_to_stage1_reference")
    expected_stage_index = 0 if fallback is True else 1
    if checkpoint.stage_index != expected_stage_index:
        raise ValueError(
            "selected checkpoint stage disagrees with shortlist fallback decision"
        )
    resolved_mode = resolve_validation_deployment_mode(
        policy_variant=protocol.model.policy_variant,
        requested_mode=stage.objective.validation_deployment_mode,
        train_latent_mode=stage.objective.train_latent_mode,
        lambda_prior=stage.objective.lambda_prior,
    )
    if resolved_mode != DEPLOYMENT_MODE:
        raise ValueError("selected checkpoint protocol stage is not deterministic-prior deployable")
    final_stage = protocol.stages[-1]
    final_mode = resolve_validation_deployment_mode(
        policy_variant=protocol.model.policy_variant,
        requested_mode=final_stage.objective.validation_deployment_mode,
        train_latent_mode=final_stage.objective.train_latent_mode,
        lambda_prior=final_stage.objective.lambda_prior,
    )
    if final_mode != DEPLOYMENT_MODE or final_stage.monitor.aggregation != AGGREGATION:
        raise ValueError(
            "final protocol stage must preregister deterministic prior and "
            "episode_uniform aggregation"
        )
    if checkpoint.config.get("validation_deployment_mode") != DEPLOYMENT_MODE:
        raise ValueError("selected checkpoint validation deployment mode is not prior")
    if checkpoint.config.get("validation_aggregation") != AGGREGATION:
        raise ValueError("selected checkpoint validation aggregation is not episode_uniform")
    if protocol.model.policy_variant not in {
        "force_aware_act",
        "force_aware_contact_cvae",
    }:
        raise ValueError("formal deterministic-prior test requires a contact-prior policy")

    provenance = checkpoint.config.get("data_provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("selected checkpoint is missing data_provenance")
    if provenance.get("dataset_manifest_sha256") != manifest_sha256:
        raise ValueError("selected checkpoint dataset manifest SHA256 mismatch")
    run_evidence = selection.document["run_evidence"]
    run_prefix = "reference" if fallback is True else "candidate"
    if checkpoint.config.get("run_id") != run_evidence.get(f"{run_prefix}_run_id"):
        raise ValueError("selected checkpoint run_id disagrees with selection evidence")
    run_manifest_path = _resolve_recorded_path(
        run_evidence[f"{run_prefix}_run_manifest"],
        base_dir=selection.report_path.parent,
        context="selected checkpoint run manifest",
    )
    run_manifest_document = _strict_json_load(
        run_manifest_path, context="selected checkpoint run manifest"
    )
    run_manifest_semantic_sha256 = selection_evaluator.canonical_json_sha256(
        run_manifest_document
    )
    if checkpoint.config.get("run_manifest_sha256") != run_manifest_semantic_sha256:
        raise ValueError(
            "selected checkpoint run_manifest_sha256 disagrees with run evidence"
        )
    if run_prefix == "candidate" and run_evidence.get(
        "candidate_run_manifest_sha256"
    ) != run_manifest_semantic_sha256:
        raise ValueError("selection candidate run manifest semantic SHA256 mismatch")
    list_hashes = provenance.get("episode_list_sha256")
    counts = provenance.get("episode_counts")
    test_counts = counts.get("tests") if isinstance(counts, Mapping) else None
    assignments = provenance.get("test_domain_assignments")
    if not isinstance(list_hashes, Mapping):
        raise ValueError("selected checkpoint is missing episode-list hashes")
    if not isinstance(test_counts, Mapping):
        raise ValueError("selected checkpoint is missing test episode counts")
    if not isinstance(assignments, Mapping):
        raise ValueError("selected checkpoint is missing test domain assignments")
    if set(assignments) != {domain.name for domain in domains}:
        raise ValueError("selected checkpoint test domain names disagree with protocol")
    if set(test_counts) != {domain.name for domain in domains}:
        raise ValueError("selected checkpoint test count domains disagree with protocol")
    for domain in domains:
        if list_hashes.get(str(domain.episode_list)) != domain.episode_list_sha256:
            raise ValueError(f"selected checkpoint test list SHA256 mismatch: {domain.name}")
        if test_counts.get(domain.name) != len(domain.episode_paths):
            raise ValueError(f"selected checkpoint test count mismatch: {domain.name}")
        assignment = assignments.get(domain.name)
        expected_assignment = {
            "domain": domain.manifest_domain,
            "episode_list": str(domain.episode_list),
            "resolved_paths_sha256": selection_evaluator.canonical_json_sha256(
                [str(path) for path in domain.episode_paths]
            ),
        }
        if assignment != expected_assignment:
            raise ValueError(f"selected checkpoint test assignment mismatch: {domain.name}")
    # Final-test deploy_loss always uses the final-stage preregistered weight,
    # including the explicit stage-1 fallback case, so model selection cannot
    # change the reported metric definition.
    return checkpoint, final_stage


def build_test_loaders(
    protocol: ResolvedProtocol,
    domains: Sequence[FrozenTestDomain],
    *,
    batch_size: int,
) -> Mapping[str, DataLoader]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
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
        indexed_paths = {Path(index.episode_path).resolve() for index in dataset.indices}
        missing = sorted(set(domain.episode_paths) - indexed_paths)
        if missing:
            raise ValueError(
                f"test domain {domain.name!r} contains episodes with zero usable samples: "
                + ", ".join(str(path) for path in missing)
            )
        loaders[domain.name] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            drop_last=False,
        )
    return loaders


def _normalized_batch(
    batch: Mapping[str, Any],
    stats: Mapping[str, Any],
    device: torch.device,
) -> Dict[str, Any]:
    normalized = dict(batch)
    normalized["qpos"] = normalize_tensor(
        normalized["qpos"], stats["qpos_mean"], stats["qpos_std"]
    )
    normalized["action_chunk"] = normalize_tensor(
        normalized["action_chunk"], stats["action_mean"], stats["action_std"]
    )
    normalized["force_window"] = normalize_tensor(
        normalized["force_window"], stats["force_mean"], stats["force_std"]
    )
    normalized["future_force_chunk"] = normalize_tensor(
        normalized["future_force_chunk"], stats["force_mean"], stats["force_std"]
    )
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in normalized.items()
    }


def evaluate_episode_metrics(
    *,
    model: torch.nn.Module,
    dataloaders: Mapping[str, DataLoader],
    domains: Sequence[FrozenTestDomain],
    manifest_entries: Mapping[Path, Any],
    device: torch.device,
    normalization_stats: Mapping[str, Any],
    lambda_force: float,
) -> Tuple[Mapping[str, Any], ...]:
    """Evaluate prior inference and retain one mean for each independent episode."""

    if not math.isfinite(lambda_force) or lambda_force < 0.0:
        raise ValueError("lambda_force must be finite and non-negative")
    domain_by_name = {domain.name: domain for domain in domains}
    if set(dataloaders) != set(domain_by_name):
        raise ValueError("test dataloaders must exactly match protocol test domains")
    module_states = [(module, module.training) for module in model.modules()]
    rows = []
    model.eval()
    try:
        with torch.inference_mode():
            for domain in domains:
                allowed_paths = set(domain.episode_paths)
                accumulators: Dict[Path, Dict[str, Any]] = {
                    path: {"count": 0, "action": 0.0, "force": 0.0, "indices": set()}
                    for path in domain.episode_paths
                }
                for raw_batch in dataloaders[domain.name]:
                    episode_paths = raw_batch.get("episode_path")
                    state_indices = raw_batch.get("state_index")
                    if not isinstance(episode_paths, (list, tuple)) or not torch.is_tensor(
                        state_indices
                    ):
                        raise ValueError(
                            "frozen test batches require episode_path and state_index provenance"
                        )
                    canonical_paths = tuple(Path(path).resolve() for path in episode_paths)
                    if any(path not in allowed_paths for path in canonical_paths):
                        raise ValueError("test dataloader yielded an unregistered episode")
                    indices = [int(value) for value in state_indices.detach().cpu().tolist()]
                    if len(indices) != len(canonical_paths):
                        raise ValueError("test batch state-index count mismatch")
                    batch = _normalized_batch(raw_batch, normalization_stats, device)
                    outputs = model(
                        images=batch["images"],
                        qpos=batch["qpos"],
                        force_window=batch["force_window"],
                        action_chunk=None,
                        future_force_chunk=None,
                        is_training=False,
                        contact_latent_mode=DEPLOYMENT_MODE,
                        deterministic_prior=True,
                    )
                    if not isinstance(outputs, Mapping):
                        raise ValueError("selected model output must be a mapping")
                    pred_action = outputs.get("pred_action")
                    pred_force = outputs.get("pred_force")
                    if not torch.is_tensor(pred_action) or not torch.is_tensor(pred_force):
                        raise ValueError("selected model must output pred_action and pred_force tensors")
                    if pred_action.shape != batch["action_chunk"].shape:
                        raise ValueError("pred_action shape differs from the registered target")
                    if pred_force.shape != batch["future_force_chunk"].shape:
                        raise ValueError("pred_force shape differs from the registered target")
                    action_values = (
                        (pred_action - batch["action_chunk"]).abs().flatten(1).mean(1)
                    )
                    force_values = (
                        (pred_force - batch["future_force_chunk"]).abs().flatten(1).mean(1)
                    )
                    if not torch.isfinite(action_values).all() or not torch.isfinite(
                        force_values
                    ).all():
                        raise ValueError("frozen test produced non-finite per-sample metrics")
                    for path, state_index, action_value, force_value in zip(
                        canonical_paths,
                        indices,
                        action_values.detach().cpu().tolist(),
                        force_values.detach().cpu().tolist(),
                    ):
                        accumulator = accumulators[path]
                        if state_index in accumulator["indices"]:
                            raise ValueError(
                                f"test dataloader repeated sample {path}:{state_index}"
                            )
                        accumulator["indices"].add(state_index)
                        accumulator["count"] += 1
                        accumulator["action"] += float(action_value)
                        accumulator["force"] += float(force_value)
                for episode_order, path in enumerate(domain.episode_paths):
                    accumulator = accumulators[path]
                    if accumulator["count"] <= 0:
                        raise ValueError(f"test episode produced zero samples: {path}")
                    action_l1 = accumulator["action"] / accumulator["count"]
                    force_l1 = accumulator["force"] / accumulator["count"]
                    deploy_loss = action_l1 + lambda_force * force_l1
                    if not all(math.isfinite(value) for value in (action_l1, force_l1, deploy_loss)):
                        raise ValueError(f"test episode produced non-finite metrics: {path}")
                    entry = manifest_entries[path]
                    rows.append(
                        {
                            "test_name": domain.name,
                            "manifest_domain": domain.manifest_domain,
                            "episode_order": episode_order,
                            "episode_uuid": entry.identity.episode_uuid,
                            "episode_path": str(path),
                            "episode_file_sha256": entry.identity.file_sha256,
                            "num_samples": accumulator["count"],
                            "action_l1": action_l1,
                            "force_l1": force_l1,
                            "deploy_loss": deploy_loss,
                        }
                    )
    finally:
        for module, training in module_states:
            module.training = training
    return tuple(rows)


def bootstrap_episode_mean(
    values: Sequence[float],
    *,
    seed: int,
    replicates: int,
    confidence: float = 0.95,
) -> Mapping[str, float]:
    """Return a deterministic percentile CI by resampling whole episodes."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or len(array) == 0 or not np.isfinite(array).all():
        raise ValueError("bootstrap values must be a non-empty finite vector")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("bootstrap seed must be a non-negative integer")
    if isinstance(replicates, bool) or not isinstance(replicates, int) or replicates < MIN_BOOTSTRAP_REPLICATES:
        raise ValueError(
            f"bootstrap replicates must be at least {MIN_BOOTSTRAP_REPLICATES}"
        )
    if not math.isfinite(confidence) or not 0.0 < confidence < 1.0:
        raise ValueError("bootstrap confidence must be in (0, 1)")
    generator = np.random.default_rng(seed)
    sample_indices = generator.integers(
        0, len(array), size=(replicates, len(array)), endpoint=False
    )
    bootstrap_means = array[sample_indices].mean(axis=1)
    tail = (1.0 - confidence) / 2.0
    return {
        "mean": float(array.mean()),
        "ci_low": float(np.quantile(bootstrap_means, tail)),
        "ci_high": float(np.quantile(bootstrap_means, 1.0 - tail)),
    }


def aggregate_domain_metrics(
    episode_rows: Sequence[Mapping[str, Any]],
    domains: Sequence[FrozenTestDomain],
    *,
    bootstrap_seed: int,
    bootstrap_replicates: int,
) -> Mapping[str, Any]:
    by_domain: Dict[str, list[Mapping[str, Any]]] = {domain.name: [] for domain in domains}
    for row in episode_rows:
        name = row.get("test_name")
        if name not in by_domain:
            raise ValueError(f"episode metric has unknown test domain: {name!r}")
        by_domain[name].append(row)
    summaries = {}
    for domain in domains:
        rows = by_domain[domain.name]
        if len(rows) != len(domain.episode_paths):
            raise ValueError(f"test domain {domain.name!r} episode metric count mismatch")
        domain_seed = int.from_bytes(
            hashlib.sha256(f"{bootstrap_seed}:{domain.name}".encode("utf-8")).digest()[:8],
            byteorder="big",
            signed=False,
        )
        metrics = {
            metric: bootstrap_episode_mean(
                [float(row[metric]) for row in rows],
                seed=domain_seed,
                replicates=bootstrap_replicates,
            )
            for metric in ("action_l1", "force_l1", "deploy_loss")
        }
        summaries[domain.name] = {
            "manifest_domain": domain.manifest_domain,
            "num_episodes": len(rows),
            "num_samples": sum(int(row["num_samples"]) for row in rows),
            "bootstrap_seed": domain_seed,
            "metrics": metrics,
        }
    return {
        "schema_version": FROZEN_TEST_SCHEMA_VERSION,
        "aggregation": AGGREGATION,
        "metric_units": "normalized_l1",
        "bootstrap": {
            "method": "episode_resampling_percentile",
            "confidence": 0.95,
            "base_seed": bootstrap_seed,
            "replicates": bootstrap_replicates,
        },
        "domains": summaries,
    }


def verify_immutable_inputs(
    *,
    protocol_path: Path,
    protocol_file_sha256: str,
    selection: SelectionArtifact,
    normalization_path: Path,
    normalization_file_sha256: str,
    manifest_path: Path,
    manifest_file_sha256: str,
    domains: Sequence[FrozenTestDomain],
    manifest_entries: Mapping[Path, Any],
) -> None:
    """Fail closed if any contract or test byte changed during evaluation."""

    expected_files: Dict[str, Tuple[Path, str]] = {
        "protocol": (protocol_path, protocol_file_sha256),
        "selection report": (selection.report_path, selection.report_sha256),
        "selected checkpoint": (selection.checkpoint_path, selection.checkpoint_sha256),
        "normalization statistics": (normalization_path, normalization_file_sha256),
        "dataset manifest": (manifest_path, manifest_file_sha256),
    }
    report_files = selection.document["report_files"]
    for name, expected_sha256 in selection.companion_sha256.items():
        expected_files[f"selection companion {name}"] = (
            _resolve_recorded_path(
                report_files[name],
                base_dir=selection.report_path.parent,
                context=f"selection companion {name}",
            ),
            expected_sha256,
        )
    run_evidence = selection.document["run_evidence"]
    for prefix in ("reference", "candidate"):
        for suffix, hash_suffix in (
            ("run_manifest", "run_manifest_file_sha256"),
            ("stage_completion", "stage_completion_sha256"),
        ):
            path_key = f"{prefix}_{suffix}"
            hash_key = f"{prefix}_{hash_suffix}"
            expected_files[f"selection {path_key}"] = (
                _resolve_recorded_path(
                    run_evidence[path_key],
                    base_dir=selection.report_path.parent,
                    context=f"selection {path_key}",
                ),
                run_evidence[hash_key],
            )
    for domain in domains:
        expected_files[f"test list {domain.name}"] = (
            domain.episode_list,
            domain.episode_list_sha256,
        )
        for index, episode_path in enumerate(domain.episode_paths):
            expected_files[f"test episode {domain.name}[{index}]"] = (
                episode_path,
                manifest_entries[episode_path].identity.file_sha256,
            )
    for context, (path, expected_sha256) in expected_files.items():
        path = _strict_regular_file(path, context=context)
        actual_sha256 = file_sha256(path)
        if actual_sha256 != expected_sha256:
            raise ValueError(
                f"{context} changed during frozen-test evaluation: "
                f"expected={expected_sha256} actual={actual_sha256}"
            )


def create_output_directory(path: Path) -> Tuple[Path, Mapping[str, Path]]:
    """Create one output directory exclusively and reject symlink traversal."""

    output_dir = _absolute_without_resolving(path)
    _reject_symlink_components(output_dir.parent, context="output directory parent")
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"frozen test refuses to reuse output directory: {output_dir}")
    if not output_dir.parent.is_dir():
        raise FileNotFoundError(
            f"frozen test output parent does not exist: {output_dir.parent}"
        )
    os.mkdir(output_dir, 0o750)
    paths = {name: output_dir / filename for name, filename in OUTPUT_FILENAMES.items()}
    return output_dir, paths


def _atomic_write_new(path: Path, payload: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite frozen-test artifact: {path}")
    temporary = path.parent / f".{path.name}.tmp-{uuid.uuid4().hex}"
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o640)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        # A hard-link publication is atomic and fails if the destination was
        # created concurrently; unlike replace(), it can never overwrite it.
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise FileExistsError(
                f"refusing to overwrite frozen-test artifact: {path}"
            ) from error
        temporary.unlink()
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        if temporary.exists():
            temporary.unlink()


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _episode_csv_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=EPISODE_METRIC_FIELDS, extrasaction="raise")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def write_frozen_test_outputs(
    *,
    output_paths: Mapping[str, Path],
    episode_rows: Sequence[Mapping[str, Any]],
    domain_metrics: Mapping[str, Any],
    inputs: Mapping[str, Any],
    evaluation_contract: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Atomically publish artifacts, writing the completion attestation last."""

    expected_names = set(OUTPUT_FILENAMES)
    if set(output_paths) != expected_names:
        raise ValueError("output path set is incomplete")
    episode_payload = _episode_csv_bytes(episode_rows)
    domain_payload = _json_bytes(domain_metrics)
    _atomic_write_new(output_paths["episode_metrics"], episode_payload)
    _atomic_write_new(output_paths["domain_metrics"], domain_payload)
    report = {
        "schema_version": FROZEN_TEST_SCHEMA_VERSION,
        "status": "evaluated",
        "inputs": inputs,
        "evaluation_contract": evaluation_contract,
        "domain_metrics": domain_metrics,
        "artifacts": {
            "episode_metrics": {
                "path": str(output_paths["episode_metrics"]),
                "sha256": hashlib.sha256(episode_payload).hexdigest(),
            },
            "domain_metrics": {
                "path": str(output_paths["domain_metrics"]),
                "sha256": hashlib.sha256(domain_payload).hexdigest(),
            },
        },
    }
    report_payload = _json_bytes(report)
    _atomic_write_new(output_paths["report"], report_payload)
    completion = {
        "schema_version": FROZEN_TEST_SCHEMA_VERSION,
        "status": "complete",
        "report_sha256": hashlib.sha256(report_payload).hexdigest(),
        "artifacts": {
            name: {
                "path": str(output_paths[name]),
                "sha256": file_sha256(output_paths[name]),
            }
            for name in ("episode_metrics", "domain_metrics", "report")
        },
    }
    _atomic_write_new(output_paths["completion"], _json_bytes(completion))
    return completion


def _resolve_device(value: str) -> torch.device:
    try:
        device = torch.device(value)
    except (TypeError, RuntimeError) as error:
        raise ValueError(f"invalid --device value: {value!r}") from error
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but torch.cuda.is_available() is false")
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    if device.type == "mps":
        backend = getattr(torch.backends, "mps", None)
        if backend is None or not backend.is_available():
            raise RuntimeError("MPS device requested but the MPS backend is unavailable")
    if device.type not in {"cpu", "cuda", "mps"}:
        raise ValueError("--device must select cpu, cuda, or mps")
    return device


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


def run(args: argparse.Namespace) -> Mapping[str, Any]:
    if args.batch_size != DEFAULT_BATCH_SIZE:
        raise ValueError(
            f"formal frozen test fixes --batch-size to {DEFAULT_BATCH_SIZE}"
        )
    if args.bootstrap_seed != DEFAULT_BOOTSTRAP_SEED:
        raise ValueError(
            "formal frozen test fixes --bootstrap-seed to "
            f"{DEFAULT_BOOTSTRAP_SEED}"
        )
    if args.bootstrap_replicates != DEFAULT_BOOTSTRAP_REPLICATES:
        raise ValueError(
            "formal frozen test fixes --bootstrap-replicates to "
            f"{DEFAULT_BOOTSTRAP_REPLICATES}"
        )
    protocol_path = _strict_regular_file(args.protocol, context="protocol")
    protocol_file_sha256 = file_sha256(protocol_path)
    protocol = load_protocol(protocol_path)
    if not protocol.deterministic:
        raise ValueError("formal frozen test requires protocol.deterministic=true")
    if protocol.normalization.expected_sha256 is None:
        raise ValueError("formal frozen test requires pinned normalization.sha256")
    normalization_path = (
        protocol.normalization.stats_path
        if args.normalization_stats is None
        else args.normalization_stats
    )
    normalization_path = _strict_regular_file(
        normalization_path, context="normalization statistics"
    )
    normalization_file_sha256 = file_sha256(normalization_path)
    domains = resolve_frozen_test_domains(protocol)
    selection = load_selection_artifact(
        args.selection_report,
        expected_sha256=args.selection_report_sha256,
        protocol=protocol,
        normalization_path=normalization_path,
    )
    manifest, manifest_sha256, manifest_entries = load_and_validate_manifest(
        protocol, domains
    )
    manifest_path = _strict_regular_file(
        protocol.dataset_manifest.path, context="dataset manifest"
    )
    manifest_file_sha256 = file_sha256(manifest_path)
    stats, normalization_sha256 = selection_evaluator.load_normalization_strict(
        normalization_path,
        expected_action_mode=protocol.dataset.action_mode,
        expected_domain_weights=protocol.normalization.domain_weights,
        strict_lengths=protocol.dataset.strict_lengths,
        expected_sha256=protocol.normalization.expected_sha256,
        validation_domains=domains,
        expected_qpos_dim=7,
        expected_action_dim=protocol.model.action_dim,
        expected_force_dim=protocol.model.force_dim,
    )
    selection_evaluator.validate_normalization_dataset_config(stats, protocol)
    checkpoint, stage = validate_selected_checkpoint(
        selection,
        protocol=protocol,
        normalization_sha256=normalization_sha256,
        manifest_sha256=manifest_sha256,
        domains=domains,
    )
    device = _resolve_device(args.device)
    # Claim the destination before the first model inference.  A failed run
    # deliberately leaves a non-complete directory that cannot be mistaken for
    # a valid result or silently reused by a concurrent/retry process.
    output_dir, output_paths = create_output_directory(args.output_dir)
    _enable_determinism(args.bootstrap_seed)
    loaders = build_test_loaders(protocol, domains, batch_size=args.batch_size)
    model = selection_evaluator._build_model_from_config(checkpoint.config).to(device)
    model.load_state_dict(checkpoint.payload["model_state_dict"], strict=True)
    episode_rows = evaluate_episode_metrics(
        model=model,
        dataloaders=loaders,
        domains=domains,
        manifest_entries=manifest_entries,
        device=device,
        normalization_stats=stats,
        lambda_force=float(stage.objective.lambda_force),
    )
    domain_metrics = aggregate_domain_metrics(
        episode_rows,
        domains,
        bootstrap_seed=args.bootstrap_seed,
        bootstrap_replicates=args.bootstrap_replicates,
    )
    verify_immutable_inputs(
        protocol_path=protocol_path,
        protocol_file_sha256=protocol_file_sha256,
        selection=selection,
        normalization_path=normalization_path,
        normalization_file_sha256=normalization_file_sha256,
        manifest_path=manifest_path,
        manifest_file_sha256=manifest_file_sha256,
        domains=domains,
        manifest_entries=manifest_entries,
    )
    inputs = {
        "protocol": {
            "path": str(protocol_path),
            "file_sha256": protocol_file_sha256,
            "content_sha256": protocol.content_sha256,
        },
        "selection_report": {
            "path": str(selection.report_path),
            "sha256": selection.report_sha256,
            "companion_sha256": selection.companion_sha256,
            "run_evidence": selection.document["run_evidence"],
        },
        "selected_checkpoint": {
            "candidate_id": selection.selected_id,
            "path": str(checkpoint.path),
            "sha256": checkpoint.file_sha256,
            "stage": checkpoint.stage_name,
            "stage_index": checkpoint.stage_index,
            "epoch": checkpoint.epoch,
            "step": checkpoint.step,
        },
        "normalization": {
            "path": str(normalization_path),
            "file_sha256": normalization_file_sha256,
            "content_sha256": normalization_sha256,
        },
        "dataset_manifest": {
            "path": str(manifest_path),
            "file_sha256": manifest_file_sha256,
            "content_sha256": manifest.content_sha256,
        },
        "test_domains": [
            {
                "name": domain.name,
                "manifest_domain": domain.manifest_domain,
                "episode_list": str(domain.episode_list),
                "episode_list_sha256": domain.episode_list_sha256,
                "episodes": [
                    {
                        "episode_uuid": manifest_entries[path].identity.episode_uuid,
                        "path": str(path),
                        "file_sha256": manifest_entries[path].identity.file_sha256,
                    }
                    for path in domain.episode_paths
                ],
            }
            for domain in domains
        ],
    }
    evaluation_contract = {
        "deployment_mode": DEPLOYMENT_MODE,
        "deterministic_prior": True,
        "aggregation": AGGREGATION,
        "metric_units": "normalized_l1",
        "test_domains": FORMAL_TEST_DOMAIN_COUNT,
        "episodes_per_domain": FORMAL_EPISODES_PER_DOMAIN,
        "batch_size": args.batch_size,
        "num_workers": 0,
        "device": str(device),
        "deterministic_algorithms": True,
        "bootstrap_seed": args.bootstrap_seed,
        "bootstrap_replicates": args.bootstrap_replicates,
    }
    completion = write_frozen_test_outputs(
        output_paths=output_paths,
        episode_rows=episode_rows,
        domain_metrics=domain_metrics,
        inputs=inputs,
        evaluation_contract=evaluation_contract,
    )
    print(
        json.dumps(
            {
                "status": completion["status"],
                "selected_checkpoint_sha256": checkpoint.file_sha256,
                "output_dir": str(output_dir),
                "completion": str(output_paths["completion"]),
            },
            sort_keys=True,
        )
    )
    return completion


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--selection-report", type=Path, required=True)
    parser.add_argument("--selection-report-sha256", required=True)
    parser.add_argument("--normalization-stats", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    parser.add_argument(
        "--bootstrap-replicates", type=int, default=DEFAULT_BOOTSTRAP_REPLICATES
    )
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
        OSError,
        TypeError,
        ValueError,
        RuntimeError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
