#!/usr/bin/env python3
"""Protocol-driven staged training for spatial and contact curricula.

This entry point intentionally does not replace ``train_minimal.py``.  It is a
strict path for new experiments: data populations are validated before model
construction, source/phase quotas are explicit, stage initialization and exact
resume have different CLI flags, and checkpoint selection keeps validation
domains separate.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import io
import json
import math
import os
import random
import re
import shutil
import sys
import tempfile
import uuid
from contextlib import ExitStack, contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
import h5py
import torch
from torch.optim import AdamW
from torch.utils.data import ConcatDataset, DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from force_aware_act.data import (  # noqa: E402
    ContactForceHDF5Dataset,
    validate_balanced_normalization_contract,
    validate_episode_uuid_provenance,
    validate_normalization_provenance_hashes,
)
from force_aware_act.data.manifest import (  # noqa: E402
    DatasetManifest,
    canonical_json_sha256,
)
from force_aware_act.training.catalog import PhaseCatalog  # noqa: E402
from force_aware_act.training.checkpointing import (  # noqa: E402
    INIT_COMPATIBILITY_KEYS,
    RESUME_COMPATIBILITY_KEYS,
    build_checkpoint_v2,
    file_sha256,
    initialize_model_from_checkpoint,
    resume_training_from_checkpoint,
    save_checkpoint_atomic,
    validate_checkpoint_v2_payload,
)
from force_aware_act.training.control import (  # noqa: E402
    EarlyStoppingState,
    RetentionGatedCheckpointSelector,
    evaluate_named_deployment_metrics,
    resolve_validation_deployment_mode,
)
from force_aware_act.training.engine import (  # noqa: E402
    move_batch_to_device,
    normalize_training_batch,
    train_one_update,
    validate_normalization_stats,
)
from force_aware_act.training.optim import (  # noqa: E402
    build_parameter_groups_from_specs,
)
from force_aware_act.training.policies import (  # noqa: E402
    build_policy,
    resolved_model_config,
)
from force_aware_act.training.protocol import (  # noqa: E402
    ResolvedProtocol,
    SourceSpec,
    StageSpec,
    load_protocol,
)
from force_aware_act.training.sampling import (  # noqa: E402
    DomainPhaseBatchSampler,
    SampleDescriptor,
)
from force_aware_act.utils import resolve_episode_paths  # noqa: E402


TRAIN_LOG_FIELDS = (
    "global_step",
    "stage_step",
    "epoch",
    "batch_in_epoch",
    "loss_total",
    "loss_action",
    "loss_force",
    "kl_motion",
    "kl_contact",
    "loss_prior",
    "beta_motion",
    "beta_contact",
    "gradient_norm",
    "gradient_was_clipped",
    "sampler_domain_counts",
    "sampler_phase_counts",
)
VALIDATION_LOG_FIELDS = (
    "validation_index",
    "global_step",
    "stage_step",
    "epoch",
    "domain",
    "deployment_mode",
    "deploy_loss",
    "action_l1",
    "force_l1",
    "num_samples",
    "num_episodes",
    "selected",
    "retention_passed",
    "decision_reason",
)
STAGE_COMPLETION_FILENAME = "stage_completion.json"
TRAINING_LOCK_FILENAME = ".train_staged.lock"
RUN_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")


@dataclass
class PreparedStage:
    protocol: ResolvedProtocol
    stage: StageSpec
    stage_index: int
    source_paths: Mapping[str, Tuple[Path, ...]]
    validation_paths: Mapping[str, Tuple[Path, ...]]
    test_paths: Mapping[str, Tuple[Path, ...]]
    normalization_population: Tuple[Path, ...]
    normalization_stats: Mapping[str, Any]
    normalization_sha256: str
    dataset_manifest: Optional[DatasetManifest]
    dataset_manifest_sha256: Optional[str]
    phase_catalogs: Mapping[str, PhaseCatalog]
    data_provenance: Mapping[str, Any]


@dataclass
class MonitorRuntime:
    kind: str
    early_stopping: Optional[EarlyStoppingState]
    retention_selector: Optional[RetentionGatedCheckpointSelector]
    validation_count: int = 0
    validations_without_selection: int = 0
    last_metrics: Optional[Mapping[str, Mapping[str, float]]] = None

    def state_dict(self) -> Dict[str, Any]:
        state: Dict[str, Any] = {
            "version": 1,
            "kind": self.kind,
            "validation_count": self.validation_count,
            "validations_without_selection": self.validations_without_selection,
            "last_metrics": self.last_metrics,
        }
        if self.early_stopping is not None:
            state["early_stopping"] = self.early_stopping.checkpoint_metadata()
        if self.retention_selector is not None:
            state["retention_selector"] = self.retention_selector.state_dict()
        return state


@dataclass(frozen=True)
class ResumeArtifactPlan:
    """Validated, recoverable filesystem changes for rollback-style resume."""

    quarantine_dir: Optional[Path]
    moves: Tuple[Tuple[Path, Path], ...]
    best_restore_source: Optional[Path]
    best_alias: Path


def configure_reproducibility(seed: int, deterministic: bool) -> None:
    """Configure all process RNGs before dataset/model construction."""

    if deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _validate_run_id(value: Any, context: str = "run_id") -> str:
    if not isinstance(value, str) or RUN_ID_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{context} must be 32 lowercase hexadecimal characters")
    return value


def seed_worker(_worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def _data_loader_generator(seed: int) -> torch.Generator:
    """Keep DataLoader iterator bookkeeping off the model's global RNG stream."""

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed % (2**63 - 1))
    return generator


def _training_code_sha256() -> str:
    """Hash the executable staged-training Python source tree."""

    files = [Path(__file__).resolve()]
    files.extend(sorted(SRC_ROOT.rglob("*.py")))
    descriptor = {
        str(path.relative_to(REPO_ROOT)): file_sha256(path)
        for path in files
        if path.is_file()
    }
    return canonical_json_sha256(descriptor)


def _runtime_versions() -> Dict[str, Any]:
    return {
        "python": sys.version,
        "torch": str(torch.__version__),
        "numpy": str(np.__version__),
        "h5py": str(h5py.__version__),
        "cuda_runtime": str(torch.version.cuda) if torch.version.cuda else None,
        "cudnn": (
            int(torch.backends.cudnn.version())
            if torch.backends.cudnn.is_available()
            else None
        ),
    }


def _torch_load(path: Path, map_location: str = "cpu") -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _load_json_object(path: Path, *, context: str) -> Mapping[str, Any]:
    try:
        document = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=lambda pairs: _reject_duplicate_pairs(
                pairs, context=context
            ),
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"{context} contains non-finite value {value}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid {context}: {path}") from error
    if not isinstance(document, Mapping):
        raise ValueError(f"{context} must contain a JSON object: {path}")
    return document


def _reject_duplicate_pairs(pairs, *, context: str):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"{context} contains duplicate JSON key {key!r}")
        result[key] = value
    return result


def _load_checkpoint_snapshot(
    path: Path,
    *,
    map_location: Any,
) -> Tuple[Mapping[str, Any], str, Path]:
    """Load and hash the same immutable byte snapshot, rejecting symlinks."""

    candidate = Path(path).expanduser()
    if candidate.is_symlink():
        raise ValueError(f"checkpoint path must not be a symlink: {candidate}")
    resolved = candidate.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {resolved}")
    payload_bytes = resolved.read_bytes()
    digest = hashlib.sha256(payload_bytes).hexdigest()
    try:
        payload = torch.load(
            io.BytesIO(payload_bytes),
            map_location=map_location,
            weights_only=False,
        )
    except TypeError:
        payload = torch.load(io.BytesIO(payload_bytes), map_location=map_location)
    if not isinstance(payload, Mapping):
        raise ValueError(f"checkpoint must contain a mapping: {resolved}")
    return payload, digest, resolved


@contextmanager
def _exclusive_training_lock(output_dir: Path):
    """Hold a non-blocking single-writer lock for one formal stage run.

    The lock file is deliberately persistent: deleting it on release would
    introduce an inode-replacement race between waiting processes.  ``flock``
    state itself is released by the kernel when the descriptor closes.
    """

    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    lock_path = output_dir / TRAINING_LOCK_FILENAME
    if lock_path.is_symlink():
        raise ValueError(f"training lock path must not be a symlink: {lock_path}")
    flags = os.O_CREAT | os.O_RDWR
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(lock_path, flags, 0o600)
    except OSError as error:
        raise RuntimeError(f"cannot open training lock: {lock_path}") from error
    handle = os.fdopen(descriptor, "r+", encoding="utf-8")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                f"another staged training writer is active: {output_dir}"
            ) from error
        handle.seek(0)
        handle.truncate()
        json.dump({"pid": os.getpid()}, handle, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
        yield lock_path
    finally:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def _resolve_list(path: Path) -> Tuple[Path, ...]:
    resolved = resolve_episode_paths(
        [], path, project_root=REPO_ROOT, deduplicate=False
    )
    if not resolved:
        raise ValueError(f"episode list is empty: {path}")
    canonical = tuple(Path(item).resolve() for item in resolved)
    if len(set(canonical)) != len(canonical):
        raise ValueError(f"episode list contains duplicate episodes: {path}")
    return canonical


def _canonical_paths(paths: Iterable[Path]) -> Tuple[Path, ...]:
    return tuple(sorted({Path(path).expanduser().resolve() for path in paths}))


def _require_disjoint(
    first_name: str,
    first: Iterable[Path],
    second_name: str,
    second: Iterable[Path],
) -> None:
    overlap = set(_canonical_paths(first)) & set(_canonical_paths(second))
    if overlap:
        preview = ", ".join(str(path) for path in sorted(overlap)[:5])
        raise ValueError(
            f"episode leakage between {first_name} and {second_name}: "
            f"count={len(overlap)} examples=[{preview}]"
        )


def _normalization_semantic_sha256(stats: Mapping[str, Any], stats_path: Path) -> str:
    recorded = stats.get("normalization_content_sha256")
    if recorded is None:
        return file_sha256(stats_path)
    if not isinstance(recorded, str) or len(recorded) != 64:
        raise ValueError("normalization_content_sha256 must be a SHA256 hex digest")
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
            "normalization statistics fail semantic content verification: "
            f"recorded={recorded} actual={actual}"
        )
    return actual


def _load_normalization(
    protocol: ResolvedProtocol,
    population: Sequence[Path],
    *,
    allow_legacy: bool,
    manifest: Optional[DatasetManifest] = None,
) -> Tuple[Mapping[str, Any], str]:
    stats_path = protocol.normalization.stats_path
    if not stats_path.is_file():
        raise FileNotFoundError(f"normalization stats do not exist: {stats_path}")
    stats = _torch_load(stats_path)
    if not isinstance(stats, Mapping):
        raise ValueError("normalization stats file must contain a mapping")
    validate_normalization_stats(stats)
    validate_normalization_provenance_hashes(
        stats,
        require_components=not allow_legacy,
    )
    if stats.get("action_mode") not in (None, protocol.dataset.action_mode):
        raise ValueError(
            "normalization action_mode mismatch: "
            f"stats={stats.get('action_mode')!r} protocol={protocol.dataset.action_mode!r}"
        )
    if not allow_legacy:
        validate_balanced_normalization_contract(
            stats,
            expected_action_mode=protocol.dataset.action_mode,
            expected_domain_weights=protocol.normalization.domain_weights,
            strict_lengths=protocol.dataset.strict_lengths,
        )
    expected_dimensions = {
        "qpos_mean": 7,
        "qpos_std": 7,
        "action_mean": protocol.model.action_dim,
        "action_std": protocol.model.action_dim,
        "force_mean": protocol.model.force_dim,
        "force_std": protocol.model.force_dim,
    }
    for key, expected_dimension in expected_dimensions.items():
        if stats[key].numel() != expected_dimension:
            raise ValueError(
                f"normalization {key} dimension mismatch: "
                f"stats={stats[key].numel()} expected={expected_dimension}"
            )
    _validate_normalization_dataset_semantics(
        stats,
        protocol,
        allow_legacy=allow_legacy,
    )
    recorded_paths = stats.get("population_paths", stats.get("episode_paths"))
    if recorded_paths is None:
        if not allow_legacy:
            raise ValueError(
                "normalization stats has no population provenance; "
                "use --allow-legacy-normalization only for temporary compatibility checks"
            )
    if manifest is not None:
        _validate_normalization_manifest_identities(
            stats,
            manifest,
            allow_legacy=allow_legacy,
        )
    else:
        if not isinstance(recorded_paths, (list, tuple)):
            raise ValueError("normalization population paths must be a list or tuple")
        expected = set(_canonical_paths(population))
        recorded = set(_canonical_paths(Path(path) for path in recorded_paths))
        if recorded != expected:
            raise ValueError(
                "normalization population does not equal protocol train union: "
                f"outside_union={len(recorded - expected)} "
                f"missing_from_stats={len(expected - recorded)}"
            )
    semantic_hash = _normalization_semantic_sha256(stats, stats_path)
    if (
        "normalization_content_sha256" not in stats
        and not allow_legacy
    ):
        raise ValueError(
            "legacy normalization stats lacks a semantic content hash; "
            "regenerate it with --domain or use --allow-legacy-normalization"
        )
    expected_hash = protocol.normalization.expected_sha256
    if expected_hash is None and not allow_legacy:
        raise ValueError(
            "formal training requires normalization.sha256 to pin the precomputed "
            "statistics"
        )
    if expected_hash is not None and semantic_hash != expected_hash:
        raise ValueError(
            "normalization SHA256 mismatch: "
            f"protocol={expected_hash} actual={semantic_hash}"
        )
    return stats, semantic_hash


def _validate_normalization_dataset_semantics(
    stats: Mapping[str, Any],
    protocol: ResolvedProtocol,
    *,
    allow_legacy: bool,
) -> None:
    """Reject stats generated for different input/target semantics."""

    expected = {
        "chunk_len": protocol.dataset.chunk_len,
        "force_window_len": protocol.dataset.force_window_len,
        "force_window_duration": protocol.dataset.force_window_duration,
        "camera_names": tuple(protocol.dataset.camera_names),
        "image_size": tuple(protocol.dataset.image_size),
        "imagenet_normalize": protocol.dataset.imagenet_normalize,
    }
    missing = []
    mismatches = []
    for key, expected_value in expected.items():
        if key not in stats:
            missing.append(key)
            continue
        actual_value = stats[key]
        if key in {"camera_names", "image_size"}:
            if not isinstance(actual_value, (list, tuple)):
                mismatches.append(f"{key}=invalid:{actual_value!r}")
                continue
            actual_value = tuple(actual_value)
        if actual_value != expected_value:
            mismatches.append(
                f"{key}: stats={actual_value!r} protocol={expected_value!r}"
            )
    if mismatches:
        raise ValueError(
            "normalization dataset semantics mismatch: " + "; ".join(mismatches)
        )
    if missing and not allow_legacy:
        raise ValueError(
            "normalization stats lacks dataset semantics fields: "
            + ", ".join(sorted(missing))
        )
    estimator = stats.get("normalization_estimator")
    if not allow_legacy and estimator != "balanced_raw":
        raise ValueError(
            "formal staged training requires normalization_estimator='balanced_raw'; "
            f"got {estimator!r}"
        )


def _validate_normalization_manifest_identities(
    stats: Mapping[str, Any],
    manifest: DatasetManifest,
    *,
    allow_legacy: bool,
) -> None:
    """Bind normalization provenance to manifest train paths, domains, and bytes."""

    identities = stats.get("population_identities")
    if identities is None:
        if allow_legacy:
            return
        raise ValueError(
            "normalization stats has no population_identities; regenerate balanced "
            "statistics from the manifest train lists"
        )
    if not isinstance(identities, (list, tuple)):
        raise ValueError("normalization population_identities must be a list or tuple")

    recorded: Dict[Path, Tuple[str, str]] = {}
    for index, identity in enumerate(identities):
        if not isinstance(identity, Mapping):
            raise ValueError(
                f"normalization population_identities[{index}] must be a mapping"
            )
        try:
            path = Path(identity["path"]).expanduser().resolve()
            domain = identity["domain"]
            digest = identity["file_sha256"]
        except KeyError as error:
            raise ValueError(
                f"normalization population_identities[{index}] is missing {error.args[0]!r}"
            ) from error
        if not isinstance(domain, str) or not domain:
            raise ValueError(
                f"normalization population_identities[{index}].domain must be non-empty"
            )
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(
                f"normalization population_identities[{index}].file_sha256 is invalid"
            )
        if path in recorded:
            raise ValueError(
                "normalization population_identities contains a duplicate path: "
                f"{path}"
            )
        recorded[path] = (domain, digest)

    manifest_train = {
        entry.identity.path: (entry.domain, entry.identity.file_sha256)
        for entry in manifest.episodes
        if entry.split == "train"
    }
    if recorded != manifest_train:
        missing = sorted(set(manifest_train) - set(recorded))
        unexpected = sorted(set(recorded) - set(manifest_train))
        mismatched = sorted(
            path
            for path in set(recorded) & set(manifest_train)
            if recorded[path] != manifest_train[path]
        )
        raise ValueError(
            "normalization identities do not match manifest train population: "
            f"missing={len(missing)} unexpected={len(unexpected)} "
            f"domain_or_sha_mismatch={len(mismatched)}"
        )


def _load_manifest(
    protocol: ResolvedProtocol,
    *,
    allow_legacy_data_contract: bool,
    verify_dataset_files: bool,
) -> Tuple[Optional[DatasetManifest], Optional[str]]:
    spec = protocol.dataset_manifest
    if spec is None:
        if allow_legacy_data_contract:
            return None, None
        raise ValueError(
            "protocol has no dataset_manifest; use --allow-legacy-data-contract "
            "only for temporary checks of historical datasets"
        )
    if spec.expected_sha256 is None and not allow_legacy_data_contract:
        raise ValueError(
            "formal training requires dataset_manifest.sha256 to pin the manifest"
        )
    manifest = DatasetManifest.load(spec.path, verify_files=verify_dataset_files)
    validate_episode_uuid_provenance(
        manifest,
        allow_derived=allow_legacy_data_contract,
    )
    digest = manifest.content_sha256
    if spec.expected_sha256 is not None and digest != spec.expected_sha256:
        raise ValueError(
            f"dataset manifest SHA256 mismatch: protocol={spec.expected_sha256} actual={digest}"
        )
    return manifest, digest


def _validate_manifest_assignments(
    manifest: DatasetManifest,
    protocol: ResolvedProtocol,
    all_stage_sources: Mapping[Tuple[str, str], Sequence[Path]],
    all_validation: Mapping[Tuple[str, str], Sequence[Path]],
    tests: Mapping[str, Sequence[Path]],
    normalization_population: Sequence[Path],
) -> None:
    entries = {entry.identity.path: entry for entry in manifest.episodes}

    def require(path: Path, expected_split: str, expected_domain: Optional[str], label: str) -> None:
        canonical = Path(path).resolve()
        if canonical not in entries:
            raise ValueError(f"{label} episode is absent from dataset manifest: {canonical}")
        entry = entries[canonical]
        if entry.split != expected_split:
            raise ValueError(
                f"{label} split mismatch for {canonical}: "
                f"manifest={entry.split!r} expected={expected_split!r}"
            )
        if expected_domain is not None and entry.domain != expected_domain:
            raise ValueError(
                f"{label} domain mismatch for {canonical}: "
                f"manifest={entry.domain!r} expected={expected_domain!r}"
            )

    for (stage_name, source_name), paths in all_stage_sources.items():
        source = next(
            source
            for stage in protocol.stages
            if stage.name == stage_name
            for source in stage.sources
            if source.name == source_name
        )
        for path in paths:
            require(path, "train", source.domain, f"stage={stage_name} source={source_name}")
    for (stage_name, validation_name), paths in all_validation.items():
        validation = next(
            validation
            for stage in protocol.stages
            if stage.name == stage_name
            for validation in stage.validation_domains
            if validation.name == validation_name
        )
        for path in paths:
            require(path, "val", validation.domain, f"stage={stage_name} validation={validation_name}")
    for name, paths in tests.items():
        expected_domain = protocol.test_episode_lists[name].domain
        for path in paths:
            require(path, "test", expected_domain, f"test={name}")
    manifest_train = {
        entry.identity.path for entry in manifest.episodes if entry.split == "train"
    }
    if set(_canonical_paths(normalization_population)) != manifest_train:
        raise ValueError(
            "normalization train union must exactly equal manifest train population: "
            f"manifest_only={len(manifest_train - set(normalization_population))} "
            f"protocol_only={len(set(normalization_population) - manifest_train)}"
        )


def _validate_phase_catalog_identity_binding(
    catalog: PhaseCatalog,
    *,
    source: SourceSpec,
    source_paths: Sequence[Path],
    manifest: Optional[DatasetManifest],
    manifest_sha256: Optional[str],
) -> None:
    """Bind a phase catalog to the pinned manifest and exact source population."""

    if not source.phase_quotas or source.sample_catalog is None:
        raise ValueError(
            f"source {source.name!r} has a phase catalog without phase_quotas"
        )
    expected_catalog_sha256 = source.sample_catalog_sha256
    if expected_catalog_sha256 is None:
        raise ValueError(
            f"source {source.name!r} is missing sample_catalog_sha256"
        )
    if catalog.content_sha256 != expected_catalog_sha256:
        raise ValueError(
            f"source {source.name!r} phase catalog SHA256 mismatch: "
            f"protocol={expected_catalog_sha256} actual={catalog.content_sha256}"
        )
    if manifest is None or manifest_sha256 is None:
        raise ValueError(
            f"source {source.name!r} phase catalog requires a pinned dataset manifest"
        )
    # Phase catalogs are a new formal contract.  Historical SHA-derived UUIDs
    # remain usable only for catalog-free compatibility checks.
    validate_episode_uuid_provenance(manifest, allow_derived=False)
    if catalog.dataset_manifest_sha256 != manifest_sha256:
        raise ValueError(
            f"source {source.name!r} phase catalog dataset manifest SHA256 mismatch: "
            f"catalog={catalog.dataset_manifest_sha256} actual={manifest_sha256}"
        )

    expected_paths = {Path(path).expanduser().resolve() for path in source_paths}
    catalog_paths = {episode.episode_path for episode in catalog.episodes}
    if catalog_paths != expected_paths:
        raise ValueError(
            f"source {source.name!r} phase catalog population mismatch: "
            f"source_only={len(expected_paths - catalog_paths)} "
            f"catalog_only={len(catalog_paths - expected_paths)}"
        )

    manifest_by_path = {entry.identity.path: entry for entry in manifest.episodes}
    for path in sorted(expected_paths):
        entry = manifest_by_path.get(path)
        if entry is None:
            raise ValueError(
                f"source {source.name!r} phase catalog episode is absent from "
                f"dataset manifest: {path}"
            )
        catalog_episode = catalog.episode_for(path)
        mismatches = []
        if entry.split != "train":
            mismatches.append(f"manifest split={entry.split!r}, expected 'train'")
        if entry.domain != source.domain:
            mismatches.append(
                f"manifest domain={entry.domain!r}, source={source.domain!r}"
            )
        if catalog_episode.domain != source.domain:
            mismatches.append(
                f"catalog domain={catalog_episode.domain!r}, source={source.domain!r}"
            )
        if catalog_episode.episode_uuid != entry.identity.episode_uuid:
            mismatches.append(
                "episode_uuid differs: "
                f"catalog={catalog_episode.episode_uuid} "
                f"manifest={entry.identity.episode_uuid}"
            )
        if catalog_episode.file_sha256 != entry.identity.file_sha256:
            mismatches.append(
                "file_sha256 differs: "
                f"catalog={catalog_episode.file_sha256} "
                f"manifest={entry.identity.file_sha256}"
            )
        if mismatches:
            raise ValueError(
                f"source {source.name!r} phase catalog identity mismatch for {path}: "
                + "; ".join(mismatches)
            )


def prepare_stage(
    protocol: ResolvedProtocol,
    stage_name: str,
    *,
    allow_legacy_data_contract: bool,
    allow_legacy_normalization: bool,
    verify_dataset_files: bool = True,
) -> PreparedStage:
    """Resolve and validate every population before constructing a model."""

    stage = protocol.stage(stage_name)
    stage_index = [item.name for item in protocol.stages].index(stage_name)
    list_cache: Dict[Path, Tuple[Path, ...]] = {}

    def resolve(path: Path) -> Tuple[Path, ...]:
        if path not in list_cache:
            if not path.is_file():
                raise FileNotFoundError(f"episode list does not exist: {path}")
            list_cache[path] = _resolve_list(path)
        return list_cache[path]

    def require_count(
        paths: Sequence[Path], expected_count: int, context: str
    ) -> None:
        if len(paths) != expected_count:
            raise ValueError(
                f"{context} episode count mismatch: "
                f"protocol={expected_count} resolved={len(paths)}"
            )

    normalization_lists = [resolve(path) for path in protocol.normalization.population_episode_lists]
    normalization_population = _canonical_paths(
        path for paths in normalization_lists for path in paths
    )
    if sum(len(paths) for paths in normalization_lists) != len(normalization_population):
        raise ValueError("normalization population lists contain duplicate episodes")

    all_stage_sources: Dict[Tuple[str, str], Tuple[Path, ...]] = {}
    all_validation: Dict[Tuple[str, str], Tuple[Path, ...]] = {}
    for protocol_stage in protocol.stages:
        stage_union = []
        for source in protocol_stage.sources:
            paths = resolve(source.episode_list)
            require_count(
                paths,
                source.expected_episode_count,
                f"stage {protocol_stage.name!r} source {source.name!r}",
            )
            all_stage_sources[(protocol_stage.name, source.name)] = paths
            stage_union.extend(paths)
        if len(set(stage_union)) != len(stage_union):
            raise ValueError(f"stage {protocol_stage.name!r} sources overlap")
        if not set(stage_union).issubset(set(normalization_population)):
            raise ValueError(
                f"stage {protocol_stage.name!r} training population is not a subset "
                "of the normalization train union"
            )
        stage_validation_union = []
        for validation in protocol_stage.validation_domains:
            paths = resolve(validation.episode_list)
            require_count(
                paths,
                validation.expected_episode_count,
                f"stage {protocol_stage.name!r} validation {validation.name!r}",
            )
            all_validation[(protocol_stage.name, validation.name)] = paths
            _require_disjoint(
                f"stage {protocol_stage.name} train",
                stage_union,
                f"stage {protocol_stage.name} validation {validation.name}",
                paths,
            )
            overlap = set(stage_validation_union) & set(paths)
            if overlap:
                raise ValueError(
                    f"stage {protocol_stage.name!r} validation domains overlap: "
                    f"count={len(overlap)}"
                )
            stage_validation_union.extend(paths)

    tests = {}
    for name, spec in protocol.test_episode_lists.items():
        paths = resolve(spec.episode_list)
        require_count(paths, spec.expected_episode_count, f"test {name!r}")
        tests[name] = paths
    seen_test_paths = set()
    for name, paths in tests.items():
        overlap = seen_test_paths & set(paths)
        if overlap:
            raise ValueError(f"test domains overlap at {name!r}: count={len(overlap)}")
        seen_test_paths.update(paths)
    validation_union = [path for paths in all_validation.values() for path in paths]
    test_union = [path for paths in tests.values() for path in paths]
    all_source_union = {
        path for paths in all_stage_sources.values() for path in paths
    }
    if set(normalization_population) != all_source_union:
        raise ValueError(
            "normalization train union must exactly equal all staged source episodes: "
            f"normalization_only={len(set(normalization_population) - all_source_union)} "
            f"source_only={len(all_source_union - set(normalization_population))}"
        )
    _require_disjoint("normalization train union", normalization_population, "validation", validation_union)
    _require_disjoint("normalization train union", normalization_population, "test", test_union)
    _require_disjoint("validation", validation_union, "test", test_union)

    manifest, manifest_hash = _load_manifest(
        protocol,
        allow_legacy_data_contract=allow_legacy_data_contract,
        verify_dataset_files=verify_dataset_files,
    )
    if manifest is not None:
        _validate_manifest_assignments(
            manifest,
            protocol,
            all_stage_sources,
            all_validation,
            tests,
            normalization_population,
        )

    all_phase_catalogs: Dict[Tuple[str, str], PhaseCatalog] = {}
    for protocol_stage in protocol.stages:
        for source in protocol_stage.sources:
            if source.sample_catalog is None:
                continue
            catalog = PhaseCatalog.load(source.sample_catalog)
            _validate_phase_catalog_identity_binding(
                catalog,
                source=source,
                source_paths=all_stage_sources[(protocol_stage.name, source.name)],
                manifest=manifest,
                manifest_sha256=manifest_hash,
            )
            all_phase_catalogs[(protocol_stage.name, source.name)] = catalog

    stats, normalization_hash = _load_normalization(
        protocol,
        normalization_population,
        allow_legacy=allow_legacy_normalization,
        manifest=manifest,
    )
    source_paths = {
        source.name: all_stage_sources[(stage.name, source.name)]
        for source in stage.sources
    }
    validation_paths = {
        validation.name: all_validation[(stage.name, validation.name)]
        for validation in stage.validation_domains
    }
    list_hashes = {
        str(path): file_sha256(path)
        for path in sorted(list_cache)
    }
    phase_catalogs = {
        source.name: all_phase_catalogs[(stage.name, source.name)]
        for source in stage.sources
        if source.sample_catalog is not None
    }
    catalog_hashes = {
        name: catalog.content_sha256 for name, catalog in phase_catalogs.items()
    }
    data_provenance = {
        "dataset_manifest_sha256": manifest_hash,
        "episode_list_sha256": list_hashes,
        "sample_catalog_sha256": catalog_hashes,
        "episode_counts": {
            "sources": {name: len(paths) for name, paths in source_paths.items()},
            "validation": {
                name: len(paths) for name, paths in validation_paths.items()
            },
            "tests": {name: len(paths) for name, paths in tests.items()},
            "normalization_population": len(normalization_population),
        },
        "stage_source_paths_sha256": canonical_json_sha256(
            {name: [str(path) for path in paths] for name, paths in source_paths.items()}
        ),
        "validation_paths_sha256": canonical_json_sha256(
            {name: [str(path) for path in paths] for name, paths in validation_paths.items()}
        ),
        "test_domain_assignments": {
            name: {
                "domain": protocol.test_episode_lists[name].domain,
                "episode_list": str(protocol.test_episode_lists[name].episode_list),
                "resolved_paths_sha256": canonical_json_sha256(
                    [str(path) for path in paths]
                ),
            }
            for name, paths in sorted(tests.items())
        },
    }
    return PreparedStage(
        protocol=protocol,
        stage=stage,
        stage_index=stage_index,
        source_paths=source_paths,
        validation_paths=validation_paths,
        test_paths=tests,
        normalization_population=normalization_population,
        normalization_stats=stats,
        normalization_sha256=normalization_hash,
        dataset_manifest=manifest,
        dataset_manifest_sha256=manifest_hash,
        phase_catalogs=phase_catalogs,
        data_provenance=data_provenance,
    )


def _build_dataset(protocol: ResolvedProtocol, paths: Sequence[Path]):
    spec = protocol.dataset
    return ContactForceHDF5Dataset(
        paths,
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


def _require_dataset_episode_coverage(
    dataset: ContactForceHDF5Dataset,
    paths: Sequence[Path],
    *,
    context: str,
) -> None:
    """Reject listed episodes that yield no usable training/validation state."""

    indexed_paths = {
        Path(index.episode_path).resolve() for index in dataset.indices
    }
    expected_paths = {Path(path).resolve() for path in paths}
    missing = sorted(expected_paths - indexed_paths)
    if missing:
        raise ValueError(
            f"{context} contains {len(missing)} episodes with zero usable samples: "
            + ", ".join(str(path) for path in missing[:5])
        )


def _validate_phase_catalog_semantics(
    catalog: PhaseCatalog,
    protocol: ResolvedProtocol,
    *,
    source_name: str,
) -> None:
    """Bind manual phase labels to the exact dataset indexing semantics."""

    semantics = catalog.labeler.get("dataset_semantics")
    if not isinstance(semantics, Mapping):
        raise ValueError(
            f"source {source_name!r} phase catalog is missing labeler.dataset_semantics"
        )
    dataset = protocol.dataset
    expected = {
        "action_mode": dataset.action_mode,
        "chunk_len": dataset.chunk_len,
        "force_window_len": dataset.force_window_len,
        "force_window_duration": dataset.force_window_duration,
        "camera_names": list(dataset.camera_names),
        "image_size": list(dataset.image_size),
        "normalize_images": True,
        "imagenet_normalize": dataset.imagenet_normalize,
        "image_alignment": dataset.image_alignment,
        "max_image_lag_seconds": dataset.max_image_lag_seconds,
        "include_force": True,
        "tolerate_length_mismatch": not dataset.strict_lengths,
        "max_length_mismatch": 0 if dataset.strict_lengths else 1,
    }
    mismatches = [
        f"{key}: catalog={semantics.get(key)!r} protocol={expected_value!r}"
        for key, expected_value in expected.items()
        if semantics.get(key) != expected_value
    ]
    if mismatches:
        raise ValueError(
            f"source {source_name!r} phase catalog dataset semantics mismatch: "
            + "; ".join(mismatches)
        )


def _validate_phase_label_set(
    *,
    source_name: str,
    requested_phases: Iterable[str],
    catalog_phases: Iterable[str],
) -> None:
    requested = set(requested_phases)
    actual = set(catalog_phases)
    if requested != actual:
        raise ValueError(
            f"source {source_name!r} phase labels must exactly match phase_quotas: "
            f"missing_from_catalog={sorted(requested - actual)} "
            f"unrequested_in_catalog={sorted(actual - requested)}"
        )


def _build_train_loader(
    prepared: PreparedStage,
    *,
    num_workers: int,
) -> Tuple[DataLoader, DomainPhaseBatchSampler, Mapping[str, int]]:
    datasets = []
    descriptors = []
    domain_quotas = {}
    phase_quotas = {}
    source_lengths = {}
    for source in prepared.stage.sources:
        dataset = _build_dataset(prepared.protocol, prepared.source_paths[source.name])
        if len(dataset) == 0:
            raise ValueError(f"source dataset is empty: {source.name}")
        _require_dataset_episode_coverage(
            dataset,
            prepared.source_paths[source.name],
            context=f"source {source.name!r}",
        )
        datasets.append(dataset)
        source_lengths[source.name] = len(dataset)
        domain_quotas[source.domain] = source.batch_quota
        if source.phase_quotas:
            if source.sample_catalog is None:
                raise ValueError(
                    f"source {source.name!r} defines phase_quotas but no sample_catalog"
                )
            try:
                catalog = prepared.phase_catalogs[source.name]
            except KeyError as error:
                raise ValueError(
                    f"source {source.name!r} has no prepared phase catalog"
                ) from error
            _validate_phase_catalog_identity_binding(
                catalog,
                source=source,
                source_paths=prepared.source_paths[source.name],
                manifest=prepared.dataset_manifest,
                manifest_sha256=prepared.dataset_manifest_sha256,
            )
            _validate_phase_catalog_semantics(
                catalog,
                prepared.protocol,
                source_name=source.name,
            )
            catalog.validate_indices(dataset.indices)
            phases = dict(source.phase_quotas)
        else:
            catalog = None
            phases = {"all": source.batch_quota}
        phase_quotas[source.domain] = phases
        source_descriptors = []
        catalog_phases = set()
        phase_episodes: Dict[str, set[str]] = {}
        for index in dataset.indices:
            phase = (
                catalog.phase_for(index.episode_path, index.state_index)
                if catalog is not None
                else "all"
            )
            catalog_phases.add(phase)
            phase_episodes.setdefault(phase, set()).add(
                str(Path(index.episode_path).resolve())
            )
            source_descriptors.append(
                SampleDescriptor(
                    domain=source.domain,
                    episode_id=str(Path(index.episode_path).resolve()),
                    phase=phase,
                )
            )
        if catalog is not None:
            _validate_phase_label_set(
                source_name=source.name,
                requested_phases=phases,
                catalog_phases=catalog_phases,
            )
            minimum = source.min_episodes_per_phase
            if minimum is None:
                raise ValueError(
                    f"source {source.name!r} is missing min_episodes_per_phase"
                )
            insufficient = {
                phase: len(phase_episodes.get(phase, set()))
                for phase in phases
                if len(phase_episodes.get(phase, set())) < minimum
            }
            if insufficient:
                raise ValueError(
                    f"source {source.name!r} phase episode coverage is below "
                    f"min_episodes_per_phase={minimum}: {insufficient}"
                )
        descriptors.extend(source_descriptors)
    combined = datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)
    batches_per_epoch = prepared.stage.samples_per_epoch // prepared.stage.batch_size
    sampler = DomainPhaseBatchSampler(
        descriptors,
        domain_quotas=domain_quotas,
        phase_quotas=phase_quotas,
        batches_per_epoch=batches_per_epoch,
        seed=prepared.protocol.seed + prepared.stage_index * 1009,
    )
    loader = DataLoader(
        combined,
        batch_sampler=sampler,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        # DataLoader draws a base seed whenever an iterator is constructed,
        # even with num_workers=0. A private generator prevents a mid-epoch
        # resume from consuming an extra model/dropout RNG value.
        generator=_data_loader_generator(
            prepared.protocol.seed + prepared.stage_index * 1009 + 17
        ),
    )
    return loader, sampler, source_lengths


def _build_validation_loaders(
    prepared: PreparedStage,
    *,
    num_workers: int,
) -> Tuple[Mapping[str, DataLoader], Mapping[str, int]]:
    loaders = {}
    lengths = {}
    for validation_index, validation in enumerate(prepared.stage.validation_domains):
        dataset = _build_dataset(prepared.protocol, prepared.validation_paths[validation.name])
        if len(dataset) == 0:
            raise ValueError(f"validation dataset is empty: {validation.name}")
        _require_dataset_episode_coverage(
            dataset,
            prepared.validation_paths[validation.name],
            context=f"validation {validation.name!r}",
        )
        lengths[validation.name] = len(dataset)
        loaders[validation.name] = DataLoader(
            dataset,
            batch_size=prepared.stage.batch_size,
            shuffle=False,
            num_workers=num_workers,
            worker_init_fn=seed_worker,
            generator=_data_loader_generator(
                prepared.protocol.seed
                + prepared.stage_index * 1009
                + validation_index
                + 101
            ),
        )
    return loaders, lengths


def _optimizer_group_manifest(parameter_groups: Sequence[Mapping[str, Any]]) -> list:
    return [
        {
            "name": group["name"],
            "param_names": list(group["param_names"]),
            "lr_multiplier": float(group["lr_multiplier"]),
            "lr": float(group["lr"]),
            "weight_decay": float(group["weight_decay"]),
        }
        for group in parameter_groups
    ]


def _checkpoint_config(
    prepared: PreparedStage,
    optimizer_groups: Sequence[Mapping[str, Any]],
    deployment_mode: str,
    training_device: str,
    run_id: str,
    stage_initial_global_step: int,
) -> Dict[str, Any]:
    protocol = prepared.protocol
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
        "optimizer_groups": _optimizer_group_manifest(optimizer_groups),
        "data_provenance": dict(prepared.data_provenance),
        "training_stage": prepared.stage.name,
        "stage_index": prepared.stage_index,
        "normalization_stats_path": str(protocol.normalization.stats_path),
        "normalization_sha256": prepared.normalization_sha256,
        "validation_deployment_mode": deployment_mode,
        "validation_aggregation": prepared.stage.monitor.aggregation,
        "freeze_vision_batch_norm": prepared.stage.freeze_vision_batch_norm,
        "training_device": training_device,
        "run_id": run_id,
        "stage_initial_global_step": stage_initial_global_step,
        "checkpoint_every_steps": prepared.stage.checkpoint_every_steps,
        "validation_every_steps": prepared.stage.validation_every_steps,
        "minimum_validations": prepared.stage.monitor.min_validations,
        "training_code_sha256": _training_code_sha256(),
        "runtime_versions": _runtime_versions(),
        "protocol_path": str(protocol.source_path),
        "protocol_sha256": protocol.content_sha256,
    }


def _validate_stage_transition_checkpoint(
    payload: Mapping[str, Any],
    prepared: PreparedStage,
) -> None:
    """Require Stage N to start from Stage N-1 with identical data semantics."""

    if prepared.stage_index <= 0:
        return
    validate_checkpoint_v2_payload(payload)
    expected_parent_index = prepared.stage_index - 1
    expected_parent_name = prepared.protocol.stages[expected_parent_index].name
    stage = payload["stage"]
    if stage.get("index") != expected_parent_index or stage.get("name") != expected_parent_name:
        raise ValueError(
            "stage transition checkpoint is not the immediately preceding stage: "
            f"checkpoint=({stage.get('name')!r}, {stage.get('index')!r}) "
            f"expected=({expected_parent_name!r}, {expected_parent_index})"
        )
    integrity = payload["integrity"]
    config = payload["config"]
    for location, value in (
        ("integrity.protocol_sha256", integrity.get("protocol_sha256")),
        ("config.protocol_sha256", config.get("protocol_sha256")),
    ):
        if value != prepared.protocol.content_sha256:
            raise ValueError(
                f"stage transition {location} mismatch: "
                f"checkpoint={value!r} current={prepared.protocol.content_sha256!r}"
            )
    for location, value in (
        ("integrity.normalization_sha256", integrity.get("normalization_sha256")),
        ("config.normalization_sha256", config.get("normalization_sha256")),
    ):
        if value != prepared.normalization_sha256:
            raise ValueError(
                f"stage transition {location} mismatch: "
                f"checkpoint={value!r} current={prepared.normalization_sha256!r}"
            )
    parent_provenance = config.get("data_provenance")
    if not isinstance(parent_provenance, Mapping):
        raise ValueError("stage transition checkpoint is missing data_provenance")
    for key in ("dataset_manifest_sha256", "episode_list_sha256"):
        if parent_provenance.get(key) != prepared.data_provenance.get(key):
            raise ValueError(
                f"stage transition data provenance mismatch for {key}"
            )
    monitor_state = payload["monitor_state"]
    training_state = payload["training_state"]
    kind = monitor_state.get("kind")
    if kind == "single":
        monitor_payload = monitor_state.get("early_stopping")
    elif kind == "retention":
        monitor_payload = monitor_state.get("retention_selector")
    else:
        raise ValueError("stage transition checkpoint has an invalid monitor kind")
    if not isinstance(monitor_payload, Mapping):
        raise ValueError("stage transition checkpoint is missing monitor selection state")
    selected_global_step = monitor_payload.get("best_step")
    is_retention_fallback = (
        kind == "retention"
        and selected_global_step is None
        and training_state.get("stage_step") == 0
    )
    if not is_retention_fallback and selected_global_step != training_state.get(
        "global_step"
    ):
        raise ValueError(
            "stage transition checkpoint is not the monitor-selected best: "
            f"selected_global_step={selected_global_step!r} "
            f"checkpoint_global_step={training_state.get('global_step')!r}"
        )


def _validate_completed_stage_best_checkpoint(
    checkpoint_path: Path,
    checkpoint_payload: Mapping[str, Any],
    checkpoint_sha256: Optional[str] = None,
) -> None:
    """Bind a stage transition to the completed run's final selected artifact."""

    if checkpoint_sha256 is None:
        checkpoint_sha256 = file_sha256(checkpoint_path)
    completion_path = checkpoint_path.parent / STAGE_COMPLETION_FILENAME
    if completion_path.is_symlink():
        raise ValueError(f"stage completion attestation must not be a symlink: {completion_path}")
    if not completion_path.is_file():
        raise FileNotFoundError(
            "stage transition requires a completed parent stage attestation: "
            f"{completion_path}"
        )
    completion = _load_json_object(
        completion_path, context="stage completion attestation"
    )
    if not isinstance(completion, Mapping) or completion.get("schema_version") != 2:
        raise ValueError("unsupported stage completion attestation")
    checkpoint_run_id = _validate_run_id(
        checkpoint_payload["config"].get("run_id"),
        "stage-transition checkpoint run_id",
    )
    if completion.get("run_id") != checkpoint_run_id:
        raise ValueError("stage completion run_id mismatch")
    run_manifest_path = checkpoint_path.parent / "run_manifest.json"
    if run_manifest_path.is_symlink() or not run_manifest_path.is_file():
        raise ValueError("completed stage run manifest is missing or symlinked")
    run_manifest = _load_json_object(
        run_manifest_path, context="completed stage run manifest"
    )
    run_manifest_sha256 = canonical_json_sha256(run_manifest)
    if completion.get("run_manifest_sha256") != run_manifest_sha256 or (
        checkpoint_payload["config"].get("run_manifest_sha256")
        != run_manifest_sha256
    ):
        raise ValueError("completed stage run manifest SHA256 mismatch")
    if completion.get("selected_best_checkpoint_sha256") != checkpoint_sha256:
        raise ValueError(
            "stage transition checkpoint is not the completed stage's final selected best"
        )

    final_path = checkpoint_path.parent / "checkpoint.pt"
    final_payload, final_sha256, _ = _load_checkpoint_snapshot(
        final_path,
        map_location="cpu",
    )
    validate_checkpoint_v2_payload(final_payload)
    if completion.get("final_checkpoint_sha256") != final_sha256:
        raise ValueError("stage completion final checkpoint SHA256 mismatch")
    if final_payload["stage"] != checkpoint_payload["stage"]:
        raise ValueError("completed final checkpoint stage mismatch")
    for key in ("config_sha256", "protocol_sha256", "normalization_sha256"):
        if final_payload["integrity"].get(key) != checkpoint_payload["integrity"].get(key):
            raise ValueError(f"completed final checkpoint {key} mismatch")
    if (
        final_payload["lineage"].get("parent_checkpoint_sha256")
        != checkpoint_payload["lineage"].get("parent_checkpoint_sha256")
    ):
        raise ValueError("completed final checkpoint parent lineage mismatch")
    completion_expected = {
        "stage": final_payload["stage"],
        "protocol_sha256": final_payload["integrity"]["protocol_sha256"],
        "normalization_sha256": final_payload["integrity"]["normalization_sha256"],
        "final_global_step": final_payload["training_state"]["global_step"],
        "final_stage_step": final_payload["training_state"]["stage_step"],
        "final_model_sha256": final_payload["integrity"]["model_state_sha256"],
    }
    for key, expected_value in completion_expected.items():
        if completion.get(key) != expected_value:
            raise ValueError(f"stage completion attestation {key} mismatch")
    expected_best_steps = _expected_best_checkpoint_steps(final_payload)
    if expected_best_steps is None or not _artifact_represents_selected_best(
        checkpoint_payload,
        expected_best_steps,
    ):
        raise ValueError(
            "stage transition checkpoint does not match the completed stage monitor"
        )
    if completion.get("selected_best_global_step") != expected_best_steps[0] or (
        completion.get("selected_best_stage_step") != expected_best_steps[1]
    ):
        raise ValueError("stage completion selected-best step mismatch")
    if (
        completion.get("selected_best_model_sha256")
        != checkpoint_payload["integrity"]["model_state_sha256"]
    ):
        raise ValueError("stage completion selected-best model SHA256 mismatch")


def _new_monitor(stage: StageSpec, initial_metrics=None) -> MonitorRuntime:
    monitor = stage.monitor
    if monitor.retention_domain is None:
        return MonitorRuntime(
            kind="single",
            early_stopping=EarlyStoppingState(
                patience=monitor.patience,
                min_epochs=0,
                min_delta=monitor.min_delta,
            ),
            retention_selector=None,
        )
    if initial_metrics is None:
        raise ValueError("retention-gated stages require initial validation metrics")
    baseline = float(initial_metrics[monitor.retention_domain][monitor.metric])
    initial_objective = float(initial_metrics[monitor.primary_domain][monitor.metric])
    selector = RetentionGatedCheckpointSelector(
        objective_domain=monitor.primary_domain,
        retention_domain=monitor.retention_domain,
        retention_baseline=baseline,
        metric=monitor.metric,
        max_relative_degradation=monitor.max_retention_regression,
        min_relative_improvement=monitor.min_delta,
        best_objective_value=initial_objective,
        best_retention_value=baseline,
    )
    return MonitorRuntime(
        kind="retention",
        early_stopping=None,
        retention_selector=selector,
        last_metrics=initial_metrics,
    )


def _restore_monitor(stage: StageSpec, state: Mapping[str, Any]) -> MonitorRuntime:
    if state.get("version") != 1:
        raise ValueError("unsupported monitor checkpoint state")
    kind = state.get("kind")
    runtime = MonitorRuntime(
        kind=str(kind),
        early_stopping=None,
        retention_selector=None,
        validation_count=int(state.get("validation_count", 0)),
        validations_without_selection=int(state.get("validations_without_selection", 0)),
        last_metrics=state.get("last_metrics"),
    )
    if kind == "single":
        metadata = state.get("early_stopping")
        if not isinstance(metadata, Mapping):
            raise ValueError("resume monitor is missing early_stopping state")
        runtime.early_stopping = EarlyStoppingState(
            patience=stage.monitor.patience,
            min_epochs=0,
            min_delta=stage.monitor.min_delta,
            best_metric=metadata.get("best_metric"),
            best_epoch=metadata.get("best_epoch"),
            best_step=metadata.get("best_step"),
            epochs_without_improvement=int(metadata.get("epochs_without_improvement", 0)),
        )
    elif kind == "retention":
        selector_state = state.get("retention_selector")
        if not isinstance(selector_state, Mapping):
            raise ValueError("resume monitor is missing retention_selector state")
        selector = RetentionGatedCheckpointSelector(
            objective_domain=stage.monitor.primary_domain,
            retention_domain=stage.monitor.retention_domain,
            retention_baseline=float(selector_state["retention_baseline"]),
            metric=stage.monitor.metric,
            max_relative_degradation=stage.monitor.max_retention_regression,
            min_relative_improvement=stage.monitor.min_delta,
        )
        selector.load_state_dict(selector_state)
        runtime.retention_selector = selector
    else:
        raise ValueError(f"unknown monitor kind in checkpoint: {kind!r}")
    return runtime


def _update_monitor(
    runtime: MonitorRuntime,
    stage: StageSpec,
    metrics: Mapping[str, Mapping[str, float]],
    *,
    epoch: int,
    global_step: int,
) -> Tuple[bool, bool, bool, str]:
    runtime.validation_count += 1
    runtime.last_metrics = metrics
    if runtime.kind == "single":
        value = float(metrics[stage.monitor.primary_domain][stage.monitor.metric])
        selected, _ = runtime.early_stopping.update(value, epoch=epoch, step=global_step)
        retention_passed = True
        reason = "selected" if selected else "objective_not_improved"
    else:
        decision = runtime.retention_selector.update(metrics, epoch=epoch, step=global_step)
        selected = decision.selected
        retention_passed = decision.retention_passed
        reason = decision.reason
    if selected:
        runtime.validations_without_selection = 0
    else:
        runtime.validations_without_selection += 1
    should_stop = _monitor_should_stop(runtime, stage)
    return selected, retention_passed, should_stop, reason


def _validate_validation_episode_counts(
    metrics: Mapping[str, Mapping[str, float]], stage: StageSpec
) -> None:
    expected = {
        domain.name: domain.expected_episode_count
        for domain in stage.validation_domains
    }
    if set(metrics) != set(expected):
        raise ValueError("validation metric domains differ from the stage protocol")
    for domain, expected_count in expected.items():
        try:
            observed = float(metrics[domain]["num_episodes"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"validation metrics for {domain!r} are missing num_episodes"
            ) from error
        if not math.isfinite(observed) or observed != float(expected_count):
            raise ValueError(
                f"validation domain {domain!r} observed episode count mismatch: "
                f"protocol={expected_count} observed={observed!r}"
            )


def _monitor_should_stop(runtime: MonitorRuntime, stage: StageSpec) -> bool:
    """Return the persisted early-stop condition without mutating state."""

    return (
        runtime.validation_count >= stage.monitor.min_validations
        and runtime.validations_without_selection >= stage.monitor.patience
    )


def _open_csv(
    stack: ExitStack,
    path: Path,
    fields: Sequence[str],
    *,
    resume: bool,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise ValueError(f"managed log path must not be a symlink: {path}")
    if resume:
        if not path.is_file():
            raise FileNotFoundError(f"exact resume requires existing log: {path}")
        with path.open("r", newline="") as existing:
            reader = csv.reader(existing)
            header = next(reader, None)
        if header != list(fields):
            raise ValueError(f"resume log header mismatch: {path}")
        handle = stack.enter_context(path.open("a", newline=""))
        return csv.DictWriter(handle, fieldnames=fields), handle
    if path.exists():
        raise FileExistsError(f"new training run refuses to overwrite log: {path}")
    handle = stack.enter_context(path.open("x", newline=""))
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    handle.flush()
    return writer, handle


def _flush_and_fsync_logs(*handles: Any) -> None:
    """Make CSV evidence durable before publishing a dependent checkpoint."""

    for handle in handles:
        handle.flush()
        os.fsync(handle.fileno())


def _checkpoint_payload(
    *,
    model,
    optimizer,
    config,
    prepared,
    sampler,
    monitor,
    global_step,
    stage_step,
    parent_checkpoint_sha256,
    stop_reason,
    resumed_from_sha256,
):
    batches_per_epoch = len(sampler)
    epoch = (stage_step - 1) // batches_per_epoch + 1 if stage_step else 0
    step_in_epoch = (stage_step - 1) % batches_per_epoch + 1 if stage_step else 0
    payload = build_checkpoint_v2(
        model=model,
        optimizer=optimizer,
        config=config,
        global_step=global_step,
        stage_step=stage_step,
        epoch=epoch,
        step_in_epoch=step_in_epoch,
        stage_name=prepared.stage.name,
        stage_index=prepared.stage_index,
        protocol_sha256=prepared.protocol.content_sha256,
        normalization_sha256=prepared.normalization_sha256,
        sampler=sampler,
        monitor_state=monitor.state_dict(),
        parent_checkpoint_sha256=parent_checkpoint_sha256,
        resumed_from_checkpoint_sha256=resumed_from_sha256,
    )
    payload["stop_reason"] = stop_reason
    payload["last_validation_metrics"] = monitor.last_metrics
    return payload


def _save_checkpoint(path: Path, **kwargs) -> str:
    return save_checkpoint_atomic(_checkpoint_payload(**kwargs), path)


def _save_best_checkpoint(output_dir: Path, *, stage_step: int, **kwargs) -> str:
    """Save an immutable best history entry before updating the stable alias."""

    history_path = output_dir / f"checkpoint_best_step_{stage_step:08d}.pt"
    if history_path.exists():
        raise FileExistsError(f"best checkpoint history already exists: {history_path}")
    payload = _checkpoint_payload(stage_step=stage_step, **kwargs)
    save_checkpoint_atomic(payload, history_path)
    best_alias = output_dir / "checkpoint_best.pt"
    _copy_file_atomic(history_path, best_alias)
    return file_sha256(best_alias)


def _write_run_manifest(
    path: Path,
    prepared: PreparedStage,
    config: Mapping[str, Any],
    source_lengths: Mapping[str, int],
    validation_lengths: Mapping[str, int],
    parent_checkpoint_sha256: Optional[str],
) -> None:
    if path.is_symlink():
        raise ValueError(f"run manifest path must not be a symlink: {path}")
    document = _run_manifest_document(
        prepared,
        config,
        source_lengths,
        validation_lengths,
        parent_checkpoint_sha256,
    )
    expected_sha256 = config.get("run_manifest_sha256")
    actual_sha256 = canonical_json_sha256(document)
    if expected_sha256 != actual_sha256:
        raise ValueError(
            "checkpoint config run_manifest_sha256 disagrees with run manifest: "
            f"config={expected_sha256!r} actual={actual_sha256}"
        )
    payload = (json.dumps(document, indent=2, sort_keys=True) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=str(path.parent),
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        # A hard link gives us atomic exclusive creation: unlike os.replace,
        # it cannot silently replace a run manifest created by another process.
        os.link(temporary_path, path)
        try:
            directory_fd = os.open(str(path.parent), os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            pass
    except FileExistsError as error:
        raise FileExistsError(f"refusing to overwrite run manifest: {path}") from error
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _run_manifest_document(
    prepared: PreparedStage,
    config: Mapping[str, Any],
    source_lengths: Mapping[str, int],
    validation_lengths: Mapping[str, int],
    parent_checkpoint_sha256: Optional[str],
) -> Dict[str, Any]:
    manifest_config = {
        key: value
        for key, value in config.items()
        if key != "run_manifest_sha256"
    }
    return {
        "schema_version": 3,
        "run_id": manifest_config.get("run_id"),
        "protocol_sha256": prepared.protocol.content_sha256,
        "normalization_sha256": prepared.normalization_sha256,
        "dataset_manifest_sha256": prepared.dataset_manifest_sha256,
        "stage": prepared.stage.name,
        "stage_index": prepared.stage_index,
        "source_dataset_samples": dict(source_lengths),
        "validation_dataset_samples": dict(validation_lengths),
        "config": manifest_config,
        "data_provenance": dict(prepared.data_provenance),
        "parent_checkpoint_sha256": parent_checkpoint_sha256,
        "initialization_mode": (
            "weights_only_stage_transition"
            if parent_checkpoint_sha256 is not None
            else "fresh"
        ),
    }


def _read_csv_rows(path: Path, fields: Sequence[str]) -> list[Dict[str, str]]:
    if path.is_symlink():
        raise ValueError(f"managed log path must not be a symlink: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"exact resume requires existing log: {path}")
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != list(fields):
            raise ValueError(f"resume log header mismatch: {path}")
        rows = []
        for row_number, row in enumerate(reader, start=2):
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"malformed resume log row {row_number}: {path}")
            rows.append(dict(row))
    return rows


def _rewrite_csv_atomic(
    path: Path,
    fields: Sequence[str],
    rows: Sequence[Mapping[str, str]],
) -> None:
    if path.is_symlink():
        raise ValueError(f"managed log path must not be a symlink: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _parse_log_int(row: Mapping[str, str], key: str, path: Path) -> int:
    try:
        value = int(row[key])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"resume log {path} has invalid integer field {key!r}") from error
    if value < 0:
        raise ValueError(f"resume log {path} has negative field {key!r}")
    return value


def _parse_log_bool(row: Mapping[str, str], key: str, path: Path) -> bool:
    value = row.get(key)
    if value == "True":
        return True
    if value == "False":
        return False
    raise ValueError(f"resume log {path} has invalid boolean field {key!r}")


def _parse_log_float(row: Mapping[str, str], key: str, path: Path) -> float:
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"resume log {path} has invalid float field {key!r}") from error
    if not math.isfinite(value):
        raise ValueError(f"resume log {path} has non-finite field {key!r}")
    return value


def _reconcile_resume_logs(
    *,
    train_log: Path,
    validation_log: Path,
    stage_step: int,
    global_step: int,
    validation_count: int,
    validation_domains: Sequence[str],
    has_initial_baseline: bool,
    monitor_state: Mapping[str, Any],
    allow_trim: bool,
) -> None:
    """Validate logs at a checkpoint boundary and optionally trim crash-ahead rows."""

    train_rows = _read_csv_rows(train_log, TRAIN_LOG_FIELDS)
    train_steps = [_parse_log_int(row, "stage_step", train_log) for row in train_rows]
    if train_steps != list(range(1, len(train_steps) + 1)):
        raise ValueError("resume train log stage_step values must be contiguous from 1")
    if len(train_rows) < stage_step:
        raise ValueError(
            f"resume train log is behind checkpoint: rows={len(train_rows)} "
            f"checkpoint_stage_step={stage_step}"
        )
    if stage_step > 0:
        recorded_global_step = _parse_log_int(
            train_rows[stage_step - 1], "global_step", train_log
        )
        if recorded_global_step != global_step:
            raise ValueError(
                "resume train log global_step disagrees with checkpoint: "
                f"log={recorded_global_step} checkpoint={global_step}"
            )
    train_ahead = len(train_rows) > stage_step

    validation_rows = _read_csv_rows(validation_log, VALIDATION_LOG_FIELDS)
    grouped: Dict[int, list[Dict[str, str]]] = {}
    row_indices = []
    for row in validation_rows:
        index = _parse_log_int(row, "validation_index", validation_log)
        row_indices.append(index)
        grouped.setdefault(index, []).append(row)
    compact_indices = [
        index
        for position, index in enumerate(row_indices)
        if position == 0 or index != row_indices[position - 1]
    ]
    if compact_indices != sorted(set(row_indices)):
        raise ValueError("resume validation log indices must be grouped and increasing")
    expected_indices = set(range(1, validation_count + 1))
    if has_initial_baseline:
        expected_indices.add(0)
    missing_indices = sorted(expected_indices - set(grouped))
    if missing_indices:
        raise ValueError(
            f"resume validation log is behind checkpoint; missing indices={missing_indices}"
        )
    domain_set = set(validation_domains)
    if not domain_set or len(domain_set) != len(validation_domains):
        raise ValueError("resume validation domains must be non-empty and unique")
    global_offset = global_step - stage_step
    expected_deployment_mode: Optional[str] = None
    group_metadata: Dict[int, Dict[str, Any]] = {}
    for index in sorted(grouped):
        rows = grouped[index]
        row_domains = [row["domain"] for row in rows]
        if len(row_domains) != len(set(row_domains)) or set(row_domains) != domain_set:
            raise ValueError(
                f"resume validation index {index} must contain each domain exactly once"
            )
        shared_int_fields = ("global_step", "stage_step", "epoch")
        parsed_ints = {
            key: {_parse_log_int(row, key, validation_log) for row in rows}
            for key in shared_int_fields
        }
        if any(len(values) != 1 for values in parsed_ints.values()):
            raise ValueError(
                f"resume validation index {index} has inconsistent step metadata"
            )
        parsed_bools = {
            key: {_parse_log_bool(row, key, validation_log) for row in rows}
            for key in ("selected", "retention_passed")
        }
        if any(len(values) != 1 for values in parsed_bools.values()):
            raise ValueError(
                f"resume validation index {index} has inconsistent decisions"
            )
        reasons = {row["decision_reason"] for row in rows}
        deployment_modes = {row["deployment_mode"] for row in rows}
        if len(reasons) != 1 or len(deployment_modes) != 1 or not next(
            iter(deployment_modes)
        ):
            raise ValueError(
                f"resume validation index {index} has inconsistent metadata"
            )
        deployment_mode = next(iter(deployment_modes))
        if expected_deployment_mode is None:
            expected_deployment_mode = deployment_mode
        elif deployment_mode != expected_deployment_mode:
            raise ValueError("resume validation deployment mode changed within the log")
        row_global_step = next(iter(parsed_ints["global_step"]))
        row_stage_step = next(iter(parsed_ints["stage_step"]))
        if row_global_step - row_stage_step != global_offset:
            raise ValueError(
                f"resume validation index {index} has an invalid global/stage step offset"
            )
        if index == 0:
            if not has_initial_baseline or row_stage_step != 0:
                raise ValueError("resume validation baseline must be at stage_step 0")
            if not next(iter(parsed_bools["selected"])) or next(iter(reasons)) != (
                "stage_initialization_baseline"
            ):
                raise ValueError("resume validation baseline decision is invalid")
        elif row_stage_step <= 0:
            raise ValueError(
                f"resume validation index {index} must have a positive stage_step"
            )
        group_metadata[index] = {
            "global_step": row_global_step,
            "stage_step": row_stage_step,
            "selected": next(iter(parsed_bools["selected"])),
        }

    positive_indices = sorted(index for index in grouped if index > 0)
    positive_steps = [group_metadata[index]["stage_step"] for index in positive_indices]
    if positive_steps != sorted(set(positive_steps)):
        raise ValueError("resume validation stage_step values must be strictly increasing")
    for index in sorted(expected_indices):
        if group_metadata[index]["stage_step"] > stage_step:
            raise ValueError(
                f"resume validation index {index} is ahead of the checkpoint boundary"
            )
    validation_ahead_indices = sorted(set(grouped) - expected_indices)
    if validation_ahead_indices and validation_ahead_indices != list(
        range(validation_count + 1, max(validation_ahead_indices) + 1)
    ):
        raise ValueError("resume validation log has non-contiguous future indices")
    for index in validation_ahead_indices:
        if group_metadata[index]["stage_step"] <= stage_step:
            raise ValueError(
                f"future validation index {index} is not ahead of the checkpoint boundary"
            )

    checkpoint_indices = sorted(index for index in expected_indices if index > 0)
    selected_steps = [
        group_metadata[index]["global_step"]
        for index in checkpoint_indices
        if group_metadata[index]["selected"]
    ]
    monitor_kind = monitor_state.get("kind")
    if monitor_kind == "single":
        selection_state = monitor_state.get("early_stopping")
    elif monitor_kind == "retention":
        selection_state = monitor_state.get("retention_selector")
    else:
        raise ValueError("resume monitor state has an invalid kind")
    if not isinstance(selection_state, Mapping):
        raise ValueError("resume monitor state is missing selection metadata")
    monitor_best_step = selection_state.get("best_step")
    logged_best_step = selected_steps[-1] if selected_steps else None
    if monitor_best_step != logged_best_step:
        raise ValueError(
            "resume validation selections disagree with checkpoint monitor best_step"
        )
    trailing_unselected = 0
    for index in reversed(checkpoint_indices):
        if group_metadata[index]["selected"]:
            break
        trailing_unselected += 1
    if monitor_state.get("validations_without_selection") != trailing_unselected:
        raise ValueError(
            "resume validation selections disagree with checkpoint monitor patience state"
        )

    metrics_index: Optional[int]
    if validation_count > 0:
        metrics_index = validation_count
    elif has_initial_baseline:
        metrics_index = 0
    else:
        metrics_index = None
    last_metrics = monitor_state.get("last_metrics")
    if metrics_index is not None:
        if not isinstance(last_metrics, Mapping):
            raise ValueError("resume monitor is missing last validation metrics")
        rows_by_domain = {row["domain"]: row for row in grouped[metrics_index]}
        for domain in validation_domains:
            recorded_metrics = last_metrics.get(domain)
            if not isinstance(recorded_metrics, Mapping):
                raise ValueError(
                    f"resume monitor is missing last metrics for domain {domain!r}"
                )
            for key in (
                "deploy_loss",
                "action_l1",
                "force_l1",
                "num_samples",
                "num_episodes",
            ):
                log_value = _parse_log_float(
                    rows_by_domain[domain], key, validation_log
                )
                try:
                    monitor_value = float(recorded_metrics[key])
                except (KeyError, TypeError, ValueError) as error:
                    raise ValueError(
                        f"resume monitor has invalid {key!r} for domain {domain!r}"
                    ) from error
                if not math.isfinite(monitor_value) or not math.isclose(
                    log_value, monitor_value, rel_tol=1e-12, abs_tol=1e-12
                ):
                    raise ValueError(
                        "resume validation metrics disagree with checkpoint monitor: "
                        f"domain={domain!r} metric={key!r}"
                    )

    if (train_ahead or validation_ahead_indices) and not allow_trim:
        raise ValueError(
            "resume logs are ahead of the selected checkpoint; use the latest "
            "checkpoint or pass --trim-resume-logs-to-checkpoint"
        )
    if allow_trim and train_ahead:
        _rewrite_csv_atomic(train_log, TRAIN_LOG_FIELDS, train_rows[:stage_step])
    if allow_trim and validation_ahead_indices:
        kept = [
            row
            for row in validation_rows
            if _parse_log_int(row, "validation_index", validation_log) in expected_indices
        ]
        _rewrite_csv_atomic(validation_log, VALIDATION_LOG_FIELDS, kept)


def _recover_stage_zero_logs_from_checkpoint(
    *,
    train_log: Path,
    validation_log: Path,
    resume_payload: Mapping[str, Any],
    validation_domains: Sequence[str],
) -> None:
    """Recover the narrow crash window after a retention baseline checkpoint."""

    state = resume_payload["training_state"]
    if state.get("stage_step") != 0:
        return
    monitor_state = resume_payload["monitor_state"]
    if monitor_state.get("kind") != "retention":
        return
    metrics = monitor_state.get("last_metrics")
    if not isinstance(metrics, Mapping) or set(metrics) != set(validation_domains):
        raise ValueError("stage-zero checkpoint cannot reconstruct validation metrics")
    deployment_mode = resume_payload["config"].get("validation_deployment_mode")
    if not isinstance(deployment_mode, str) or not deployment_mode:
        raise ValueError("stage-zero checkpoint is missing validation deployment mode")
    validation_rows: list[Dict[str, str]] = []
    for domain in validation_domains:
        domain_metrics = metrics[domain]
        if not isinstance(domain_metrics, Mapping):
            raise ValueError("stage-zero checkpoint has invalid domain metrics")
        validation_rows.append(
            {
                "validation_index": "0",
                "global_step": str(state["global_step"]),
                "stage_step": "0",
                "epoch": "0",
                "domain": domain,
                "deployment_mode": deployment_mode,
                "deploy_loss": str(domain_metrics["deploy_loss"]),
                "action_l1": str(domain_metrics["action_l1"]),
                "force_l1": str(domain_metrics["force_l1"]),
                "num_samples": str(domain_metrics["num_samples"]),
                "num_episodes": str(domain_metrics["num_episodes"]),
                "selected": "True",
                "retention_passed": "True",
                "decision_reason": "stage_initialization_baseline",
            }
        )

    # A process may stop after the baseline checkpoint is durable but before
    # either CSV is created, after only one header is created, or midway
    # through the baseline rows.  The stage-zero checkpoint is authoritative,
    # but recovery is safe only while no optimizer update or future validation
    # has been logged.
    if train_log.exists():
        if _read_csv_rows(train_log, TRAIN_LOG_FIELDS):
            return
    existing_validation_rows: list[Dict[str, str]] = []
    if validation_log.exists():
        existing_validation_rows = _read_csv_rows(
            validation_log, VALIDATION_LOG_FIELDS
        )
        expected_by_domain = {row["domain"]: row for row in validation_rows}
        observed_domains: set[str] = set()
        for row in existing_validation_rows:
            if _parse_log_int(row, "validation_index", validation_log) != 0:
                return
            domain = row.get("domain")
            if domain in observed_domains or domain not in expected_by_domain:
                raise ValueError("stage-zero validation log is not reconstructable")
            observed_domains.add(domain)
            expected = expected_by_domain[domain]
            for key in (
                "global_step",
                "stage_step",
                "epoch",
                "deployment_mode",
                "domain",
                "selected",
                "retention_passed",
                "decision_reason",
            ):
                if row.get(key) != expected[key]:
                    raise ValueError(
                        "stage-zero validation log disagrees with checkpoint: "
                        f"domain={domain!r} field={key!r}"
                    )
            for key in (
                "deploy_loss",
                "action_l1",
                "force_l1",
                "num_samples",
                "num_episodes",
            ):
                observed = _parse_log_float(row, key, validation_log)
                expected_value = float(expected[key])
                if not math.isclose(
                    observed, expected_value, rel_tol=1e-12, abs_tol=1e-12
                ):
                    raise ValueError(
                        "stage-zero validation log disagrees with checkpoint: "
                        f"domain={domain!r} field={key!r}"
                    )
    _rewrite_csv_atomic(train_log, TRAIN_LOG_FIELDS, [])
    _rewrite_csv_atomic(validation_log, VALIDATION_LOG_FIELDS, validation_rows)


def _load_matching_resume_artifact(
    path: Path,
    resume_payload: Mapping[str, Any],
) -> Mapping[str, Any]:
    payload = _torch_load(path)
    if not isinstance(payload, Mapping):
        raise ValueError(f"checkpoint artifact must contain a mapping: {path}")
    validate_checkpoint_v2_payload(payload)
    if payload["stage"] != resume_payload["stage"]:
        raise ValueError(f"checkpoint artifact belongs to a different stage: {path}")
    if (
        payload["integrity"]["config_sha256"]
        != resume_payload["integrity"]["config_sha256"]
    ):
        raise ValueError(f"checkpoint artifact config differs from resume checkpoint: {path}")
    for key in ("protocol_sha256", "normalization_sha256"):
        if payload["integrity"][key] != resume_payload["integrity"][key]:
            raise ValueError(f"checkpoint artifact {key} mismatch: {path}")
    if (
        payload["lineage"].get("parent_checkpoint_sha256")
        != resume_payload["lineage"].get("parent_checkpoint_sha256")
    ):
        raise ValueError(f"checkpoint artifact parent lineage mismatch: {path}")
    return payload


def _expected_best_checkpoint_steps(
    resume_payload: Mapping[str, Any],
) -> Optional[Tuple[int, int]]:
    """Return ``(global_step, stage_step)`` selected by the restored monitor."""

    monitor_state = resume_payload["monitor_state"]
    progress = resume_payload["training_state"]
    global_step = int(progress["global_step"])
    stage_step = int(progress["stage_step"])
    global_offset = global_step - stage_step
    kind = monitor_state.get("kind")
    if kind == "single":
        selection_state = monitor_state.get("early_stopping")
        if not isinstance(selection_state, Mapping):
            raise ValueError("resume monitor is missing early_stopping selection state")
        best_global_step = selection_state.get("best_step")
        if best_global_step is None:
            return None
    elif kind == "retention":
        selection_state = monitor_state.get("retention_selector")
        if not isinstance(selection_state, Mapping):
            raise ValueError("resume monitor is missing retention selection state")
        best_global_step = selection_state.get("best_step")
        if best_global_step is None:
            return global_offset, 0
    else:
        raise ValueError(f"resume monitor has unknown kind: {kind!r}")
    if (
        isinstance(best_global_step, bool)
        or not isinstance(best_global_step, int)
        or best_global_step < global_offset
        or best_global_step > global_step
    ):
        raise ValueError("resume monitor best_step is outside the restored stage history")
    return best_global_step, best_global_step - global_offset


def _artifact_represents_selected_best(
    payload: Mapping[str, Any],
    expected_steps: Tuple[int, int],
) -> bool:
    expected_global_step, expected_stage_step = expected_steps
    progress = payload["training_state"]
    if (
        progress.get("global_step") != expected_global_step
        or progress.get("stage_step") != expected_stage_step
    ):
        return False
    monitor_state = payload["monitor_state"]
    kind = monitor_state.get("kind")
    if kind == "single":
        selection_state = monitor_state.get("early_stopping")
        return bool(
            isinstance(selection_state, Mapping)
            and selection_state.get("best_step") == expected_global_step
        )
    if kind == "retention":
        selection_state = monitor_state.get("retention_selector")
        if not isinstance(selection_state, Mapping):
            return False
        selected_step = selection_state.get("best_step")
        return bool(
            selected_step == expected_global_step
            or (selected_step is None and expected_stage_step == 0)
        )
    return False


def _plan_resume_artifact_reconciliation(
    *,
    output_dir: Path,
    resume_payload: Mapping[str, Any],
    resume_checkpoint_sha256: str,
    allow_trim: bool,
) -> ResumeArtifactPlan:
    """Validate checkpoint artifacts and plan recoverable rollback operations."""

    resume_progress = resume_payload["training_state"]
    resume_stage_step = int(resume_progress["stage_step"])
    resume_model_sha256 = resume_payload["integrity"]["model_state_sha256"]
    loaded: Dict[Path, Mapping[str, Any]] = {}
    output_root = output_dir.resolve()

    def load(path: Path) -> Mapping[str, Any]:
        if path.is_symlink():
            raise ValueError(f"managed checkpoint artifact must not be a symlink: {path}")
        resolved = path.resolve()
        if resolved.parent != output_root:
            raise ValueError(f"checkpoint artifact escapes the output directory: {path}")
        if resolved not in loaded:
            loaded[resolved] = _load_matching_resume_artifact(resolved, resume_payload)
        return loaded[resolved]

    artifacts_to_move: Dict[Path, str] = {}
    periodic_by_stage_step: Dict[int, Path] = {}
    history_by_stage_step: Dict[int, Path] = {}
    named_patterns = (
        (
            "checkpoint_step_*.pt",
            re.compile(r"checkpoint_step_(\d{8,})\.pt"),
            periodic_by_stage_step,
        ),
        (
            "checkpoint_best_step_*.pt",
            re.compile(r"checkpoint_best_step_(\d{8,})\.pt"),
            history_by_stage_step,
        ),
    )
    for glob_pattern, pattern, destination in named_patterns:
        for path in sorted(output_dir.glob(glob_pattern)):
            match = pattern.fullmatch(path.name)
            if match is None or not path.is_file():
                continue
            filename_stage_step = int(match.group(1))
            payload = load(path)
            payload_stage_step = int(payload["training_state"]["stage_step"])
            if payload_stage_step != filename_stage_step:
                raise ValueError(
                    "checkpoint artifact filename disagrees with payload stage_step: "
                    f"{path}"
                )
            destination[filename_stage_step] = path.resolve()
            if filename_stage_step > resume_stage_step:
                artifacts_to_move[path.resolve()] = "future_named_checkpoint"

    for name in ("checkpoint.pt", "checkpoint_latest.pt"):
        path = output_dir / name
        if not path.is_file():
            continue
        payload = load(path)
        same_boundary = (
            payload["training_state"]["stage_step"] == resume_stage_step
            and payload["integrity"]["model_state_sha256"] == resume_model_sha256
        )
        if not same_boundary:
            artifacts_to_move[path.resolve()] = "stale_or_future_alias"

    completion_path = output_dir / STAGE_COMPLETION_FILENAME
    if completion_path.is_symlink():
        raise ValueError(
            f"managed stage completion artifact must not be a symlink: {completion_path}"
        )
    if completion_path.is_file():
        completion = _load_json_object(
            completion_path, context="managed stage completion artifact"
        )
        if not isinstance(completion, Mapping) or completion.get("schema_version") != 2:
            raise ValueError(f"invalid stage completion artifact: {completion_path}")
        completion_matches_boundary = (
            completion.get("final_stage_step") == resume_stage_step
            and completion.get("stage") == resume_payload["stage"]
            and completion.get("protocol_sha256")
            == resume_payload["integrity"]["protocol_sha256"]
            and completion.get("normalization_sha256")
            == resume_payload["integrity"]["normalization_sha256"]
            and completion.get("final_model_sha256") == resume_model_sha256
        )
        if not completion_matches_boundary:
            artifacts_to_move[completion_path.resolve()] = "future_stage_completion"

    expected_best_steps = _expected_best_checkpoint_steps(resume_payload)
    best_alias_path = output_dir / "checkpoint_best.pt"
    if best_alias_path.is_symlink():
        raise ValueError(
            f"managed checkpoint artifact must not be a symlink: {best_alias_path}"
        )
    best_alias = best_alias_path.resolve()
    if best_alias.parent != output_root:
        raise ValueError(
            f"checkpoint artifact escapes the output directory: {best_alias_path}"
        )
    best_alias_payload = load(best_alias) if best_alias.is_file() else None
    alias_is_correct = False
    if expected_best_steps is None:
        alias_is_correct = best_alias_payload is None
    elif best_alias_payload is not None:
        alias_is_correct = _artifact_represents_selected_best(
            best_alias_payload,
            expected_best_steps,
        )

    best_restore_source: Optional[Path] = None
    if not alias_is_correct:
        if best_alias_payload is not None:
            artifacts_to_move[best_alias] = "stale_or_future_best_alias"
        if expected_best_steps is not None:
            expected_global_step, expected_stage_step = expected_best_steps
            candidate_sources = (
                history_by_stage_step.get(expected_stage_step),
                periodic_by_stage_step.get(expected_stage_step),
            )
            for source in candidate_sources:
                if source is None:
                    continue
                source_payload = load(source)
                if _artifact_represents_selected_best(
                    source_payload,
                    (expected_global_step, expected_stage_step),
                ):
                    best_restore_source = source
                    break
            if best_restore_source is None:
                raise ValueError(
                    "cannot restore checkpoint_best.pt for the selected resume "
                    f"boundary; expected_stage_step={expected_stage_step}"
                )

    reconciliation_required = bool(artifacts_to_move) or best_restore_source is not None
    if reconciliation_required and not allow_trim:
        raise ValueError(
            "checkpoint artifacts are ahead of or inconsistent with the selected "
            "resume checkpoint; use the latest checkpoint or pass "
            "--trim-resume-logs-to-checkpoint to quarantine them"
        )
    if not reconciliation_required:
        return ResumeArtifactPlan(None, (), None, best_alias)

    quarantine_root = output_dir / "resume_quarantine"
    if quarantine_root.is_symlink():
        raise ValueError(f"resume quarantine root must not be a symlink: {quarantine_root}")
    quarantine_stem = (
        f"stage_step_{resume_stage_step:08d}_{resume_checkpoint_sha256[:12]}"
    )
    quarantine_dir = quarantine_root / quarantine_stem
    suffix = 1
    while quarantine_dir.exists():
        quarantine_dir = quarantine_root / f"{quarantine_stem}_{suffix:03d}"
        suffix += 1
    moves = tuple(
        (source, quarantine_dir / source.name)
        for source in sorted(artifacts_to_move)
    )
    return ResumeArtifactPlan(
        quarantine_dir=quarantine_dir,
        moves=moves,
        best_restore_source=best_restore_source,
        best_alias=best_alias,
    )


def _copy_file_atomic(source: Path, destination: Path) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        dir=str(destination.parent), prefix=f".{destination.name}.", suffix=".tmp"
    )
    temporary_path = Path(temporary_name)
    try:
        with source.open("rb") as source_handle, os.fdopen(
            descriptor, "wb"
        ) as destination_handle:
            shutil.copyfileobj(source_handle, destination_handle, length=1024 * 1024)
            destination_handle.flush()
            os.fsync(destination_handle.fileno())
        os.replace(temporary_path, destination)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _write_json_atomic(
    path: Path, document: Mapping[str, Any], *, overwrite: bool
) -> None:
    if path.is_symlink():
        raise ValueError(f"JSON artifact path must not be a symlink: {path}")
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite JSON artifact: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(document, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        if overwrite:
            os.replace(temporary_path, path)
        else:
            os.link(temporary_path, path)
        try:
            directory_fd = os.open(str(path.parent), os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            pass
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _periodic_candidate_records(
    output_dir: Path, final_payload: Mapping[str, Any]
) -> list[Dict[str, Any]]:
    """Return the complete, cadence-defined periodic checkpoint universe."""

    config = final_payload["config"]
    final_state = final_payload["training_state"]
    cadence = config.get("checkpoint_every_steps")
    if isinstance(cadence, bool) or not isinstance(cadence, int) or cadence < 0:
        raise ValueError("checkpoint config has invalid checkpoint_every_steps")
    final_stage_step = int(final_state["stage_step"])
    initial_global_step = config.get("stage_initial_global_step")
    if (
        isinstance(initial_global_step, bool)
        or not isinstance(initial_global_step, int)
        or initial_global_step < 0
    ):
        raise ValueError("checkpoint config has invalid stage_initial_global_step")
    expected_steps = (
        tuple(range(cadence, final_stage_step + 1, cadence)) if cadence else ()
    )
    pattern = re.compile(r"checkpoint_step_(\d{8,})\.pt")
    actual_by_step: Dict[int, Path] = {}
    for path in sorted(output_dir.glob("checkpoint_step_*.pt")):
        if path.is_symlink():
            raise ValueError(f"periodic checkpoint must not be a symlink: {path}")
        match = pattern.fullmatch(path.name)
        if match is None:
            raise ValueError(f"invalid periodic checkpoint filename: {path.name}")
        step = int(match.group(1))
        if step in actual_by_step:
            raise ValueError(f"duplicate periodic checkpoint stage_step={step}")
        actual_by_step[step] = path
    if tuple(sorted(actual_by_step)) != expected_steps:
        raise ValueError(
            "periodic checkpoint universe differs from the configured cadence: "
            f"expected={list(expected_steps)} actual={sorted(actual_by_step)}"
        )

    records = []
    for stage_step in expected_steps:
        payload, digest, resolved = _load_checkpoint_snapshot(
            actual_by_step[stage_step], map_location="cpu"
        )
        if not isinstance(payload, Mapping):
            raise ValueError(f"periodic checkpoint must contain a mapping: {resolved}")
        validate_checkpoint_v2_payload(payload)
        state = payload["training_state"]
        if state.get("stage_step") != stage_step:
            raise ValueError(f"periodic checkpoint stage_step mismatch: {resolved}")
        if state.get("global_step") != initial_global_step + stage_step:
            raise ValueError(f"periodic checkpoint global_step mismatch: {resolved}")
        if payload.get("stop_reason") != "periodic_checkpoint":
            raise ValueError(f"periodic checkpoint stop_reason mismatch: {resolved}")
        if payload["stage"] != final_payload["stage"]:
            raise ValueError(f"periodic checkpoint stage mismatch: {resolved}")
        if (
            payload["integrity"].get("config_sha256")
            != final_payload["integrity"].get("config_sha256")
        ):
            raise ValueError(f"periodic checkpoint config mismatch: {resolved}")
        if payload["lineage"].get("parent_checkpoint_sha256") != final_payload[
            "lineage"
        ].get("parent_checkpoint_sha256"):
            raise ValueError(f"periodic checkpoint lineage mismatch: {resolved}")
        records.append(
            {
                "stage_step": stage_step,
                "global_step": state["global_step"],
                "epoch": state["epoch"],
                "checkpoint_path": str(resolved),
                "checkpoint_sha256": digest,
                "model_sha256": payload["integrity"]["model_state_sha256"],
            }
        )
    return records


def _write_stage_completion(
    output_dir: Path,
    final_payload: Mapping[str, Any],
    *,
    overwrite: bool,
    minimum_validations: int,
) -> None:
    validation_count = final_payload["monitor_state"].get("validation_count")
    if (
        isinstance(validation_count, bool)
        or not isinstance(validation_count, int)
        or validation_count < minimum_validations
    ):
        raise ValueError(
            "cannot attest stage completion before monitor.min_validations: "
            f"observed={validation_count!r} required={minimum_validations}"
        )
    config = final_payload["config"]
    run_id = _validate_run_id(config.get("run_id"), "checkpoint run_id")
    run_manifest_path = output_dir / "run_manifest.json"
    if run_manifest_path.is_symlink():
        raise ValueError(f"run manifest must not be a symlink: {run_manifest_path}")
    if not run_manifest_path.is_file():
        raise FileNotFoundError(f"stage completion requires run manifest: {run_manifest_path}")
    run_manifest = _load_json_object(run_manifest_path, context="run manifest")
    run_manifest_sha256 = canonical_json_sha256(run_manifest)
    if run_manifest_sha256 != config.get("run_manifest_sha256"):
        raise ValueError("run manifest SHA256 disagrees with final checkpoint")
    if run_manifest.get("run_id") != run_id:
        raise ValueError("run manifest run_id disagrees with final checkpoint")
    candidate_checkpoints = _periodic_candidate_records(output_dir, final_payload)
    final_path = output_dir / "checkpoint.pt"
    best_path = output_dir / "checkpoint_best.pt"
    best_payload, best_sha256, _ = _load_checkpoint_snapshot(
        best_path,
        map_location="cpu",
    )
    validate_checkpoint_v2_payload(best_payload)
    expected_best_steps = _expected_best_checkpoint_steps(final_payload)
    if expected_best_steps is None or not _artifact_represents_selected_best(
        best_payload,
        expected_best_steps,
    ):
        raise ValueError("cannot attest stage completion without the final selected best")
    document = {
        "schema_version": 2,
        "run_id": run_id,
        "run_manifest": str(run_manifest_path.resolve()),
        "run_manifest_sha256": run_manifest_sha256,
        "stage": dict(final_payload["stage"]),
        "protocol_sha256": final_payload["integrity"]["protocol_sha256"],
        "normalization_sha256": final_payload["integrity"]["normalization_sha256"],
        "final_checkpoint": str(final_path.resolve()),
        "final_checkpoint_sha256": file_sha256(final_path),
        "final_global_step": final_payload["training_state"]["global_step"],
        "final_stage_step": final_payload["training_state"]["stage_step"],
        "final_model_sha256": final_payload["integrity"]["model_state_sha256"],
        "validation_count": validation_count,
        "minimum_validations": minimum_validations,
        "checkpoint_every_steps": config["checkpoint_every_steps"],
        "stage_initial_global_step": config["stage_initial_global_step"],
        "candidate_checkpoints": candidate_checkpoints,
        "selected_best_checkpoint": str(best_path.resolve()),
        "selected_best_checkpoint_sha256": best_sha256,
        "selected_best_global_step": expected_best_steps[0],
        "selected_best_stage_step": expected_best_steps[1],
        "selected_best_model_sha256": best_payload["integrity"][
            "model_state_sha256"
        ],
    }
    destination = output_dir / STAGE_COMPLETION_FILENAME
    if destination.is_symlink():
        raise ValueError(f"stage completion path must not be a symlink: {destination}")
    if destination.exists() and not overwrite:
        raise FileExistsError(f"stage completion already exists: {destination}")
    _write_json_atomic(destination, document, overwrite=overwrite)


def _apply_resume_artifact_plan(plan: ResumeArtifactPlan) -> None:
    if plan.quarantine_dir is None:
        return
    records = [
        {
            "source": str(source),
            "quarantined_to": str(destination),
            "file_sha256": file_sha256(source),
        }
        for source, destination in plan.moves
    ]
    plan.quarantine_dir.mkdir(parents=True, exist_ok=False)
    manifest_path = plan.quarantine_dir / "quarantine_manifest.json"
    manifest = {
        "schema_version": 2,
        "status": "moving",
        "moved_artifacts": records,
        "restored_best_from": (
            str(plan.best_restore_source)
            if plan.best_restore_source is not None
            else None
        ),
        "restored_best_to": (
            str(plan.best_alias) if plan.best_restore_source is not None else None
        ),
    }
    _write_json_atomic(manifest_path, manifest, overwrite=False)
    for source, destination in plan.moves:
        os.replace(source, destination)
    if plan.best_restore_source is not None:
        _copy_file_atomic(plan.best_restore_source, plan.best_alias)
    manifest["status"] = "complete"
    _write_json_atomic(manifest_path, manifest, overwrite=True)


def _train_impl(
    args: argparse.Namespace,
    *,
    loaded_protocol: Optional[ResolvedProtocol] = None,
    locked_output_dir: Optional[Path] = None,
) -> int:
    compatibility_flags = {
        "--allow-legacy-data-contract": args.allow_legacy_data_contract,
        "--allow-legacy-normalization": args.allow_legacy_normalization,
        "--skip-dataset-file-verification": args.skip_dataset_file_verification,
    }
    enabled_compatibility_flags = [
        name for name, enabled in compatibility_flags.items() if enabled
    ]
    if enabled_compatibility_flags and not args.dry_run:
        raise ValueError(
            "historical compatibility flags are restricted to --dry-run: "
            + ", ".join(enabled_compatibility_flags)
        )
    if args.num_workers != 0:
        raise ValueError(
            "staged training currently requires --num-workers 0 so exact resume "
            "cannot be invalidated by DataLoader prefetch"
        )
    protocol = loaded_protocol if loaded_protocol is not None else load_protocol(args.protocol)
    prepared = prepare_stage(
        protocol,
        args.stage,
        allow_legacy_data_contract=args.allow_legacy_data_contract,
        allow_legacy_normalization=args.allow_legacy_normalization,
        verify_dataset_files=not args.skip_dataset_file_verification,
    )
    if args.init_from is not None and args.resume_from is not None:
        raise ValueError("--init-from and --resume-from are mutually exclusive")
    if prepared.stage_index > 0 and args.init_from is None and args.resume_from is None:
        raise ValueError("non-initial stages require --init-from or --resume-from")
    if args.trim_resume_logs_to_checkpoint and args.resume_from is None:
        raise ValueError("--trim-resume-logs-to-checkpoint requires --resume-from")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but torch.cuda.is_available() is false")
    if device.type == "cuda":
        device_index = (
            torch.cuda.current_device() if device.index is None else device.index
        )
        if device_index < 0 or device_index >= torch.cuda.device_count():
            raise ValueError(f"CUDA device index is out of range: {device_index}")
        device = torch.device("cuda", device_index)
    elif device.type == "mps":
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is None or not mps_backend.is_available():
            raise RuntimeError("MPS device requested but the MPS backend is unavailable")
        device = torch.device("mps")
    elif device.type == "cpu":
        if device.index is not None:
            raise ValueError("CPU device must not include an index")
        device = torch.device("cpu")
    else:
        raise ValueError(f"unsupported training device: {device}")
    training_device = str(device)

    parent_checkpoint_sha256 = None
    resumed_from_sha256 = None
    global_step = 0
    stage_step = 0
    monitor = None
    resume_payload: Optional[Mapping[str, Any]] = None
    transition_payload: Optional[Mapping[str, Any]] = None
    transition_sha256: Optional[str] = None
    resume_mode = args.resume_from is not None
    if args.init_from is not None:
        loaded, transition_sha256, transition_path = _load_checkpoint_snapshot(
            args.init_from, map_location=device
        )
        if not isinstance(loaded, Mapping):
            raise ValueError("stage-transition checkpoint must contain a mapping")
        transition_payload = loaded
        if prepared.stage_index > 0:
            _validate_stage_transition_checkpoint(transition_payload, prepared)
            _validate_completed_stage_best_checkpoint(
                transition_path,
                transition_payload,
                transition_sha256,
            )
        parent_checkpoint_sha256 = transition_sha256
    elif resume_mode:
        loaded, resume_snapshot_sha256, _ = _load_checkpoint_snapshot(
            args.resume_from, map_location=device
        )
        if not isinstance(loaded, Mapping):
            raise ValueError("resume checkpoint must contain a mapping")
        resume_payload = loaded
        resumed_from_sha256 = resume_snapshot_sha256
        parent_checkpoint_sha256 = resume_payload.get("lineage", {}).get(
            "parent_checkpoint_sha256"
        )

    if resume_payload is not None:
        resume_config = resume_payload.get("config")
        if not isinstance(resume_config, Mapping):
            raise ValueError("resume checkpoint is missing config")
        run_id = _validate_run_id(
            resume_config.get("run_id"), "resume checkpoint run_id"
        )
        stage_initial_global_step = int(
            resume_payload["training_state"]["global_step"]
        ) - int(resume_payload["training_state"]["stage_step"])
    elif transition_payload is not None:
        run_id = (
            hashlib.sha256(
                (
                    protocol.content_sha256
                    + prepared.stage.name
                    + training_device
                    + "dry-run"
                ).encode("utf-8")
            ).hexdigest()[:32]
            if args.dry_run
            else uuid.uuid4().hex
        )
        stage_initial_global_step = int(
            transition_payload["training_state"]["global_step"]
        )
    elif args.dry_run:
        run_id = hashlib.sha256(
            (
                protocol.content_sha256
                + prepared.stage.name
                + training_device
                + "dry-run"
            ).encode("utf-8")
        ).hexdigest()[:32]
        stage_initial_global_step = 0
    else:
        run_id = uuid.uuid4().hex
        stage_initial_global_step = 0

    configure_reproducibility(protocol.seed, protocol.deterministic)
    train_loader, sampler, source_lengths = _build_train_loader(
        prepared, num_workers=args.num_workers
    )
    validation_loaders, validation_lengths = _build_validation_loaders(
        prepared, num_workers=args.num_workers
    )
    # Keep model initialization independent of HDF5 header traversal.
    configure_reproducibility(protocol.seed, protocol.deterministic)
    model = build_policy(protocol.model, protocol.dataset).to(device)
    parameter_groups = build_parameter_groups_from_specs(
        model,
        specs=prepared.stage.optimizer.parameter_groups,
        base_lr=prepared.stage.optimizer.base_lr,
        default_weight_decay=prepared.stage.optimizer.weight_decay,
    )
    optimizer = AdamW(parameter_groups)
    deployment_mode = resolve_validation_deployment_mode(
        policy_variant=protocol.model.policy_variant,
        requested_mode=prepared.stage.objective.validation_deployment_mode,
        train_latent_mode=prepared.stage.objective.train_latent_mode,
        lambda_prior=prepared.stage.objective.lambda_prior,
    )
    config = _checkpoint_config(
        prepared,
        parameter_groups,
        deployment_mode,
        training_device,
        run_id,
        stage_initial_global_step,
    )
    run_manifest_document = _run_manifest_document(
        prepared,
        config,
        source_lengths,
        validation_lengths,
        parent_checkpoint_sha256,
    )
    config["run_manifest_sha256"] = canonical_json_sha256(run_manifest_document)

    if transition_payload is not None:
        init_result = initialize_model_from_checkpoint(
            model,
            transition_payload,
            expected_config=config,
            compatibility_keys=INIT_COMPATIBILITY_KEYS,
            map_location=device,
        )
        global_step = init_result.source_step or 0
    elif resume_payload is not None:
        resume_result = resume_training_from_checkpoint(
            model=model,
            optimizer=optimizer,
            source=resume_payload,
            expected_config=config,
            expected_stage_name=prepared.stage.name,
            expected_stage_index=prepared.stage_index,
            expected_protocol_sha256=protocol.content_sha256,
            expected_normalization_sha256=prepared.normalization_sha256,
            compatibility_keys=RESUME_COMPATIBILITY_KEYS + ("data_provenance",),
            sampler=sampler,
            map_location=device,
            strict_cuda_rng=True,
        )
        global_step = resume_result.global_step
        stage_step = resume_result.stage_step
        monitor = _restore_monitor(prepared.stage, resume_result.monitor_state)

    dry_run_report = {
        "protocol_sha256": protocol.content_sha256,
        "normalization_sha256": prepared.normalization_sha256,
        "dataset_manifest_sha256": prepared.dataset_manifest_sha256,
        "stage": prepared.stage.name,
        "stage_index": prepared.stage_index,
        "source_dataset_samples": source_lengths,
        "validation_dataset_samples": validation_lengths,
        "batch_size": sampler.batch_size,
        "batches_per_epoch": len(sampler),
        "domain_quotas": sampler.domain_quotas,
        "phase_quotas": sampler.phase_quotas,
        "phase_episode_counts": sampler.phase_episode_counts,
        "optimizer_groups": config["optimizer_groups"],
        "training_device": training_device,
        "run_id": run_id,
        "run_manifest_sha256": config["run_manifest_sha256"],
        "init_from": str(args.init_from) if args.init_from else None,
        "resume_from": str(args.resume_from) if args.resume_from else None,
    }
    if args.dry_run:
        print(json.dumps(dry_run_report, indent=2, sort_keys=True, default=str))
        return 0

    computed_output_dir = args.output_dir or (
        REPO_ROOT / "outputs" / "staged" / protocol.run_name / prepared.stage.name
    )
    computed_output_dir = Path(computed_output_dir).expanduser().resolve()
    if locked_output_dir is not None:
        output_dir = Path(locked_output_dir).expanduser().resolve()
        if output_dir != computed_output_dir:
            raise RuntimeError(
                "locked output directory differs from the resolved training output"
            )
    else:
        output_dir = computed_output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    train_log = output_dir / "train_log.csv"
    validation_log = output_dir / "validation_log.csv"
    run_manifest_path = output_dir / "run_manifest.json"
    if resume_mode:
        if resume_payload is None or resumed_from_sha256 is None:
            raise RuntimeError("resume checkpoint state was not initialized")
        if run_manifest_path.is_symlink():
            raise ValueError(
                f"managed run manifest must not be a symlink: {run_manifest_path}"
            )
        if not run_manifest_path.is_file():
            raise FileNotFoundError(f"exact resume requires run manifest: {run_manifest_path}")
        previous_manifest = _load_json_object(
            run_manifest_path, context="resume run manifest"
        )
        expected_manifest = _run_manifest_document(
            prepared,
            config,
            source_lengths,
            validation_lengths,
            parent_checkpoint_sha256,
        )
        previous_manifest_sha256 = canonical_json_sha256(previous_manifest)
        if previous_manifest_sha256 != canonical_json_sha256(expected_manifest):
            raise ValueError(
                "run manifest does not match the resume checkpoint and current stage"
            )
        if previous_manifest_sha256 != config["run_manifest_sha256"]:
            raise ValueError("resume checkpoint run_manifest_sha256 mismatch")
        artifact_plan = _plan_resume_artifact_reconciliation(
            output_dir=output_dir,
            resume_payload=resume_payload,
            resume_checkpoint_sha256=resumed_from_sha256,
            # Plan and validate before touching logs; policy is enforced after
            # the log audit so legacy "logs ahead" diagnostics remain precise.
            allow_trim=True,
        )
        validation_domain_names = [
            validation.name for validation in prepared.stage.validation_domains
        ]
        _recover_stage_zero_logs_from_checkpoint(
            train_log=train_log,
            validation_log=validation_log,
            resume_payload=resume_payload,
            validation_domains=validation_domain_names,
        )
        _reconcile_resume_logs(
            train_log=train_log,
            validation_log=validation_log,
            stage_step=stage_step,
            global_step=global_step,
            validation_count=monitor.validation_count,
            validation_domains=validation_domain_names,
            has_initial_baseline=(
                prepared.stage.monitor.retention_domain is not None
            ),
            monitor_state=monitor.state_dict(),
            allow_trim=args.trim_resume_logs_to_checkpoint,
        )
        if artifact_plan.quarantine_dir is not None:
            if not args.trim_resume_logs_to_checkpoint:
                raise ValueError(
                    "checkpoint artifacts are ahead of or inconsistent with the "
                    "selected resume checkpoint; use the latest checkpoint or pass "
                    "--trim-resume-logs-to-checkpoint to quarantine them"
                )
            _apply_resume_artifact_plan(artifact_plan)
    else:
        for protected in (
            run_manifest_path,
            output_dir / "checkpoint.pt",
            output_dir / "checkpoint_best.pt",
            output_dir / "checkpoint_latest.pt",
            output_dir / STAGE_COMPLETION_FILENAME,
            train_log,
            validation_log,
        ):
            if protected.is_symlink():
                raise ValueError(
                    f"new training run refuses managed symlink: {protected}"
                )
            if protected.exists():
                raise FileExistsError(f"new training run refuses to overwrite: {protected}")
        stale_checkpoint_artifacts = sorted(output_dir.glob("checkpoint_*.pt"))
        if stale_checkpoint_artifacts:
            raise FileExistsError(
                "new training run refuses existing checkpoint artifact: "
                f"{stale_checkpoint_artifacts[0]}"
            )
        _write_run_manifest(
            run_manifest_path,
            prepared,
            config,
            source_lengths,
            validation_lengths,
            parent_checkpoint_sha256,
        )

    initial_validation_metrics = None
    if monitor is None:
        if prepared.stage.monitor.retention_domain is not None:
            initial_metrics = evaluate_named_deployment_metrics(
                model=model,
                dataloaders=validation_loaders,
                device=device,
                policy_variant=protocol.model.policy_variant,
                deployment_mode=deployment_mode,
                normalization_stats=prepared.normalization_stats,
                lambda_force=prepared.stage.objective.lambda_force,
                aggregation=prepared.stage.monitor.aggregation,
            )
            _validate_validation_episode_counts(initial_metrics, prepared.stage)
            monitor = _new_monitor(prepared.stage, initial_metrics=initial_metrics)
            initial_validation_metrics = initial_metrics
            _save_best_checkpoint(
                output_dir,
                model=model,
                optimizer=optimizer,
                config=config,
                prepared=prepared,
                sampler=sampler,
                monitor=monitor,
                global_step=global_step,
                stage_step=stage_step,
                parent_checkpoint_sha256=parent_checkpoint_sha256,
                stop_reason="stage_initialization_fallback",
                resumed_from_sha256=resumed_from_sha256,
            )
        else:
            monitor = _new_monitor(prepared.stage)

    resume_is_terminal = bool(
        resume_mode and _monitor_should_stop(monitor, prepared.stage)
    )
    stop_reason = "early_stopping" if resume_is_terminal else "max_steps"
    batch_iterator = iter(train_loader) if not resume_is_terminal else None
    with ExitStack() as stack:
        train_writer, train_handle = _open_csv(
            stack, train_log, TRAIN_LOG_FIELDS, resume=resume_mode
        )
        validation_writer, validation_handle = _open_csv(
            stack, validation_log, VALIDATION_LOG_FIELDS, resume=resume_mode
        )
        if initial_validation_metrics is not None:
            for domain, domain_metrics in initial_validation_metrics.items():
                validation_writer.writerow(
                    {
                        "validation_index": 0,
                        "global_step": global_step,
                        "stage_step": stage_step,
                        "epoch": 0,
                        "domain": domain,
                        "deployment_mode": deployment_mode,
                        "deploy_loss": domain_metrics.get("deploy_loss"),
                        "action_l1": domain_metrics.get("action_l1"),
                        "force_l1": domain_metrics.get("force_l1"),
                        "num_samples": domain_metrics.get("num_samples"),
                        "num_episodes": domain_metrics.get("num_episodes"),
                        "selected": True,
                        "retention_passed": True,
                        "decision_reason": "stage_initialization_baseline",
                    }
                )
            _flush_and_fsync_logs(train_handle, validation_handle)
        while not resume_is_terminal and stage_step < prepared.stage.max_steps:
            try:
                raw_batch = next(batch_iterator)
            except StopIteration:
                batch_iterator = iter(train_loader)
                raw_batch = next(batch_iterator)
            stage_step += 1
            global_step += 1
            epoch = (stage_step - 1) // len(sampler) + 1
            batch_in_epoch = (stage_step - 1) % len(sampler) + 1
            batch = move_batch_to_device(raw_batch, device)
            batch = normalize_training_batch(batch, prepared.normalization_stats)
            update = train_one_update(
                model=model,
                optimizer=optimizer,
                batch=batch,
                policy_variant=protocol.model.policy_variant,
                objective=prepared.stage.objective,
                stage_step=stage_step,
                max_grad_norm=prepared.stage.optimizer.max_grad_norm,
                freeze_vision_batch_norm=prepared.stage.freeze_vision_batch_norm,
            )
            sampler_counts = sampler.realized_quota_summary()
            train_writer.writerow(
                {
                    "global_step": global_step,
                    "stage_step": stage_step,
                    "epoch": epoch,
                    "batch_in_epoch": batch_in_epoch,
                    "loss_total": update.losses.get("loss_total", 0.0),
                    "loss_action": update.losses.get("loss_action", 0.0),
                    "loss_force": update.losses.get("loss_force", 0.0),
                    "kl_motion": update.losses.get("kl_motion", 0.0),
                    "kl_contact": update.losses.get("kl_contact", 0.0),
                    "loss_prior": update.losses.get("loss_prior", 0.0),
                    "beta_motion": update.objective.get("beta_motion", 0.0),
                    "beta_contact": update.objective.get("beta_contact", 0.0),
                    "gradient_norm": update.gradient_norm,
                    "gradient_was_clipped": update.gradient_was_clipped,
                    "sampler_domain_counts": json.dumps(
                        sampler_counts["domain"], sort_keys=True
                    ),
                    "sampler_phase_counts": json.dumps(
                        sampler_counts["phase"], sort_keys=True
                    ),
                }
            )
            train_handle.flush()

            should_validate = (
                stage_step % prepared.stage.validation_every_steps == 0
                or stage_step == prepared.stage.max_steps
            )
            should_stop = False
            if should_validate:
                metrics = evaluate_named_deployment_metrics(
                    model=model,
                    dataloaders=validation_loaders,
                    device=device,
                    policy_variant=protocol.model.policy_variant,
                    deployment_mode=deployment_mode,
                    normalization_stats=prepared.normalization_stats,
                    lambda_force=prepared.stage.objective.lambda_force,
                    aggregation=prepared.stage.monitor.aggregation,
                )
                _validate_validation_episode_counts(metrics, prepared.stage)
                selected, retention_passed, should_stop, reason = _update_monitor(
                    monitor,
                    prepared.stage,
                    metrics,
                    epoch=epoch,
                    global_step=global_step,
                )
                for domain, domain_metrics in metrics.items():
                    validation_writer.writerow(
                        {
                            "validation_index": monitor.validation_count,
                            "global_step": global_step,
                            "stage_step": stage_step,
                            "epoch": epoch,
                            "domain": domain,
                            "deployment_mode": deployment_mode,
                            "deploy_loss": domain_metrics.get("deploy_loss"),
                            "action_l1": domain_metrics.get("action_l1"),
                            "force_l1": domain_metrics.get("force_l1"),
                            "num_samples": domain_metrics.get("num_samples"),
                            "num_episodes": domain_metrics.get("num_episodes"),
                            "selected": selected,
                            "retention_passed": retention_passed,
                            "decision_reason": reason,
                        }
                    )
                _flush_and_fsync_logs(train_handle, validation_handle)
                if selected:
                    _save_best_checkpoint(
                        output_dir,
                        model=model,
                        optimizer=optimizer,
                        config=config,
                        prepared=prepared,
                        sampler=sampler,
                        monitor=monitor,
                        global_step=global_step,
                        stage_step=stage_step,
                        parent_checkpoint_sha256=parent_checkpoint_sha256,
                        stop_reason="best_validation_metric",
                        resumed_from_sha256=resumed_from_sha256,
                    )

            checkpoint_every = prepared.stage.checkpoint_every_steps
            if checkpoint_every and stage_step % checkpoint_every == 0:
                checkpoint_kwargs = dict(
                    model=model,
                    optimizer=optimizer,
                    config=config,
                    prepared=prepared,
                    sampler=sampler,
                    monitor=monitor,
                    global_step=global_step,
                    stage_step=stage_step,
                    parent_checkpoint_sha256=parent_checkpoint_sha256,
                    stop_reason="periodic_checkpoint",
                    resumed_from_sha256=resumed_from_sha256,
                )
                _save_checkpoint(
                    output_dir / f"checkpoint_step_{stage_step:08d}.pt",
                    **checkpoint_kwargs,
                )
                _save_checkpoint(output_dir / "checkpoint_latest.pt", **checkpoint_kwargs)
            if should_stop:
                stop_reason = "early_stopping"
                break
        _flush_and_fsync_logs(train_handle, validation_handle)

    final_kwargs = dict(
        model=model,
        optimizer=optimizer,
        config=config,
        prepared=prepared,
        sampler=sampler,
        monitor=monitor,
        global_step=global_step,
        stage_step=stage_step,
        parent_checkpoint_sha256=parent_checkpoint_sha256,
        stop_reason=stop_reason,
        resumed_from_sha256=resumed_from_sha256,
    )
    final_payload = _checkpoint_payload(**final_kwargs)
    save_checkpoint_atomic(final_payload, output_dir / "checkpoint.pt")
    _copy_file_atomic(
        output_dir / "checkpoint.pt",
        output_dir / "checkpoint_latest.pt",
    )
    _write_stage_completion(
        output_dir,
        final_payload,
        overwrite=resume_mode,
        minimum_validations=prepared.stage.monitor.min_validations,
    )
    print(
        json.dumps(
            {
                "stage": prepared.stage.name,
                "global_step": global_step,
                "stage_step": stage_step,
                "stop_reason": stop_reason,
                "output_dir": str(output_dir),
            },
            sort_keys=True,
        )
    )
    return 0


def train(args: argparse.Namespace) -> int:
    """Run one stage, serializing all formal writes per output directory."""

    if args.dry_run:
        return _train_impl(args)
    protocol = load_protocol(args.protocol)
    output_dir = args.output_dir or (
        REPO_ROOT / "outputs" / "staged" / protocol.run_name / args.stage
    )
    output_dir = Path(output_dir).expanduser().resolve()
    with _exclusive_training_lock(output_dir):
        return _train_impl(
            args,
            loaded_protocol=protocol,
            locked_output_dir=output_dir,
        )


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--stage", required=True)
    transition = parser.add_mutually_exclusive_group()
    transition.add_argument("--init-from", type=Path, default=None)
    transition.add_argument("--resume-from", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--trim-resume-logs-to-checkpoint",
        action="store_true",
        help=(
            "On rollback-style exact resume, atomically trim CSV rows and move "
            "newer checkpoint artifacts into a hashed quarantine. Without this "
            "explicit flag, any future state is rejected."
        ),
    )
    parser.add_argument(
        "--allow-legacy-data-contract",
        action="store_true",
        help="Allow a protocol without dataset_manifest for historical-data smoke checks.",
    )
    parser.add_argument(
        "--allow-legacy-normalization",
        action="store_true",
        help="Allow old stats without semantic content hashing; never use for formal runs.",
    )
    parser.add_argument(
        "--skip-dataset-file-verification",
        action="store_true",
        help="Skip hashing episode files against the manifest; intended only for fast dry-runs.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        return train(args)
    except (
        FileNotFoundError,
        FileExistsError,
        FloatingPointError,
        KeyError,
        TypeError,
        ValueError,
        RuntimeError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
