"""Versioned, reproducible checkpoint utilities for staged training.

There are deliberately two different load paths:

``initialize_model_from_checkpoint``
    Loads model weights only.  It accepts legacy checkpoints and is the correct
    operation at a stage boundary, where optimizer and early-stopping state must
    be reset.

``resume_training_from_checkpoint``
    Restores an exact version-2 training state.  It rejects legacy checkpoints
    and validates model configuration, stage identity, data provenance, RNG,
    optimizer, scheduler, and sampler state before continuing the same stage.

Checkpoint files are assumed to be trusted local artifacts.  PyTorch checkpoint
loading uses pickle and must not be used on untrusted files.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import random
import struct
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Union

import numpy as np
import torch


CHECKPOINT_SCHEMA_VERSION = 2
CHECKPOINT_TYPE = "force_aware_act_training"
SHA256_HEX_LENGTH = 64

# These values change the meaning or shape of model inputs/outputs.  Runtime
# settings such as output paths, number of epochs, and learning rate are omitted
# intentionally, so a same-stage resume can extend its stopping horizon.
INIT_COMPATIBILITY_KEYS = (
    "policy_variant",
    "action_mode",
    "chunk_len",
    "force_window_len",
    "force_window_duration",
    "image_size",
    "camera_names",
    "imagenet_normalize",
    "image_alignment",
    "max_image_lag_seconds",
    "model",
)
# Backward-friendly generic name: weights-only initialization is the less
# restrictive operation. Exact resume uses its own stricter constant below.
DEFAULT_COMPATIBILITY_KEYS = INIT_COMPATIBILITY_KEYS
RESUME_COMPATIBILITY_KEYS = INIT_COMPATIBILITY_KEYS + (
    "optimizer_groups",
    "training_device",
    "freeze_vision_batch_norm",
    "run_id",
    "run_manifest_sha256",
    "stage_initial_global_step",
    "checkpoint_every_steps",
    "validation_every_steps",
    "minimum_validations",
    "training_code_sha256",
    "runtime_versions",
)

CheckpointSource = Union[Path, str, Mapping[str, Any]]


class CheckpointCompatibilityError(ValueError):
    """Raised when a checkpoint cannot safely initialize the requested run."""


class CheckpointIntegrityError(ValueError):
    """Raised when checkpoint contents do not match their recorded hashes."""


@dataclass(frozen=True)
class InitializationResult:
    """Metadata returned after a weights-only initialization."""

    schema_version: Optional[int]
    checkpoint_sha256: Optional[str]
    model_state_sha256: str
    source_step: Optional[int]
    source_stage_name: Optional[str]
    source_stage_index: Optional[int]
    compatibility_validated: bool


@dataclass(frozen=True)
class ResumeResult:
    """Restored counters and auxiliary state from an exact resume."""

    checkpoint_sha256: Optional[str]
    model_state_sha256: str
    global_step: int
    stage_step: int
    epoch: int
    step_in_epoch: int
    stage_name: str
    stage_index: int
    monitor_state: Mapping[str, Any]


def _canonicalize(value: Any) -> Any:
    """Convert configuration-like values to deterministic JSON values."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("values used for hashing must be finite")
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, np.generic):
        return _canonicalize(value.item())
    if isinstance(value, Mapping):
        canonical: Dict[str, Any] = {}
        for key in sorted(value, key=lambda item: str(item)):
            if not isinstance(key, str):
                raise TypeError("mapping keys used for hashing must be strings")
            canonical[key] = _canonicalize(value[key])
        return canonical
    if isinstance(value, (list, tuple)):
        return [_canonicalize(item) for item in value]
    raise TypeError(f"unsupported value for deterministic hashing: {type(value).__name__}")


def mapping_sha256(value: Mapping[str, Any]) -> str:
    """Return a deterministic SHA256 for a JSON-like mapping."""

    encoded = json.dumps(
        _canonicalize(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def state_dict_sha256(state_dict: Mapping[str, Any]) -> str:
    """Hash tensor values, dtypes, shapes, and keys in a model state dict."""

    digest = hashlib.sha256()
    for key in sorted(state_dict):
        if not isinstance(key, str):
            raise TypeError("state_dict keys must be strings")
        value = state_dict[key]
        if not torch.is_tensor(value):
            raise TypeError(
                f"state_dict value {key!r} must be a tensor, got {type(value).__name__}"
            )
        tensor = value.detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(b"\0")
        # Viewing as bytes supports dtypes (notably bfloat16) that NumPy cannot
        # represent directly.
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes(order="C"))
        digest.update(b"\0")
    return digest.hexdigest()


def structured_state_sha256(value: Any) -> str:
    """Hash nested resume state, including tensors and non-string mapping keys."""

    digest = hashlib.sha256()

    def write(data: bytes) -> None:
        digest.update(struct.pack(">Q", len(data)))
        digest.update(data)

    def visit(item: Any) -> None:
        if item is None:
            write(b"none")
        elif isinstance(item, bool):
            write(b"bool")
            write(b"1" if item else b"0")
        elif isinstance(item, int):
            write(b"int")
            write(str(item).encode("ascii"))
        elif isinstance(item, float):
            if not math.isfinite(item):
                raise ValueError("structured state contains a non-finite float")
            write(b"float64")
            write(struct.pack(">d", item))
        elif isinstance(item, str):
            write(b"str")
            write(item.encode("utf-8"))
        elif isinstance(item, Path):
            write(b"path")
            write(str(item).encode("utf-8"))
        elif isinstance(item, torch.device):
            write(b"device")
            write(str(item).encode("ascii"))
        elif isinstance(item, np.generic):
            visit(item.item())
        elif torch.is_tensor(item):
            tensor = item.detach().cpu().contiguous()
            write(b"tensor")
            write(str(tensor.dtype).encode("ascii"))
            write(str(tuple(tensor.shape)).encode("ascii"))
            write(tensor.reshape(-1).view(torch.uint8).numpy().tobytes(order="C"))
        elif isinstance(item, np.ndarray):
            if item.dtype.hasobject:
                raise TypeError("structured state object arrays are unsupported")
            array = np.ascontiguousarray(item)
            write(b"ndarray")
            write(array.dtype.str.encode("ascii"))
            write(str(tuple(array.shape)).encode("ascii"))
            write(array.tobytes(order="C"))
        elif isinstance(item, Mapping):
            write(b"mapping")
            ordered_items = sorted(
                item.items(), key=lambda pair: (type(pair[0]).__name__, repr(pair[0]))
            )
            write(str(len(ordered_items)).encode("ascii"))
            for key, child in ordered_items:
                if not isinstance(key, (str, int, float, bool)):
                    raise TypeError(
                        "structured state mapping keys must be scalar strings or numbers"
                    )
                visit(key)
                visit(child)
        elif isinstance(item, (list, tuple)):
            write(b"tuple" if isinstance(item, tuple) else b"list")
            write(str(len(item)).encode("ascii"))
            for child in item:
                visit(child)
        else:
            raise TypeError(
                f"unsupported structured state value: {type(item).__name__}"
            )

    visit(value)
    return digest.hexdigest()


def file_sha256(path: Union[Path, str], chunk_size: int = 1024 * 1024) -> str:
    """Stream a file and return its SHA256 without loading it into memory."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _validate_sha256(value: str, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a 64-character SHA256 hex digest")
    normalized = value.lower()
    if len(normalized) != SHA256_HEX_LENGTH or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError(f"{field_name} must be a 64-character SHA256 hex digest")
    return normalized


def capture_rng_state(
    named_generators: Optional[Mapping[str, torch.Generator]] = None,
) -> Dict[str, Any]:
    """Capture Python, NumPy, Torch CPU/CUDA, and named generator RNG states."""

    generator_states: Dict[str, torch.Tensor] = {}
    for name, generator in sorted((named_generators or {}).items()):
        if not isinstance(name, str) or not name:
            raise ValueError("named generator keys must be non-empty strings")
        if not isinstance(generator, torch.Generator):
            raise TypeError(f"named generator {name!r} must be torch.Generator")
        generator_states[name] = generator.get_state().clone().cpu()

    cuda_states: Optional[list[torch.Tensor]]
    if torch.cuda.is_available():
        cuda_states = [state.clone().cpu() for state in torch.cuda.get_rng_state_all()]
    else:
        cuda_states = None

    numpy_state = np.random.get_state()
    serialized_numpy_state = {
        "bit_generator": numpy_state[0],
        # A tensor keeps the v2 payload compatible with torch.load's restricted
        # weights-only unpickler; raw NumPy arrays are not allow-listed there.
        "keys": torch.from_numpy(numpy_state[1].astype(np.int64, copy=True)),
        "position": int(numpy_state[2]),
        "has_gauss": int(numpy_state[3]),
        "cached_gaussian": float(numpy_state[4]),
    }
    payload = {
        "python": random.getstate(),
        "numpy": serialized_numpy_state,
        "torch_cpu": torch.get_rng_state().clone().cpu(),
        "torch_cuda": cuda_states,
        "named_generators": generator_states,
    }
    return payload


def restore_rng_state(
    rng_state: Mapping[str, Any],
    named_generators: Optional[Mapping[str, torch.Generator]] = None,
    *,
    strict_cuda: bool = True,
) -> None:
    """Restore RNG state, rejecting incomplete state needed for exact replay."""

    required = {"python", "numpy", "torch_cpu", "torch_cuda", "named_generators"}
    missing = sorted(required - set(rng_state))
    if missing:
        raise CheckpointCompatibilityError(
            "RNG state is incomplete; missing keys: " + ", ".join(missing)
        )
    torch_cpu = rng_state["torch_cpu"]
    if not torch.is_tensor(torch_cpu):
        raise CheckpointCompatibilityError("RNG torch_cpu state must be a tensor")
    numpy_state = rng_state["numpy"]
    if not isinstance(numpy_state, Mapping):
        raise CheckpointCompatibilityError("RNG numpy state must be a mapping")
    required_numpy = {
        "bit_generator",
        "keys",
        "position",
        "has_gauss",
        "cached_gaussian",
    }
    missing_numpy = sorted(required_numpy - set(numpy_state))
    if missing_numpy:
        raise CheckpointCompatibilityError(
            "RNG numpy state is incomplete; missing keys: " + ", ".join(missing_numpy)
        )
    numpy_keys = numpy_state["keys"]
    if not torch.is_tensor(numpy_keys) or numpy_keys.ndim != 1:
        raise CheckpointCompatibilityError("RNG numpy keys must be a one-dimensional tensor")

    saved_named = rng_state["named_generators"]
    if not isinstance(saved_named, Mapping):
        raise CheckpointCompatibilityError("RNG named_generators state must be a mapping")
    current_named = named_generators or {}
    if set(saved_named) != set(current_named):
        raise CheckpointCompatibilityError(
            "named generator mismatch: "
            f"checkpoint={sorted(saved_named)} current={sorted(current_named)}"
        )
    for name, generator in current_named.items():
        state = saved_named[name]
        if not torch.is_tensor(state):
            raise CheckpointCompatibilityError(
                f"RNG state for named generator {name!r} must be a tensor"
            )

    saved_cuda = rng_state["torch_cuda"]
    cuda_available = torch.cuda.is_available()
    if strict_cuda:
        if saved_cuda is None and cuda_available:
            raise CheckpointCompatibilityError(
                "checkpoint has no CUDA RNG state but CUDA is active"
            )
        if saved_cuda is not None and not cuda_available:
            raise CheckpointCompatibilityError(
                "checkpoint has CUDA RNG state but CUDA is unavailable"
            )
        if saved_cuda is not None and len(saved_cuda) != torch.cuda.device_count():
            raise CheckpointCompatibilityError(
                "CUDA device count differs from the checkpoint RNG state"
            )

    # Validation above intentionally completes before mutating any RNG.
    random.setstate(rng_state["python"])
    np.random.set_state(
        (
            numpy_state["bit_generator"],
            numpy_keys.detach().cpu().numpy().astype(np.uint32, copy=False),
            int(numpy_state["position"]),
            int(numpy_state["has_gauss"]),
            float(numpy_state["cached_gaussian"]),
        )
    )
    torch.set_rng_state(torch_cpu.cpu())
    if saved_cuda is not None and cuda_available:
        torch.cuda.set_rng_state_all([state.cpu() for state in saved_cuda])
    for name, generator in current_named.items():
        generator.set_state(saved_named[name].cpu())


def validate_checkpoint_compatibility(
    saved_config: Mapping[str, Any],
    current_config: Mapping[str, Any],
    *,
    keys: Sequence[str] = DEFAULT_COMPATIBILITY_KEYS,
    require_keys: bool = True,
) -> None:
    """Validate architecture and input-semantics fields across two configs."""

    if not keys:
        raise ValueError("compatibility keys must not be empty")
    duplicate_keys = sorted({key for key in keys if keys.count(key) > 1})
    if duplicate_keys:
        raise ValueError("duplicate compatibility keys: " + ", ".join(duplicate_keys))

    errors = []
    for key in keys:
        saved_present = key in saved_config
        current_present = key in current_config
        if require_keys and (not saved_present or not current_present):
            missing_from = []
            if not saved_present:
                missing_from.append("checkpoint")
            if not current_present:
                missing_from.append("current config")
            errors.append(f"{key}: missing from {' and '.join(missing_from)}")
            continue
        if not saved_present and not current_present:
            continue
        if saved_present != current_present:
            errors.append(f"{key}: present in only one config")
            continue
        if _canonicalize(saved_config[key]) != _canonicalize(current_config[key]):
            errors.append(
                f"{key}: checkpoint={saved_config[key]!r} current={current_config[key]!r}"
            )
    if errors:
        raise CheckpointCompatibilityError(
            "checkpoint configuration is incompatible: " + "; ".join(errors)
        )


def _state_from_object(stateful: Optional[Any], field_name: str) -> Optional[Mapping[str, Any]]:
    if stateful is None:
        return None
    state_dict = getattr(stateful, "state_dict", None)
    if not callable(state_dict):
        raise TypeError(f"{field_name} must define state_dict()")
    state = state_dict()
    if not isinstance(state, Mapping):
        raise TypeError(f"{field_name}.state_dict() must return a mapping")
    return copy.deepcopy(dict(state))


def build_checkpoint_v2(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    config: Mapping[str, Any],
    global_step: int,
    stage_step: int,
    epoch: int,
    step_in_epoch: int,
    stage_name: str,
    stage_index: int,
    protocol_sha256: str,
    normalization_sha256: str,
    scheduler: Optional[Any] = None,
    sampler: Optional[Any] = None,
    monitor_state: Optional[Mapping[str, Any]] = None,
    named_generators: Optional[Mapping[str, torch.Generator]] = None,
    parent_checkpoint_sha256: Optional[str] = None,
    resumed_from_checkpoint_sha256: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a self-describing version-2 checkpoint payload.

    The returned payload retains the legacy top-level ``model_state_dict``,
    ``optimizer_state_dict``, ``config``, and ``step`` fields so existing
    evaluation code can continue to read new checkpoints.
    """

    for name, value in (
        ("global_step", global_step),
        ("stage_step", stage_step),
        ("epoch", epoch),
        ("step_in_epoch", step_in_epoch),
        ("stage_index", stage_index),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    if not isinstance(stage_name, str) or not stage_name.strip():
        raise ValueError("stage_name must be a non-empty string")
    if not isinstance(config, Mapping):
        raise TypeError("config must be a mapping")
    protocol_sha256 = _validate_sha256(protocol_sha256, "protocol_sha256")
    normalization_sha256 = _validate_sha256(
        normalization_sha256, "normalization_sha256"
    )
    if parent_checkpoint_sha256 is not None:
        parent_checkpoint_sha256 = _validate_sha256(
            parent_checkpoint_sha256, "parent_checkpoint_sha256"
        )
    if resumed_from_checkpoint_sha256 is not None:
        resumed_from_checkpoint_sha256 = _validate_sha256(
            resumed_from_checkpoint_sha256,
            "resumed_from_checkpoint_sha256",
        )

    model_state = model.state_dict()
    optimizer_state = optimizer.state_dict()
    copied_config = copy.deepcopy(dict(config))
    model_hash = state_dict_sha256(model_state)
    config_hash = mapping_sha256(copied_config)
    scheduler_state = _state_from_object(scheduler, "scheduler")
    sampler_state = _state_from_object(sampler, "sampler")

    payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "checkpoint_type": CHECKPOINT_TYPE,
        # Backward-compatible top-level fields.
        "model_state_dict": model_state,
        "optimizer_state_dict": optimizer_state,
        "config": copied_config,
        "step": global_step,
        "epoch": epoch,
        "step_in_epoch": step_in_epoch,
        # Version-2 training state.
        "scheduler_state_dict": scheduler_state,
        "sampler_state_dict": sampler_state,
        "monitor_state": copy.deepcopy(dict(monitor_state or {})),
        "training_state": {
            "global_step": global_step,
            "stage_step": stage_step,
            "epoch": epoch,
            "step_in_epoch": step_in_epoch,
        },
        "stage": {"name": stage_name, "index": stage_index},
        "lineage": {
            "parent_checkpoint_sha256": parent_checkpoint_sha256,
            "resumed_from_checkpoint_sha256": resumed_from_checkpoint_sha256,
        },
        "rng_state": capture_rng_state(named_generators),
        "integrity": {
            "model_state_sha256": model_hash,
            "config_sha256": config_hash,
            "protocol_sha256": protocol_sha256,
            "normalization_sha256": normalization_sha256,
        },
    }
    auxiliary_state = {
        key: payload[key]
        for key in (
            "optimizer_state_dict",
            "scheduler_state_dict",
            "sampler_state_dict",
            "monitor_state",
            "training_state",
            "stage",
            "lineage",
            "rng_state",
        )
    }
    payload["integrity"]["auxiliary_state_sha256"] = structured_state_sha256(
        auxiliary_state
    )
    return payload


def _require_mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise CheckpointCompatibilityError(f"checkpoint {key!r} must be a mapping")
    return value


def validate_checkpoint_v2_payload(
    payload: Mapping[str, Any], *, verify_hashes: bool = True
) -> None:
    """Validate required version-2 structure and recorded integrity hashes."""

    if payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise CheckpointCompatibilityError(
            f"exact resume requires checkpoint schema version {CHECKPOINT_SCHEMA_VERSION}"
        )
    if payload.get("checkpoint_type") != CHECKPOINT_TYPE:
        raise CheckpointCompatibilityError(
            f"unexpected checkpoint_type: {payload.get('checkpoint_type')!r}"
        )
    for key in ("model_state_dict", "optimizer_state_dict", "config", "training_state"):
        _require_mapping(payload, key)
    stage = _require_mapping(payload, "stage")
    lineage = _require_mapping(payload, "lineage")
    integrity = _require_mapping(payload, "integrity")
    _require_mapping(payload, "rng_state")
    _require_mapping(payload, "monitor_state")

    if not isinstance(stage.get("name"), str) or not stage["name"]:
        raise CheckpointCompatibilityError("checkpoint stage name must be non-empty")
    if not isinstance(stage.get("index"), int) or isinstance(stage["index"], bool):
        raise CheckpointCompatibilityError("checkpoint stage index must be an integer")
    if stage["index"] < 0:
        raise CheckpointCompatibilityError("checkpoint stage index must be non-negative")

    training_state = payload["training_state"]
    for key in ("global_step", "stage_step", "epoch", "step_in_epoch"):
        value = training_state.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise CheckpointCompatibilityError(
                f"checkpoint training_state.{key} must be a non-negative integer"
            )
    if payload.get("step") != training_state["global_step"]:
        raise CheckpointCompatibilityError(
            "legacy step field disagrees with training_state.global_step"
        )
    if payload.get("epoch") != training_state["epoch"]:
        raise CheckpointCompatibilityError(
            "legacy epoch field disagrees with training_state.epoch"
        )
    if payload.get("step_in_epoch") != training_state["step_in_epoch"]:
        raise CheckpointCompatibilityError(
            "legacy step_in_epoch field disagrees with training_state.step_in_epoch"
        )
    if training_state["stage_step"] > training_state["global_step"]:
        raise CheckpointCompatibilityError(
            "training_state.stage_step cannot exceed training_state.global_step"
        )

    for key in ("scheduler_state_dict", "sampler_state_dict"):
        if key not in payload:
            raise CheckpointCompatibilityError(f"checkpoint is missing {key!r}")
        value = payload[key]
        if value is not None and not isinstance(value, Mapping):
            raise CheckpointCompatibilityError(
                f"checkpoint {key!r} must be a mapping or None"
            )

    for key in (
        "model_state_sha256",
        "config_sha256",
        "protocol_sha256",
        "normalization_sha256",
        "auxiliary_state_sha256",
    ):
        value = integrity.get(key)
        if not isinstance(value, str):
            raise CheckpointCompatibilityError(f"checkpoint integrity.{key} is missing")
        _validate_sha256(value, f"integrity.{key}")
    parent_hash = lineage.get("parent_checkpoint_sha256")
    if parent_hash is not None:
        if not isinstance(parent_hash, str):
            raise CheckpointCompatibilityError(
                "lineage.parent_checkpoint_sha256 must be a string or None"
            )
        _validate_sha256(parent_hash, "lineage.parent_checkpoint_sha256")
    if "resumed_from_checkpoint_sha256" not in lineage:
        raise CheckpointCompatibilityError(
            "lineage.resumed_from_checkpoint_sha256 is missing"
        )
    resumed_hash = lineage.get("resumed_from_checkpoint_sha256")
    if resumed_hash is not None:
        if not isinstance(resumed_hash, str):
            raise CheckpointCompatibilityError(
                "lineage.resumed_from_checkpoint_sha256 must be a string or None"
            )
        _validate_sha256(
            resumed_hash,
            "lineage.resumed_from_checkpoint_sha256",
        )

    if verify_hashes:
        actual_model_hash = state_dict_sha256(payload["model_state_dict"])
        if actual_model_hash != integrity["model_state_sha256"]:
            raise CheckpointIntegrityError(
                "model_state_dict does not match integrity.model_state_sha256"
            )
        actual_config_hash = mapping_sha256(payload["config"])
        if actual_config_hash != integrity["config_sha256"]:
            raise CheckpointIntegrityError(
                "config does not match integrity.config_sha256"
            )
        auxiliary_state = {
            key: payload[key]
            for key in (
                "optimizer_state_dict",
                "scheduler_state_dict",
                "sampler_state_dict",
                "monitor_state",
                "training_state",
                "stage",
                "lineage",
                "rng_state",
            )
        }
        actual_auxiliary_hash = structured_state_sha256(auxiliary_state)
        if actual_auxiliary_hash != integrity["auxiliary_state_sha256"]:
            raise CheckpointIntegrityError(
                "resume auxiliary state does not match "
                "integrity.auxiliary_state_sha256"
            )


def save_checkpoint_atomic(payload: Mapping[str, Any], checkpoint_path: Union[Path, str]) -> str:
    """Atomically save a checkpoint in its destination directory.

    Returns the SHA256 of the completed file, suitable for recording as a child
    stage's ``parent_checkpoint_sha256``.
    """

    if not isinstance(payload, Mapping):
        raise TypeError("checkpoint payload must be a mapping")
    if payload.get("schema_version") == CHECKPOINT_SCHEMA_VERSION:
        validate_checkpoint_v2_payload(payload)

    destination = Path(checkpoint_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=str(destination.parent),
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        torch.save(dict(payload), temporary_path)
        with temporary_path.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(str(temporary_path), str(destination))
        # Persist the rename itself when the filesystem supports directory fsync.
        try:
            directory_fd = os.open(str(destination.parent), os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            pass
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    return file_sha256(destination)


def _torch_load(path: Path, map_location: Any) -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        # Compatibility with PyTorch versions predating the weights_only kwarg.
        return torch.load(path, map_location=map_location)


def _load_source(
    source: CheckpointSource, map_location: Any
) -> tuple[Mapping[str, Any], Optional[str]]:
    if isinstance(source, Mapping):
        return source, None
    path = Path(source)
    loaded = _torch_load(path, map_location)
    if not isinstance(loaded, Mapping):
        raise CheckpointCompatibilityError("checkpoint file must contain a mapping")
    return loaded, file_sha256(path)


def _looks_like_raw_state_dict(payload: Mapping[str, Any]) -> bool:
    return bool(payload) and all(
        isinstance(key, str) and torch.is_tensor(value)
        for key, value in payload.items()
    )


def _extract_model_state(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    if "model_state_dict" in payload:
        state = payload["model_state_dict"]
    elif "state_dict" in payload:
        state = payload["state_dict"]
    elif _looks_like_raw_state_dict(payload):
        state = payload
    else:
        raise CheckpointCompatibilityError(
            "checkpoint has no model_state_dict (or legacy state_dict)"
        )
    if not isinstance(state, Mapping):
        raise CheckpointCompatibilityError("checkpoint model state must be a mapping")
    return state


def initialize_model_from_checkpoint(
    model: torch.nn.Module,
    source: CheckpointSource,
    *,
    expected_config: Optional[Mapping[str, Any]] = None,
    compatibility_keys: Sequence[str] = INIT_COMPATIBILITY_KEYS,
    map_location: Any = "cpu",
    strict_model: bool = True,
) -> InitializationResult:
    """Initialize model weights while intentionally resetting training state."""

    payload, checkpoint_hash = _load_source(source, map_location)
    schema_version = payload.get("schema_version")
    if schema_version == CHECKPOINT_SCHEMA_VERSION:
        validate_checkpoint_v2_payload(payload)
    elif schema_version is not None and not _looks_like_raw_state_dict(payload):
        raise CheckpointCompatibilityError(
            f"unsupported checkpoint schema_version={schema_version!r}"
        )
    state = _extract_model_state(payload)

    compatibility_validated = False
    saved_config = payload.get("config")
    if expected_config is not None and isinstance(saved_config, Mapping):
        validate_checkpoint_compatibility(
            saved_config,
            expected_config,
            keys=compatibility_keys,
            require_keys=True,
        )
        compatibility_validated = True

    model.load_state_dict(state, strict=strict_model)
    model_hash = state_dict_sha256(model.state_dict())
    if schema_version == CHECKPOINT_SCHEMA_VERSION:
        recorded_hash = payload["integrity"]["model_state_sha256"]
        if model_hash != recorded_hash:
            raise CheckpointIntegrityError(
                "loaded model hash differs from the version-2 checkpoint"
            )

    stage = payload.get("stage")
    return InitializationResult(
        schema_version=schema_version if isinstance(schema_version, int) else None,
        checkpoint_sha256=checkpoint_hash,
        model_state_sha256=model_hash,
        source_step=payload.get("step") if isinstance(payload.get("step"), int) else None,
        source_stage_name=(
            stage.get("name") if isinstance(stage, Mapping) else None
        ),
        source_stage_index=(
            stage.get("index") if isinstance(stage, Mapping) else None
        ),
        compatibility_validated=compatibility_validated,
    )


def _validate_resume_provenance(
    payload: Mapping[str, Any],
    *,
    expected_stage_name: str,
    expected_stage_index: int,
    expected_protocol_sha256: str,
    expected_normalization_sha256: str,
) -> None:
    if not isinstance(expected_stage_name, str) or not expected_stage_name:
        raise ValueError("expected_stage_name must be a non-empty string")
    if (
        not isinstance(expected_stage_index, int)
        or isinstance(expected_stage_index, bool)
        or expected_stage_index < 0
    ):
        raise ValueError("expected_stage_index must be a non-negative integer")
    stage = payload["stage"]
    if stage["name"] != expected_stage_name or stage["index"] != expected_stage_index:
        raise CheckpointCompatibilityError(
            "resume checkpoint belongs to a different stage: "
            f"checkpoint=({stage['name']!r}, {stage['index']}) "
            f"current=({expected_stage_name!r}, {expected_stage_index})"
        )
    expected_protocol_sha256 = _validate_sha256(
        expected_protocol_sha256, "expected_protocol_sha256"
    )
    expected_normalization_sha256 = _validate_sha256(
        expected_normalization_sha256, "expected_normalization_sha256"
    )
    integrity = payload["integrity"]
    if integrity["protocol_sha256"] != expected_protocol_sha256:
        raise CheckpointCompatibilityError("training protocol SHA256 mismatch")
    if integrity["normalization_sha256"] != expected_normalization_sha256:
        raise CheckpointCompatibilityError("normalization SHA256 mismatch")


def _validate_optional_state_pair(
    payload: Mapping[str, Any], key: str, stateful: Optional[Any]
) -> None:
    saved = payload.get(key)
    if (saved is None) != (stateful is None):
        raise CheckpointCompatibilityError(
            f"exact resume requires matching {key}: "
            f"checkpoint_present={saved is not None} current_present={stateful is not None}"
        )
    if saved is not None:
        if not isinstance(saved, Mapping):
            raise CheckpointCompatibilityError(f"checkpoint {key} must be a mapping or None")
        loader = getattr(stateful, "load_state_dict", None)
        if not callable(loader):
            raise TypeError(f"object for {key} must define load_state_dict()")


def resume_training_from_checkpoint(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    source: CheckpointSource,
    expected_config: Mapping[str, Any],
    expected_stage_name: str,
    expected_stage_index: int,
    expected_protocol_sha256: str,
    expected_normalization_sha256: str,
    compatibility_keys: Sequence[str] = RESUME_COMPATIBILITY_KEYS,
    scheduler: Optional[Any] = None,
    sampler: Optional[Any] = None,
    named_generators: Optional[Mapping[str, torch.Generator]] = None,
    map_location: Any = "cpu",
    strict_cuda_rng: bool = True,
) -> ResumeResult:
    """Strictly restore the same stage; legacy/partial checkpoints are rejected."""

    payload, checkpoint_hash = _load_source(source, map_location)
    validate_checkpoint_v2_payload(payload)
    validate_checkpoint_compatibility(
        payload["config"],
        expected_config,
        keys=compatibility_keys,
        require_keys=True,
    )
    _validate_resume_provenance(
        payload,
        expected_stage_name=expected_stage_name,
        expected_stage_index=expected_stage_index,
        expected_protocol_sha256=expected_protocol_sha256,
        expected_normalization_sha256=expected_normalization_sha256,
    )
    _validate_optional_state_pair(payload, "scheduler_state_dict", scheduler)
    _validate_optional_state_pair(payload, "sampler_state_dict", sampler)

    # RNG validation occurs before model/optimizer mutation.  Capture current
    # state so validation can be performed without consuming or changing it.
    current_rng = capture_rng_state(named_generators)
    try:
        restore_rng_state(
            payload["rng_state"],
            named_generators,
            strict_cuda=strict_cuda_rng,
        )
    finally:
        restore_rng_state(current_rng, named_generators, strict_cuda=strict_cuda_rng)

    model.load_state_dict(payload["model_state_dict"], strict=True)
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    if scheduler is not None:
        scheduler.load_state_dict(payload["scheduler_state_dict"])
    if sampler is not None:
        sampler.load_state_dict(payload["sampler_state_dict"])
    restore_rng_state(
        payload["rng_state"],
        named_generators,
        strict_cuda=strict_cuda_rng,
    )

    training_state = payload["training_state"]
    return ResumeResult(
        checkpoint_sha256=checkpoint_hash,
        model_state_sha256=payload["integrity"]["model_state_sha256"],
        global_step=training_state["global_step"],
        stage_step=training_state["stage_step"],
        epoch=training_state["epoch"],
        step_in_epoch=training_state["step_in_epoch"],
        stage_name=payload["stage"]["name"],
        stage_index=payload["stage"]["index"],
        monitor_state=copy.deepcopy(dict(payload["monitor_state"])),
    )
