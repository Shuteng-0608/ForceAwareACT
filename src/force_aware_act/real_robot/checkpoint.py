"""Strict real-robot checkpoint identity, independent from six-axis wrench runs."""

from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
import torch

from force_aware_act.models.act_aligned.config import canonical_model_config


FORMAT_VERSION = "real_robot_training_v1"


def canonical(value):
    """Normalize tuples/JSON lists before comparing a serialized data contract."""
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def sampler_states_equal(left, right):
    """Compare legacy generator tensors and versioned episodic sampler states."""
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        return isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, dict) or isinstance(right, dict):
        return (isinstance(left, dict) and isinstance(right, dict) and left.keys() == right.keys()
                and all(sampler_states_equal(left[key], right[key]) for key in left))
    if isinstance(left, (tuple, list)) or isinstance(right, (tuple, list)):
        return (type(left) is type(right) and len(left) == len(right)
                and all(sampler_states_equal(a, b) for a, b in zip(left, right)))
    return left == right


def source_fingerprint(path):
    path = Path(path)
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"source changed while hashing: {path}")
    return {"sha256": digest.hexdigest(), "bytes": after.st_size}


def read_checkpoint(path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format_version") != FORMAT_VERSION:
        raise ValueError("checkpoint is not a real_robot_training_v1 checkpoint")
    for key in ("model_kind", "model_config", "data_config", "data_contract", "normalization",
                "split_manifest", "source_fingerprints", "training_config", "run_config",
                "model_state", "optimizer_state", "global_step", "rng_state", "sampler_state"):
        if key not in payload:
            raise ValueError(f"checkpoint missing {key}")
    return payload


def save_checkpoint(path, *, model, optimizer, sampler, global_step, metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(metadata)
    payload.update(
        format_version=FORMAT_VERSION, model_state=model.state_dict(),
        optimizer_state=optimizer.state_dict(), global_step=int(global_step),
        sampler_state=sampler.get_state(),
        rng_state={"python": random.getstate(), "numpy": np.random.get_state(),
                   "torch": torch.get_rng_state(),
                   "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None},
    )
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def restore_checkpoint(payload, *, model, optimizer, sampler, expected):
    # Check everything before mutating model/optimizer or the random streams.
    for key in ("model_kind", "model_config", "data_config", "data_contract", "normalization",
                "split_manifest", "source_fingerprints", "training_config"):
        left, right = payload[key], expected[key]
        if key == "model_config":
            left, right = canonical_model_config(left), canonical_model_config(right)
        if canonical(left) != canonical(right):
            raise ValueError(f"checkpoint {key} differs from this run")
    for key in ("batch_size", "seed", "sampling", "mode"):
        if payload["run_config"][key] != expected["run_config"][key]:
            raise ValueError(f"checkpoint run_config.{key} differs from this run")
    for key in ("evaluation_config", "early_stopping_config"):
        if canonical(payload["run_config"].get(key)) != canonical(expected["run_config"].get(key)):
            raise ValueError(f"checkpoint run_config.{key} differs from this run")
    if hasattr(sampler, "validate_state"):
        sampler.validate_state(payload["sampler_state"])
    rng = payload["rng_state"]
    if rng["cuda"] is not None and torch.cuda.is_available():
        if len(rng["cuda"]) != torch.cuda.device_count():
            raise ValueError("CUDA device count differs from checkpoint RNG state")
    model.load_state_dict(payload["model_state"], strict=True)
    optimizer.load_state_dict(payload["optimizer_state"])
    sampler_state = payload["sampler_state"]
    sampler.set_state(sampler_state.cpu() if isinstance(sampler_state, torch.Tensor) else sampler_state)
    random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])
    torch.set_rng_state(rng["torch"].cpu())
    if rng["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([state.cpu() for state in rng["cuda"]])
    return int(payload["global_step"])
