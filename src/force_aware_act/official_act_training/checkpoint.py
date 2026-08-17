"""Versioned, atomic official ACT checkpoint persistence."""

from __future__ import annotations

import os
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np
import torch

from force_aware_act.models.official_act import OfficialACTPolicy
from force_aware_act.official_act_training.config import (
    OfficialACTTrainingConfig,
)
from force_aware_act.official_act_training.data import (
    OfficialACTNormalizationStats,
    OfficialACTSplitManifest,
)
from force_aware_act.official_act_training.optimizer import (
    build_official_act_optimizer,
)


OFFICIAL_ACT_CHECKPOINT_VERSION = "official_act_checkpoint_v1"


def save_official_act_checkpoint(
    path: Path,
    *,
    model: OfficialACTPolicy,
    optimizer: torch.optim.Optimizer,
    training_config: OfficialACTTrainingConfig,
    normalization: OfficialACTNormalizationStats,
    split_manifest: OfficialACTSplitManifest,
    epoch: int,
    global_step: int,
    best_metric: float,
    best_epoch: int = -1,
    best_model_state: Any = None,
    experiment_manifest: Optional[Mapping[str, Any]] = None,
    run_control: Optional[Mapping[str, Any]] = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": OFFICIAL_ACT_CHECKPOINT_VERSION,
        "architecture_version": model.config.architecture_version,
        "training_version": training_config.training_version,
        "model_config": asdict(model.config),
        "training_config": asdict(training_config),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "normalization": normalization.to_dict(),
        "split_manifest": split_manifest.to_dict(),
        "experiment_manifest": (
            None
            if experiment_manifest is None
            else dict(experiment_manifest)
        ),
        "run_control": None if run_control is None else dict(run_control),
        "progress": {
            "epoch": int(epoch),
            "global_step": int(global_step),
            "best_metric": float(best_metric),
            "best_epoch": int(best_epoch),
        },
        "best_model_state": best_model_state,
        "rng_state": _capture_rng(),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def read_official_act_checkpoint(
    path: Path,
    *,
    map_location: Any = "cpu",
) -> dict[str, Any]:
    payload = torch.load(
        Path(path),
        map_location=map_location,
        weights_only=False,
    )
    if payload.get("format_version") != OFFICIAL_ACT_CHECKPOINT_VERSION:
        raise ValueError("unsupported official ACT checkpoint")
    return payload


def load_official_act_checkpoint(
    path: Path,
    *,
    model: OfficialACTPolicy,
    optimizer: torch.optim.Optimizer,
    training_config: OfficialACTTrainingConfig,
    map_location: Any,
) -> dict[str, Any]:
    payload = read_official_act_checkpoint(path, map_location=map_location)
    if payload["model_config"] != asdict(model.config):
        raise ValueError("checkpoint model config mismatch")
    if payload["training_config"] != asdict(training_config):
        raise ValueError("checkpoint training config mismatch")
    model.load_state_dict(payload["model_state"], strict=True)
    optimizer.load_state_dict(payload["optimizer_state"])
    _restore_rng(payload["rng_state"])
    return payload


def construct_official_act_from_checkpoint(
    path: Path,
    *,
    map_location: Any,
) -> tuple[
    OfficialACTPolicy,
    torch.optim.Optimizer,
    OfficialACTTrainingConfig,
    dict[str, Any],
]:
    from force_aware_act.models.official_act import OfficialACTConfig

    payload = read_official_act_checkpoint(path, map_location="cpu")
    model = OfficialACTPolicy(
        OfficialACTConfig(**payload["model_config"])
    ).to(map_location)
    config = OfficialACTTrainingConfig(**payload["training_config"])
    optimizer = build_official_act_optimizer(model, config)
    loaded = load_official_act_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        training_config=config,
        map_location=map_location,
    )
    return model, optimizer, config, loaded


def _capture_rng() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": (
            torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else None
        ),
    }


def _restore_rng(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(_cpu_rng_state(state["torch"]))
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(
            [_cpu_rng_state(value) for value in state["cuda"]]
        )


def _cpu_rng_state(value: torch.Tensor) -> torch.Tensor:
    """Return the CPU ByteTensor required by PyTorch RNG restoration.

    A checkpoint loaded with ``map_location='cuda'`` also maps saved CPU RNG
    tensors to CUDA. Both ``torch.set_rng_state`` and
    ``torch.cuda.set_rng_state_all`` require CPU uint8 state tensors.
    """

    if not isinstance(value, torch.Tensor):
        raise TypeError("RNG state must be a torch.Tensor")
    state = value.detach().cpu()
    if state.dtype is not torch.uint8:
        raise TypeError("RNG state must have dtype torch.uint8")
    return state
