"""Versioned checkpoint persistence for the independent ACT-aligned trainer."""

from __future__ import annotations

import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

from force_aware_act.act_aligned_training.config import ACTAlignedTrainingConfig
from force_aware_act.act_aligned_training.normalization import NormalizationStats
from force_aware_act.act_aligned_training.split import EpisodeSplitManifest
from force_aware_act.models.act_aligned.policy import ACTAlignedContactCVAEPolicy


CHECKPOINT_FORMAT_VERSION = "act_aligned_checkpoint_v1"


@dataclass(frozen=True)
class TrainingProgress:
    epoch: int
    global_step: int
    best_metric: float


@dataclass(frozen=True)
class LoadedCheckpoint:
    progress: TrainingProgress
    normalization: NormalizationStats
    split_manifest: EpisodeSplitManifest
    dataloader_generator_state: Optional[torch.Tensor]


def save_act_aligned_checkpoint(
    path: Path,
    *,
    model: ACTAlignedContactCVAEPolicy,
    optimizer: torch.optim.Optimizer,
    training_config: ACTAlignedTrainingConfig,
    progress: TrainingProgress,
    normalization: NormalizationStats,
    split_manifest: EpisodeSplitManifest,
    dataloader_generator: Optional[torch.Generator] = None,
) -> None:
    """Atomically save a complete, resume-capable checkpoint."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "architecture_version": model.config.architecture_version,
        "training_version": training_config.training_version,
        "model_config": asdict(model.config),
        "training_config": asdict(training_config),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "progress": asdict(progress),
        "normalization": normalization.to_dict(),
        "split_manifest": split_manifest.to_dict(),
        "rng_state": _capture_rng_state(),
        "dataloader_generator_state": (
            None
            if dataloader_generator is None
            else dataloader_generator.get_state()
        ),
    }
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary_path)
    os.replace(temporary_path, path)


def load_act_aligned_checkpoint(
    path: Path,
    *,
    model: ACTAlignedContactCVAEPolicy,
    optimizer: torch.optim.Optimizer,
    training_config: ACTAlignedTrainingConfig,
    restore_rng: bool = True,
    map_location: Any = "cpu",
) -> LoadedCheckpoint:
    """Strictly load model, optimizer, progress, data contract, and RNG."""

    payload = read_act_aligned_checkpoint(path, map_location=map_location)
    if payload["model_config"] != asdict(model.config):
        raise ValueError("checkpoint model_config does not match the model")
    if payload["training_config"] != asdict(training_config):
        raise ValueError("checkpoint training_config does not match")
    model.load_state_dict(payload["model_state"], strict=True)
    optimizer.load_state_dict(payload["optimizer_state"])
    if restore_rng:
        _restore_rng_state(payload["rng_state"])
    progress = TrainingProgress(**payload["progress"])
    return LoadedCheckpoint(
        progress=progress,
        normalization=NormalizationStats.from_dict(payload["normalization"]),
        split_manifest=EpisodeSplitManifest.from_dict(
            payload["split_manifest"]
        ),
        dataloader_generator_state=payload.get("dataloader_generator_state"),
    )


def read_act_aligned_checkpoint(
    path: Path,
    *,
    map_location: Any = "cpu",
) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {path}")
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if payload.get("format_version") != CHECKPOINT_FORMAT_VERSION:
        raise ValueError("unsupported ACT-aligned checkpoint format")
    required = (
        "architecture_version",
        "training_version",
        "model_config",
        "training_config",
        "model_state",
        "optimizer_state",
        "progress",
        "normalization",
        "split_manifest",
        "rng_state",
    )
    for key in required:
        if key not in payload:
            raise KeyError(f"checkpoint is missing {key!r}")
    return payload


def _capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        ),
    }


def _restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
