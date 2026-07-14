#!/usr/bin/env python3
"""Minimal ForceAwareACT training script.

Example:
    PYTHONPATH=src .venv/bin/python scripts/train_minimal.py test_data/episode.hdf5 --max-steps 20
    PYTHONPATH=src .venv/bin/python scripts/train_minimal.py test_data/episode.hdf5 --log-csv outputs/minimal_train/train_log.csv
    PYTHONPATH=src .venv/bin/python scripts/train_minimal.py test_data/episode.hdf5 --lambda-prior 0.01
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import random
import sys
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from force_aware_act.data import ContactForceHDF5Dataset, normalize_tensor  # noqa: E402
from force_aware_act.models import (  # noqa: E402
    ForceAwareACTContactCVAEPolicy,
    ForceAwareACTMotionCVAEPolicy,
    ForceAwareACTPolicy,
)
from force_aware_act.training import (  # noqa: E402
    EARLY_STOP_METRICS,
    VALIDATION_DEPLOYMENT_MODES,
    EarlyStoppingState,
    compute_force_aware_act_loss,
    compute_force_aware_contact_cvae_loss,
    compute_force_aware_motion_cvae_loss,
    compute_steps_per_epoch,
    evaluate_deployment_metrics,
    linear_warmup,
    resolve_validation_deployment_mode,
    validate_disjoint_episode_splits,
    validate_normalization_training_episodes,
)
from force_aware_act.utils import resolve_episode_paths, validate_episode_paths  # noqa: E402


ACTION_MODE_CHOICES = (
    "joint_pos",
    "action",
    "joint_pos_command",
    "delta_joint_cmd",
    "delta_joint_pos_command",
)
TRAIN_LATENT_MODE_CHOICES = ("posterior", "zero")
TRAIN_CONTACT_LATENT_MODE_CHOICES = ("posterior",)
POLICY_VARIANT_CHOICES = (
    "force_aware_act",
    "force_aware_motion_cvae",
    "force_aware_contact_cvae",
)
DEFAULT_POLICY_VARIANT = "force_aware_act"
DEFAULT_TRAINING_SEED = 0
DATALOADER_SEED_OFFSET = 1


def positive_int(value: str) -> int:
    """Parse a strictly positive integer for argparse."""

    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def configure_cpu_threads(
    torch_num_threads: Optional[int] = None,
    torch_num_interop_threads: Optional[int] = None,
) -> tuple[int, int]:
    """Apply optional PyTorch CPU thread limits and return resolved values."""

    if torch_num_threads is not None and torch_num_threads <= 0:
        raise ValueError("torch_num_threads must be a positive integer")
    if torch_num_interop_threads is not None and torch_num_interop_threads <= 0:
        raise ValueError("torch_num_interop_threads must be a positive integer")

    # Inter-op configuration must happen before inter-op parallel work begins.
    if torch_num_interop_threads is not None:
        torch.set_num_interop_threads(torch_num_interop_threads)
    if torch_num_threads is not None:
        torch.set_num_threads(torch_num_threads)
    return torch.get_num_threads(), torch.get_num_interop_threads()


def _configure_cpu_threads_from_args(args: argparse.Namespace) -> tuple[int, int]:
    if getattr(args, "_cpu_threads_configured", False):
        return args.resolved_torch_num_threads, args.resolved_torch_num_interop_threads

    resolved = configure_cpu_threads(
        getattr(args, "torch_num_threads", None),
        getattr(args, "torch_num_interop_threads", None),
    )
    args.resolved_torch_num_threads = resolved[0]
    args.resolved_torch_num_interop_threads = resolved[1]
    args._cpu_threads_configured = True
    return resolved


def configure_reproducibility(seed: int, deterministic: bool = False) -> None:
    """Configure process RNGs and optional deterministic execution."""

    if deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_dataloader_worker(_worker_id: int) -> None:
    """Seed Python and NumPy from the worker seed assigned by DataLoader."""

    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def compute_initial_model_sha256(model: torch.nn.Module) -> str:
    """Return a deterministic fingerprint of a model state dictionary."""

    digest = hashlib.sha256()
    state_dict = model.state_dict()
    for key in sorted(state_dict):
        tensor = state_dict[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _move_batch_to_device(batch: Dict[str, object], device: torch.device) -> Dict[str, object]:
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def _cycle_batches(dataloader: DataLoader) -> Iterable[Dict[str, object]]:
    while True:
        for batch in dataloader:
            yield batch


def _load_normalization_stats(stats_path: Optional[Path]) -> Optional[Dict[str, Any]]:
    if stats_path is None:
        return None
    stats = torch.load(stats_path, map_location="cpu")
    if not isinstance(stats, dict):
        raise ValueError("normalization stats file must contain a dict")
    required_keys = (
        "qpos_mean",
        "qpos_std",
        "action_mean",
        "action_std",
        "force_mean",
        "force_std",
    )
    for key in required_keys:
        if key not in stats:
            raise KeyError(f"normalization stats missing required key: {key}")
        if not torch.is_tensor(stats[key]):
            raise ValueError(f"normalization stats {key!r} must be a torch.Tensor")
    return stats


def _validate_normalization_action_mode(
    stats: Optional[Dict[str, Any]],
    action_mode: str,
) -> None:
    if stats is None or "action_mode" not in stats:
        return
    stats_action_mode = stats["action_mode"]
    if stats_action_mode != action_mode:
        raise ValueError(
            "normalization stats action_mode mismatch: "
            f"stats action_mode={stats_action_mode!r}, requested action_mode={action_mode!r}. "
            "Recompute normalization stats for the requested action_mode."
        )


def _normalize_batch(
    batch: Dict[str, object],
    stats: Dict[str, Any],
) -> Dict[str, object]:
    normalized = dict(batch)
    normalized["qpos"] = normalize_tensor(
        normalized["qpos"],
        stats["qpos_mean"],
        stats["qpos_std"],
    )
    normalized["force_window"] = normalize_tensor(
        normalized["force_window"],
        stats["force_mean"],
        stats["force_std"],
    )
    normalized["action_chunk"] = normalize_tensor(
        normalized["action_chunk"],
        stats["action_mean"],
        stats["action_std"],
    )
    normalized["future_force_chunk"] = normalize_tensor(
        normalized["future_force_chunk"],
        stats["force_mean"],
        stats["force_std"],
    )
    return normalized


def resolve_checkpoint_steps(
    *,
    max_steps: int,
    save_every: int,
    save_steps: Sequence[int],
) -> list[int]:
    if max_steps <= 0:
        raise ValueError("--max-steps must be positive")
    if save_every < 0:
        raise ValueError("--save-every must be non-negative")

    resolved_steps = set()
    for step in save_steps:
        if step <= 0:
            raise ValueError(f"--save-steps values must be positive integers; got {step}")
        if step > max_steps:
            raise ValueError(
                f"--save-steps values must be <= --max-steps ({max_steps}); got {step}"
            )
        resolved_steps.add(step)

    if save_every > 0:
        resolved_steps.update(range(save_every, max_steps + 1, save_every))

    return sorted(resolved_steps)


def checkpoint_step_path(output_dir: Path, step: int) -> Path:
    return output_dir / f"checkpoint_step_{step:08d}.pt"


def build_checkpoint_payload(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    config: Dict[str, object],
    step: int,
    training_seed: int = DEFAULT_TRAINING_SEED,
    dataloader_seed: int = DEFAULT_TRAINING_SEED + DATALOADER_SEED_OFFSET,
    deterministic_enabled: bool = False,
    initial_model_sha256: Optional[str] = None,
    torch_num_threads: Optional[int] = None,
    torch_num_interop_threads: Optional[int] = None,
    epoch: Optional[int] = None,
    step_in_epoch: Optional[int] = None,
    best_metric: Optional[float] = None,
    best_epoch: Optional[int] = None,
    best_step: Optional[int] = None,
    epochs_without_improvement: int = 0,
    stop_reason: Optional[str] = None,
) -> Dict[str, object]:
    if initial_model_sha256 is None:
        initial_model_sha256 = compute_initial_model_sha256(model)
    if torch_num_threads is None:
        torch_num_threads = torch.get_num_threads()
    if torch_num_interop_threads is None:
        torch_num_interop_threads = torch.get_num_interop_threads()
    return {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config,
        "step": step,
        "training_seed": training_seed,
        "dataloader_seed": dataloader_seed,
        "deterministic_enabled": deterministic_enabled,
        "initial_model_sha256": initial_model_sha256,
        "torch_num_threads": torch_num_threads,
        "torch_num_interop_threads": torch_num_interop_threads,
        "epoch": epoch,
        "step_in_epoch": step_in_epoch,
        "best_metric": best_metric,
        "best_epoch": best_epoch,
        "best_step": best_step,
        "epochs_without_improvement": epochs_without_improvement,
        "stop_reason": stop_reason,
    }


def save_checkpoint_atomic(payload: Dict[str, object], checkpoint_path: Path) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
    if temporary_path.exists():
        temporary_path.unlink()
    try:
        torch.save(payload, temporary_path)
        temporary_path.replace(checkpoint_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _config_from_args(args: argparse.Namespace) -> Dict[str, object]:
    policy_variant = getattr(args, "policy_variant", DEFAULT_POLICY_VARIANT)
    train_contact_latent_mode = getattr(args, "train_contact_latent_mode", "posterior")
    training_seed = getattr(args, "seed", DEFAULT_TRAINING_SEED)
    deterministic_enabled = getattr(args, "deterministic", False)
    torch_num_threads = getattr(args, "resolved_torch_num_threads", torch.get_num_threads())
    torch_num_interop_threads = getattr(
        args,
        "resolved_torch_num_interop_threads",
        torch.get_num_interop_threads(),
    )
    architecture_metadata: Dict[str, object] = {}
    if policy_variant == "force_aware_contact_cvae":
        architecture_metadata = {
            "uses_motion_latent": False,
            "uses_contact_latent": True,
            "train_contact_latent_mode": train_contact_latent_mode,
            "deployment_contact_latent_modes": ["zero", "prior"],
        }
    return {
        "episode_paths": [str(path) for path in args.episode_paths],
        "action_mode": args.action_mode,
        "policy_variant": policy_variant,
        "train_latent_mode": args.train_latent_mode,
        "train_contact_latent_mode": train_contact_latent_mode,
        "chunk_len": args.chunk_len,
        "force_window_len": args.force_window_len,
        "force_window_duration": args.force_window_duration,
        "image_size": tuple(args.image_size),
        "camera_names": tuple(args.camera_names),
        "imagenet_normalize": args.imagenet_normalize,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "training_seed": training_seed,
        "dataloader_seed": training_seed + DATALOADER_SEED_OFFSET,
        "deterministic_enabled": deterministic_enabled,
        "torch_num_threads": torch_num_threads,
        "torch_num_interop_threads": torch_num_interop_threads,
        "max_steps": args.max_steps,
        "max_epochs": getattr(args, "max_epochs", None),
        "steps_per_epoch": getattr(args, "steps_per_epoch", None),
        "effective_max_steps": getattr(args, "effective_max_steps", args.max_steps),
        "val_episode_paths": [str(path) for path in getattr(args, "val_episode_paths", ())],
        "val_every_epochs": getattr(args, "val_every_epochs", 1),
        "early_stop_patience": getattr(args, "early_stop_patience", 8),
        "early_stop_min_epochs": getattr(args, "early_stop_min_epochs", 10),
        "early_stop_min_delta": getattr(args, "early_stop_min_delta", 0.005),
        "early_stop_metric": getattr(args, "early_stop_metric", "deploy_loss"),
        "validation_deployment_mode": getattr(
            args, "resolved_validation_deployment_mode", ""
        ),
        "learning_rate": args.learning_rate,
        "lambda_force": args.lambda_force,
        "lambda_prior": args.lambda_prior,
        "prior_loss_mode": args.prior_loss_mode,
        "beta_motion_max": args.beta_motion_max,
        "beta_contact_max": args.beta_contact_max,
        "warmup_steps": args.warmup_steps,
        "save_every": getattr(args, "save_every", 0),
        "save_steps": tuple(getattr(args, "save_steps", ())),
        "intermediate_checkpoint_steps": tuple(
            getattr(args, "intermediate_checkpoint_steps", ())
        ),
        "output_dir": str(args.output_dir),
        "log_csv": str(args.log_csv),
        "validation_log": str(getattr(args, "validation_log", "")),
        "device": args.device,
        "normalization_stats_path": (
            str(args.normalization_stats) if args.normalization_stats is not None else None
        ),
        "model": {
            "pretrained_resnet18": False,
            "d_model": 128,
            "z_dim": 16,
            "action_dim": 7,
            "force_dim": 6,
            "chunk_len": args.chunk_len,
            "nhead": 4,
            "num_encoder_layers": 1,
            "num_decoder_layers": 1,
            "dim_feedforward": 256,
        },
        **architecture_metadata,
    }


def _build_training_dataset(
    args: argparse.Namespace,
    episode_paths: Optional[Sequence[Path]] = None,
) -> ContactForceHDF5Dataset:
    return ContactForceHDF5Dataset(
        args.episode_paths if episode_paths is None else episode_paths,
        camera_names=tuple(args.camera_names),
        action_mode=args.action_mode,
        chunk_len=args.chunk_len,
        force_window_len=args.force_window_len,
        force_window_duration=args.force_window_duration,
        image_size=tuple(args.image_size),
        imagenet_normalize=args.imagenet_normalize,
    )


def train(args: argparse.Namespace) -> int:
    torch_num_threads, torch_num_interop_threads = _configure_cpu_threads_from_args(args)
    policy_variant = getattr(args, "policy_variant", DEFAULT_POLICY_VARIANT)
    args.policy_variant = policy_variant
    training_seed = getattr(args, "seed", DEFAULT_TRAINING_SEED)
    deterministic_enabled = getattr(args, "deterministic", False)
    dataloader_seed = training_seed + DATALOADER_SEED_OFFSET
    configure_reproducibility(training_seed, deterministic_enabled)
    intermediate_checkpoint_steps = resolve_checkpoint_steps(
        max_steps=args.max_steps,
        save_every=getattr(args, "save_every", 0),
        save_steps=getattr(args, "save_steps", ()),
    )
    args.intermediate_checkpoint_steps = intermediate_checkpoint_steps
    intermediate_checkpoint_step_set = set(intermediate_checkpoint_steps)
    device = torch.device(args.device)
    normalization_stats = _load_normalization_stats(args.normalization_stats)
    _validate_normalization_action_mode(normalization_stats, args.action_mode)
    val_episode_paths = list(getattr(args, "val_episode_paths", ()))
    if val_episode_paths:
        validate_disjoint_episode_splits(args.episode_paths, val_episode_paths)
        validate_normalization_training_episodes(normalization_stats, args.episode_paths)
    dataset = _build_training_dataset(args)
    if len(dataset) == 0:
        print("error: dataset is empty for the requested settings", file=sys.stderr)
        return 1

    dataloader_generator = torch.Generator(device="cpu")
    dataloader_generator.manual_seed(dataloader_seed)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        generator=dataloader_generator,
        worker_init_fn=seed_dataloader_worker,
    )
    steps_per_epoch = compute_steps_per_epoch(len(dataset), args.batch_size)
    max_epochs = getattr(args, "max_epochs", None)
    effective_max_steps = args.max_steps
    if max_epochs is not None:
        effective_max_steps = min(effective_max_steps, max_epochs * steps_per_epoch)
    args.steps_per_epoch = steps_per_epoch
    args.effective_max_steps = effective_max_steps

    val_dataloader = None
    validation_deployment_mode = ""
    early_stopping = None
    if val_episode_paths:
        val_dataset = _build_training_dataset(args, val_episode_paths)
        if len(val_dataset) == 0:
            print("error: validation dataset is empty for the requested settings", file=sys.stderr)
            return 1
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            worker_init_fn=seed_dataloader_worker,
        )
        validation_deployment_mode = resolve_validation_deployment_mode(
            policy_variant=policy_variant,
            requested_mode=getattr(args, "validation_deployment_mode", "auto"),
            train_latent_mode=args.train_latent_mode,
            lambda_prior=args.lambda_prior,
        )
        early_stopping = EarlyStoppingState(
            patience=getattr(args, "early_stop_patience", 8),
            min_epochs=getattr(args, "early_stop_min_epochs", 10),
            min_delta=getattr(args, "early_stop_min_delta", 0.005),
        )
    args.resolved_validation_deployment_mode = validation_deployment_mode
    config = _config_from_args(args)

    model_kwargs = {
        "pretrained_resnet18": False,
        "d_model": 128,
        "z_dim": 16,
        "action_dim": 7,
        "force_dim": 6,
        "chunk_len": args.chunk_len,
        "nhead": 4,
        "num_encoder_layers": 1,
        "num_decoder_layers": 1,
        "dim_feedforward": 256,
        "dropout": 0.0,
        "max_force_window_len": max(args.force_window_len, 20),
    }
    # Keep initialization independent of any RNG consumption during dataset or
    # DataLoader construction.
    configure_reproducibility(training_seed, deterministic_enabled)
    if policy_variant == "force_aware_motion_cvae":
        model = ForceAwareACTMotionCVAEPolicy(**model_kwargs).to(device)
    elif policy_variant == "force_aware_contact_cvae":
        model = ForceAwareACTContactCVAEPolicy(**model_kwargs).to(device)
    else:
        model = ForceAwareACTPolicy(**model_kwargs).to(device)
    initial_model_sha256 = compute_initial_model_sha256(model)
    config["initial_model_sha256"] = initial_model_sha256
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    model.train()

    print(f"dataset_length={len(dataset)}")
    print(f"steps_per_epoch={steps_per_epoch}")
    print(f"effective_max_steps={effective_max_steps}")
    print(f"training_seed={training_seed}")
    print(f"dataloader_seed={dataloader_seed}")
    print(f"deterministic_enabled={deterministic_enabled}")
    print(f"initial_model_sha256={initial_model_sha256}")
    print(f"deterministic_algorithms_enabled={torch.are_deterministic_algorithms_enabled()}")
    print(f"torch_num_threads={torch_num_threads}")
    print(f"torch_num_interop_threads={torch_num_interop_threads}")
    print(f"action_mode={args.action_mode}")
    print(f"policy_variant={policy_variant}")
    print(f"train_latent_mode={args.train_latent_mode}")
    if policy_variant == "force_aware_contact_cvae":
        print(f"train_contact_latent_mode={getattr(args, 'train_contact_latent_mode', 'posterior')}")
    print(f"checkpoint_path={args.output_dir / 'checkpoint.pt'}")
    print(f"intermediate_checkpoint_steps={intermediate_checkpoint_steps}")
    print(f"log_csv={args.log_csv}")
    if val_dataloader is not None:
        print(f"validation_dataset_length={len(val_dataloader.dataset)}")
        print(f"validation_deployment_mode={validation_deployment_mode}")
        print(f"early_stop_metric={getattr(args, 'early_stop_metric', 'deploy_loss')}")
        print(f"validation_log={args.validation_log}")
    print(f"normalization_enabled={normalization_stats is not None}")
    if normalization_stats is not None:
        print(f"normalization_stats_path={args.normalization_stats}")

    batch_iter = _cycle_batches(dataloader)
    last_step = 0
    last_epoch = 0
    last_step_in_epoch = 0
    stop_reason = "max_steps"
    if max_epochs is not None and effective_max_steps == max_epochs * steps_per_epoch:
        stop_reason = "max_epochs"
    args.log_csv.parent.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        log_file = stack.enter_context(args.log_csv.open("w", newline=""))
        log_writer = csv.DictWriter(
            log_file,
            fieldnames=[
                "step",
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
                "lambda_prior",
                "prior_loss_mode",
                "train_latent_mode",
                "uses_posterior_latent",
                "uses_zero_latent",
                "normalization_enabled",
                "policy_variant",
                "train_contact_latent_mode",
            ],
        )
        log_writer.writeheader()
        validation_writer = None
        if val_dataloader is not None:
            args.validation_log.parent.mkdir(parents=True, exist_ok=True)
            validation_file = stack.enter_context(args.validation_log.open("w", newline=""))
            validation_writer = csv.DictWriter(
                validation_file,
                fieldnames=[
                    "epoch",
                    "step",
                    "deployment_mode",
                    "deploy_loss",
                    "action_l1",
                    "force_l1",
                    "monitored_metric",
                    "improved",
                    "epochs_without_improvement",
                    "best_metric",
                    "best_epoch",
                    "best_step",
                ],
            )
            validation_writer.writeheader()
        for step in range(1, effective_max_steps + 1):
            last_step = step
            epoch = (step - 1) // steps_per_epoch + 1
            step_in_epoch = (step - 1) % steps_per_epoch + 1
            last_epoch = epoch
            last_step_in_epoch = step_in_epoch
            batch = _move_batch_to_device(next(batch_iter), device)
            if normalization_stats is not None:
                batch = _normalize_batch(batch, normalization_stats)
            beta_motion = linear_warmup(step, args.warmup_steps, args.beta_motion_max)
            beta_contact = linear_warmup(step, args.warmup_steps, args.beta_contact_max)
            uses_motion_cvae = policy_variant == "force_aware_motion_cvae"
            uses_contact_cvae = policy_variant == "force_aware_contact_cvae"
            train_contact_latent_mode = getattr(args, "train_contact_latent_mode", "posterior")
            uses_posterior_latent = (
                uses_motion_cvae or uses_contact_cvae or args.train_latent_mode == "posterior"
            )
            uses_zero_latent = args.train_latent_mode == "zero"
            loss_beta_motion = beta_motion if uses_posterior_latent and not uses_contact_cvae else 0.0
            loss_beta_contact = (
                0.0 if uses_motion_cvae else beta_contact if uses_posterior_latent else 0.0
            )
            loss_lambda_prior = (
                0.0 if uses_motion_cvae else args.lambda_prior if uses_posterior_latent else 0.0
            )

            optimizer.zero_grad(set_to_none=True)
            if uses_motion_cvae:
                outputs = model(
                    images=batch["images"],
                    qpos=batch["qpos"],
                    force_window=batch["force_window"],
                    action_chunk=batch["action_chunk"],
                    future_force_chunk=batch["future_force_chunk"],
                    is_training=True,
                )
                losses = compute_force_aware_motion_cvae_loss(
                    outputs=outputs,
                    action_chunk=batch["action_chunk"],
                    future_force_chunk=batch["future_force_chunk"],
                    lambda_force=args.lambda_force,
                    beta_motion=loss_beta_motion,
                )
                kl_contact_value = 0.0
                loss_prior_value = 0.0
            elif uses_contact_cvae:
                outputs = model(
                    images=batch["images"],
                    qpos=batch["qpos"],
                    force_window=batch["force_window"],
                    action_chunk=batch["action_chunk"],
                    future_force_chunk=batch["future_force_chunk"],
                    is_training=True,
                    contact_latent_mode=train_contact_latent_mode,
                )
                losses = compute_force_aware_contact_cvae_loss(
                    outputs=outputs,
                    action_chunk=batch["action_chunk"],
                    future_force_chunk=batch["future_force_chunk"],
                    lambda_force=args.lambda_force,
                    beta_contact=loss_beta_contact,
                    lambda_prior=loss_lambda_prior,
                    prior_loss_mode=args.prior_loss_mode,
                )
                kl_contact_value = losses["kl_contact"].item()
                loss_prior_value = losses["loss_prior"].item()
            else:
                outputs = model(
                    images=batch["images"],
                    qpos=batch["qpos"],
                    force_window=batch["force_window"],
                    action_chunk=batch["action_chunk"],
                    future_force_chunk=batch["future_force_chunk"],
                    is_training=True,
                    contact_latent_mode=args.train_latent_mode,
                )
                losses = compute_force_aware_act_loss(
                    outputs=outputs,
                    action_chunk=batch["action_chunk"],
                    future_force_chunk=batch["future_force_chunk"],
                    lambda_force=args.lambda_force,
                    beta_motion=loss_beta_motion,
                    beta_contact=loss_beta_contact,
                    lambda_prior=loss_lambda_prior,
                    prior_loss_mode=args.prior_loss_mode,
                    use_posterior_kl=uses_posterior_latent,
                )
                kl_contact_value = losses["kl_contact"].item()
                loss_prior_value = losses["loss_prior"].item()
            losses["loss_total"].backward()
            optimizer.step()

            if step in intermediate_checkpoint_step_set:
                checkpoint_path = checkpoint_step_path(args.output_dir, step)
                checkpoint = build_checkpoint_payload(
                    model=model,
                    optimizer=optimizer,
                    config=config,
                    step=step,
                    training_seed=training_seed,
                    dataloader_seed=dataloader_seed,
                    deterministic_enabled=deterministic_enabled,
                    initial_model_sha256=initial_model_sha256,
                    torch_num_threads=torch_num_threads,
                    torch_num_interop_threads=torch_num_interop_threads,
                    epoch=epoch,
                    step_in_epoch=step_in_epoch,
                    **(early_stopping.checkpoint_metadata() if early_stopping else {}),
                )
                save_checkpoint_atomic(checkpoint, checkpoint_path)
                print(f"saved intermediate checkpoint: {checkpoint_path}", flush=True)

            log_writer.writerow(
                {
                    "step": step,
                    "epoch": epoch,
                    "batch_in_epoch": step_in_epoch,
                    "loss_total": losses["loss_total"].item(),
                    "loss_action": losses["loss_action"].item(),
                    "loss_force": losses["loss_force"].item(),
                    "kl_motion": losses["kl_motion"].item() if "kl_motion" in losses else 0.0,
                    "kl_contact": kl_contact_value,
                    "loss_prior": loss_prior_value,
                    "beta_motion": 0.0 if uses_contact_cvae else beta_motion,
                    "beta_contact": loss_beta_contact,
                    "lambda_prior": loss_lambda_prior,
                    "prior_loss_mode": args.prior_loss_mode,
                    "train_latent_mode": args.train_latent_mode,
                    "uses_posterior_latent": uses_posterior_latent,
                    "uses_zero_latent": uses_zero_latent,
                    "normalization_enabled": normalization_stats is not None,
                    "policy_variant": policy_variant,
                    "train_contact_latent_mode": (
                        train_contact_latent_mode if uses_contact_cvae else ""
                    ),
                }
            )

            loss_parts = [
                f"step={step}",
                f"epoch={epoch}",
                f"loss_total={losses['loss_total'].item():.6g}",
                f"loss_action={losses['loss_action'].item():.6g}",
                f"loss_force={losses['loss_force'].item():.6g}",
                f"kl_motion={(losses['kl_motion'].item() if 'kl_motion' in losses else 0.0):.6g}",
                f"kl_contact={kl_contact_value:.6g}",
                f"train_latent_mode={args.train_latent_mode}",
            ]
            if loss_lambda_prior > 0:
                loss_parts.append(f"loss_prior={losses['loss_prior'].item():.6g}")
                loss_parts.append(f"lambda_prior={loss_lambda_prior:.6g}")
                loss_parts.append(f"prior_loss_mode={args.prior_loss_mode}")
            loss_parts.extend(
                [
                    f"beta_motion={(0.0 if uses_contact_cvae else beta_motion):.6g}",
                    f"beta_contact={loss_beta_contact:.6g}",
                ]
            )
            print(
                " ".join(loss_parts),
                flush=True,
            )

            epoch_finished = step_in_epoch == steps_per_epoch
            val_every_epochs = getattr(args, "val_every_epochs", 1)
            should_validate = val_dataloader is not None and (
                (epoch_finished and epoch % val_every_epochs == 0)
                or step == effective_max_steps
            )
            if should_validate:
                metrics = evaluate_deployment_metrics(
                    model=model,
                    dataloader=val_dataloader,
                    device=device,
                    policy_variant=policy_variant,
                    deployment_mode=validation_deployment_mode,
                    normalization_stats=normalization_stats,
                    lambda_force=args.lambda_force,
                )
                monitored_name = getattr(args, "early_stop_metric", "deploy_loss")
                monitored_metric = metrics[monitored_name]
                improved, should_stop = early_stopping.update(
                    monitored_metric,
                    epoch=epoch,
                    step=step,
                )
                validation_writer.writerow(
                    {
                        "epoch": epoch,
                        "step": step,
                        "deployment_mode": validation_deployment_mode,
                        "deploy_loss": metrics["deploy_loss"],
                        "action_l1": metrics["action_l1"],
                        "force_l1": metrics["force_l1"],
                        "monitored_metric": monitored_metric,
                        "improved": improved,
                        **early_stopping.checkpoint_metadata(),
                    }
                )
                validation_file.flush()
                print(
                    " ".join(
                        [
                            f"validation epoch={epoch}",
                            f"step={step}",
                            f"mode={validation_deployment_mode}",
                            f"deploy_loss={metrics['deploy_loss']:.6g}",
                            f"action_l1={metrics['action_l1']:.6g}",
                            f"force_l1={metrics['force_l1']:.6g}",
                            f"improved={improved}",
                            "epochs_without_improvement="
                            f"{early_stopping.epochs_without_improvement}",
                        ]
                    ),
                    flush=True,
                )
                if improved:
                    best_path = args.output_dir / "checkpoint_best.pt"
                    best_checkpoint = build_checkpoint_payload(
                        model=model,
                        optimizer=optimizer,
                        config=config,
                        step=step,
                        training_seed=training_seed,
                        dataloader_seed=dataloader_seed,
                        deterministic_enabled=deterministic_enabled,
                        initial_model_sha256=initial_model_sha256,
                        torch_num_threads=torch_num_threads,
                        torch_num_interop_threads=torch_num_interop_threads,
                        epoch=epoch,
                        step_in_epoch=step_in_epoch,
                        stop_reason="best_validation_metric",
                        **early_stopping.checkpoint_metadata(),
                    )
                    save_checkpoint_atomic(best_checkpoint, best_path)
                    print(f"saved best checkpoint: {best_path}", flush=True)
                if should_stop:
                    stop_reason = "early_stopping"
                    print(
                        f"early stopping at epoch={epoch} step={step} "
                        f"best_epoch={early_stopping.best_epoch} "
                        f"best_step={early_stopping.best_step}",
                        flush=True,
                    )
                    break

    checkpoint_path = args.output_dir / "checkpoint.pt"
    checkpoint = build_checkpoint_payload(
        model=model,
        optimizer=optimizer,
        config=config,
        step=last_step,
        training_seed=training_seed,
        dataloader_seed=dataloader_seed,
        deterministic_enabled=deterministic_enabled,
        initial_model_sha256=initial_model_sha256,
        torch_num_threads=torch_num_threads,
        torch_num_interop_threads=torch_num_interop_threads,
        epoch=last_epoch,
        step_in_epoch=last_step_in_epoch,
        stop_reason=stop_reason,
        **(early_stopping.checkpoint_metadata() if early_stopping else {}),
    )
    save_checkpoint_atomic(checkpoint, checkpoint_path)
    print(f"saved final checkpoint: {checkpoint_path}")
    return 0


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal ForceAwareACT training loop.")
    parser.add_argument("episode_paths", type=Path, nargs="*", help="One or more HDF5 episodes.")
    parser.add_argument("--episode-list", type=Path, default=None)
    parser.add_argument(
        "--val-episode-list",
        type=Path,
        default=None,
        help="Episode-level validation split; enables validation and early stopping.",
    )
    parser.add_argument("--action-mode", choices=ACTION_MODE_CHOICES, default="joint_pos")
    parser.add_argument(
        "--policy-variant",
        choices=POLICY_VARIANT_CHOICES,
        default="force_aware_act",
    )
    parser.add_argument("--train-latent-mode", choices=TRAIN_LATENT_MODE_CHOICES, default="posterior")
    parser.add_argument(
        "--train-contact-latent-mode",
        choices=TRAIN_CONTACT_LATENT_MODE_CHOICES,
        default="posterior",
    )
    parser.add_argument("--chunk-len", type=int, default=10)
    parser.add_argument("--force-window-len", type=int, default=20)
    parser.add_argument("--force-window-duration", type=float, default=0.25)
    parser.add_argument("--image-size", type=int, nargs=2, default=(224, 224))
    parser.add_argument("--camera-names", nargs="+", default=("ee_cam", "base_top_cam"))
    parser.add_argument("--imagenet-normalize", action="store_true")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--torch-num-threads",
        type=positive_int,
        default=None,
        help="Set PyTorch intra-op CPU threads; omitted preserves the current default.",
    )
    parser.add_argument(
        "--torch-num-interop-threads",
        type=positive_int,
        default=None,
        help="Set PyTorch inter-op CPU threads; omitted preserves the current default.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_TRAINING_SEED,
        help=(
            "Seed for model initialization and stochastic training operations; "
            "the reproducible DataLoader seed is seed + 1."
        ),
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Enable strict deterministic PyTorch, cuDNN, and cuBLAS execution.",
    )
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument(
        "--max-epochs",
        type=positive_int,
        default=None,
        help="Optional epoch cap; training stops at the first max-steps/max-epochs limit.",
    )
    parser.add_argument("--val-every-epochs", type=positive_int, default=1)
    parser.add_argument("--early-stop-patience", type=positive_int, default=8)
    parser.add_argument("--early-stop-min-epochs", type=int, default=10)
    parser.add_argument("--early-stop-min-delta", type=float, default=0.005)
    parser.add_argument("--early-stop-metric", choices=EARLY_STOP_METRICS, default="deploy_loss")
    parser.add_argument(
        "--validation-deployment-mode",
        choices=VALIDATION_DEPLOYMENT_MODES,
        default="auto",
    )
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--lambda-force", type=float, default=0.1)
    parser.add_argument("--lambda-prior", type=float, default=0.0)
    parser.add_argument("--prior-loss-mode", choices=("mse_mu", "kl_q_to_p"), default="mse_mu")
    parser.add_argument("--beta-motion-max", type=float, default=1.0e-4)
    parser.add_argument("--beta-contact-max", type=float, default=1.0e-4)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument("--save-steps", type=int, nargs="*", default=[])
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/minimal_train"))
    parser.add_argument("--log-csv", type=Path, default=Path("outputs/minimal_train/train_log.csv"))
    parser.add_argument("--validation-log", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--normalization-stats", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    _configure_cpu_threads_from_args(args)
    policy_variant = getattr(args, "policy_variant", DEFAULT_POLICY_VARIANT)
    args.policy_variant = policy_variant
    args.episode_paths = resolve_episode_paths(
        args.episode_paths, args.episode_list, project_root=REPO_ROOT
    )
    args.val_episode_paths = resolve_episode_paths(
        [], args.val_episode_list, project_root=REPO_ROOT
    )
    if args.normalization_stats is not None:
        args.normalization_stats = args.normalization_stats.expanduser()
    args.log_csv = args.log_csv.expanduser()
    args.output_dir = args.output_dir.expanduser()
    if args.validation_log is None:
        args.validation_log = args.output_dir / "validation_log.csv"
    else:
        args.validation_log = args.validation_log.expanduser()
    if not args.episode_paths:
        print("error: provide episode paths or --episode-list", file=sys.stderr)
        return 2
    if not validate_episode_paths(args.episode_paths):
        return 2
    if args.chunk_len <= 0:
        print("error: --chunk-len must be positive", file=sys.stderr)
        return 2
    if args.force_window_len <= 0:
        print("error: --force-window-len must be positive", file=sys.stderr)
        return 2
    if args.batch_size <= 0:
        print("error: --batch-size must be positive", file=sys.stderr)
        return 2
    if len(args.image_size) != 2 or args.image_size[0] <= 0 or args.image_size[1] <= 0:
        print("error: --image-size must be two positive integers", file=sys.stderr)
        return 2
    if not args.camera_names:
        print("error: --camera-names must include at least one camera", file=sys.stderr)
        return 2
    if args.max_steps <= 0:
        print("error: --max-steps must be positive", file=sys.stderr)
        return 2
    if args.early_stop_min_epochs < 0:
        print("error: --early-stop-min-epochs must be non-negative", file=sys.stderr)
        return 2
    if not 0.0 <= args.early_stop_min_delta < 1.0:
        print("error: --early-stop-min-delta must be in [0, 1)", file=sys.stderr)
        return 2
    try:
        resolve_checkpoint_steps(
            max_steps=args.max_steps,
            save_every=args.save_every,
            save_steps=args.save_steps,
        )
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    if args.lambda_prior < 0:
        print("error: --lambda-prior must be non-negative", file=sys.stderr)
        return 2
    if policy_variant == "force_aware_motion_cvae" and args.train_latent_mode != "posterior":
        print(
            "error: ForceAwareACT-MotionCVAE training requires --train-latent-mode posterior",
            file=sys.stderr,
        )
        return 2
    if policy_variant == "force_aware_motion_cvae" and args.lambda_prior != 0:
        print(
            "error: ForceAwareACT-MotionCVAE has no contact prior; use --lambda-prior 0",
            file=sys.stderr,
        )
        return 2
    if policy_variant == "force_aware_contact_cvae" and args.train_latent_mode != "posterior":
        print(
            "error: ForceAwareACT-ContactCVAE training requires --train-latent-mode posterior",
            file=sys.stderr,
        )
        return 2
    if (
        policy_variant == "force_aware_contact_cvae"
        and getattr(args, "train_contact_latent_mode", "posterior") != "posterior"
    ):
        print(
            "error: ForceAwareACT-ContactCVAE training requires "
            "--train-contact-latent-mode posterior",
            file=sys.stderr,
        )
        return 2
    if args.normalization_stats is not None and not args.normalization_stats.is_file():
        print(
            f"error: normalization stats file does not exist: {args.normalization_stats}",
            file=sys.stderr,
        )
        return 2

    try:
        return train(args)
    except Exception as error:
        print(f"error: minimal training failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
