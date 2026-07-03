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
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from force_aware_act.data import ContactForceHDF5Dataset, normalize_tensor  # noqa: E402
from force_aware_act.models import ForceAwareACTMotionCVAEPolicy, ForceAwareACTPolicy  # noqa: E402
from force_aware_act.training import (  # noqa: E402
    compute_force_aware_act_loss,
    compute_force_aware_motion_cvae_loss,
    linear_warmup,
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
POLICY_VARIANT_CHOICES = ("force_aware_act", "force_aware_motion_cvae")
DEFAULT_POLICY_VARIANT = "force_aware_act"


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
) -> Dict[str, object]:
    return {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config,
        "step": step,
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
    return {
        "episode_paths": [str(path) for path in args.episode_paths],
        "action_mode": args.action_mode,
        "policy_variant": policy_variant,
        "train_latent_mode": args.train_latent_mode,
        "chunk_len": args.chunk_len,
        "force_window_len": args.force_window_len,
        "force_window_duration": args.force_window_duration,
        "image_size": tuple(args.image_size),
        "camera_names": tuple(args.camera_names),
        "imagenet_normalize": args.imagenet_normalize,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "max_steps": args.max_steps,
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
    }


def _build_training_dataset(args: argparse.Namespace) -> ContactForceHDF5Dataset:
    return ContactForceHDF5Dataset(
        args.episode_paths,
        camera_names=tuple(args.camera_names),
        action_mode=args.action_mode,
        chunk_len=args.chunk_len,
        force_window_len=args.force_window_len,
        force_window_duration=args.force_window_duration,
        image_size=tuple(args.image_size),
        imagenet_normalize=args.imagenet_normalize,
    )


def train(args: argparse.Namespace) -> int:
    policy_variant = getattr(args, "policy_variant", DEFAULT_POLICY_VARIANT)
    args.policy_variant = policy_variant
    intermediate_checkpoint_steps = resolve_checkpoint_steps(
        max_steps=args.max_steps,
        save_every=getattr(args, "save_every", 0),
        save_steps=getattr(args, "save_steps", ()),
    )
    args.intermediate_checkpoint_steps = intermediate_checkpoint_steps
    intermediate_checkpoint_step_set = set(intermediate_checkpoint_steps)
    config = _config_from_args(args)
    device = torch.device(args.device)
    normalization_stats = _load_normalization_stats(args.normalization_stats)
    _validate_normalization_action_mode(normalization_stats, args.action_mode)
    dataset = _build_training_dataset(args)
    if len(dataset) == 0:
        print("error: dataset is empty for the requested settings", file=sys.stderr)
        return 1

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

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
    if policy_variant == "force_aware_motion_cvae":
        model = ForceAwareACTMotionCVAEPolicy(**model_kwargs).to(device)
    else:
        model = ForceAwareACTPolicy(**model_kwargs).to(device)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    model.train()

    print(f"dataset_length={len(dataset)}")
    print(f"action_mode={args.action_mode}")
    print(f"policy_variant={policy_variant}")
    print(f"train_latent_mode={args.train_latent_mode}")
    print(f"checkpoint_path={args.output_dir / 'checkpoint.pt'}")
    print(f"intermediate_checkpoint_steps={intermediate_checkpoint_steps}")
    print(f"log_csv={args.log_csv}")
    print(f"normalization_enabled={normalization_stats is not None}")
    if normalization_stats is not None:
        print(f"normalization_stats_path={args.normalization_stats}")

    batch_iter = _cycle_batches(dataloader)
    last_step = 0
    args.log_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.log_csv.open("w", newline="") as log_file:
        log_writer = csv.DictWriter(
            log_file,
            fieldnames=[
                "step",
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
            ],
        )
        log_writer.writeheader()
        for step in range(1, args.max_steps + 1):
            last_step = step
            batch = _move_batch_to_device(next(batch_iter), device)
            if normalization_stats is not None:
                batch = _normalize_batch(batch, normalization_stats)
            beta_motion = linear_warmup(step, args.warmup_steps, args.beta_motion_max)
            beta_contact = linear_warmup(step, args.warmup_steps, args.beta_contact_max)
            uses_motion_cvae = policy_variant == "force_aware_motion_cvae"
            uses_posterior_latent = uses_motion_cvae or args.train_latent_mode == "posterior"
            uses_zero_latent = args.train_latent_mode == "zero"
            loss_beta_motion = beta_motion if uses_posterior_latent else 0.0
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
                )
                save_checkpoint_atomic(checkpoint, checkpoint_path)
                print(f"saved intermediate checkpoint: {checkpoint_path}", flush=True)

            log_writer.writerow(
                {
                    "step": step,
                    "loss_total": losses["loss_total"].item(),
                    "loss_action": losses["loss_action"].item(),
                    "loss_force": losses["loss_force"].item(),
                    "kl_motion": losses["kl_motion"].item(),
                    "kl_contact": kl_contact_value,
                    "loss_prior": loss_prior_value,
                    "beta_motion": beta_motion,
                    "beta_contact": loss_beta_contact,
                    "lambda_prior": loss_lambda_prior,
                    "prior_loss_mode": args.prior_loss_mode,
                    "train_latent_mode": args.train_latent_mode,
                    "uses_posterior_latent": uses_posterior_latent,
                    "uses_zero_latent": uses_zero_latent,
                    "normalization_enabled": normalization_stats is not None,
                }
            )

            loss_parts = [
                f"step={step}",
                f"loss_total={losses['loss_total'].item():.6g}",
                f"loss_action={losses['loss_action'].item():.6g}",
                f"loss_force={losses['loss_force'].item():.6g}",
                f"kl_motion={losses['kl_motion'].item():.6g}",
                f"kl_contact={kl_contact_value:.6g}",
                f"train_latent_mode={args.train_latent_mode}",
            ]
            if loss_lambda_prior > 0:
                loss_parts.append(f"loss_prior={losses['loss_prior'].item():.6g}")
                loss_parts.append(f"lambda_prior={loss_lambda_prior:.6g}")
                loss_parts.append(f"prior_loss_mode={args.prior_loss_mode}")
            loss_parts.extend(
                [
                    f"beta_motion={beta_motion:.6g}",
                    f"beta_contact={loss_beta_contact:.6g}",
                ]
            )
            print(
                " ".join(loss_parts),
                flush=True,
            )

    checkpoint_path = args.output_dir / "checkpoint.pt"
    checkpoint = build_checkpoint_payload(
        model=model,
        optimizer=optimizer,
        config=config,
        step=last_step,
    )
    save_checkpoint_atomic(checkpoint, checkpoint_path)
    print(f"saved final checkpoint: {checkpoint_path}")
    return 0


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal ForceAwareACT training loop.")
    parser.add_argument("episode_paths", type=Path, nargs="*", help="One or more HDF5 episodes.")
    parser.add_argument("--episode-list", type=Path, default=None)
    parser.add_argument("--action-mode", choices=ACTION_MODE_CHOICES, default="joint_pos")
    parser.add_argument(
        "--policy-variant",
        choices=POLICY_VARIANT_CHOICES,
        default="force_aware_act",
    )
    parser.add_argument("--train-latent-mode", choices=TRAIN_LATENT_MODE_CHOICES, default="posterior")
    parser.add_argument("--chunk-len", type=int, default=10)
    parser.add_argument("--force-window-len", type=int, default=20)
    parser.add_argument("--force-window-duration", type=float, default=0.25)
    parser.add_argument("--image-size", type=int, nargs=2, default=(224, 224))
    parser.add_argument("--camera-names", nargs="+", default=("ee_cam", "base_top_cam"))
    parser.add_argument("--imagenet-normalize", action="store_true")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=20)
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
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--normalization-stats", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    policy_variant = getattr(args, "policy_variant", DEFAULT_POLICY_VARIANT)
    args.policy_variant = policy_variant
    args.episode_paths = resolve_episode_paths(
        args.episode_paths, args.episode_list, project_root=REPO_ROOT
    )
    if args.normalization_stats is not None:
        args.normalization_stats = args.normalization_stats.expanduser()
    args.log_csv = args.log_csv.expanduser()
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
