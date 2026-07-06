#!/usr/bin/env python3
"""Train the structurally force-free ACT Motion-CVAE baseline."""

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
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from force_aware_act.data import ContactForceHDF5Dataset, normalize_tensor  # noqa: E402
from force_aware_act.models import ACTPolicyBaseline  # noqa: E402
from force_aware_act.training import compute_act_baseline_loss, linear_warmup  # noqa: E402
from force_aware_act.utils import resolve_episode_paths, validate_episode_paths  # noqa: E402
from scripts.train_minimal import (  # noqa: E402
    build_checkpoint_payload,
    checkpoint_step_path,
    resolve_checkpoint_steps,
    save_checkpoint_atomic,
)


ACTION_MODE_CHOICES = (
    "joint_pos",
    "action",
    "joint_pos_command",
    "delta_joint_cmd",
    "delta_joint_pos_command",
)


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
    for key in ("qpos_mean", "qpos_std", "action_mean", "action_std"):
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


def _normalize_batch(batch: Dict[str, object], stats: Dict[str, Any]) -> Dict[str, object]:
    normalized = dict(batch)
    normalized["qpos"] = normalize_tensor(
        normalized["qpos"],
        stats["qpos_mean"],
        stats["qpos_std"],
    )
    normalized["action_chunk"] = normalize_tensor(
        normalized["action_chunk"],
        stats["action_mean"],
        stats["action_std"],
    )
    return normalized


def _model_config_from_args(args: argparse.Namespace) -> Dict[str, object]:
    return {
        "pretrained_resnet18": False,
        "freeze_resnet18": False,
        "d_model": args.d_model,
        "z_dim": args.z_dim,
        "q_dim": 7,
        "action_dim": 7,
        "chunk_len": args.chunk_len,
        "nhead": args.nhead,
        "num_encoder_layers": args.num_encoder_layers,
        "num_decoder_layers": args.num_decoder_layers,
        "dim_feedforward": args.dim_feedforward,
        "dropout": args.dropout,
    }


def _config_from_args(args: argparse.Namespace) -> Dict[str, object]:
    return {
        "policy_variant": "act_baseline",
        "act_baseline_version": ACTPolicyBaseline.act_baseline_version,
        "uses_force": False,
        "uses_contact_latent": False,
        "motion_latent_mode": "posterior_train_zero_deploy",
        "train_latent_mode": "posterior",
        "episode_paths": [str(path) for path in args.episode_paths],
        "action_mode": args.action_mode,
        "chunk_len": args.chunk_len,
        "image_size": tuple(args.image_size),
        "camera_names": tuple(args.camera_names),
        "imagenet_normalize": args.imagenet_normalize,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "max_steps": args.max_steps,
        "learning_rate": args.learning_rate,
        "optimizer": "AdamW",
        "beta_motion_max": args.beta_motion_max,
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
        "model": _model_config_from_args(args),
    }


def _build_training_dataset(args: argparse.Namespace) -> ContactForceHDF5Dataset:
    return ContactForceHDF5Dataset(
        args.episode_paths,
        camera_names=tuple(args.camera_names),
        action_mode=args.action_mode,
        chunk_len=args.chunk_len,
        force_window_len=1,
        force_window_duration=0.0,
        image_size=tuple(args.image_size),
        imagenet_normalize=args.imagenet_normalize,
        include_force=False,
    )


def train(args: argparse.Namespace) -> int:
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

    model = ACTPolicyBaseline(**_model_config_from_args(args)).to(device)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    model.train()

    print(f"dataset_length={len(dataset)}")
    print(f"policy_variant=act_baseline")
    print(f"uses_force=False")
    print(f"uses_contact_latent=False")
    print(f"act_baseline_version={ACTPolicyBaseline.act_baseline_version}")
    print(f"motion_latent_mode=posterior_train_zero_deploy")
    print(f"train_latent_mode=posterior")
    print(f"action_mode={args.action_mode}")
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
                "kl_motion",
                "beta_motion",
                "policy_variant",
                "uses_force",
                "uses_contact_latent",
                "motion_latent_mode",
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
            optimizer.zero_grad(set_to_none=True)
            outputs = model(
                images=batch["images"],
                qpos=batch["qpos"],
                action_chunk=batch["action_chunk"],
                is_training=True,
            )
            losses = compute_act_baseline_loss(
                outputs=outputs,
                action_chunk=batch["action_chunk"],
                beta_motion=beta_motion,
            )
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
                    "kl_motion": losses["kl_motion"].item(),
                    "beta_motion": beta_motion,
                    "policy_variant": "act_baseline",
                    "uses_force": False,
                    "uses_contact_latent": False,
                    "motion_latent_mode": "posterior_train_zero_deploy",
                    "train_latent_mode": "posterior",
                    "uses_posterior_latent": True,
                    "uses_zero_latent": False,
                    "normalization_enabled": normalization_stats is not None,
                }
            )
            print(
                " ".join(
                    [
                        f"step={step}",
                        f"loss_total={losses['loss_total'].item():.6g}",
                        f"loss_action={losses['loss_action'].item():.6g}",
                        f"kl_motion={losses['kl_motion'].item():.6g}",
                        f"beta_motion={beta_motion:.6g}",
                        "policy_variant=act_baseline",
                        "train_latent_mode=posterior",
                    ]
                ),
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
    parser = argparse.ArgumentParser(description="ACT Motion-CVAE baseline training loop.")
    parser.add_argument("episode_paths", type=Path, nargs="*", help="One or more HDF5 episodes.")
    parser.add_argument("--episode-list", type=Path, default=None)
    parser.add_argument("--action-mode", choices=ACTION_MODE_CHOICES, default="joint_pos")
    parser.add_argument("--chunk-len", type=int, default=10)
    parser.add_argument("--image-size", type=int, nargs=2, default=(224, 224))
    parser.add_argument("--camera-names", nargs="+", default=("ee_cam", "base_top_cam"))
    parser.add_argument("--imagenet-normalize", action="store_true")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--beta-motion-max", type=float, default=1.0e-4)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument("--save-steps", type=int, nargs="*", default=[])
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--z-dim", type=int, default=16)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num-encoder-layers", type=int, default=1)
    parser.add_argument("--num-decoder-layers", type=int, default=1)
    parser.add_argument("--dim-feedforward", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/act_baseline_train"))
    parser.add_argument(
        "--log-csv",
        type=Path,
        default=Path("outputs/act_baseline_train/train_log.csv"),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--normalization-stats", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    args.episode_paths = resolve_episode_paths(
        args.episode_paths, args.episode_list, project_root=REPO_ROOT
    )
    if args.normalization_stats is not None:
        args.normalization_stats = args.normalization_stats.expanduser()
    args.log_csv = args.log_csv.expanduser()
    args.output_dir = args.output_dir.expanduser()
    if not args.episode_paths:
        print("error: provide episode paths or --episode-list", file=sys.stderr)
        return 2
    if not validate_episode_paths(args.episode_paths):
        return 2
    if args.chunk_len <= 0:
        print("error: --chunk-len must be positive", file=sys.stderr)
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
    if args.beta_motion_max < 0:
        print("error: --beta-motion-max must be non-negative", file=sys.stderr)
        return 2
    if args.warmup_steps < 0:
        print("error: --warmup-steps must be non-negative", file=sys.stderr)
        return 2
    if args.d_model <= 0 or args.z_dim <= 0 or args.nhead <= 0:
        print("error: model dimensions must be positive", file=sys.stderr)
        return 2
    if args.d_model % args.nhead != 0:
        print("error: --d-model must be divisible by --nhead", file=sys.stderr)
        return 2
    if args.num_encoder_layers <= 0 or args.num_decoder_layers <= 0:
        print("error: transformer layer counts must be positive", file=sys.stderr)
        return 2
    if args.dim_feedforward <= 0:
        print("error: --dim-feedforward must be positive", file=sys.stderr)
        return 2
    if args.dropout < 0:
        print("error: --dropout must be non-negative", file=sys.stderr)
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
        print(f"error: ACT baseline training failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
