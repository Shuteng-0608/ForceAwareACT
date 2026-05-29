#!/usr/bin/env python3
"""Minimal ForceAwareACT training script.

Example:
    PYTHONPATH=src .venv/bin/python scripts/train_minimal.py test_data/episode.hdf5 --max-steps 20
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from force_aware_act.data import ContactForceHDF5Dataset  # noqa: E402
from force_aware_act.models import ForceAwareACTPolicy  # noqa: E402
from force_aware_act.training import compute_force_aware_act_loss, linear_warmup  # noqa: E402


def _move_batch_to_device(batch: Dict[str, object], device: torch.device) -> Dict[str, object]:
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def _cycle_batches(dataloader: DataLoader) -> Iterable[Dict[str, object]]:
    while True:
        for batch in dataloader:
            yield batch


def _config_from_args(args: argparse.Namespace) -> Dict[str, object]:
    return {
        "episode_paths": [str(path) for path in args.episode_paths],
        "chunk_len": args.chunk_len,
        "force_window_len": args.force_window_len,
        "force_window_duration": args.force_window_duration,
        "imagenet_normalize": args.imagenet_normalize,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "max_steps": args.max_steps,
        "learning_rate": args.learning_rate,
        "lambda_force": args.lambda_force,
        "beta_motion_max": args.beta_motion_max,
        "beta_contact_max": args.beta_contact_max,
        "warmup_steps": args.warmup_steps,
        "output_dir": str(args.output_dir),
        "device": args.device,
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


def train(args: argparse.Namespace) -> int:
    device = torch.device(args.device)
    dataset = ContactForceHDF5Dataset(
        args.episode_paths,
        camera_names=("ee_cam", "base_top_cam"),
        action_mode="joint_pos",
        chunk_len=args.chunk_len,
        force_window_len=args.force_window_len,
        force_window_duration=args.force_window_duration,
        image_size=(224, 224),
        imagenet_normalize=args.imagenet_normalize,
    )
    if len(dataset) == 0:
        print("error: dataset is empty for the requested settings", file=sys.stderr)
        return 1

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    model = ForceAwareACTPolicy(
        pretrained_resnet18=False,
        d_model=128,
        z_dim=16,
        action_dim=7,
        force_dim=6,
        chunk_len=args.chunk_len,
        nhead=4,
        num_encoder_layers=1,
        num_decoder_layers=1,
        dim_feedforward=256,
        dropout=0.0,
        max_force_window_len=max(args.force_window_len, 20),
    ).to(device)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    model.train()

    print(f"dataset_length={len(dataset)}")
    print(f"checkpoint_path={args.output_dir / 'checkpoint.pt'}")

    batch_iter = _cycle_batches(dataloader)
    last_step = 0
    for step in range(1, args.max_steps + 1):
        last_step = step
        batch = _move_batch_to_device(next(batch_iter), device)
        beta_motion = linear_warmup(step, args.warmup_steps, args.beta_motion_max)
        beta_contact = linear_warmup(step, args.warmup_steps, args.beta_contact_max)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(
            images=batch["images"],
            qpos=batch["qpos"],
            force_window=batch["force_window"],
            action_chunk=batch["action_chunk"],
            future_force_chunk=batch["future_force_chunk"],
            is_training=True,
        )
        losses = compute_force_aware_act_loss(
            outputs=outputs,
            action_chunk=batch["action_chunk"],
            future_force_chunk=batch["future_force_chunk"],
            lambda_force=args.lambda_force,
            beta_motion=beta_motion,
            beta_contact=beta_contact,
        )
        losses["loss_total"].backward()
        optimizer.step()

        print(
            " ".join(
                [
                    f"step={step}",
                    f"loss_total={losses['loss_total'].item():.6g}",
                    f"loss_action={losses['loss_action'].item():.6g}",
                    f"loss_force={losses['loss_force'].item():.6g}",
                    f"kl_motion={losses['kl_motion'].item():.6g}",
                    f"kl_contact={losses['kl_contact'].item():.6g}",
                    f"beta_motion={beta_motion:.6g}",
                    f"beta_contact={beta_contact:.6g}",
                ]
            ),
            flush=True,
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": _config_from_args(args),
        "step": last_step,
    }
    torch.save(checkpoint, args.output_dir / "checkpoint.pt")
    print(f"saved_checkpoint={args.output_dir / 'checkpoint.pt'}")
    return 0


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal ForceAwareACT training loop.")
    parser.add_argument("episode_paths", type=Path, nargs="+", help="One or more HDF5 episodes.")
    parser.add_argument("--chunk-len", type=int, default=10)
    parser.add_argument("--force-window-len", type=int, default=20)
    parser.add_argument("--force-window-duration", type=float, default=0.25)
    parser.add_argument("--imagenet-normalize", action="store_true")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--lambda-force", type=float, default=0.1)
    parser.add_argument("--beta-motion-max", type=float, default=1.0e-4)
    parser.add_argument("--beta-contact-max", type=float, default=1.0e-4)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/minimal_train"))
    parser.add_argument("--device", default="cpu")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    args.episode_paths = [path.expanduser() for path in args.episode_paths]
    for path in args.episode_paths:
        if not path.exists():
            print(f"error: file does not exist: {path}", file=sys.stderr)
            return 2
        if not path.is_file():
            print(f"error: path is not a file: {path}", file=sys.stderr)
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
    if args.max_steps <= 0:
        print("error: --max-steps must be positive", file=sys.stderr)
        return 2

    try:
        return train(args)
    except Exception as error:
        print(f"error: minimal training failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
