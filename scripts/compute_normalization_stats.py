#!/usr/bin/env python3
"""Compute ForceAwareACT dataset normalization statistics.

Example:
    PYTHONPATH=src .venv/bin/python scripts/compute_normalization_stats.py test_data/episode.hdf5 --output outputs/normalization_stats.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from force_aware_act.data import ContactForceHDF5Dataset, compute_normalization_stats  # noqa: E402


def compute_and_save(args: argparse.Namespace) -> int:
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
    print(f"dataset_length={len(dataset)}")
    stats = compute_normalization_stats(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        eps=args.eps,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(stats, args.output)
    print(f"saved_stats={args.output}")
    for key, value in stats.items():
        print(f"{key}: shape={tuple(value.shape)}")
    return 0


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute qpos/action/force normalization stats from HDF5 episodes.",
    )
    parser.add_argument("episode_paths", type=Path, nargs="+", help="One or more HDF5 episodes.")
    parser.add_argument("--output", type=Path, default=Path("outputs/normalization_stats.pt"))
    parser.add_argument("--chunk-len", type=int, default=50)
    parser.add_argument("--force-window-len", type=int, default=50)
    parser.add_argument("--force-window-duration", type=float, default=0.25)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--eps", type=float, default=1.0e-6)
    parser.add_argument("--imagenet-normalize", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    args.episode_paths = [path.expanduser() for path in args.episode_paths]
    args.output = args.output.expanduser()
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
    if args.eps <= 0:
        print("error: --eps must be positive", file=sys.stderr)
        return 2

    try:
        return compute_and_save(args)
    except Exception as error:
        print(f"error: failed to compute normalization stats: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
