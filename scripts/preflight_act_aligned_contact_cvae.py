#!/usr/bin/env python3
"""Audit one canonical ACT-aligned forward/backward before long training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from force_aware_act.act_aligned_training import (  # noqa: E402
    ACTAlignedBatch,
    ACTAlignedCriterion,
    ACTAlignedHDF5Dataset,
    ACTAlignedTrainingConfig,
    build_act_aligned_optimizer,
    collate_act_aligned_samples,
    compute_normalization_stats,
    create_episode_split,
    run_training_preflight,
)
from force_aware_act.models.act_aligned import (  # noqa: E402
    ACTAlignedConfig,
    ACTAlignedContactCVAEPolicy,
)


def main() -> None:
    args = _parse_args()
    if args.device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    torch.manual_seed(args.seed)
    if args.device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    model_config = _model_config(args.smoke, args.no_pretrained)
    training_config = ACTAlignedTrainingConfig(
        batch_size=args.batch_size,
        seed=args.seed,
    )
    model = ACTAlignedContactCVAEPolicy(model_config).to(args.device)
    criterion = ACTAlignedCriterion(training_config)
    optimizer = build_act_aligned_optimizer(model, training_config)
    batch = (
        _real_batch(args.data_root, model_config, args.batch_size, args.seed)
        if args.data_root is not None
        else _synthetic_batch(model_config, args.batch_size)
    ).to(args.device)

    report = run_training_preflight(model, criterion, optimizer, batch)
    report["input_source"] = (
        str(args.data_root.resolve())
        if args.data_root is not None
        else "synthetic"
    )
    serialized = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)


def _model_config(smoke: bool, no_pretrained: bool) -> ACTAlignedConfig:
    if smoke:
        return ACTAlignedConfig(
            d_model=32,
            nhead=4,
            dim_feedforward=64,
            dropout=0.0,
            chunk_len=6,
            force_window_len=5,
            image_height=64,
            image_width=64,
            pretrained_backbone=False,
            imagenet_normalize=False,
        )
    return ACTAlignedConfig.canonical_act(
        pretrained_backbone=not no_pretrained,
        imagenet_normalize=not no_pretrained,
    )


def _synthetic_batch(
    config: ACTAlignedConfig,
    batch_size: int,
) -> ACTAlignedBatch:
    return ACTAlignedBatch(
        images=torch.randn(
            batch_size,
            config.num_cameras,
            3,
            config.image_height,
            config.image_width,
        ),
        qpos=torch.randn(batch_size, config.q_dim),
        force_history=torch.randn(
            batch_size,
            config.force_window_len,
            config.force_dim,
        ),
        force_padding_mask=torch.zeros(
            batch_size,
            config.force_window_len,
            dtype=torch.bool,
        ),
        action_chunk=torch.randn(
            batch_size,
            config.chunk_len,
            config.action_dim,
        ),
        future_force_chunk=torch.randn(
            batch_size,
            config.chunk_len,
            config.force_dim,
        ),
        future_padding_mask=torch.zeros(
            batch_size,
            config.chunk_len,
            dtype=torch.bool,
        ),
    )


def _real_batch(
    data_root: Path,
    config: ACTAlignedConfig,
    batch_size: int,
    seed: int,
) -> ACTAlignedBatch:
    manifest = create_episode_split(
        data_root,
        validation_fraction=0.1,
        seed=seed,
    )
    normalization = compute_normalization_stats(
        data_root,
        manifest.train_episodes,
    )
    dataset = ACTAlignedHDF5Dataset(
        data_root,
        manifest.train_episodes,
        normalization,
        config,
    )
    try:
        samples = [dataset[index] for index in range(batch_size)]
        return collate_act_aligned_samples(samples)
    finally:
        dataset.close()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path)
    parser.add_argument(
        "--device",
        type=torch.device,
        default=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="keep canonical dimensions but avoid pretrained weight loading",
    )
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("batch-size must be positive")
    if args.smoke and args.no_pretrained:
        parser.error("--smoke already disables pretrained weights")
    if args.data_root is not None and not args.data_root.is_dir():
        parser.error(f"data root does not exist: {args.data_root}")
    return args


if __name__ == "__main__":
    main()
