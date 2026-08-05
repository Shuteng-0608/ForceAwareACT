#!/usr/bin/env python3
"""Audit the native 500 Hz contact-CVAE before a long training run."""

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
    ACTAlignedHighRateCriterion,
    ACTAlignedHighRateHDF5Dataset,
    ACTAlignedHighRateTrainingConfig,
    build_act_aligned_high_rate_optimizer,
    collate_high_rate_samples,
    compute_high_rate_normalization_stats,
    create_episode_split,
    run_high_rate_training_preflight,
)
from force_aware_act.models.act_aligned import (  # noqa: E402
    ACTAlignedHighRateConfig,
    ACTAlignedHighRateContactCVAEPolicy,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("data_root", type=Path)
    parser.add_argument("--device", type=torch.device, default=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true")
    args = parser.parse_args()
    if not args.data_root.is_dir():
        parser.error(f"data root does not exist: {args.data_root}")
    if args.batch_size <= 0:
        parser.error("batch-size must be positive")
    if args.device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    torch.manual_seed(args.seed)
    config = (
        ACTAlignedHighRateConfig(
            d_model=32, nhead=4, dim_feedforward=64, local_force_dim=32,
            dropout=0.0, chunk_len=6, image_height=64, image_width=64,
            pretrained_backbone=False, imagenet_normalize=False,
        )
        if args.smoke
        else ACTAlignedHighRateConfig.canonical_act(
            pretrained_backbone=not args.no_pretrained,
            imagenet_normalize=not args.no_pretrained,
        )
    )
    training_config = ACTAlignedHighRateTrainingConfig(batch_size=args.batch_size, seed=args.seed)
    manifest = create_episode_split(args.data_root, validation_fraction=0.1, seed=args.seed)
    normalization = compute_high_rate_normalization_stats(args.data_root, manifest.train_episodes)
    dataset = ACTAlignedHighRateHDF5Dataset(args.data_root, manifest.train_episodes, normalization, config)
    try:
        batch = collate_high_rate_samples([dataset[index] for index in range(args.batch_size)]).to(args.device)
        model = ACTAlignedHighRateContactCVAEPolicy(config).to(args.device)
        criterion = ACTAlignedHighRateCriterion(training_config)
        optimizer = build_act_aligned_high_rate_optimizer(model, training_config)
        report = run_high_rate_training_preflight(model, criterion, optimizer, batch)
    finally:
        dataset.close()
    report["input_source"] = str(args.data_root.resolve())
    serialized = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)


if __name__ == "__main__":
    main()
