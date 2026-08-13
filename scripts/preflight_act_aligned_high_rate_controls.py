#!/usr/bin/env python3
"""Audit native-rate Motion-CVAE or Dual-Zero before long training."""

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
    ACTAlignedHighRateDualZeroCriterion,
    ACTAlignedHighRateDualZeroTrainingConfig,
    ACTAlignedHighRateHDF5Dataset,
    ACTAlignedHighRateMotionCriterion,
    ACTAlignedHighRateMotionTrainingConfig,
    build_act_aligned_high_rate_dual_zero_optimizer,
    build_act_aligned_high_rate_motion_optimizer,
    collate_high_rate_samples,
    compute_high_rate_normalization_stats,
    create_episode_split,
    run_high_rate_control_preflight,
)
from force_aware_act.models.act_aligned import (  # noqa: E402
    ACTAlignedHighRateConfig,
    ACTAlignedHighRateDualZeroPolicy,
    ACTAlignedHighRateMotionCVAEPolicy,
)


def main() -> None:
    args = _parse_args()
    torch.manual_seed(args.seed)
    stack = _stack(args.variant)
    config = stack["config"](args.smoke, args.no_pretrained)
    training_config = stack["training_config"](
        batch_size=args.batch_size,
        seed=args.seed,
    )
    manifest = create_episode_split(
        args.data_root,
        validation_fraction=0.1,
        seed=args.seed,
    )
    normalization = compute_high_rate_normalization_stats(
        args.data_root,
        manifest.train_episodes,
    )
    dataset = ACTAlignedHighRateHDF5Dataset(
        args.data_root,
        manifest.train_episodes,
        normalization,
        config,
    )
    try:
        batch = collate_high_rate_samples(
            [dataset[index] for index in range(args.batch_size)]
        ).to(args.device)
        model = stack["policy"](config).to(args.device)
        criterion = stack["criterion"](training_config)
        optimizer = stack["optimizer"](model, training_config)
        report = run_high_rate_control_preflight(
            model,
            criterion,
            optimizer,
            batch,
        )
    finally:
        dataset.close()
    report["variant"] = args.variant
    report["input_source"] = str(args.data_root.resolve())
    serialized = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)


def _stack(variant: str) -> dict:
    if variant == "motion_cvae":
        return {
            "config": _motion_config,
            "training_config": ACTAlignedHighRateMotionTrainingConfig,
            "policy": ACTAlignedHighRateMotionCVAEPolicy,
            "criterion": ACTAlignedHighRateMotionCriterion,
            "optimizer": build_act_aligned_high_rate_motion_optimizer,
        }
    return {
        "config": _dual_zero_config,
        "training_config": ACTAlignedHighRateDualZeroTrainingConfig,
        "policy": ACTAlignedHighRateDualZeroPolicy,
        "criterion": ACTAlignedHighRateDualZeroCriterion,
        "optimizer": build_act_aligned_high_rate_dual_zero_optimizer,
    }


def _motion_config(smoke: bool, no_pretrained: bool):
    return _model_config(
        ACTAlignedHighRateConfig.motion_control,
        smoke,
        no_pretrained,
    )


def _dual_zero_config(smoke: bool, no_pretrained: bool):
    return _model_config(
        ACTAlignedHighRateConfig.dual_zero,
        smoke,
        no_pretrained,
    )


def _model_config(factory, smoke: bool, no_pretrained: bool):
    if smoke:
        return factory(
            d_model=32,
            nhead=4,
            dim_feedforward=64,
            local_force_dim=32,
            dropout=0.0,
            chunk_len=6,
            image_height=64,
            image_width=64,
            pretrained_backbone=False,
            imagenet_normalize=False,
        )
    return factory(
        pretrained_backbone=not no_pretrained,
        imagenet_normalize=not no_pretrained,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("variant", choices=("motion_cvae", "dual_zero"))
    parser.add_argument("data_root", type=Path)
    parser.add_argument(
        "--device",
        type=torch.device,
        default=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )
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
        parser.error("CUDA was requested but is unavailable")
    if args.smoke and args.no_pretrained:
        parser.error("--smoke already disables pretrained weights")
    return args


if __name__ == "__main__":
    main()
