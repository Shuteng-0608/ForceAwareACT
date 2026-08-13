#!/usr/bin/env python3
"""Audit latent-free official ACT on real HDF5 episodes before training."""

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

from force_aware_act.act_aligned_training.split import (  # noqa: E402
    discover_episodes,
)
from force_aware_act.models.official_act import (  # noqa: E402
    OfficialACTNoLatentConfig,
    OfficialACTNoLatentPolicy,
)
from force_aware_act.official_act_training import (  # noqa: E402
    OfficialACTNoLatentCriterion,
    OfficialACTNoLatentTrainingConfig,
    OfficialACTEpisodicDataset,
    build_official_act_no_latent_optimizer,
    collate_official_act,
    compute_official_act_stats,
    create_official_act_split,
    evaluate_official_act_no_latent_batch,
    train_official_act_no_latent_step,
)


def main() -> None:
    args = _parse_args()
    torch.manual_seed(args.seed)
    model_config = (
        OfficialACTNoLatentConfig.compact_smoke()
        if args.smoke
        else OfficialACTNoLatentConfig.canonical(
            pretrained_backbone=not args.no_pretrained,
            imagenet_normalize=not args.no_pretrained,
        )
    )
    training_config = OfficialACTNoLatentTrainingConfig(
        batch_size=args.batch_size,
        seed=args.seed,
    )
    manifest = create_official_act_split(
        args.data_root,
        validation_fraction=training_config.validation_fraction,
        split_seed=training_config.split_seed,
    )
    normalization = compute_official_act_stats(
        args.data_root,
        discover_episodes(args.data_root),
    )
    dataset = OfficialACTEpisodicDataset(
        args.data_root,
        manifest.train_episodes,
        normalization,
        model_config,
        sampling_seed=training_config.seed,
        stream_id=0,
    )
    if args.batch_size > len(dataset):
        raise ValueError("batch size exceeds the training episode count")
    batch = collate_official_act(
        [dataset[index] for index in range(args.batch_size)]
    ).to(args.device)
    model = OfficialACTNoLatentPolicy(model_config).to(args.device)
    _audit_structure(model)
    criterion = OfficialACTNoLatentCriterion(training_config)
    optimizer = build_official_act_no_latent_optimizer(
        model,
        training_config,
    )
    validation_before = evaluate_official_act_no_latent_batch(
        model,
        criterion,
        batch,
    )
    train_metrics = train_official_act_no_latent_step(
        model,
        criterion,
        optimizer,
        batch,
        training_config,
    )
    validation_after = evaluate_official_act_no_latent_batch(
        model,
        criterion,
        batch,
    )
    with torch.no_grad():
        output = model(batch.images, batch.qpos)
    report = {
        "passed": True,
        "architecture_version": model_config.architecture_version,
        "input_source": str(args.data_root.resolve()),
        "batch_size": batch.batch_size,
        "image_shape": list(batch.images.shape),
        "qpos_shape": list(batch.qpos.shape),
        "action_shape": list(output["pred_action"].shape),
        "memory_shape": list(output["memory_tokens"].shape),
        "latent_mechanism": output["latent_mechanism"],
        "force_input": "none",
        "train": train_metrics,
        "validation_before": validation_before,
        "validation_after": validation_after,
    }
    serialized = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)


def _audit_structure(model: OfficialACTNoLatentPolicy) -> None:
    forbidden = ("latent", "posterior", "prior")
    violations = [
        name
        for name, _module in model.named_modules()
        if any(token in name for token in forbidden)
    ]
    violations.extend(
        name
        for name, _parameter in model.named_parameters()
        if any(token in name for token in forbidden)
    )
    if violations:
        raise RuntimeError(
            "latent-free policy contains forbidden state: "
            + ", ".join(sorted(set(violations)))
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
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
