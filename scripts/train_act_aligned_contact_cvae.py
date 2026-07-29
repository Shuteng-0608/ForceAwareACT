#!/usr/bin/env python3
"""Train only the new ACT-aligned contact-CVAE stack."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from force_aware_act.act_aligned_training import (  # noqa: E402
    ACTAlignedCriterion,
    ACTAlignedHDF5Dataset,
    ACTAlignedTrainingConfig,
    EpisodeSplitManifest,
    NormalizationStats,
    TrainingProgress,
    build_act_aligned_optimizer,
    collate_act_aligned_samples,
    compute_normalization_stats,
    create_episode_split,
    load_act_aligned_checkpoint,
    read_act_aligned_checkpoint,
    run_training_epoch,
    run_validation_epoch,
    save_act_aligned_checkpoint,
)
from force_aware_act.models.act_aligned import (  # noqa: E402
    ACTAlignedConfig,
    ACTAlignedContactCVAEPolicy,
)


def main() -> None:
    args = _parse_args()
    device = torch.device(args.device)
    _seed_everything(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.resume is None:
        model_config = (
            ACTAlignedConfig(
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
            if args.smoke
            else ACTAlignedConfig.canonical_act()
        )
        training_config = ACTAlignedTrainingConfig(
            batch_size=2 if args.smoke else args.batch_size,
            num_epochs=1 if args.smoke else args.epochs,
            seed=args.seed,
        )
        manifest = create_episode_split(
            args.data_root,
            validation_fraction=args.validation_fraction,
            seed=args.seed,
        )
        normalization = compute_normalization_stats(
            args.data_root,
            manifest.train_episodes,
        )
        start_epoch = 0
        global_step = 0
        best_metric = float("inf")
    else:
        payload = read_act_aligned_checkpoint(args.resume, map_location="cpu")
        model_config = ACTAlignedConfig(**payload["model_config"])
        training_config = ACTAlignedTrainingConfig(**payload["training_config"])
        manifest = EpisodeSplitManifest.from_dict(payload["split_manifest"])
        normalization = NormalizationStats.from_dict(payload["normalization"])
        start_epoch = int(payload["progress"]["epoch"])
        global_step = int(payload["progress"]["global_step"])
        best_metric = float(payload["progress"]["best_metric"])

    model = ACTAlignedContactCVAEPolicy(model_config).to(device)
    criterion = ACTAlignedCriterion(training_config)
    optimizer = build_act_aligned_optimizer(model, training_config)
    generator = torch.Generator().manual_seed(training_config.seed + 1)
    if args.resume is not None:
        loaded = load_act_aligned_checkpoint(
            args.resume,
            model=model,
            optimizer=optimizer,
            training_config=training_config,
            restore_rng=True,
            map_location=device,
        )
        if loaded.dataloader_generator_state is not None:
            generator.set_state(loaded.dataloader_generator_state)

    train_dataset = ACTAlignedHDF5Dataset(
        args.data_root,
        manifest.train_episodes,
        normalization,
        model_config,
    )
    validation_dataset = ACTAlignedHDF5Dataset(
        args.data_root,
        manifest.validation_episodes,
        normalization,
        model_config,
    )
    train_source = (
        Subset(train_dataset, range(min(4, len(train_dataset))))
        if args.smoke
        else train_dataset
    )
    validation_source = (
        Subset(validation_dataset, range(min(2, len(validation_dataset))))
        if args.smoke
        else validation_dataset
    )
    train_loader = _make_loader(
        train_source,
        batch_size=training_config.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        generator=generator,
        pin_memory=device.type == "cuda",
    )
    validation_loader = _make_loader(
        validation_source,
        batch_size=training_config.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        generator=None,
        pin_memory=device.type == "cuda",
    )

    log_path = args.output_dir / "metrics.jsonl"
    try:
        for epoch in range(start_epoch, training_config.num_epochs):
            train_metrics = run_training_epoch(
                model,
                criterion,
                optimizer,
                train_loader,
                training_config,
                device=device,
            )
            global_step += int(train_metrics["optimizer_steps"])
            validation_metrics = {}
            if (epoch + 1) % training_config.validation_interval == 0:
                validation_metrics = run_validation_epoch(
                    model,
                    criterion,
                    validation_loader,
                    training_config,
                    device=device,
                )
                selected = validation_metrics[training_config.selection_metric]
                if selected < best_metric:
                    best_metric = selected
                    _save(
                        args.output_dir / "best.pt",
                        model,
                        optimizer,
                        training_config,
                        epoch + 1,
                        global_step,
                        best_metric,
                        normalization,
                        manifest,
                        generator,
                    )
            record = {
                "epoch": epoch + 1,
                "global_step": global_step,
                "train": train_metrics,
                "validation": validation_metrics,
            }
            with log_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, sort_keys=True) + "\n")
            print(json.dumps(record, sort_keys=True), flush=True)
            if (epoch + 1) % training_config.checkpoint_interval == 0:
                _save(
                    args.output_dir / f"epoch_{epoch + 1:04d}.pt",
                    model,
                    optimizer,
                    training_config,
                    epoch + 1,
                    global_step,
                    best_metric,
                    normalization,
                    manifest,
                    generator,
                )
            _save(
                args.output_dir / "last.pt",
                model,
                optimizer,
                training_config,
                epoch + 1,
                global_step,
                best_metric,
                normalization,
                manifest,
                generator,
            )
    finally:
        train_dataset.close()
        validation_dataset.close()


def _make_loader(
    dataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    generator,
    pin_memory: bool,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_act_aligned_samples,
        generator=generator,
        worker_init_fn=_seed_worker,
        pin_memory=pin_memory,
    )


def _save(
    path,
    model,
    optimizer,
    training_config,
    epoch,
    global_step,
    best_metric,
    normalization,
    manifest,
    generator,
) -> None:
    save_act_aligned_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        training_config=training_config,
        progress=TrainingProgress(epoch, global_step, best_metric),
        normalization=normalization,
        split_manifest=manifest,
        dataloader_generator=generator,
    )


def _seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("data_root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.batch_size <= 0 or args.epochs <= 0 or args.num_workers < 0:
        parser.error("batch-size/epochs must be positive and num-workers non-negative")
    return args


if __name__ == "__main__":
    main()
