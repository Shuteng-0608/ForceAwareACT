#!/usr/bin/env python3
"""Train the structurally latent-free official ACT baseline."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from force_aware_act.act_aligned_training.split import (  # noqa: E402
    discover_episodes,
)
from force_aware_act.experiments import (  # noqa: E402
    load_paired_episode_subset_manifest,
    validate_checkpoint_experiment_provenance,
)
from force_aware_act.models.official_act import (  # noqa: E402
    OfficialACTNoLatentConfig,
    OfficialACTNoLatentPolicy,
)
from force_aware_act.official_act_training import (  # noqa: E402
    OfficialACTEpisodicDataset,
    OfficialACTNoLatentCriterion,
    OfficialACTNoLatentTrainingConfig,
    OfficialACTNormalizationStats,
    OfficialACTSplitManifest,
    build_official_act_no_latent_optimizer,
    collate_official_act,
    compute_official_act_stats,
    create_official_act_split,
    load_official_act_no_latent_checkpoint,
    read_official_act_no_latent_checkpoint,
    run_official_act_no_latent_training_epoch,
    run_official_act_no_latent_validation_epoch,
    save_official_act_no_latent_checkpoint,
)
from force_aware_act.official_act_training.data import (  # noqa: E402
    OFFICIAL_ACT_SPLIT_VERSION,
)


def main() -> None:
    args = _parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"
    if args.resume is None and metrics_path.exists():
        raise FileExistsError(
            "output directory already contains metrics.jsonl; use a new "
            "directory or --resume"
        )
    device = torch.device(args.device)
    _seed_everything(args.seed)
    experiment_subset = (
        None
        if args.experiment_manifest is None
        else load_paired_episode_subset_manifest(
            args.experiment_manifest,
            data_root=args.data_root,
        )
    )

    if args.resume is None:
        model_config = (
            OfficialACTNoLatentConfig.compact_smoke()
            if args.smoke
            else OfficialACTNoLatentConfig.canonical()
        )
        training_config = OfficialACTNoLatentTrainingConfig(
            batch_size=2 if args.smoke else args.batch_size,
            num_epochs=1 if args.smoke else args.num_epochs,
            seed=args.seed,
            split_seed=(
                experiment_subset.seed
                if experiment_subset is not None
                else 1
            ),
            checkpoint_interval_epochs=(
                1 if args.smoke else args.checkpoint_interval_epochs
            ),
        )
        if experiment_subset is None:
            full_manifest = create_official_act_split(
                args.data_root,
                validation_fraction=training_config.validation_fraction,
                split_seed=training_config.split_seed,
            )
            if args.smoke:
                manifest = OfficialACTSplitManifest(
                    format_version=full_manifest.format_version,
                    data_root=full_manifest.data_root,
                    split_seed=full_manifest.split_seed,
                    validation_fraction=full_manifest.validation_fraction,
                    train_episodes=full_manifest.train_episodes[:4],
                    validation_episodes=full_manifest.validation_episodes[:2],
                )
            else:
                manifest = full_manifest
            normalization_episodes = discover_episodes(args.data_root)
            normalization_scope = "all_episodes_official_behavior"
            experiment_provenance = None
        else:
            manifest = OfficialACTSplitManifest(
                format_version=OFFICIAL_ACT_SPLIT_VERSION,
                data_root=str(args.data_root.resolve()),
                split_seed=experiment_subset.seed,
                validation_fraction=(
                    len(experiment_subset.validation_episodes)
                    / (
                        len(experiment_subset.train_episodes)
                        + len(experiment_subset.validation_episodes)
                    )
                ),
                train_episodes=experiment_subset.train_episodes,
                validation_episodes=experiment_subset.validation_episodes,
            )
            normalization_episodes = manifest.train_episodes
            normalization_scope = "paired_manifest_train_episodes_only"
            experiment_provenance = (
                experiment_subset.checkpoint_provenance()
            )
        normalization = compute_official_act_stats(
            args.data_root,
            normalization_episodes,
        )
        start_epoch = 0
        global_step = 0
        best_metric = float("inf")
        best_epoch = -1
        best_model_state = None
    else:
        payload = read_official_act_no_latent_checkpoint(
            args.resume,
            map_location="cpu",
        )
        model_config = OfficialACTNoLatentConfig(**payload["model_config"])
        training_config = OfficialACTNoLatentTrainingConfig(
            **payload["training_config"]
        )
        manifest = OfficialACTSplitManifest.from_dict(
            payload["split_manifest"]
        )
        normalization = OfficialACTNormalizationStats.from_dict(
            payload["normalization"]
        )
        start_epoch = int(payload["progress"]["epoch"])
        global_step = int(payload["progress"]["global_step"])
        best_metric = float(payload["progress"]["best_metric"])
        best_epoch = int(payload["progress"].get("best_epoch", -1))
        best_model_state = payload.get("best_model_state")
        experiment_provenance = payload.get("experiment_manifest")
        normalization_scope = (
            "paired_manifest_train_episodes_only"
            if experiment_provenance is not None
            else "all_episodes_official_behavior"
        )
        if experiment_subset is not None:
            validate_checkpoint_experiment_provenance(
                experiment_provenance,
                experiment_subset,
            )

    model = OfficialACTNoLatentPolicy(model_config).to(device)
    criterion = OfficialACTNoLatentCriterion(training_config)
    optimizer = build_official_act_no_latent_optimizer(
        model,
        training_config,
    )
    if args.resume is not None:
        load_official_act_no_latent_checkpoint(
            args.resume,
            model=model,
            optimizer=optimizer,
            training_config=training_config,
            map_location=device,
        )

    train_dataset = OfficialACTEpisodicDataset(
        args.data_root,
        manifest.train_episodes,
        normalization,
        model_config,
        sampling_seed=training_config.seed,
        stream_id=0,
    )
    validation_dataset = OfficialACTEpisodicDataset(
        args.data_root,
        manifest.validation_episodes,
        normalization,
        model_config,
        sampling_seed=training_config.seed,
        stream_id=1,
    )
    steps_per_epoch = (
        len(train_dataset) + training_config.batch_size - 1
    ) // training_config.batch_size
    expected_total_steps = steps_per_epoch * training_config.num_epochs
    training_metadata = training_config.checkpoint_metadata()
    training_metadata["normalization_scope"] = normalization_scope
    run_metadata = {
        "model": model_config.checkpoint_metadata(),
        "training": training_metadata,
        "train_episodes": len(train_dataset),
        "validation_episodes": len(validation_dataset),
        "steps_per_epoch": steps_per_epoch,
        "expected_total_steps": expected_total_steps,
        "normalization_scope": normalization_scope,
        "experiment_manifest": experiment_provenance,
        "formal_native_image_resolution": (
            not args.smoke
            and (
                model_config.image_height,
                model_config.image_width,
            )
            == (480, 640)
        ),
    }
    _atomic_json(output_dir / "run_metadata.json", run_metadata)
    print(json.dumps(run_metadata, sort_keys=True), flush=True)

    for epoch in range(start_epoch, training_config.num_epochs):
        train_dataset.set_epoch(epoch)
        validation_dataset.set_epoch(epoch)
        validation_metrics = run_official_act_no_latent_validation_epoch(
            model,
            criterion,
            _make_loader(
                validation_dataset,
                batch_size=training_config.batch_size,
                shuffle=True,
                num_workers=args.num_workers,
                seed=training_config.seed + 1_000_000 + epoch,
                pin_memory=device.type == "cuda",
            ),
            device=device,
        )
        selected = validation_metrics[training_config.selection_metric]
        if selected < best_metric:
            best_metric = selected
            best_epoch = epoch
            best_model_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }

        train_loader = _make_loader(
            train_dataset,
            batch_size=training_config.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            seed=training_config.seed + epoch,
            pin_memory=device.type == "cuda",
        )

        def log_step(
            current_global_step: int,
            metrics: dict[str, float],
        ) -> None:
            if (
                current_global_step == global_step + 1
                or current_global_step % args.log_interval == 0
                or current_global_step == global_step + steps_per_epoch
            ):
                record = {
                    "record_type": "step",
                    "epoch": epoch,
                    "step_in_epoch": current_global_step - global_step,
                    "global_step": current_global_step,
                    "metrics": metrics,
                    "cuda_memory": _cuda_memory(device),
                }
                _append_json(metrics_path, record)
                print(json.dumps(record, sort_keys=True), flush=True)

        train_metrics = run_official_act_no_latent_training_epoch(
            model,
            criterion,
            optimizer,
            train_loader,
            training_config,
            device=device,
            global_step_start=global_step,
            step_callback=log_step,
        )
        global_step += int(train_metrics["optimizer_steps"])
        record = {
            "record_type": "epoch_segment",
            "epoch": epoch,
            "completed_epoch": True,
            "step_in_epoch": 0,
            "global_step": global_step,
            "train": train_metrics,
            "validation": validation_metrics,
        }
        _append_json(metrics_path, record)
        print(json.dumps(record, sort_keys=True), flush=True)

        completed_epochs = epoch + 1
        if (
            completed_epochs
            % training_config.checkpoint_interval_epochs
            == 0
            or completed_epochs == training_config.num_epochs
        ):
            save_official_act_no_latent_checkpoint(
                output_dir / "latest.pt",
                model=model,
                optimizer=optimizer,
                training_config=training_config,
                normalization=normalization,
                split_manifest=manifest,
                epoch=completed_epochs,
                global_step=global_step,
                best_metric=best_metric,
                best_epoch=best_epoch,
                best_model_state=best_model_state,
                experiment_manifest=experiment_provenance,
            )

    validation_dataset.set_epoch(training_config.num_epochs)
    final_validation = run_official_act_no_latent_validation_epoch(
        model,
        criterion,
        _make_loader(
            validation_dataset,
            batch_size=training_config.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            seed=training_config.seed + 2_000_000,
            pin_memory=device.type == "cuda",
        ),
        device=device,
    )
    final_path = output_dir / "final.pt"
    save_official_act_no_latent_checkpoint(
        final_path,
        model=model,
        optimizer=optimizer,
        training_config=training_config,
        normalization=normalization,
        split_manifest=manifest,
        epoch=training_config.num_epochs,
        global_step=global_step,
        best_metric=best_metric,
        best_epoch=best_epoch,
        best_model_state=best_model_state,
        experiment_manifest=experiment_provenance,
    )
    if best_model_state is None:
        raise RuntimeError("latent-free ACT training did not select a best model")
    _atomic_torch_save(
        output_dir / "best_policy.pt",
        {
            "architecture_version": model_config.architecture_version,
            "model_config": asdict(model_config),
            "architecture_metadata": model_config.checkpoint_metadata(),
            "best_epoch": best_epoch,
            "best_metric": best_metric,
            "model_state": best_model_state,
            "normalization": normalization.to_dict(),
            "split_manifest": manifest.to_dict(),
            "experiment_manifest": experiment_provenance,
        },
    )
    reloaded = load_official_act_no_latent_checkpoint(
        final_path,
        model=model,
        optimizer=optimizer,
        training_config=training_config,
        map_location=device,
    )
    summary = {
        "passed": True,
        "architecture_version": model_config.architecture_version,
        "training_version": training_config.training_version,
        "epochs": training_config.num_epochs,
        "global_step": global_step,
        "expected_total_steps": expected_total_steps,
        "normalization_scope": normalization_scope,
        "experiment_manifest": experiment_provenance,
        "best_metric": best_metric,
        "best_epoch": best_epoch,
        "final_validation": final_validation,
        "checkpoint": str(final_path.resolve()),
        "reload_audit": {
            "passed": True,
            "epoch": int(reloaded["progress"]["epoch"]),
            "global_step": int(reloaded["progress"]["global_step"]),
        },
    }
    _atomic_json(output_dir / "training_summary.json", summary)
    print(json.dumps(summary, sort_keys=True), flush=True)


def _make_loader(
    dataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
    pin_memory: bool,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_official_act,
        generator=torch.Generator().manual_seed(seed),
        worker_init_fn=_seed_worker,
        pin_memory=pin_memory,
    )


def _seed_worker(_worker_id: int) -> None:
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _append_json(path: Path, value: dict) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, sort_keys=True) + "\n")


def _atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_torch_save(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _cuda_memory(device: torch.device) -> dict:
    if device.type != "cuda":
        return {
            "device": str(device),
            "allocated_bytes": None,
            "reserved_bytes": None,
            "peak_allocated_bytes": None,
            "peak_reserved_bytes": None,
        }
    return {
        "device": str(device),
        "allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "reserved_bytes": int(torch.cuda.memory_reserved(device)),
        "peak_allocated_bytes": int(
            torch.cuda.max_memory_allocated(device)
        ),
        "peak_reserved_bytes": int(
            torch.cuda.max_memory_reserved(device)
        ),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-epochs", type=int, default=2000)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint-interval-epochs", type=int, default=100)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--experiment-manifest", type=Path)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if (
        args.batch_size <= 0
        or args.num_epochs <= 0
        or args.num_workers < 0
        or args.checkpoint_interval_epochs <= 0
        or args.log_interval <= 0
    ):
        parser.error("numeric training arguments are invalid")
    if args.smoke and args.experiment_manifest is not None:
        parser.error("--smoke cannot be combined with --experiment-manifest")
    return args


if __name__ == "__main__":
    main()
