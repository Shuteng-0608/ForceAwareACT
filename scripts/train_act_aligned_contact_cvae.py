#!/usr/bin/env python3
"""Train only the new ACT-aligned contact-CVAE stack."""

from __future__ import annotations

import argparse
import json
import os
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
        resume_step_in_epoch = 0
        global_step = 0
        best_metric = float("inf")
    else:
        payload = read_act_aligned_checkpoint(args.resume, map_location="cpu")
        model_config = ACTAlignedConfig(**payload["model_config"])
        training_config = ACTAlignedTrainingConfig(**payload["training_config"])
        manifest = EpisodeSplitManifest.from_dict(payload["split_manifest"])
        normalization = NormalizationStats.from_dict(payload["normalization"])
        start_epoch = int(payload["progress"]["epoch"])
        resume_step_in_epoch = int(
            payload["progress"].get("step_in_epoch", 0)
        )
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
    if args.max_train_steps is not None:
        if global_step >= args.max_train_steps:
            raise ValueError(
                "checkpoint global_step has already reached "
                "--max-train-steps"
            )
        if resume_step_in_epoch >= len(train_loader):
            raise ValueError(
                "checkpoint step_in_epoch must be smaller than epoch length"
            )
    run_start_global_step = global_step
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    initial_memory = _cuda_memory_snapshot(device)
    stopped_at_limit = False
    try:
        for epoch in range(start_epoch, training_config.num_epochs):
            step_in_epoch = (
                resume_step_in_epoch if epoch == start_epoch else 0
            )
            epoch_generator_state = generator.get_state()
            remaining_steps = (
                None
                if args.max_train_steps is None
                else args.max_train_steps - global_step
            )
            segment_start_global_step = global_step

            def log_step(
                segment_step: int,
                step_metrics: dict[str, float],
            ) -> None:
                current_global_step = (
                    segment_start_global_step + segment_step
                )
                should_log = (
                    current_global_step == run_start_global_step + 1
                    or current_global_step % args.log_interval == 0
                    or (
                        args.max_train_steps is not None
                        and current_global_step == args.max_train_steps
                    )
                )
                if not should_log:
                    return
                record = {
                    "record_type": "step",
                    "epoch": epoch,
                    "step_in_epoch": step_in_epoch + segment_step,
                    "global_step": current_global_step,
                    "metrics": step_metrics,
                    "cuda_memory": _cuda_memory_snapshot(device),
                }
                _append_json_record(log_path, record)
                print(json.dumps(record, sort_keys=True), flush=True)

            train_metrics = run_training_epoch(
                model,
                criterion,
                optimizer,
                train_loader,
                training_config,
                device=device,
                max_optimizer_steps=remaining_steps,
                skip_batches=step_in_epoch,
                step_callback=log_step,
            )
            optimizer_steps = int(train_metrics["optimizer_steps"])
            global_step += optimizer_steps
            next_step_in_epoch = step_in_epoch + optimizer_steps
            completed_epoch = next_step_in_epoch >= len(train_loader)
            progress = TrainingProgress(
                epoch=epoch + 1 if completed_epoch else epoch,
                global_step=global_step,
                best_metric=best_metric,
                step_in_epoch=0 if completed_epoch else next_step_in_epoch,
            )
            checkpoint_generator = (
                generator
                if completed_epoch
                else _generator_from_state(epoch_generator_state)
            )
            training_memory = _cuda_memory_snapshot(device)
            stopped_at_limit = (
                args.max_train_steps is not None
                and global_step >= args.max_train_steps
            )
            validation_metrics = {}
            should_validate = stopped_at_limit or (
                completed_epoch
                and (epoch + 1) % training_config.validation_interval == 0
            )
            if should_validate:
                training_rng_state = _capture_runtime_rng_state()
                try:
                    validation_metrics = run_validation_epoch(
                        model,
                        criterion,
                        validation_loader,
                        training_config,
                        device=device,
                    )
                finally:
                    _restore_runtime_rng_state(training_rng_state)
                selected = validation_metrics[training_config.selection_metric]
                if selected < best_metric:
                    best_metric = selected
                    progress = TrainingProgress(
                        progress.epoch,
                        progress.global_step,
                        best_metric,
                        progress.step_in_epoch,
                    )
                    _save(
                        args.output_dir / "best.pt",
                        model,
                        optimizer,
                        training_config,
                        progress,
                        normalization,
                        manifest,
                        checkpoint_generator,
                    )
            validation_memory = _cuda_memory_snapshot(device)
            record = {
                "record_type": "epoch_segment",
                "epoch": epoch,
                "completed_epoch": completed_epoch,
                "step_in_epoch": progress.step_in_epoch,
                "global_step": global_step,
                "train": train_metrics,
                "validation": validation_metrics,
            }
            _append_json_record(log_path, record)
            print(json.dumps(record, sort_keys=True), flush=True)
            if (
                completed_epoch
                and (epoch + 1) % training_config.checkpoint_interval == 0
            ):
                _save(
                    args.output_dir / f"epoch_{epoch + 1:04d}.pt",
                    model,
                    optimizer,
                    training_config,
                    progress,
                    normalization,
                    manifest,
                    checkpoint_generator,
                )
            _save(
                args.output_dir / "last.pt",
                model,
                optimizer,
                training_config,
                progress,
                normalization,
                manifest,
                checkpoint_generator,
            )
            if stopped_at_limit:
                burn_in_path = args.output_dir / "burn_in.pt"
                _save(
                    burn_in_path,
                    model,
                    optimizer,
                    training_config,
                    progress,
                    normalization,
                    manifest,
                    checkpoint_generator,
                )
                expected_rng_state = _capture_runtime_rng_state()
                reload_audit = _audit_checkpoint_reload(
                    burn_in_path,
                    model=model,
                    optimizer=optimizer,
                    training_config=training_config,
                    expected_progress=progress,
                    expected_generator_state=(
                        checkpoint_generator.get_state()
                    ),
                    expected_rng_state=expected_rng_state,
                )
                summary = {
                    "mode": "burn_in",
                    "passed": True,
                    "stop_reason": "max_train_steps_reached",
                    "requested_max_train_steps": args.max_train_steps,
                    "run_start_global_step": run_start_global_step,
                    "global_step": global_step,
                    "progress": {
                        "epoch": progress.epoch,
                        "step_in_epoch": progress.step_in_epoch,
                        "best_metric": progress.best_metric,
                    },
                    "completed_epoch": completed_epoch,
                    "train": train_metrics,
                    "validation": validation_metrics,
                    "memory": {
                        "before_training": initial_memory,
                        "after_training": training_memory,
                        "after_validation": validation_memory,
                    },
                    "optimizer_state": {
                        "parameter_entries": len(optimizer.state),
                        "tensor_bytes": _optimizer_state_tensor_bytes(
                            optimizer
                        ),
                    },
                    "checkpoint": {
                        "path": str(burn_in_path.resolve()),
                        "reload_audit": reload_audit,
                    },
                }
                _atomic_write_json(
                    args.output_dir / "burn_in_summary.json",
                    summary,
                )
                print(json.dumps(summary, sort_keys=True), flush=True)
                break
            resume_step_in_epoch = 0
    finally:
        train_dataset.close()
        validation_dataset.close()
    if args.max_train_steps is not None and not stopped_at_limit:
        raise RuntimeError(
            "training ended before --max-train-steps was reached"
        )


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
    progress,
    normalization,
    manifest,
    generator,
) -> None:
    save_act_aligned_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        training_config=training_config,
        progress=progress,
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
    parser.add_argument(
        "--max-train-steps",
        type=int,
        help="absolute global optimizer-step limit for a controlled burn-in",
    )
    parser.add_argument("--log-interval", type=int, default=10)
    args = parser.parse_args()
    if (
        args.batch_size <= 0
        or args.epochs <= 0
        or args.num_workers < 0
        or args.log_interval <= 0
    ):
        parser.error(
            "batch-size/epochs/log-interval must be positive and "
            "num-workers non-negative"
        )
    if args.max_train_steps is not None and args.max_train_steps <= 0:
        parser.error("max-train-steps must be positive")
    return args


def _append_json_record(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")


def _generator_from_state(state: torch.Tensor) -> torch.Generator:
    generator = torch.Generator()
    generator.set_state(state)
    return generator


def _cuda_memory_snapshot(device: torch.device) -> dict:
    if device.type != "cuda":
        return {
            "device": str(device),
            "allocated_bytes": None,
            "reserved_bytes": None,
            "peak_allocated_bytes": None,
            "peak_reserved_bytes": None,
        }
    torch.cuda.synchronize(device)
    return {
        "device": str(device),
        "allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "reserved_bytes": int(torch.cuda.memory_reserved(device)),
        "peak_allocated_bytes": int(
            torch.cuda.max_memory_allocated(device)
        ),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }


def _optimizer_state_tensor_bytes(
    optimizer: torch.optim.Optimizer,
) -> int:
    return sum(
        value.numel() * value.element_size()
        for state in optimizer.state.values()
        for value in state.values()
        if isinstance(value, torch.Tensor)
    )


def _capture_runtime_rng_state() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": (
            torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else None
        ),
    }


def _restore_runtime_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def _audit_checkpoint_reload(
    path: Path,
    *,
    model: ACTAlignedContactCVAEPolicy,
    optimizer: torch.optim.Optimizer,
    training_config: ACTAlignedTrainingConfig,
    expected_progress: TrainingProgress,
    expected_generator_state: torch.Tensor,
    expected_rng_state: dict,
) -> dict:
    loaded = load_act_aligned_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        training_config=training_config,
        restore_rng=True,
        map_location="cpu",
    )
    if loaded.progress != expected_progress:
        raise RuntimeError("checkpoint reload progress mismatch")
    if loaded.dataloader_generator_state is None:
        raise RuntimeError("checkpoint is missing DataLoader generator state")
    if not torch.equal(
        loaded.dataloader_generator_state,
        expected_generator_state,
    ):
        raise RuntimeError("checkpoint reload DataLoader state mismatch")
    actual_rng_state = _capture_runtime_rng_state()
    if not _rng_states_equal(actual_rng_state, expected_rng_state):
        raise RuntimeError("checkpoint reload RNG state mismatch")
    return {
        "passed": True,
        "epoch": loaded.progress.epoch,
        "step_in_epoch": loaded.progress.step_in_epoch,
        "global_step": loaded.progress.global_step,
        "optimizer_parameter_entries": len(optimizer.state),
        "rng_restore_tested": True,
    }


def _rng_states_equal(first: dict, second: dict) -> bool:
    if first["python"] != second["python"]:
        return False
    first_numpy = first["numpy"]
    second_numpy = second["numpy"]
    if (
        first_numpy[0] != second_numpy[0]
        or not np.array_equal(first_numpy[1], second_numpy[1])
        or first_numpy[2:] != second_numpy[2:]
    ):
        return False
    if not torch.equal(first["torch"], second["torch"]):
        return False
    if first["cuda"] is None or second["cuda"] is None:
        return first["cuda"] is None and second["cuda"] is None
    return len(first["cuda"]) == len(second["cuda"]) and all(
        torch.equal(left, right)
        for left, right in zip(first["cuda"], second["cuda"])
    )


def _atomic_write_json(path: Path, payload: dict) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_path, path)


if __name__ == "__main__":
    main()
