#!/usr/bin/env python3
"""Train the faithful official ACT baseline on compact MuJoCo episodes."""

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
    OfficialACTConfig,
    OfficialACTPolicy,
)
from force_aware_act.official_act_training import (  # noqa: E402
    OfficialACTCriterion,
    OfficialACTEpisodicDataset,
    OfficialACTWindowDataset,
    OfficialACTNormalizationStats,
    OfficialACTSplitManifest,
    OfficialACTTrainingConfig,
    build_official_act_optimizer,
    collate_official_act,
    compute_official_act_stats,
    create_official_act_split,
    load_official_act_checkpoint,
    read_official_act_checkpoint,
    run_official_act_training_epoch,
    run_official_act_full_window_validation_epoch,
    run_official_act_validation_epoch,
    save_official_act_checkpoint,
)
from force_aware_act.official_act_training.data import (  # noqa: E402
    OFFICIAL_ACT_SPLIT_VERSION,
)
from force_aware_act.training import (  # noqa: E402
    best_policy_artifact,
    checkpoint_identity,
    resolve_convergence_monitor,
    resolve_training_horizon,
    selection_metric_migration,
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

    payload = None
    if args.resume is None:
        model_config = (
            OfficialACTConfig.compact_smoke()
            if args.smoke
            else OfficialACTConfig.canonical()
        )
        training_config = OfficialACTTrainingConfig(
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
        checkpoint_best_metric = None
        best_epoch = -1
        best_model_state = None
        prior_run_control = None
    else:
        payload = read_official_act_checkpoint(
            args.resume,
            map_location="cpu",
        )
        model_config = OfficialACTConfig(**payload["model_config"])
        training_config = OfficialACTTrainingConfig(
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
        checkpoint_best_metric = best_metric
        best_epoch = int(payload["progress"].get("best_epoch", -1))
        best_model_state = payload.get("best_model_state")
        prior_run_control = payload.get("run_control")
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
    parent_checkpoint = (
        None
        if args.resume is None
        else checkpoint_identity(args.resume, payload=payload)
    )

    model = OfficialACTPolicy(model_config).to(device)
    criterion = OfficialACTCriterion(training_config)
    optimizer = build_official_act_optimizer(model, training_config)
    if args.resume is not None:
        load_official_act_checkpoint(
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
    full_validation_dataset = OfficialACTWindowDataset(
        args.data_root,
        manifest.validation_episodes,
        normalization,
        model_config,
    )
    full_validation_loader = _make_loader(
        full_validation_dataset,
        batch_size=training_config.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        seed=training_config.seed + 3_000_000,
        pin_memory=device.type == "cuda",
    )
    steps_per_epoch = (
        len(train_dataset) + training_config.batch_size - 1
    ) // training_config.batch_size
    source_optimizer_step_limit = (
        steps_per_epoch * training_config.num_epochs
    )
    requested_target_optimizer_steps = args.target_optimizer_steps
    if (
        requested_target_optimizer_steps is None
        and prior_run_control is not None
    ):
        requested_target_optimizer_steps = int(
            prior_run_control["target_optimizer_steps"]
        )
    horizon = resolve_training_horizon(
        source_optimizer_step_limit=source_optimizer_step_limit,
        start_global_step=global_step,
        requested_target_optimizer_steps=(
            requested_target_optimizer_steps
        ),
        steps_per_epoch=steps_per_epoch,
    )
    if horizon.target_epochs is None:
        raise RuntimeError("Official ACT horizon did not resolve target epochs")
    target_epochs = horizon.target_epochs
    expected_total_steps = horizon.target_optimizer_steps
    selection_metric = "deployment_zero_action_l1_physical"
    prior_convergence = (
        None
        if prior_run_control is None
        else prior_run_control.get("convergence")
    )
    convergence_monitor = resolve_convergence_monitor(
        metric_name=selection_metric,
        minimum_optimizer_steps=args.minimum_optimizer_steps,
        patience_validations=args.early_stop_patience_validations,
        min_relative_improvement=(
            args.early_stop_min_relative_improvement
        ),
        default_minimum_optimizer_steps=source_optimizer_step_limit,
        prior_state=prior_convergence,
    )
    if (
        convergence_monitor is not None
        and convergence_monitor.minimum_optimizer_steps
        > horizon.target_optimizer_steps
    ):
        raise ValueError(
            "minimum_optimizer_steps must not exceed the target horizon"
        )
    prior_best_artifact = (
        None
        if prior_run_control is None
        else prior_run_control.get("best_artifact")
    )
    trusted_prior_best = (
        isinstance(prior_best_artifact, dict)
        and prior_best_artifact.get("metric_name") == selection_metric
        and best_model_state is not None
    )
    runtime_state = {
        "selection_metric": selection_metric,
        "parent_checkpoint": parent_checkpoint,
        "best_artifact": prior_best_artifact,
        "selection_metric_migration": (
            None
            if prior_run_control is None
            else prior_run_control.get("selection_metric_migration")
        ),
    }
    training_metadata = training_config.checkpoint_metadata()
    training_metadata["normalization_scope"] = normalization_scope
    run_metadata = {
        "model": model_config.checkpoint_metadata(),
        "training": training_metadata,
        "train_episodes": len(train_dataset),
        "validation_episodes": len(validation_dataset),
        "validation_windows": len(full_validation_dataset),
        "steps_per_epoch": steps_per_epoch,
        "expected_total_steps": expected_total_steps,
        "run_control": _run_control_metadata(
            horizon, convergence_monitor, runtime_state
        ),
        "selection_metric": selection_metric,
        "full_validation_interval_epochs": (
            args.full_validation_interval_epochs
        ),
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
    initial_full_validation = (
        run_official_act_full_window_validation_epoch(
            model,
            full_validation_loader,
            normalization,
            device=device,
        )
    )
    initial_metric = initial_full_validation[selection_metric]
    historical_best_validation = None
    if (
        args.resume is not None
        and prior_convergence is None
        and not trusted_prior_best
        and best_model_state is not None
    ):
        model.load_state_dict(best_model_state, strict=True)
        historical_best_validation = (
            run_official_act_full_window_validation_epoch(
                model,
                full_validation_loader,
                normalization,
                device=device,
            )
        )
        model.load_state_dict(payload["model_state"], strict=True)
    if convergence_monitor is None:
        if trusted_prior_best:
            best_metric = float(prior_best_artifact["metric"])
        else:
            historical_metric = (
                None
                if historical_best_validation is None
                else historical_best_validation[selection_metric]
            )
            if (
                historical_metric is not None
                and historical_metric < initial_metric
            ):
                best_metric = historical_metric
            else:
                best_metric = initial_metric
                best_epoch = start_epoch
                best_model_state = {
                    name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()
                }
            if args.resume is not None:
                runtime_state["selection_metric_migration"] = (
                    selection_metric_migration(
                        source_metric_name=(
                            None
                            if prior_run_control is None
                            else prior_run_control.get("selection_metric")
                        ),
                        source_best_metric=checkpoint_best_metric,
                        target_metric_name=selection_metric,
                        target_metric=best_metric,
                        global_step=global_step,
                        reason="legacy_best_revalidated_in_physical_units",
                    )
                )
    elif convergence_monitor.validations_seen == 0:
        convergence_monitor.update(initial_metric, global_step=global_step)
        if convergence_monitor.best_metric is None:
            raise RuntimeError("convergence monitor did not initialize")
        best_metric = convergence_monitor.best_metric
        best_epoch = start_epoch
        best_model_state = {
            name: value.detach().cpu().clone()
            for name, value in model.state_dict().items()
        }
    else:
        if convergence_monitor.best_metric is None:
            raise RuntimeError("resumed convergence monitor has no best metric")
        if best_model_state is None:
            raise RuntimeError(
                "resumed convergence state has no embedded best model"
            )
        best_metric = convergence_monitor.best_metric
    best_step = (
        global_step
        if convergence_monitor is None
        or convergence_monitor.best_step is None
        else convergence_monitor.best_step
    )
    runtime_state["best_artifact"] = best_policy_artifact(
        output_dir / "best_policy.pt",
        metric_name=selection_metric,
        metric=best_metric,
        global_step=best_step,
        storage="lightweight_policy_checkpoint",
    )
    run_metadata["run_control"] = _run_control_metadata(
        horizon, convergence_monitor, runtime_state
    )
    _atomic_json(output_dir / "run_metadata.json", run_metadata)
    print(json.dumps(run_metadata, sort_keys=True), flush=True)
    initial_record = {
        "record_type": "full_validation",
        "stage": "run_start",
        "epoch": start_epoch,
        "global_step": global_step,
        "selection_metric": selection_metric,
        "validation": initial_full_validation,
        "historical_best_validation": historical_best_validation,
        "selection_metric_migration": runtime_state[
            "selection_metric_migration"
        ],
        "best_artifact": runtime_state["best_artifact"],
        "convergence": (
            None
            if convergence_monitor is None
            else convergence_monitor.to_dict()
        ),
    }
    _append_json(metrics_path, initial_record)
    print(json.dumps(initial_record, sort_keys=True), flush=True)
    last_full_validation = initial_full_validation
    stopped_early = False
    completed_epochs = start_epoch

    for epoch in range(start_epoch, target_epochs):
        train_dataset.set_epoch(epoch)
        validation_dataset.set_epoch(epoch)
        validation_loader = _make_loader(
            validation_dataset,
            batch_size=training_config.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            seed=training_config.seed + 1_000_000 + epoch,
            pin_memory=device.type == "cuda",
        )
        validation_metrics = run_official_act_validation_epoch(
            model,
            criterion,
            validation_loader,
            device=device,
            include_deployment_diagnostics=(
                epoch == 0
                or (epoch + 1)
                % training_config.checkpoint_interval_epochs
                == 0
                or epoch + 1 == target_epochs
            ),
        )
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
                    "step_in_epoch": (
                        current_global_step - global_step
                    ),
                    "global_step": current_global_step,
                    "metrics": metrics,
                    "cuda_memory": _cuda_memory(device),
                }
                _append_json(metrics_path, record)
                print(json.dumps(record, sort_keys=True), flush=True)

        train_metrics = run_official_act_training_epoch(
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
        completed_epochs = epoch + 1
        full_validation_metrics = {}
        if (
            completed_epochs % args.full_validation_interval_epochs == 0
            or completed_epochs == target_epochs
        ):
            full_validation_metrics = (
                run_official_act_full_window_validation_epoch(
                    model,
                    full_validation_loader,
                    normalization,
                    device=device,
                )
            )
            last_full_validation = full_validation_metrics
            selected = full_validation_metrics[selection_metric]
            checkpoint_improved = selected < best_metric
            if convergence_monitor is not None:
                convergence_update = convergence_monitor.update(
                    selected,
                    global_step=global_step,
                )
                checkpoint_improved = (
                    convergence_update.checkpoint_improved
                )
                stopped_early = convergence_update.should_stop
                if convergence_monitor.best_metric is None:
                    raise RuntimeError(
                        "convergence monitor did not record a best metric"
                    )
                best_metric = convergence_monitor.best_metric
            if checkpoint_improved:
                best_metric = selected
                best_epoch = completed_epochs
                best_model_state = {
                    name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()
                }
                runtime_state["best_artifact"] = best_policy_artifact(
                    output_dir / "best_policy.pt",
                    metric_name=selection_metric,
                    metric=best_metric,
                    global_step=global_step,
                    storage="lightweight_policy_checkpoint",
                )
            full_record = {
                "record_type": "full_validation",
                "stage": "after_training",
                "epoch": completed_epochs,
                "global_step": global_step,
                "selection_metric": (
                    selection_metric
                ),
                "validation": full_validation_metrics,
                "convergence": (
                    None
                    if convergence_monitor is None
                    else convergence_monitor.to_dict()
                ),
            }
            _append_json(metrics_path, full_record)
            print(json.dumps(full_record, sort_keys=True), flush=True)
        record = {
            "record_type": "epoch_segment",
            "epoch": epoch,
            "completed_epoch": True,
            "step_in_epoch": 0,
            "global_step": global_step,
            "train": train_metrics,
            "validation": validation_metrics,
            "full_validation": full_validation_metrics,
        }
        _append_json(metrics_path, record)
        print(json.dumps(record, sort_keys=True), flush=True)

        if (
            completed_epochs
            % training_config.checkpoint_interval_epochs
            == 0
            or completed_epochs == target_epochs
            or stopped_early
        ):
            save_official_act_checkpoint(
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
                run_control=_run_control_metadata(
                    horizon, convergence_monitor, runtime_state
                ),
            )
        if stopped_early:
            break

    validation_dataset.set_epoch(completed_epochs)
    final_validation = run_official_act_validation_epoch(
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
        include_deployment_diagnostics=True,
    )
    final_path = output_dir / "final.pt"
    save_official_act_checkpoint(
        final_path,
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
        run_control=_run_control_metadata(
            horizon, convergence_monitor, runtime_state
        ),
    )
    if best_model_state is None:
        raise RuntimeError("official ACT training did not select a best model")
    _atomic_torch_save(
        output_dir / "best_policy.pt",
        {
            "architecture_version": model_config.architecture_version,
            "model_config": asdict(model_config),
            "architecture_metadata": model_config.checkpoint_metadata(),
            "best_epoch": best_epoch,
            "best_metric": best_metric,
            "selection_metric": selection_metric,
            "model_state": best_model_state,
            "normalization": normalization.to_dict(),
            "split_manifest": manifest.to_dict(),
            "experiment_manifest": experiment_provenance,
        },
    )
    reloaded = load_official_act_checkpoint(
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
        "epochs": completed_epochs,
        "source_epochs": training_config.num_epochs,
        "global_step": global_step,
        "expected_total_steps": expected_total_steps,
        "run_control": _run_control_metadata(
            horizon, convergence_monitor, runtime_state
        ),
        "stop_reason": (
            "maximum_optimizer_steps_reached"
            if global_step >= expected_total_steps
            else "early_stopping_plateau"
        ),
        "normalization_scope": normalization_scope,
        "experiment_manifest": experiment_provenance,
        "best_metric": best_metric,
        "best_epoch": best_epoch,
        "selection_metric": "deployment_zero_action_l1_physical",
        "best_artifact": runtime_state["best_artifact"],
        "parent_checkpoint": parent_checkpoint,
        "selection_metric_migration": runtime_state[
            "selection_metric_migration"
        ],
        "final_validation": final_validation,
        "final_full_validation": last_full_validation,
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


def _run_control_metadata(
    horizon,
    convergence_monitor,
    runtime_state,
) -> dict:
    metadata = horizon.to_dict()
    metadata["convergence"] = (
        None
        if convergence_monitor is None
        else convergence_monitor.to_dict()
    )
    metadata.update(runtime_state)
    return metadata


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
    parser.add_argument(
        "--target-optimizer-steps",
        type=int,
        help=(
            "absolute optimizer-step target; on --resume this can extend "
            "the checkpoint's original training horizon and must end on an "
            "Official ACT epoch boundary"
        ),
    )
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint-interval-epochs", type=int, default=100)
    parser.add_argument(
        "--full-validation-interval-epochs",
        type=int,
        default=400,
        help=(
            "run deterministic all-window physical-action validation every "
            "this many completed epochs"
        ),
    )
    parser.add_argument("--minimum-optimizer-steps", type=int)
    parser.add_argument(
        "--early-stop-patience-validations",
        type=int,
        help="enable validation-driven early stopping with this patience",
    )
    parser.add_argument(
        "--early-stop-min-relative-improvement",
        type=float,
        default=0.01,
    )
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--experiment-manifest", type=Path)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if (
        args.batch_size <= 0
        or args.num_epochs <= 0
        or (
            args.target_optimizer_steps is not None
            and args.target_optimizer_steps <= 0
        )
        or args.num_workers < 0
        or args.checkpoint_interval_epochs <= 0
        or args.full_validation_interval_epochs <= 0
        or (
            args.minimum_optimizer_steps is not None
            and args.minimum_optimizer_steps < 0
        )
        or (
            args.early_stop_patience_validations is not None
            and args.early_stop_patience_validations <= 0
        )
        or not 0.0
        <= args.early_stop_min_relative_improvement
        < 1.0
        or args.log_interval <= 0
    ):
        parser.error("numeric training arguments are invalid")
    if args.smoke and args.experiment_manifest is not None:
        parser.error("--smoke cannot be combined with --experiment-manifest")
    if (
        args.minimum_optimizer_steps is not None
        and args.early_stop_patience_validations is None
    ):
        parser.error(
            "--minimum-optimizer-steps requires "
            "--early-stop-patience-validations"
        )
    return args


if __name__ == "__main__":
    main()
