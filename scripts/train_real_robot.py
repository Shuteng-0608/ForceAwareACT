#!/usr/bin/env python3
"""Inspect, smoke-test, or train right-arm policies from untouched raw HDF5."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import h5py
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from force_aware_act.act_aligned_training.high_rate_data import collate_high_rate_samples
from force_aware_act.models.act_aligned.config import canonical_model_config
from force_aware_act.real_robot.config import RealRobotDataConfig
from force_aware_act.real_robot.data import RealRobotHighRateDataset
from force_aware_act.real_robot.reader import RawEpisode, discover_real_robot_episodes
from force_aware_act.real_robot.normalization import compute_real_robot_normalization_stats
from force_aware_act.real_robot.checkpoint import (
    canonical, read_checkpoint, restore_checkpoint, sampler_states_equal, save_checkpoint, source_fingerprint,
)
from force_aware_act.real_robot.training_sampling import (
    EPISODIC_UNIFORM_SAMPLING, TRAIN_SAMPLING_MODES, RealRobotTrainingSampler,
    resolve_training_sampling_mode,
)
from force_aware_act.real_robot.training_control import EarlyStoppingConfig, EarlyStoppingController
from force_aware_act.real_robot.multi_metric_control import (
    MultiMetricController, resolve_metric_mode, restore_multi_metric_controller, validation_objectives,
)
from force_aware_act.real_robot.evaluation import fixed_indices, index_manifest, evaluate_episodes
from force_aware_act.real_robot.contact_latent_mode import (
    CONTACT_LATENT_MODES, contact_latent_contract, resolve_contact_latent_mode,
    validate_contact_latent_identity,
)
from force_aware_act.real_robot.training import (
    MODEL_KINDS, RealRobotOptimizationConfig, build_criterion, build_model,
    build_optimizer, evaluate, make_model_config, train_step,
)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def source_groups(root, records):
    groups = {}
    for record in records:
        with h5py.File(record.resolve(root), "r") as handle:
            # Derived clips from the same original recording must stay together.
            source = str(handle.attrs.get("source_sha256", ""))
        groups[record.episode_id] = source or record.episode_id
    return groups


def choose_split(records, groups, *, mode, seed, fraction, supplied=None):
    by_id = {record.episode_id: record for record in records}
    if len(by_id) != len(records):
        raise ValueError("duplicate episode IDs")
    if supplied is not None:
        train_ids = supplied["train_episode_ids"]
        validation_ids = supplied["validation_episode_ids"]
        test_ids = supplied.get("test_episode_ids", [])
        all_ids = train_ids + validation_ids + test_ids
        if len(all_ids) != len(set(all_ids)) or set(all_ids) != set(by_id):
            raise ValueError("split must include every discovered episode exactly once")
    elif mode == "smoke":
        train_ids, validation_ids = sorted(by_id), []
        test_ids = []
    else:
        unique_groups = sorted(set(groups.values()))
        if len(unique_groups) < 2:
            raise ValueError("formal training needs at least two independent recordings; use --mode smoke for one episode")
        random.Random(seed).shuffle(unique_groups)
        count = min(len(unique_groups) - 1, max(1, round(len(unique_groups) * fraction)))
        validation_groups = set(unique_groups[:count])
        train_ids = sorted(key for key in by_id if groups[key] not in validation_groups)
        validation_ids = sorted(key for key in by_id if groups[key] in validation_groups)
        test_ids = []
    if not train_ids or (mode in ("train", "test") and not validation_ids):
        raise ValueError("formal training requires nonempty train and held-out validation sets")
    if {groups[key] for key in train_ids} & {groups[key] for key in validation_ids}:
        raise ValueError("split leaks the same original recording into train and validation")
    if {groups[key] for key in test_ids} & {groups[key] for key in train_ids + validation_ids}:
        raise ValueError("split leaks the same original recording into the test set")
    manifest = {"version": "real_robot_recording_split_v1", "seed": seed,
                "train_episode_ids": list(train_ids), "validation_episode_ids": list(validation_ids),
                "source_groups": groups}
    if test_ids:
        manifest.update(version="real_robot_recording_split_v2", test_episode_ids=list(test_ids),
                        seed=supplied.get("seed", seed))
        if "condition_groups" in supplied:
            manifest["condition_groups"] = supplied["condition_groups"]
    return tuple(by_id[key] for key in train_ids), tuple(by_id[key] for key in validation_ids), manifest


def validation_batches(dataset, batch_size, max_windows=None):
    size = len(dataset) if max_windows is None else min(len(dataset), max_windows)
    for start in range(0, size, batch_size):
        yield collate_high_rate_samples([dataset[i] for i in range(start, min(size, start + batch_size))])


def validate_output_identity(output_dir, metadata, resume_step):
    path = output_dir / "run_manifest.json"
    if not path.exists():
        return
    previous = json.loads(path.read_text())
    validate_contact_latent_identity(previous["run_config"], metadata["run_config"])
    for key in ("model_kind", "model_config", "data_config", "data_contract", "normalization",
                "split_manifest", "source_fingerprints", "training_config"):
        left, right = previous[key], metadata[key]
        if key == "model_config":
            left, right = canonical_model_config(left), canonical_model_config(right)
        if canonical(left) != canonical(right):
            raise ValueError(f"existing output directory belongs to another run ({key})")
    for key in ("mode", "batch_size", "seed", "sampling"):
        if previous["run_config"][key] != metadata["run_config"][key]:
            raise ValueError(f"existing output directory differs in {key}")
    for key in ("evaluation_config", "early_stopping_config", "validation_objectives"):
        if canonical(previous["run_config"].get(key)) != canonical(metadata["run_config"].get(key)):
            raise ValueError(f"existing output directory differs in {key}")
    metrics_path = output_dir / "metrics.jsonl"
    if metrics_path.exists():
        last_step = 0
        with metrics_path.open() as stream:
            for line in stream:
                if line.strip():
                    last_step = max(last_step, int(json.loads(line)["step"]))
        if last_step > resume_step:
            raise ValueError("output contains steps newer than the resume checkpoint; choose a new output directory")


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--data-root", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--mode", choices=("inspect", "smoke", "test", "train"), default="inspect",
                        help="test: held-out full-network check, explicit budget up to 300 updates")
    result.add_argument("--model", choices=MODEL_KINDS, default="contact")
    result.add_argument("--data-config", type=Path)
    result.add_argument("--force-representation", choices=("joint_torque7", "flange_force3", "end_effector_wrench6", "end_effector_force3"),
                        help="Select joint torque, derived flange force3, or raw_v3 sensor wrench6/force3")
    result.add_argument("--optimization-config", type=Path)
    result.add_argument("--split", type=Path, help="JSON with train_episode_ids and validation_episode_ids")
    result.add_argument("--resume", type=Path)
    result.add_argument("--contact-latent-mode", choices=CONTACT_LATENT_MODES,
                        help="validation/selection only; training always samples the posterior. "
                             "prior_mean also logs a zero-latent reference; default: saved mode or zero")
    result.add_argument("--steps", type=int, help="total optimizer steps, including previously saved steps")
    result.add_argument("--max-sampling-epochs", type=int, help="episodic epoch budget; alternative to --steps")
    result.add_argument("--minimum-sampling-epochs", type=int, default=500)
    result.add_argument("--early-stop-patience", type=int, default=0, help="validation checks; 0 disables early stopping")
    result.add_argument("--early-stop-min-relative-improvement", type=float, default=0.005)
    result.add_argument("--early-stop-metrics", choices=("action", "multi"),
                        help="multi: stop only when every validation objective stalls and save each best; "
                             "default: action for new runs, saved mode for resume")
    result.add_argument("--validation-anchor-stride", type=int, default=1)
    result.add_argument("--selection-metric", choices=("action_mae_rad", "episode_macro_action_mae_rad"), default="action_mae_rad")
    result.add_argument("--record-timing", action="store_true")
    result.add_argument("--batch-size", type=int, default=1)
    result.add_argument("--chunk-len", type=int, default=100)
    result.add_argument("--force-prediction-steps", type=int,
                        help="future force intervals; defaults to action chunk length, or saved value on resume")
    result.add_argument("--strict-force-horizon", action=argparse.BooleanOptionalAction, default=None,
                        help="cap source timestamps at the force horizon even when action and force lengths match")
    result.add_argument("--seed", type=int, default=0)
    result.add_argument("--train-sampling-mode", choices=TRAIN_SAMPLING_MODES,
                        help="episodic_uniform: one random valid anchor per episode per epoch; "
                             "default: legacy anchor-uniform sampling, or the saved mode on resume")
    result.add_argument("--validation-fraction", type=float, default=0.2)
    result.add_argument("--validation-interval", type=int, default=1000)
    result.add_argument("--checkpoint-interval", type=int, default=1000)
    result.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    result.add_argument("--threads", type=int, default=4)
    result.add_argument("--compact", action="store_true", help="small network, full input resolution; automatic in smoke mode")
    result.add_argument("--no-pretrained", action="store_true")
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    resolve_contact_latent_mode(args.model, args.contact_latent_mode)
    for name in ("batch_size", "chunk_len", "validation_interval", "checkpoint_interval", "threads", "validation_anchor_stride"):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive")
    if args.model == "act" and args.strict_force_horizon is not None:
        raise ValueError("ACT has no force prediction horizon")
    if args.force_prediction_steps is not None:
        if args.model == "act" or not 1 <= args.force_prediction_steps <= args.chunk_len:
            raise ValueError("force-prediction-steps requires a force model and a value between 1 and chunk-len")
    if not 0 < args.validation_fraction < 1:
        raise ValueError("validation fraction must be between zero and one")
    if args.steps is not None and args.steps <= 0:
        raise ValueError("steps must be positive")
    if args.max_sampling_epochs is not None and args.max_sampling_epochs <= 0:
        raise ValueError("max sampling epochs must be positive")
    if args.steps is not None and args.max_sampling_epochs is not None:
        raise ValueError("choose either --steps or --max-sampling-epochs")
    if args.mode == "train" and args.steps is None and args.max_sampling_epochs is None:
        raise ValueError("formal training requires an explicit --steps or --max-sampling-epochs budget")
    if args.mode == "test" and (args.steps is None or args.steps > 300 or args.max_sampling_epochs is not None):
        raise ValueError("test mode requires an explicit --steps budget of at most 300 updates")
    stopping_config = EarlyStoppingConfig(args.minimum_sampling_epochs, args.early_stop_patience,
                                         args.early_stop_min_relative_improvement)
    if args.mode == "inspect" and args.resume:
        raise ValueError("inspect mode does not resume a training checkpoint")
    args.data_root = args.data_root.resolve()
    args.output_dir = args.output_dir.resolve()
    if (args.output_dir / "run_manifest.json").exists() and args.resume is None:
        raise ValueError("output already contains a run; choose a new directory or pass --resume")
    torch.set_num_threads(args.threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    data_config = RealRobotDataConfig.from_dict(json.loads(args.data_config.read_text())) if args.data_config else RealRobotDataConfig()
    if args.force_representation is not None:
        data_config = replace(data_config, force_representation=args.force_representation)
    if args.model == "act" and data_config.force_representation == "flange_force3":
        raise ValueError("ACT has no force input; choose contact/motion/dual-zero for flange_force3")
    records = discover_real_robot_episodes(args.data_root, data_config)
    if not records:
        raise ValueError("no episodes matching the selected raw schema found")
    diagnostics = {record.episode_id: RawEpisode(record.resolve(args.data_root), data_config).diagnostics for record in records}
    inspection = {"data_contract": data_config.contract_dict(), "episodes": diagnostics}
    print(json.dumps(inspection, ensure_ascii=False, allow_nan=False), flush=True)
    if args.mode == "inspect":
        write_json(args.output_dir / "data_quality.json", inspection)
        return inspection

    payload = read_checkpoint(args.resume) if args.resume else None
    contact_mode = resolve_contact_latent_mode(
        args.model, args.contact_latent_mode, payload["run_config"] if payload else None,
        resume=payload is not None)
    metric_mode = resolve_metric_mode(args.early_stop_metrics, payload["run_config"] if payload else None)
    objectives = (validation_objectives(args.model, data_config.force_representation, args.selection_metric)
                  if metric_mode == "multi" else None)
    if objectives and args.mode == "smoke":
        raise ValueError("multi-metric selection requires held-out validation; use test or train mode")
    if payload and args.force_prediction_steps is None:
        args.force_prediction_steps = payload["model_config"].get("force_prediction_steps")
    if payload and args.strict_force_horizon is None:
        args.strict_force_horizon = payload["model_config"].get("strict_force_horizon", False)
    sampling_mode = resolve_training_sampling_mode(
        args.train_sampling_mode, checkpoint_sampling=payload["run_config"]["sampling"] if payload else None)
    if (args.max_sampling_epochs is not None or stopping_config.patience) and sampling_mode != EPISODIC_UNIFORM_SAMPLING:
        raise ValueError("epoch budgets and early stopping require episodic_uniform")
    if stopping_config.patience and args.mode != "train":
        raise ValueError("early stopping requires held-out formal training")
    supplied_split = json.loads(args.split.read_text()) if args.split else (payload["split_manifest"] if payload else None)
    train_records, val_records, split = choose_split(
        records, source_groups(args.data_root, records), mode=args.mode, seed=args.seed,
        fraction=args.validation_fraction, supplied=supplied_split)
    settings = RealRobotOptimizationConfig(**json.loads(args.optimization_config.read_text())) if args.optimization_config else RealRobotOptimizationConfig()
    compact = args.compact or args.mode == "smoke"
    config = make_model_config(args.model, data_config, compact=compact, chunk_len=args.chunk_len,
                               pretrained=not args.no_pretrained, force_prediction_steps=args.force_prediction_steps,
                               strict_force_horizon=bool(args.strict_force_horizon))
    # ACT normalizes images inside its model. Its data proxy must deliver RGB/255.
    data_model_config = config if args.model != "act" else make_model_config(
        "contact", data_config, compact=compact, chunk_len=args.chunk_len, pretrained=False)
    normalization = compute_real_robot_normalization_stats(args.data_root, train_records, data_config)
    train_data = RealRobotHighRateDataset(args.data_root, train_records, normalization, data_model_config, data_config)
    valid_anchor_counts = [len(episode.valid_anchor_indices) for episode in train_data.raw_episodes]
    sampler = RealRobotTrainingSampler(valid_anchor_counts, mode=sampling_mode, seed=args.seed)
    validation_data = (RealRobotHighRateDataset(args.data_root, val_records, normalization, data_model_config, data_config)
                       if val_records else None)
    fingerprints = {r.episode_id: source_fingerprint(r.resolve(args.data_root)) for r in records}
    steps_per_epoch = (len(train_records) + args.batch_size - 1) // args.batch_size
    steps = args.max_sampling_epochs * steps_per_epoch if args.max_sampling_epochs is not None else (args.steps or 2)
    validation_plan = None
    extended_evaluation = args.validation_anchor_stride != 1 or args.selection_metric != "action_mae_rad"
    if extended_evaluation:
        if validation_data is None:
            raise ValueError("fixed per-episode validation requires a held-out validation set")
        validation_plan = fixed_indices(validation_data, args.validation_anchor_stride)
        frozen_validation = index_manifest(validation_data, validation_plan, args.validation_anchor_stride)
    run_config = {"mode": args.mode, "steps": steps, "batch_size": args.batch_size, "seed": args.seed,
                  "sampling": sampler.contract_version, "train_sampling_mode": sampling_mode,
                  "train_episode_valid_anchors": {r.episode_id: count for r, count in zip(train_records, valid_anchor_counts)},
                  "selection_metric": f"held_out_validation.{args.selection_metric}", "action_loss_reduction": "valid_scalars",
                  "normalization_source": "training_recordings_native_streams_once",
                  "uses_qvel": False, "synthetic_terminal_hold": False}
    if args.model == "contact":
        run_config["contact_latent_config"] = contact_latent_contract(contact_mode)
    if extended_evaluation:
        run_config["evaluation_config"] = {"stride": args.validation_anchor_stride, "selection_metric": args.selection_metric,
                                          "indices_sha256": frozen_validation["indices_sha256"]}
    if stopping_config.patience or objectives:
        run_config["early_stopping_config"] = asdict(stopping_config)
    if objectives:
        run_config["validation_objectives"] = objectives
    if args.max_sampling_epochs is not None:
        run_config["max_sampling_epochs"] = args.max_sampling_epochs
    if sampling_mode == EPISODIC_UNIFORM_SAMPLING:
        run_config.update(sampling_epoch_windows=len(train_records),
                          steps_per_sampling_epoch=(len(train_records) + args.batch_size - 1) // args.batch_size,
                          sampling_epoch_last_batch="partial_no_cross_epoch_fill")
    metadata = {"model_kind": args.model, "model_config": asdict(config), "data_config": data_config.to_dict(),
                "data_contract": data_config.contract_dict(), "normalization": normalization.to_dict(),
                "split_manifest": split, "source_fingerprints": fingerprints,
                "training_config": asdict(settings), "run_config": run_config}
    if args.model != "act":
        metadata["prediction_horizons"] = {
            "action_steps": config.chunk_len,
            "action_duration_s": config.chunk_len / data_config.action_rate_hz,
            "force_history_duration_s": data_config.history_duration_s,
            "force_history_interval_s": data_config.history_period_ns / 1e9,
            "force_history_intervals": data_config.history_intervals,
            "force_history_sample_capacity": data_config.history_sample_capacity,
            "future_force_interval_s": 1.0 / data_config.action_rate_hz,
            "future_force_sample_capacity": data_config.max_samples_per_interval,
            "force_prediction_steps": config.force_prediction_len,
            "force_prediction_duration_s": config.force_prediction_len / data_config.action_rate_hz,
            "force_nominal_samples": config.force_prediction_len * data_config.samples_per_action,
            "contact_posterior": ("full_action_plus_full_force" if config.force_prediction_len == config.chunk_len
                                  else "full_action_plus_force_prefix") if args.model == "contact" else "not_used",
            "force_pooling": ["mean", "max", "last"],
            "force_source_upper_bound": ("anchor_plus_force_horizon" if config.bounds_force_source_timestamps
                                         else "legacy_matching_tolerance"),
        }
    validate_output_identity(args.output_dir, metadata, payload["global_step"] if payload else 0)
    device = torch.device(args.device)
    model = build_model(args.model, asdict(config), initialize_pretrained=payload is None).to(device)
    criterion = build_criterion(args.model, settings)
    optimizer = build_optimizer(model, settings)
    start = restore_checkpoint(payload, model=model, optimizer=optimizer, sampler=sampler, expected=metadata) if payload else 0
    stopping = (restore_multi_metric_controller(payload) if payload and objectives else
                MultiMetricController(stopping_config, objectives) if objectives else
                EarlyStoppingController(stopping_config))
    if stopping_config.patience and payload and not objectives:
        stopping.load_state_dict(payload["early_stopping_state"])
    if stopping.stopped:
        raise ValueError("checkpoint already satisfied early stopping")
    if start >= steps:
        raise ValueError("--steps must exceed the checkpoint's completed steps")
    write_json(args.output_dir / "data_quality.json", inspection)
    write_json(args.output_dir / "run_manifest.json", metadata)
    if extended_evaluation:
        write_json(args.output_dir / "validation_indices.json", frozen_validation)
    write_json(args.output_dir / "sample_traces.json", [train_data.sample_trace(index)
               for index in sorted({0, len(train_data) // 2, len(train_data) - 1})])
    metrics_file = args.output_dir / "metrics.jsonl"
    last_metrics = None
    best_metric = payload.get("best_metric") if payload else None
    completed = start
    train_seconds, read_seconds, validation_seconds = 0.0, 0.0, 0.0
    if args.record_timing and device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    def checkpoint_metadata():
        result = {**metadata, "best_metric": best_metric}
        if stopping_config.patience:
            result["early_stopping_state"] = stopping.state_dict()
        if objectives:
            result["multi_metric_state"] = stopping.state_dict()
        return result

    def validation_for_mode(mode):
        evaluation_data = validation_data if validation_data is not None else train_data
        if extended_evaluation:
            return evaluate_episodes(args.model, model, validation_data, validation_plan,
                                     normalization, device, args.batch_size, contact_latent_mode=mode)
        return evaluate(args.model, model, validation_batches(
            evaluation_data, args.batch_size, max_windows=2 if args.mode == "smoke" else None),
            normalization, device, force_representation=data_config.force_representation,
            contact_latent_mode=mode)

    for step in range(start + 1, steps + 1):
        tick = time.perf_counter()
        indices = sampler.sample_batch(args.batch_size)
        batch = collate_high_rate_samples([train_data[index] for index in indices]).to(device)
        read_time = time.perf_counter() - tick
        tick = time.perf_counter()
        metrics = train_step(args.model, model, criterion, optimizer, batch, settings)
        train_time = time.perf_counter() - tick
        train_seconds += train_time
        read_seconds += read_time
        row = {"step": step, "train": metrics, "sample_indices": indices,
               "sample_episode_ids": [train_records[train_data.index[index][0]].episode_id for index in indices],
               "sample_anchor_grid_indices": [train_data.index[index][1] for index in indices],
               "batch_size_actual": len(indices)}
        if sampling_mode == EPISODIC_UNIFORM_SAMPLING:
            row.update(sampling_epoch=sampler.epoch, sampling_epoch_complete=sampler.epoch_complete)
        if args.record_timing:
            row.update(data_seconds=read_time, train_seconds=train_time)
        if step % args.validation_interval == 0 or step == steps:
            tick = time.perf_counter()
            evaluated = validation_for_mode(contact_mode)
            if args.model == "contact":
                row["validation_contact_latent_mode"] = contact_mode
            if contact_mode == "prior_mean":
                # Separate namespace: reference metrics never enter selection/stopping.
                row["zero_latent_reference"] = validation_for_mode("zero")
                row["zero_latent_reference_role"] = "validation" if validation_data is not None else "same_recording_smoke"
            validation_seconds += time.perf_counter() - tick
            # A same-recording smoke check is explicitly not a held-out estimate.
            row["held_out_validation" if validation_data is not None else "same_recording_smoke_check"] = evaluated
            selected = evaluated[args.selection_metric]
            if objectives:
                stopping.update(evaluated, completed_epochs=sampler.completed_epochs, step=step)
                best_metric = stopping.best["action"]["value"]
                row["validation_objective_values"] = dict(stopping.values)
                row["best_checkpoints"] = {name: dict(best) for name, best in stopping.best.items()}
                if stopping_config.patience:
                    row["early_stopping"] = stopping.state_dict()
                for name in stopping.improved_metrics:
                    selection = {"objective": name, **stopping.best[name]}
                    if args.model == "contact":
                        selection["contact_latent_mode"] = contact_mode
                    save_checkpoint(args.output_dir / selection["checkpoint"], model=model, optimizer=optimizer,
                                    sampler=sampler, global_step=step,
                                    metadata={**checkpoint_metadata(), "checkpoint_selection": selection})
            else:
                if stopping_config.patience:
                    stopping.update(selected, completed_epochs=sampler.completed_epochs, step=step)
                    row["early_stopping"] = stopping.state_dict()
                if validation_data is not None and (best_metric is None or selected < best_metric):
                    best_metric = selected
                    save_checkpoint(args.output_dir / "best.pt", model=model, optimizer=optimizer,
                                    sampler=sampler, global_step=step, metadata=checkpoint_metadata())
        with metrics_file.open("a") as stream:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        print(json.dumps(row, ensure_ascii=False, allow_nan=False), flush=True)
        if step % args.checkpoint_interval == 0 or step == steps or stopping.stopped:
            save_checkpoint(args.output_dir / "last.pt", model=model, optimizer=optimizer,
                            sampler=sampler, global_step=step, metadata=checkpoint_metadata())
        last_metrics = row
        completed = step
        if stopping.stopped:
            break
    # Verify save/load, strict metadata, weights and the next sampled batch without
    # allocating a second full network on the GPU.
    saved = read_checkpoint(args.output_dir / "last.pt")
    probe = RealRobotTrainingSampler(valid_anchor_counts, mode=sampling_mode, seed=args.seed)
    restored_step = restore_checkpoint(saved, model=model, optimizer=optimizer, sampler=probe, expected=metadata)
    if restored_step != completed or not sampler_states_equal(probe.get_state(), sampler.get_state()):
        raise RuntimeError("checkpoint round trip failed")
    summary = {"mode": args.mode, "model": args.model, "completed_steps": completed,
               "step_budget": steps, "termination": "early_stopping" if stopping.stopped else "step_budget",
               "train_sampling_mode": sampling_mode, "train_episodes": len(train_records),
               "train_anchors": len(train_data), "validation_anchors": len(validation_data) if validation_data else 0,
               "checkpoint_round_trip": "passed", "final_metrics": last_metrics,
               "checkpoint": str(args.output_dir / "last.pt")}
    if objectives:
        summary["best_checkpoints"] = stopping.best
        summary["multi_metric_state"] = stopping.state_dict()
    if sampling_mode == EPISODIC_UNIFORM_SAMPLING:
        summary.update(sampling_epoch=sampler.epoch, completed_sampling_epochs=sampler.completed_epochs)
    if args.record_timing:
        summary["timing"] = {"training_seconds": train_seconds, "data_seconds": read_seconds,
                             "validation_seconds": validation_seconds,
                             "seconds_per_training_step": (train_seconds + read_seconds) / (completed - start),
                             "peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
                             "peak_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None}
    write_json(args.output_dir / "summary.json", summary)
    train_data.close()
    if validation_data is not None:
        validation_data.close()
    return summary


if __name__ == "__main__":
    main()
