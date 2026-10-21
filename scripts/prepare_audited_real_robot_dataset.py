#!/usr/bin/env python3
"""Convert a passed MCAP audit, freeze splits/indices and training-only statistics."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO/"src"))
sys.path.insert(0, str(REPO/"scripts"))

import numpy as np

from audit_mcap_right_arm_dataset import fingerprint, write_json
from force_aware_act.real_robot.config import RealRobotDataConfig, RAW_SCHEMA_VERSION
from force_aware_act.real_robot.reader import RawEpisode, discover_real_robot_episodes
from force_aware_act.real_robot.normalization import compute_real_robot_normalization_stats


def balanced_recording_split(episodes, seed=20261003):
    """Every five-record condition contributes three train, one val, one test."""
    groups = {}
    ids, identities = set(), set()
    for row in episodes:
        if row["episode_id"] in ids or row["source_fingerprint"]["sha256"] in identities:
            raise ValueError("duplicate recording IDs or source contents would leak across splits")
        ids.add(row["episode_id"])
        identities.add(row["source_fingerprint"]["sha256"])
        groups.setdefault(row["group"], []).append(row["episode_id"])
    if not groups or any(len(values) != 5 for values in groups.values()):
        raise ValueError("each condition must contain exactly five accepted recordings")
    rng = random.Random(seed)
    result = {"version": "real_robot_recording_split_v2", "seed": seed,
        "train_episode_ids": [], "validation_episode_ids": [], "test_episode_ids": [],
        "condition_groups": {r["episode_id"]: r["group"] for r in episodes},
        "scope": "recording holdout within each of the ten observed positions; not unseen-position holdout"}
    for group in sorted(groups):
        values = sorted(groups[group])
        rng.shuffle(values)
        result["test_episode_ids"].append(values[0])
        result["validation_episode_ids"].append(values[1])
        result["train_episode_ids"].extend(values[2:])
    for role in ("train", "validation", "test"):
        result[f"{role}_episode_ids"].sort()
    return result


def require_passed_audit(audit):
    if audit["status"] != "passed" or not audit["episodes"]:
        raise ValueError("source audit must have passed before conversion")
    if any(not r["status"].startswith("passed") or r["errors"] for r in audit["episodes"]):
        raise ValueError("an individual recording failed its source audit")
    if any(not r.get("task_success") for r in audit["episodes"]):
        raise ValueError("this successful-demonstration preparation requires confirmed success labels")
    return balanced_recording_split(audit["episodes"])


def validate_data_batches(output_root, records, stats, config):
    """Exercise the existing data reader at start/middle/end without any model."""
    import torch
    from force_aware_act.act_aligned_training.high_rate_data import collate_high_rate_samples
    from force_aware_act.models.act_aligned.config import ACTAlignedHighRateConfig
    from force_aware_act.real_robot.data import RealRobotHighRateDataset

    torch.set_num_threads(4)
    model_config = ACTAlignedHighRateConfig(q_dim=7, action_dim=7, force_dim=7, chunk_len=100,
        image_height=config.image_height, image_width=config.image_width, pretrained_backbone=True,
        force_sample_rate_hz=config.torque_rate_hz, policy_sample_rate_hz=config.action_rate_hz,
        online_force_window_len=round(config.history_duration_s*config.torque_rate_hz),
        max_online_force_intervals=config.history_intervals,
        max_force_samples_per_interval=config.max_samples_per_interval,
        max_online_force_samples_per_interval=config.history_max_samples_per_interval,
        online_force_interval_duration_s=(config.history_period_ns / 1e9
                                         if config.history_interval_ms is not None else None))
    dataset = RealRobotHighRateDataset(output_root, records, stats, model_config, config)
    offset = 0
    chosen = []
    rows = []
    for record, raw in zip(records, dataset.raw_episodes):
        selected = sorted({0, len(raw.valid_anchor_indices)//2, len(raw.valid_anchor_indices)-1})
        for local_index in selected:
            index = offset+local_index
            trace = dataset.sample_trace(index)
            assert all(trace["observation"][key]["timestamp_ns"] <= trace["observation"]["anchor_ns"]
                       for key in ("state", *config.camera_names))
            rows.append({"episode": record.episode_id, "anchor_grid_index": dataset.index[index][1],
                "anchor_ns": trace["observation"]["anchor_ns"],
                "history_actual_sample_count": trace["history_actual_sample_count"],
                "valid_action_slots": sum(trace["action"]["valid_slots"]),
                "valid_force_endpoints": trace["future_torque"]["valid_endpoint_count"]})
            chosen.append(index)
        offset += len(raw.valid_anchor_indices)
    last_shapes = None
    for start in range(0, len(chosen), 2):
        batch = collate_high_rate_samples([dataset[i] for i in chosen[start:start+2]])
        batch.validate(model_config)
        tensors = (batch.images, batch.qpos, batch.action_chunk, batch.online_force_intervals,
                   batch.future_force_intervals)
        if any(not torch.isfinite(values).all() for values in tensors):
            raise ValueError("nonfinite normalized data batch")
        last_shapes = {"images": list(batch.images.shape), "qpos": list(batch.qpos.shape),
            "action": list(batch.action_chunk.shape), "history": list(batch.online_force_intervals.shape),
            "future_torque": list(batch.future_force_intervals.shape)}
    dataset.close()
    return {"status": "passed", "records_checked": len(records), "windows_checked": len(chosen),
        "start_middle_end_windows": rows, "batch_shapes": last_shapes,
        "model_instantiated": False, "optimizer_instantiated": False, "training_started": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dataset-config-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)
    if args.workers <= 0:
        raise ValueError("workers must be positive")
    audit_path = args.audit_manifest.resolve()
    audit = json.loads(audit_path.read_text())
    split = require_passed_audit(audit)
    config = RealRobotDataConfig.from_dict(audit["data_config"])
    if audit["data_contract"] != json.loads(json.dumps(config.contract_dict())):
        raise ValueError("audit data contract differs from the current reader")
    output, config_dir = args.output_root.resolve(), args.dataset_config_dir.resolve()
    if output.exists() or config_dir.exists():
        raise ValueError("choose new conversion and configuration directories; preserve existing versions")
    output.mkdir(parents=True)
    config_dir.mkdir(parents=True)
    write_json(config_dir/"split.json", split)
    write_json(config_dir/"data_config.json", config.to_dict())
    shutil.copyfile(audit_path, config_dir/"source_audit_manifest.json")
    manifest = {"version": "real_robot_audited_dataset_preparation_v1", "status": "preparing",
        "created_utc": datetime.now(timezone.utc).isoformat(), "source_root": audit["source_root"],
        "output_root": str(output), "source_audit_manifest": str(audit_path),
        "source_audit_manifest_sha256": fingerprint(audit_path)["sha256"],
        "raw_schema_version": RAW_SCHEMA_VERSION, "data_config": config.to_dict(),
        "data_contract": config.contract_dict(), "split_manifest": split, "episodes": [],
        "training_started": False,
        "code_sha256": {str(p.relative_to(REPO)): fingerprint(p)["sha256"] for p in (Path(__file__),
            REPO/"scripts/audit_mcap_right_arm_dataset.py", REPO/"scripts/convert_mcap_right_arm.py",
            REPO/"src/force_aware_act/real_robot/config.py", REPO/"src/force_aware_act/real_robot/reader.py",
            REPO/"src/force_aware_act/real_robot/sampling.py", REPO/"src/force_aware_act/real_robot/normalization.py",
            REPO/"src/force_aware_act/real_robot/data.py")}}
    with (output/"dataset_labels.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("episode_id", "group", "task_success", "label_provenance"))
        writer.writeheader()
        for row in audit["episodes"]:
            writer.writerow({"episode_id": row["episode_id"], "group": row["group"],
                "task_success": True, "label_provenance": row["task_success_provenance"]})
    write_json(output/"conversion_manifest.json", manifest)

    def prepare(row):
        start = time.monotonic()
        source = Path(row["source_path"])
        if fingerprint(source) != row["source_fingerprint"]:
            raise ValueError(f"source changed since audit: {source}")
        episode_dir = output/row["episode_id"]
        episode_dir.mkdir()
        hdf5 = episode_dir/"episode.hdf5"
        cmd = [sys.executable, "-u", "-c", "import cv2,runpy,sys;cv2.setNumThreads(1);"
               "p=sys.argv.pop(1);runpy.run_path(p,run_name='__main__')",
               str(REPO/"scripts/convert_mcap_right_arm.py"), str(source), str(hdf5)]
        with (episode_dir/"conversion.log").open("x") as log:
            subprocess.run(cmd, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, check=True)
        report = json.loads((episode_dir/"quality_report.json").read_text())
        verification = report["verification"]
        if (report["source_sha256"] != row["source_fingerprint"]["sha256"]
                or verification["status"] != "passed"
                or not all(verification[k] for k in ("all_selected_timestamps_exact",
                    "all_selected_numeric_float64_bytes_exact", "all_original_jpeg_bytes_exact", "all_rgb_pixels_exact"))):
            raise ValueError("complete source comparison did not pass")
        raw = RawEpisode(hdf5, config)
        audited_index = Path(audit["output_dir"])/"episodes"/row["episode_id"]/"sampling_index.npz"
        with np.load(audited_index) as index:
            for key, values in (("candidate_anchor_log_ns", raw.anchor_timestamps_ns),
                ("valid_anchor_grid_indices", raw.valid_anchor_indices),
                ("action_source_indices", raw.action_slot_indices)):
                if not np.array_equal(values, index[key]):
                    raise ValueError(f"converted reader index differs from pre-conversion audit: {key}")
        if raw.diagnostics["action_grid"] != row["alignment"]["action_grid"]:
            raise ValueError("phase metadata differs from pre-conversion audit")
        shutil.copyfile(audited_index, episode_dir/"sampling_index.npz")
        role = next(role for role in ("train", "validation", "test") if row["episode_id"] in split[f"{role}_episode_ids"])
        return {"episode_id": row["episode_id"], "group": row["group"], "task_success": True,
            "task_success_provenance": row["task_success_provenance"], "split": role,
            "source_path": str(source), "source_fingerprint": row["source_fingerprint"],
            "output": str(hdf5), "output_fingerprint": fingerprint(hdf5),
            "verification": verification, "audit_reader_indices_equal": True,
            "action_grid": raw.diagnostics["action_grid"], "valid_anchors": len(raw.valid_anchor_indices),
            "candidate_anchors": len(raw.anchor_timestamps_ns),
            "excluded_anchor_reasons": raw.diagnostics["excluded_anchor_reasons"],
            "preparation_seconds": time.monotonic()-start}

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(prepare, row) for row in audit["episodes"]]
            for future in as_completed(futures):
                result = future.result()
                manifest["episodes"].append(result)
                write_json(output/"conversion_manifest.json", manifest)
                print(json.dumps({"completed": len(manifest["episodes"]), "episode": result["episode_id"],
                    "valid_anchors": result["valid_anchors"], "bytes": result["output_fingerprint"]["bytes"]}), flush=True)
        manifest["episodes"].sort(key=lambda row: row["episode_id"])
        records = discover_real_robot_episodes(output, config)
        train = [r for r in records if r.episode_id in split["train_episode_ids"]]
        stats = compute_real_robot_normalization_stats(output, train, config)
        normalization = {"statistics": stats.to_dict(), "training_episode_ids": split["train_episode_ids"],
            "source": "training episodes only, each native qpos/action/external-disturbance message once",
            "minimum_std": .01, "internal_torque_included": False}
        write_json(config_dir/"normalization.json", normalization)
        write_json(output/"normalization.json", normalization)
        validation = validate_data_batches(output, records, stats, config)
        write_json(output/"data_loader_validation.json", validation)
        validation_indices = []
        for record in records:
            if record.episode_id not in split["validation_episode_ids"]:
                continue
            raw = RawEpisode(record.resolve(output), config)
            selected = sorted(set(range(0, len(raw.valid_anchor_indices), 10)) | {len(raw.valid_anchor_indices)-1})
            for local_index in selected:
                grid_index = int(raw.valid_anchor_indices[local_index])
                validation_indices.append({"episode_id": record.episode_id, "anchor_grid_index": grid_index,
                    "anchor_log_ns": int(raw.anchor_timestamps_ns[grid_index])})
        write_json(config_dir/"validation_indices_stride10.json", {"stride": 10, "includes_last_anchor": True,
            "windows": validation_indices, "scope": "frozen validation proposal, no model evaluation performed"})
        manifest.update(status="ready", completed_utc=datetime.now(timezone.utc).isoformat(),
            total_bytes=sum(r["output_fingerprint"]["bytes"] for r in manifest["episodes"]),
            split_valid_anchors={role: sum(r["valid_anchors"] for r in manifest["episodes"] if r["split"] == role)
                for role in ("train", "validation", "test")},
            normalization=str(config_dir/"normalization.json"), data_loader_validation=validation["status"],
            loader_windows_checked=validation["windows_checked"], frozen_validation_windows=len(validation_indices))
        write_json(output/"conversion_manifest.json", manifest)
        shutil.copyfile(output/"conversion_manifest.json", config_dir/"conversion_manifest.json")
        write_json(config_dir/"dataset.json", {"data_root": str(output), "raw_schema_version": RAW_SCHEMA_VERSION,
            "conversion_manifest": str(output/"conversion_manifest.json"),
            "conversion_manifest_sha256": fingerprint(output/"conversion_manifest.json")["sha256"],
            "source_audit": str(audit_path), "source_audit_sha256": fingerprint(audit_path)["sha256"],
            "data_config": str(config_dir/"data_config.json"), "split": str(config_dir/"split.json"),
            "normalization": str(config_dir/"normalization.json"), "status": "ready",
            "recommended_train_sampling_mode": "episodic_uniform", "training_started": False})
    except Exception as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        write_json(output/"conversion_manifest.json", manifest)
        raise
    print(json.dumps({"status": manifest["status"], "episodes": len(records),
        "split_valid_anchors": manifest["split_valid_anchors"], "total_bytes": manifest["total_bytes"],
        "data_loader_windows_checked": validation["windows_checked"], "training_started": False}), flush=True)
    return manifest


if __name__ == "__main__":
    main()
