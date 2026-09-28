#!/usr/bin/env python3
"""Convert and verify the approved 20-episode dataset, recording every source."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import h5py

from force_aware_act.real_robot.config import RealRobotDataConfig, RAW_SCHEMA_VERSION, EXTERNAL_TORQUE_TOPIC
from force_aware_act.real_robot.reader import RawEpisode


def source_fingerprint(path):
    """Hash data without importing model, optimizer or checkpoint code."""
    path = Path(path)
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"source changed while hashing: {path}")
    return {"sha256": digest.hexdigest(), "bytes": after.st_size}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def balanced_split(rows, seed=20260928):
    groups = {}
    for row in rows:
        groups.setdefault(row["group"], []).append(row)
    if len(groups) != 4 or any(len(values) != 5 for values in groups.values()):
        raise ValueError("expected four condition groups with five episodes each")
    rng = random.Random(seed)
    result = {"version": "real_robot_recording_split_v2", "seed": seed,
              "train_episode_ids": [], "validation_episode_ids": [], "test_episode_ids": [],
              "condition_groups": {r["dataset_name"]: r["group"] for r in rows}}
    for group in sorted(groups):
        values = sorted(groups[group], key=lambda r: r["dataset_name"])
        rng.shuffle(values)
        result["test_episode_ids"].append(values[0]["dataset_name"])
        result["validation_episode_ids"].append(values[1]["dataset_name"])
        result["train_episode_ids"].extend(r["dataset_name"] for r in values[2:])
    for key in ("train_episode_ids", "validation_episode_ids", "test_episode_ids"):
        result[key].sort()
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path("/home/stw/Downloads/1mm_gap_mcap"))
    parser.add_argument("--output-root", type=Path, default=Path("/home/stw/Downloads/1mm_gap_hdf5_raw_v2"))
    parser.add_argument("--experiment-dir", type=Path, default=REPO / "configs/experiments/1mm_gap_20_v1")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args(argv)
    if args.workers <= 0:
        raise ValueError("workers must be positive")
    source_root, output_root = args.source_root.resolve(), args.output_root.resolve()
    label_path = source_root / "dataset_labels.csv"
    with label_path.open() as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 20 or len({r["dataset_name"] for r in rows}) != 20:
        raise ValueError("expected exactly 20 unique labeled episodes")
    if any(r["task_success"].lower() != "true" for r in rows):
        raise ValueError("success labels do not match the approved dataset")
    phase_path = source_root / "alignment_processing_manifest_v2.json"
    phase = json.loads(phase_path.read_text())
    expected = {r["mcap_file"]: r for r in phase["files"]}
    if set(expected) != {r["mcap_file"] for r in rows}:
        raise ValueError("labels and approved alignment manifest differ")
    split = balanced_split(rows)
    config_path = REPO / "configs/real_robot/right_arm.json"
    config = RealRobotDataConfig.from_dict(json.loads(config_path.read_text()))
    output_root.mkdir(parents=True, exist_ok=True)
    copied_labels = output_root / "dataset_labels.csv"
    if copied_labels.exists() and copied_labels.read_bytes() != label_path.read_bytes():
        raise ValueError("existing output labels differ")
    if not copied_labels.exists():
        copied_labels.write_bytes(label_path.read_bytes())
    # Dataset preparation owns the recording split, not training hyperparameters.
    # Existing optimization/queue files are supplied by the training workflow.
    split_path = args.experiment_dir / "split.json"
    if split_path.exists() and json.loads(split_path.read_text()) != split:
        raise ValueError(f"existing experiment configuration differs: {split_path}")
    write_json(split_path, split)
    manifest_path = output_root / "conversion_manifest.json"
    manifest = {"version": "1mm_gap_conversion_v2", "status": "preparing", "episodes": [],
                "raw_schema_version": RAW_SCHEMA_VERSION, "data_contract": config.contract_dict(),
                "correction": {"reason": "User clarified external estimate source",
                    "external_torque": EXTERNAL_TORQUE_TOPIC + ".effort",
                    "internal_torque": "io_teleop/joint_states.effort; preserved, excluded from training",
                    "supersedes_data_root": "/home/stw/Downloads/1mm_gap_hdf5_raw_v1",
                    "old_force_model_results_valid": False},
                "source_root": str(source_root), "output_root": str(output_root),
                "label_sha256": hashlib.sha256(label_path.read_bytes()).hexdigest(),
                "approved_alignment_sha256": hashlib.sha256(phase_path.read_bytes()).hexdigest(),
                "data_config": config.to_dict(), "split_manifest": split,
                "code_sha256": {str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in (REPO / "scripts/convert_mcap_right_arm.py",
                                          REPO / "src/force_aware_act/real_robot/reader.py",
                                          REPO / "src/force_aware_act/real_robot/sampling.py")}}
    write_json(manifest_path, manifest)

    def prepare(row):
        start = time.monotonic()
        source = source_root / row["mcap_file"]
        reference = expected[row["mcap_file"]]
        if source_fingerprint(source)["sha256"] != reference["source_sha256"]:
            raise ValueError(f"source differs from approved audit: {source}")
        output = output_root / row["dataset_name"] / "episode.hdf5"
        report_path = output.with_name("quality_report.json")
        if not output.exists():
            output.parent.mkdir(parents=True, exist_ok=True)
            command = [sys.executable, "-u", "-c",
                       "import cv2,runpy,sys;cv2.setNumThreads(2);"
                       "p=sys.argv.pop(1);runpy.run_path(p,run_name='__main__')",
                       str(REPO / "scripts/convert_mcap_right_arm.py"), str(source), str(output)]
            with output.with_name("conversion.log").open("x") as log:
                subprocess.run(command, cwd=REPO, env=os.environ.copy(), stdout=log, stderr=subprocess.STDOUT, check=True)
        report = json.loads(report_path.read_text())
        verification = report["verification"]
        if (report["source_sha256"] != reference["source_sha256"] or verification["status"] != "passed"
                or not all(verification[key] for key in ("all_selected_timestamps_exact",
                            "all_selected_numeric_float64_bytes_exact", "all_original_jpeg_bytes_exact", "all_rgb_pixels_exact"))):
            raise ValueError(f"conversion verification failed: {output}")
        with h5py.File(output, "r") as handle:
            if handle.attrs["source_sha256"] != reference["source_sha256"]:
                raise ValueError("HDF5 source identity differs")
        raw = RawEpisode(output, config)
        if raw.diagnostics["action_grid"]["phase_ns"] != reference["phase_estimate"]["phase_ns"]:
            raise ValueError(f"production alignment differs from approved phase audit: {output}")
        role = next(role for role in ("train", "validation", "test")
                    if row["dataset_name"] in split[f"{role}_episode_ids"])
        return {**row, "split": role, "source_sha256": reference["source_sha256"],
                "output": str(output), "output_fingerprint": source_fingerprint(output),
                "valid_anchors": len(raw.valid_anchor_indices), "action_grid": raw.diagnostics["action_grid"],
                "previous_valid_anchors": reference["valid_anchors"],
                "valid_anchor_change": len(raw.valid_anchor_indices) - reference["valid_anchors"],
                "stream_diagnostics": raw.diagnostics["streams"],
                "excluded_anchor_reasons": raw.diagnostics["excluded_anchor_reasons"],
                "verification": verification, "preparation_seconds": time.monotonic() - start}

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(prepare, row) for row in rows]
        for future in as_completed(futures):
            result = future.result()
            manifest["episodes"].append(result)
            write_json(manifest_path, manifest)
            print(json.dumps({"completed": len(manifest["episodes"]), "episode": result["dataset_name"],
                              "valid_anchors": result["valid_anchors"], "bytes": result["output_fingerprint"]["bytes"]}), flush=True)
    manifest["episodes"].sort(key=lambda r: r["dataset_name"])
    manifest.update(status="ready", total_bytes=sum(r["output_fingerprint"]["bytes"] for r in manifest["episodes"]),
                    split_valid_anchors={role: sum(r["valid_anchors"] for r in manifest["episodes"] if r["split"] == role)
                                         for role in ("train", "validation", "test")})
    write_json(manifest_path, manifest)
    write_json(args.experiment_dir / "dataset.json", {"data_root": str(output_root),
               "conversion_manifest": str(manifest_path), "conversion_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
               "split": str((args.experiment_dir / "split.json").resolve())})
    print(json.dumps({"status": "ready", "total_bytes": manifest["total_bytes"],
                      "split_valid_anchors": manifest["split_valid_anchors"]}), flush=True)
    return manifest


if __name__ == "__main__":
    main()
