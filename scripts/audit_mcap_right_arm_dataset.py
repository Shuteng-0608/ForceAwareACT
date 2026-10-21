#!/usr/bin/env python3
"""Audit complete MCAP records before publishing any training conversion."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import cv2
from mcap.reader import make_reader
from mcap.records import DataEnd, Message
from mcap.stream_reader import StreamReader
import numpy as np

from convert_mcap_right_arm import (ACTION, CAMERAS, FIELDS, JOINTS, RATES,
                                   STREAM_TOPICS, selected_messages)
from force_aware_act.real_robot.config import RealRobotDataConfig
from force_aware_act.real_robot.sampling import (estimate_action_grid_phase_ns,
    future_torque_slot_indices, match_time_slots, matching_error_summary)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def fingerprint(path):
    path = Path(path)
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(f"source changed while hashing: {path}")
    return {"sha256": digest.hexdigest(), "bytes": after.st_size}


def quantiles(values):
    values = np.asarray(values)
    if not values.size:
        return None
    return dict(zip(("min", "p50", "p95", "p99", "max"),
                    map(float, np.quantile(values, [0, .5, .95, .99, 1]))))


def clock_quality(rows, rate):
    result = {"count": len(rows), "nominal_hz": rate, "clocks": {}}
    for clock in ("header_ns", "log_ns", "publish_ns"):
        times = np.asarray([r[clock] for r in rows], dtype=np.int64)
        dt = np.diff(times)
        duration = (int(times[-1]) - int(times[0])) * 1e-9
        gaps = np.flatnonzero(dt > round(1.5e9 / rate))
        result["clocks"][clock] = {
            "duration_s": duration, "average_hz": (len(times)-1)/duration if duration > 0 else None,
            "duplicates": int((dt == 0).sum()), "backwards": int((dt < 0).sum()),
            "interval_ms": quantiles(dt * 1e-6), "gap_count_over_1_5_period": len(gaps),
            "gap_after_indices": gaps.tolist(), "gap_duration_ms": (dt[gaps]*1e-6).tolist(),
        }
    result["log_minus_header_ms"] = quantiles(
        np.asarray([r["log_ns"] - r["header_ns"] for r in rows], dtype=np.int64) * 1e-6)
    return result


def alignment_metrics(times, config):
    """Apply the reader's time rules; compare exact indices again after conversion."""
    for stream, values in times.items():
        if len(values) < 2 or np.any(np.diff(values) <= 0):
            raise ValueError(f"{stream}: log_ns must contain strictly increasing messages")
    action_times = times["action"]
    phase = estimate_action_grid_phase_ns(action_times, config.action_period_ns,
                                         round(config.action_grid_phase_window_s * 1e9))
    anchors = np.arange(int(action_times[0]) + phase["phase_ns"], int(action_times[-1]) + 1,
                        config.action_period_ns, dtype=np.int64)
    if not len(anchors):
        raise ValueError("no candidate anchors")
    slots = match_time_slots(action_times, anchors, round(config.action_match_tolerance_ms * 1e6))

    def ages(stream):
        index = np.searchsorted(times[stream], anchors, side="right") - 1
        age = anchors - times[stream][np.maximum(index, 0)]
        return index, age

    remaining = np.ones(len(anchors), dtype=bool)
    reasons = {}
    conditions = [("missing_current_action", slots >= 0)]
    state_index, state_age = ages("state")
    conditions.append(("stale_or_missing_state", (state_index >= 0) & (state_age <= config.state_max_age_ns)))
    camera_ok = np.ones(len(anchors), dtype=bool)
    age_stats = {"state": quantiles(state_age[state_index >= 0] * 1e-6)}
    for camera in config.camera_names:
        index, age = ages(camera)
        camera_ok &= (index >= 0) & (age <= config.camera_max_age_ns)
        age_stats[camera] = quantiles(age[index >= 0] * 1e-6)
    conditions.append(("stale_or_missing_camera", camera_ok))
    force_times = times["disturbance"]
    history_lo = np.searchsorted(force_times, anchors-round(config.history_duration_s*1e9), side="right")
    history_hi = np.searchsorted(force_times, anchors, side="right")
    history_count = history_hi-history_lo
    conditions.append(("empty_torque_history", history_count > 0))
    for reason, ok in conditions:
        reasons[reason] = int((remaining & ~ok).sum())
        remaining &= ok
    valid = np.flatnonzero(remaining)
    edges = (anchors[:, None] - config.history_intervals * config.history_period_ns
             + np.arange(config.history_intervals + 1, dtype=np.int64) * config.history_period_ns)
    bin_counts = np.diff(np.searchsorted(force_times, edges, side="right"))
    maximum = int(bin_counts[valid].max()) if len(valid) else 0
    if maximum > config.history_sample_capacity:
        raise ValueError(f"torque history capacity exceeded: {maximum}>{config.history_sample_capacity}")
    force_slots, force_targets = future_torque_slot_indices(
        force_times, int(anchors[0]), len(anchors), config)
    selected = force_slots[valid]
    report = {
        "action_grid": {**phase, "timestamp_key": config.timestamp_key,
            "first_anchor_ns": int(anchors[0]), "last_anchor_ns": int(anchors[-1])},
        "candidate_anchors": len(anchors), "valid_anchors": len(valid),
        "valid_anchor_rate": len(valid)/len(anchors), "excluded_anchor_reasons": reasons,
        "action_matching": matching_error_summary(action_times, anchors, slots),
        "observation_age_ms": age_stats,
        "history_sample_count": quantiles(history_count[valid]), "max_samples_per_history_interval": maximum,
        "next_interval_force_slots": int(selected.size),
        "next_interval_force_valid_slots": int((selected >= 0).sum()),
        "next_interval_force_slot_rate": float((selected >= 0).mean()) if selected.size else 0.,
        "next_interval_endpoint_valid": int((selected[:, -1] >= 0).sum()) if len(valid) else 0,
        "next_interval_endpoint_rate": float((selected[:, -1] >= 0).mean()) if len(valid) else 0.,
        "force_matching": matching_error_summary(force_times, force_targets.reshape(-1), force_slots.reshape(-1)),
        "force_coverage_scope": "one next 10ms interval per valid anchor; includes recording boundaries",
    }
    return report, {"candidate_anchor_log_ns": anchors, "valid_anchor_grid_indices": valid,
                    "action_source_indices": slots, "next_interval_force_source_indices": force_slots}


def audit_recording(source, download_row, destination, config_dict, *, group=None,
                    task_success=None, task_success_provenance="not confirmed for this recording"):
    cv2.setNumThreads(1)
    source, destination = Path(source), Path(destination)
    config = RealRobotDataConfig.from_dict(config_dict)
    result = {"episode_id": source.stem, "source_path": str(source),
              "group": group or source.stem.rsplit("_", 1)[-1], "status": "failed", "errors": [], "findings": [],
              "task_success": task_success, "task_success_provenance": task_success_provenance}
    try:
        identity = fingerprint(source)
        result.update(source_fingerprint=identity)
        if task_success is not None and type(task_success) is not bool:
            raise ValueError("task_success must be True, False or None")
        if download_row is not None and identity["sha256"] != download_row["sha256"]:
            raise ValueError("SHA-256 differs from download manifest")
        if download_row is not None and (identity["bytes"] != int(download_row["bytes_downloaded"]) or identity["bytes"] != int(download_row["content_length"])):
            raise ValueError("size differs from download manifest")
        with source.open("rb") as stream:
            summary = make_reader(stream).get_summary()
        if summary is None or summary.statistics is None:
            raise ValueError("required MCAP summary missing")
        inventory = [{"channel_id": c.id, "topic": c.topic, "schema": summary.schemas[c.schema_id].name,
                      "count": summary.statistics.channel_message_counts.get(c.id, 0)} for c in summary.channels.values()]
        result["inventory"] = inventory
        for topic in STREAM_TOPICS.values():
            if not any(r["topic"] == topic and r["count"] for r in inventory):
                raise ValueError(f"required topic absent or empty: {topic}")
        # Read the whole file, including unselected topics, validating every
        # advertised chunk CRC and a nonzero DataEnd data-section CRC.
        actual_counts = Counter()
        data_crc_present = False
        with source.open("rb") as stream:
            for record in StreamReader(stream, validate_crcs=True).records:
                if isinstance(record, Message):
                    actual_counts[record.channel_id] += 1
                elif isinstance(record, DataEnd):
                    data_crc_present = record.data_section_crc != 0
        expected_counts = {r["channel_id"]: r["count"] for r in inventory if r["count"]}
        if dict(actual_counts) != expected_counts or sum(actual_counts.values()) != summary.statistics.message_count:
            raise ValueError("complete MCAP message counts differ from summary")
        result["integrity"] = {"complete_message_count": sum(actual_counts.values()),
            "all_records_read": True, "all_advertised_chunk_crcs_passed": True,
            "data_section_crc_present": data_crc_present, "data_section_crc_passed": True if data_crc_present else None,
            "current_sha256_matches_download": True if download_row is not None else None,
            "current_size_matches_download": True if download_row is not None else None,
            "download_crc64_match_recorded": download_row.get("crc64_match", "").lower() == "true" if download_row is not None else None,
            "identity_reference": "download_manifest" if download_row is not None else "current_local_file_only"}
        counts = Counter()
        clocks = {stream: [] for stream in RATES}
        numeric = {stream: {field: [] for field in fields} for stream, fields in FIELDS.items()}
        camera_counts = {name: sum(r["count"] for r in inventory if r["topic"] == topic) for topic, name in CAMERAS.items()}
        image_data = {camera: {"stats": [], "hashes": set(), "previous_hash": None,
            "run_start_ns": None, "longest_identical_run_s": 0., "samples": [], "shape": None,
            "black_or_white_count": 0} for camera in CAMERAS.values()}
        destination.mkdir(parents=True, exist_ok=False)
        for stream, indices, decoded, times in selected_messages(source, counts):
            clocks[stream].append(times)
            if stream in FIELDS:
                for field in FIELDS[stream]:
                    numeric[stream][field].append([getattr(decoded, field)[i] for i in indices])
                continue
            raw = bytes(decoded.data)
            bgr = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
            if bgr is None:
                raise ValueError(f"{stream}: image decode failure at {counts[stream]-1}")
            expected_shape = (config.image_height, config.image_width, 3)
            if bgr.shape != expected_shape:
                raise ValueError(f"{stream}: image shape {bgr.shape} differs from {expected_shape}")
            evidence = image_data[stream]
            evidence["shape"] = list(bgr.shape)
            small = cv2.resize(bgr, (64, 48), interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            mean, std = float(gray.mean()), float(gray.std())
            evidence["black_or_white_count"] += int(std < 1 and (mean < 2 or mean > 253))
            evidence["stats"].append((mean, std, float((gray < 5).mean()), float((gray > 250).mean()),
                                      float(cv2.Laplacian(gray, cv2.CV_64F).var())))
            digest = hashlib.sha256(raw).hexdigest()
            evidence["hashes"].add(digest)
            if digest != evidence["previous_hash"]:
                evidence["run_start_ns"] = times["log_ns"]
            evidence["longest_identical_run_s"] = max(evidence["longest_identical_run_s"],
                (times["log_ns"] - evidence["run_start_ns"]) * 1e-9)
            evidence["previous_hash"] = digest
            index = counts[stream]-1
            if index in {0, camera_counts[stream]//2, camera_counts[stream]-1}:
                name = f"{stream}_{index:06d}.png"
                if not cv2.imwrite(str(destination/name), cv2.resize(bgr, (256, 192), interpolation=cv2.INTER_AREA)):
                    raise ValueError("failed to save image evidence")
                evidence["samples"].append({"index": index, "log_ns": times["log_ns"],
                                            "header_ns": times["header_ns"], "thumbnail": str(destination/name)})
        result["streams"] = {}
        for stream, rate in RATES.items():
            if not clocks[stream]:
                raise ValueError(f"missing selected stream {stream}")
            result["streams"][stream] = clock_quality(clocks[stream], rate)
        result["numeric"] = {}
        for stream, fields in FIELDS.items():
            for field, path in fields.items():
                values = np.asarray(numeric[stream][field], dtype=np.float64)
                if not np.isfinite(values).all():
                    raise ValueError(f"{path}: NaN/Inf")
                result["numeric"][path] = {"shape": list(values.shape), "nonfinite_count": 0,
                    "min_per_joint": values.min(axis=0).tolist(), "max_per_joint": values.max(axis=0).tolist(),
                    "std_per_joint": values.std(axis=0).tolist(),
                    "max_adjacent_change_per_joint": np.abs(np.diff(values, axis=0)).max(axis=0).tolist() if len(values)>1 else [0.]*7}
        result["images"] = {}
        for camera, evidence in image_data.items():
            measurements = np.asarray(evidence["stats"])
            result["images"][camera] = {"decoded_frames": len(measurements), "decode_failures": 0,
                "shape": evidence["shape"], "unique_compressed_frames": len(evidence["hashes"]),
                "longest_identical_compressed_run_s": evidence["longest_identical_run_s"],
                "black_or_white_frame_count": evidence["black_or_white_count"],
                "metrics_at_64x48": {name: quantiles(measurements[:, i]) for i, name in enumerate(
                    ("gray_mean", "gray_std", "dark_fraction", "bright_fraction", "laplacian_variance"))},
                "samples": evidence["samples"]}
            if evidence["black_or_white_count"] or evidence["longest_identical_run_s"] > 1:
                result["findings"].append(f"{camera}: inspect blank/frozen-frame indicators")
        times = {s: np.asarray([r[config.timestamp_key] for r in rows], dtype=np.int64) for s, rows in clocks.items()}
        alignment, index = alignment_metrics(times, config)
        result["alignment"] = alignment
        if alignment["valid_anchor_rate"] < .95:
            raise ValueError("valid anchor rate below the declared 95% usability gate")
        if alignment["next_interval_force_slot_rate"] < .90:
            result["findings"].append("next-interval force slot coverage below 90%; retain missing-label masks")
        for stream, quality in result["streams"].items():
            if quality["clocks"]["header_ns"]["backwards"]:
                result["findings"].append(f"{stream}: header clock reversals; log clock is used")
        np.savez_compressed(destination/"sampling_index.npz", **index)
        if fingerprint(source) != identity:
            raise ValueError("source changed during audit")
        result["selected_message_counts"] = dict(counts)
        result["status"] = "passed_with_findings" if result["findings"] else "passed"
    except Exception as error:
        result["errors"].append(f"{type(error).__name__}: {error}")
    destination.mkdir(parents=True, exist_ok=True)
    write_json(destination/"audit.json", result)
    return result


def summarize(episodes):
    accepted = [r for r in episodes if r["status"].startswith("passed")]
    summary = {"episodes": len(episodes), "accepted_episodes": len(accepted),
        "failed_episodes": len(episodes)-len(accepted), "with_findings": sum(bool(r["findings"]) for r in accepted),
        "source_bytes": sum(r.get("source_fingerprint", {}).get("bytes", 0) for r in episodes)}
    if accepted:
        candidates = sum(r["alignment"]["candidate_anchors"] for r in accepted)
        valid = sum(r["alignment"]["valid_anchors"] for r in accepted)
        slots = sum(r["alignment"]["next_interval_force_slots"] for r in accepted)
        summary.update(candidate_anchors=candidates, valid_anchors=valid, valid_anchor_rate=valid/candidates,
            next_interval_force_slot_rate=sum(r["alignment"]["next_interval_force_valid_slots"] for r in accepted)/slots,
            next_interval_endpoint_rate=sum(r["alignment"]["next_interval_endpoint_valid"] for r in accepted)/valid,
            total_action_duration_s=sum(r["streams"]["action"]["clocks"]["log_ns"]["duration_s"] for r in accepted),
            selected_counts={s: sum(r["selected_message_counts"][s] for r in accepted) for s in RATES},
            decoded_images=sum(r["images"][c]["decoded_frames"] for r in accepted for c in CAMERAS.values()),
            full_mcap_messages=sum(r["integrity"]["complete_message_count"] for r in accepted))
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-config", type=Path, default=REPO/"configs/real_robot/right_arm.json")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--expected-episodes", type=int, default=50)
    parser.add_argument("--success-confirmed", action="store_true")
    parser.add_argument("--success-provenance", default=None)
    parser.add_argument("--local-files", action="store_true", help="audit local files without claiming download-manifest verification")
    parser.add_argument("--group", help="explicit recording condition when filenames have no condition suffix")
    args = parser.parse_args(argv)
    if args.workers <= 0:
        raise ValueError("workers must be positive")
    source_root, output = args.source_root.resolve(), args.output_dir.resolve()
    if output.exists():
        raise ValueError("audit destination already exists; preserve previous audit")
    if args.success_provenance and not args.success_confirmed:
        raise ValueError("success provenance requires explicit --success-confirmed")
    if args.local_files:
        rows = [{"saved_filename": p.name} for p in sorted(source_root.glob("*.mcap"))]
    else:
        with (source_root/"download_manifest.csv").open(encoding="utf-8-sig") as stream:
            rows = list(csv.DictReader(stream))
    filenames = [r["saved_filename"] for r in rows]
    actual = {p.name for p in source_root.glob("*.mcap")}
    if len(rows) != args.expected_episodes or len(set(filenames)) != len(rows) or set(filenames) != actual:
        raise ValueError("file inventory differs from the expected unique download manifest")
    if not args.local_files and len({r["sha256"] for r in rows}) != len(rows):
        raise ValueError("duplicate source contents require deduplication before split")
    for row in rows:
        if Path(row["saved_filename"]).name != row["saved_filename"] or (not args.group and not re.search(r"_x\d+y\d+\.mcap$", row["saved_filename"])):
            raise ValueError("unexpected recording filename")
        if not args.local_files and (row["status"] != "verified" or any(row.get(k, "").lower() != "true" for k in ("size_match", "crc64_match", "mcap_magic_valid"))):
            raise ValueError("download manifest contains an unverified file")
    config = RealRobotDataConfig.from_dict(json.loads(args.data_config.read_text()))
    output.mkdir(parents=True)
    policy = {"hard_gates": ["local SHA-256 stable during audit" if args.local_files else "download SHA-256 and size match", "complete MCAP parse and all advertised CRCs pass",
        "required topics and complete right-arm fields", "all selected numeric values finite",
        "all log clocks strictly increasing", "all images decode at configured resolution",
        "stable action phase resultant >=0.5", "history interval capacity <=8", "valid anchor rate >=95%"],
        "review_findings": ["blank/frozen image indicators", "force slot coverage below90%", "header clock reversal"],
        "gaps": "reported and masked; no interpolation, clock correction, filtering or sample deletion",
        "success_label": "explicitly confirmed; not inferred from MCAP" if args.success_confirmed else "unknown; not inferred from MCAP"}
    report = {"version": "real_robot_mcap_dataset_audit_v1", "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_root": str(source_root), "output_dir": str(output), "status": "auditing",
        "audit_policy": policy, "data_config": config.to_dict(), "data_contract": config.contract_dict(),
        "download_manifest_sha256": None if args.local_files else fingerprint(source_root/"download_manifest.csv")["sha256"],
        "code_sha256": {str(p.relative_to(REPO)): fingerprint(p)["sha256"] for p in (Path(__file__),
            REPO/"scripts/convert_mcap_right_arm.py", REPO/"src/force_aware_act/real_robot/sampling.py",
            REPO/"src/force_aware_act/real_robot/reader.py")}, "episodes": []}
    write_json(output/"audit_manifest.json", report)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(audit_recording, str(source_root/r["saved_filename"]), None if args.local_files else r,
                    str(output/"episodes"/Path(r["saved_filename"]).stem), config.to_dict(), group=args.group,
                    task_success=True if args.success_confirmed else None,
                    task_success_provenance=(args.success_provenance or "explicit --success-confirmed for this audit")
                        if args.success_confirmed else "not confirmed for this recording") for r in rows]
        for future in as_completed(futures):
            episode = future.result()
            report["episodes"].append(episode)
            write_json(output/"audit_manifest.json", report)
            print(json.dumps({"completed": len(report["episodes"]), "episode": episode["episode_id"],
                "status": episode["status"], "errors": episode["errors"],
                "valid_anchors": episode.get("alignment", {}).get("valid_anchors")}), flush=True)
    report["episodes"].sort(key=lambda r: r["episode_id"])
    report["summary"] = summarize(report["episodes"])
    hashes = Counter(r["source_fingerprint"]["sha256"] for r in report["episodes"] if "source_fingerprint" in r)
    report["duplicate_source_sha256"] = [sha for sha, count in hashes.items() if count > 1]
    report["status"] = "passed" if not report["summary"]["failed_episodes"] and not report["duplicate_source_sha256"] else "failed"
    report["completed_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(output/"audit_manifest.json", report)
    print(json.dumps({"status": report["status"], **report["summary"]}), flush=True)
    return report


if __name__ == "__main__":
    main()
