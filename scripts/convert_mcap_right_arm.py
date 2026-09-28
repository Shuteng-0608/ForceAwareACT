#!/usr/bin/env python3
"""Lossless right-arm extraction; no resampling, imputation or training adaptation.

Requires numpy, h5py, opencv-python, mcap and mcap-ros2-support.
The completed HDF5 is published only after a second full source comparison.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import cv2
import h5py
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory
import numpy as np


SCHEMA = "real_robot_right_arm_raw_v2"
JOINTS = [f"Joint{i}_R" for i in range(1, 8)]
CAMERAS = {
    "/d455/d455/color/image_raw/compressed": "d455",
    "/d405/d405/color/image_raw/compressed": "d405",
}
STATE = "io_teleop/joint_states"
ACTION = "io_teleop/joint_cmd"
DISTURBANCE = "/tianji_marvin_teleop_bridge/estimated_joint_disturbance"
FIELDS = {
    "state": {
        "position": "observations/joint_pos",
        "velocity": "observations/joint_vel",
        "effort": "observations/joint_internal_torque",
    },
    "disturbance": {"effort": "observations/joint_external_torque"},
    "action": {
        "position": "action",
        "velocity": "actions/source_velocity",
        "effort": "actions/source_effort",
    },
}
RATES = {"state": 300, "disturbance": 300, "action": 100, "d455": 30, "d405": 30}
STREAM_TOPICS = {"state": STATE, "action": ACTION, "disturbance": DISTURBANCE,
                 **{name: topic for topic, name in CAMERAS.items()}}
STR = h5py.string_dtype("utf-8")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def selected_messages(path, counters):
    """Keep MCAP file order; select joints by name on every individual message."""
    topic_indices = Counter()
    with path.open("rb") as source:
        reader = make_reader(source, decoder_factories=[DecoderFactory()], validate_crcs=True)
        for _, channel, message, decoded in reader.iter_decoded_messages(
            topics=[STATE, ACTION, DISTURBANCE, *CAMERAS], log_time_order=False
        ):
            topic = channel.topic
            topic_index = topic_indices[topic]
            topic_indices[topic] += 1
            if topic in CAMERAS:
                stream = CAMERAS[topic]
                indices = None
            else:
                names = list(decoded.name)
                if not set(JOINTS).intersection(names):
                    if topic != ACTION:
                        raise ValueError(f"{topic}: message lacks right-arm joint names")
                    counters["excluded_non_right_arm_command_messages"] += 1
                    continue
                if not set(JOINTS).issubset(names) or len(set(names)) != len(names):
                    raise ValueError(f"Incomplete/duplicate joint names: {names}")
                indices = [names.index(joint) for joint in JOINTS]
                stream = {STATE: "state", ACTION: "action", DISTURBANCE: "disturbance"}[topic]
                for field in FIELDS[stream]:
                    if len(getattr(decoded, field)) != len(names):
                        raise ValueError(f"{topic}: incomplete {field}; no values fabricated")
            times = {
                "header_ns": decoded.header.stamp.sec * 10**9 + decoded.header.stamp.nanosec,
                "log_ns": message.log_time,
                "publish_ns": message.publish_time,
                "sequence": message.sequence,
                "source_topic_message_index": topic_index,
                "frame_id": decoded.header.frame_id,
            }
            counters[stream] += 1
            yield stream, indices, decoded, times


def put(handle, path, values, **kwargs):
    parent, _, name = path.rpartition("/")
    return handle.require_group(parent or "/").create_dataset(name, data=values, **kwargs)


def quantiles(values):
    return dict(zip(("min", "p50", "p95", "p99", "max"),
                    map(float, np.quantile(values, [0, .5, .95, .99, 1]))))


def convert(source, output, source_hash):
    with source.open("rb") as f:
        reader = make_reader(f)
        summary = reader.get_summary()
        if summary is None or summary.statistics is None:
            raise ValueError("This converter requires an MCAP summary with message counts")
        inventory = [
            {"topic": c.topic, "schema": summary.schemas[c.schema_id].name,
             "count": summary.statistics.channel_message_counts[c.id]}
            for c in summary.channels.values()
        ]
    camera_counts = {
        name: sum(row["count"] for row in inventory if row["topic"] == topic)
        for topic, name in CAMERAS.items()
    }
    counts = Counter()
    timestamps = {stream: [] for stream in RATES}
    numeric = {stream: {field: [] for field in mapping} for stream, mapping in FIELDS.items()}
    with h5py.File(output, "x") as h:
        h.attrs.update({
            "schema_version": SCHEMA, "source_path": str(source),
            "source_sha256": source_hash, "source_bytes": source.stat().st_size,
            "created_utc": datetime.now(timezone.utc).isoformat(), "arm": "right",
            "force_representation": "joint_external_torque",
            "force_source_topic": DISTURBANCE, "force_source_field": "effort",
            "force_timestamp_stream": "disturbance",
            "gravity_compensation": "unverified",
            "force_semantics_provenance": "User correction: estimated_joint_disturbance.effort is the external torque estimate; joint_states.effort is internal joint torque",
            "friction_compensation": "unknown", "resampled": False,
            "imputed": False, "time_order": "MCAP file order within each selected stream",
            "training_ready": False,
        })
        put(h, "metadata/joint_names", JOINTS, dtype=STR)
        put(h, "metadata/camera_names", list(CAMERAS.values()), dtype=STR)
        put(h, "metadata/topic_inventory_json", json.dumps(inventory), dtype=STR)
        for stream, indices, decoded, times in selected_messages(source, counts):
            timestamps[stream].append(times)
            if stream in FIELDS:
                for field in FIELDS[stream]:
                    numeric[stream][field].append([getattr(decoded, field)[i] for i in indices])
                continue
            raw = np.frombuffer(bytes(decoded.data), dtype=np.uint8)
            bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)
            if bgr is None:
                raise ValueError(f"Image decode failed in {stream}; conversion stopped")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            g = h.require_group(f"observations/images/{stream}")
            if "rgb" not in g:
                n = camera_counts[stream]
                g.create_dataset("rgb", shape=(n, *rgb.shape), dtype="u1",
                                 chunks=(1, *rgb.shape), compression="lzf", fletcher32=True)
                g.create_dataset("compressed", shape=(n,), dtype=h5py.vlen_dtype(np.dtype("u1")))
                g.create_dataset("format", shape=(n,), dtype=STR)
                g.attrs.update({"color_order": "RGB", "layout": "NHWC",
                                "compressed_payload": "Original bytes, no JPEG re-encoding"})
            i = counts[stream] - 1
            if rgb.shape != g["rgb"].shape[1:]:
                raise ValueError(f"Changing resolution in {stream}; no resizing performed")
            g["rgb"][i] = rgb
            g["compressed"][i] = raw
            g["format"][i] = decoded.format
        for stream, rows in timestamps.items():
            if not rows:
                raise ValueError(f"Missing required stream {stream}")
            if stream in camera_counts and len(rows) != camera_counts[stream]:
                raise ValueError(f"Camera count differs from MCAP summary: {stream}")
            g = h.require_group(f"timestamps/{stream}")
            g.attrs["nominal_rate_hz"] = RATES[stream]
            g.attrs["source_topic"] = STREAM_TOPICS[stream]
            for key in rows[0]:
                dtype = STR if key == "frame_id" else "i8" if key == "header_ns" else "u8"
                ds = g.create_dataset(key, data=[row[key] for row in rows], dtype=dtype)
                if key.endswith("_ns"):
                    ds.attrs["unit"] = "nanoseconds; original clock, no offset correction"
        for stream, mapping in FIELDS.items():
            for field, path in mapping.items():
                ds = put(h, path, np.asarray(numeric[stream][field], dtype=np.float64),
                         compression="lzf", fletcher32=True)
                ds.attrs["source_field"] = field
                ds.attrs["source_topic"] = STREAM_TOPICS[stream]
                ds.attrs["timestamp_group"] = f"/timestamps/{stream}"
                ds.attrs["unit"] = {"position": "rad", "velocity": "rad/s", "effort": "N*m"}[field]
                if field == "effort":
                    ds.attrs["signal_semantics"] = {"state": "internal_joint_torque; retained for audit, excluded from model inputs and labels",
                        "disturbance": "estimated_external_joint_torque", "action": "command_effort; retained for audit"}[stream]
        h.attrs["selected_message_counts_json"] = json.dumps(counts)
    return inventory, dict(counts)


def verify_against_source(source, output, expected_counts):
    counts = Counter()
    rgb_hashes = {camera: set() for camera in CAMERAS.values()}
    with h5py.File(output, "r") as h:
        for stream, indices, decoded, times in selected_messages(source, counts):
            i = counts[stream] - 1
            for key, value in times.items():
                ds = h[f"timestamps/{stream}/{key}"]
                actual = ds.asstr()[i] if key == "frame_id" else int(ds[i])
                if actual != value:
                    raise ValueError(f"Timestamp/provenance mismatch: {stream}/{key}/{i}")
            if stream in FIELDS:
                for field, path in FIELDS[stream].items():
                    expected = np.asarray([getattr(decoded, field)[j] for j in indices], dtype=np.float64)
                    if h[path][i].tobytes() != expected.tobytes():
                        raise ValueError(f"Numeric byte mismatch: {path}/{i}")
            else:
                g = h[f"observations/images/{stream}"]
                raw = bytes(decoded.data)
                if g["compressed"][i].tobytes() != raw or g["format"].asstr()[i] != decoded.format:
                    raise ValueError(f"Original compressed image mismatch: {stream}/{i}")
                rgb = cv2.cvtColor(cv2.imdecode(np.frombuffer(raw, dtype="u1"), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
                actual = g["rgb"][i]
                if not np.array_equal(actual, rgb):
                    raise ValueError(f"RGB pixel mismatch: {stream}/{i}")
                rgb_hashes[stream].add(hashlib.sha256(actual.tobytes()).digest())
        if dict(counts) != expected_counts:
            raise ValueError("Source/output message counts differ")
        for stream in RATES:
            if len(h[f"timestamps/{stream}/header_ns"]) != counts[stream]:
                raise ValueError(f"Unexpected output length in {stream}")
    return {"status": "passed", "all_selected_timestamps_exact": True,
            "all_selected_numeric_float64_bytes_exact": True,
            "all_original_jpeg_bytes_exact": True, "all_rgb_pixels_exact": True,
            "counts": dict(counts),
            "unique_decoded_rgb_frames": {k: len(v) for k, v in rgb_hashes.items()}}


def audit(output):
    report = {"streams": {}, "numeric": {}}
    with h5py.File(output, "r") as h:
        for stream, rate in RATES.items():
            g = h[f"timestamps/{stream}"]
            t = g["header_ns"][:]
            dt = np.diff(t) / 1e6
            long = np.flatnonzero(dt > 1500 / rate)
            duration = float((int(t[-1]) - int(t[0])) / 1e9)
            report["streams"][stream] = {
                "count": len(t), "nominal_hz": rate, "duration_s": duration,
                "average_hz": (len(t) - 1) / duration if duration > 0 else None,
                "header_duplicate_count": int((dt == 0).sum()),
                "header_backwards_count": int((dt < 0).sum()),
                "interval_ms": quantiles(dt),
                "log_minus_header_ms": quantiles((g["log_ns"][:].astype("i8") - t) / 1e6),
                "intervals_over_1_5_nominal_period": len(long),
                "gap_after_indices": long.tolist(),
                "gap_durations_ms": dt[long].tolist(),
                "gap_note": "Timing flags only; no samples inserted or removed; does not prove packet loss",
            }
            if stream in CAMERAS.values():
                report["streams"][stream]["rgb_shape"] = list(h[f"observations/images/{stream}/rgb"].shape)
        for mapping in FIELDS.values():
            for path in mapping.values():
                a = h[path][:]
                finite = np.isfinite(a)
                report["numeric"][path] = {
                    "shape": list(a.shape), "nonfinite_count": int((~finite).sum()),
                    "min_per_joint": a.min(axis=0).tolist() if finite.all() else None,
                    "max_per_joint": a.max(axis=0).tolist() if finite.all() else None,
                    "std_per_joint": a.std(axis=0).tolist() if finite.all() else None,
                }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    temporary = output.with_suffix(output.suffix + ".partial")
    report_path = output.with_name("quality_report.json")
    if any(p.exists() for p in [output, temporary, report_path]):
        parser.error("Output or report already exists; choose a new destination")
    output.parent.mkdir(parents=True, exist_ok=True)
    print("Hashing source...", flush=True)
    source_hash = sha256(source)
    print("Extracting native-rate streams...", flush=True)
    inventory, counts = convert(source, temporary, source_hash)
    print("Re-reading MCAP and checking every selected value, timestamp and image...", flush=True)
    verification = verify_against_source(source, temporary, counts)
    if sha256(source) != source_hash:
        raise ValueError("Source changed during conversion")
    quality = audit(temporary)
    with h5py.File(temporary, "r+") as h:
        h.attrs["source_comparison"] = "passed: complete second read"
    temporary.rename(output)
    report = {"source": str(source), "source_sha256": source_hash,
              "output": str(output), "output_bytes": output.stat().st_size,
              "schema_version": SCHEMA, "inventory": inventory,
              "verification": verification, "quality": quality}
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(output), "report": str(report_path),
                      "bytes": output.stat().st_size, "verification": verification}, indent=2))


if __name__ == "__main__":
    main()
