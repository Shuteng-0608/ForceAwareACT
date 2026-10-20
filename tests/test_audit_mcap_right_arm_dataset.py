"""MCAP integrity and pre-conversion alignment must match the real reader."""

import importlib.util
import json
from pathlib import Path
import struct
import subprocess
import sys

import cv2
import h5py
import numpy as np
import pytest
pytest.importorskip("mcap_ros2.writer")
from mcap_ros2.writer import Writer

from force_aware_act.real_robot.reader import RawEpisode


def _module(filename, name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parents[1]/filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _audit():
    return _module("scripts/audit_mcap_right_arm_dataset.py", "mcap_audit_test")


def _fixture(tmp_path):
    helper = _module("tests/test_real_robot_data.py", "audit_raw_fixture")
    path = helper._write_raw(tmp_path/"raw", missing_action_ms=(50,),
                             missing_disturbance_ms=(50,), disturbance_delay_ms=2)
    return path, helper._config()


def test_audit_alignment_matches_reader_indices_masks_phase_and_capacity(tmp_path):
    path, config = _fixture(tmp_path)
    raw = RawEpisode(path, config)
    metrics, index = _audit().alignment_metrics(raw.timestamps, config)
    np.testing.assert_array_equal(index["candidate_anchor_log_ns"], raw.anchor_timestamps_ns)
    np.testing.assert_array_equal(index["valid_anchor_grid_indices"], raw.valid_anchor_indices)
    np.testing.assert_array_equal(index["action_source_indices"], raw.action_slot_indices)
    assert metrics["action_grid"] == raw.diagnostics["action_grid"]
    assert metrics["excluded_anchor_reasons"] == raw.diagnostics["excluded_anchor_reasons"]
    assert metrics["next_interval_force_slot_rate"] < 1
    times = dict(raw.timestamps)
    times["disturbance"] = np.r_[times["disturbance"], times["disturbance"][-1]]
    with pytest.raises(ValueError, match="strictly increasing"):
        _audit().alignment_metrics(times, config)


def _encoded_source(tmp_path, nonce=""):
    path, config = _fixture(tmp_path)
    audit = _audit()
    source = tmp_path/"demo_x13y08.mcap"
    header_def = """\n================================================================================
MSG: std_msgs/Header
builtin_interfaces/Time stamp
string frame_id
================================================================================
MSG: builtin_interfaces/Time
int32 sec
uint32 nanosec
"""
    writer = Writer(str(source))
    joints = writer.register_msgdef("sensor_msgs/msg/JointState",
        "std_msgs/Header header\nstring[] name\nfloat64[] position\nfloat64[] velocity\nfloat64[] effort\n"+header_def)
    images = writer.register_msgdef("sensor_msgs/msg/CompressedImage",
        "std_msgs/Header header\nstring format\nuint8[] data\n"+header_def)
    text = writer.register_msgdef("std_msgs/msg/String", "string data")
    writer.write_message("unselected_metadata", text, {"data": "audit all topics"+nonce}, log_time=1)
    with h5py.File(path, "r") as handle:
        for stream, topic in audit.STREAM_TOPICS.items():
            logs = handle[f"timestamps/{stream}/log_ns"][:]
            headers = handle[f"timestamps/{stream}/header_ns"][:]
            for i, (log, stamp) in enumerate(zip(logs, headers)):
                message = {"header": {"stamp": {"sec": int(stamp)//10**9,
                    "nanosec": int(stamp)%10**9}, "frame_id": "right"}}
                if stream in audit.FIELDS:
                    message["name"] = list(reversed(config.joint_names))
                    for field in ("position", "velocity", "effort"):
                        mapping = audit.FIELDS[stream].get(field)
                        message[field] = handle[mapping][i][::-1].tolist() if mapping and mapping in handle else [0.]*7
                    schema = joints
                else:
                    _, jpg = cv2.imencode(".jpg", handle[f"observations/images/{stream}/rgb"][i][:,:,::-1])
                    message.update(format="jpeg", data=jpg.tolist())
                    schema = images
                writer.write_message(topic, schema, message, log_time=int(log), publish_time=int(log))
    writer.finish()
    identity = audit.fingerprint(source)
    row = {"sha256": identity["sha256"], "content_length": str(identity["bytes"]),
           "bytes_downloaded": str(identity["bytes"]), "crc64_match": "True"}
    return audit, source, row, config


def test_complete_mcap_audit_decodes_all_images_and_preserves_source(tmp_path):
    audit, source, row, config = _encoded_source(tmp_path)
    result = audit.audit_recording(source, row, tmp_path/"evidence", config.to_dict())
    assert result["status"] == "passed", result["errors"]
    assert result["integrity"]["all_records_read"]
    assert result["integrity"]["complete_message_count"] > sum(result["selected_message_counts"].values())
    assert result["images"]["d455"]["decoded_frames"] == 13
    assert result["images"]["d405"]["decoded_frames"] == 12
    assert len(result["images"]["d455"]["samples"]) == 3
    assert audit.fingerprint(source)["sha256"] == row["sha256"]


def test_download_identity_failure_blocks_conversion_eligibility(tmp_path):
    audit, source, row, config = _encoded_source(tmp_path)
    row["sha256"] = "0"*64
    result = audit.audit_recording(source, row, tmp_path/"evidence", config.to_dict())
    assert result["status"] == "failed"
    assert "SHA-256" in result["errors"][0]


def test_local_files_do_not_inherit_download_verification_or_success_labels(tmp_path):
    audit, source, _, config = _encoded_source(tmp_path)
    result = audit.audit_recording(source, None, tmp_path/"evidence", config.to_dict(), group="x14y08")
    assert result["status"] == "passed", result["errors"]
    assert result["group"] == "x14y08"
    assert result["task_success"] is None
    assert "2026-10-03" not in result["task_success_provenance"]
    assert result["integrity"]["current_sha256_matches_download"] is None
    assert result["integrity"]["current_size_matches_download"] is None
    assert result["integrity"]["download_crc64_match_recorded"] is None
    assert result["integrity"]["all_advertised_chunk_crcs_passed"]
    result = audit.audit_recording(source, None, tmp_path/"explicit", config.to_dict(),
        task_success=False, task_success_provenance="synthetic failed trial")
    assert result["task_success"] is False
    assert result["task_success_provenance"] == "synthetic failed trial"


def test_local_cli_accepts_condition_without_manifest_and_keeps_unknown_label(tmp_path):
    audit, source, _, config = _encoded_source(tmp_path/"source")
    config_path = tmp_path/"config.json"
    audit.write_json(config_path, config.to_dict())
    subprocess.run([sys.executable, str(Path(__file__).parents[1]/"scripts/audit_mcap_right_arm_dataset.py"),
        "--source-root", str(source.parent), "--output-dir", str(tmp_path/"audit"),
        "--data-config", str(config_path), "--expected-episodes", "1", "--local-files", "--group", "x14y08", "--workers", "1"],
        check=True, capture_output=True, text=True)
    result = json.loads((tmp_path/"audit/audit_manifest.json").read_text())
    assert result["status"] == "passed"
    assert result["download_manifest_sha256"] is None
    assert result["episodes"][0]["task_success"] is None
    assert result["episodes"][0]["group"] == "x14y08"
    assert result["duplicate_source_sha256"] == []


def test_complete_crc_failure_is_detected_even_when_download_hash_is_updated(tmp_path):
    audit, source, row, config = _encoded_source(tmp_path)
    data = bytearray(source.read_bytes())
    offset = 8
    while offset+9 < len(data):
        opcode, length = data[offset], struct.unpack_from("<Q", data, offset+1)[0]
        if opcode == 6:  # Chunk: start/end/size precede uncompressed CRC32.
            data[offset+9+24] ^= 1
            break
        offset += 9+length
    else:
        raise AssertionError("fixture lacks MCAP chunk")
    source.write_bytes(data)
    row["sha256"] = audit.fingerprint(source)["sha256"]
    result = audit.audit_recording(source, row, tmp_path/"evidence", config.to_dict())
    assert result["status"] == "failed"
    assert "CRC" in result["errors"][0]
