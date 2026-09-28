"""Verify force provenance against an encoded MCAP with different stream clocks."""

import importlib.util
from pathlib import Path

import h5py
import numpy as np
import pytest

pytest.importorskip("mcap_ros2.writer")
from mcap_ros2.writer import Writer
import cv2


HEADER = """\n================================================================================
MSG: std_msgs/Header
builtin_interfaces/Time stamp
string frame_id
================================================================================
MSG: builtin_interfaces/Time
int32 sec
uint32 nanosec
"""


def _converter():
    path = Path(__file__).parents[1] / "scripts/convert_mcap_right_arm.py"
    spec = importlib.util.spec_from_file_location("right_arm_conversion_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _mcap(path, *, include_disturbance=True, incomplete_effort=False):
    module = _converter()
    writer = Writer(str(path))
    joint = writer.register_msgdef("sensor_msgs/msg/JointState",
        "std_msgs/Header header\nstring[] name\nfloat64[] position\nfloat64[] velocity\nfloat64[] effort\n" + HEADER)
    image = writer.register_msgdef("sensor_msgs/msg/CompressedImage",
        "std_msgs/Header header\nstring format\nuint8[] data\n" + HEADER)
    names = list(reversed(module.JOINTS))
    header = {"stamp": {"sec": 1, "nanosec": 123}, "frame_id": "right"}
    topics = [(module.STATE, [0, 3_333_333, 6_666_667], 100),
              (module.ACTION, [0, 10_000_000], 200)]
    if include_disturbance:
        topics.append((module.DISTURBANCE, [2_000_000, 5_333_333], 300))
    for topic, offsets, base in topics:
        for index, offset in enumerate(offsets):
            effort = [base + int(name[5]) + index / 10 for name in names]
            if incomplete_effort and topic == module.DISTURBANCE:
                effort = []
            writer.write_message(topic, joint, {"header": header, "name": names,
                "position": [0.] * 7, "velocity": [1.] * 7, "effort": effort},
                log_time=1_100_000_000 + offset, publish_time=1_090_000_000 + offset, sequence=index)
    _, jpg = cv2.imencode(".jpg", np.full((2, 3, 3), 70, np.uint8))
    for topic in module.CAMERAS:
        for offset in (0, 33_333_333):
            writer.write_message(topic, image, {"header": header, "format": "jpeg", "data": jpg.tolist()},
                                 log_time=1_100_000_000 + offset)
    writer.finish()
    return module


def test_conversion_and_source_verification_preserve_distinct_effort_streams(tmp_path):
    source, output = tmp_path / "recording.mcap", tmp_path / "episode.hdf5"
    module = _mcap(source)
    original_hash = module.sha256(source)
    _, counts = module.convert(source, output, original_hash)
    assert counts["state"] == 3 and counts["disturbance"] == 2
    verified = module.verify_against_source(source, output, counts)
    assert verified["status"] == "passed"
    assert module.sha256(source) == original_hash
    with h5py.File(output, "r") as handle:
        assert handle.attrs["schema_version"] == "real_robot_right_arm_raw_v2"
        np.testing.assert_array_equal(handle["observations/joint_internal_torque"][0], np.arange(101, 108))
        np.testing.assert_array_equal(handle["observations/joint_external_torque"][0], np.arange(301, 308))
        assert handle["observations/joint_external_torque"].attrs["timestamp_group"] == "/timestamps/disturbance"
        assert handle["observations/joint_external_torque"].attrs["source_topic"] == module.DISTURBANCE
        assert handle["timestamps/disturbance/log_ns"][0] == 1_102_000_000
        assert handle["timestamps/state/log_ns"][0] == 1_100_000_000
        assert "gravity_compensated" not in handle.attrs
    assert module.audit(output)["numeric"]["observations/joint_external_torque"]["nonfinite_count"] == 0


@pytest.mark.parametrize("damage", ["missing_stream", "missing_effort"])
def test_converter_refuses_missing_external_estimate_instead_of_using_internal_effort(tmp_path, damage):
    source = tmp_path / "recording.mcap"
    module = _mcap(source, include_disturbance=damage != "missing_stream",
                   incomplete_effort=damage == "missing_effort")
    with pytest.raises(ValueError, match="Missing required stream disturbance|incomplete effort"):
        module.convert(source, tmp_path / "episode.hdf5", module.sha256(source))
