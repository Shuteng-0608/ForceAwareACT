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


def _mcap(path, *, include_disturbance=True, incomplete_effort=False, include_ft=False, ft_damage=None):
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
    if include_ft:
        for index, offset in enumerate((1_000_000, 4_333_333)):
            axes = list(reversed(module.FT_AXES))
            canonical = [1.25 + index, -2.5, 3.75, -0.0, 0.25, -0.5]
            effort = list(reversed(canonical))
            if ft_damage == "axes":
                axes[0] = axes[1]
            if ft_damage == "nonfinite":
                effort[0] = float("nan")
            writer.write_message(module.FT_SENSOR, joint,
                {"header": {"stamp": {"sec": 1, "nanosec": offset}, "frame_id": "right_ft_measurement"},
                 "name": axes, "position": [], "velocity": [], "effort": effort},
                log_time=1_100_000_000+offset, publish_time=1_095_000_000+offset, sequence=index+20)
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


def test_sensor_conversion_preserves_six_axes_and_independent_clocks_without_recompensation(tmp_path):
    source, output = tmp_path/"sensor.mcap", tmp_path/"episode.hdf5"
    module = _mcap(source, include_ft=True)
    report = module.convert_verified(source, output, include_ft_sensor=True,
        ft_gravity_compensated=True, ft_semantics_provenance="synthetic user confirmation")
    assert report["schema_version"] == module.SENSOR_SCHEMA
    assert report["verification"]["ft_sensor_included"]
    assert report["verification"]["counts"]["ft_sensor"] == 2
    with h5py.File(output, "r") as h:
        assert h.attrs["force_source_topic"] == module.FT_SENSOR
        assert h.attrs["gravity_compensation"] == "applied_upstream"
        assert not h.attrs["additional_gravity_compensation_applied"]
        assert not h.attrs["training_ready"]
        assert list(h["metadata/ee_wrench_axes"].asstr()[:]) == module.FT_AXES
        expected = np.array([1.25, -2.5, 3.75, -0.0, .25, -.5], dtype=np.float64)
        assert h[module.FT_DATASET][0].tobytes() == expected.tobytes()
        assert h[module.FT_DATASET].shape == (2, 6)
        assert h["observations/joint_external_torque"].shape == (2, 7)
        assert h["observations/joint_internal_torque"].shape == (3, 7)
        assert h["timestamps/ft_sensor/log_ns"][0] == 1_101_000_000
        assert h["timestamps/ft_sensor/header_ns"][0] == 1_001_000_000
        assert h["timestamps/ft_sensor/publish_ns"][0] == 1_096_000_000
        assert h["timestamps/ft_sensor/sequence"][0] == 20
        assert h["timestamps/ft_sensor/frame_id"].asstr()[0] == "right_ft_measurement"
    assert report["quality"]["numeric"][module.FT_DATASET]["nonfinite_count"] == 0


@pytest.mark.parametrize("damage", ["missing", "axes", "nonfinite"])
def test_sensor_mode_refuses_missing_or_malformed_sensor_without_fallback(tmp_path, damage):
    source = tmp_path/"sensor.mcap"
    module = _mcap(source, include_ft=damage != "missing", ft_damage=damage)
    with pytest.raises(ValueError, match="Missing required stream ft_sensor|FT axes|six finite"):
        module.convert(source, tmp_path/"episode.hdf5", module.sha256(source), include_ft_sensor=True)


@pytest.mark.parametrize("damage", ["value", "timestamp", "provenance"])
def test_second_source_read_detects_sensor_corruption(tmp_path, damage):
    source, output = tmp_path/"sensor.mcap", tmp_path/"episode.hdf5"
    module = _mcap(source, include_ft=True)
    _, counts = module.convert(source, output, module.sha256(source), include_ft_sensor=True)
    with h5py.File(output, "r+") as h:
        if damage == "value":
            h[module.FT_DATASET][0, 0] += 1
        elif damage == "timestamp":
            h["timestamps/ft_sensor/header_ns"][0] += 1
        else:
            h.attrs["force_source_topic"] = module.DISTURBANCE
    with pytest.raises(ValueError, match="Numeric byte mismatch|Timestamp/provenance mismatch|FT provenance"):
        module.verify_against_source(source, output, counts)


def test_changed_reviewed_source_is_not_published(tmp_path):
    source, output = tmp_path/"sensor.mcap", tmp_path/"episode.hdf5"
    module = _mcap(source, include_ft=True)
    with pytest.raises(ValueError, match="changed since reviewed"):
        module.convert_verified(source, output, include_ft_sensor=True, expected_source_sha256="0"*64)
    assert not output.exists()
