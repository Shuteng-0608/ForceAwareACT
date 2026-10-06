"""Real-message causality and action-clock tests without robot command output."""

from dataclasses import asdict
from pathlib import Path
import runpy

import h5py
import numpy as np
import pytest
import torch

from force_aware_act.inference.real_robot import (
    ActionUnavailableError,
    ObservationUnavailableError,
    RealRobotObservationBuffer,
    RealRobotPolicyAdapter,
    TimestampActionChunkBuffer,
    validate_data_contract,
)
from force_aware_act.real_robot.config import RealRobotDataConfig, RAW_SCHEMA_VERSION, EXTERNAL_TORQUE_TOPIC
from force_aware_act.real_robot.sampling import pack_torque_history


ORIGIN = 1_700_000_000_000_000_000
MS = 1_000_000


def _config(**values):
    return RealRobotDataConfig(image_height=32, image_width=32, **values)


def _stats():
    return {f"{name}_{suffix}": [value] * 7
            for name in ("qpos", "action", "force")
            for suffix, value in (("mean", 1.0), ("std", 2.0))}


def _buffer(config=None):
    config = config or _config()
    buffer = RealRobotObservationBuffer(config)
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    for name in config.camera_names:
        buffer.append_image(name, ORIGIN, image)
    return buffer


def test_actual_state_gaps_are_retained_and_future_messages_are_invisible():
    buffer = _buffer()
    for offset in (0, 3, 10, 20, 40):
        buffer.append_state(ORIGIN + offset * MS, np.full(7, offset),
                            np.full(7, offset + 100))
    buffer.append_image("d455", ORIGIN + 40 * MS, np.full((32, 32, 3), 255, np.uint8))
    observation = buffer.observation_at(ORIGIN + 20 * MS)
    np.testing.assert_array_equal(observation.qpos, np.full(7, 20))
    np.testing.assert_array_equal(observation.torque_timestamps_ns - ORIGIN,
                                  np.asarray([0, 3, 10, 20]) * MS)
    assert observation.image_timestamps_ns == (ORIGIN, ORIGIN)
    assert observation.images[0].max() == 0


def test_receive_availability_gates_late_messages_even_when_stamp_is_in_past():
    buffer = _buffer()
    buffer.append_state(ORIGIN, np.zeros(7), np.zeros(7))
    buffer.append_state(ORIGIN + 10 * MS, np.ones(7), np.ones(7),
                        available_ns=ORIGIN + 30 * MS)
    observation = buffer.observation_at(ORIGIN + 20 * MS)
    np.testing.assert_array_equal(observation.qpos, np.zeros(7))
    assert len(observation.torque_values) == 1
    assert buffer.observation_at(ORIGIN + 30 * MS).state_timestamp_ns == ORIGIN + 10 * MS


def test_independent_external_torque_arrival_does_not_use_state_rows_or_future_values():
    buffer = _buffer()
    for offset in (0, 10, 20):
        buffer.append_state(ORIGIN + offset * MS, np.full(7, offset))
    buffer.append_external_torque(ORIGIN + 2 * MS, np.full(7, 12))
    buffer.append_external_torque(ORIGIN + 15 * MS, np.full(7, 25), available_ns=ORIGIN + 25 * MS)
    observation = buffer.observation_at(ORIGIN + 20 * MS)
    assert observation.state_timestamp_ns == ORIGIN + 20 * MS
    assert observation.torque_timestamps_ns.tolist() == [ORIGIN + 2 * MS]
    assert observation.torque_values[:, 0].tolist() == [12]
    later = buffer.observation_at(ORIGIN + 25 * MS)
    assert later.torque_timestamps_ns.tolist() == [ORIGIN + 2 * MS, ORIGIN + 15 * MS]


def test_old_checkpoint_contract_cannot_be_reinterpreted_as_external_estimate():
    config = _config()
    old = config.contract_dict()
    old["raw_schema_version"] = "real_robot_right_arm_raw_v1"
    old.pop("force_source_topic")
    with pytest.raises(ValueError, match="raw_schema_version|force_source_topic"):
        validate_data_contract(old, config)


def test_stale_state_and_camera_fail_explicitly_without_repeating_frames():
    buffer = _buffer()
    buffer.append_state(ORIGIN, np.zeros(7), np.zeros(7))
    with pytest.raises(ObservationUnavailableError, match="state"):
        buffer.observation_at(ORIGIN + 21 * MS)
    buffer.append_state(ORIGIN + 101 * MS, np.zeros(7), np.zeros(7))
    with pytest.raises(ObservationUnavailableError, match="camera d455"):
        buffer.observation_at(ORIGIN + 101 * MS)


def test_bad_message_shape_order_and_unknown_camera_are_rejected():
    buffer = _buffer()
    with pytest.raises(ValueError, match="shape"):
        buffer.append_state(ORIGIN, np.zeros(6), np.zeros(7))
    buffer.append_state(ORIGIN, np.zeros(7), np.zeros(7))
    with pytest.raises(ValueError, match="strictly increasing"):
        buffer.append_state(ORIGIN, np.zeros(7), np.zeros(7))
    with pytest.raises(ValueError, match="unknown camera"):
        buffer.append_image("left", ORIGIN, np.zeros((32, 32, 3), np.uint8))
    with pytest.raises(TypeError, match="integer"):
        buffer.observation_at(float(ORIGIN))


def test_contract_validates_joint_order_camera_order_signal_and_clock():
    config = _config()
    contract = config.contract_dict()
    validate_data_contract(contract, config)
    changed = {**contract, "contact_representation": "cartesian_wrench"}
    with pytest.raises(ValueError, match="contact_representation"):
        validate_data_contract(changed, config)
    for field in ("joint_names", "camera_names"):
        changed = {**contract, "config": {**contract["config"],
                    field: tuple(reversed(contract["config"][field]))}}
        with pytest.raises(ValueError, match="config"):
            validate_data_contract(changed, config)


@pytest.mark.parametrize("query_interval", [4, 5])
def test_100hz_execution_with_25_or_20hz_queries_uses_each_chunk_tick(query_interval):
    buffer = TimestampActionChunkBuffer(ORIGIN)
    for tick in range(13):
        timestamp = ORIGIN + tick * 10 * MS
        if tick % query_interval == 0:
            actions = np.repeat(np.arange(tick, tick + 10)[:, None], 7, axis=1)
            buffer.submit(actions, prediction_start_ns=timestamp, available_ns=timestamp)
        result = buffer.action_at(timestamp)
        assert result.action[0] == tick
        assert result.chunk_index == tick % query_interval


def test_asynchronous_30hz_result_skips_elapsed_ticks_and_keeps_latest_observation():
    buffer = TimestampActionChunkBuffer(ORIGIN)
    buffer.submit(np.repeat(np.arange(10)[:, None], 7, axis=1),
                  prediction_start_ns=ORIGIN, available_ns=ORIGIN)
    # A query anchored at tick 3 (near the next 30 Hz camera frame) completes
    # between ticks.  Tick 5 consumes element 2, not element 0 of that result.
    buffer.submit(np.repeat(np.arange(103, 113)[:, None], 7, axis=1),
                  prediction_start_ns=ORIGIN + 30 * MS,
                  available_ns=ORIGIN + 45 * MS)
    assert buffer.action_at(ORIGIN + 40 * MS).action[0] == 4
    assert buffer.action_at(ORIGIN + 50 * MS).action[0] == 105
    # Older query completes even later; it must not overwrite newer context.
    buffer.submit(np.full((10, 7), -100), prediction_start_ns=ORIGIN + 10 * MS,
                  available_ns=ORIGIN + 55 * MS)
    assert buffer.action_at(ORIGIN + 60 * MS).action[0] == 106


def test_exhausted_chunk_never_implicitly_holds_terminal_action():
    buffer = TimestampActionChunkBuffer(ORIGIN)
    buffer.submit(np.zeros((2, 7)), prediction_start_ns=ORIGIN, available_ns=ORIGIN)
    buffer.action_at(ORIGIN + 10 * MS)
    with pytest.raises(ActionUnavailableError, match="tick 2"):
        buffer.action_at(ORIGIN + 20 * MS)
    with pytest.raises(ValueError, match="100 Hz"):
        buffer.action_at(ORIGIN + 15 * MS)


class _CaptureModel(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.anchor = torch.nn.Parameter(torch.zeros(()))

    def forward(self, images, qpos, *arguments, **kwargs):
        return {"pred_action": qpos[:, None].expand(-1, self.config.chunk_len, -1)}


def test_preparation_uses_shared_history_and_image_transforms():
    from force_aware_act.real_robot.data import preprocess_images
    from force_aware_act.real_robot.training import make_model_config

    config = _config()
    model_config = make_model_config("contact", config, compact=True, chunk_len=3)
    adapter = RealRobotPolicyAdapter(_CaptureModel(model_config), model_kind="contact",
                                     normalization=_stats(), data_config=config,
                                     data_contract=config.contract_dict())
    buffer = _buffer(config)
    times = ORIGIN + np.asarray([0, 3, 10, 20, 40]) * MS
    torques = np.repeat(np.arange(5, dtype=np.float32)[:, None], 7, axis=1)
    for timestamp, torque in zip(times, torques):
        buffer.append_state(timestamp, torque, torque)
    inputs = adapter.prepare_observation(buffer, ORIGIN + 20 * MS)
    packed = pack_torque_history(times, torques, ORIGIN + 20 * MS, config)
    expected = (torch.from_numpy(packed.values) - 1) / 2
    expected[packed.sample_padding_mask] = 0
    torch.testing.assert_close(inputs.online_force_intervals[0], expected)
    torch.testing.assert_close(inputs.online_force_relative_time[0],
                               torch.from_numpy(packed.relative_times))
    torch.testing.assert_close(inputs.images[0], preprocess_images(inputs.source.images, model_config))
    assert int((~inputs.online_force_sample_padding_mask).sum()) == 4
    result = adapter.predict(buffer, ORIGIN + 20 * MS)
    np.testing.assert_array_equal(result["pred_action"], np.full((3, 7), 3))


@pytest.mark.parametrize("kind", ["contact", "motion", "dual-zero", "act"])
def test_checkpoint_adapter_load_and_inference_for_all_current_models(kind):
    from force_aware_act.real_robot.training import build_model, make_model_config

    config = _config(history_duration_s=.02)
    model_config = make_model_config(kind, config, compact=True, chunk_len=3)
    model = build_model(kind, asdict(model_config), initialize_pretrained=False)
    checkpoint = {
        "format_version": "real_robot_training_v1", "model_kind": kind,
        "model_config": asdict(model_config), "model_state": model.state_dict(),
        "data_config": config.to_dict(), "data_contract": config.contract_dict(),
        "normalization": _stats(),
    }
    adapter = RealRobotPolicyAdapter.from_checkpoint(checkpoint)
    buffer = _buffer(config)
    buffer.append_state(ORIGIN, np.zeros(7), np.zeros(7))
    buffer.append_state(ORIGIN + 10 * MS, np.ones(7), np.ones(7))
    output = adapter.predict(buffer, ORIGIN + 10 * MS)
    assert output["pred_action"].shape == (3, 7)
    assert np.isfinite(output["pred_action"]).all()
    assert ("pred_force" in output) == (kind != "act")
    if kind != "act":
        assert output["pred_force"].shape == (3, 7)
        assert output["pred_force_highrate"].shape == (3, 3, 7)
        expected_offsets = np.rint(np.arange(1, 10) * (1e9 / 300)).astype(np.int64).reshape(3, 3)
        np.testing.assert_array_equal(output["pred_force_highrate_timestamps_ns"],
                                      ORIGIN + 10 * MS + expected_offsets)
    else:
        assert adapter.prepare_observation(buffer, ORIGIN + 10 * MS).online_force_intervals is None


def test_seven_dimensional_normalization_is_required():
    from force_aware_act.real_robot.training import make_model_config

    config = _config()
    model = _CaptureModel(make_model_config("contact", config, compact=True))
    with pytest.raises(ValueError, match="force_std"):
        RealRobotPolicyAdapter(model, model_kind="contact",
                              normalization={**_stats(), "force_std": [1] * 6},
                              data_config=config, data_contract=config.contract_dict())


def test_nominal_history_length_must_match_checkpoint_contract():
    from dataclasses import replace
    from force_aware_act.real_robot.training import make_model_config

    config = _config()
    model_config = replace(make_model_config("contact", config, compact=True),
                           online_force_window_len=100)
    with pytest.raises(ValueError, match="nominal history"):
        RealRobotPolicyAdapter(_CaptureModel(model_config), model_kind="contact",
                              normalization=_stats(), data_config=config,
                              data_contract=config.contract_dict())


def test_saved_checkpoint_replay_matches_offline_data_and_never_publishes(tmp_path):
    from force_aware_act.act_aligned_training.normalization import NormalizationStats
    from force_aware_act.real_robot.checkpoint import save_checkpoint
    from force_aware_act.real_robot.training import build_model, make_model_config

    config = _config()
    model_config = make_model_config("dual-zero", config, compact=True, chunk_len=3)
    model = build_model("dual-zero", asdict(model_config), initialize_pretrained=False)
    path = tmp_path / "episode.hdf5"
    offsets = {
        "state": np.asarray([0, 3, 10, 20, 30, 40, 50, 60, 70]) * MS,
        "disturbance": np.asarray([2, 6, 15, 22, 36, 45, 62, 69]) * MS,
        "action": np.arange(8) * 10 * MS,
        "d455": np.asarray([0, 33, 66]) * MS,
        "d405": np.asarray([0, 33, 66]) * MS,
    }
    with h5py.File(path, "w") as handle:
        handle.attrs.update(schema_version=RAW_SCHEMA_VERSION, arm="right",
                            force_representation="joint_external_torque", gravity_compensation="unverified")
        handle.create_dataset("metadata/joint_names", data=np.asarray(config.joint_names, dtype="S"))
        handle.create_dataset("metadata/camera_names", data=np.asarray(config.camera_names, dtype="S"))
        for stream, times in offsets.items():
            for clock in ("log_ns", "header_ns", "publish_ns"):
                handle.create_dataset(f"timestamps/{stream}/{clock}", data=ORIGIN + times)
        for field in ("joint_pos", "joint_vel"):
            handle.create_dataset(f"observations/{field}", data=np.repeat(
                np.arange(len(offsets["state"]))[:, None], 7, axis=1).astype(np.float64))
        force = handle.create_dataset("observations/joint_external_torque", data=np.repeat(
            np.arange(len(offsets["disturbance"]))[:, None], 7, axis=1).astype(np.float64))
        force.attrs.update(source_topic=EXTERNAL_TORQUE_TOPIC, source_field="effort",
                           timestamp_group="/timestamps/disturbance")
        handle["timestamps/disturbance"].attrs["source_topic"] = EXTERNAL_TORQUE_TOPIC
        handle.create_dataset("action", data=np.zeros((len(offsets["action"]), 7)))
        for camera in config.camera_names:
            handle.create_dataset(f"observations/images/{camera}/rgb",
                                  data=np.zeros((len(offsets[camera]), 32, 32, 3), np.uint8))
    checkpoint = tmp_path / "checkpoint.pt"
    metadata = {
        "model_kind": "dual-zero", "model_config": asdict(model_config),
        "data_config": config.to_dict(), "data_contract": config.contract_dict(),
        "normalization": NormalizationStats(**_stats()).to_dict(),
        "split_manifest": {}, "source_fingerprints": {}, "training_config": {}, "run_config": {},
    }
    save_checkpoint(checkpoint, model=model, optimizer=torch.optim.AdamW(model.parameters()),
                    sampler=torch.Generator(), global_step=0, metadata=metadata)
    replay = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/replay_real_robot.py"))["replay"]
    report = replay(checkpoint, path, max_samples=3)
    assert report["offline_online_inputs_equal"]
    assert report["replayed_anchors"] == 3
    assert report["robot_commands_published"] is False
    assert all(row["action_shape"] == [3, 7] for row in report["samples"])
