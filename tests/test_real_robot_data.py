from dataclasses import replace

import h5py
import numpy as np
import pytest
import torch

from force_aware_act.real_robot import (
    RawEpisode, RealRobotDataConfig, RealRobotHighRateDataset,
    compute_real_robot_normalization_stats, discover_real_robot_episodes,
)
from force_aware_act.real_robot.sampling import (
    causal_index, estimate_action_grid_phase_ns, match_time_slots,
    pack_future_torque, pack_torque_history,
)
from force_aware_act.models.act_aligned.config import ACTAlignedHighRateConfig
from force_aware_act.models.official_act.config import OfficialACTConfig
from force_aware_act.real_robot.data import preprocess_images
from force_aware_act.real_robot.config import RAW_SCHEMA_VERSION, EXTERNAL_TORQUE_TOPIC
from force_aware_act.act_aligned_training.high_rate_data import collate_high_rate_samples


EPOCH = 1_750_000_000_000_000_000


def _write_raw(root, name="demo", *, offset=0.0, missing_action_ms=(), missing_state_ms=(),
               missing_disturbance_ms=None, disturbance_delay_ms=0):
    path = root / name / "episode.hdf5"
    path.parent.mkdir(parents=True)
    state_offsets = np.rint(np.arange(121) * 1e9 / 300).astype(np.int64)
    state_offsets = state_offsets[~np.isin(state_offsets, np.asarray(missing_state_ms) * 1_000_000)]
    disturbance_offsets = np.rint(np.arange(121) * 1e9 / 300).astype(np.int64)
    missing_disturbance_ms = missing_state_ms if missing_disturbance_ms is None else missing_disturbance_ms
    disturbance_offsets = disturbance_offsets[~np.isin(disturbance_offsets, np.asarray(missing_disturbance_ms) * 1_000_000)]
    disturbance_offsets += disturbance_delay_ms * 1_000_000
    action_offsets = np.arange(41, dtype=np.int64) * 10_000_000
    action_offsets = action_offsets[~np.isin(action_offsets, np.asarray(missing_action_ms) * 1_000_000)]
    cameras = {"d455": np.arange(13, dtype=np.int64) * 33_000_000,
               "d405": 4_000_000 + np.arange(12, dtype=np.int64) * 33_000_000}
    with h5py.File(path, "w") as h:
        h.attrs.update(schema_version=RAW_SCHEMA_VERSION, arm="right",
                       force_representation="joint_external_torque", gravity_compensation="unverified")
        h.create_dataset("metadata/joint_names", data=np.asarray([f"Joint{i}_R".encode() for i in range(1, 8)]))
        h.create_dataset("metadata/camera_names", data=np.asarray([b"d455", b"d405"]))
        for stream, offsets in {"state": state_offsets, "disturbance": disturbance_offsets,
                                "action": action_offsets, **cameras}.items():
            times = offsets + EPOCH
            h.create_dataset(f"timestamps/{stream}/log_ns", data=times.astype(np.uint64))
            h.create_dataset(f"timestamps/{stream}/header_ns", data=times - 100_000_000)
            h.create_dataset(f"timestamps/{stream}/publish_ns", data=times.astype(np.uint64))
        h["timestamps/disturbance"].attrs["source_topic"] = EXTERNAL_TORQUE_TOPIC
        for key in ("joint_pos", "joint_vel"):
            h.create_dataset(f"observations/{key}", data=np.repeat((state_offsets * 1e-6 + offset)[:, None], 7, axis=1))
        h.create_dataset("observations/joint_internal_torque", data=np.full((len(state_offsets), 7), 99999.0))
        force = h.create_dataset("observations/joint_external_torque",
                                 data=np.repeat((disturbance_offsets * 1e-6 + offset)[:, None], 7, axis=1))
        force.attrs.update(source_topic=EXTERNAL_TORQUE_TOPIC, source_field="effort",
                           timestamp_group="/timestamps/disturbance", unit="N*m")
        h.create_dataset("action", data=np.repeat((action_offsets * 1e-6 + offset)[:, None], 7, axis=1))
        for camera, offsets in cameras.items():
            pixels = np.repeat((np.arange(len(offsets), dtype=np.uint8) + 10)[:, None, None, None], 2 * 3 * 3, axis=1)
            h.create_dataset(f"observations/images/{camera}/rgb", data=pixels.reshape(len(offsets), 2, 3, 3))
    return path


def test_external_torque_has_independent_counts_times_and_internal_torque_is_excluded(tmp_path):
    path = _write_raw(tmp_path, missing_disturbance_ms=(50, 60), disturbance_delay_ms=2)
    config = _config()
    raw = RawEpisode(path, config)
    assert len(raw.qpos) == 121 and len(raw.torque) == 119
    assert raw.torque_timestamps_ns[0] == EPOCH + 2_000_000
    records = discover_real_robot_episodes(tmp_path, config)
    stats = compute_real_robot_normalization_stats(tmp_path, records, config)
    assert stats.force_mean == pytest.approx(raw.torque.mean(axis=0))
    assert max(stats.force_mean) < 1000  # internal torque has the distinct sentinel 99999
    dataset = RealRobotHighRateDataset(tmp_path, records, stats, _model(config), config)
    index = dataset.index.index((0, 4))
    sample = dataset[index]
    expected_history = pack_torque_history(raw.torque_timestamps_ns, raw.torque, EPOCH + 40_000_000, config)
    expected_future = pack_future_torque(raw.torque_timestamps_ns, raw.torque, EPOCH + 40_000_000, 3, config)
    assert sample.online_raw_valid_count == expected_history.valid_sample_count
    assert sample.future_force_sample_padding_mask.tolist() == expected_future.sample_padding_mask.tolist()
    trace = dataset.sample_trace(index)
    assert trace["force_timestamp_stream"] == "disturbance"
    assert trace["force_source_topic"] == EXTERNAL_TORQUE_TOPIC
    for slot in trace["future_torque"]["first_interval"]:
        if slot["hdf5_index"] >= 0:
            assert slot["source_timestamp_ns"] == raw.torque_timestamps_ns[slot["hdf5_index"]]
    # Mutating audit-only internal torques cannot affect training views or stats.
    with h5py.File(path, "r+") as handle:
        handle["observations/joint_internal_torque"][:] = -88888
    changed_stats = compute_real_robot_normalization_stats(tmp_path, records, config)
    assert changed_stats == stats
    changed = RealRobotHighRateDataset(tmp_path, records, changed_stats, _model(config), config)[index]
    torch.testing.assert_close(changed.online_force_intervals, sample.online_force_intervals, rtol=0, atol=0)
    torch.testing.assert_close(changed.future_force_intervals, sample.future_force_intervals, rtol=0, atol=0)


@pytest.mark.parametrize("damage", ["old_schema", "wrong_topic", "wrong_field", "state_time_axis"])
def test_external_force_source_contract_rejects_old_or_mislabelled_data(tmp_path, damage):
    path = _write_raw(tmp_path)
    with h5py.File(path, "r+") as handle:
        if damage == "old_schema":
            handle.attrs["schema_version"] = "real_robot_right_arm_raw_v1"
        else:
            key, value = {"wrong_topic": ("source_topic", "io_teleop/joint_states"),
                          "wrong_field": ("source_field", "position"),
                          "state_time_axis": ("timestamp_group", "/timestamps/state")}[damage]
            handle["observations/joint_external_torque"].attrs[key] = value
    with pytest.raises(ValueError, match="schema|source_topic|source_field|timestamp_group"):
        RawEpisode(path, _config())


def _config(**kwargs):
    return RealRobotDataConfig(image_height=2, image_width=3, **kwargs)


def _model(config):
    return ACTAlignedHighRateConfig(
        d_model=32, nhead=4, dim_feedforward=64, chunk_len=3, force_dim=7,
        image_height=2, image_width=3, pretrained_backbone=False, imagenet_normalize=False,
        force_sample_rate_hz=300, policy_sample_rate_hz=100, online_force_window_len=60,
        max_online_force_intervals=config.history_intervals,
        max_force_samples_per_interval=config.max_samples_per_interval, local_force_dim=16,
    )


def test_independent_streams_discovered_without_sidecar_and_no_future_inputs(tmp_path):
    path = _write_raw(tmp_path, missing_action_ms=(50,))
    config = _config()
    records = discover_real_robot_episodes(tmp_path, config)
    assert len(records) == 1 and records[0].num_steps > 0
    raw = RawEpisode(path, config)
    assert len(raw.qpos) != len(raw.action) != len(raw.timestamps["d455"])
    anchor = EPOCH + 40_000_000
    state, cameras = raw.observation_indices(anchor)
    assert raw.state_timestamps_ns[state] <= anchor
    assert all(raw.timestamps[camera][index] <= anchor for camera, index in cameras.items())
    assert raw.timestamps["d455"][cameras["d455"]] != raw.timestamps["d405"][cameras["d405"]]
    # A header stamp is not permission to see a message that arrives 100 ms later.
    assert raw.qpos[state, 0] == pytest.approx(40.0)
    assert 5 not in raw.valid_anchor_indices
    assert raw.action_slot_indices[5] == -1
    assert raw.action_slot_indices[6] == 5
    assert raw.diagnostics["excluded_anchor_reasons"]["missing_current_action"] == 1
    trace = raw.observation_trace(anchor)
    assert trace["state"]["timestamp_ns"] <= trace["anchor_ns"]
    assert all(trace[camera]["age_ms"] >= 0 for camera in config.camera_names)
    assert raw.diagnostics["action_matching"]["missing_count"] == 1


def test_missing_fixed_slots_do_not_shift_and_endpoint_has_separate_mask(tmp_path):
    _write_raw(tmp_path, missing_action_ms=(50,), missing_state_ms=(50,))
    config = _config()
    records = discover_real_robot_episodes(tmp_path, config)
    stats = compute_real_robot_normalization_stats(tmp_path, records, config)
    dataset = RealRobotHighRateDataset(tmp_path, records, stats, _model(config), config)
    location = dataset.index.index((0, 4))
    sample = dataset[location]
    assert sample.action_padding_mask.tolist() == [False, True, False]
    assert sample.future_force_sample_padding_mask[0, :3].tolist() == [False, False, True]
    assert not sample.future_force_interval_padding_mask[0]
    assert sample.future_force_target_padding_mask[0]
    assert sample.future_force_target[0].eq(0).all()
    assert sample.future_force_intervals[0, 2].eq(0).all()
    assert sample.future_force_relative_time[0, :3].tolist() == pytest.approx([1/300, 2/300, .01])
    batch = collate_high_rate_samples([sample])
    batch.validate(_model(config))
    assert batch.force_target_padding_mask[0, 0]
    trace = dataset.sample_trace(location)
    assert trace["action"]["valid_slots"] == [True, False, True]
    assert trace["future_torque"]["first_interval"][2]["hdf5_index"] == -1
    assert trace["future_torque"]["endpoint_padding_mask"][0]


def test_future_never_reuses_anchor_sample_and_keeps_episode_end_masked():
    config = _config()
    times = EPOCH + np.asarray([0, 3_333_333, 6_666_667, 10_000_000], dtype=np.int64)
    values = np.repeat(np.arange(4, dtype=np.float32)[:, None], 7, axis=1)
    future = pack_future_torque(times, values, EPOCH, 2, config)
    assert future.values[0, :3, 0].tolist() == [1, 2, 3]
    assert future.sample_padding_mask[1].all()
    assert future.sample_counts.tolist() == [3, 0]
    history = pack_torque_history(times, values, EPOCH, config)
    assert history.valid_sample_count == 1
    assert history.values[-1, 0, 0] == 0


def test_causal_age_rejection_and_true_gap_history():
    config = _config()
    times = EPOCH + np.asarray([0, 1_000_000, 31_000_000], dtype=np.int64)
    assert causal_index(times, EPOCH + 30_000_000, 20_000_000) is None
    assert causal_index(times, EPOCH - 1, 20_000_000) is None
    values = np.ones((3, 7), dtype=np.float32)
    history = pack_torque_history(times, values, EPOCH + 40_000_000, config)
    assert history.valid_sample_count == 3
    assert history.interval_padding_mask[-3:-1].all()
    # A duplicated fallback value is never synthesized to fill empty intervals.
    assert (~history.sample_padding_mask).sum() == 3


def test_one_to_one_nearest_integer_matching_retains_nanosecond_precision():
    targets = EPOCH + np.asarray([1, 3, 5], dtype=np.int64)
    times = EPOCH + np.asarray([2, 5], dtype=np.int64)
    assert match_time_slots(times, targets, 1).tolist() == [-1, 0, 1]
    assert match_time_slots(EPOCH + np.asarray([1, 3, 5]), targets, 0).tolist() == [0, 1, 2]


def test_action_grid_phase_uses_initial_receive_times_and_records_provenance(tmp_path):
    times = EPOCH + np.arange(301, dtype=np.int64) * 10_000_000
    times[1:] -= 4_000_000
    times[201:] += 1_000_000
    estimate = estimate_action_grid_phase_ns(times, 10_000_000, 2_000_000_000)
    assert estimate["sample_count"] == 201
    assert estimate["phase_ns"] == pytest.approx(6_000_000, abs=50_000)
    assert estimate["resultant_length"] > 0.99

    path = _write_raw(tmp_path)
    with h5py.File(path, "r+") as handle:
        recorded = handle["timestamps/action/log_ns"][:].astype(np.int64)
        recorded[1:] += -4_500_000 + np.where(np.arange(1, len(recorded)) % 2, -1_000_000, 1_000_000)
        handle["timestamps/action/log_ns"][:] = recorded.astype(np.uint64)
        # Keep the original header stamps: grid phase must use receive times.
    robust = RawEpisode(path, _config())
    legacy = RawEpisode(path, _config(action_grid_phase_mode="first_action"))
    grid = robust.diagnostics["action_grid"]
    assert grid["timestamp_key"] == "log_ns"
    assert grid["window_ns"] == 2_000_000_000
    assert grid["sample_count"] == 41
    assert grid["first_anchor_ns"] == robust.action_timestamps_ns[0] + grid["phase_ns"]
    assert robust.diagnostics["action_matching"]["missing_count"] < legacy.diagnostics["action_matching"]["missing_count"]


def test_statistics_only_use_explicit_training_episodes(tmp_path):
    path = _write_raw(tmp_path, "train")
    _write_raw(tmp_path, "validation", offset=10000)
    config = _config()
    records = discover_real_robot_episodes(tmp_path, config)
    train = [record for record in records if record.episode_id == "train"]
    stats = compute_real_robot_normalization_stats(tmp_path, train, config)
    raw = RawEpisode(path, config)
    assert stats.qpos_mean == pytest.approx(raw.qpos.mean(axis=0))
    assert stats.force_mean == pytest.approx(raw.torque.mean(axis=0))
    assert stats.minimum_std == 1e-2
    original = torch.as_tensor(raw.action[:3], dtype=torch.float32)
    assert torch.allclose(stats.denormalize_action(stats.normalize_action(original)), original, atol=2e-5)


@pytest.mark.parametrize("damage", ["nan", "clock", "joint_order", "camera_shape"])
def test_invalid_raw_measurements_fail_explicitly(tmp_path, damage):
    path = _write_raw(tmp_path)
    with h5py.File(path, "r+") as h:
        if damage == "nan":
            h["observations/joint_external_torque"][1, 2] = np.nan
        elif damage == "clock":
            h["timestamps/state/log_ns"][1] = h["timestamps/state/log_ns"][0]
        elif damage == "joint_order":
            h["metadata/joint_names"][0] = b"Joint7_R"
        else:
            del h["observations/images/d405/rgb"]
            h.create_dataset("observations/images/d405/rgb", shape=(12, 3, 2, 3), dtype="u1")
    with pytest.raises(ValueError):
        RawEpisode(path, _config())


def test_config_round_trip_and_capacity_overflow_are_explicit():
    config = _config()
    assert RealRobotDataConfig.from_dict(config.to_dict()) == config
    assert config.contract_dict()["qvel_input"] is False
    old_config = dict(config.to_dict())
    old_config.pop("action_grid_phase_mode")
    old_config.pop("action_grid_phase_window_s")
    legacy = RealRobotDataConfig.from_dict(old_config)
    assert legacy.action_grid_phase_mode == "first_action"
    assert legacy.to_dict() == old_config
    assert legacy.contract_dict()["sampling_version"].endswith("_v1")
    assert config.contract_dict()["sampling_version"].endswith("_v2")
    times = EPOCH + np.arange(9, dtype=np.int64)
    with pytest.raises(ValueError, match="capacity"):
        pack_torque_history(times, np.zeros((9, 7), dtype=np.float32), times[-1], config)
    with pytest.raises(ValueError, match="half"):
        replace(config, torque_match_tolerance_ms=2.0)
    with pytest.raises(ValueError, match="availability"):
        replace(config, timestamp_key="header_ns")


def test_official_act_preprocessing_does_not_double_normalize():
    config = OfficialACTConfig(image_height=2, image_width=3)
    images = [np.full((2, 3, 3), 128, dtype=np.uint8) for _ in range(2)]
    tensor = preprocess_images(images, config)
    assert torch.allclose(tensor, torch.full_like(tensor, 128 / 255))


@pytest.mark.parametrize("path,attribute,value", [
    ("observations/joint_pos", "unit", "degree"),
    ("observations/joint_vel", "unit", "deg/s"),
    ("observations/joint_external_torque", "unit", "N"),
    ("action", "unit", "degree"),
    ("observations/images/d405", "color_order", "BGR"),
    ("observations/images/d455", "layout", "NCHW"),
])
def test_present_metadata_must_not_contradict_units_or_rgb(tmp_path, path, attribute, value):
    episode = _write_raw(tmp_path)
    with h5py.File(episode, "r+") as handle:
        handle[path].attrs[attribute] = value
    with pytest.raises(ValueError, match="unit|color_order|layout"):
        RawEpisode(episode, _config())


@pytest.mark.parametrize("torque_unit", ["N*m", "Nm", "N·m"])
def test_confirmed_units_and_rgb_metadata_are_accepted(tmp_path, torque_unit):
    episode = _write_raw(tmp_path)
    with h5py.File(episode, "r+") as handle:
        handle["observations/joint_pos"].attrs["unit"] = "rad"
        handle["observations/joint_vel"].attrs["unit"] = "rad/s"
        handle["observations/joint_external_torque"].attrs["unit"] = torque_unit
        handle["action"].attrs["unit"] = "rad"
        for camera in ("d455", "d405"):
            handle[f"observations/images/{camera}"].attrs.update(color_order="RGB", layout="NHWC")
    assert len(RawEpisode(episode, _config()).valid_anchor_indices) > 0
