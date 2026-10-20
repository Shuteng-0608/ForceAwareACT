"""Read raw asynchronous episodes without changing or completing their streams."""

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

from .config import (RAW_SCHEMA_VERSION, EXTERNAL_TORQUE_TOPIC,
                     EXTERNAL_TORQUE_FIELD, WRENCH_TOPIC, WRENCH_AXES, RealRobotDataConfig)
from .sampling import (causal_index, estimate_action_grid_phase_ns,
                       match_time_slots, matching_error_summary, pack_torque_history)


def _text(value):
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


@dataclass(frozen=True)
class RealRobotEpisodeRecord:
    episode_id: str
    relative_hdf5_path: str
    num_steps: int
    camera_names: tuple[str, ...]

    def resolve(self, data_root):
        return Path(data_root) / self.relative_hdf5_path


def _integer_times(values, name):
    array = np.asarray(values)
    if array.ndim != 1 or not len(array) or array.dtype.kind not in "iu":
        raise ValueError(f"{name} must be a nonempty integer nanosecond vector")
    if array.dtype.kind == "u" and np.any(array > np.iinfo(np.int64).max):
        raise ValueError(f"{name} exceeds int64 nanosecond range")
    return array.astype(np.int64)


class RawEpisode:
    """Cache compact numeric streams and indices; read image pixels on demand."""

    def __init__(self, path, config=None):
        self.path = Path(path).resolve()
        self.config = config or RealRobotDataConfig()
        self.timestamps = {}
        self.header_timestamps = {}
        self.diagnostics = {"timestamp_key": self.config.timestamp_key, "streams": {}}
        with h5py.File(self.path, "r") as handle:
            schema = self.config.raw_schema_version
            wrench6 = self.config.force_dim == 6
            force_key = "observations/ee_wrench" if wrench6 else "observations/joint_external_torque"
            force_stream = self.config.force_timestamp_stream
            force_topic = WRENCH_TOPIC if wrench6 else EXTERNAL_TORQUE_TOPIC
            if _text(handle.attrs.get("schema_version", "")) != schema:
                raise ValueError(f"{self.path}: unsupported raw schema; expected {schema}")
            if _text(handle.attrs.get("arm", "")) != "right":
                raise ValueError("raw episode must explicitly identify the right arm")
            expected_rep = "end_effector_wrench6" if wrench6 else "joint_external_torque"
            if _text(handle.attrs.get("force_representation", "")) != expected_rep:
                raise ValueError(f"expected {expected_rep}")
            force = handle[force_key]
            for name, expected in (("source_topic", force_topic), ("source_field", "effort"),
                                   ("timestamp_group", "/timestamps/" + force_stream)):
                if _text(force.attrs.get(name, "")) != expected:
                    raise ValueError(f"force {name} must be {expected!r}")
            if _text(handle["timestamps/" + force_stream].attrs.get("source_topic", "")) != force_topic:
                raise ValueError("force timestamps must originate from the selected force topic")
            if wrench6:
                axes = tuple(_text(x) for x in handle["metadata/ee_wrench_axes"][...])
                if axes != WRENCH_AXES:
                    raise ValueError("ee_wrench axes must be Fx/Fy/Fz/Tx/Ty/Tz")
                if _text(handle.attrs.get("primary_force_dataset", "")) != force_key:
                    raise ValueError("raw primary_force_dataset must identify ee_wrench")
                for attrs in (handle.attrs, force.attrs):
                    if _text(attrs.get("gravity_compensation", "")) != "applied_upstream":
                        raise ValueError("wrench gravity compensation must be applied_upstream")
                frames = handle["timestamps/ft_sensor/frame_id"][...]
                if len(frames) != len(force) or any(_text(f) != self.config.wrench_frame for f in frames):
                    raise ValueError("wrench frame_id differs from configured measurement frame")
                unit = _text(force.attrs.get("unit", ""))
                if unit not in {"source_native", "source_native; no unit conversion", "N,N,N,N*m,N*m,N*m"}:
                    raise ValueError("ee_wrench unit metadata contradicts configured N/N*m")
            joints = tuple(_text(x) for x in handle["metadata/joint_names"][...])
            if joints != tuple(self.config.joint_names):
                raise ValueError("joint order does not match the right-arm sampling contract")
            cameras = tuple(_text(x) for x in handle["metadata/camera_names"][...])
            if set(cameras) != set(self.config.camera_names):
                raise ValueError("camera names do not match the sampling contract")
            self.provenance = {
                "path": str(self.path), "schema_version": schema,
                "source_sha256": _text(handle.attrs.get("source_sha256", "")),
                "source_path": _text(handle.attrs.get("source_path", "")),
                "friction_compensation": _text(handle.attrs.get("friction_compensation", "unknown")),
                "force_source_topic": force_topic,
                "force_source_field": EXTERNAL_TORQUE_FIELD,
                "force_timestamp_stream": force_stream,
            }
            for stream in ("state", force_stream, "action", *self.config.camera_names):
                group = handle[f"timestamps/{stream}"]
                times = _integer_times(group[self.config.timestamp_key][...], stream)
                if np.any(np.diff(times) <= 0):
                    raise ValueError(f"{stream}: chosen clock timestamps must be strictly increasing")
                header = _integer_times(group["header_ns"][...], f"{stream}/header_ns")
                logs = _integer_times(group["log_ns"][...], f"{stream}/log_ns")
                publish = _integer_times(group["publish_ns"][...], f"{stream}/publish_ns")
                if not len(times) == len(header) == len(logs) == len(publish):
                    raise ValueError(f"{stream}: clock array lengths disagree")
                self.timestamps[stream] = times
                self.header_timestamps[stream] = header
                delta = np.diff(times)
                self.diagnostics["streams"][stream] = {
                    "count": len(times), "duration_s": (int(times[-1]) - int(times[0])) * 1e-9,
                    "max_interval_ms": float(delta.max() * 1e-6) if len(delta) else 0.0,
                    "header_nonincreasing_count": int((np.diff(header) <= 0).sum()),
                    "log_minus_header_median_ms": float(np.median(logs - header) * 1e-6),
                }
            numeric = {}
            allowed_units = {
                "observations/joint_pos": {"rad"},
                "observations/joint_vel": {"rad/s"},
                "observations/joint_external_torque": {"N*m", "Nm", "N·m"},
                "action": {"rad"},
            }
            for key, stream in (("observations/joint_pos", "state"),
                                ("observations/joint_vel", "state"),
                                (force_key, force_stream), ("action", "action")):
                if key in allowed_units and "unit" in handle[key].attrs:
                    unit = _text(handle[key].attrs["unit"])
                    if unit not in allowed_units[key]:
                        raise ValueError(f"{key}: unit {unit!r} contradicts the real-robot data contract")
                values = np.asarray(handle[key][...], dtype=np.float64)
                dimension = 6 if wrench6 and key == force_key else 7
                if values.shape != (len(self.timestamps[stream]), dimension):
                    raise ValueError(f"{key}: expected independent stream shape [N_{stream}, {dimension}]")
                if not np.isfinite(values).all():
                    raise ValueError(f"{key}: non-finite raw measurements require explicit quality review")
                numeric[key] = values
            self.qpos = numeric["observations/joint_pos"]
            self.qvel = numeric["observations/joint_vel"]  # retained for audit; never a model input
            # Historical attribute name; for raw_v3 this is exclusively measured wrench6.
            self.torque = numeric[force_key]
            self.action = numeric["action"]
            for camera in self.config.camera_names:
                image_group = handle[f"observations/images/{camera}"]
                for attribute, required in (("color_order", "RGB"), ("layout", "NHWC")):
                    if attribute in image_group.attrs and _text(image_group.attrs[attribute]) != required:
                        raise ValueError(f"{camera}: {attribute} must be {required}")
                images = handle[f"observations/images/{camera}/rgb"]
                expected = (len(self.timestamps[camera]), self.config.image_height, self.config.image_width, 3)
                if images.shape != expected or images.dtype != np.dtype("uint8"):
                    raise ValueError(f"{camera}: RGB data must have shape {expected} and uint8 dtype")
        self.state_timestamps_ns = self.timestamps["state"]
        self.torque_timestamps_ns = self.timestamps[force_stream]
        self.force = self.torque
        if self.config.force_representation == "flange_force3":
            from .flange_force import FlangeForceStream
            self.force = FlangeForceStream(self.qpos, self.torque,
                self.header_timestamps["state"], self.header_timestamps["disturbance"],
                self.timestamps["state"], self.timestamps["disturbance"])
            self.torque_timestamps_ns = self.force.timestamps_ns
            self.diagnostics["force_conversion"] = self.force.diagnostics
        self.action_timestamps_ns = self.timestamps["action"]
        if self.config.action_grid_phase_mode == "first_action":
            phase = {"method": "first_action", "phase_ns": 0,
                     "period_ns": self.config.action_period_ns, "window_ns": 0,
                     "window_start_ns": int(self.action_timestamps_ns[0]),
                     "window_end_ns": int(self.action_timestamps_ns[0]),
                     "sample_count": 1, "resultant_length": 1.0}
        else:
            phase = estimate_action_grid_phase_ns(
                self.action_timestamps_ns, self.config.action_period_ns,
                round(self.config.action_grid_phase_window_s * 1e9))
        first_anchor_ns = int(self.action_timestamps_ns[0]) + phase["phase_ns"]
        self.anchor_timestamps_ns = np.arange(
            first_anchor_ns, self.action_timestamps_ns[-1] + 1,
            self.config.action_period_ns, dtype=np.int64)
        self.diagnostics["action_grid"] = {
            **phase, "timestamp_key": self.config.timestamp_key,
            "first_anchor_ns": first_anchor_ns,
            "last_anchor_ns": int(self.anchor_timestamps_ns[-1]) if len(self.anchor_timestamps_ns) else None,
        }
        self.action_slot_indices = match_time_slots(
            self.action_timestamps_ns, self.anchor_timestamps_ns,
            round(self.config.action_match_tolerance_ms * 1e6))
        self.diagnostics["action_matching"] = matching_error_summary(
            self.action_timestamps_ns, self.anchor_timestamps_ns, self.action_slot_indices)
        self.diagnostics["action_matching"]["unassigned_raw_messages"] = (
            len(self.action_timestamps_ns) - int((self.action_slot_indices >= 0).sum()))
        reasons = {"missing_current_action": 0, "stale_or_missing_state": 0,
                   "stale_or_missing_camera": 0, "empty_torque_history": 0}
        valid = []
        force_valid = (self.force.valid_samples if self.config.force_dim == 3
                       else np.ones(len(self.torque), dtype=np.bool_))
        force_prefix = np.r_[0, np.cumsum(force_valid)]
        for index, anchor in enumerate(self.anchor_timestamps_ns):
            if self.action_slot_indices[index] < 0:
                reasons["missing_current_action"] += 1
                continue
            if causal_index(self.state_timestamps_ns, anchor, self.config.state_max_age_ns) is None:
                reasons["stale_or_missing_state"] += 1
                continue
            if any(causal_index(self.timestamps[name], anchor, self.config.camera_max_age_ns) is None
                   for name in self.config.camera_names):
                reasons["stale_or_missing_camera"] += 1
                continue
            left = int(anchor) - round(self.config.history_duration_s * 1e9)
            lo, hi = np.searchsorted(self.torque_timestamps_ns, [left, anchor], side="right")
            if force_prefix[hi] == force_prefix[lo]:
                reasons["empty_torque_history"] += 1
                continue
            valid.append(index)
        self.valid_anchor_indices = np.asarray(valid, dtype=np.int64)
        self.valid_anchor_ns = self.anchor_timestamps_ns[self.valid_anchor_indices]
        self.diagnostics.update({"candidate_anchors": len(self.anchor_timestamps_ns),
                                 "valid_anchors": len(valid), "excluded_anchor_reasons": reasons})

    def observation_indices(self, anchor_ns):
        state = causal_index(self.state_timestamps_ns, anchor_ns, self.config.state_max_age_ns)
        cameras = {name: causal_index(self.timestamps[name], anchor_ns, self.config.camera_max_age_ns)
                   for name in self.config.camera_names}
        if state is None or any(index is None for index in cameras.values()):
            raise ValueError("anchor has missing or stale observations")
        return state, cameras

    def images_at(self, anchor_ns):
        _, indices = self.observation_indices(anchor_ns)
        with h5py.File(self.path, "r") as handle:
            return [np.asarray(handle[f"observations/images/{camera}/rgb"][indices[camera]])
                    for camera in self.config.camera_names]

    def observation_trace(self, anchor_ns):
        """Selected HDF5 rows and observed age in the recorder availability clock."""
        state, cameras = self.observation_indices(anchor_ns)
        trace = {"anchor_ns": int(anchor_ns), "timestamp_key": self.config.timestamp_key,
                 "clock_semantics": "recorder availability proxy"}
        for stream, index in {"state": state, **cameras}.items():
            timestamp = int(self.timestamps[stream][index])
            trace[stream] = {"hdf5_index": int(index), "timestamp_ns": timestamp,
                             "header_ns": int(self.header_timestamps[stream][index]),
                             "age_ms": (int(anchor_ns) - timestamp) * 1e-6}
        return trace


def discover_real_robot_episodes(data_root, config=None):
    """Discover raw files by their own metadata, including a single smoke episode."""
    root = Path(data_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    config = config or RealRobotDataConfig()
    records = []
    for path in sorted(root.rglob("episode.hdf5")):
        with h5py.File(path, "r") as handle:
            if _text(handle.attrs.get("schema_version", "")) != config.raw_schema_version:
                continue
        episode = RawEpisode(path, config)
        records.append(RealRobotEpisodeRecord(
            path.parent.relative_to(root).as_posix(), path.relative_to(root).as_posix(),
            len(episode.valid_anchor_indices), tuple(episode.config.camera_names)))
    if not records:
        raise ValueError(f"no {config.raw_schema_version} episodes under {root}")
    return tuple(records)
