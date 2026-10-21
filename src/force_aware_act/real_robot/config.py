"""Versioned sampling contract for untouched asynchronous right-arm recordings."""

from dataclasses import asdict, dataclass
import math


RAW_SCHEMA_VERSION = "real_robot_right_arm_raw_v2"
EXTERNAL_TORQUE_TOPIC = "/tianji_marvin_teleop_bridge/estimated_joint_disturbance"
EXTERNAL_TORQUE_FIELD = "effort"
WRENCH_SCHEMA_VERSION = "real_robot_right_arm_raw_v3"
WRENCH_TOPIC = "/io_teleop/right/ft_sensor"
WRENCH_AXES = ("Fx", "Fy", "Fz", "Tx", "Ty", "Tz")
WRENCH_UNITS = ("N", "N", "N", "N*m", "N*m", "N*m")
LEGACY_REAL_ROBOT_SAMPLING_VERSION = "right_arm_available_clock_100hz_torque300hz_v1"
REAL_ROBOT_SAMPLING_VERSION = "right_arm_available_clock_100hz_torque300hz_phase_v2"


@dataclass(frozen=True)
class RealRobotDataConfig:
    timestamp_key: str = "log_ns"
    action_rate_hz: int = 100
    action_grid_phase_mode: str = "circular_mean_initial_window"
    action_grid_phase_window_s: float = 2.0
    torque_rate_hz: int = 300
    history_duration_s: float = 0.2
    camera_names: tuple[str, ...] = ("d455", "d405")
    joint_names: tuple[str, ...] = tuple(f"Joint{i}_R" for i in range(1, 8))
    image_height: int = 480
    image_width: int = 640
    state_max_age_ms: float = 20.0
    camera_max_age_ms: float = 100.0
    action_match_tolerance_ms: float = 5.0
    torque_match_tolerance_ms: float = 1.666666
    max_samples_per_interval: int = 8
    force_representation: str = "joint_torque7"
    wrench_component_units: tuple[str, ...] | None = None
    wrench_frame: str | None = None
    force_std_floor: tuple[float, ...] | None = None

    def __post_init__(self):
        if self.force_representation not in {"joint_torque7", "flange_force3", "end_effector_wrench6", "end_effector_force3"}:
            raise ValueError("unsupported force_representation")
        if self.uses_wrench_sensor:
            if self.wrench_component_units is None or tuple(self.wrench_component_units) != WRENCH_UNITS[:self.force_dim]:
                raise ValueError("wrench_component_units must explicitly confirm selected SI units in fixed axis order")
            if not isinstance(self.wrench_frame, str) or not self.wrench_frame.strip():
                raise ValueError("wrench_frame must explicitly identify the measurement frame")
            if (self.force_std_floor is None or len(self.force_std_floor) != self.force_dim
                    or any(isinstance(v, bool) or not math.isfinite(v) or v <= 0 for v in self.force_std_floor)):
                raise ValueError(f"force_std_floor requires {self.force_dim} finite positive per-axis values")
        elif any(v is not None for v in (self.wrench_component_units, self.wrench_frame, self.force_std_floor)):
            raise ValueError("wrench-only configuration on a legacy force representation")
        if self.timestamp_key != "log_ns":
            raise ValueError("real-robot sampling requires log_ns availability; header clocks need an explicit synchronized availability mapping")
        if self.action_grid_phase_mode not in {"first_action", "circular_mean_initial_window"}:
            raise ValueError("unsupported action grid phase mode")
        if not math.isfinite(self.action_grid_phase_window_s) or self.action_grid_phase_window_s <= 0:
            raise ValueError("action grid phase window must be finite and positive")
        for name in ("action_rate_hz", "torque_rate_hz"):
            if type(getattr(self, name)) is not int:
                raise ValueError(f"{name} must be an integer to preserve nanosecond arithmetic")
        if self.action_rate_hz != 100 or self.torque_rate_hz != 300:
            raise ValueError("this version requires 100 Hz actions and 300 Hz torque slots")
        if tuple(self.joint_names) != tuple(f"Joint{i}_R" for i in range(1, 8)):
            raise ValueError("joint_names must retain Joint1_R through Joint7_R order")
        if len(self.camera_names) != 2 or len(set(self.camera_names)) != 2:
            raise ValueError("exactly two distinct camera names are required")
        for name in ("image_height", "image_width", "max_samples_per_interval"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.max_samples_per_interval < self.samples_per_action:
            raise ValueError("sample capacity must include all three future torque slots")
        for name in ("history_duration_s", "state_max_age_ms", "camera_max_age_ms",
                     "action_match_tolerance_ms", "torque_match_tolerance_ms"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isclose(self.history_duration_s * self.action_rate_hz,
                            round(self.history_duration_s * self.action_rate_hz), abs_tol=1e-8):
            raise ValueError("history duration must contain a whole number of action intervals")
        if self.history_intervals < 1:
            raise ValueError("history duration must include at least one action interval")
        if self.action_match_tolerance_ms * 1e6 > self.action_period_ns / 2:
            raise ValueError("action tolerance must not exceed half an action period")
        if self.torque_match_tolerance_ms * 1e6 > 1e9 / self.torque_rate_hz / 2:
            raise ValueError("torque tolerance must not exceed half a torque period")

    @property
    def force_dim(self):
        return {"flange_force3": 3, "joint_torque7": 7, "end_effector_wrench6": 6,
                "end_effector_force3": 3}[self.force_representation]

    @property
    def uses_wrench_sensor(self):
        return self.force_representation in {"end_effector_wrench6", "end_effector_force3"}

    @property
    def raw_schema_version(self):
        return WRENCH_SCHEMA_VERSION if self.uses_wrench_sensor else RAW_SCHEMA_VERSION

    @property
    def force_timestamp_stream(self):
        return "ft_sensor" if self.uses_wrench_sensor else "disturbance"

    @property
    def action_period_ns(self):
        return 1_000_000_000 // self.action_rate_hz

    @property
    def history_intervals(self):
        return round(self.history_duration_s * self.action_rate_hz)

    @property
    def samples_per_action(self):
        return self.torque_rate_hz // self.action_rate_hz

    @property
    def state_max_age_ns(self):
        return round(self.state_max_age_ms * 1e6)

    @property
    def camera_max_age_ns(self):
        return round(self.camera_max_age_ms * 1e6)

    def to_dict(self):
        values = asdict(self)
        for name in ("wrench_component_units", "wrench_frame", "force_std_floor"):
            if values[name] is None:
                values.pop(name)
        if self.force_representation == "joint_torque7":
            values.pop("force_representation")  # Exact compatibility with existing checkpoints.
        if self.action_grid_phase_mode == "first_action":
            # Preserve the exact serialized v1 config for existing checkpoints.
            values.pop("action_grid_phase_mode")
            values.pop("action_grid_phase_window_s")
        return values

    @classmethod
    def from_dict(cls, values):
        values = dict(values)
        if "action_grid_phase_mode" not in values:
            values["action_grid_phase_mode"] = "first_action"
        for name in ("camera_names", "joint_names", "wrench_component_units", "force_std_floor"):
            if name in values and values[name] is not None:
                values[name] = tuple(values[name])
        return cls(**values)

    def contract_dict(self):
        version = (LEGACY_REAL_ROBOT_SAMPLING_VERSION if self.action_grid_phase_mode == "first_action"
                   else REAL_ROBOT_SAMPLING_VERSION)
        result = {"sampling_version": version,
                "raw_schema_version": RAW_SCHEMA_VERSION,
                "contact_representation": "joint_external_torque",
                "force_source_topic": EXTERNAL_TORQUE_TOPIC,
                "force_source_field": EXTERNAL_TORQUE_FIELD,
                "force_timestamp_stream": "disturbance",
                "gravity_compensation": "unverified", "internal_torque_input": False,
                "qvel_input": False,
                "qpos_unit": "rad", "action_unit": "rad", "torque_unit": "N*m",
                "joint_names": tuple(self.joint_names), "camera_names": tuple(self.camera_names),
                "image_preprocessing": "RGB uint8 / 255; bilinear antialias resize to model dimensions; optional ImageNet normalization from model config",
                "synthetic_terminal_hold": False,
                "input_clock_semantics": "recorder availability proxy; headers are diagnostics only",
                "config": self.to_dict()}
        if self.force_representation == "flange_force3":
            from .flange_force import conversion_contract
            result.update(contact_representation="flange_force3", force_unit="N",
                          force_dim=3, force_conversion=conversion_contract(),
                          input_clock_semantics="recorder availability proxy; exact header identity pairs source frames; force available at max(source log times)")
        if self.uses_wrench_sensor:
            result.update(raw_schema_version=WRENCH_SCHEMA_VERSION,
                          contact_representation=self.force_representation, force_dim=self.force_dim,
                          primary_force_dataset="observations/ee_wrench", force_source_topic=WRENCH_TOPIC,
                          force_timestamp_stream="ft_sensor", gravity_compensation="applied_upstream",
                          force_axes=WRENCH_AXES[:self.force_dim], force_component_units=WRENCH_UNITS[:self.force_dim],
                          force_coordinate_frame=self.wrench_frame,
                          force_reference_point="sensor measurement reference; no spatial transform applied",
                          force_units_provenance="explicit training configuration; raw source_native metadata retained",
                          normalization_policy="training native samples once; per-axis z-score with configured floors; no clipping",
                          additional_gravity_compensation=False)
            if self.force_representation == "end_effector_force3":
                result.pop("torque_unit")
                result.update(force_unit="N", source_force_axes=WRENCH_AXES,
                              source_force_component_units=WRENCH_UNITS, force_channel_indices=(0, 1, 2),
                              channel_selection="observations/ee_wrench[:, :3]; before normalization and encoding",
                              moment_input=False, moment_target=False)
        return result
