"""Causal real-robot observation buffers and timestamped action execution.

This module does not publish robot commands.  Message availability and action
execution time are explicit, so offline replay exercises the same rules as a
future hardware integration without assuming ideal sensor arrival rates.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import torch


class ObservationUnavailableError(ValueError):
    """A causal observation is missing or older than the configured limit."""


class ActionUnavailableError(ValueError):
    """No available predicted chunk covers the requested execution tick."""


def _timestamp(value: int, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer nanosecond timestamp")
    return int(value)


def _vector7(value: np.ndarray, name: str) -> np.ndarray:
    values = np.asarray(value)
    if values.shape != (7,) or values.dtype.kind not in "iuf":
        raise ValueError(f"{name} must have shape [7]")
    if not np.isfinite(values).all():
        raise ValueError(f"{name} must contain finite values")
    return values.astype(np.float32, copy=True)


@dataclass(frozen=True)
class _StateMessage:
    timestamp_ns: int
    available_ns: int
    qpos: np.ndarray


@dataclass(frozen=True)
class _TorqueMessage:
    timestamp_ns: int
    available_ns: int
    torque: np.ndarray
    valid: bool = True


@dataclass(frozen=True)
class _ImageMessage:
    timestamp_ns: int
    available_ns: int
    rgb: np.ndarray


@dataclass(frozen=True)
class RealRobotObservation:
    """One unnormalized causal observation plus its traceable source times."""

    anchor_ns: int
    images: tuple[np.ndarray, ...]
    image_timestamps_ns: tuple[int, ...]
    qpos: np.ndarray
    state_timestamp_ns: int
    torque_timestamps_ns: np.ndarray
    torque_values: np.ndarray
    torque_valid_mask: np.ndarray | None = None


class RealRobotObservationBuffer:
    """Append independent joint-state, external disturbance and RGB streams.

    ``timestamp_ns`` is in the clock selected by the data configuration.
    ``available_ns`` is the receive/record time in the *same clock domain* as
    the replay anchor; it prevents header-stamped, late-arriving messages from
    entering an earlier observation.  The default log clock uses the same
    timestamp for both.  Unknown clock offsets must be resolved before using
    a different clock, never inferred inside this buffer.
    """

    def __init__(self, config: Any, *, state_capacity: int = 4096,
                 image_capacity: int = 128) -> None:
        for name, value in (("state_capacity", state_capacity),
                            ("image_capacity", image_capacity)):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if state_capacity < config.history_intervals * config.history_sample_capacity:
            raise ValueError("state_capacity cannot truncate a full configured history window")
        self.config = config
        self._states: deque[_StateMessage] = deque(maxlen=state_capacity)
        self._torques: deque[_TorqueMessage] = deque(maxlen=state_capacity)
        self._images: dict[str, deque[_ImageMessage]] = {
            name: deque(maxlen=image_capacity) for name in config.camera_names
        }

    @staticmethod
    def _message_times(timestamp_ns: int, available_ns: int | None,
                       previous: Any | None) -> tuple[int, int]:
        timestamp_ns = _timestamp(timestamp_ns, "timestamp_ns")
        available_ns = (timestamp_ns if available_ns is None
                        else _timestamp(available_ns, "available_ns"))
        if previous is not None:
            if timestamp_ns <= previous.timestamp_ns:
                raise ValueError("message timestamps must be strictly increasing")
            if available_ns < previous.available_ns:
                raise ValueError("message availability timestamps must not decrease")
        return timestamp_ns, available_ns

    def append_state(self, timestamp_ns: int, qpos: np.ndarray,
                     external_joint_torque: np.ndarray | None = None, *,
                     available_ns: int | None = None) -> None:
        """Append qpos. Optional torque preserves the explicit combined API.

        The optional vector must already be an external estimate with the same
        timestamps. Never pass joint_states.effort here. Independent ROS topics
        must call append_external_torque with their own timestamps instead.
        """
        timestamp_ns, available_ns = self._message_times(
            timestamp_ns, available_ns, self._states[-1] if self._states else None
        )
        source_qpos = qpos
        qpos = _vector7(qpos, "qpos")
        if external_joint_torque is not None:
            # Validate the second stream before mutating either buffer.
            self.append_external_torque(timestamp_ns, external_joint_torque,
                                        available_ns=available_ns, qpos=source_qpos)
        self._states.append(_StateMessage(timestamp_ns, available_ns, qpos))

    def append_external_torque(self, timestamp_ns: int, external_joint_torque: np.ndarray,
                               *, available_ns: int | None = None,
                               qpos: np.ndarray | None = None) -> None:
        """Append estimated_joint_disturbance.effort, preserving its own clock."""
        timestamp_ns, available_ns = self._message_times(
            timestamp_ns, available_ns, self._torques[-1] if self._torques else None
        )
        torque = _vector7(external_joint_torque, "external_joint_torque")
        valid = True
        if self.config.force_dim == 3:
            if qpos is None:
                raise ValueError("flange_force3 requires same-frame qpos with external torque")
            from force_aware_act.real_robot.flange_force import joint_torque_to_flange_force
            # Preserve float64 source precision through inversion, then cast model input.
            result = joint_torque_to_flange_force(qpos, external_joint_torque)
            torque, valid = result.force.astype(np.float32), bool(result.valid)
        self._torques.append(_TorqueMessage(timestamp_ns, available_ns, torque, valid))

    def append_image(self, camera_name: str, timestamp_ns: int, rgb: np.ndarray,
                     *, available_ns: int | None = None) -> None:
        if camera_name not in self._images:
            raise ValueError(f"unknown camera: {camera_name!r}")
        messages = self._images[camera_name]
        timestamp_ns, available_ns = self._message_times(
            timestamp_ns, available_ns, messages[-1] if messages else None
        )
        image = np.asarray(rgb)
        expected = (self.config.image_height, self.config.image_width, 3)
        if image.shape != expected or image.dtype != np.uint8:
            raise ValueError(f"{camera_name} RGB must be uint8 with shape {expected}")
        messages.append(_ImageMessage(timestamp_ns, available_ns, image.copy()))

    def observation_at(self, anchor_ns: int) -> RealRobotObservation:
        from force_aware_act.real_robot.sampling import causal_index

        anchor_ns = _timestamp(anchor_ns, "anchor_ns")
        # Filtering on availability is required even when replay buffers already
        # contain messages newer than the requested observation time.
        states = [item for item in self._states if item.available_ns <= anchor_ns]
        state_times = np.asarray([item.timestamp_ns for item in states], dtype=np.int64)
        state_index = causal_index(state_times, anchor_ns, self.config.state_max_age_ns)
        if state_index is None:
            raise ObservationUnavailableError("joint state is missing or stale")
        images = []
        image_times = []
        for name in self.config.camera_names:
            messages = [item for item in self._images[name]
                        if item.available_ns <= anchor_ns]
            times = np.asarray([item.timestamp_ns for item in messages], dtype=np.int64)
            index = causal_index(times, anchor_ns, self.config.camera_max_age_ns)
            if index is None:
                raise ObservationUnavailableError(f"camera {name} is missing or stale")
            images.append(messages[index].rgb.copy())
            image_times.append(messages[index].timestamp_ns)
        history_start = anchor_ns - round(self.config.history_duration_s * 1e9)
        history = [item for item in self._torques
                   if item.available_ns <= anchor_ns
                   and history_start < item.timestamp_ns <= anchor_ns]
        return RealRobotObservation(
            anchor_ns=anchor_ns,
            images=tuple(images),
            image_timestamps_ns=tuple(image_times),
            qpos=states[state_index].qpos.copy(),
            state_timestamp_ns=states[state_index].timestamp_ns,
            torque_timestamps_ns=np.asarray([item.timestamp_ns for item in history], dtype=np.int64),
            torque_values=(np.stack([item.torque for item in history]) if history
                           else np.empty((0, self.config.force_dim), dtype=np.float32)),
            torque_valid_mask=(np.asarray([item.valid for item in history], dtype=np.bool_)
                               if self.config.force_dim == 3 else None),
        )


def validate_data_contract(contract: Mapping[str, Any], config: Any) -> None:
    """Reject a checkpoint whose signal/order/clock contract has changed."""

    def canonical(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {key: canonical(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [canonical(item) for item in value]
        return value

    if not isinstance(contract, Mapping):
        raise ValueError("checkpoint data_contract must be a mapping")
    expected = canonical(config.contract_dict())
    actual = canonical(contract)
    if actual != expected:
        keys = sorted(key for key in set(actual) | set(expected)
                      if actual.get(key) != expected.get(key))
        raise ValueError(f"checkpoint data contract mismatch: {', '.join(keys)}")


@dataclass(frozen=True)
class RealRobotPolicyInputs:
    images: torch.Tensor
    qpos: torch.Tensor
    source: RealRobotObservation
    online_force_intervals: torch.Tensor | None = None
    online_force_relative_time: torch.Tensor | None = None
    online_force_sample_padding_mask: torch.Tensor | None = None
    online_force_interval_padding_mask: torch.Tensor | None = None


class RealRobotPolicyAdapter:
    """Use the offline image/history transforms for causal model inference."""

    def __init__(self, model: torch.nn.Module, *, model_kind: str,
                 normalization: Mapping[str, Any], data_config: Any,
                 data_contract: Mapping[str, Any]) -> None:
        if model_kind not in {"contact", "motion", "dual-zero", "act"}:
            raise ValueError("unsupported real-robot model kind")
        validate_data_contract(data_contract, data_config)
        if data_config.timestamp_key != "log_ns":
            raise ValueError("real-robot replay currently requires the confirmed log_ns clock")
        self.model = model
        self.model_kind = model_kind
        self.normalization = dict(normalization)
        self.data_config = data_config
        self._validate_model_contract()
        self.model.eval()

    @classmethod
    def from_checkpoint(cls, checkpoint: Mapping[str, Any], *,
                        device: torch.device | str = "cpu",
                        expected_data_config: Any | None = None) -> "RealRobotPolicyAdapter":
        from force_aware_act.real_robot.config import RealRobotDataConfig
        from force_aware_act.real_robot.training import build_model

        if checkpoint.get("format_version") != "real_robot_training_v1":
            raise ValueError("unsupported real-robot checkpoint format")
        config = RealRobotDataConfig.from_dict(checkpoint["data_config"])
        validate_data_contract(checkpoint["data_contract"], config)
        if expected_data_config is not None:
            validate_data_contract(checkpoint["data_contract"], expected_data_config)
        model = build_model(checkpoint["model_kind"], checkpoint["model_config"],
                            initialize_pretrained=False)
        model.load_state_dict(checkpoint["model_state"], strict=True)
        model.to(device)
        return cls(model, model_kind=checkpoint["model_kind"],
                   normalization=checkpoint["normalization"], data_config=config,
                   data_contract=checkpoint["data_contract"])

    def _validate_model_contract(self) -> None:
        config = self.model.config
        if config.q_dim != 7 or config.action_dim != 7 or config.num_cameras != 2:
            raise ValueError("real-robot model requires seven joints/actions and two cameras")
        dimensions = {"qpos": 7, "action": 7}
        if self.model_kind != "act":
            dimensions["force"] = self.data_config.force_dim
            if config.force_dim != self.data_config.force_dim:
                raise ValueError("real-robot force dimension differs from its representation")
            if config.force_sample_rate_hz != self.data_config.torque_rate_hz:
                raise ValueError("model torque rate differs from the data contract")
            if config.policy_sample_rate_hz != self.data_config.action_rate_hz:
                raise ValueError("model action rate differs from the data contract")
            if config.max_online_force_intervals != self.data_config.history_intervals:
                raise ValueError("model history intervals differ from the data contract")
            nominal_history_samples = round(self.data_config.history_duration_s * self.data_config.torque_rate_hz)
            if config.online_force_window_len != nominal_history_samples:
                raise ValueError("model nominal history sample count differs from the data contract")
            if config.max_force_samples_per_interval != self.data_config.max_samples_per_interval:
                raise ValueError("model interval capacity differs from the data contract")
            if config.online_force_sample_capacity != self.data_config.history_sample_capacity:
                raise ValueError("model online capacity differs from the data contract")
            if round(config.online_force_interval_s * 1e9) != self.data_config.history_period_ns:
                raise ValueError("model history interval duration differs from the data contract")
        for prefix, dimension in dimensions.items():
            for suffix in ("mean", "std"):
                values = np.asarray(self.normalization.get(f"{prefix}_{suffix}"), dtype=np.float64)
                if values.shape != (dimension,) or not np.isfinite(values).all():
                    raise ValueError(f"invalid {prefix}_{suffix} normalization")
                if suffix == "std" and np.any(values <= 0):
                    raise ValueError(f"{prefix}_std must be positive")

    def _normalize(self, values: torch.Tensor, prefix: str) -> torch.Tensor:
        mean = values.new_tensor(self.normalization[f"{prefix}_mean"])
        std = values.new_tensor(self.normalization[f"{prefix}_std"])
        return (values - mean) / std

    def prepare_observation(self, buffer: RealRobotObservationBuffer,
                            anchor_ns: int) -> RealRobotPolicyInputs:
        validate_data_contract(buffer.config.contract_dict(), self.data_config)
        return self.prepare_snapshot(buffer.observation_at(anchor_ns))

    def prepare_snapshot(self, observation: RealRobotObservation) -> RealRobotPolicyInputs:
        """Preprocess a copied observation after releasing the receive-buffer lock.

        The snapshot must come from a buffer configured with this adapter's
        data_config. The ordinary prepare_observation API checks that contract.
        Snapshot selection/copying is the only operation needing the receive
        lock; image transforms, history packing and device copies do not.
        """
        from force_aware_act.real_robot.data import preprocess_images
        from force_aware_act.real_robot.sampling import pack_torque_history

        parameter = next(self.model.parameters())
        images = preprocess_images(observation.images, self.model.config).to(
            device=parameter.device, dtype=parameter.dtype
        ).unsqueeze(0)
        qpos = torch.as_tensor(observation.qpos, device=parameter.device,
                               dtype=parameter.dtype)
        values: dict[str, Any] = {
            "images": images,
            "qpos": self._normalize(qpos, "qpos").unsqueeze(0),
            "source": observation,
        }
        if self.model_kind != "act":
            force_values = observation.torque_values
            if observation.torque_valid_mask is not None:
                class MaskedForce:
                    def read(self, indices):
                        return observation.torque_values[indices], observation.torque_valid_mask[indices]
                force_values = MaskedForce()
            packed = pack_torque_history(observation.torque_timestamps_ns,
                                         force_values,
                                         observation.anchor_ns, self.data_config)
            if packed.sample_padding_mask.all():
                raise ObservationUnavailableError("no causal torque sample exists in the history window")
            sample_mask = torch.as_tensor(packed.sample_padding_mask,
                                           device=parameter.device, dtype=torch.bool)
            torque = torch.as_tensor(packed.values, device=parameter.device,
                                      dtype=parameter.dtype)
            torque = self._normalize(torque, "force").masked_fill(sample_mask.unsqueeze(-1), 0)
            values.update({
                "online_force_intervals": torque.unsqueeze(0),
                "online_force_relative_time": torch.as_tensor(
                    packed.relative_times, device=parameter.device,
                    dtype=parameter.dtype).unsqueeze(0),
                "online_force_sample_padding_mask": sample_mask.unsqueeze(0),
                "online_force_interval_padding_mask": torch.as_tensor(
                    packed.interval_padding_mask, device=parameter.device,
                    dtype=torch.bool).unsqueeze(0),
            })
        return RealRobotPolicyInputs(**values)

    def predict(self, buffer: RealRobotObservationBuffer, anchor_ns: int, *,
                contact_latent_mode: str = "zero") -> dict[str, np.ndarray]:
        if contact_latent_mode not in {"zero", "prior"}:
            raise ValueError("deployment contact mode must be zero or prior")
        if self.model_kind != "contact" and contact_latent_mode != "zero":
            raise ValueError("only the contact model supports prior inference")
        inputs = self.prepare_observation(buffer, anchor_ns)
        return self.predict_prepared(inputs, contact_latent_mode=contact_latent_mode)

    def predict_prepared(self, inputs: RealRobotPolicyInputs, *,
                         contact_latent_mode: str = "zero") -> dict[str, np.ndarray]:
        """Predict after a locked snapshot; acquisition can continue during GPU work."""
        if contact_latent_mode not in {"zero", "prior"}:
            raise ValueError("deployment contact mode must be zero or prior")
        if self.model_kind != "contact" and contact_latent_mode != "zero":
            raise ValueError("only the contact model supports prior inference")
        anchor_ns = inputs.source.anchor_ns
        with torch.inference_mode():
            if self.model_kind == "act":
                output = self.model(inputs.images, inputs.qpos)
            else:
                arguments = (inputs.images, inputs.qpos,
                             inputs.online_force_intervals,
                             inputs.online_force_relative_time,
                             inputs.online_force_sample_padding_mask,
                             inputs.online_force_interval_padding_mask)
                if self.model_kind == "contact":
                    output = self.model(*arguments,
                                        contact_latent_mode=contact_latent_mode,
                                        deterministic_prior=True)
                else:
                    output = self.model(*arguments)
        result = {}
        for key, prefix in (("pred_action", "action"), ("pred_force", "force"),
                            ("pred_force_highrate", "force")):
            if key not in output:
                continue
            prediction = output[key]
            physical = (prediction * prediction.new_tensor(self.normalization[f"{prefix}_std"])
                        + prediction.new_tensor(self.normalization[f"{prefix}_mean"]))
            if key == "pred_force_highrate":
                # The shared head reserves capacity for irregular input bursts.
                # Only the first three fixed future slots have supervision in
                # this 300/100 Hz profile; extra head channels are not estimates.
                physical = physical[:, :, :self.data_config.samples_per_action]
            if not torch.isfinite(physical).all():
                raise FloatingPointError(f"{key} contains non-finite predictions")
            result[key] = physical.squeeze(0).cpu().numpy()
            if key == "pred_force_highrate":
                count = physical.shape[1] * self.data_config.samples_per_action
                offsets = np.rint(np.arange(1, count + 1, dtype=np.float64)
                                  * (1e9 / self.data_config.torque_rate_hz)).astype(np.int64)
                result["pred_force_highrate_timestamps_ns"] = (
                    int(anchor_ns) + offsets.reshape(physical.shape[1], self.data_config.samples_per_action)
                )
        return result


@dataclass(frozen=True)
class TimestampedAction:
    action: np.ndarray
    execution_tick: int
    prediction_start_tick: int
    chunk_index: int
    available_ns: int


@dataclass(frozen=True)
class _ActionChunk:
    start_tick: int
    available_ns: int
    actions: np.ndarray


class TimestampActionChunkBuffer:
    """Consume 100 Hz action predictions independently of model query rate.

    The observation/query scheduler supplies the chunk's first target tick.
    A late result skips elapsed elements instead of restarting at element zero.
    This supports 20/25 Hz queries and irregular or asynchronous 30 Hz queries;
    no resampling or terminal holding is performed.  Only an exact control-tick
    timestamp can request an action.
    """

    period_ns = 10_000_000

    def __init__(self, origin_ns: int, *, capacity: int = 128) -> None:
        self.origin_ns = _timestamp(origin_ns, "origin_ns")
        if not isinstance(capacity, int) or isinstance(capacity, bool) or capacity <= 0:
            raise ValueError("capacity must be a positive integer")
        self._chunks: deque[_ActionChunk] = deque(maxlen=capacity)

    def tick_at(self, timestamp_ns: int) -> int:
        elapsed = _timestamp(timestamp_ns, "timestamp_ns") - self.origin_ns
        if elapsed < 0 or elapsed % self.period_ns:
            raise ValueError("timestamp must lie on the 100 Hz execution grid")
        return elapsed // self.period_ns

    def submit(self, actions: np.ndarray, *, prediction_start_ns: int,
               available_ns: int) -> None:
        start_tick = self.tick_at(prediction_start_ns)
        available_ns = _timestamp(available_ns, "available_ns")
        values = np.asarray(actions)
        if values.ndim != 2 or values.shape[0] <= 0 or values.shape[1] != 7:
            raise ValueError("action chunk must have shape [K, 7] with K > 0")
        if values.dtype.kind not in "iuf" or not np.isfinite(values).all():
            raise ValueError("action chunk must contain finite numeric values")
        if available_ns < self.origin_ns:
            raise ValueError("chunk availability cannot precede the execution origin")
        self._chunks.append(_ActionChunk(start_tick, available_ns,
                                         values.astype(np.float64, copy=True)))

    def action_at(self, timestamp_ns: int) -> TimestampedAction:
        timestamp_ns = _timestamp(timestamp_ns, "timestamp_ns")
        tick = self.tick_at(timestamp_ns)
        candidates = [chunk for chunk in self._chunks
                      if chunk.available_ns <= timestamp_ns
                      and 0 <= tick - chunk.start_tick < len(chunk.actions)]
        if not candidates:
            raise ActionUnavailableError(
                f"no available action chunk covers execution tick {tick}"
            )
        # Newest observation wins even if an older query completes later.
        selected = max(candidates, key=lambda item: (item.start_tick, item.available_ns))
        index = tick - selected.start_tick
        return TimestampedAction(selected.actions[index].copy(), tick,
                                 selected.start_tick, index, selected.available_ns)
