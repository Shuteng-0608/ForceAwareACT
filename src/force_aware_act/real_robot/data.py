"""Model-ready views of original asynchronous raw HDF5 streams."""

import numpy as np
import torch
import torch.nn.functional as functional
from torch.utils.data import Dataset

from force_aware_act.act_aligned_training.high_rate_data import ACTAlignedHighRateSample
from force_aware_act.models.official_act.config import OfficialACTConfig
from .config import RealRobotDataConfig
from .reader import RawEpisode
from .sampling import (future_torque_slot_indices, matching_error_summary,
                       pack_future_torque, pack_torque_history)


def preprocess_images(images, model_config):
    """Shared offline/online RGB preparation, with no channel or camera reordering."""
    tensor = torch.stack([torch.from_numpy(np.array(image, dtype=np.uint8, copy=True)).permute(2, 0, 1)
                          for image in images]).float().div_(255.0)
    size = (model_config.image_height, model_config.image_width)
    if tensor.shape[-2:] != size:
        tensor = functional.interpolate(tensor, size=size, mode="bilinear", align_corners=False, antialias=True)
    # Official ACT normalizes inside its visual backbone; its flag must not
    # cause a second normalization in this shared input adapter.
    if not isinstance(model_config, OfficialACTConfig) and getattr(model_config, "imagenet_normalize", False):
        mean = tensor.new_tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
        std = tensor.new_tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
        tensor = (tensor - mean) / std
    return tensor


class RealRobotHighRateDataset(Dataset):
    """Use the same valid observation anchors for force-aware and ACT baselines."""

    def __init__(self, data_root, episodes, normalization, model_config, data_config=None):
        self.data_root = data_root
        self.episodes = tuple(episodes)
        self.normalization = normalization
        self.model_config = model_config
        self.data_config = data_config or RealRobotDataConfig()
        if not self.episodes:
            raise ValueError("dataset requires at least one raw episode")
        if (model_config.q_dim, model_config.action_dim, model_config.force_dim) != (7, 7, self.data_config.force_dim):
            raise ValueError("right-arm dimensions must be qpos=action=7 and force match its representation")
        if model_config.num_cameras != len(self.data_config.camera_names):
            raise ValueError("model camera count does not match raw data contract")
        if model_config.max_online_force_intervals != self.data_config.history_intervals:
            raise ValueError("model history interval count does not match real time duration")
        if model_config.max_force_samples_per_interval != self.data_config.max_samples_per_interval:
            raise ValueError("model force sample capacity does not match data contract")
        if model_config.force_sample_rate_hz != self.data_config.torque_rate_hz:
            raise ValueError("model force sample rate does not match 300 Hz data contract")
        if model_config.policy_sample_rate_hz != self.data_config.action_rate_hz:
            raise ValueError("model action interval rate does not match the 100 Hz output grid")
        if model_config.online_force_window_len != round(self.data_config.history_duration_s * self.data_config.torque_rate_hz):
            raise ValueError("model nominal history length does not match the configured duration")
        for name in ("qpos_mean", "qpos_std", "action_mean", "action_std", "force_mean", "force_std"):
            values = np.asarray(getattr(normalization, name))
            dimension = self.data_config.force_dim if name.startswith("force_") else 7
            if values.shape != (dimension,) or not np.isfinite(values).all():
                raise ValueError(f"{name} must contain {dimension} finite real-robot statistics")
            if name.endswith("std") and np.any(values <= 0):
                raise ValueError(f"{name} must be positive")
        self.raw_episodes = tuple(RawEpisode(record.resolve(data_root), self.data_config) for record in self.episodes)
        self.index = tuple((i, int(anchor_index)) for i, episode in enumerate(self.raw_episodes)
                           for anchor_index in episode.valid_anchor_indices)
        if not self.index:
            raise ValueError("no valid anchors remain; inspect clocks and observation age thresholds")

    def __len__(self):
        return len(self.index)

    @property
    def diagnostics(self):
        return {record.episode_id: episode.diagnostics for record, episode in zip(self.episodes, self.raw_episodes)}

    def close(self):
        pass  # no persistent HDF5 handles cross DataLoader process boundaries

    def sample_trace(self, item):
        """Read no image pixels: expose one decision and its supervised slot mapping."""
        episode_index, timestep = self.index[item]
        episode = self.raw_episodes[episode_index]
        anchor = int(episode.anchor_timestamps_ns[timestep])
        count = self.model_config.chunk_len
        force_count = self.model_config.force_prediction_len
        strict_horizon = self.model_config.bounds_force_source_timestamps
        action_indices = np.full(count, -1, dtype=np.int64)
        slots = episode.action_slot_indices[timestep:timestep + count]
        action_indices[:len(slots)] = slots
        action_targets = anchor + np.arange(count, dtype=np.int64) * self.data_config.action_period_ns
        action_trace = matching_error_summary(episode.action_timestamps_ns, action_targets, action_indices)
        action_trace["hdf5_indices"] = action_indices.tolist()
        action_trace["valid_slots"] = (action_indices >= 0).tolist()
        force_indices, force_targets = future_torque_slot_indices(
            episode.torque_timestamps_ns, anchor, force_count, self.data_config, strict_horizon=strict_horizon)
        force_trace = matching_error_summary(episode.torque_timestamps_ns, force_targets, force_indices)
        force_trace["prediction_steps"] = force_count
        force_trace["prediction_duration_s"] = force_count / self.data_config.action_rate_hz
        force_trace["valid_endpoint_count"] = int((force_indices[:, -1] >= 0).sum())
        force_trace["endpoint_padding_mask"] = (force_indices[:, -1] < 0).tolist()
        force_trace["first_interval"] = [
            {"slot": slot, "target_ns": int(force_targets[0, slot]),
             "hdf5_index": int(index),
             "source_timestamp_ns": None if index < 0 else int(episode.torque_timestamps_ns[index])}
            for slot, index in enumerate(force_indices[0])]
        history = pack_torque_history(episode.torque_timestamps_ns, episode.force, anchor, self.data_config)
        if self.data_config.force_dim == 3:
            future = pack_future_torque(episode.torque_timestamps_ns, episode.force, anchor, force_count, self.data_config, strict_horizon=strict_horizon)
            endpoint = self.data_config.samples_per_action - 1
            force_trace.update(
                valid_conversion_slots=(~future.sample_padding_mask[:, :self.data_config.samples_per_action]).tolist(),
                endpoint_padding_mask=future.sample_padding_mask[:, endpoint].tolist(),
                valid_endpoint_count=int((~future.sample_padding_mask[:, endpoint]).sum()),
                timestamp_semantics="max(raw disturbance log_ns, paired state log_ns)")
            for slot in force_trace["first_interval"]:
                index = slot["hdf5_index"]
                slot["paired_state_hdf5_index"] = None if index < 0 else int(episode.force.state_indices[index])
                slot["raw_disturbance_log_ns"] = None if index < 0 else int(episode.timestamps["disturbance"][index])
        return {"episode_id": self.episodes[episode_index].episode_id, "anchor_grid_index": timestep,
                "observation": episode.observation_trace(anchor), "action": action_trace,
                ("future_force" if self.data_config.force_dim == 3 else "future_torque"): force_trace,
                "history_actual_sample_count": history.valid_sample_count,
                "force_source_topic": episode.provenance["force_source_topic"],
                "force_timestamp_stream": "disturbance",
                "qpos_unit": "rad", "action_unit": "rad", "torque_unit": "N*m",
                **({"force_representation": "flange_force3", "force_unit": "N",
                    "force_conversion": episode.force.diagnostics,
                    "future_force_valid_samples": int((~future.sample_padding_mask).sum())}
                   if self.data_config.force_dim == 3 else {})}

    def __getitem__(self, item):
        episode_index, timestep = self.index[item]
        record = self.episodes[episode_index]
        episode = self.raw_episodes[episode_index]
        anchor = int(episode.anchor_timestamps_ns[timestep])
        state_index, _ = episode.observation_indices(anchor)
        chunk_len = self.model_config.chunk_len
        actions = np.zeros((chunk_len, 7), dtype=np.float32)
        action_mask = np.ones(chunk_len, dtype=np.bool_)
        slots = episode.action_slot_indices[timestep:timestep + chunk_len]
        for j, source_index in enumerate(slots):
            if source_index >= 0:
                actions[j] = episode.action[source_index]
                action_mask[j] = False
        action_tensor = self.normalization.normalize_action(torch.from_numpy(actions))
        action_tensor.masked_fill_(torch.from_numpy(action_mask).unsqueeze(-1), 0)
        history = pack_torque_history(episode.torque_timestamps_ns, episode.force, anchor, self.data_config)
        future = pack_future_torque(episode.torque_timestamps_ns, episode.force, anchor, self.model_config.force_prediction_len, self.data_config,
                                   strict_horizon=self.model_config.bounds_force_source_timestamps)

        def normalize_force(values, mask):
            normalized = self.normalization.normalize_force(torch.from_numpy(values.copy()))
            return normalized.masked_fill(torch.from_numpy(mask).unsqueeze(-1), 0)

        future_tensor = normalize_force(future.values, future.sample_padding_mask)
        endpoint_slot = self.data_config.samples_per_action - 1
        endpoint_mask = future.sample_padding_mask[:, endpoint_slot].copy()
        return ACTAlignedHighRateSample(
            images=preprocess_images(episode.images_at(anchor), self.model_config),
            qpos=self.normalization.normalize_qpos(torch.from_numpy(episode.qpos[state_index].astype(np.float32))),
            action_chunk=action_tensor, action_padding_mask=torch.from_numpy(action_mask),
            online_force_intervals=normalize_force(history.values, history.sample_padding_mask),
            online_force_relative_time=torch.from_numpy(history.relative_times),
            online_force_sample_padding_mask=torch.from_numpy(history.sample_padding_mask),
            online_force_interval_padding_mask=torch.from_numpy(history.interval_padding_mask),
            future_force_intervals=future_tensor,
            future_force_relative_time=torch.from_numpy(future.relative_times),
            future_force_sample_padding_mask=torch.from_numpy(future.sample_padding_mask),
            future_force_interval_padding_mask=torch.from_numpy(future.interval_padding_mask),
            future_force_target=future_tensor[:, endpoint_slot].clone(),
            future_force_target_padding_mask=torch.from_numpy(endpoint_mask),
            episode_id=record.episode_id, timestep=timestep,
            online_raw_valid_count=history.valid_sample_count)
