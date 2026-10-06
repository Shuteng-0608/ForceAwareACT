#!/usr/bin/env python3
"""Replay raw HDF5 through causal inference buffers without commanding hardware."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import h5py
import numpy as np
import torch

from force_aware_act.inference.real_robot import RealRobotObservationBuffer, RealRobotPolicyAdapter
from force_aware_act.real_robot.checkpoint import read_checkpoint
from force_aware_act.real_robot.data import preprocess_images
from force_aware_act.real_robot.reader import RawEpisode
from force_aware_act.real_robot.sampling import pack_torque_history


def replay(checkpoint_path, episode_path, *, device="cpu", max_samples=10):
    """Verify offline/online input equivalence and infer at sampled valid anchors."""
    if max_samples <= 0:
        raise ValueError("max_samples must be positive")
    payload = read_checkpoint(checkpoint_path)
    adapter = RealRobotPolicyAdapter.from_checkpoint(payload, device=device)
    raw = RawEpisode(episode_path, adapter.data_config)
    if not len(raw.valid_anchor_ns):
        raise ValueError("episode has no valid replay anchors")
    selected = np.unique(np.linspace(0, len(raw.valid_anchor_ns) - 1,
                                     min(max_samples, len(raw.valid_anchor_ns)), dtype=int))
    buffer = RealRobotObservationBuffer(adapter.data_config)
    state_cursor = 0
    torque_cursor = 0
    image_cursors = dict.fromkeys(adapter.data_config.camera_names, 0)
    timings = []
    rows = []
    with h5py.File(raw.path, "r") as handle:
        for sample_index in selected:
            anchor = int(raw.valid_anchor_ns[sample_index])
            while state_cursor < len(raw.state_timestamps_ns) and raw.state_timestamps_ns[state_cursor] <= anchor:
                buffer.append_state(raw.state_timestamps_ns[state_cursor], raw.qpos[state_cursor])
                state_cursor += 1
            while torque_cursor < len(raw.torque_timestamps_ns) and raw.torque_timestamps_ns[torque_cursor] <= anchor:
                buffer.append_external_torque(raw.torque_timestamps_ns[torque_cursor], raw.torque[torque_cursor])
                torque_cursor += 1
            for camera in adapter.data_config.camera_names:
                cursor = image_cursors[camera]
                times = raw.timestamps[camera]
                while cursor < len(times) and times[cursor] <= anchor:
                    buffer.append_image(camera, times[cursor],
                                        handle[f"observations/images/{camera}/rgb"][cursor])
                    cursor += 1
                image_cursors[camera] = cursor
            inputs = adapter.prepare_observation(buffer, anchor)
            expected_images = preprocess_images(raw.images_at(anchor), adapter.model.config)
            torch.testing.assert_close(inputs.images[0].cpu(), expected_images)
            state_index, _ = raw.observation_indices(anchor)
            expected_qpos = (raw.qpos[state_index].astype(np.float32)
                             - np.asarray(adapter.normalization["qpos_mean"], dtype=np.float32))
            expected_qpos /= np.asarray(adapter.normalization["qpos_std"], dtype=np.float32)
            np.testing.assert_allclose(inputs.qpos[0].cpu().numpy(), expected_qpos, rtol=1e-6, atol=1e-6)
            valid_torque = None
            if adapter.model_kind != "act":
                packed = pack_torque_history(raw.torque_timestamps_ns, raw.torque, anchor, adapter.data_config)
                expected_torque = (packed.values - np.asarray(adapter.normalization["force_mean"], np.float32))
                expected_torque /= np.asarray(adapter.normalization["force_std"], np.float32)
                expected_torque[packed.sample_padding_mask] = 0
                np.testing.assert_allclose(inputs.online_force_intervals[0].cpu().numpy(), expected_torque,
                                           rtol=1e-6, atol=1e-6)
                np.testing.assert_array_equal(inputs.online_force_sample_padding_mask[0].cpu().numpy(),
                                              packed.sample_padding_mask)
                np.testing.assert_array_equal(inputs.online_force_relative_time[0].cpu().numpy(), packed.relative_times)
                valid_torque = int((~packed.sample_padding_mask).sum())
            if str(device).startswith("cuda"):
                torch.cuda.synchronize()
            start = time.perf_counter()
            output = adapter.predict(buffer, anchor)
            if str(device).startswith("cuda"):
                torch.cuda.synchronize()
            timings.append((time.perf_counter() - start) * 1000)
            rows.append({
                "anchor_ns": anchor, "state_age_ms": (anchor - inputs.source.state_timestamp_ns) * 1e-6,
                "image_age_ms": {name: (anchor - timestamp) * 1e-6
                                 for name, timestamp in zip(adapter.data_config.camera_names,
                                                            inputs.source.image_timestamps_ns)},
                "valid_torque_samples": valid_torque,
                "action_shape": list(output["pred_action"].shape),
                "first_predicted_action": output["pred_action"][0].tolist(),
            })
    return {
        "episode": str(raw.path), "checkpoint": str(Path(checkpoint_path).resolve()),
        "checkpoint_global_step": payload["global_step"],
        "model_kind": adapter.model_kind, "timestamp_key": adapter.data_config.timestamp_key,
        "force_source_topic": raw.provenance["force_source_topic"],
        "force_timestamp_stream": "disturbance",
        "raw_valid_anchors": len(raw.valid_anchor_ns), "replayed_anchors": len(rows),
        "offline_online_inputs_equal": True, "robot_commands_published": False,
        "timing_includes": "input preparation and model prediction; first call included; HDF5 reads excluded",
        "latency_ms": {"mean": float(np.mean(timings)), "max": float(np.max(timings))},
        "samples": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-samples", type=int, default=10)
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.torch_threads <= 0:
        parser.error("--torch-threads must be positive")
    torch.set_num_threads(args.torch_threads)
    result = replay(args.checkpoint, args.episode, device=args.device, max_samples=args.max_samples)
    serialized = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)


if __name__ == "__main__":
    main()
