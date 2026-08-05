#!/usr/bin/env python3
"""Audit the timestamp-level 500 Hz force contract on real HDF5 episodes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from force_aware_act.high_rate_force import (
    DEFAULT_HIGH_RATE_FORCE_CONTRACT,
    HIGH_RATE_FORCE_CONTRACT_VERSION,
    build_online_force_intervals,
)


def _frequency(timestamps: np.ndarray) -> float:
    if timestamps.shape[0] < 2:
        return float("nan")
    return float(1.0 / np.median(np.diff(timestamps)))


def audit_episode(path: Path) -> dict[str, object]:
    contract = DEFAULT_HIGH_RATE_FORCE_CONTRACT
    with h5py.File(path, "r") as handle:
        state_timestamps = np.asarray(handle["timestamps/state"], dtype=np.float64)
        force_timestamps = np.asarray(handle["timestamps/force"], dtype=np.float64)
        force_values = np.asarray(handle["observations/ft_wrench"], dtype=np.float32)

    starts = np.searchsorted(force_timestamps, state_timestamps[:-1], side="right")
    ends = np.searchsorted(force_timestamps, state_timestamps[1:], side="right")
    interval_counts = ends - starts
    assigned_count = int(interval_counts.sum())
    expected_assigned_count = int(
        np.count_nonzero(
            (force_timestamps > state_timestamps[0])
            & (force_timestamps <= state_timestamps[-1])
        )
    )

    online_valid_counts: list[int] = []
    online_grouped_counts: list[int] = []
    online_interval_counts: list[int] = []
    online_full_window_spans: list[float] = []
    for state_index in range(state_timestamps.shape[0]):
        window, packed = build_online_force_intervals(
            force_timestamps,
            force_values,
            state_timestamps,
            state_index=state_index,
            contract=contract,
        )
        online_valid_counts.append(window.valid_count)
        online_grouped_counts.append(packed.valid_sample_count)
        online_interval_counts.append(
            int((~packed.interval_padding_mask).sum())
        )
        if window.valid_count == contract.online_window_len:
            valid_timestamps = window.timestamps[~window.padding_mask]
            online_full_window_spans.append(
                float(valid_timestamps[-1] - valid_timestamps[0])
            )

    interval_histogram = {
        str(int(value)): int(np.count_nonzero(interval_counts == value))
        for value in np.unique(interval_counts)
    }

    return {
        "episode_id": path.parent.name,
        "num_state_samples": int(state_timestamps.shape[0]),
        "num_force_samples": int(force_timestamps.shape[0]),
        "state_rate_hz": _frequency(state_timestamps),
        "force_rate_hz": _frequency(force_timestamps),
        "action_intervals": int(interval_counts.shape[0]),
        "force_samples_per_action_min": int(interval_counts.min()),
        "force_samples_per_action_max": int(interval_counts.max()),
        "force_samples_per_action_values": sorted(
            int(value) for value in np.unique(interval_counts)
        ),
        "force_samples_per_action_histogram": interval_histogram,
        "assigned_force_samples": assigned_count,
        "expected_assigned_force_samples": expected_assigned_count,
        "assigned_exactly_once": assigned_count == expected_assigned_count,
        "online_valid_samples_min": min(online_valid_counts),
        "online_valid_samples_max": max(online_valid_counts),
        "online_grouping_preserved_all_samples": (
            online_valid_counts == online_grouped_counts
        ),
        "online_nonempty_intervals_min": min(online_interval_counts),
        "online_nonempty_intervals_max": max(online_interval_counts),
        "online_seven_interval_anchor_count": int(
            np.count_nonzero(np.asarray(online_interval_counts) == 7)
        ),
        "online_full_window_span_min": min(online_full_window_spans),
        "online_full_window_span_max": max(online_full_window_spans),
    }


def audit_collection(data_root: Path) -> dict[str, object]:
    episode_paths = sorted(data_root.glob("*/episode.hdf5"))
    if not episode_paths:
        raise FileNotFoundError(f"no episode.hdf5 files found below {data_root}")
    episodes = [audit_episode(path) for path in episode_paths]
    all_interval_values = sorted(
        {
            value
            for episode in episodes
            for value in episode["force_samples_per_action_values"]
        }
    )
    interval_histogram: dict[str, int] = {}
    for episode in episodes:
        for count, occurrences in episode[
            "force_samples_per_action_histogram"
        ].items():
            interval_histogram[count] = (
                interval_histogram.get(count, 0) + occurrences
            )
    failures = [
        episode["episode_id"]
        for episode in episodes
        if not episode["assigned_exactly_once"]
        or not episode["online_grouping_preserved_all_samples"]
        or episode["force_samples_per_action_max"]
        > DEFAULT_HIGH_RATE_FORCE_CONTRACT.max_samples_per_interval
        or episode["online_nonempty_intervals_max"]
        > DEFAULT_HIGH_RATE_FORCE_CONTRACT.max_online_intervals
    ]
    result = {
        "contract_version": HIGH_RATE_FORCE_CONTRACT_VERSION,
        "contract": {
            "sample_rate_hz": DEFAULT_HIGH_RATE_FORCE_CONTRACT.sample_rate_hz,
            "online_window_len": (
                DEFAULT_HIGH_RATE_FORCE_CONTRACT.online_window_len
            ),
            "max_samples_per_interval": (
                DEFAULT_HIGH_RATE_FORCE_CONTRACT.max_samples_per_interval
            ),
            "max_online_intervals": (
                DEFAULT_HIGH_RATE_FORCE_CONTRACT.max_online_intervals
            ),
            "future_interval_boundary": "(state_timestamp_j, state_timestamp_j+1]",
        },
        "data_root": str(data_root.resolve()),
        "episode_count": len(episodes),
        "state_rate_hz_min": min(item["state_rate_hz"] for item in episodes),
        "state_rate_hz_max": max(item["state_rate_hz"] for item in episodes),
        "force_rate_hz_min": min(item["force_rate_hz"] for item in episodes),
        "force_rate_hz_max": max(item["force_rate_hz"] for item in episodes),
        "force_samples_per_action_values": all_interval_values,
        "force_samples_per_action_histogram": interval_histogram,
        "force_samples_per_action_max": max(
            item["force_samples_per_action_max"] for item in episodes
        ),
        "online_nonempty_intervals_max": max(
            item["online_nonempty_intervals_max"] for item in episodes
        ),
        "online_seven_interval_anchor_count": sum(
            item["online_seven_interval_anchor_count"] for item in episodes
        ),
        "online_full_window_span_min": min(
            item["online_full_window_span_min"] for item in episodes
        ),
        "online_full_window_span_max": max(
            item["online_full_window_span_max"] for item in episodes
        ),
        "episodes_failing_contract": failures,
        "passed": not failures,
        "episodes": episodes,
    }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("data_root", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = audit_collection(args.data_root)
    payload = json.dumps(result, indent=2, sort_keys=True)
    print(payload)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
