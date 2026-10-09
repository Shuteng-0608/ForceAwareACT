"""Fixed per-episode validation indices and macro/micro deployment metrics."""

import hashlib
import json

from force_aware_act.act_aligned_training.high_rate_data import collate_high_rate_samples
from .training import evaluate


def fixed_indices(dataset, stride=1):
    if type(stride) is not int or stride <= 0:
        raise ValueError("validation stride must be a positive integer")
    by_episode = {record.episode_id: [] for record in dataset.episodes}
    for index, (episode, _) in enumerate(dataset.index):
        by_episode[dataset.episodes[episode].episode_id].append(index)
    result = {}
    for episode, indices in by_episode.items():
        if not indices:
            raise ValueError(f"no valid evaluation anchors for {episode}")
        result[episode] = indices[::stride]
        if result[episode][-1] != indices[-1]:
            result[episode].append(indices[-1])
    return result


def index_manifest(dataset, indices, stride):
    entries = []
    for episode_id, locations in indices.items():
        for location in locations:
            episode, grid = dataset.index[location]
            entries.append({"dataset_index": location, "episode_id": episode_id,
                            "anchor_grid_index": grid, "anchor_log_ns": int(dataset.raw_episodes[episode].anchor_timestamps_ns[grid])})
    digest = hashlib.sha256(json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {"version": "real_robot_fixed_validation_v1", "stride": stride,
            "entries": entries, "indices_sha256": digest}


def batches_for_indices(dataset, indices, batch_size):
    for start in range(0, len(indices), batch_size):
        yield collate_high_rate_samples([dataset[i] for i in indices[start:start + batch_size]])


def aggregate_episode_metrics(per_episode):
    if not per_episode:
        raise ValueError("evaluation requires at least one episode")
    result = {"windows": sum(row["windows"] for row in per_episode.values()), "per_episode": per_episode}
    force3 = next(iter(per_episode.values())).get("force_representation") == "flange_force3"
    metrics = (("action", "rad"), ("force", "n"), ("force_endpoint", "n")) if force3 else (("action", "rad"), ("torque", "nm"), ("torque_endpoint", "nm"))
    for name, unit in metrics:
        key, count_key = f"{name}_mae_{unit}", f"{name}_valid_scalars"
        values = [row[key] for row in per_episode.values() if row[key] is not None]
        count = sum(row[count_key] for row in per_episode.values())
        result[count_key] = count
        result[key] = sum(row[key] * row[count_key] for row in per_episode.values() if row[key] is not None) / count if count else None
        result[f"episode_macro_{key}"] = sum(values) / len(values) if values else None
    if force3:
        result.update(force_representation="flange_force3", force_unit="N")
        for name in ("force", "force_endpoint"):
            key = f"{name}_mae_per_axis_n"
            rows = [row for row in per_episode.values() if row.get(key) is not None]
            count = sum(row[f"{name}_valid_scalars"] for row in rows)
            result[key] = [sum(row[key][i] * row[f"{name}_valid_scalars"] for row in rows) / count for i in range(3)] if count else None
    holds = [row["hold_current_qpos_mae_rad"] for row in per_episode.values() if "hold_current_qpos_mae_rad" in row]
    if holds:
        result["episode_macro_hold_current_qpos_mae_rad"] = sum(holds) / len(holds)
    return result


def evaluate_episodes(kind, model, dataset, indices, normalization, device, batch_size, *, detailed=False):
    per_episode = {
        episode: evaluate(kind, model, batches_for_indices(dataset, locations, batch_size), normalization, device, detailed=detailed)
        for episode, locations in indices.items()
    }
    return aggregate_episode_metrics(per_episode)
