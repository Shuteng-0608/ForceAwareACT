#!/usr/bin/env python3
"""Analyze posterior contact latents from a trained ForceAwareACT checkpoint."""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence

import h5py
import numpy as np
import torch
import torch.nn.functional as functional
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from force_aware_act.data import ContactForceHDF5Dataset, nearest_index, normalize_tensor  # noqa: E402
from force_aware_act.models import ForceAwareACTPolicy  # noqa: E402
from script_utils import resolve_episode_paths, validate_episode_paths  # noqa: E402


COLOR_COLUMNS = {
    "future_force_mean": ("future_force_mean_raw", "future force mean raw"),
    "future_force_mean_raw": ("future_force_mean_raw", "future force mean raw"),
    "future_force_mean_normalized": (
        "future_force_mean_normalized",
        "future force mean normalized",
    ),
    "future_force_max": ("force_norm_future_max", "future force max"),
    "force_delta": ("force_norm_future_delta", "future force mean - current force"),
    "current_force": ("force_norm_current", "current force"),
    "current_torque": ("torque_norm_current", "current torque"),
    "future_torque_mean": ("torque_norm_future_mean", "future torque mean"),
    "time": ("t_state", "time"),
    "loss_force_per_sample": ("loss_force_per_sample", "force loss per sample"),
    "episode_id": ("episode_numeric_id", "episode id"),
}


FORCE_BINS = ("low", "mid", "high")


@dataclass(frozen=True)
class SampleCandidate:
    episode_id: int
    local_index: int
    global_index: int
    future_force_mean_raw: Optional[float] = None
    force_bin: str = ""


class MultiEpisodeIndexedDataset(Dataset):
    """Return selected samples with global index and episode id."""

    def __init__(
        self,
        datasets: Sequence[Dataset],
        episode_paths: Sequence[Path],
        indices: Sequence[SampleCandidate],
    ) -> None:
        self.datasets = list(datasets)
        self.episode_paths = [str(path) for path in episode_paths]
        self.indices = list(indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> dict:
        candidate = self.indices[item]
        sample = dict(self.datasets[candidate.episode_id][candidate.local_index])
        sample["episode_id"] = self.episode_paths[candidate.episode_id]
        sample["episode_numeric_id"] = candidate.episode_id
        sample["dataset_index"] = candidate.global_index
        sample["sample_future_force_mean_raw"] = (
            float("nan")
            if candidate.future_force_mean_raw is None
            else candidate.future_force_mean_raw
        )
        sample["sample_force_bin"] = candidate.force_bin
        return sample


def _load_stats(path: Path) -> Dict[str, torch.Tensor]:
    stats = torch.load(path, map_location="cpu")
    if not isinstance(stats, dict):
        raise ValueError("normalization stats file must contain a dict")
    for key in ("qpos_mean", "qpos_std", "action_mean", "action_std", "force_mean", "force_std"):
        if key not in stats:
            raise KeyError(f"normalization stats missing required key: {key}")
        if not torch.is_tensor(stats[key]):
            raise ValueError(f"normalization stats {key!r} must be a torch.Tensor")
    return stats


def _normalize_batch(batch: dict, stats: Dict[str, torch.Tensor]) -> dict:
    normalized = dict(batch)
    normalized["qpos"] = normalize_tensor(batch["qpos"], stats["qpos_mean"], stats["qpos_std"])
    normalized["force_window"] = normalize_tensor(
        batch["force_window"],
        stats["force_mean"],
        stats["force_std"],
    )
    normalized["action_chunk"] = normalize_tensor(
        batch["action_chunk"],
        stats["action_mean"],
        stats["action_std"],
    )
    normalized["future_force_chunk"] = normalize_tensor(
        batch["future_force_chunk"],
        stats["force_mean"],
        stats["force_std"],
    )
    return normalized


def _model_kwargs_from_checkpoint(checkpoint: dict) -> dict:
    config = checkpoint.get("config", {})
    model_config = dict(config.get("model", {}))
    if not model_config:
        raise KeyError("checkpoint config is missing model settings")
    if "pretrained_vision" in model_config and "pretrained_resnet18" not in model_config:
        model_config["pretrained_resnet18"] = model_config.pop("pretrained_vision")
    model_config.setdefault("pretrained_resnet18", False)
    model_config.setdefault("dropout", 0.0)
    model_config.setdefault("max_force_window_len", max(int(config.get("force_window_len", 20)), 20))
    return model_config


def _dataset_kwargs_from_args(args: argparse.Namespace) -> dict:
    return {
        "camera_names": tuple(args.camera_names),
        "action_mode": "joint_pos",
        "chunk_len": args.chunk_len,
        "force_window_len": args.force_window_len,
        "force_window_duration": args.force_window_duration,
        "image_size": tuple(args.image_size),
        "imagenet_normalize": False,
    }


def _episode_offsets(datasets: Sequence[Dataset]) -> list[int]:
    offsets = []
    running_total = 0
    for dataset in datasets:
        offsets.append(running_total)
        running_total += len(dataset)
    return offsets


def _candidate_grid(datasets: Sequence[Dataset], stride: int) -> list[list[SampleCandidate]]:
    offsets = _episode_offsets(datasets)
    return [
        [
            SampleCandidate(
                episode_id=episode_id,
                local_index=local_index,
                global_index=offsets[episode_id] + local_index,
            )
            for local_index in range(0, len(dataset), stride)
        ]
        for episode_id, dataset in enumerate(datasets)
    ]


def _estimate_episode_future_force_means(
    dataset: ContactForceHDF5Dataset,
    candidates: Sequence[SampleCandidate],
) -> list[SampleCandidate]:
    if not candidates:
        return []

    with h5py.File(dataset.episode_paths[0], "r") as handle:
        state_ts = np.asarray(handle["timestamps/state_episode"])
        force_ts = np.asarray(handle["timestamps/force_episode"])
        ft_wrench = np.asarray(handle["observations/ft_wrench"])

    estimated = []
    for candidate in candidates:
        state_index = dataset.indices[candidate.local_index].state_index
        future_force_indices = np.array(
            [
                nearest_index(force_ts, float(state_ts[state_index + step]))
                for step in range(dataset.chunk_len)
            ],
            dtype=np.int64,
        )
        future_force_chunk = ft_wrench[future_force_indices]
        future_force_mean = float(np.linalg.norm(future_force_chunk[:, :3], axis=1).mean())
        estimated.append(
            SampleCandidate(
                episode_id=candidate.episode_id,
                local_index=candidate.local_index,
                global_index=candidate.global_index,
                future_force_mean_raw=future_force_mean,
                force_bin=candidate.force_bin,
            )
        )
    return estimated


def _assign_force_bins(candidates: Sequence[SampleCandidate]) -> list[SampleCandidate]:
    if not candidates:
        return []
    values = np.array([candidate.future_force_mean_raw for candidate in candidates], dtype=np.float64)
    finite_mask = np.isfinite(values)
    if not finite_mask.any():
        return [
            SampleCandidate(
                episode_id=candidate.episode_id,
                local_index=candidate.local_index,
                global_index=candidate.global_index,
                future_force_mean_raw=candidate.future_force_mean_raw,
                force_bin="unknown",
            )
            for candidate in candidates
        ]

    low_cut, high_cut = np.quantile(values[finite_mask], [1.0 / 3.0, 2.0 / 3.0])
    binned = []
    for candidate, value in zip(candidates, values):
        if not np.isfinite(value):
            force_bin = "unknown"
        elif value <= low_cut:
            force_bin = "low"
        elif value <= high_cut:
            force_bin = "mid"
        else:
            force_bin = "high"
        binned.append(
            SampleCandidate(
                episode_id=candidate.episode_id,
                local_index=candidate.local_index,
                global_index=candidate.global_index,
                future_force_mean_raw=candidate.future_force_mean_raw,
                force_bin=force_bin,
            )
        )
    return binned


def _with_force_estimates(
    datasets: Sequence[ContactForceHDF5Dataset],
    candidates_by_episode: Sequence[Sequence[SampleCandidate]],
) -> list[list[SampleCandidate]]:
    estimated_by_episode = []
    for dataset, candidates in zip(datasets, candidates_by_episode):
        estimated = _estimate_episode_future_force_means(dataset, candidates)
        estimated_by_episode.append(_assign_force_bins(estimated))
    return estimated_by_episode


def _force_stats(candidates: Sequence[SampleCandidate]) -> Optional[dict[str, float]]:
    values = np.array(
        [
            candidate.future_force_mean_raw
            for candidate in candidates
            if candidate.future_force_mean_raw is not None
        ],
        dtype=np.float64,
    )
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    return {
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "std": float(values.std()),
    }


def _print_force_stats(label: str, candidates: Sequence[SampleCandidate]) -> None:
    stats = _force_stats(candidates)
    if stats is None:
        print(f"{label}=unavailable")
        return
    print(
        f"{label}=min={stats['min']:.6g} "
        f"max={stats['max']:.6g} "
        f"mean={stats['mean']:.6g} "
        f"std={stats['std']:.6g}"
    )


def _force_bin_counts(candidates: Sequence[SampleCandidate]) -> dict[str, int]:
    counts = {force_bin: 0 for force_bin in FORCE_BINS}
    for candidate in candidates:
        if candidate.force_bin in counts:
            counts[candidate.force_bin] += 1
        elif candidate.force_bin:
            counts[candidate.force_bin] = counts.get(candidate.force_bin, 0) + 1
    return counts


def _flatten_candidates(candidates_by_episode: Sequence[Sequence[SampleCandidate]]) -> list[SampleCandidate]:
    return [
        candidate
        for candidates in candidates_by_episode
        for candidate in candidates
    ]


def _select_stratified_episode(
    candidates_by_episode: Sequence[Sequence[SampleCandidate]],
    max_samples: Optional[int],
) -> list[SampleCandidate]:
    if max_samples is None or len(candidates_by_episode) <= 1:
        indices = _flatten_candidates(candidates_by_episode)
        return indices if max_samples is None else indices[:max_samples]

    target_count = min(max_samples, sum(len(candidates) for candidates in candidates_by_episode))
    base_count = target_count // len(candidates_by_episode)
    remainder = target_count % len(candidates_by_episode)
    selected_by_episode: list[list[SampleCandidate]] = []
    next_position_by_episode = []

    for episode_id, candidates in enumerate(candidates_by_episode):
        requested = base_count + (1 if episode_id < remainder else 0)
        selected = list(candidates[:requested])
        selected_by_episode.append(selected)
        next_position_by_episode.append(len(selected))

    remaining = target_count - sum(len(selected) for selected in selected_by_episode)
    while remaining > 0:
        made_progress = False
        for episode_id, candidates in enumerate(candidates_by_episode):
            next_position = next_position_by_episode[episode_id]
            if next_position >= len(candidates):
                continue
            selected_by_episode[episode_id].append(candidates[next_position])
            next_position_by_episode[episode_id] += 1
            remaining -= 1
            made_progress = True
            if remaining == 0:
                break
        if not made_progress:
            break

    return _flatten_candidates(selected_by_episode)


def _take_force_balanced(candidates: Sequence[SampleCandidate], requested: int) -> list[SampleCandidate]:
    if requested <= 0 or not candidates:
        return []

    by_bin = {
        force_bin: [candidate for candidate in candidates if candidate.force_bin == force_bin]
        for force_bin in FORCE_BINS
    }
    selected_by_bin = {force_bin: [] for force_bin in FORCE_BINS}
    base_count = requested // len(FORCE_BINS)
    remainder = requested % len(FORCE_BINS)

    for bin_index, force_bin in enumerate(FORCE_BINS):
        bin_requested = base_count + (1 if bin_index < remainder else 0)
        selected_by_bin[force_bin] = by_bin[force_bin][:bin_requested]

    remaining = min(requested, len(candidates)) - sum(
        len(selected) for selected in selected_by_bin.values()
    )
    positions = {
        force_bin: len(selected_by_bin[force_bin])
        for force_bin in FORCE_BINS
    }
    while remaining > 0:
        made_progress = False
        for force_bin in FORCE_BINS:
            position = positions[force_bin]
            if position >= len(by_bin[force_bin]):
                continue
            selected_by_bin[force_bin].append(by_bin[force_bin][position])
            positions[force_bin] += 1
            remaining -= 1
            made_progress = True
            if remaining == 0:
                break
        if not made_progress:
            break

    selected = []
    for force_bin in FORCE_BINS:
        selected.extend(selected_by_bin[force_bin])
    return sorted(selected, key=lambda candidate: candidate.local_index)


def _select_force_balanced(
    candidates_by_episode: Sequence[Sequence[SampleCandidate]],
    max_samples: Optional[int],
) -> list[SampleCandidate]:
    total_count = sum(len(candidates) for candidates in candidates_by_episode)
    if max_samples is None:
        target_count = total_count
    else:
        target_count = min(max_samples, total_count)

    base_count = target_count // len(candidates_by_episode)
    remainder = target_count % len(candidates_by_episode)
    selected_by_episode = []
    for episode_id, candidates in enumerate(candidates_by_episode):
        requested = base_count + (1 if episode_id < remainder else 0)
        selected_by_episode.append(_take_force_balanced(candidates, requested))

    remaining = target_count - sum(len(selected) for selected in selected_by_episode)
    while remaining > 0:
        made_progress = False
        for episode_id, candidates in enumerate(candidates_by_episode):
            selected_ids = {candidate.local_index for candidate in selected_by_episode[episode_id]}
            extras = [
                candidate
                for candidate in candidates
                if candidate.local_index not in selected_ids
            ]
            if not extras:
                continue
            selected_by_episode[episode_id].append(extras[0])
            remaining -= 1
            made_progress = True
            if remaining == 0:
                break
        if not made_progress:
            break

    return _flatten_candidates(selected_by_episode)


def _sample_indices(
    datasets: Sequence[Dataset],
    max_samples: Optional[int],
    stride: int,
) -> list[SampleCandidate]:
    candidates_by_episode = _candidate_grid(datasets, stride)
    indices = _select_stratified_episode(candidates_by_episode, max_samples)
    if not indices:
        raise ValueError("no dataset indices selected")
    return indices


def _select_sample_indices(
    datasets: Sequence[ContactForceHDF5Dataset],
    max_samples: Optional[int],
    stride: int,
    sampling_mode: str,
) -> tuple[list[SampleCandidate], list[SampleCandidate]]:
    candidates_by_episode = _candidate_grid(datasets, stride)
    candidates_by_episode = _with_force_estimates(datasets, candidates_by_episode)
    all_candidates = _flatten_candidates(candidates_by_episode)
    if sampling_mode == "uniform":
        selected = all_candidates if max_samples is None else all_candidates[:max_samples]
        return selected, all_candidates
    if sampling_mode == "stratified_episode":
        selected = _select_stratified_episode(candidates_by_episode, max_samples)
        return selected, all_candidates
    if sampling_mode == "force_balanced":
        selected = _select_force_balanced(candidates_by_episode, max_samples)
        return selected, all_candidates
    raise ValueError(f"unsupported sampling_mode: {sampling_mode}")


def _sampled_force_bin_counts(rows: list[dict]) -> dict[str, int]:
    counts = {force_bin: 0 for force_bin in FORCE_BINS}
    for row in rows:
        force_bin = str(row.get("force_bin", ""))
        if force_bin in counts:
            counts[force_bin] += 1
        elif force_bin:
            counts[force_bin] = counts.get(force_bin, 0) + 1
    return counts


def _episode_counts(rows: list[dict]) -> dict[int, int]:
    counts: dict[int, int] = {}
    for row in rows:
        episode_id = int(row["episode_numeric_id"])
        counts[episode_id] = counts.get(episode_id, 0) + 1
    return dict(sorted(counts.items()))


def _tensor_to_float_list(tensor: torch.Tensor) -> list[float]:
    return [float(value) for value in tensor.detach().cpu().tolist()]


def _write_rows(output_csv: Path, rows: list[dict], z_dim: int, include_prior: bool) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset_index",
        "episode_id",
        "episode_numeric_id",
        "t_state",
        "force_norm_current",
        "force_norm_future_mean",
        "future_force_mean_raw",
        "future_force_mean_normalized",
        "force_norm_future_max",
        "force_norm_future_delta",
        "torque_norm_current",
        "torque_norm_future_mean",
        "force_bin",
        *[f"mu_contact_{index}" for index in range(z_dim)],
        *([f"mu_contact_prior_{index}" for index in range(z_dim)] if include_prior else []),
        *[f"mu_motion_{index}" for index in range(z_dim)],
        *(
            [
                "prior_mu_mse",
                "prior_mu_l2",
                "prior_mu_cosine_similarity",
            ]
            if include_prior
            else []
        ),
        "loss_force_per_sample",
    ]
    with output_csv.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _save_plot(rows: list[dict], z_dim: int, plot_path: Path, color_by: str) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not available; skipping plot", file=sys.stderr)
        return

    color_column, color_label = COLOR_COLUMNS[color_by]
    print(f"color_mode={color_by}")
    pca_result = _compute_contact_pca(rows, z_dim, color_column)
    print(f"pca_rows_used={pca_result['rows_used']}")
    if pca_result["singular_values"] is not None:
        singular_values = pca_result["singular_values"]
        print(f"singular_values_first_two={singular_values[:2].tolist()}")
    if pca_result["pcs"] is None:
        print(pca_result["skip_reason"], file=sys.stderr)
        return

    pcs = pca_result["pcs"]
    colors = pca_result["colors"]
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    scatter = plt.scatter(pcs[:, 0], pcs[:, 1], c=colors, cmap="viridis", s=16)
    plt.xlabel("mu_contact PC1")
    plt.ylabel("mu_contact PC2")
    plt.title(f"mu_contact PCA colored by {color_label}")
    plt.colorbar(scatter, label=color_label)
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    print(f"saved_plot={plot_path}")


def _save_prior_overlay_plot(rows: list[dict], z_dim: int, plot_path: Path, color_by: str) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not available; skipping prior overlay plot", file=sys.stderr)
        return

    color_column, color_label = COLOR_COLUMNS[color_by]
    print(f"prior_overlay_color_mode={color_by}")
    overlay = _compute_prior_overlay_pca(rows, z_dim, color_column)
    print(f"prior_overlay_rows_used={overlay['rows_used']}")
    if overlay["singular_values"] is not None:
        print(f"prior_overlay_singular_values_first_two={overlay['singular_values'][:2].tolist()}")
    if "posterior_pcs_finite" in overlay:
        print(f"prior_overlay_posterior_pcs_finite={overlay['posterior_pcs_finite']}")
    if "prior_pcs_finite" in overlay:
        print(f"prior_overlay_prior_pcs_finite={overlay['prior_pcs_finite']}")
    if overlay["posterior_pcs"] is None:
        print(overlay["skip_reason"], file=sys.stderr)
        return

    posterior_pcs = overlay["posterior_pcs"]
    prior_pcs = overlay["prior_pcs"]
    colors = overlay["colors"]
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    for posterior_point, prior_point in zip(posterior_pcs, prior_pcs):
        plt.plot(
            [posterior_point[0], prior_point[0]],
            [posterior_point[1], prior_point[1]],
            color="0.6",
            alpha=0.25,
            linewidth=0.75,
        )
    posterior_scatter = plt.scatter(
        posterior_pcs[:, 0],
        posterior_pcs[:, 1],
        c=colors,
        cmap="viridis",
        s=18,
        marker="o",
        label="posterior",
    )
    plt.scatter(
        prior_pcs[:, 0],
        prior_pcs[:, 1],
        c=colors,
        cmap="viridis",
        s=24,
        marker="x",
        label="prior",
    )
    plt.xlabel("mu_contact posterior PC1")
    plt.ylabel("mu_contact posterior PC2")
    plt.title(f"contact prior vs posterior PCA colored by {color_label}")
    plt.colorbar(posterior_scatter, label=color_label)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    print(f"saved_prior_overlay_plot={plot_path}")


def _compute_contact_pca(rows: list[dict], z_dim: int, color_column: str) -> dict:
    mu_contact = np.array(
        [[row[f"mu_contact_{index}"] for index in range(z_dim)] for row in rows],
        dtype=np.float64,
    )
    colors = np.array([row[color_column] for row in rows], dtype=np.float64)
    finite_mask = np.isfinite(mu_contact).all(axis=1) & np.isfinite(colors)
    mu_contact = mu_contact[finite_mask]
    colors = colors[finite_mask]

    print(f"mu_contact_shape={mu_contact.shape}")
    print(f"finite_rows={mu_contact.shape[0]}")
    if mu_contact.shape[0] == 0:
        print("mu_contact_stats=unavailable")
        print("color_min_max=unavailable")
        return {
            "pcs": None,
            "colors": colors,
            "rows_used": 0,
            "singular_values": None,
            "skip_reason": "fewer than 2 valid samples for PCA plot; skipping plot",
        }

    print(
        "mu_contact_stats="
        f"min={mu_contact.min():.6g} "
        f"max={mu_contact.max():.6g} "
        f"mean={mu_contact.mean():.6g} "
        f"std={mu_contact.std():.6g}"
    )
    print(f"color_min_max=min={colors.min():.6g} max={colors.max():.6g}")
    if mu_contact.shape[0] < 2:
        return {
            "pcs": None,
            "colors": colors,
            "rows_used": mu_contact.shape[0],
            "singular_values": None,
            "skip_reason": "fewer than 2 valid samples for PCA plot; skipping plot",
        }

    centered = mu_contact - mu_contact.mean(axis=0, keepdims=True)
    centered = np.nan_to_num(centered, nan=0.0, posinf=0.0, neginf=0.0)
    u, s, vh = np.linalg.svd(centered, full_matrices=False)
    pcs = u[:, :2] * s[:2]
    if pcs.shape[1] == 1:
        pcs = np.concatenate([pcs, np.zeros((pcs.shape[0], 1), dtype=np.float64)], axis=1)
    if not np.isfinite(pcs).all():
        return {
            "pcs": None,
            "colors": colors,
            "rows_used": mu_contact.shape[0],
            "singular_values": s[:2],
            "skip_reason": "PCA produced non-finite coordinates; skipping plot",
        }
    return {
        "pcs": pcs,
        "colors": colors,
        "rows_used": mu_contact.shape[0],
        "singular_values": s[:2],
        "skip_reason": None,
    }


def _compute_prior_overlay_pca(rows: list[dict], z_dim: int, color_column: str) -> dict:
    mu_contact = np.array(
        [[row[f"mu_contact_{index}"] for index in range(z_dim)] for row in rows],
        dtype=np.float64,
    )
    mu_prior = np.array(
        [[row[f"mu_contact_prior_{index}"] for index in range(z_dim)] for row in rows],
        dtype=np.float64,
    )
    colors = np.array([row[color_column] for row in rows], dtype=np.float64)
    finite_mask = (
        np.isfinite(mu_contact).all(axis=1)
        & np.isfinite(mu_prior).all(axis=1)
        & np.isfinite(colors)
    )
    mu_contact = mu_contact[finite_mask]
    mu_prior = mu_prior[finite_mask]
    colors = colors[finite_mask]

    print(f"prior_overlay_mu_contact_shape={mu_contact.shape}")
    print(f"prior_overlay_finite_rows={mu_contact.shape[0]}")
    if mu_contact.shape[0] < 2:
        return {
            "posterior_pcs": None,
            "prior_pcs": None,
            "colors": colors,
            "rows_used": mu_contact.shape[0],
            "singular_values": None,
            "skip_reason": "fewer than 2 valid samples for prior overlay plot; skipping plot",
        }

    posterior_mean = mu_contact.mean(axis=0, keepdims=True)
    centered_posterior = mu_contact - posterior_mean
    centered_posterior = np.nan_to_num(
        centered_posterior,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    u, s, vh = np.linalg.svd(centered_posterior, full_matrices=False)
    posterior_pcs = u[:, :2] * s[:2]
    prior_centered = mu_prior - posterior_mean
    prior_centered = np.nan_to_num(
        prior_centered,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    print(
        "prior_centered_stats="
        f"min={prior_centered.min():.6g} "
        f"max={prior_centered.max():.6g} "
        f"mean={prior_centered.mean():.6g} "
        f"std={prior_centered.std():.6g}"
    )
    prior_centered = np.asarray(prior_centered, dtype=np.float64)
    components = np.asarray(vh[:2], dtype=np.float64)
    prior_centered = np.nan_to_num(
        prior_centered,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    components = np.nan_to_num(
        components,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    prior_centered_finite = bool(np.isfinite(prior_centered).all())
    pca_components_finite = bool(np.isfinite(components).all())
    print(
        "vh_first_two_stats="
        f"min={components.min():.6g} "
        f"max={components.max():.6g}"
    )
    print(f"prior_centered_finite={prior_centered_finite}")
    print(f"pca_components_finite={pca_components_finite}")
    prior_pcs = np.stack(
        [
            np.sum(prior_centered * components[0][None, :], axis=1),
            np.sum(prior_centered * components[1][None, :], axis=1),
        ],
        axis=1,
    )
    posterior_pcs_finite = bool(np.isfinite(posterior_pcs).all())
    prior_pcs_finite = bool(np.isfinite(prior_pcs).all())
    print(f"posterior_pcs_finite={posterior_pcs_finite}")
    print(f"prior_pcs_finite={prior_pcs_finite}")
    if prior_pcs.size > 0:
        print(
            "prior_pcs_stats="
            f"min={prior_pcs.min():.6g} "
            f"max={prior_pcs.max():.6g} "
            f"mean={prior_pcs.mean():.6g} "
            f"std={prior_pcs.std():.6g}"
        )
    if posterior_pcs.shape[1] == 1:
        posterior_pcs = np.concatenate(
            [posterior_pcs, np.zeros((posterior_pcs.shape[0], 1), dtype=np.float64)],
            axis=1,
        )
        prior_pcs = np.concatenate(
            [prior_pcs, np.zeros((prior_pcs.shape[0], 1), dtype=np.float64)],
            axis=1,
        )
    if not np.isfinite(posterior_pcs).all() or not np.isfinite(prior_pcs).all():
        return {
            "posterior_pcs": None,
            "prior_pcs": None,
            "colors": colors,
            "rows_used": mu_contact.shape[0],
            "singular_values": s[:2],
            "posterior_pcs_finite": posterior_pcs_finite,
            "prior_pcs_finite": prior_pcs_finite,
            "skip_reason": "prior overlay PCA produced non-finite coordinates; skipping plot",
        }
    return {
        "posterior_pcs": posterior_pcs,
        "prior_pcs": prior_pcs,
        "colors": colors,
        "rows_used": mu_contact.shape[0],
        "singular_values": s[:2],
        "posterior_pcs_finite": posterior_pcs_finite,
        "prior_pcs_finite": prior_pcs_finite,
        "skip_reason": None,
    }


def _print_prior_metric_summary(rows: list[dict]) -> None:
    for column in ("prior_mu_mse", "prior_mu_l2", "prior_mu_cosine_similarity"):
        values = np.array([row[column] for row in rows], dtype=np.float64)
        values = values[np.isfinite(values)]
        if values.size == 0:
            print(f"{column}_summary=unavailable")
            continue
        print(
            f"{column}_summary="
            f"mean={values.mean():.6g} "
            f"std={values.std():.6g} "
            f"min={values.min():.6g} "
            f"max={values.max():.6g}"
        )


def analyze(args: argparse.Namespace) -> int:
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint must contain a dict")

    stats = _load_stats(args.normalization_stats)
    dataset_kwargs = _dataset_kwargs_from_args(args)
    print("effective_dataset_config:")
    print(f"chunk_len={dataset_kwargs['chunk_len']}")
    print(f"force_window_len={dataset_kwargs['force_window_len']}")
    print(f"force_window_duration={dataset_kwargs['force_window_duration']}")
    print(f"image_size={dataset_kwargs['image_size']}")
    print(f"camera_names={dataset_kwargs['camera_names']}")
    datasets = [
        ContactForceHDF5Dataset(episode_path, **dataset_kwargs)
        for episode_path in args.episode_paths
    ]
    total_dataset_len = sum(len(dataset) for dataset in datasets)
    if total_dataset_len == 0:
        raise ValueError("datasets are empty")

    model = ForceAwareACTPolicy(**_model_kwargs_from_checkpoint(checkpoint)).to(args.device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    indices, candidate_indices = _select_sample_indices(
        datasets,
        args.max_samples,
        args.stride,
        args.sampling_mode,
    )
    print(f"sampling_mode={args.sampling_mode}")
    _print_force_stats("candidate_future_force_mean_raw", candidate_indices)
    indexed_dataset = MultiEpisodeIndexedDataset(datasets, args.episode_paths, indices)
    dataloader = DataLoader(indexed_dataset, batch_size=args.batch_size, shuffle=False)

    rows = []
    with torch.no_grad():
        for raw_batch in dataloader:
            batch = {
                key: value.to(args.device) if torch.is_tensor(value) else value
                for key, value in raw_batch.items()
            }
            normalized = _normalize_batch(batch, stats)
            outputs = model(
                images=normalized["images"],
                qpos=normalized["qpos"],
                force_window=normalized["force_window"],
                action_chunk=normalized["action_chunk"],
                future_force_chunk=normalized["future_force_chunk"],
                is_training=True,
            )
            prior_available = all(
                key in outputs
                for key in ("mu_contact_prior", "logvar_contact_prior")
            )
            if args.include_prior and not prior_available:
                print(
                    "include_prior requested, but model outputs do not contain contact prior; "
                    "skipping prior export for this batch",
                    file=sys.stderr,
                )

            loss_force_per_sample = functional.l1_loss(
                outputs["pred_force"],
                normalized["future_force_chunk"],
                reduction="none",
            ).mean(dim=(1, 2))
            current_wrench = raw_batch["force_window"][:, -1, :]
            future_wrench = raw_batch["future_force_chunk"]
            force_norm_current = current_wrench[:, :3].norm(dim=-1)
            torque_norm_current = current_wrench[:, 3:6].norm(dim=-1)
            future_force_norms = future_wrench[:, :, :3].norm(dim=-1)
            future_torque_norms = future_wrench[:, :, 3:6].norm(dim=-1)
            force_norm_future_mean = future_force_norms.mean(dim=1)
            force_norm_future_max = future_force_norms.max(dim=1).values
            force_norm_future_delta = force_norm_future_mean - force_norm_current
            torque_norm_future_mean = future_torque_norms.mean(dim=1)
            normalized_future_force_mean = normalized["future_force_chunk"][:, :, :3].norm(dim=-1).mean(dim=1)
            if args.include_prior and prior_available:
                prior_delta = outputs["mu_contact_prior"] - outputs["mu_contact"]
                prior_mu_mse = prior_delta.pow(2).mean(dim=-1)
                prior_mu_l2 = prior_delta.norm(dim=-1)
                prior_mu_cosine_similarity = functional.cosine_similarity(
                    outputs["mu_contact_prior"],
                    outputs["mu_contact"],
                    dim=-1,
                )

            for row_index in range(outputs["mu_contact"].shape[0]):
                sample_force_bin = raw_batch["sample_force_bin"][row_index]
                if not sample_force_bin:
                    sample_force_bin = "unknown"
                row = {
                    "dataset_index": int(raw_batch["dataset_index"][row_index]),
                    "episode_id": str(raw_batch["episode_id"][row_index]),
                    "episode_numeric_id": int(raw_batch["episode_numeric_id"][row_index]),
                    "t_state": float(raw_batch["t_state"][row_index]),
                    "force_norm_current": float(force_norm_current[row_index]),
                    "force_norm_future_mean": float(force_norm_future_mean[row_index]),
                    "future_force_mean_raw": float(force_norm_future_mean[row_index]),
                    "future_force_mean_normalized": float(
                        normalized_future_force_mean[row_index].detach().cpu()
                    ),
                    "force_norm_future_max": float(force_norm_future_max[row_index]),
                    "force_norm_future_delta": float(force_norm_future_delta[row_index]),
                    "torque_norm_current": float(torque_norm_current[row_index]),
                    "torque_norm_future_mean": float(torque_norm_future_mean[row_index]),
                    "force_bin": str(sample_force_bin),
                    "loss_force_per_sample": float(loss_force_per_sample[row_index].cpu()),
                }
                for latent_index, value in enumerate(
                    _tensor_to_float_list(outputs["mu_contact"][row_index])
                ):
                    row[f"mu_contact_{latent_index}"] = value
                if args.include_prior and prior_available:
                    for latent_index, value in enumerate(
                        _tensor_to_float_list(outputs["mu_contact_prior"][row_index])
                    ):
                        row[f"mu_contact_prior_{latent_index}"] = value
                    row["prior_mu_mse"] = float(prior_mu_mse[row_index].detach().cpu())
                    row["prior_mu_l2"] = float(prior_mu_l2[row_index].detach().cpu())
                    row["prior_mu_cosine_similarity"] = float(
                        prior_mu_cosine_similarity[row_index].detach().cpu()
                    )
                for latent_index, value in enumerate(
                    _tensor_to_float_list(outputs["mu_motion"][row_index])
                ):
                    row[f"mu_motion_{latent_index}"] = value
                rows.append(row)

    z_dim = int(outputs["mu_contact"].shape[1])
    prior_rows_available = bool(rows) and all(
        f"mu_contact_prior_{index}" in rows[0]
        for index in range(z_dim)
    )
    _write_rows(args.output_csv, rows, z_dim, include_prior=prior_rows_available)
    sampled_episode_counts = _episode_counts(rows)
    unique_episode_numeric_ids = list(sampled_episode_counts.keys())
    sampled_future_force_candidates = [
        SampleCandidate(
            episode_id=int(row["episode_numeric_id"]),
            local_index=0,
            global_index=int(row["dataset_index"]),
            future_force_mean_raw=float(row["future_force_mean_raw"]),
            force_bin=str(row["force_bin"]),
        )
        for row in rows
    ]
    print(f"dataset_length={total_dataset_len}")
    print(f"sampled_rows={len(rows)}")
    print(f"sampled_episode_counts={sampled_episode_counts}")
    print(f"unique_episode_numeric_ids={unique_episode_numeric_ids}")
    _print_force_stats("sampled_future_force_mean_raw", sampled_future_force_candidates)
    print(f"sampled_force_quantile_counts={_sampled_force_bin_counts(rows)}")
    print(f"prior_analysis_found={prior_rows_available}")
    if prior_rows_available:
        _print_prior_metric_summary(rows)
    print(f"saved_csv={args.output_csv}")
    episode_mapping = {
        str(path): index
        for index, path in enumerate(args.episode_paths)
    }
    print(f"episode_id_mapping={episode_mapping}")
    if args.plot is not None:
        _save_plot(rows, z_dim, args.plot, args.color_by)
    if args.plot_prior_overlay is not None:
        if not prior_rows_available:
            print(
                "prior overlay requested, but prior columns are unavailable; skipping overlay",
                file=sys.stderr,
            )
        else:
            _save_prior_overlay_plot(rows, z_dim, args.plot_prior_overlay, args.color_by)
    return 0


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze ForceAwareACT contact posterior latents.")
    parser.add_argument("episode_paths", type=Path, nargs="*")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("normalization_stats", type=Path)
    parser.add_argument("output_csv", type=Path)
    parser.add_argument("--episode-list", type=Path, default=None)
    parser.add_argument("--chunk-len", type=int, default=10)
    parser.add_argument("--force-window-len", type=int, default=20)
    parser.add_argument("--force-window-duration", type=float, default=0.25)
    parser.add_argument("--image-size", type=int, nargs=2, default=(224, 224))
    parser.add_argument("--camera-names", nargs="+", default=("ee_cam", "base_top_cam"))
    parser.add_argument(
        "--sampling-mode",
        choices=("uniform", "stratified_episode", "force_balanced"),
        default="stratified_episode",
    )
    parser.add_argument("--include-prior", action="store_true")
    parser.add_argument("--plot", type=Path, default=None)
    parser.add_argument("--plot-prior-overlay", type=Path, default=None)
    parser.add_argument(
        "--color-by",
        choices=sorted(COLOR_COLUMNS.keys()),
        default="future_force_mean",
    )
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    args.episode_paths = resolve_episode_paths(args.episode_paths, args.episode_list)
    args.checkpoint = args.checkpoint.expanduser()
    args.normalization_stats = args.normalization_stats.expanduser()
    args.output_csv = args.output_csv.expanduser()
    if args.plot is not None:
        args.plot = args.plot.expanduser()
    if args.plot_prior_overlay is not None:
        args.plot_prior_overlay = args.plot_prior_overlay.expanduser()
    if args.plot_prior_overlay is not None and not args.include_prior:
        print("error: --plot-prior-overlay requires --include-prior", file=sys.stderr)
        return 2
    if not args.episode_paths:
        print("error: provide episode paths or --episode-list", file=sys.stderr)
        return 2
    if not validate_episode_paths(args.episode_paths):
        return 2
    for path_name in ("checkpoint", "normalization_stats"):
        path = getattr(args, path_name)
        if not path.is_file():
            print(f"error: {path_name} does not exist: {path}", file=sys.stderr)
            return 2
    if args.max_samples is not None and args.max_samples <= 0:
        print("error: --max-samples must be positive", file=sys.stderr)
        return 2
    if args.stride <= 0:
        print("error: --stride must be positive", file=sys.stderr)
        return 2
    if args.batch_size <= 0:
        print("error: --batch-size must be positive", file=sys.stderr)
        return 2
    if args.chunk_len <= 0:
        print("error: --chunk-len must be positive", file=sys.stderr)
        return 2
    if args.force_window_len <= 0:
        print("error: --force-window-len must be positive", file=sys.stderr)
        return 2
    if len(args.image_size) != 2 or args.image_size[0] <= 0 or args.image_size[1] <= 0:
        print("error: --image-size must be two positive integers", file=sys.stderr)
        return 2
    if not args.camera_names:
        print("error: --camera-names must include at least one camera", file=sys.stderr)
        return 2

    try:
        return analyze(args)
    except Exception as error:
        print(f"error: failed to analyze contact latents: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
