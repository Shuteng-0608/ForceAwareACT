#!/usr/bin/env python3
"""Analyze posterior contact latents from a trained ForceAwareACT checkpoint."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as functional
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from force_aware_act.data import ContactForceHDF5Dataset, normalize_tensor  # noqa: E402
from force_aware_act.models import ForceAwareACTPolicy  # noqa: E402


COLOR_COLUMNS = {
    "future_force_mean": ("force_norm_future_mean", "future force mean"),
    "future_force_max": ("force_norm_future_max", "future force max"),
    "force_delta": ("force_norm_future_delta", "future force mean - current force"),
    "current_force": ("force_norm_current", "current force"),
    "current_torque": ("torque_norm_current", "current torque"),
    "future_torque_mean": ("torque_norm_future_mean", "future torque mean"),
    "time": ("t_state", "time"),
    "loss_force_per_sample": ("loss_force_per_sample", "force loss per sample"),
}


class IndexedDataset(Dataset):
    """Return selected dataset samples with their original dataset index."""

    def __init__(self, dataset: Dataset, indices: Sequence[int]) -> None:
        self.dataset = dataset
        self.indices = list(indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> dict:
        dataset_index = self.indices[item]
        sample = dict(self.dataset[dataset_index])
        sample["dataset_index"] = dataset_index
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


def _dataset_kwargs_from_checkpoint(checkpoint: dict) -> dict:
    config = checkpoint.get("config", {})
    return {
        "camera_names": ("ee_cam", "base_top_cam"),
        "action_mode": "joint_pos",
        "chunk_len": int(config.get("chunk_len", 10)),
        "force_window_len": int(config.get("force_window_len", 20)),
        "force_window_duration": float(config.get("force_window_duration", 0.25)),
        "image_size": (224, 224),
        "imagenet_normalize": bool(config.get("imagenet_normalize", False)),
    }


def _sample_indices(dataset_len: int, max_samples: Optional[int], stride: int) -> list[int]:
    indices = list(range(0, dataset_len, stride))
    if max_samples is not None:
        indices = indices[:max_samples]
    if not indices:
        raise ValueError("no dataset indices selected")
    return indices


def _tensor_to_float_list(tensor: torch.Tensor) -> list[float]:
    return [float(value) for value in tensor.detach().cpu().tolist()]


def _write_rows(output_csv: Path, rows: list[dict], z_dim: int) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset_index",
        "t_state",
        "force_norm_current",
        "force_norm_future_mean",
        "force_norm_future_max",
        "force_norm_future_delta",
        "torque_norm_current",
        "torque_norm_future_mean",
        *[f"mu_contact_{index}" for index in range(z_dim)],
        *[f"mu_motion_{index}" for index in range(z_dim)],
        "loss_force_per_sample",
    ]
    with output_csv.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _save_plot(rows: list[dict], z_dim: int, plot_path: Path, color_by: str) -> None:
    try:
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


def analyze(args: argparse.Namespace) -> int:
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint must contain a dict")

    stats = _load_stats(args.normalization_stats)
    dataset = ContactForceHDF5Dataset(args.episode_path, **_dataset_kwargs_from_checkpoint(checkpoint))
    if len(dataset) == 0:
        raise ValueError("dataset is empty")

    model = ForceAwareACTPolicy(**_model_kwargs_from_checkpoint(checkpoint)).to(args.device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    indices = _sample_indices(len(dataset), args.max_samples, args.stride)
    indexed_dataset = IndexedDataset(dataset, indices)
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

            for row_index in range(outputs["mu_contact"].shape[0]):
                row = {
                    "dataset_index": int(raw_batch["dataset_index"][row_index]),
                    "t_state": float(raw_batch["t_state"][row_index]),
                    "force_norm_current": float(force_norm_current[row_index]),
                    "force_norm_future_mean": float(force_norm_future_mean[row_index]),
                    "force_norm_future_max": float(force_norm_future_max[row_index]),
                    "force_norm_future_delta": float(force_norm_future_delta[row_index]),
                    "torque_norm_current": float(torque_norm_current[row_index]),
                    "torque_norm_future_mean": float(torque_norm_future_mean[row_index]),
                    "loss_force_per_sample": float(loss_force_per_sample[row_index].cpu()),
                }
                for latent_index, value in enumerate(
                    _tensor_to_float_list(outputs["mu_contact"][row_index])
                ):
                    row[f"mu_contact_{latent_index}"] = value
                for latent_index, value in enumerate(
                    _tensor_to_float_list(outputs["mu_motion"][row_index])
                ):
                    row[f"mu_motion_{latent_index}"] = value
                rows.append(row)

    z_dim = int(outputs["mu_contact"].shape[1])
    _write_rows(args.output_csv, rows, z_dim)
    print(f"dataset_length={len(dataset)}")
    print(f"sampled_rows={len(rows)}")
    print(f"saved_csv={args.output_csv}")
    if args.plot is not None:
        _save_plot(rows, z_dim, args.plot, args.color_by)
    return 0


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze ForceAwareACT contact posterior latents.")
    parser.add_argument("episode_path", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("normalization_stats", type=Path)
    parser.add_argument("output_csv", type=Path)
    parser.add_argument("--plot", type=Path, default=None)
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
    args.episode_path = args.episode_path.expanduser()
    args.checkpoint = args.checkpoint.expanduser()
    args.normalization_stats = args.normalization_stats.expanduser()
    args.output_csv = args.output_csv.expanduser()
    if args.plot is not None:
        args.plot = args.plot.expanduser()
    for path_name in ("episode_path", "checkpoint", "normalization_stats"):
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

    try:
        return analyze(args)
    except Exception as error:
        print(f"error: failed to analyze contact latents: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
