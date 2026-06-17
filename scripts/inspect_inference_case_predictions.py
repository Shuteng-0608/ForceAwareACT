#!/usr/bin/env python3
"""Inspect zero, prior, and posterior prediction chunks for one HDF5 state."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as functional

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from force_aware_act.data import (  # noqa: E402
    ContactForceHDF5Dataset,
    denormalize_tensor,
    normalize_tensor,
)
from force_aware_act.models import ForceAwareACTPolicy  # noqa: E402


def _load_stats(path: Path) -> Dict[str, torch.Tensor]:
    stats = torch.load(path, map_location="cpu")
    if not isinstance(stats, dict):
        raise ValueError("normalization stats file must contain a dict")
    required = ("qpos_mean", "qpos_std", "action_mean", "action_std", "force_mean", "force_std")
    for key in required:
        if key not in stats or not torch.is_tensor(stats[key]):
            raise KeyError(f"normalization stats missing tensor: {key}")
    return stats


def _model_kwargs(checkpoint: dict, force_window_len: int) -> dict:
    model_config = dict(checkpoint.get("config", {}).get("model", {}))
    if not model_config:
        raise KeyError("checkpoint config is missing model settings")
    if "pretrained_vision" in model_config and "pretrained_resnet18" not in model_config:
        model_config["pretrained_resnet18"] = model_config.pop("pretrained_vision")
    model_config.setdefault("pretrained_resnet18", False)
    model_config.setdefault("dropout", 0.0)
    model_config.setdefault("max_force_window_len", max(force_window_len, 20))
    return model_config


def _find_sample(dataset: ContactForceHDF5Dataset, state_index: int) -> dict:
    for dataset_index, episode_index in enumerate(dataset.indices):
        if episode_index.state_index == state_index:
            sample = dataset[dataset_index]
            sample["global_dataset_index"] = dataset_index
            return sample
    raise ValueError(
        f"state index {state_index} is not a valid dataset sample for chunk_len={dataset.chunk_len}"
    )


def _make_batch(sample: dict, stats: Dict[str, torch.Tensor]) -> dict:
    batch = {
        key: value.unsqueeze(0) if torch.is_tensor(value) else value
        for key, value in sample.items()
    }
    batch["qpos"] = normalize_tensor(batch["qpos"], stats["qpos_mean"], stats["qpos_std"])
    batch["force_window"] = normalize_tensor(
        batch["force_window"], stats["force_mean"], stats["force_std"]
    )
    batch["action_chunk"] = normalize_tensor(
        batch["action_chunk"], stats["action_mean"], stats["action_std"]
    )
    batch["future_force_chunk"] = normalize_tensor(
        batch["future_force_chunk"], stats["force_mean"], stats["force_std"]
    )
    return batch


def _run_modes(model: ForceAwareACTPolicy, batch: dict) -> tuple[dict, dict, dict]:
    with torch.no_grad():
        zero = model(
            images=batch["images"],
            qpos=batch["qpos"],
            force_window=batch["force_window"],
            action_chunk=None,
            future_force_chunk=None,
            is_training=False,
            contact_latent_mode="zero",
        )
        prior = model(
            images=batch["images"],
            qpos=batch["qpos"],
            force_window=batch["force_window"],
            action_chunk=None,
            future_force_chunk=None,
            is_training=False,
            contact_latent_mode="prior",
            deterministic_prior=True,
        )
        posterior = model(
            images=batch["images"],
            qpos=batch["qpos"],
            force_window=batch["force_window"],
            action_chunk=batch["action_chunk"],
            future_force_chunk=batch["future_force_chunk"],
            is_training=True,
            contact_latent_mode="posterior",
        )
    return zero, prior, posterior


def _numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().squeeze(0).numpy()


def _load_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _save_component_plot(
    path: Path,
    arrays: dict[str, np.ndarray],
    labels: Sequence[str],
    title: str,
) -> None:
    plt = _load_matplotlib()
    figure, axes = plt.subplots(len(labels), 1, figsize=(10, 2.2 * len(labels)), sharex=True)
    steps = np.arange(next(iter(arrays.values())).shape[0])
    for dimension, (axis, label) in enumerate(zip(axes, labels)):
        for name, values in arrays.items():
            axis.plot(steps, values[:, dimension], marker="o", linewidth=1.5, label=name)
        axis.set_ylabel(label)
        axis.grid(alpha=0.25)
    axes[0].legend(ncol=4)
    axes[-1].set_xlabel("future chunk step")
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)
    print(f"saved_plot={path}")


def _save_force_norm_plot(path: Path, arrays: dict[str, np.ndarray]) -> None:
    plt = _load_matplotlib()
    figure, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    steps = np.arange(next(iter(arrays.values())).shape[0])
    for name, values in arrays.items():
        axes[0].plot(steps, np.linalg.norm(values[:, :3], axis=1), marker="o", label=name)
        axes[1].plot(steps, np.linalg.norm(values[:, 3:6], axis=1), marker="o", label=name)
    axes[0].set_ylabel("translational force norm")
    axes[1].set_ylabel("torque norm")
    axes[1].set_xlabel("future chunk step")
    axes[0].legend(ncol=4)
    for axis in axes:
        axis.grid(alpha=0.25)
    figure.suptitle("Future Force and Torque Norms")
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)
    print(f"saved_plot={path}")


def _save_error_plot(
    path: Path,
    action_gt: torch.Tensor,
    force_gt: torch.Tensor,
    outputs: dict[str, dict],
) -> None:
    plt = _load_matplotlib()
    figure, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    steps = np.arange(action_gt.shape[1])
    for name, output in outputs.items():
        action_error = (output["pred_action"] - action_gt).abs().mean(dim=-1).squeeze(0)
        force_error = (output["pred_force"] - force_gt).abs().mean(dim=-1).squeeze(0)
        axes[0].plot(steps, _numpy(action_error), marker="o", label=name)
        axes[1].plot(steps, _numpy(force_error), marker="o", label=name)
    axes[0].set_ylabel("action L1, normalized")
    axes[1].set_ylabel("force L1, normalized")
    axes[1].set_xlabel("future chunk step")
    axes[0].legend(ncol=3)
    for axis in axes:
        axis.grid(alpha=0.25)
    figure.suptitle("Per-Step Prediction Errors")
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)
    print(f"saved_plot={path}")


def _metric(tensor: torch.Tensor) -> float:
    return float(tensor.detach().cpu().item())


def inspect(args: argparse.Namespace) -> int:
    stats = _load_stats(args.normalization_stats)
    dataset = ContactForceHDF5Dataset(
        args.episode,
        action_mode="joint_pos",
        chunk_len=args.chunk_len,
        force_window_len=args.force_window_len,
        force_window_duration=args.force_window_duration,
        image_size=(224, 224),
        imagenet_normalize=False,
    )
    sample = _find_sample(dataset, args.state_index)
    batch = _make_batch(sample, stats)

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint must contain a dict")
    model = ForceAwareACTPolicy(**_model_kwargs(checkpoint, args.force_window_len))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    zero, prior, posterior = _run_modes(model, batch)
    outputs = {"zero": zero, "prior": prior, "posterior": posterior}

    action_gt_normalized = batch["action_chunk"]
    force_gt_normalized = batch["future_force_chunk"]
    action_physical = {
        "gt": _numpy(
            denormalize_tensor(action_gt_normalized, stats["action_mean"], stats["action_std"])
        ),
        **{
            name: _numpy(
                denormalize_tensor(
                    output["pred_action"], stats["action_mean"], stats["action_std"]
                )
            )
            for name, output in outputs.items()
        },
    }
    force_physical = {
        "gt": _numpy(
            denormalize_tensor(force_gt_normalized, stats["force_mean"], stats["force_std"])
        ),
        **{
            name: _numpy(
                denormalize_tensor(output["pred_force"], stats["force_mean"], stats["force_std"])
            )
            for name, output in outputs.items()
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output_dir / "prediction_chunks.npz",
        action_gt=action_physical["gt"],
        action_zero=action_physical["zero"],
        action_prior=action_physical["prior"],
        action_posterior=action_physical["posterior"],
        force_gt=force_physical["gt"],
        force_zero=force_physical["zero"],
        force_prior=force_physical["prior"],
        force_posterior=force_physical["posterior"],
        mu_contact_prior=_numpy(prior["mu_contact_prior"]),
        mu_contact_posterior=_numpy(posterior["mu_contact"]),
        episode_path=np.asarray(str(args.episode)),
        state_index=np.asarray(args.state_index),
        timestamp=np.asarray(sample["t_state"]),
        global_dataset_index=np.asarray(sample["global_dataset_index"]),
        array_space=np.asarray("denormalized"),
    )
    print(f"saved_npz={args.output_dir / 'prediction_chunks.npz'}")

    _save_component_plot(
        args.output_dir / "force_chunk_components.png",
        force_physical,
        ("Fx", "Fy", "Fz", "Tx", "Ty", "Tz"),
        "Future Wrench Chunk",
    )
    _save_force_norm_plot(args.output_dir / "force_chunk_norm.png", force_physical)
    _save_component_plot(
        args.output_dir / "action_chunk_joints.png",
        action_physical,
        tuple(f"joint {index}" for index in range(7)),
        "Future Action Chunk",
    )
    _save_error_plot(
        args.output_dir / "prediction_errors.png",
        action_gt_normalized,
        force_gt_normalized,
        outputs,
    )

    action_losses = {
        name: _metric(functional.l1_loss(output["pred_action"], action_gt_normalized))
        for name, output in outputs.items()
    }
    force_losses = {
        name: _metric(functional.l1_loss(output["pred_force"], force_gt_normalized))
        for name, output in outputs.items()
    }
    mu_delta = prior["mu_contact_prior"] - posterior["mu_contact"]
    action_improvement = (action_losses["zero"] - action_losses["prior"]) / action_losses["zero"]
    force_improvement = (force_losses["zero"] - force_losses["prior"]) / force_losses["zero"]
    force_norm_means = {
        name: float(np.linalg.norm(values[:, :3], axis=1).mean())
        for name, values in force_physical.items()
    }

    print(f"episode_path={args.episode}")
    print(f"state_index={args.state_index}")
    print(f"timestamp={sample['t_state']:.9g}")
    for mode in ("zero", "prior", "posterior"):
        print(f"action_l1_{mode}={action_losses[mode]:.9g}")
    for mode in ("zero", "prior", "posterior"):
        print(f"force_l1_{mode}={force_losses[mode]:.9g}")
    print(f"action_prior_improvement_vs_zero={action_improvement:.9g}")
    print(f"force_prior_improvement_vs_zero={force_improvement:.9g}")
    print(f"mu_prior_to_mu_posterior_mse={_metric(mu_delta.pow(2).mean()):.9g}")
    print(f"mu_prior_to_mu_posterior_l2={_metric(mu_delta.norm(dim=-1).mean()):.9g}")
    print(
        "mu_prior_to_mu_posterior_cosine="
        f"{_metric(functional.cosine_similarity(prior['mu_contact_prior'], posterior['mu_contact'], dim=-1).mean()):.9g}"
    )
    for name in ("gt", "zero", "prior", "posterior"):
        print(f"mean_translational_force_norm_{name}={force_norm_means[name]:.9g}")
    print(
        "prior_mean_force_norm_bias="
        f"{force_norm_means['prior'] - force_norm_means['gt']:.9g}"
    )
    print(f"output_dir={args.output_dir}")
    return 0


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect one inference prediction case.")
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--state-index", type=int, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--normalization-stats", type=Path, required=True)
    parser.add_argument("--chunk-len", type=int, default=10)
    parser.add_argument("--force-window-len", type=int, default=20)
    parser.add_argument("--force-window-duration", type=float, default=0.25)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    for key in ("episode", "checkpoint", "normalization_stats", "output_dir"):
        setattr(args, key, getattr(args, key).expanduser())
    for key in ("episode", "checkpoint", "normalization_stats"):
        if not getattr(args, key).is_file():
            print(f"error: {key.replace('_', ' ')} does not exist: {getattr(args, key)}", file=sys.stderr)
            return 2
    if args.state_index < 0:
        print("error: --state-index must be non-negative", file=sys.stderr)
        return 2
    if args.chunk_len <= 0 or args.force_window_len <= 0:
        print("error: --chunk-len and --force-window-len must be positive", file=sys.stderr)
        return 2
    if args.force_window_duration < 0:
        print("error: --force-window-duration must be non-negative", file=sys.stderr)
        return 2
    try:
        return inspect(args)
    except Exception as error:
        print(f"error: failed to inspect inference case: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
