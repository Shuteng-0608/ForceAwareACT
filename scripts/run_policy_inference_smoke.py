#!/usr/bin/env python3
"""Run deployable ForceAwareACT inference on one recorded HDF5 sample."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import torch

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
    for key in ("qpos_mean", "qpos_std", "action_mean", "action_std", "force_mean", "force_std"):
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
        f"state index {state_index} is not a valid sample for chunk_len={dataset.chunk_len}"
    )


def _online_batch(sample: dict, stats: Dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    images = sample["images"].unsqueeze(0)
    qpos = sample["qpos"].unsqueeze(0)
    force_window = sample["force_window"].unsqueeze(0)
    return {
        "images": images,
        "qpos": normalize_tensor(qpos, stats["qpos_mean"], stats["qpos_std"]),
        "force_window": normalize_tensor(force_window, stats["force_mean"], stats["force_std"]),
    }


def _run_mode(model: ForceAwareACTPolicy, batch: dict[str, torch.Tensor], mode: str) -> dict:
    with torch.no_grad():
        return model(
            images=batch["images"],
            qpos=batch["qpos"],
            force_window=batch["force_window"],
            action_chunk=None,
            future_force_chunk=None,
            is_training=False,
            contact_latent_mode=mode,
            deterministic_prior=True,
        )


def _denormalized_predictions(
    output: dict,
    stats: Dict[str, torch.Tensor],
) -> tuple[np.ndarray, np.ndarray]:
    action = denormalize_tensor(output["pred_action"], stats["action_mean"], stats["action_std"])
    force = denormalize_tensor(output["pred_force"], stats["force_mean"], stats["force_std"])
    return action.squeeze(0).cpu().numpy(), force.squeeze(0).cpu().numpy()


def _load_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _save_action_plot(path: Path, actions: dict[str, np.ndarray]) -> None:
    plt = _load_matplotlib()
    figure, axes = plt.subplots(7, 1, figsize=(10, 14), sharex=True)
    steps = np.arange(next(iter(actions.values())).shape[0])
    for joint, axis in enumerate(axes):
        for mode, values in actions.items():
            axis.plot(steps, values[:, joint], marker="o", label=mode)
        axis.set_ylabel(f"joint {joint}")
        axis.grid(alpha=0.25)
    axes[0].legend()
    axes[-1].set_xlabel("future chunk step")
    figure.suptitle("Predicted Action Chunks")
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)
    print(f"saved_plot={path}")


def _save_force_plot(path: Path, forces: dict[str, np.ndarray]) -> None:
    plt = _load_matplotlib()
    steps = np.arange(next(iter(forces.values())).shape[0])
    for mode, values in forces.items():
        plt.plot(steps, np.linalg.norm(values[:, :3], axis=1), marker="o", label=mode)
    plt.xlabel("future chunk step")
    plt.ylabel("predicted translational force norm")
    plt.title("Predicted Future Force Norm")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    print(f"saved_plot={path}")


def smoke(args: argparse.Namespace) -> int:
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
    batch = _online_batch(sample, stats)

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint must contain a dict")
    model = ForceAwareACTPolicy(**_model_kwargs(checkpoint, args.force_window_len))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    outputs = {"zero": _run_mode(model, batch, "zero")}
    if args.contact_latent_mode == "prior":
        outputs["prior"] = _run_mode(model, batch, "prior")
    selected_output = outputs[args.contact_latent_mode]
    predictions = {
        mode: _denormalized_predictions(output, stats) for mode, output in outputs.items()
    }
    actions = {mode: prediction[0] for mode, prediction in predictions.items()}
    forces = {mode: prediction[1] for mode, prediction in predictions.items()}
    selected_action = actions[args.contact_latent_mode]
    selected_force = forces[args.contact_latent_mode]
    force_norms = np.linalg.norm(selected_force[:, :3], axis=1)

    print(f"episode_path={args.episode}")
    print(f"state_index={args.state_index}")
    print(f"timestamp={sample['t_state']:.9g}")
    print(f"contact_latent_mode={args.contact_latent_mode}")
    print(f"images_shape={tuple(batch['images'].shape)}")
    print(f"qpos_shape={tuple(batch['qpos'].shape)}")
    print(f"force_window_shape={tuple(batch['force_window'].shape)}")
    print(f"qpos={np.array2string(sample['qpos'].numpy(), precision=6, separator=',')}")
    print(f"action_chunk_predicted_shape={selected_action.shape}")
    print(f"force_chunk_predicted_shape={selected_force.shape}")
    print(
        "first_predicted_action="
        f"{np.array2string(selected_action[0], precision=6, separator=',')}"
    )
    print(f"action_min={float(selected_action.min()):.9g}")
    print(f"action_max={float(selected_action.max()):.9g}")
    print(
        "predicted_force_norm_per_step="
        f"{np.array2string(force_norms, precision=6, separator=',')}"
    )
    if "prior" in outputs:
        print(
            "prior_vs_zero_action_mean_abs_difference="
            f"{float(np.abs(actions['prior'] - actions['zero']).mean()):.9g}"
        )
        print(
            "prior_vs_zero_force_mean_abs_difference="
            f"{float(np.abs(forces['prior'] - forces['zero']).mean()):.9g}"
        )

    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        save_values = {
            "episode_path": np.asarray(str(args.episode)),
            "state_index": np.asarray(args.state_index),
            "timestamp": np.asarray(sample["t_state"]),
            "qpos": sample["qpos"].numpy(),
            "contact_latent_mode": np.asarray(args.contact_latent_mode),
        }
        for mode in actions:
            save_values[f"action_{mode}"] = actions[mode]
            save_values[f"force_{mode}"] = forces[mode]
        np.savez(args.output_dir / "predictions.npz", **save_values)
        print(f"saved_npz={args.output_dir / 'predictions.npz'}")
        _save_action_plot(args.output_dir / "action_chunk.png", actions)
        _save_force_plot(args.output_dir / "force_chunk_norm.png", forces)
    return 0


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run deployable ForceAwareACT inference smoke test.")
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--state-index", type=int, default=0)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--normalization-stats", type=Path, required=True)
    parser.add_argument("--chunk-len", type=int, default=10)
    parser.add_argument("--force-window-len", type=int, default=20)
    parser.add_argument("--force-window-duration", type=float, default=0.25)
    parser.add_argument("--contact-latent-mode", choices=("zero", "prior"), default="prior")
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    for key in ("episode", "checkpoint", "normalization_stats"):
        setattr(args, key, getattr(args, key).expanduser())
        if not getattr(args, key).is_file():
            print(f"error: {key.replace('_', ' ')} does not exist: {getattr(args, key)}", file=sys.stderr)
            return 2
    if args.output_dir is not None:
        args.output_dir = args.output_dir.expanduser()
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
        return smoke(args)
    except Exception as error:
        print(f"error: policy inference smoke test failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
