#!/usr/bin/env python3
"""Compare zero, prior, and posterior contact latents on one real-HDF5 batch."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Optional, Sequence

import torch
import torch.nn.functional as functional
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from force_aware_act.data import ContactForceHDF5Dataset, normalize_tensor  # noqa: E402
from force_aware_act.models import ForceAwareACTPolicy  # noqa: E402
from script_utils import resolve_episode_paths, validate_episode_paths  # noqa: E402


def _load_normalization_stats(path: Path) -> Dict[str, torch.Tensor]:
    stats = torch.load(path, map_location="cpu")
    if not isinstance(stats, dict):
        raise ValueError("normalization stats file must contain a dict")
    for key in ("qpos_mean", "qpos_std", "action_mean", "action_std", "force_mean", "force_std"):
        if key not in stats:
            raise KeyError(f"normalization stats missing required key: {key}")
        if not torch.is_tensor(stats[key]):
            raise ValueError(f"normalization stats {key!r} must be a torch.Tensor")
    return stats


def _normalize_batch(batch: Dict[str, object], stats: Dict[str, torch.Tensor]) -> Dict[str, object]:
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


def _move_batch_to_device(batch: Dict[str, object], device: torch.device) -> Dict[str, object]:
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def _model_kwargs_from_checkpoint(checkpoint: dict, force_window_len: int) -> dict:
    config = checkpoint.get("config", {})
    model_config = dict(config.get("model", {}))
    if not model_config:
        raise KeyError("checkpoint config is missing model settings")
    if "pretrained_vision" in model_config and "pretrained_resnet18" not in model_config:
        model_config["pretrained_resnet18"] = model_config.pop("pretrained_vision")
    model_config.setdefault("pretrained_resnet18", False)
    model_config.setdefault("dropout", 0.0)
    model_config.setdefault("max_force_window_len", max(int(force_window_len), 20))
    return model_config


def _shape(tensor: torch.Tensor) -> tuple[int, ...]:
    return tuple(tensor.shape)


def _print_shapes(outputs_zero: dict, outputs_prior: dict, outputs_posterior: dict) -> None:
    print("Tensor Shapes")
    print("-------------")
    shape_items = {
        "pred_action_zero": outputs_zero["pred_action"],
        "pred_action_prior": outputs_prior["pred_action"],
        "pred_action_posterior": outputs_posterior["pred_action"],
        "pred_force_zero": outputs_zero["pred_force"],
        "pred_force_prior": outputs_prior["pred_force"],
        "pred_force_posterior": outputs_posterior["pred_force"],
        "z_contact_zero": outputs_zero["z_contact"],
        "z_contact_prior": outputs_prior["z_contact"],
        "z_contact_posterior": outputs_posterior["z_contact"],
    }
    for name, tensor in shape_items.items():
        print(f"{name}: {_shape(tensor)}")


def _print_metric(name: str, value: torch.Tensor) -> None:
    print(f"{name}: {value.detach().cpu().item():.6g}")


def run_debug(args: argparse.Namespace) -> int:
    device = torch.device(args.device)
    stats = _load_normalization_stats(args.normalization_stats)
    dataset = ContactForceHDF5Dataset(
        args.episode_paths,
        camera_names=tuple(args.camera_names),
        action_mode="joint_pos",
        chunk_len=args.chunk_len,
        force_window_len=args.force_window_len,
        force_window_duration=args.force_window_duration,
        image_size=tuple(args.image_size),
        imagenet_normalize=False,
    )
    if len(dataset) == 0:
        print("error: dataset is empty for the requested settings", file=sys.stderr)
        return 1
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    batch = _move_batch_to_device(next(iter(dataloader)), device)
    batch = _normalize_batch(batch, stats)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint must contain a dict")
    model = ForceAwareACTPolicy(
        **_model_kwargs_from_checkpoint(checkpoint, args.force_window_len)
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    with torch.no_grad():
        outputs_zero = model(
            images=batch["images"],
            qpos=batch["qpos"],
            force_window=batch["force_window"],
            action_chunk=None,
            future_force_chunk=None,
            is_training=False,
            contact_latent_mode="zero",
        )
        outputs_prior = model(
            images=batch["images"],
            qpos=batch["qpos"],
            force_window=batch["force_window"],
            action_chunk=None,
            future_force_chunk=None,
            is_training=False,
            contact_latent_mode="prior",
        )
        outputs_posterior = model(
            images=batch["images"],
            qpos=batch["qpos"],
            force_window=batch["force_window"],
            action_chunk=batch["action_chunk"],
            future_force_chunk=batch["future_force_chunk"],
            is_training=True,
            contact_latent_mode="posterior",
        )

    print(f"dataset_length={len(dataset)}")
    print(f"batch_size={batch['qpos'].shape[0]}")
    _print_shapes(outputs_zero, outputs_prior, outputs_posterior)

    action_target = batch["action_chunk"]
    force_target = batch["future_force_chunk"]
    z_delta = outputs_prior["z_contact"] - outputs_posterior["z_contact"]

    print("\nMetrics")
    print("-------")
    _print_metric(
        "action_l1_zero_to_target",
        functional.l1_loss(outputs_zero["pred_action"], action_target),
    )
    _print_metric(
        "action_l1_prior_to_target",
        functional.l1_loss(outputs_prior["pred_action"], action_target),
    )
    _print_metric(
        "action_l1_posterior_to_target",
        functional.l1_loss(outputs_posterior["pred_action"], action_target),
    )
    _print_metric(
        "force_l1_zero_to_target",
        functional.l1_loss(outputs_zero["pred_force"], force_target),
    )
    _print_metric(
        "force_l1_prior_to_target",
        functional.l1_loss(outputs_prior["pred_force"], force_target),
    )
    _print_metric(
        "force_l1_posterior_to_target",
        functional.l1_loss(outputs_posterior["pred_force"], force_target),
    )
    _print_metric("z_prior_to_posterior_mse", z_delta.pow(2).mean())
    _print_metric("z_prior_to_posterior_l2", z_delta.norm(dim=-1).mean())
    _print_metric(
        "z_prior_to_posterior_cosine",
        functional.cosine_similarity(
            outputs_prior["z_contact"],
            outputs_posterior["z_contact"],
            dim=-1,
        ).mean(),
    )
    _print_metric(
        "pred_action_zero_prior_mean_abs_diff",
        (outputs_zero["pred_action"] - outputs_prior["pred_action"]).abs().mean(),
    )
    _print_metric(
        "pred_force_zero_prior_mean_abs_diff",
        (outputs_zero["pred_force"] - outputs_prior["pred_force"]).abs().mean(),
    )
    return 0


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare ForceAwareACT inference contact modes.")
    parser.add_argument("episode_paths", type=Path, nargs="*", help="One or more HDF5 episodes.")
    parser.add_argument("--episode-list", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--normalization-stats", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--chunk-len", type=int, default=10)
    parser.add_argument("--force-window-len", type=int, default=20)
    parser.add_argument("--force-window-duration", type=float, default=0.25)
    parser.add_argument("--image-size", type=int, nargs=2, default=(224, 224))
    parser.add_argument("--camera-names", nargs="+", default=("ee_cam", "base_top_cam"))
    parser.add_argument("--device", default="cpu")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    args.episode_paths = resolve_episode_paths(args.episode_paths, args.episode_list)
    args.checkpoint = args.checkpoint.expanduser()
    args.normalization_stats = args.normalization_stats.expanduser()
    if not args.episode_paths:
        print("error: provide episode paths or --episode-list", file=sys.stderr)
        return 2
    if not validate_episode_paths(args.episode_paths):
        return 2
    if not args.checkpoint.is_file():
        print(f"error: checkpoint does not exist: {args.checkpoint}", file=sys.stderr)
        return 2
    if not args.normalization_stats.is_file():
        print(
            f"error: normalization stats file does not exist: {args.normalization_stats}",
            file=sys.stderr,
        )
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
        return run_debug(args)
    except Exception as error:
        print(f"error: inference mode debug failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
