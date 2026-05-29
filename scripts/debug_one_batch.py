#!/usr/bin/env python3
"""Run one read-only real-HDF5 batch through ForceAwareACTPolicy."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Mapping, Optional, Sequence

import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from force_aware_act.data import ContactForceHDF5Dataset  # noqa: E402
from force_aware_act.models import ForceAwareACTPolicy  # noqa: E402
from force_aware_act.training import compute_force_aware_act_loss  # noqa: E402


def _shape(value) -> tuple:
    return tuple(value.shape) if hasattr(value, "shape") else ()


def _print_tensor_shapes(title: str, tensors: Mapping[str, object]) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    for key, value in tensors.items():
        if torch.is_tensor(value):
            print(f"{key}: {_shape(value)}")


def _has_nonzero_gradient(model: torch.nn.Module) -> bool:
    return any(
        parameter.grad is not None and parameter.grad.detach().abs().sum().item() > 0
        for parameter in model.parameters()
    )


def run_debug(args: argparse.Namespace) -> int:
    dataset = ContactForceHDF5Dataset(
        args.episode_path,
        camera_names=("ee_cam", "base_top_cam"),
        action_mode="joint_pos",
        chunk_len=args.chunk_len,
        force_window_len=args.force_window_len,
        force_window_duration=0.25,
        image_size=(224, 224),
        imagenet_normalize=args.imagenet_normalize,
    )
    print(f"Dataset length: {len(dataset)}")
    if len(dataset) == 0:
        print("error: dataset is empty for the requested chunk length", file=sys.stderr)
        return 1

    dataloader = DataLoader(dataset, batch_size=2, shuffle=False)
    batch = next(iter(dataloader))
    _print_tensor_shapes(
        "Batch Tensor Shapes",
        {
            "images": batch["images"],
            "qpos": batch["qpos"],
            "force_window": batch["force_window"],
            "action_chunk": batch["action_chunk"],
            "future_force_chunk": batch["future_force_chunk"],
        },
    )

    model = ForceAwareACTPolicy(
        pretrained_resnet18=False,
        d_model=128,
        z_dim=16,
        action_dim=7,
        force_dim=6,
        chunk_len=args.chunk_len,
        nhead=4,
        num_encoder_layers=1,
        num_decoder_layers=1,
        dim_feedforward=256,
        dropout=0.0,
        max_force_window_len=max(args.force_window_len, 20),
    )
    model.train()

    outputs = model(
        images=batch["images"],
        qpos=batch["qpos"],
        force_window=batch["force_window"],
        action_chunk=batch["action_chunk"],
        future_force_chunk=batch["future_force_chunk"],
        is_training=True,
    )
    _print_tensor_shapes(
        "Output Tensor Shapes",
        {
            "pred_action": outputs["pred_action"],
            "pred_force": outputs["pred_force"],
            "visual_tokens": outputs["visual_tokens"],
            "z_q": outputs["z_q"],
            "z_F_online": outputs["z_F_online"],
            "z_VF": outputs["z_VF"],
            "z_motion": outputs["z_motion"],
            "z_contact": outputs["z_contact"],
            "decoder_hidden": outputs["decoder_hidden"],
            "mu_motion": outputs["mu_motion"],
            "logvar_motion": outputs["logvar_motion"],
            "mu_contact": outputs["mu_contact"],
            "logvar_contact": outputs["logvar_contact"],
        },
    )

    losses = compute_force_aware_act_loss(
        outputs=outputs,
        action_chunk=batch["action_chunk"],
        future_force_chunk=batch["future_force_chunk"],
        lambda_force=0.1,
        beta_motion=1.0e-4,
        beta_contact=1.0e-4,
    )
    losses["loss_total"].backward()

    print("\nLoss Values")
    print("-----------")
    for key in ("loss_total", "loss_action", "loss_force", "kl_motion", "kl_contact"):
        print(f"{key}: {losses[key].item():.6g}")
    print(f"lambda_force: {losses['lambda_force']}")
    print(f"beta_motion: {losses['beta_motion']}")
    print(f"beta_contact: {losses['beta_contact']}")

    has_gradient = _has_nonzero_gradient(model)
    print("\nGradient Check")
    print("--------------")
    print(f"nonzero_model_gradient: {has_gradient}")
    if not has_gradient:
        print("error: no model parameter received a nonzero gradient", file=sys.stderr)
        return 1
    return 0


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one ForceAwareACTPolicy debug batch from a read-only HDF5 episode.",
    )
    parser.add_argument("episode_path", type=Path, help="Path to one HDF5 episode file.")
    parser.add_argument("--chunk-len", type=int, default=10)
    parser.add_argument("--force-window-len", type=int, default=20)
    parser.add_argument("--imagenet-normalize", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    args.episode_path = args.episode_path.expanduser()
    if not args.episode_path.exists():
        print(f"error: file does not exist: {args.episode_path}", file=sys.stderr)
        return 2
    if not args.episode_path.is_file():
        print(f"error: path is not a file: {args.episode_path}", file=sys.stderr)
        return 2
    if args.chunk_len <= 0:
        print("error: --chunk-len must be positive", file=sys.stderr)
        return 2
    if args.force_window_len <= 0:
        print("error: --force-window-len must be positive", file=sys.stderr)
        return 2

    try:
        return run_debug(args)
    except Exception as error:
        print(f"error: one-batch debug failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
