#!/usr/bin/env python3
"""Stage-2 training for the ForceAwareACT contact prior.

Example:
    PYTHONPATH=src .venv/bin/python scripts/train_contact_prior_stage2.py \
      --episode-list peg_in_hole_hdf5/episodes_train.txt \
      --checkpoint outputs/minimal_train/checkpoint.pt \
      --normalization-stats outputs/normalization_stats_10eps.pt \
      --max-steps 1000
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence

import torch
import torch.nn.functional as functional
from torch.optim import AdamW
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from force_aware_act.data import ContactForceHDF5Dataset, normalize_tensor  # noqa: E402
from force_aware_act.models import ForceAwareACTPolicy  # noqa: E402
from force_aware_act.training import compute_contact_prior_distillation_loss  # noqa: E402
from force_aware_act.utils import resolve_episode_paths, validate_episode_paths  # noqa: E402


def _move_batch_to_device(batch: Dict[str, object], device: torch.device) -> Dict[str, object]:
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def _cycle_batches(dataloader: DataLoader) -> Iterable[Dict[str, object]]:
    while True:
        for batch in dataloader:
            yield batch


def _load_normalization_stats(stats_path: Path) -> Dict[str, torch.Tensor]:
    stats = torch.load(stats_path, map_location="cpu")
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
    normalized["qpos"] = normalize_tensor(
        normalized["qpos"],
        stats["qpos_mean"],
        stats["qpos_std"],
    )
    normalized["force_window"] = normalize_tensor(
        normalized["force_window"],
        stats["force_mean"],
        stats["force_std"],
    )
    normalized["action_chunk"] = normalize_tensor(
        normalized["action_chunk"],
        stats["action_mean"],
        stats["action_std"],
    )
    normalized["future_force_chunk"] = normalize_tensor(
        normalized["future_force_chunk"],
        stats["force_mean"],
        stats["force_std"],
    )
    return normalized


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


def _freeze_except_contact_prior(model: ForceAwareACTPolicy) -> None:
    for name, parameter in model.named_parameters():
        parameter.requires_grad = name.startswith("contact_prior.")


def _trainable_parameter_report(model: ForceAwareACTPolicy) -> tuple[list[str], int]:
    trainable_names = []
    trainable_count = 0
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            trainable_names.append(name)
            trainable_count += parameter.numel()
    return trainable_names, trainable_count


def _assert_gradient_state(model: ForceAwareACTPolicy) -> None:
    contact_prior_has_grad = False
    frozen_with_grad = []
    for name, parameter in model.named_parameters():
        grad_nonzero = parameter.grad is not None and parameter.grad.abs().sum() > 0
        if name.startswith("contact_prior."):
            contact_prior_has_grad = contact_prior_has_grad or grad_nonzero
        elif parameter.grad is not None and parameter.grad.abs().sum() > 0:
            frozen_with_grad.append(name)

    if frozen_with_grad:
        raise RuntimeError(
            "frozen parameters unexpectedly received gradients: "
            + ", ".join(frozen_with_grad[:10])
        )
    if not contact_prior_has_grad:
        raise RuntimeError("no contact_prior parameter received nonzero gradient")


def _prior_metrics(outputs: Dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    prior_delta = outputs["mu_contact_prior"] - outputs["mu_contact"].detach()
    return {
        "prior_mu_mse": prior_delta.pow(2).mean(),
        "prior_mu_l2": prior_delta.norm(dim=-1).mean(),
        "prior_mu_cosine_similarity": functional.cosine_similarity(
            outputs["mu_contact_prior"],
            outputs["mu_contact"].detach(),
            dim=-1,
        ).mean(),
    }


def _config_from_args(args: argparse.Namespace, model_config: dict) -> Dict[str, object]:
    return {
        "episode_paths": [str(path) for path in args.episode_paths],
        "checkpoint": str(args.checkpoint),
        "normalization_stats": str(args.normalization_stats),
        "output_dir": str(args.output_dir),
        "log_csv": str(args.log_csv),
        "max_steps": args.max_steps,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "prior_loss_mode": args.prior_loss_mode,
        "chunk_len": args.chunk_len,
        "force_window_len": args.force_window_len,
        "force_window_duration": args.force_window_duration,
        "image_size": tuple(args.image_size),
        "camera_names": tuple(args.camera_names),
        "device": args.device,
        "model": model_config,
    }


def train(args: argparse.Namespace) -> int:
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

    checkpoint = torch.load(args.checkpoint, map_location=device)
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint must contain a dict")
    model_config = _model_kwargs_from_checkpoint(checkpoint, args.force_window_len)
    model = ForceAwareACTPolicy(**model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    _freeze_except_contact_prior(model)
    model.eval()
    model.contact_prior.train()

    trainable_names, trainable_count = _trainable_parameter_report(model)
    print(f"dataset_length={len(dataset)}")
    print(f"stage1_checkpoint={args.checkpoint}")
    print(f"normalization_stats={args.normalization_stats}")
    print(f"checkpoint_path={args.output_dir / 'checkpoint.pt'}")
    print(f"log_csv={args.log_csv}")
    print("trainable_parameters:")
    for name in trainable_names:
        print(f"  {name}")
    print(f"trainable_parameter_count={trainable_count}")

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    optimizer = AdamW(
        [parameter for parameter in model.contact_prior.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
    )
    batch_iter = _cycle_batches(dataloader)
    last_step = 0
    args.log_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.log_csv.open("w", newline="") as log_file:
        log_writer = csv.DictWriter(
            log_file,
            fieldnames=[
                "step",
                "loss_prior",
                "prior_loss_mode",
                "prior_mu_mse",
                "prior_mu_l2",
                "prior_mu_cosine_similarity",
            ],
        )
        log_writer.writeheader()
        for step in range(1, args.max_steps + 1):
            last_step = step
            batch = _move_batch_to_device(next(batch_iter), device)
            batch = _normalize_batch(batch, stats)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(
                images=batch["images"],
                qpos=batch["qpos"],
                force_window=batch["force_window"],
                action_chunk=batch["action_chunk"],
                future_force_chunk=batch["future_force_chunk"],
                is_training=True,
            )
            prior_losses = compute_contact_prior_distillation_loss(
                mu_prior=outputs["mu_contact_prior"],
                logvar_prior=outputs["logvar_contact_prior"],
                mu_posterior=outputs["mu_contact"],
                logvar_posterior=outputs["logvar_contact"],
                mode=args.prior_loss_mode,
            )
            metrics = _prior_metrics(outputs)
            prior_losses["loss_prior"].backward()
            _assert_gradient_state(model)
            optimizer.step()

            log_writer.writerow(
                {
                    "step": step,
                    "loss_prior": prior_losses["loss_prior"].item(),
                    "prior_loss_mode": args.prior_loss_mode,
                    "prior_mu_mse": metrics["prior_mu_mse"].item(),
                    "prior_mu_l2": metrics["prior_mu_l2"].item(),
                    "prior_mu_cosine_similarity": metrics[
                        "prior_mu_cosine_similarity"
                    ].item(),
                }
            )
            print(
                " ".join(
                    [
                        f"step={step}",
                        f"loss_prior={prior_losses['loss_prior'].item():.6g}",
                        f"prior_mu_mse={metrics['prior_mu_mse'].item():.6g}",
                        f"prior_mu_l2={metrics['prior_mu_l2'].item():.6g}",
                        "prior_mu_cosine_similarity="
                        f"{metrics['prior_mu_cosine_similarity'].item():.6g}",
                    ]
                ),
                flush=True,
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": _config_from_args(args, model_config),
            "step": last_step,
        },
        args.output_dir / "checkpoint.pt",
    )
    print(f"saved_checkpoint={args.output_dir / 'checkpoint.pt'}")
    return 0


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage-2 contact-prior distillation training.")
    parser.add_argument("episode_paths", type=Path, nargs="*", help="One or more HDF5 episodes.")
    parser.add_argument("--episode-list", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--normalization-stats", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/contact_prior_stage2"))
    parser.add_argument(
        "--log-csv",
        type=Path,
        default=Path("outputs/contact_prior_stage2/train_log.csv"),
    )
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--prior-loss-mode", choices=("mse_mu", "kl_q_to_p"), default="mse_mu")
    parser.add_argument("--chunk-len", type=int, default=10)
    parser.add_argument("--force-window-len", type=int, default=20)
    parser.add_argument("--force-window-duration", type=float, default=0.25)
    parser.add_argument("--image-size", type=int, nargs=2, default=(224, 224))
    parser.add_argument("--camera-names", nargs="+", default=("ee_cam", "base_top_cam"))
    parser.add_argument("--device", default="cpu")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    args.episode_paths = resolve_episode_paths(
        args.episode_paths, args.episode_list, project_root=REPO_ROOT
    )
    args.checkpoint = args.checkpoint.expanduser()
    args.normalization_stats = args.normalization_stats.expanduser()
    args.output_dir = args.output_dir.expanduser()
    args.log_csv = args.log_csv.expanduser()

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
    if args.max_steps <= 0:
        print("error: --max-steps must be positive", file=sys.stderr)
        return 2
    if args.batch_size <= 0:
        print("error: --batch-size must be positive", file=sys.stderr)
        return 2
    if args.learning_rate <= 0:
        print("error: --learning-rate must be positive", file=sys.stderr)
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
        return train(args)
    except Exception as error:
        print(f"error: stage-2 contact-prior training failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
