#!/usr/bin/env python3
"""Evaluate ForceAwareACT zero, prior, and posterior contact modes over many batches."""

from __future__ import annotations

import argparse
import csv
import statistics
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
from force_aware_act.utils import resolve_episode_paths, validate_episode_paths  # noqa: E402


METRIC_COLUMNS = (
    "action_l1_zero",
    "action_l1_prior",
    "action_l1_posterior",
    "force_l1_zero",
    "force_l1_prior",
    "force_l1_posterior",
    "mu_prior_to_mu_posterior_mse",
    "mu_prior_to_mu_posterior_l2",
    "mu_prior_to_mu_posterior_cosine",
    "pred_action_zero_prior_mean_abs_diff",
    "pred_force_zero_prior_mean_abs_diff",
)


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


def _scalar(tensor: torch.Tensor) -> float:
    return float(tensor.detach().cpu().item())


def _batch_metrics(model: ForceAwareACTPolicy, batch: Dict[str, object]) -> dict[str, float]:
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
            deterministic_prior=True,
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

    action_target = batch["action_chunk"]
    force_target = batch["future_force_chunk"]
    mu_delta = outputs_prior["mu_contact_prior"] - outputs_posterior["mu_contact"]

    return {
        "action_l1_zero": _scalar(functional.l1_loss(outputs_zero["pred_action"], action_target)),
        "action_l1_prior": _scalar(functional.l1_loss(outputs_prior["pred_action"], action_target)),
        "action_l1_posterior": _scalar(
            functional.l1_loss(outputs_posterior["pred_action"], action_target)
        ),
        "force_l1_zero": _scalar(functional.l1_loss(outputs_zero["pred_force"], force_target)),
        "force_l1_prior": _scalar(functional.l1_loss(outputs_prior["pred_force"], force_target)),
        "force_l1_posterior": _scalar(
            functional.l1_loss(outputs_posterior["pred_force"], force_target)
        ),
        "mu_prior_to_mu_posterior_mse": _scalar(mu_delta.pow(2).mean()),
        "mu_prior_to_mu_posterior_l2": _scalar(mu_delta.norm(dim=-1).mean()),
        "mu_prior_to_mu_posterior_cosine": _scalar(
            functional.cosine_similarity(
                outputs_prior["mu_contact_prior"],
                outputs_posterior["mu_contact"],
                dim=-1,
            ).mean()
        ),
        "pred_action_zero_prior_mean_abs_diff": _scalar(
            (outputs_zero["pred_action"] - outputs_prior["pred_action"]).abs().mean()
        ),
        "pred_force_zero_prior_mean_abs_diff": _scalar(
            (outputs_zero["pred_force"] - outputs_prior["pred_force"]).abs().mean()
        ),
    }


def _summarize(values: Sequence[float]) -> dict[str, float]:
    if not values:
        raise ValueError("cannot summarize empty metric values")
    return {
        "mean": statistics.fmean(values),
        "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
        "median": statistics.median(values),
    }


def _print_summary(rows: list[dict[str, float]]) -> None:
    print("\nAggregate Metrics")
    print("-----------------")
    for column in METRIC_COLUMNS:
        stats = _summarize([row[column] for row in rows])
        print(
            f"{column}: "
            f"mean={stats['mean']:.6g} "
            f"std={stats['std']:.6g} "
            f"min={stats['min']:.6g} "
            f"max={stats['max']:.6g} "
            f"median={stats['median']:.6g}"
        )

    action_zero = [row["action_l1_zero"] for row in rows]
    action_prior = [row["action_l1_prior"] for row in rows]
    force_zero = [row["force_l1_zero"] for row in rows]
    force_prior = [row["force_l1_prior"] for row in rows]
    action_improvements = [
        (zero - prior) / zero for zero, prior in zip(action_zero, action_prior) if zero != 0
    ]
    force_improvements = [
        (zero - prior) / zero for zero, prior in zip(force_zero, force_prior) if zero != 0
    ]
    print("\nImprovement Ratios")
    print("------------------")
    for name, values in (
        ("action_prior_improvement_vs_zero", action_improvements),
        ("force_prior_improvement_vs_zero", force_improvements),
    ):
        stats = _summarize(values)
        print(
            f"{name}: "
            f"mean={stats['mean']:.6g} "
            f"std={stats['std']:.6g} "
            f"min={stats['min']:.6g} "
            f"max={stats['max']:.6g} "
            f"median={stats['median']:.6g}"
        )


def _write_csv(path: Path, rows: list[dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["batch_index", *METRIC_COLUMNS]
    with path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved_csv={path}")


def evaluate(args: argparse.Namespace) -> int:
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
    model = ForceAwareACTPolicy(
        **_model_kwargs_from_checkpoint(checkpoint, args.force_window_len)
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    rows = []
    for batch_index, batch in enumerate(dataloader, start=1):
        if batch_index > args.max_batches:
            break
        batch = _move_batch_to_device(batch, device)
        batch = _normalize_batch(batch, stats)
        row = {"batch_index": batch_index, **_batch_metrics(model, batch)}
        rows.append(row)
        print(
            " ".join(
                [
                    f"batch={batch_index}",
                    f"action_l1_zero={row['action_l1_zero']:.6g}",
                    f"action_l1_prior={row['action_l1_prior']:.6g}",
                    f"force_l1_zero={row['force_l1_zero']:.6g}",
                    f"force_l1_prior={row['force_l1_prior']:.6g}",
                ]
            ),
            flush=True,
        )

    if not rows:
        print("error: no batches were evaluated", file=sys.stderr)
        return 1

    print(f"dataset_length={len(dataset)}")
    print(f"evaluated_batches={len(rows)}")
    _print_summary(rows)
    if args.output_csv is not None:
        _write_csv(args.output_csv, rows)
    return 0


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate ForceAwareACT inference contact modes.")
    parser.add_argument("episode_paths", type=Path, nargs="*", help="One or more HDF5 episodes.")
    parser.add_argument("--episode-list", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--normalization-stats", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-batches", type=int, default=50)
    parser.add_argument("--chunk-len", type=int, default=10)
    parser.add_argument("--force-window-len", type=int, default=20)
    parser.add_argument("--force-window-duration", type=float, default=0.25)
    parser.add_argument("--image-size", type=int, nargs=2, default=(224, 224))
    parser.add_argument("--camera-names", nargs="+", default=("ee_cam", "base_top_cam"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-csv", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    args.episode_paths = resolve_episode_paths(
        args.episode_paths, args.episode_list, project_root=REPO_ROOT
    )
    args.checkpoint = args.checkpoint.expanduser()
    args.normalization_stats = args.normalization_stats.expanduser()
    if args.output_csv is not None:
        args.output_csv = args.output_csv.expanduser()
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
    if args.max_batches <= 0:
        print("error: --max-batches must be positive", file=sys.stderr)
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
        return evaluate(args)
    except Exception as error:
        print(f"error: inference mode evaluation failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
