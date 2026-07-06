#!/usr/bin/env python3
"""Evaluate ACT baseline zero and posterior motion latent modes."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from force_aware_act.data import ContactForceHDF5Dataset, normalize_tensor  # noqa: E402
from force_aware_act.models import ACTPolicyBaseline  # noqa: E402
from force_aware_act.utils import resolve_episode_paths, validate_episode_paths  # noqa: E402


ACTION_MODE_CHOICES = (
    "joint_pos",
    "action",
    "joint_pos_command",
    "delta_joint_cmd",
    "delta_joint_pos_command",
)
POSTERIOR_MODE_CHOICES = ("mean", "sample")
POLICY_VARIANT = "act_baseline"
ACT_BASELINE_VERSION = ACTPolicyBaseline.act_baseline_version
RATIO_EPSILON = 1.0e-12
CSV_COLUMNS = (
    "global_sample_index",
    "dataset_index",
    "episode_path",
    "episode_identifier",
    "timestep_index",
    "action_l1_zero",
    "action_l1_posterior",
    "action_l1_zero_minus_posterior",
    "action_l1_zero_over_posterior",
    "pred_action_zero_posterior_mean_abs_diff",
    "kl_motion",
    "mu_motion_l2",
    "mu_motion_abs_mean",
    "logvar_motion_mean",
    "posterior_std_mean",
)
METRIC_COLUMNS = CSV_COLUMNS[5:]


@dataclass(frozen=True)
class ACTModeOutputs:
    zero: Dict[str, Any]
    posterior: Dict[str, Any]
    mu_motion: torch.Tensor
    logvar_motion: torch.Tensor
    z_motion_posterior: torch.Tensor


def resolve_device(device_name: str) -> torch.device:
    if device_name == "cuda" and not torch.cuda.is_available():
        raise ValueError("--device cuda was requested, but CUDA is not available")
    if device_name == "mps":
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is None or not mps_backend.is_available():
            raise ValueError("--device mps was requested, but MPS is not available")
    if device_name not in {"cpu", "cuda", "mps"}:
        raise ValueError("--device must be one of: cpu, mps, cuda")
    return torch.device(device_name)


def load_normalization_stats(path: Path) -> Dict[str, torch.Tensor]:
    stats = torch.load(path, map_location="cpu")
    if not isinstance(stats, dict):
        raise ValueError("normalization stats file must contain a dict")
    for key in ("qpos_mean", "qpos_std", "action_mean", "action_std"):
        if key not in stats:
            raise KeyError(f"normalization stats missing required key: {key}")
        if not torch.is_tensor(stats[key]):
            raise ValueError(f"normalization stats {key!r} must be a torch.Tensor")
    return stats


def validate_normalization_action_mode(stats: Mapping[str, object], action_mode: str) -> None:
    if "action_mode" not in stats:
        return
    stats_action_mode = stats["action_mode"]
    if stats_action_mode != action_mode:
        raise ValueError(
            "normalization stats action_mode mismatch: "
            f"stats action_mode={stats_action_mode!r}, requested action_mode={action_mode!r}. "
            "Recompute normalization stats for the requested action_mode."
        )


def normalize_batch(
    batch: Dict[str, object],
    stats: Mapping[str, torch.Tensor],
) -> Dict[str, object]:
    normalized = dict(batch)
    normalized["qpos"] = normalize_tensor(
        normalized["qpos"],
        stats["qpos_mean"],
        stats["qpos_std"],
    )
    normalized["action_chunk"] = normalize_tensor(
        normalized["action_chunk"],
        stats["action_mean"],
        stats["action_std"],
    )
    return normalized


def move_batch_to_device(batch: Dict[str, object], device: torch.device) -> Dict[str, object]:
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def build_evaluation_dataset(args: argparse.Namespace) -> ContactForceHDF5Dataset:
    return ContactForceHDF5Dataset(
        args.episode_paths,
        camera_names=tuple(args.camera_names),
        action_mode=args.action_mode,
        chunk_len=args.chunk_len,
        force_window_len=1,
        force_window_duration=0.0,
        image_size=tuple(args.image_size),
        imagenet_normalize=False,
        include_force=False,
    )


def _state_dict_from_checkpoint(checkpoint: object) -> Mapping[str, torch.Tensor]:
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
    if not isinstance(state_dict, Mapping):
        raise ValueError("checkpoint must be a state_dict or contain model_state_dict")
    if not all(torch.is_tensor(value) for value in state_dict.values()):
        raise ValueError("model state_dict must map parameter names to tensors")
    return state_dict


def _model_config_from_checkpoint(checkpoint: object, args: argparse.Namespace) -> dict[str, object]:
    if not isinstance(checkpoint, dict):
        raise ValueError("ACT baseline checkpoint must contain config metadata")
    config = checkpoint.get("config", {})
    if not isinstance(config, dict):
        raise ValueError("ACT baseline checkpoint config must be a dict")
    policy_variant = str(config.get("policy_variant", ""))
    if policy_variant != POLICY_VARIANT:
        raise ValueError(f"checkpoint policy_variant must be {POLICY_VARIANT!r}, got {policy_variant!r}")
    version = str(config.get("act_baseline_version", ""))
    if version != ACT_BASELINE_VERSION:
        raise ValueError(
            "ACT baseline evaluator requires act_baseline_version="
            f"{ACT_BASELINE_VERSION!r}; got {version!r}. Legacy zero-latent "
            "checkpoints do not contain a motion posterior."
        )
    model_config = dict(config.get("model", {}))
    if not model_config:
        raise KeyError("checkpoint config is missing model settings")
    if "pretrained_vision" in model_config and "pretrained_resnet18" not in model_config:
        model_config["pretrained_resnet18"] = model_config.pop("pretrained_vision")
    model_config.setdefault("pretrained_resnet18", False)
    model_config.setdefault("dropout", 0.0)
    model_config.setdefault("freeze_resnet18", False)
    if int(model_config.get("chunk_len", args.chunk_len)) != args.chunk_len:
        raise ValueError(
            f"--chunk-len={args.chunk_len} does not match checkpoint chunk_len="
            f"{model_config.get('chunk_len')}"
        )
    model_config.pop("force_dim", None)
    model_config.pop("max_force_window_len", None)
    return model_config


def load_act_baseline_checkpoint(
    checkpoint_path: Path,
    args: argparse.Namespace,
    device: torch.device,
) -> ACTPolicyBaseline:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model_config = _model_config_from_checkpoint(checkpoint, args)
    state_dict = _state_dict_from_checkpoint(checkpoint)
    model = ACTPolicyBaseline(**model_config).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def run_act_modes(
    model: ACTPolicyBaseline,
    batch: Mapping[str, torch.Tensor],
    posterior_mode: str,
) -> ACTModeOutputs:
    with torch.no_grad():
        outputs_zero = model(
            images=batch["images"],
            qpos=batch["qpos"],
            action_chunk=None,
            is_training=False,
        )
        mu_motion, logvar_motion, z_motion_sample = model.encode_motion_posterior(
            batch["qpos"],
            batch["action_chunk"],
        )
        z_motion_posterior = mu_motion if posterior_mode == "mean" else z_motion_sample
        outputs_posterior = model(
            images=batch["images"],
            qpos=batch["qpos"],
            action_chunk=None,
            is_training=False,
            motion_latent_override=z_motion_posterior,
        )
    validate_prediction_shapes(outputs_zero, batch["action_chunk"], "zero")
    validate_prediction_shapes(outputs_posterior, batch["action_chunk"], "posterior")
    return ACTModeOutputs(
        zero=outputs_zero,
        posterior=outputs_posterior,
        mu_motion=mu_motion,
        logvar_motion=logvar_motion,
        z_motion_posterior=z_motion_posterior,
    )


def validate_prediction_shapes(
    outputs: Mapping[str, Any],
    action_target: torch.Tensor,
    label: str,
) -> None:
    for key in ("pred_action", "z_motion"):
        if key not in outputs or not torch.is_tensor(outputs[key]):
            raise KeyError(f"{label} outputs missing tensor key: {key}")
    if outputs["pred_action"].shape != action_target.shape:
        raise ValueError(
            f"{label} pred_action shape {tuple(outputs['pred_action'].shape)} "
            f"does not match target {tuple(action_target.shape)}"
        )


def compute_per_sample_l1(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if prediction.shape != target.shape:
        raise ValueError(
            f"prediction shape {tuple(prediction.shape)} does not match target {tuple(target.shape)}"
        )
    return (prediction - target).abs().mean(dim=tuple(range(1, prediction.ndim)))


def compute_safe_ratio(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
    return numerator / denominator.clamp_min(RATIO_EPSILON)


def compute_motion_kl_per_sample(mu_motion: torch.Tensor, logvar_motion: torch.Tensor) -> torch.Tensor:
    if mu_motion.shape != logvar_motion.shape:
        raise ValueError("mu_motion and logvar_motion must have matching shapes")
    return -0.5 * torch.sum(
        1 + logvar_motion - mu_motion.pow(2) - logvar_motion.exp(),
        dim=-1,
    )


def compute_motion_latent_stats(
    mu_motion: torch.Tensor,
    logvar_motion: torch.Tensor,
) -> dict[str, torch.Tensor]:
    posterior_std = torch.exp(0.5 * logvar_motion)
    return {
        "kl_motion": compute_motion_kl_per_sample(mu_motion, logvar_motion),
        "mu_motion_l2": mu_motion.norm(dim=-1),
        "mu_motion_abs_mean": mu_motion.abs().mean(dim=-1),
        "logvar_motion_mean": logvar_motion.mean(dim=-1),
        "posterior_std_mean": posterior_std.mean(dim=-1),
    }


def compute_sample_metrics(
    outputs: ACTModeOutputs,
    action_target: torch.Tensor,
) -> dict[str, torch.Tensor]:
    action_l1_zero = compute_per_sample_l1(outputs.zero["pred_action"], action_target)
    action_l1_posterior = compute_per_sample_l1(outputs.posterior["pred_action"], action_target)
    action_diff = compute_per_sample_l1(outputs.zero["pred_action"], outputs.posterior["pred_action"])
    metrics = {
        "action_l1_zero": action_l1_zero,
        "action_l1_posterior": action_l1_posterior,
        "action_l1_zero_minus_posterior": action_l1_zero - action_l1_posterior,
        "action_l1_zero_over_posterior": compute_safe_ratio(
            action_l1_zero,
            action_l1_posterior,
        ),
        "pred_action_zero_posterior_mean_abs_diff": action_diff,
    }
    metrics.update(compute_motion_latent_stats(outputs.mu_motion, outputs.logvar_motion))
    _validate_finite_metric_tensors(metrics)
    return metrics


def _validate_finite_metric_tensors(metrics: Mapping[str, torch.Tensor]) -> None:
    for name, value in metrics.items():
        if not torch.isfinite(value).all().item():
            raise ValueError(f"metric {name} contains NaN or Inf")


def _batch_value(batch: Mapping[str, object], key: str, sample_index: int) -> object:
    value = batch.get(key)
    if value is None:
        return ""
    if torch.is_tensor(value):
        return value[sample_index].detach().cpu().item()
    return value[sample_index]


def _episode_identifier(episode_path: object) -> str:
    if episode_path == "":
        return ""
    return Path(str(episode_path)).parent.name


def sample_rows_from_metrics(
    batch: Mapping[str, object],
    metrics: Mapping[str, torch.Tensor],
    global_offset: int,
) -> list[dict[str, object]]:
    batch_size = int(next(iter(metrics.values())).shape[0])
    rows: list[dict[str, object]] = []
    for sample_index in range(batch_size):
        episode_path = _batch_value(batch, "episode_path", sample_index)
        row: dict[str, object] = {
            "global_sample_index": global_offset + sample_index,
            "dataset_index": global_offset + sample_index,
            "episode_path": episode_path,
            "episode_identifier": _episode_identifier(episode_path),
            "timestep_index": _batch_value(batch, "state_index", sample_index),
        }
        for column in METRIC_COLUMNS:
            row[column] = float(metrics[column][sample_index].detach().cpu().item())
        rows.append(row)
    return rows


def aggregate_sample_metrics(rows: Sequence[Mapping[str, object]]) -> dict[str, float]:
    if not rows:
        raise ValueError("cannot aggregate zero sample rows")
    aggregate: dict[str, float] = {}
    for column in METRIC_COLUMNS:
        values = [float(row[column]) for row in rows]
        value = sum(values) / len(values)
        if not math.isfinite(value):
            raise ValueError(f"aggregate metric {column} is not finite")
        aggregate[column] = value
    aggregate["action_l1_zero_over_posterior"] = aggregate["action_l1_zero"] / max(
        aggregate["action_l1_posterior"],
        RATIO_EPSILON,
    )
    return aggregate


def write_sample_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(CSV_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)


def print_summary(
    checkpoint: Path,
    posterior_mode: str,
    num_batches: int,
    num_samples: int,
    aggregate: Mapping[str, float],
    output_csv: Optional[Path],
) -> None:
    print("ACT baseline inference mode evaluation")
    print(f"checkpoint={checkpoint}")
    print(f"posterior_mode={posterior_mode}")
    print(f"num_batches={num_batches}")
    print(f"num_samples={num_samples}")
    print("")
    for key in METRIC_COLUMNS:
        print(f"{key}={aggregate[key]:.8f}")
    print(f"output_csv={output_csv if output_csv is not None else ''}")


def evaluate(args: argparse.Namespace) -> int:
    device = resolve_device(args.device)
    stats = load_normalization_stats(args.normalization_stats)
    validate_normalization_action_mode(stats, args.action_mode)
    dataset = build_evaluation_dataset(args)
    if len(dataset) == 0:
        print("error: dataset is empty for the requested settings", file=sys.stderr)
        return 1
    model = load_act_baseline_checkpoint(args.checkpoint, args, device)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    rows: list[dict[str, object]] = []
    num_batches = 0
    for batch_index, batch in enumerate(dataloader, start=1):
        if args.max_batches is not None and batch_index > args.max_batches:
            break
        normalized_batch = normalize_batch(batch, stats)
        device_batch = move_batch_to_device(normalized_batch, device)
        tensor_batch = {
            key: value for key, value in device_batch.items() if torch.is_tensor(value)
        }
        outputs = run_act_modes(model, tensor_batch, posterior_mode=args.posterior_mode)
        metrics = compute_sample_metrics(outputs, action_target=tensor_batch["action_chunk"])
        rows.extend(sample_rows_from_metrics(batch, metrics, global_offset=len(rows)))
        num_batches += 1

    if num_batches == 0:
        print("error: no batches were evaluated", file=sys.stderr)
        return 1

    aggregate = aggregate_sample_metrics(rows)
    if args.output_csv is not None:
        write_sample_csv(args.output_csv, rows)
    print_summary(
        checkpoint=args.checkpoint,
        posterior_mode=args.posterior_mode,
        num_batches=num_batches,
        num_samples=len(rows),
        aggregate=aggregate,
        output_csv=args.output_csv,
    )
    return 0


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate ACT baseline zero and posterior motion latent modes."
    )
    parser.add_argument("episode_paths", type=Path, nargs="*", help="One or more HDF5 episodes.")
    parser.add_argument("--episode-list", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--normalization-stats", type=Path, required=True)
    parser.add_argument("--action-mode", choices=ACTION_MODE_CHOICES, default="joint_pos")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--chunk-len", type=int, default=10)
    parser.add_argument("--image-size", type=int, nargs=2, default=(224, 224))
    parser.add_argument("--camera-names", nargs="+", default=("ee_cam", "base_top_cam"))
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--posterior-mode", choices=POSTERIOR_MODE_CHOICES, default="mean")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.episode_list is not None:
        args.episode_list = args.episode_list.expanduser()
        if not args.episode_list.is_file():
            print(f"error: episode list does not exist: {args.episode_list}", file=sys.stderr)
            return 2
    args.episode_paths = resolve_episode_paths(
        args.episode_paths,
        args.episode_list,
        project_root=REPO_ROOT,
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
    if args.max_batches is not None and args.max_batches <= 0:
        print("error: --max-batches must be positive when provided", file=sys.stderr)
        return 2
    if args.num_workers < 0:
        print("error: --num-workers must be non-negative", file=sys.stderr)
        return 2
    if args.chunk_len <= 0:
        print("error: --chunk-len must be positive", file=sys.stderr)
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
        if args.device == "mps":
            print(
                "error: ACT baseline evaluation failed on MPS. "
                "Retry with --device cpu if this PyTorch/torchvision build lacks MPS support "
                f"for an operator. Details: {error}",
                file=sys.stderr,
            )
        else:
            print(f"error: ACT baseline evaluation failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
