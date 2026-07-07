#!/usr/bin/env python3
"""Evaluate ForceAwareACT Contact-CVAE deployable and oracle contact modes."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import torch
import torch.nn.functional as functional
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from force_aware_act.data import ContactForceHDF5Dataset, normalize_tensor  # noqa: E402
from force_aware_act.models import ForceAwareACTContactCVAEPolicy  # noqa: E402
from force_aware_act.utils import resolve_episode_paths, validate_episode_paths  # noqa: E402


ACTION_MODE_CHOICES = (
    "joint_pos",
    "action",
    "joint_pos_command",
    "delta_joint_cmd",
    "delta_joint_pos_command",
)
POLICY_VARIANT = "force_aware_contact_cvae"
RATIO_EPSILON = 1.0e-12
CSV_COLUMNS = (
    "global_sample_index",
    "dataset_index",
    "episode_path",
    "episode_identifier",
    "state_index",
    "t_state",
    "action_l1_zero",
    "action_l1_prior",
    "action_l1_posterior",
    "force_l1_zero",
    "force_l1_prior",
    "force_l1_posterior",
    "pred_action_zero_prior_mean_abs_diff",
    "pred_action_zero_posterior_mean_abs_diff",
    "pred_action_prior_posterior_mean_abs_diff",
    "pred_force_zero_prior_mean_abs_diff",
    "pred_force_zero_posterior_mean_abs_diff",
    "pred_force_prior_posterior_mean_abs_diff",
    "kl_contact",
    "mu_contact_l2",
    "posterior_std_mean",
    "mu_prior_to_mu_posterior_mse",
    "mu_prior_to_mu_posterior_l2",
    "mu_prior_to_mu_posterior_cosine",
)
METRIC_COLUMNS = CSV_COLUMNS[6:]


@dataclass(frozen=True)
class ContactModeOutputs:
    zero: Dict[str, Any]
    prior: Dict[str, Any]
    posterior: Dict[str, Any]
    mu_contact: torch.Tensor
    logvar_contact: torch.Tensor
    z_contact_posterior: torch.Tensor


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
    for key in ("qpos_mean", "qpos_std", "action_mean", "action_std", "force_mean", "force_std"):
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


def normalize_batch(batch: Dict[str, object], stats: Mapping[str, torch.Tensor]) -> Dict[str, object]:
    normalized = dict(batch)
    normalized["qpos"] = normalize_tensor(normalized["qpos"], stats["qpos_mean"], stats["qpos_std"])
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
        force_window_len=args.force_window_len,
        force_window_duration=args.force_window_duration,
        image_size=tuple(args.image_size),
        imagenet_normalize=False,
    )


def checkpoint_policy_variant(checkpoint: object) -> str:
    if isinstance(checkpoint, dict):
        config = checkpoint.get("config", {})
        if isinstance(config, dict):
            return str(config.get("policy_variant", "force_aware_act"))
    return "force_aware_act"


def state_dict_from_checkpoint(checkpoint: object) -> Mapping[str, torch.Tensor]:
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
    if not isinstance(state_dict, Mapping):
        raise ValueError("checkpoint must be a state_dict or contain model_state_dict")
    if not all(torch.is_tensor(value) for value in state_dict.values()):
        raise ValueError("model state_dict must map parameter names to tensors")
    return state_dict


def model_config_from_args(args: argparse.Namespace) -> dict[str, object]:
    return {
        "pretrained_resnet18": False,
        "d_model": 128,
        "z_dim": 16,
        "action_dim": 7,
        "force_dim": 6,
        "chunk_len": args.chunk_len,
        "nhead": 4,
        "num_encoder_layers": 1,
        "num_decoder_layers": 1,
        "dim_feedforward": 256,
        "dropout": 0.0,
        "max_force_window_len": max(args.force_window_len, 20),
    }


def model_config_from_checkpoint(checkpoint: object, args: argparse.Namespace) -> dict[str, object]:
    if not isinstance(checkpoint, dict):
        return model_config_from_args(args)
    config = checkpoint.get("config", {})
    if not isinstance(config, dict):
        return model_config_from_args(args)
    model_config = dict(config.get("model", {}))
    if not model_config:
        return model_config_from_args(args)
    if "pretrained_vision" in model_config and "pretrained_resnet18" not in model_config:
        model_config["pretrained_resnet18"] = model_config.pop("pretrained_vision")
    model_config.setdefault("pretrained_resnet18", False)
    model_config.setdefault("dropout", 0.0)
    model_config.setdefault("max_force_window_len", max(int(args.force_window_len), 20))
    return model_config


def load_contact_cvae_checkpoint(
    checkpoint_path: Path,
    args: argparse.Namespace,
    device: torch.device,
) -> ForceAwareACTContactCVAEPolicy:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    policy_variant = checkpoint_policy_variant(checkpoint)
    if policy_variant != POLICY_VARIANT:
        raise ValueError(
            f"checkpoint policy_variant must be {POLICY_VARIANT!r}, got {policy_variant!r}"
        )
    model_config = model_config_from_checkpoint(checkpoint, args)
    if int(model_config.get("chunk_len", args.chunk_len)) != args.chunk_len:
        raise ValueError(
            f"--chunk-len={args.chunk_len} does not match checkpoint chunk_len="
            f"{model_config.get('chunk_len')}"
        )
    model = ForceAwareACTContactCVAEPolicy(**model_config).to(device)
    model.load_state_dict(state_dict_from_checkpoint(checkpoint), strict=True)
    model.eval()
    return model


def run_contact_modes(
    model: ForceAwareACTContactCVAEPolicy,
    batch: Mapping[str, torch.Tensor],
    posterior_mode: str,
) -> ContactModeOutputs:
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
        mu_contact, logvar_contact, z_contact_sample = model.encode_contact_posterior(
            batch["qpos"],
            batch["action_chunk"],
            batch["future_force_chunk"],
        )
        z_contact_posterior = mu_contact if posterior_mode == "mean" else z_contact_sample
        outputs_posterior = model(
            images=batch["images"],
            qpos=batch["qpos"],
            force_window=batch["force_window"],
            action_chunk=None,
            future_force_chunk=None,
            is_training=False,
            contact_latent_override=z_contact_posterior,
        )

    return ContactModeOutputs(
        zero=outputs_zero,
        prior=outputs_prior,
        posterior=outputs_posterior,
        mu_contact=mu_contact,
        logvar_contact=logvar_contact,
        z_contact_posterior=z_contact_posterior,
    )


def compute_per_sample_l1(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if prediction.shape != target.shape:
        raise ValueError(
            f"prediction shape {tuple(prediction.shape)} does not match target {tuple(target.shape)}"
        )
    return (prediction - target).abs().mean(dim=tuple(range(1, prediction.ndim)))


def compute_contact_kl_per_sample(mu_contact: torch.Tensor, logvar_contact: torch.Tensor) -> torch.Tensor:
    if mu_contact.shape != logvar_contact.shape:
        raise ValueError("mu_contact and logvar_contact must have matching shapes")
    return -0.5 * torch.sum(
        1 + logvar_contact - mu_contact.pow(2) - logvar_contact.exp(),
        dim=-1,
    )


def safe_cosine_similarity(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    numerator = (left * right).sum(dim=-1)
    denominator = left.norm(dim=-1) * right.norm(dim=-1)
    return numerator / denominator.clamp_min(RATIO_EPSILON)


def compute_sample_metrics(
    outputs: ContactModeOutputs,
    action_target: torch.Tensor,
    force_target: torch.Tensor,
) -> dict[str, torch.Tensor]:
    mu_prior = outputs.prior["mu_contact_prior"]
    mu_delta = mu_prior - outputs.mu_contact
    posterior_std = torch.exp(0.5 * outputs.logvar_contact)
    metrics = {
        "action_l1_zero": compute_per_sample_l1(outputs.zero["pred_action"], action_target),
        "action_l1_prior": compute_per_sample_l1(outputs.prior["pred_action"], action_target),
        "action_l1_posterior": compute_per_sample_l1(
            outputs.posterior["pred_action"],
            action_target,
        ),
        "force_l1_zero": compute_per_sample_l1(outputs.zero["pred_force"], force_target),
        "force_l1_prior": compute_per_sample_l1(outputs.prior["pred_force"], force_target),
        "force_l1_posterior": compute_per_sample_l1(outputs.posterior["pred_force"], force_target),
        "pred_action_zero_prior_mean_abs_diff": compute_per_sample_l1(
            outputs.zero["pred_action"],
            outputs.prior["pred_action"],
        ),
        "pred_action_zero_posterior_mean_abs_diff": compute_per_sample_l1(
            outputs.zero["pred_action"],
            outputs.posterior["pred_action"],
        ),
        "pred_action_prior_posterior_mean_abs_diff": compute_per_sample_l1(
            outputs.prior["pred_action"],
            outputs.posterior["pred_action"],
        ),
        "pred_force_zero_prior_mean_abs_diff": compute_per_sample_l1(
            outputs.zero["pred_force"],
            outputs.prior["pred_force"],
        ),
        "pred_force_zero_posterior_mean_abs_diff": compute_per_sample_l1(
            outputs.zero["pred_force"],
            outputs.posterior["pred_force"],
        ),
        "pred_force_prior_posterior_mean_abs_diff": compute_per_sample_l1(
            outputs.prior["pred_force"],
            outputs.posterior["pred_force"],
        ),
        "kl_contact": compute_contact_kl_per_sample(outputs.mu_contact, outputs.logvar_contact),
        "mu_contact_l2": outputs.mu_contact.norm(dim=-1),
        "posterior_std_mean": posterior_std.mean(dim=-1),
        "mu_prior_to_mu_posterior_mse": mu_delta.pow(2).mean(dim=-1),
        "mu_prior_to_mu_posterior_l2": mu_delta.norm(dim=-1),
        "mu_prior_to_mu_posterior_cosine": safe_cosine_similarity(mu_prior, outputs.mu_contact),
    }
    validate_finite_metric_tensors(metrics)
    return metrics


def validate_finite_metric_tensors(metrics: Mapping[str, torch.Tensor]) -> None:
    for name, value in metrics.items():
        if not torch.isfinite(value).all().item():
            raise ValueError(f"metric {name} contains NaN or Inf")


def batch_value(batch: Mapping[str, object], key: str, sample_index: int) -> object:
    value = batch.get(key)
    if value is None:
        return ""
    if torch.is_tensor(value):
        return value[sample_index].detach().cpu().item()
    return value[sample_index]


def episode_identifier(episode_path: object) -> str:
    if episode_path == "":
        return ""
    path = Path(str(episode_path))
    return path.parent.name or path.stem


def sample_rows_from_metrics(
    batch: Mapping[str, object],
    metrics: Mapping[str, torch.Tensor],
    global_offset: int,
) -> list[dict[str, object]]:
    batch_size = int(next(iter(metrics.values())).shape[0])
    rows: list[dict[str, object]] = []
    for sample_index in range(batch_size):
        episode_path = batch_value(batch, "episode_path", sample_index)
        row: dict[str, object] = {
            "global_sample_index": global_offset + sample_index,
            "dataset_index": global_offset + sample_index,
            "episode_path": episode_path,
            "episode_identifier": episode_identifier(episode_path),
            "state_index": batch_value(batch, "state_index", sample_index),
            "t_state": batch_value(batch, "t_state", sample_index),
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
    print("Contact-CVAE inference mode evaluation")
    print(f"checkpoint={checkpoint}")
    print("zero=deployable")
    print("prior=deployable")
    print("posterior=oracle-only")
    print(f"posterior_mode={posterior_mode}")
    print(f"num_batches={num_batches}")
    print(f"num_samples={num_samples}")
    print("")
    for key in METRIC_COLUMNS:
        print(f"{key}={aggregate[key]:.8f}")
    print(f"output_csv={output_csv if output_csv is not None else ''}")


def evaluate(args: argparse.Namespace) -> int:
    device = resolve_device(args.device)
    torch.manual_seed(args.seed)
    stats = load_normalization_stats(args.normalization_stats)
    validate_normalization_action_mode(stats, args.action_mode)
    dataset = build_evaluation_dataset(args)
    if len(dataset) == 0:
        print("error: dataset is empty for the requested settings", file=sys.stderr)
        return 1
    model = load_contact_cvae_checkpoint(args.checkpoint, args, device)
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
        tensor_batch = {key: value for key, value in device_batch.items() if torch.is_tensor(value)}
        outputs = run_contact_modes(model, tensor_batch, posterior_mode=args.posterior_mode)
        metrics = compute_sample_metrics(
            outputs,
            action_target=tensor_batch["action_chunk"],
            force_target=tensor_batch["future_force_chunk"],
        )
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
        description="Evaluate ForceAwareACT Contact-CVAE zero, prior, and oracle posterior modes."
    )
    parser.add_argument("episode_paths", type=Path, nargs="*", help="One or more HDF5 episodes.")
    parser.add_argument("--episode-list", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--normalization-stats", type=Path, required=True)
    parser.add_argument("--action-mode", choices=ACTION_MODE_CHOICES, default="joint_pos")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--chunk-len", type=int, default=10)
    parser.add_argument("--force-window-len", type=int, default=20)
    parser.add_argument("--force-window-duration", type=float, default=0.25)
    parser.add_argument("--image-size", type=int, nargs=2, default=(224, 224))
    parser.add_argument("--camera-names", nargs="+", default=("ee_cam", "base_top_cam"))
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--posterior-mode", choices=("mean", "sample"), default="mean")
    parser.add_argument("--seed", type=int, default=0)
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
        print(f"error: Contact-CVAE evaluation failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
