#!/usr/bin/env python3
"""Train the structurally force-free ACT Motion-CVAE baseline."""

from __future__ import annotations

import argparse
import csv
import sys
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from force_aware_act.data import ContactForceHDF5Dataset, normalize_tensor  # noqa: E402
from force_aware_act.models import ACTPolicyBaseline  # noqa: E402
from force_aware_act.training import (  # noqa: E402
    EARLY_STOP_METRICS,
    EarlyStoppingState,
    compute_act_baseline_loss,
    compute_steps_per_epoch,
    evaluate_deployment_metrics,
    linear_warmup,
    validate_disjoint_episode_splits,
    validate_normalization_training_episodes,
)
from force_aware_act.utils import resolve_episode_paths, validate_episode_paths  # noqa: E402
from scripts.train_minimal import (  # noqa: E402
    build_checkpoint_payload,
    checkpoint_step_path,
    positive_int,
    resolve_checkpoint_steps,
    save_checkpoint_atomic,
)


ACTION_MODE_CHOICES = (
    "joint_pos",
    "action",
    "joint_pos_command",
    "delta_joint_cmd",
    "delta_joint_pos_command",
)


def _move_batch_to_device(batch: Dict[str, object], device: torch.device) -> Dict[str, object]:
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def _cycle_batches(dataloader: DataLoader) -> Iterable[Dict[str, object]]:
    while True:
        for batch in dataloader:
            yield batch


def _load_normalization_stats(stats_path: Optional[Path]) -> Optional[Dict[str, Any]]:
    if stats_path is None:
        return None
    stats = torch.load(stats_path, map_location="cpu")
    if not isinstance(stats, dict):
        raise ValueError("normalization stats file must contain a dict")
    for key in ("qpos_mean", "qpos_std", "action_mean", "action_std"):
        if key not in stats:
            raise KeyError(f"normalization stats missing required key: {key}")
        if not torch.is_tensor(stats[key]):
            raise ValueError(f"normalization stats {key!r} must be a torch.Tensor")
    return stats


def _validate_normalization_action_mode(
    stats: Optional[Dict[str, Any]],
    action_mode: str,
) -> None:
    if stats is None or "action_mode" not in stats:
        return
    stats_action_mode = stats["action_mode"]
    if stats_action_mode != action_mode:
        raise ValueError(
            "normalization stats action_mode mismatch: "
            f"stats action_mode={stats_action_mode!r}, requested action_mode={action_mode!r}. "
            "Recompute normalization stats for the requested action_mode."
        )


def _normalize_batch(batch: Dict[str, object], stats: Dict[str, Any]) -> Dict[str, object]:
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


def _model_config_from_args(args: argparse.Namespace) -> Dict[str, object]:
    return {
        "pretrained_resnet18": False,
        "freeze_resnet18": False,
        "d_model": args.d_model,
        "z_dim": args.z_dim,
        "q_dim": 7,
        "action_dim": 7,
        "chunk_len": args.chunk_len,
        "nhead": args.nhead,
        "num_encoder_layers": args.num_encoder_layers,
        "num_decoder_layers": args.num_decoder_layers,
        "dim_feedforward": args.dim_feedforward,
        "dropout": args.dropout,
    }


def _config_from_args(args: argparse.Namespace) -> Dict[str, object]:
    return {
        "policy_variant": "act_baseline",
        "act_baseline_version": ACTPolicyBaseline.act_baseline_version,
        "uses_force": False,
        "uses_contact_latent": False,
        "motion_latent_mode": "posterior_train_zero_deploy",
        "train_latent_mode": "posterior",
        "episode_paths": [str(path) for path in args.episode_paths],
        "action_mode": args.action_mode,
        "chunk_len": args.chunk_len,
        "image_size": tuple(args.image_size),
        "camera_names": tuple(args.camera_names),
        "imagenet_normalize": args.imagenet_normalize,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "max_steps": args.max_steps,
        "max_epochs": getattr(args, "max_epochs", None),
        "steps_per_epoch": getattr(args, "steps_per_epoch", None),
        "effective_max_steps": getattr(args, "effective_max_steps", args.max_steps),
        "val_episode_paths": [str(path) for path in getattr(args, "val_episode_paths", ())],
        "val_every_epochs": getattr(args, "val_every_epochs", 1),
        "early_stop_patience": getattr(args, "early_stop_patience", 8),
        "early_stop_min_epochs": getattr(args, "early_stop_min_epochs", 10),
        "early_stop_min_delta": getattr(args, "early_stop_min_delta", 0.005),
        "early_stop_metric": getattr(args, "early_stop_metric", "deploy_loss"),
        "validation_deployment_mode": "zero",
        "learning_rate": args.learning_rate,
        "optimizer": "AdamW",
        "beta_motion_max": args.beta_motion_max,
        "warmup_steps": args.warmup_steps,
        "save_every": getattr(args, "save_every", 0),
        "save_steps": tuple(getattr(args, "save_steps", ())),
        "intermediate_checkpoint_steps": tuple(
            getattr(args, "intermediate_checkpoint_steps", ())
        ),
        "output_dir": str(args.output_dir),
        "log_csv": str(args.log_csv),
        "validation_log": str(getattr(args, "validation_log", "")),
        "device": args.device,
        "normalization_stats_path": (
            str(args.normalization_stats) if args.normalization_stats is not None else None
        ),
        "model": _model_config_from_args(args),
    }


def _build_training_dataset(
    args: argparse.Namespace,
    episode_paths: Optional[Sequence[Path]] = None,
) -> ContactForceHDF5Dataset:
    return ContactForceHDF5Dataset(
        args.episode_paths if episode_paths is None else episode_paths,
        camera_names=tuple(args.camera_names),
        action_mode=args.action_mode,
        chunk_len=args.chunk_len,
        force_window_len=1,
        force_window_duration=0.0,
        image_size=tuple(args.image_size),
        imagenet_normalize=args.imagenet_normalize,
        include_force=False,
    )


def train(args: argparse.Namespace) -> int:
    intermediate_checkpoint_steps = resolve_checkpoint_steps(
        max_steps=args.max_steps,
        save_every=getattr(args, "save_every", 0),
        save_steps=getattr(args, "save_steps", ()),
    )
    args.intermediate_checkpoint_steps = intermediate_checkpoint_steps
    intermediate_checkpoint_step_set = set(intermediate_checkpoint_steps)
    device = torch.device(args.device)
    normalization_stats = _load_normalization_stats(args.normalization_stats)
    _validate_normalization_action_mode(normalization_stats, args.action_mode)
    val_episode_paths = list(getattr(args, "val_episode_paths", ()))
    if val_episode_paths:
        validate_disjoint_episode_splits(args.episode_paths, val_episode_paths)
        validate_normalization_training_episodes(normalization_stats, args.episode_paths)
    dataset = _build_training_dataset(args)
    if len(dataset) == 0:
        print("error: dataset is empty for the requested settings", file=sys.stderr)
        return 1

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    steps_per_epoch = compute_steps_per_epoch(len(dataset), args.batch_size)
    max_epochs = getattr(args, "max_epochs", None)
    effective_max_steps = args.max_steps
    if max_epochs is not None:
        effective_max_steps = min(effective_max_steps, max_epochs * steps_per_epoch)
    args.steps_per_epoch = steps_per_epoch
    args.effective_max_steps = effective_max_steps

    val_dataloader = None
    early_stopping = None
    if val_episode_paths:
        val_dataset = _build_training_dataset(args, val_episode_paths)
        if len(val_dataset) == 0:
            print("error: validation dataset is empty for the requested settings", file=sys.stderr)
            return 1
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )
        early_stopping = EarlyStoppingState(
            patience=getattr(args, "early_stop_patience", 8),
            min_epochs=getattr(args, "early_stop_min_epochs", 10),
            min_delta=getattr(args, "early_stop_min_delta", 0.005),
        )
    config = _config_from_args(args)

    model = ACTPolicyBaseline(**_model_config_from_args(args)).to(device)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    model.train()

    print(f"dataset_length={len(dataset)}")
    print(f"steps_per_epoch={steps_per_epoch}")
    print(f"effective_max_steps={effective_max_steps}")
    print(f"policy_variant=act_baseline")
    print(f"uses_force=False")
    print(f"uses_contact_latent=False")
    print(f"act_baseline_version={ACTPolicyBaseline.act_baseline_version}")
    print(f"motion_latent_mode=posterior_train_zero_deploy")
    print(f"train_latent_mode=posterior")
    print(f"action_mode={args.action_mode}")
    print(f"checkpoint_path={args.output_dir / 'checkpoint.pt'}")
    print(f"intermediate_checkpoint_steps={intermediate_checkpoint_steps}")
    print(f"log_csv={args.log_csv}")
    if val_dataloader is not None:
        print(f"validation_dataset_length={len(val_dataloader.dataset)}")
        print("validation_deployment_mode=zero")
        print(f"early_stop_metric={getattr(args, 'early_stop_metric', 'deploy_loss')}")
        print(f"validation_log={args.validation_log}")
    print(f"normalization_enabled={normalization_stats is not None}")
    if normalization_stats is not None:
        print(f"normalization_stats_path={args.normalization_stats}")

    batch_iter = _cycle_batches(dataloader)
    last_step = 0
    last_epoch = 0
    last_step_in_epoch = 0
    stop_reason = "max_steps"
    if max_epochs is not None and effective_max_steps == max_epochs * steps_per_epoch:
        stop_reason = "max_epochs"
    args.log_csv.parent.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        log_file = stack.enter_context(args.log_csv.open("w", newline=""))
        log_writer = csv.DictWriter(
            log_file,
            fieldnames=[
                "step",
                "epoch",
                "batch_in_epoch",
                "loss_total",
                "loss_action",
                "kl_motion",
                "beta_motion",
                "policy_variant",
                "uses_force",
                "uses_contact_latent",
                "motion_latent_mode",
                "train_latent_mode",
                "uses_posterior_latent",
                "uses_zero_latent",
                "normalization_enabled",
            ],
        )
        log_writer.writeheader()
        validation_writer = None
        if val_dataloader is not None:
            args.validation_log.parent.mkdir(parents=True, exist_ok=True)
            validation_file = stack.enter_context(args.validation_log.open("w", newline=""))
            validation_writer = csv.DictWriter(
                validation_file,
                fieldnames=[
                    "epoch",
                    "step",
                    "deployment_mode",
                    "deploy_loss",
                    "action_l1",
                    "monitored_metric",
                    "improved",
                    "epochs_without_improvement",
                    "best_metric",
                    "best_epoch",
                    "best_step",
                ],
            )
            validation_writer.writeheader()
        for step in range(1, effective_max_steps + 1):
            last_step = step
            epoch = (step - 1) // steps_per_epoch + 1
            step_in_epoch = (step - 1) % steps_per_epoch + 1
            last_epoch = epoch
            last_step_in_epoch = step_in_epoch
            batch = _move_batch_to_device(next(batch_iter), device)
            if normalization_stats is not None:
                batch = _normalize_batch(batch, normalization_stats)

            beta_motion = linear_warmup(step, args.warmup_steps, args.beta_motion_max)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(
                images=batch["images"],
                qpos=batch["qpos"],
                action_chunk=batch["action_chunk"],
                is_training=True,
            )
            losses = compute_act_baseline_loss(
                outputs=outputs,
                action_chunk=batch["action_chunk"],
                beta_motion=beta_motion,
            )
            losses["loss_total"].backward()
            optimizer.step()

            if step in intermediate_checkpoint_step_set:
                checkpoint_path = checkpoint_step_path(args.output_dir, step)
                checkpoint = build_checkpoint_payload(
                    model=model,
                    optimizer=optimizer,
                    config=config,
                    step=step,
                    epoch=epoch,
                    step_in_epoch=step_in_epoch,
                    **(early_stopping.checkpoint_metadata() if early_stopping else {}),
                )
                save_checkpoint_atomic(checkpoint, checkpoint_path)
                print(f"saved intermediate checkpoint: {checkpoint_path}", flush=True)

            log_writer.writerow(
                {
                    "step": step,
                    "epoch": epoch,
                    "batch_in_epoch": step_in_epoch,
                    "loss_total": losses["loss_total"].item(),
                    "loss_action": losses["loss_action"].item(),
                    "kl_motion": losses["kl_motion"].item(),
                    "beta_motion": beta_motion,
                    "policy_variant": "act_baseline",
                    "uses_force": False,
                    "uses_contact_latent": False,
                    "motion_latent_mode": "posterior_train_zero_deploy",
                    "train_latent_mode": "posterior",
                    "uses_posterior_latent": True,
                    "uses_zero_latent": False,
                    "normalization_enabled": normalization_stats is not None,
                }
            )
            print(
                " ".join(
                    [
                        f"step={step}",
                        f"epoch={epoch}",
                        f"loss_total={losses['loss_total'].item():.6g}",
                        f"loss_action={losses['loss_action'].item():.6g}",
                        f"kl_motion={losses['kl_motion'].item():.6g}",
                        f"beta_motion={beta_motion:.6g}",
                        "policy_variant=act_baseline",
                        "train_latent_mode=posterior",
                    ]
                ),
                flush=True,
            )

            epoch_finished = step_in_epoch == steps_per_epoch
            should_validate = val_dataloader is not None and (
                (epoch_finished and epoch % getattr(args, "val_every_epochs", 1) == 0)
                or step == effective_max_steps
            )
            if should_validate:
                metrics = evaluate_deployment_metrics(
                    model=model,
                    dataloader=val_dataloader,
                    device=device,
                    policy_variant="act_baseline",
                    deployment_mode="zero",
                    normalization_stats=normalization_stats,
                    lambda_force=0.0,
                )
                monitored_name = getattr(args, "early_stop_metric", "deploy_loss")
                if monitored_name == "force_l1":
                    raise ValueError("ACT baseline validation does not provide force_l1")
                monitored_metric = metrics[monitored_name]
                improved, should_stop = early_stopping.update(
                    monitored_metric,
                    epoch=epoch,
                    step=step,
                )
                validation_writer.writerow(
                    {
                        "epoch": epoch,
                        "step": step,
                        "deployment_mode": "zero",
                        "deploy_loss": metrics["deploy_loss"],
                        "action_l1": metrics["action_l1"],
                        "monitored_metric": monitored_metric,
                        "improved": improved,
                        **early_stopping.checkpoint_metadata(),
                    }
                )
                validation_file.flush()
                print(
                    f"validation epoch={epoch} step={step} mode=zero "
                    f"deploy_loss={metrics['deploy_loss']:.6g} "
                    f"action_l1={metrics['action_l1']:.6g} improved={improved} "
                    "epochs_without_improvement="
                    f"{early_stopping.epochs_without_improvement}",
                    flush=True,
                )
                if improved:
                    best_path = args.output_dir / "checkpoint_best.pt"
                    best_checkpoint = build_checkpoint_payload(
                        model=model,
                        optimizer=optimizer,
                        config=config,
                        step=step,
                        epoch=epoch,
                        step_in_epoch=step_in_epoch,
                        stop_reason="best_validation_metric",
                        **early_stopping.checkpoint_metadata(),
                    )
                    save_checkpoint_atomic(best_checkpoint, best_path)
                    print(f"saved best checkpoint: {best_path}", flush=True)
                if should_stop:
                    stop_reason = "early_stopping"
                    print(
                        f"early stopping at epoch={epoch} step={step} "
                        f"best_epoch={early_stopping.best_epoch} "
                        f"best_step={early_stopping.best_step}",
                        flush=True,
                    )
                    break

    checkpoint_path = args.output_dir / "checkpoint.pt"
    checkpoint = build_checkpoint_payload(
        model=model,
        optimizer=optimizer,
        config=config,
        step=last_step,
        epoch=last_epoch,
        step_in_epoch=last_step_in_epoch,
        stop_reason=stop_reason,
        **(early_stopping.checkpoint_metadata() if early_stopping else {}),
    )
    save_checkpoint_atomic(checkpoint, checkpoint_path)
    print(f"saved final checkpoint: {checkpoint_path}")
    return 0


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ACT Motion-CVAE baseline training loop.")
    parser.add_argument("episode_paths", type=Path, nargs="*", help="One or more HDF5 episodes.")
    parser.add_argument("--episode-list", type=Path, default=None)
    parser.add_argument(
        "--val-episode-list",
        type=Path,
        default=None,
        help="Episode-level validation split; enables validation and early stopping.",
    )
    parser.add_argument("--action-mode", choices=ACTION_MODE_CHOICES, default="joint_pos")
    parser.add_argument("--chunk-len", type=int, default=10)
    parser.add_argument("--image-size", type=int, nargs=2, default=(224, 224))
    parser.add_argument("--camera-names", nargs="+", default=("ee_cam", "base_top_cam"))
    parser.add_argument("--imagenet-normalize", action="store_true")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--max-epochs", type=positive_int, default=None)
    parser.add_argument("--val-every-epochs", type=positive_int, default=1)
    parser.add_argument("--early-stop-patience", type=positive_int, default=8)
    parser.add_argument("--early-stop-min-epochs", type=int, default=10)
    parser.add_argument("--early-stop-min-delta", type=float, default=0.005)
    parser.add_argument("--early-stop-metric", choices=EARLY_STOP_METRICS, default="deploy_loss")
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--beta-motion-max", type=float, default=1.0e-4)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument("--save-steps", type=int, nargs="*", default=[])
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--z-dim", type=int, default=16)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num-encoder-layers", type=int, default=1)
    parser.add_argument("--num-decoder-layers", type=int, default=1)
    parser.add_argument("--dim-feedforward", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/act_baseline_train"))
    parser.add_argument(
        "--log-csv",
        type=Path,
        default=Path("outputs/act_baseline_train/train_log.csv"),
    )
    parser.add_argument("--validation-log", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--normalization-stats", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    args.episode_paths = resolve_episode_paths(
        args.episode_paths, args.episode_list, project_root=REPO_ROOT
    )
    args.val_episode_paths = resolve_episode_paths(
        [], args.val_episode_list, project_root=REPO_ROOT
    )
    if args.normalization_stats is not None:
        args.normalization_stats = args.normalization_stats.expanduser()
    args.log_csv = args.log_csv.expanduser()
    args.output_dir = args.output_dir.expanduser()
    if args.validation_log is None:
        args.validation_log = args.output_dir / "validation_log.csv"
    else:
        args.validation_log = args.validation_log.expanduser()
    if not args.episode_paths:
        print("error: provide episode paths or --episode-list", file=sys.stderr)
        return 2
    if not validate_episode_paths(args.episode_paths):
        return 2
    if args.chunk_len <= 0:
        print("error: --chunk-len must be positive", file=sys.stderr)
        return 2
    if args.batch_size <= 0:
        print("error: --batch-size must be positive", file=sys.stderr)
        return 2
    if len(args.image_size) != 2 or args.image_size[0] <= 0 or args.image_size[1] <= 0:
        print("error: --image-size must be two positive integers", file=sys.stderr)
        return 2
    if not args.camera_names:
        print("error: --camera-names must include at least one camera", file=sys.stderr)
        return 2
    if args.max_steps <= 0:
        print("error: --max-steps must be positive", file=sys.stderr)
        return 2
    if args.early_stop_min_epochs < 0:
        print("error: --early-stop-min-epochs must be non-negative", file=sys.stderr)
        return 2
    if not 0.0 <= args.early_stop_min_delta < 1.0:
        print("error: --early-stop-min-delta must be in [0, 1)", file=sys.stderr)
        return 2
    if args.early_stop_metric == "force_l1":
        print("error: ACT baseline validation does not provide force_l1", file=sys.stderr)
        return 2
    try:
        resolve_checkpoint_steps(
            max_steps=args.max_steps,
            save_every=args.save_every,
            save_steps=args.save_steps,
        )
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    if args.beta_motion_max < 0:
        print("error: --beta-motion-max must be non-negative", file=sys.stderr)
        return 2
    if args.warmup_steps < 0:
        print("error: --warmup-steps must be non-negative", file=sys.stderr)
        return 2
    if args.d_model <= 0 or args.z_dim <= 0 or args.nhead <= 0:
        print("error: model dimensions must be positive", file=sys.stderr)
        return 2
    if args.d_model % args.nhead != 0:
        print("error: --d-model must be divisible by --nhead", file=sys.stderr)
        return 2
    if args.num_encoder_layers <= 0 or args.num_decoder_layers <= 0:
        print("error: transformer layer counts must be positive", file=sys.stderr)
        return 2
    if args.dim_feedforward <= 0:
        print("error: --dim-feedforward must be positive", file=sys.stderr)
        return 2
    if args.dropout < 0:
        print("error: --dropout must be non-negative", file=sys.stderr)
        return 2
    if args.normalization_stats is not None and not args.normalization_stats.is_file():
        print(
            f"error: normalization stats file does not exist: {args.normalization_stats}",
            file=sys.stderr,
        )
        return 2

    try:
        return train(args)
    except Exception as error:
        print(f"error: ACT baseline training failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
