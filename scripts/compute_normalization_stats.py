#!/usr/bin/env python3
"""Compute ForceAwareACT dataset normalization statistics.

Example:
    PYTHONPATH=src .venv/bin/python scripts/compute_normalization_stats.py test_data/episode.hdf5 --output outputs/normalization_stats.pt
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from force_aware_act.data import ContactForceHDF5Dataset, compute_normalization_stats  # noqa: E402
from force_aware_act.data.normalization import (  # noqa: E402
    compute_balanced_normalization_stats,
)
from force_aware_act.utils import resolve_episode_paths, validate_episode_paths  # noqa: E402


ACTION_MODE_CHOICES = (
    "joint_pos",
    "action",
    "joint_pos_command",
    "delta_joint_cmd",
    "delta_joint_pos_command",
)


def _save_stats_atomic(
    stats: dict,
    output: Path,
    *,
    allow_overwrite: bool,
) -> None:
    """Durably publish one stats artifact without exposing a partial file."""

    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=str(output.parent),
        prefix=f".{output.name}.",
        suffix=".tmp",
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        torch.save(stats, temporary_path)
        with temporary_path.open("rb") as handle:
            os.fsync(handle.fileno())
        if allow_overwrite:
            os.replace(temporary_path, output)
        else:
            try:
                os.link(temporary_path, output)
            except FileExistsError as error:
                raise FileExistsError(
                    f"refusing to overwrite existing normalization stats: {output}"
                ) from error
        try:
            directory_fd = os.open(str(output.parent), os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            pass
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def compute_and_save(args: argparse.Namespace) -> int:
    estimator = getattr(args, "estimator", "legacy_chunked")
    if (
        estimator == "balanced_raw"
        and args.output.exists()
        and not getattr(args, "overwrite_existing", False)
    ):
        raise FileExistsError(
            "balanced normalization refuses to overwrite an existing artifact: "
            f"{args.output}; pass --overwrite-existing only after preserving its hash"
        )
    domain_episode_paths = getattr(args, "domain_episode_paths", None)
    if domain_episode_paths is None:
        domain_episode_paths = {"default": tuple(args.episode_paths)}

    if estimator == "legacy_chunked":
        dataset = ContactForceHDF5Dataset(
            args.episode_paths,
            action_mode=args.action_mode,
            chunk_len=args.chunk_len,
            force_window_len=args.force_window_len,
            force_window_duration=args.force_window_duration,
            image_size=tuple(args.image_size),
            camera_names=tuple(args.camera_names),
            imagenet_normalize=args.imagenet_normalize,
        )
        print(f"dataset_length={len(dataset)}")
        stats = compute_normalization_stats(
            dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            eps=args.eps,
        )
        stats["normalization_method"] = "legacy_overlapping_act_chunks"
    elif estimator == "balanced_raw":
        strict_lengths = bool(getattr(args, "strict_lengths", False))
        stats = compute_balanced_normalization_stats(
            domain_episode_paths,
            action_mode=args.action_mode,
            domain_weights=getattr(args, "domain_weights", None),
            eps=args.eps,
            tolerate_length_mismatch=not strict_lengths,
            max_length_mismatch=(
                0 if strict_lengths else getattr(args, "max_length_mismatch", 1)
            ),
            read_chunk_size=getattr(args, "read_chunk_size", 65536),
        )
        print(f"dataset_episode_count={sum(len(paths) for paths in domain_episode_paths.values())}")
    else:
        raise ValueError(f"unsupported normalization estimator: {estimator!r}")

    flattened_paths = stats.get(
        "population_paths",
        [
            str(Path(path).expanduser().resolve())
            for domain_name in sorted(domain_episode_paths)
            for path in domain_episode_paths[domain_name]
        ],
    )
    stats.update(
        {
            "action_mode": args.action_mode,
            "chunk_len": args.chunk_len,
            "force_window_len": args.force_window_len,
            "force_window_duration": args.force_window_duration,
            "camera_names": tuple(args.camera_names),
            "image_size": tuple(args.image_size),
            "imagenet_normalize": bool(args.imagenet_normalize),
            "episode_paths": flattened_paths,
            "episode_list": str(args.episode_list) if args.episode_list is not None else None,
            "domain_episode_lists": {
                name: str(path)
                for name, path in getattr(args, "domain_episode_lists", {}).items()
            },
            "normalization_estimator": estimator,
        }
    )

    _save_stats_atomic(
        stats,
        args.output,
        allow_overwrite=(
            estimator == "legacy_chunked"
            or getattr(args, "overwrite_existing", False)
        ),
    )
    print(f"saved_stats={args.output}")
    print(f"action_mode={args.action_mode}")
    for key, value in stats.items():
        if torch.is_tensor(value):
            print(f"{key}: shape={tuple(value.shape)}")
        else:
            print(f"{key}: {value}")
    return 0


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute qpos/action/force normalization stats from HDF5 episodes.",
    )
    parser.add_argument("episode_paths", type=Path, nargs="*", help="One or more HDF5 episodes.")
    parser.add_argument("--episode-list", type=Path, default=None)
    parser.add_argument(
        "--domain",
        dest="domain_specs",
        action="append",
        type=_parse_domain_spec,
        default=[],
        metavar="NAME=EPISODE_LIST",
        help=(
            "Balanced raw-stream domain and its training episode list. Repeat once per "
            "domain; cannot be mixed with positional paths or --episode-list."
        ),
    )
    parser.add_argument(
        "--domain-weight",
        dest="domain_weight_specs",
        action="append",
        type=_parse_domain_weight_spec,
        default=[],
        metavar="NAME=WEIGHT",
        help="Optional positive domain mixture weight; specify every domain if used.",
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/normalization_stats.pt"))
    parser.add_argument(
        "--overwrite-existing",
        action="store_true",
        help=(
            "Allow replacement of an existing stats file. Balanced formal runs "
            "refuse replacement by default; preserve the old artifact and hash first."
        ),
    )
    parser.add_argument("--action-mode", choices=ACTION_MODE_CHOICES, default="joint_pos")
    parser.add_argument(
        "--estimator",
        choices=("auto", "balanced_raw", "legacy_chunked"),
        default="auto",
        help=(
            "auto preserves the historical estimator for legacy positional/list input "
            "and selects balanced_raw for --domain input."
        ),
    )
    parser.add_argument("--chunk-len", type=int, default=50)
    parser.add_argument("--force-window-len", type=int, default=50)
    parser.add_argument("--force-window-duration", type=float, default=0.25)
    parser.add_argument("--image-size", type=int, nargs=2, default=(224, 224))
    parser.add_argument("--camera-names", nargs="+", default=("ee_cam", "base_top_cam"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--eps", type=float, default=1.0e-6)
    parser.add_argument("--imagenet-normalize", action="store_true")
    parser.add_argument("--strict-lengths", action="store_true")
    parser.add_argument("--max-length-mismatch", type=int, default=1)
    parser.add_argument("--read-chunk-size", type=int, default=65536)
    return parser.parse_args(argv)


def _parse_domain_spec(value: str) -> Tuple[str, Path]:
    name, separator, raw_path = value.partition("=")
    if not separator or not name or name != name.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError("--domain must use NAME=EPISODE_LIST")
    return name, Path(raw_path.strip())


def _parse_domain_weight_spec(value: str) -> Tuple[str, float]:
    name, separator, raw_weight = value.partition("=")
    if not separator or not name or name != name.strip() or not raw_weight.strip():
        raise argparse.ArgumentTypeError("--domain-weight must use NAME=WEIGHT")
    try:
        weight = float(raw_weight)
    except ValueError as error:
        raise argparse.ArgumentTypeError("--domain-weight WEIGHT must be numeric") from error
    return name, weight


def _resolve_domain_inputs(args: argparse.Namespace) -> Optional[str]:
    """Resolve new domain-list inputs, returning an error string on misuse."""

    if args.domain_specs:
        if args.episode_paths or args.episode_list is not None:
            return "--domain cannot be mixed with positional episode paths or --episode-list"
        if args.estimator == "auto":
            args.estimator = "balanced_raw"
        if args.estimator != "balanced_raw":
            return "--domain requires --estimator balanced_raw"
        domains: Dict[str, Tuple[Path, ...]] = {}
        domain_lists: Dict[str, Path] = {}
        for name, episode_list in args.domain_specs:
            if name in domains:
                return f"duplicate --domain name: {name}"
            resolved_paths = resolve_episode_paths(
                [],
                episode_list,
                project_root=REPO_ROOT,
            )
            if not resolved_paths:
                return f"--domain {name} episode list is empty: {episode_list}"
            domains[name] = tuple(resolved_paths)
            list_path = episode_list.expanduser()
            if not list_path.is_absolute():
                list_path = REPO_ROOT / list_path
            domain_lists[name] = list_path.resolve()
        args.domain_episode_paths = domains
        args.domain_episode_lists = domain_lists
        args.episode_paths = [path for paths in domains.values() for path in paths]
    else:
        if args.estimator == "auto":
            args.estimator = "legacy_chunked"
        args.episode_paths = resolve_episode_paths(
            args.episode_paths,
            args.episode_list,
            project_root=REPO_ROOT,
        )
        args.domain_episode_paths = {"default": tuple(args.episode_paths)}
        if args.episode_list is None:
            args.domain_episode_lists = {}
        else:
            list_path = args.episode_list.expanduser()
            if not list_path.is_absolute():
                list_path = REPO_ROOT / list_path
            args.domain_episode_lists = {"default": list_path.resolve()}

    if args.domain_weight_specs:
        if args.estimator != "balanced_raw":
            return "--domain-weight requires --estimator balanced_raw"
        weights: Dict[str, float] = {}
        for name, weight in args.domain_weight_specs:
            if name in weights:
                return f"duplicate --domain-weight name: {name}"
            if not math.isfinite(weight) or weight <= 0:
                return f"--domain-weight for {name} must be positive and finite"
            weights[name] = weight
        domain_names = set(args.domain_episode_paths)
        if set(weights) != domain_names:
            return (
                "--domain-weight names must exactly match domains: "
                f"domains={sorted(domain_names)}; weights={sorted(weights)}"
            )
        args.domain_weights = weights
    else:
        args.domain_weights = None
    return None


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    input_error = _resolve_domain_inputs(args)
    if input_error is not None:
        print(f"error: {input_error}", file=sys.stderr)
        return 2
    args.output = args.output.expanduser()
    if not args.episode_paths:
        print("error: provide episode paths or --episode-list", file=sys.stderr)
        return 2
    if not validate_episode_paths(args.episode_paths):
        return 2
    if args.chunk_len <= 0:
        print("error: --chunk-len must be positive", file=sys.stderr)
        return 2
    if args.force_window_len <= 0:
        print("error: --force-window-len must be positive", file=sys.stderr)
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
    if not math.isfinite(args.eps) or args.eps <= 0:
        print("error: --eps must be positive and finite", file=sys.stderr)
        return 2
    if args.max_length_mismatch < 0:
        print("error: --max-length-mismatch must be non-negative", file=sys.stderr)
        return 2
    if args.read_chunk_size <= 0:
        print("error: --read-chunk-size must be positive", file=sys.stderr)
        return 2

    try:
        return compute_and_save(args)
    except Exception as error:
        print(f"error: failed to compute normalization stats: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
