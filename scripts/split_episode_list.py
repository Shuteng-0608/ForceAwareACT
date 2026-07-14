#!/usr/bin/env python3
"""Create deterministic, episode-level train/validation/test split files."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Iterable, Optional, Sequence


def read_episode_entries(path: Path) -> list[str]:
    entries = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    duplicates = len(entries) - len(set(entries))
    if duplicates:
        raise ValueError(f"input episode list contains {duplicates} duplicate entries")
    if not entries:
        raise ValueError("input episode list is empty")
    return entries


def split_episode_entries(
    entries: Iterable[str],
    *,
    train_count: int,
    val_count: int,
    seed: int,
) -> tuple[list[str], list[str], list[str]]:
    entries = list(entries)
    if train_count <= 0 or val_count <= 0:
        raise ValueError("train_count and val_count must be positive")
    if train_count + val_count >= len(entries):
        raise ValueError("train_count + val_count must leave at least one test episode")
    shuffled = sorted(entries)
    random.Random(seed).shuffle(shuffled)
    train = sorted(shuffled[:train_count])
    val = sorted(shuffled[train_count : train_count + val_count])
    test = sorted(shuffled[train_count + val_count :])
    return train, val, test


def write_split(path: Path, entries: Sequence[str], *, split: str, seed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = f"# split={split} seed={seed} episode_count={len(entries)}"
    path.write_text(header + "\n" + "\n".join(entries) + "\n", encoding="utf-8")


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--train-output", type=Path, required=True)
    parser.add_argument("--val-output", type=Path, required=True)
    parser.add_argument("--test-output", type=Path, required=True)
    parser.add_argument("--train-count", type=int, required=True)
    parser.add_argument("--val-count", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if not args.input.is_file():
        print(f"error: input episode list does not exist: {args.input}", file=sys.stderr)
        return 2
    try:
        entries = read_episode_entries(args.input)
        train, val, test = split_episode_entries(
            entries,
            train_count=args.train_count,
            val_count=args.val_count,
            seed=args.seed,
        )
        write_split(args.train_output, train, split="train", seed=args.seed)
        write_split(args.val_output, val, split="validation", seed=args.seed)
        write_split(args.test_output, test, split="test", seed=args.seed)
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(f"train_count={len(train)} train_output={args.train_output}")
    print(f"val_count={len(val)} val_output={args.val_output}")
    print(f"test_count={len(test)} test_output={args.test_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
