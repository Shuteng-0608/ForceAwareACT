"""Shared command-line helpers for ForceAwareACT scripts."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable, Optional


def resolve_episode_paths(
    direct_paths: Optional[Iterable[Path]],
    episode_list: Optional[Path],
) -> list[Path]:
    paths: list[Path] = []
    if direct_paths is not None:
        paths.extend(path.expanduser() for path in direct_paths)

    if episode_list is not None:
        episode_list = episode_list.expanduser()
        list_parent = episode_list.parent
        with episode_list.open("r") as list_file:
            for line in list_file:
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                path = Path(stripped).expanduser()
                if not path.is_absolute():
                    path = list_parent / path
                paths.append(path)

    resolved = [path.resolve() for path in paths]
    print(f"resolved_episode_count={len(resolved)}")
    for path in resolved:
        print(f"resolved_episode={path}")
    return resolved


def validate_episode_paths(paths: Iterable[Path]) -> bool:
    ok = True
    for path in paths:
        if not path.exists():
            print(f"error: file does not exist: {path}", file=sys.stderr)
            ok = False
        elif not path.is_file():
            print(f"error: path is not a file: {path}", file=sys.stderr)
            ok = False
    return ok
