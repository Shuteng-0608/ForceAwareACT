"""Shared episode-list path resolution utilities."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable, Optional


def _existing_file(path: Path, description: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"{description} does not exist: {resolved}")
    if not resolved.is_file():
        raise FileNotFoundError(f"{description} is not a file: {resolved}")
    return resolved


def resolve_episode_entry(entry: str | Path, project_root: Path, episode_list_parent: Path) -> Path:
    """Resolve one list entry, preferring project-root-relative paths."""

    original = Path(entry).expanduser()
    project_root = project_root.expanduser().resolve()
    episode_list_parent = episode_list_parent.expanduser().resolve()

    if original.is_absolute():
        if original.exists() and original.is_file():
            return original.resolve()
        project_path = original.resolve()
        list_parent_path = original.resolve()
    else:
        project_path = (project_root / original).resolve()
        list_parent_path = (episode_list_parent / original).resolve()
    if project_path.exists() and project_path.is_file():
        return project_path
    if list_parent_path.exists() and list_parent_path.is_file():
        return list_parent_path
    raise FileNotFoundError(
        "episode entry not found: "
        f"original={entry!s}; "
        f"attempted_project_root={project_path}; "
        f"attempted_list_parent={list_parent_path}"
    )


def resolve_episode_paths(
    direct_paths: Optional[Iterable[Path]],
    episode_list: Optional[Path],
    project_root: Optional[Path] = None,
) -> list[Path]:
    """Resolve direct episode paths and entries from an optional episode list."""

    root = (Path.cwd() if project_root is None else project_root).expanduser().resolve()
    resolved: list[Path] = []

    if direct_paths is not None:
        for path in direct_paths:
            candidate = path.expanduser()
            if not candidate.is_absolute():
                candidate = root / candidate
            resolved.append(_existing_file(candidate, "episode path"))

    if episode_list is not None:
        list_candidate = episode_list.expanduser()
        if not list_candidate.is_absolute():
            list_candidate = root / list_candidate
        resolved_list = _existing_file(list_candidate, "episode list")
        with resolved_list.open("r", encoding="utf-8") as list_file:
            for line in list_file:
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                resolved.append(resolve_episode_entry(stripped, root, resolved_list.parent))

    print(f"resolved_episode_count={len(resolved)}")
    for path in resolved:
        print(f"resolved_episode={path}")
    return resolved


def validate_episode_paths(paths: Iterable[Path]) -> bool:
    """Compatibility validator for already-resolved episode paths."""

    ok = True
    for path in paths:
        if not path.exists():
            print(f"error: file does not exist: {path}", file=sys.stderr)
            ok = False
        elif not path.is_file():
            print(f"error: path is not a file: {path}", file=sys.stderr)
            ok = False
    return ok
