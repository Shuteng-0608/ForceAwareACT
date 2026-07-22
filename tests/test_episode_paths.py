from pathlib import Path

import pytest

from force_aware_act.utils import resolve_episode_paths


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    return path.resolve()


def _write_episode_list(path: Path, entry: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"# episode list\n\n{entry}\n", encoding="utf-8")
    return path


def test_resolve_absolute_episode_entry(tmp_path):
    project_root = tmp_path / "project"
    episode = _touch(tmp_path / "data" / "episode.hdf5")
    episode_list = _write_episode_list(project_root / "configs" / "episodes.txt", str(episode))

    assert resolve_episode_paths([], episode_list, project_root) == [episode]


def test_resolve_project_root_relative_episode_entry(tmp_path):
    project_root = tmp_path / "project"
    episode = _touch(project_root / "data" / "episode.hdf5")
    episode_list = _write_episode_list(
        project_root / "configs" / "episodes.txt",
        "data/episode.hdf5",
    )

    assert resolve_episode_paths([], episode_list, project_root) == [episode]


def test_resolve_list_parent_relative_episode_entry(tmp_path):
    project_root = tmp_path / "project"
    episode_list = project_root / "configs" / "splits" / "episodes.txt"
    episode = _touch(episode_list.parent / "local" / "episode.hdf5")
    _write_episode_list(episode_list, "local/episode.hdf5")

    assert resolve_episode_paths([], episode_list, project_root) == [episode]


def test_missing_episode_entry_reports_attempted_paths(tmp_path):
    project_root = tmp_path / "project"
    episode_list = _write_episode_list(
        project_root / "configs" / "episodes.txt",
        "missing/episode.hdf5",
    )

    with pytest.raises(FileNotFoundError) as error:
        resolve_episode_paths([], episode_list, project_root)

    message = str(error.value)
    assert "original=missing/episode.hdf5" in message
    assert f"attempted_project_root={project_root / 'missing/episode.hdf5'}" in message
    assert f"attempted_list_parent={episode_list.parent / 'missing/episode.hdf5'}" in message


def test_project_root_relative_entry_is_preferred(tmp_path):
    project_root = tmp_path / "project"
    episode_list = project_root / "configs" / "episodes.txt"
    project_episode = _touch(project_root / "shared" / "episode.hdf5")
    _touch(episode_list.parent / "shared" / "episode.hdf5")
    _write_episode_list(episode_list, "shared/episode.hdf5")

    assert resolve_episode_paths([], episode_list, project_root) == [project_episode]


def test_resolve_episode_paths_canonically_deduplicates_by_default(tmp_path, capsys):
    project_root = tmp_path / "project"
    episode = _touch(project_root / "data" / "episode.hdf5")
    episode_list = _write_episode_list(
        project_root / "configs" / "episodes.txt",
        "data/../data/episode.hdf5",
    )

    resolved = resolve_episode_paths([episode], episode_list, project_root)

    assert resolved == [episode]
    assert "duplicate episode path ignored" in capsys.readouterr().err


def test_resolve_episode_paths_can_preserve_duplicates_for_legacy_callers(tmp_path):
    project_root = tmp_path / "project"
    episode = _touch(project_root / "data" / "episode.hdf5")
    episode_list = _write_episode_list(
        project_root / "configs" / "episodes.txt",
        "data/episode.hdf5",
    )

    resolved = resolve_episode_paths(
        [episode],
        episode_list,
        project_root,
        deduplicate=False,
    )

    assert resolved == [episode, episode]
