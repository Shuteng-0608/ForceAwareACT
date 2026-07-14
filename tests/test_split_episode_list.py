import pytest

from scripts.split_episode_list import read_episode_entries, split_episode_entries


def test_episode_split_is_deterministic_disjoint_and_complete():
    entries = [f"episode_{index}.hdf5" for index in range(20)]
    first = split_episode_entries(entries, train_count=12, val_count=4, seed=7)
    second = split_episode_entries(reversed(entries), train_count=12, val_count=4, seed=7)

    assert first == second
    train, val, test = first
    assert len(train) == 12
    assert len(val) == 4
    assert len(test) == 4
    assert set(train).isdisjoint(val)
    assert set(train).isdisjoint(test)
    assert set(val).isdisjoint(test)
    assert set(train) | set(val) | set(test) == set(entries)


def test_read_episode_entries_rejects_duplicates(tmp_path):
    episode_list = tmp_path / "episodes.txt"
    episode_list.write_text("a.hdf5\na.hdf5\n", encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate"):
        read_episode_entries(episode_list)
