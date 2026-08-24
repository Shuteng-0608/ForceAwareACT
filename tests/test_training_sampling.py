from dataclasses import dataclass

import pytest

from force_aware_act.training import (
    DENSE_WINDOW_SAMPLING,
    EPISODIC_UNIFORM_SAMPLING,
    EpisodicUniformWindowDataset,
    sample_uniform_episode_timestep,
    validate_train_sampling_mode,
)
from scripts.train_act_aligned_contact_cvae import (
    _episodic_validation_interval,
)


@dataclass(frozen=True)
class _Episode:
    num_steps: int


class _DenseDataset:
    def __init__(self, length: int) -> None:
        self.values = tuple(range(length))
        self.closed = False

    def __len__(self) -> int:
        return len(self.values)

    def __getitem__(self, index: int) -> int:
        return self.values[index]

    def close(self) -> None:
        self.closed = True


def test_sampling_modes_are_explicitly_validated():
    assert validate_train_sampling_mode(EPISODIC_UNIFORM_SAMPLING) == (
        EPISODIC_UNIFORM_SAMPLING
    )
    assert validate_train_sampling_mode(DENSE_WINDOW_SAMPLING) == (
        DENSE_WINDOW_SAMPLING
    )
    with pytest.raises(ValueError, match="sampling mode"):
        validate_train_sampling_mode("random")


def test_uniform_timestep_is_keyed_by_epoch_and_episode():
    first = sample_uniform_episode_timestep(
        sampling_seed=7,
        stream_id=2,
        epoch=3,
        episode_index=4,
        episode_steps=101,
    )
    repeated = sample_uniform_episode_timestep(
        sampling_seed=7,
        stream_id=2,
        epoch=3,
        episode_index=4,
        episode_steps=101,
    )
    changed_epoch = sample_uniform_episode_timestep(
        sampling_seed=7,
        stream_id=2,
        epoch=4,
        episode_index=4,
        episode_steps=101,
    )

    assert first == repeated
    assert first != changed_epoch
    assert 0 <= first < 101


def test_episodic_view_maps_each_episode_into_its_dense_range():
    dense = _DenseDataset(12)
    dataset = EpisodicUniformWindowDataset(
        dense,
        (_Episode(3), _Episode(4), _Episode(5)),
        sampling_seed=11,
        stream_id=0,
    )

    assert len(dataset) == 3
    first_epoch = [dataset.dense_window_index(index) for index in range(3)]
    assert 0 <= first_epoch[0] < 3
    assert 3 <= first_epoch[1] < 7
    assert 7 <= first_epoch[2] < 12
    assert [dataset[index] for index in range(3)] == first_epoch

    dataset.set_epoch(1)
    second_epoch = [dataset.dense_window_index(index) for index in range(3)]
    assert first_epoch != second_epoch
    dataset.close()
    assert dense.closed


def test_episodic_view_rejects_an_inconsistent_dense_dataset():
    with pytest.raises(ValueError, match="length"):
        EpisodicUniformWindowDataset(
            _DenseDataset(4),
            (_Episode(2), _Episode(3)),
            sampling_seed=0,
            stream_id=0,
        )


def test_episodic_validation_cadence_matches_one_dense_epoch():
    episodes = tuple(
        [_Episode(618) for _ in range(40)]
        + [_Episode(645) for _ in range(40)]
    )

    assert _episodic_validation_interval(episodes, batch_size=8) == 632
