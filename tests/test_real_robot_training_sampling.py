"""Episode balance, valid-window selection and resumable sampling contracts."""

import copy
from collections import Counter

import pytest
import torch

from force_aware_act.real_robot.checkpoint import sampler_states_equal
from force_aware_act.real_robot.training_sampling import (
    EPISODIC_UNIFORM_SAMPLING, EPISODIC_UNIFORM_VERSION, UNIFORM_VALID_ANCHOR_SAMPLING,
    RealRobotTrainingSampler, resolve_training_sampling_mode,
)


def _episode(index, counts):
    offset = 0
    for episode, count in enumerate(counts):
        if offset <= index < offset + count:
            return episode
        offset += count
    raise AssertionError("sample is outside the valid-anchor index")


@pytest.mark.parametrize("batch_size", [1, 2, 3, 8])
def test_unequal_episodes_each_contribute_once_per_epoch_without_cross_epoch_fill(batch_size):
    # A dense index may represent sparse grid anchors: selection only sees the
    # actual valid entries, rather than sampling unfiltered grid timesteps.
    counts = [2, 37, 1]
    sampler = RealRobotTrainingSampler(counts, mode=EPISODIC_UNIFORM_SAMPLING, seed=17)
    appearances, windows = Counter(), set()
    orders = []
    for epoch in range(1, 21):
        samples, sizes = [], []
        for _ in range((len(counts) + batch_size - 1) // batch_size):
            indices = sampler.sample_batch(batch_size)
            assert sampler.epoch == epoch
            sizes.append(len(indices))
            samples.extend(indices)
        episodes = [_episode(index, counts) for index in samples]
        assert sorted(episodes) == [0, 1, 2]
        assert sizes == [min(batch_size, len(counts) - start) for start in range(0, len(counts), batch_size)]
        assert sampler.epoch_complete and sampler.completed_epochs == epoch
        appearances.update(episodes)
        windows.update(samples)
        orders.append(tuple(episodes))
    assert appearances == {0: 20, 1: 20, 2: 20}
    assert len(set(orders)) > 1
    assert len(windows) > len(counts)  # new epochs reselect timesteps


@pytest.mark.parametrize("batches_before_save", [0, 1, 3, 4])
def test_episodic_resume_preserves_next_batches_at_mid_epoch_and_boundaries(batches_before_save):
    counts = [3, 15, 2, 9, 1]
    original = RealRobotTrainingSampler(counts, mode=EPISODIC_UNIFORM_SAMPLING, seed=4)
    for _ in range(batches_before_save):
        original.sample_batch(2)
    state = original.get_state()
    restored = RealRobotTrainingSampler(counts, mode=EPISODIC_UNIFORM_SAMPLING, seed=999)
    restored.set_state(state)
    assert sampler_states_equal(original.get_state(), restored.get_state())
    for _ in range(10):
        assert original.sample_batch(2) == restored.sample_batch(2)
        assert original.epoch == restored.epoch
        assert original.epoch_complete == restored.epoch_complete
    # Consuming either sampler cannot change the already saved epoch/order.
    assert restored.get_state()["episode_order"] is not state["episode_order"]


def test_legacy_sampling_retains_exact_generator_sequence_and_tensor_checkpoint():
    sampler = RealRobotTrainingSampler([0, 2, 37, 1], seed=17)
    reference = torch.Generator().manual_seed(17)
    for batch_size in (1, 3, 2):
        assert sampler.sample_batch(batch_size) == torch.randint(40, (batch_size,), generator=reference).tolist()
    state = sampler.get_state()
    assert isinstance(state, torch.Tensor)
    assert torch.equal(state, reference.get_state())
    restored = RealRobotTrainingSampler([0, 2, 37, 1], seed=999)
    restored.set_state(state)
    assert sampler.sample_batch(5) == restored.sample_batch(5)


@pytest.mark.parametrize("field,value", [
    ("version", "unknown"), ("valid_anchor_counts", [2, 3]),
    ("epoch", -1), ("epoch", True), ("cursor", 4),
    ("episode_order", [0, 0, 2]), ("episode_order", [False, 1, 2]),
    ("generator_state", torch.zeros(2, dtype=torch.uint8)),
])
def test_invalid_sampler_state_is_rejected_without_mutation(field, value):
    sampler = RealRobotTrainingSampler([2, 3, 4], mode=EPISODIC_UNIFORM_SAMPLING, seed=9)
    sampler.sample_batch(2)
    before = sampler.get_state()
    invalid = copy.deepcopy(before)
    invalid[field] = value
    with pytest.raises(ValueError, match="sampler"):
        sampler.set_state(invalid)
    assert sampler_states_equal(before, sampler.get_state())


@pytest.mark.parametrize("counts", [[], [0, 0], [-1, 2], [1.5], [True]])
def test_invalid_anchor_counts_are_rejected(counts):
    with pytest.raises(ValueError, match="anchor counts"):
        RealRobotTrainingSampler(counts)


def test_episodic_mode_rejects_an_empty_episode_instead_of_silently_reweighting():
    with pytest.raises(ValueError, match="every training episode"):
        RealRobotTrainingSampler([2, 0], mode=EPISODIC_UNIFORM_SAMPLING)


def test_mode_resolution_preserves_default_and_inherits_old_or_new_checkpoint():
    assert resolve_training_sampling_mode() == UNIFORM_VALID_ANCHOR_SAMPLING
    assert resolve_training_sampling_mode(checkpoint_sampling=UNIFORM_VALID_ANCHOR_SAMPLING) == UNIFORM_VALID_ANCHOR_SAMPLING
    assert resolve_training_sampling_mode(checkpoint_sampling=EPISODIC_UNIFORM_VERSION) == EPISODIC_UNIFORM_SAMPLING
    assert resolve_training_sampling_mode(EPISODIC_UNIFORM_SAMPLING) == EPISODIC_UNIFORM_SAMPLING
    with pytest.raises(ValueError, match="sampling mode"):
        resolve_training_sampling_mode("dense_windows")
    with pytest.raises(ValueError, match="sampling contract"):
        resolve_training_sampling_mode(checkpoint_sampling="unknown")
