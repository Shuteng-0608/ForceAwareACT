"""Checkpointable training batches over the real-robot valid-anchor index."""

from __future__ import annotations

import torch

from force_aware_act.training.sampling import EPISODIC_UNIFORM_SAMPLING


UNIFORM_VALID_ANCHOR_SAMPLING = "uniform_valid_anchor_with_replacement_v1"
EPISODIC_UNIFORM_VERSION = "episodic_uniform_valid_anchor_per_epoch_v1"
TRAIN_SAMPLING_MODES = (UNIFORM_VALID_ANCHOR_SAMPLING, EPISODIC_UNIFORM_SAMPLING)
SAMPLING_VERSIONS = {
    UNIFORM_VALID_ANCHOR_SAMPLING: UNIFORM_VALID_ANCHOR_SAMPLING,
    EPISODIC_UNIFORM_SAMPLING: EPISODIC_UNIFORM_VERSION,
}


def resolve_training_sampling_mode(requested=None, *, checkpoint_sampling=None):
    """Keep the legacy default and inherit an existing run's mode on resume."""
    if requested is not None:
        if requested not in TRAIN_SAMPLING_MODES:
            raise ValueError(f"unsupported real-robot train sampling mode: {requested}")
        return requested
    if checkpoint_sampling is None:
        return UNIFORM_VALID_ANCHOR_SAMPLING
    for mode, version in SAMPLING_VERSIONS.items():
        if checkpoint_sampling == version:
            return mode
    raise ValueError(f"unsupported checkpoint sampling contract: {checkpoint_sampling}")


class RealRobotTrainingSampler:
    """Select valid windows; episodic mode visits each episode once per epoch.

    The dataset groups valid anchors contiguously by episode. Counts must refer
    to those valid anchors, not native state/action counts or unfiltered grids.
    Batches stop at the epoch boundary; the final batch may be smaller than the
    requested size. The legacy mode keeps its original generator/tensor state.
    """

    def __init__(self, valid_anchor_counts, *, mode=UNIFORM_VALID_ANCHOR_SAMPLING, seed=0):
        self.mode = resolve_training_sampling_mode(mode)
        counts = tuple(valid_anchor_counts)
        if not counts or any(type(count) is not int or count < 0 for count in counts) or sum(counts) == 0:
            raise ValueError("valid anchor counts must be nonnegative integers with a positive total")
        if self.mode == EPISODIC_UNIFORM_SAMPLING and any(count == 0 for count in counts):
            raise ValueError("episodic_uniform requires at least one valid anchor in every training episode")
        self.valid_anchor_counts = counts
        offsets, total = [], 0
        for count in counts:
            offsets.append(total)
            total += count
        self._offsets = tuple(offsets)
        self._total = total
        self.generator = torch.Generator().manual_seed(seed)
        self.epoch = 0  # one-based once the first episodic sampling epoch begins
        self._episode_order = []
        self._cursor = 0

    @property
    def contract_version(self):
        return SAMPLING_VERSIONS[self.mode]

    @property
    def epoch_complete(self):
        return bool(self._episode_order) and self._cursor == len(self._episode_order)

    @property
    def completed_epochs(self):
        return self.epoch - int(bool(self._episode_order) and not self.epoch_complete)

    def sample_batch(self, batch_size):
        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError("batch size must be a positive integer")
        if self.mode == UNIFORM_VALID_ANCHOR_SAMPLING:
            return torch.randint(self._total, (batch_size,), generator=self.generator).tolist()
        if self._cursor == len(self._episode_order):
            self.epoch += 1
            self._episode_order = torch.randperm(len(self.valid_anchor_counts), generator=self.generator).tolist()
            self._cursor = 0
        stop = min(self._cursor + batch_size, len(self._episode_order))
        indices = [
            self._offsets[episode] + int(torch.randint(
                self.valid_anchor_counts[episode], (1,), generator=self.generator).item())
            for episode in self._episode_order[self._cursor:stop]
        ]
        self._cursor = stop
        return indices

    def get_state(self):
        if self.mode == UNIFORM_VALID_ANCHOR_SAMPLING:
            return self.generator.get_state()
        return {"version": self.contract_version, "generator_state": self.generator.get_state(),
                "valid_anchor_counts": list(self.valid_anchor_counts), "epoch": self.epoch,
                "episode_order": list(self._episode_order), "cursor": self._cursor}

    def validate_state(self, state):
        """Validate a checkpoint before changing model, optimizer or RNG state."""
        if self.mode == UNIFORM_VALID_ANCHOR_SAMPLING:
            generator_state = state
        else:
            keys = {"version", "generator_state", "valid_anchor_counts", "epoch", "episode_order", "cursor"}
            if not isinstance(state, dict) or set(state) != keys or state["version"] != self.contract_version:
                raise ValueError("checkpoint episodic sampler state has an incompatible version or fields")
            if state["valid_anchor_counts"] != list(self.valid_anchor_counts):
                raise ValueError("checkpoint sampler valid anchor counts differ from this run")
            epoch, cursor, order = state["epoch"], state["cursor"], state["episode_order"]
            if type(epoch) is not int or epoch < 0 or type(cursor) is not int or not isinstance(order, list):
                raise ValueError("checkpoint sampler epoch/order/cursor is invalid")
            if epoch == 0:
                valid = cursor == 0 and not order
            else:
                valid = (all(type(index) is int for index in order)
                         and sorted(order) == list(range(len(self.valid_anchor_counts)))
                         and 0 <= cursor <= len(order))
            if not valid:
                raise ValueError("checkpoint sampler epoch/order/cursor is invalid")
            generator_state = state["generator_state"]
        if not isinstance(generator_state, torch.Tensor):
            raise ValueError("checkpoint sampler generator state must be a tensor")
        try:
            torch.Generator().set_state(generator_state.cpu())
        except (RuntimeError, TypeError) as error:
            raise ValueError("checkpoint sampler generator state is invalid") from error

    def set_state(self, state):
        self.validate_state(state)
        if self.mode == UNIFORM_VALID_ANCHOR_SAMPLING:
            self.generator.set_state(state.cpu())
        else:
            self.generator.set_state(state["generator_state"].cpu())
            self.epoch = state["epoch"]
            self._episode_order = list(state["episode_order"])
            self._cursor = state["cursor"]
