"""Deterministic batch sampling for staged, multi-domain training.

The sampler operates on a descriptor for every global dataset index.  Each
batch first satisfies exact domain and phase quotas, then samples episodes
uniformly inside every requested ``(domain, phase)`` bucket.  Sampling uses
replacement across cycles so rare contact phases can be represented without
silently changing the requested batch composition.
"""

from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass
from typing import Any, Hashable, Mapping, Sequence, Tuple, Union

import torch
from torch.utils.data import Sampler


@dataclass(frozen=True)
class SampleDescriptor:
    """Sampling metadata associated with one global dataset index."""

    domain: Hashable
    episode_id: Hashable
    phase: Hashable


DescriptorLike = Union[
    SampleDescriptor,
    Mapping[str, Hashable],
    Tuple[Hashable, Hashable, Hashable],
]


def _stable_key(value: Hashable) -> tuple[str, str, str]:
    value_type = type(value)
    return (value_type.__module__, value_type.__qualname__, repr(value))


def _require_hashable(value: Any, name: str, index: int) -> Hashable:
    try:
        hash(value)
    except TypeError as error:
        raise ValueError(f"descriptor {index} {name} must be hashable") from error
    return value


def _coerce_descriptor(value: DescriptorLike, index: int) -> SampleDescriptor:
    if isinstance(value, SampleDescriptor):
        descriptor = value
    elif isinstance(value, Mapping):
        domain = value.get("domain", value.get("source"))
        if domain is None or "episode_id" not in value or "phase" not in value:
            raise ValueError(
                f"descriptor {index} mapping must contain domain, episode_id, and phase"
            )
        descriptor = SampleDescriptor(
            domain=domain,
            episode_id=value["episode_id"],
            phase=value["phase"],
        )
    elif isinstance(value, tuple) and len(value) == 3:
        descriptor = SampleDescriptor(value[0], value[1], value[2])
    else:
        try:
            descriptor = SampleDescriptor(
                domain=getattr(value, "domain"),
                episode_id=getattr(value, "episode_id"),
                phase=getattr(value, "phase"),
            )
        except AttributeError as error:
            raise ValueError(
                f"descriptor {index} must provide domain, episode_id, and phase"
            ) from error

    return SampleDescriptor(
        domain=_require_hashable(descriptor.domain, "domain", index),
        episode_id=_require_hashable(descriptor.episode_id, "episode_id", index),
        phase=_require_hashable(descriptor.phase, "phase", index),
    )


def _validate_positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _descriptor_signature(descriptors: Sequence[SampleDescriptor]) -> str:
    digest = hashlib.sha256()
    for index, descriptor in enumerate(descriptors):
        digest.update(
            repr((index, descriptor.domain, descriptor.episode_id, descriptor.phase)).encode(
                "utf-8"
            )
        )
        digest.update(b"\n")
    return digest.hexdigest()


class _EpisodeUniformCursor:
    """Cycle uniformly over episodes, and over samples within each episode."""

    def __init__(
        self,
        episode_to_indices: Mapping[Hashable, Sequence[int]],
        generator: torch.Generator,
    ) -> None:
        self._generator = generator
        self._episodes = sorted(episode_to_indices, key=_stable_key)
        self._indices = {
            episode: sorted(int(index) for index in episode_to_indices[episode])
            for episode in self._episodes
        }
        self._episode_order: list[Hashable] = []
        self._episode_position = 0
        self._sample_orders: dict[Hashable, list[int]] = {
            episode: [] for episode in self._episodes
        }
        self._sample_positions: dict[Hashable, int] = {
            episode: 0 for episode in self._episodes
        }

    def _permuted(self, values: Sequence[Any]) -> list[Any]:
        if len(values) <= 1:
            return list(values)
        order = torch.randperm(len(values), generator=self._generator).tolist()
        return [values[position] for position in order]

    def draw(self) -> tuple[int, Hashable]:
        if self._episode_position >= len(self._episode_order):
            self._episode_order = self._permuted(self._episodes)
            self._episode_position = 0
        episode = self._episode_order[self._episode_position]
        self._episode_position += 1

        sample_order = self._sample_orders[episode]
        sample_position = self._sample_positions[episode]
        if sample_position >= len(sample_order):
            sample_order = self._permuted(self._indices[episode])
            self._sample_orders[episode] = sample_order
            sample_position = 0
        sample_index = sample_order[sample_position]
        self._sample_positions[episode] = sample_position + 1
        return sample_index, episode


class DomainPhaseBatchSampler(Sampler[list[int]]):
    """Yield batches with exact domain/phase quotas and uniform episode draws.

    Args:
        descriptors: One descriptor for every global dataset index.  The
            sequence position is the index yielded to the DataLoader.
        domain_quotas: Exact number of samples from every requested domain in
            each batch.
        phase_quotas: Nested ``domain -> phase -> count`` quotas.  Counts for a
            domain must sum to that domain's quota.
        batches_per_epoch: Number of batches in one sampler epoch.
        seed: Base random seed.
        shuffle_within_batch: Randomly permute each completed batch.

    ``state_dict`` records both ``epoch`` and ``next_batch``.  Loading that
    state reconstructs and advances the deterministic sampling stream, so the
    next yielded batch is exactly the one that had not yet been consumed.
    """

    def __init__(
        self,
        descriptors: Sequence[DescriptorLike],
        *,
        domain_quotas: Mapping[Hashable, int],
        phase_quotas: Mapping[Hashable, Mapping[Hashable, int]],
        batches_per_epoch: int,
        seed: int = 0,
        shuffle_within_batch: bool = True,
    ) -> None:
        if not descriptors:
            raise ValueError("descriptors must not be empty")
        self.descriptors = tuple(
            _coerce_descriptor(descriptor, index)
            for index, descriptor in enumerate(descriptors)
        )
        self.domain_quotas = self._validate_domain_quotas(domain_quotas)
        self.phase_quotas = self._validate_phase_quotas(phase_quotas)
        self.batches_per_epoch = _validate_positive_integer(
            batches_per_epoch, "batches_per_epoch"
        )
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if not isinstance(shuffle_within_batch, bool):
            raise ValueError("shuffle_within_batch must be a boolean")
        self.seed = seed
        self.shuffle_within_batch = shuffle_within_batch
        self.batch_size = sum(self.domain_quotas.values())
        self._descriptor_signature = _descriptor_signature(self.descriptors)
        self._buckets = self._build_buckets()
        self._epoch = 0
        self._next_batch = 0
        self._last_epoch_counts = self._empty_counts(epoch=0)

    @staticmethod
    def _validate_domain_quotas(
        quotas: Mapping[Hashable, int],
    ) -> dict[Hashable, int]:
        if not isinstance(quotas, Mapping) or not quotas:
            raise ValueError("domain_quotas must be a non-empty mapping")
        validated: dict[Hashable, int] = {}
        for domain, quota in quotas.items():
            _require_hashable(domain, "domain quota key", -1)
            validated[domain] = _validate_positive_integer(
                quota, f"domain quota for {domain!r}"
            )
        return validated

    def _validate_phase_quotas(
        self,
        quotas: Mapping[Hashable, Mapping[Hashable, int]],
    ) -> dict[Hashable, dict[Hashable, int]]:
        if not isinstance(quotas, Mapping):
            raise ValueError("phase_quotas must be a mapping")
        if set(quotas) != set(self.domain_quotas):
            missing = set(self.domain_quotas) - set(quotas)
            extra = set(quotas) - set(self.domain_quotas)
            raise ValueError(
                "phase_quotas domains must exactly match domain_quotas; "
                f"missing={sorted(map(repr, missing))} extra={sorted(map(repr, extra))}"
            )

        validated: dict[Hashable, dict[Hashable, int]] = {}
        for domain in self.domain_quotas:
            domain_phases = quotas[domain]
            if not isinstance(domain_phases, Mapping) or not domain_phases:
                raise ValueError(f"phase quotas for domain {domain!r} must be non-empty")
            validated_phases: dict[Hashable, int] = {}
            for phase, quota in domain_phases.items():
                _require_hashable(phase, "phase quota key", -1)
                validated_phases[phase] = _validate_positive_integer(
                    quota, f"phase quota for domain={domain!r} phase={phase!r}"
                )
            total = sum(validated_phases.values())
            if total != self.domain_quotas[domain]:
                raise ValueError(
                    f"phase quotas for domain {domain!r} sum to {total}, "
                    f"expected {self.domain_quotas[domain]}"
                )
            validated[domain] = validated_phases
        return validated

    def _build_buckets(
        self,
    ) -> dict[tuple[Hashable, Hashable], dict[Hashable, list[int]]]:
        buckets: dict[tuple[Hashable, Hashable], dict[Hashable, list[int]]] = {}
        for domain, phases in self.phase_quotas.items():
            for phase in phases:
                buckets[(domain, phase)] = {}

        for index, descriptor in enumerate(self.descriptors):
            key = (descriptor.domain, descriptor.phase)
            if key not in buckets:
                continue
            episode_bucket = buckets[key]
            episode_bucket.setdefault(descriptor.episode_id, []).append(index)

        for domain, phases in self.phase_quotas.items():
            for phase in phases:
                if not buckets[(domain, phase)]:
                    raise ValueError(
                        "empty sampling bucket for "
                        f"domain={domain!r} phase={phase!r}"
                    )
        return buckets

    @property
    def epoch(self) -> int:
        return self._epoch

    @property
    def next_batch(self) -> int:
        return self._next_batch

    @property
    def last_epoch_counts(self) -> dict[str, Any]:
        """Return a defensive copy of realized domain/phase/episode counts."""

        return copy.deepcopy(self._last_epoch_counts)

    @property
    def phase_episode_counts(self) -> dict[Hashable, dict[Hashable, int]]:
        """Return independent episode coverage for every sampled bucket."""

        return {
            domain: {
                phase: len(self._buckets[(domain, phase)])
                for phase in self.phase_quotas[domain]
            }
            for domain in self.domain_quotas
        }

    def realized_quota_summary(self) -> dict[str, Any]:
        """Alias suitable for structured training logs."""

        return self.last_epoch_counts

    def _empty_counts(self, *, epoch: int) -> dict[str, Any]:
        return {
            "epoch": epoch,
            "batches": 0,
            "complete": False,
            "domain": {domain: 0 for domain in self.domain_quotas},
            "phase": {
                domain: {phase: 0 for phase in phases}
                for domain, phases in self.phase_quotas.items()
            },
            "episode": {domain: {} for domain in self.domain_quotas},
        }

    def set_epoch(self, epoch: int) -> None:
        """Start a named epoch from its first batch."""

        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise ValueError("epoch must be a non-negative integer")
        self._epoch = epoch
        self._next_batch = 0
        self._last_epoch_counts = self._empty_counts(epoch=epoch)

    def _generator_for_epoch(self) -> torch.Generator:
        generator = torch.Generator(device="cpu")
        epoch_seed = (self.seed + self._epoch * 1_000_003) % (2**63 - 1)
        generator.manual_seed(epoch_seed)
        return generator

    def _build_cursors(
        self, generator: torch.Generator
    ) -> dict[tuple[Hashable, Hashable], _EpisodeUniformCursor]:
        return {
            key: _EpisodeUniformCursor(episode_to_indices, generator)
            for key, episode_to_indices in sorted(
                self._buckets.items(),
                key=lambda item: (_stable_key(item[0][0]), _stable_key(item[0][1])),
            )
        }

    def _draw_batch(
        self,
        cursors: Mapping[tuple[Hashable, Hashable], _EpisodeUniformCursor],
        generator: torch.Generator,
    ) -> tuple[list[int], list[tuple[Hashable, Hashable, Hashable]]]:
        batch: list[int] = []
        draws: list[tuple[Hashable, Hashable, Hashable]] = []
        for domain in sorted(self.domain_quotas, key=_stable_key):
            for phase in sorted(self.phase_quotas[domain], key=_stable_key):
                quota = self.phase_quotas[domain][phase]
                cursor = cursors[(domain, phase)]
                for _ in range(quota):
                    sample_index, episode = cursor.draw()
                    batch.append(sample_index)
                    draws.append((domain, phase, episode))

        if self.shuffle_within_batch and len(batch) > 1:
            permutation = torch.randperm(len(batch), generator=generator).tolist()
            batch = [batch[position] for position in permutation]
            draws = [draws[position] for position in permutation]
        return batch, draws

    def _record_draws(
        self, draws: Sequence[tuple[Hashable, Hashable, Hashable]]
    ) -> None:
        self._last_epoch_counts["batches"] += 1
        for domain, phase, episode in draws:
            self._last_epoch_counts["domain"][domain] += 1
            self._last_epoch_counts["phase"][domain][phase] += 1
            episode_counts = self._last_epoch_counts["episode"][domain]
            episode_counts[episode] = episode_counts.get(episode, 0) + 1

    def __iter__(self):
        resume_batch = self._next_batch
        generator = self._generator_for_epoch()
        cursors = self._build_cursors(generator)
        self._last_epoch_counts = self._empty_counts(epoch=self._epoch)

        for batch_index in range(self.batches_per_epoch):
            batch, draws = self._draw_batch(cursors, generator)
            self._record_draws(draws)
            if batch_index < resume_batch:
                continue
            self._next_batch = batch_index + 1
            yield batch

        self._last_epoch_counts["complete"] = True
        self._epoch += 1
        self._next_batch = 0

    def __len__(self) -> int:
        return self.batches_per_epoch

    def state_dict(self) -> dict[str, Any]:
        """Return exact-resume state at the next unconsumed batch boundary."""

        epoch = self._epoch
        next_batch = self._next_batch
        if next_batch == self.batches_per_epoch:
            epoch += 1
            next_batch = 0
        return {
            "version": 1,
            "epoch": epoch,
            "next_batch": next_batch,
            "seed": self.seed,
            "batches_per_epoch": self.batches_per_epoch,
            "shuffle_within_batch": self.shuffle_within_batch,
            "domain_quotas": copy.deepcopy(self.domain_quotas),
            "phase_quotas": copy.deepcopy(self.phase_quotas),
            "descriptor_count": len(self.descriptors),
            "descriptor_signature": self._descriptor_signature,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore exact-resume state and reject incompatible sampler configs."""

        if not isinstance(state, Mapping):
            raise ValueError("sampler state must be a mapping")
        if state.get("version") != 1:
            raise ValueError(f"unsupported sampler state version: {state.get('version')!r}")
        compatibility_checks = {
            "batches_per_epoch": self.batches_per_epoch,
            "shuffle_within_batch": self.shuffle_within_batch,
            "domain_quotas": self.domain_quotas,
            "phase_quotas": self.phase_quotas,
            "descriptor_count": len(self.descriptors),
            "descriptor_signature": self._descriptor_signature,
        }
        for key, expected in compatibility_checks.items():
            if state.get(key) != expected:
                raise ValueError(
                    f"sampler state {key} is incompatible: "
                    f"checkpoint={state.get(key)!r} current={expected!r}"
                )

        epoch = state.get("epoch")
        next_batch = state.get("next_batch")
        seed = state.get("seed")
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise ValueError("sampler state epoch must be a non-negative integer")
        if (
            isinstance(next_batch, bool)
            or not isinstance(next_batch, int)
            or not 0 <= next_batch <= self.batches_per_epoch
        ):
            raise ValueError(
                "sampler state next_batch must be in "
                f"[0, {self.batches_per_epoch}]"
            )
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("sampler state seed must be a non-negative integer")
        if next_batch == self.batches_per_epoch:
            epoch += 1
            next_batch = 0

        self.seed = seed
        self._epoch = epoch
        self._next_batch = next_batch
        self._last_epoch_counts = self._empty_counts(epoch=epoch)
