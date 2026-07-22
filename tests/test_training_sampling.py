from collections import Counter

import pytest

from force_aware_act.training.sampling import (
    DomainPhaseBatchSampler,
    SampleDescriptor,
)


def _descriptor_counts(batch, descriptors):
    return Counter(
        (descriptors[index].domain, descriptors[index].phase)
        for index in batch
    )


def test_batches_satisfy_exact_domain_and_phase_quotas_and_log_counts():
    descriptors = [
        SampleDescriptor("r60", "r60_a", "free"),
        SampleDescriptor("r60", "r60_b", "free"),
        SampleDescriptor("r60", "r60_a", "contact"),
        SampleDescriptor("r60", "r60_b", "contact"),
        SampleDescriptor("r2", "r2_a", "precontact"),
        SampleDescriptor("r2", "r2_b", "precontact"),
        SampleDescriptor("r2", "r2_a", "contact"),
        SampleDescriptor("r2", "r2_b", "contact"),
    ]
    sampler = DomainPhaseBatchSampler(
        descriptors,
        domain_quotas={"r60": 2, "r2": 4},
        phase_quotas={
            "r60": {"free": 1, "contact": 1},
            "r2": {"precontact": 1, "contact": 3},
        },
        batches_per_epoch=5,
        seed=11,
    )

    batches = list(sampler)

    assert len(batches) == 5
    assert sampler.batch_size == 6
    for batch in batches:
        assert len(batch) == 6
        assert _descriptor_counts(batch, descriptors) == {
            ("r60", "free"): 1,
            ("r60", "contact"): 1,
            ("r2", "precontact"): 1,
            ("r2", "contact"): 3,
        }

    summary = sampler.realized_quota_summary()
    assert summary["epoch"] == 0
    assert summary["batches"] == 5
    assert summary["complete"] is True
    assert summary["domain"] == {"r60": 10, "r2": 20}
    assert summary["phase"]["r2"] == {"precontact": 5, "contact": 15}


def test_episode_selection_is_uniform_even_when_episode_lengths_differ():
    descriptors = [SampleDescriptor("r2", "short", "contact")]
    descriptors.extend(
        SampleDescriptor("r2", "long", "contact") for _ in range(100)
    )
    sampler = DomainPhaseBatchSampler(
        descriptors,
        domain_quotas={"r2": 1},
        phase_quotas={"r2": {"contact": 1}},
        batches_per_epoch=20,
        seed=3,
        shuffle_within_batch=False,
    )

    list(sampler)

    assert sampler.last_epoch_counts["episode"]["r2"] == {
        "short": 10,
        "long": 10,
    }


def test_same_seed_and_epoch_are_reproducible_and_new_epoch_changes_order():
    descriptors = [
        SampleDescriptor("r60", f"episode_{index % 3}", "free")
        for index in range(18)
    ]

    def build():
        return DomainPhaseBatchSampler(
            descriptors,
            domain_quotas={"r60": 4},
            phase_quotas={"r60": {"free": 4}},
            batches_per_epoch=5,
            seed=91,
        )

    first = build()
    second = build()
    assert list(first) == list(second)

    first.set_epoch(7)
    second.set_epoch(7)
    epoch_seven = list(first)
    assert epoch_seven == list(second)

    third = build()
    third.set_epoch(8)
    assert epoch_seven != list(third)


def test_state_dict_resume_continues_at_exact_next_batch():
    descriptors = [
        SampleDescriptor("r60", f"r60_{index % 2}", "free")
        for index in range(12)
    ] + [
        SampleDescriptor("r2", f"r2_{index % 3}", "contact")
        for index in range(18)
    ]

    def build(seed=17):
        return DomainPhaseBatchSampler(
            descriptors,
            domain_quotas={"r60": 2, "r2": 2},
            phase_quotas={"r60": {"free": 2}, "r2": {"contact": 2}},
            batches_per_epoch=7,
            seed=seed,
        )

    expected = list(build())
    interrupted = build()
    iterator = iter(interrupted)
    consumed = [next(iterator), next(iterator), next(iterator)]
    state = interrupted.state_dict()
    assert state["epoch"] == 0
    assert state["next_batch"] == 3

    resumed = build(seed=999)
    resumed.load_state_dict(state)
    remaining = list(resumed)

    assert consumed + remaining == expected
    assert resumed.last_epoch_counts["batches"] == 7
    assert resumed.last_epoch_counts["complete"] is True


def test_state_after_last_consumed_batch_points_to_next_epoch():
    descriptors = [SampleDescriptor("r2", "episode", "contact")]
    sampler = DomainPhaseBatchSampler(
        descriptors,
        domain_quotas={"r2": 1},
        phase_quotas={"r2": {"contact": 1}},
        batches_per_epoch=2,
    )
    iterator = iter(sampler)
    next(iterator)
    next(iterator)

    assert sampler.state_dict()["epoch"] == 1
    assert sampler.state_dict()["next_batch"] == 0


def test_empty_requested_bucket_is_an_explicit_error():
    descriptors = [SampleDescriptor("r2", "episode", "free")]

    with pytest.raises(ValueError, match="empty sampling bucket.*contact"):
        DomainPhaseBatchSampler(
            descriptors,
            domain_quotas={"r2": 1},
            phase_quotas={"r2": {"contact": 1}},
            batches_per_epoch=1,
        )


def test_phase_quota_must_sum_to_domain_quota():
    descriptors = [SampleDescriptor("r2", "episode", "contact")]

    with pytest.raises(ValueError, match="sum to 1, expected 2"):
        DomainPhaseBatchSampler(
            descriptors,
            domain_quotas={"r2": 2},
            phase_quotas={"r2": {"contact": 1}},
            batches_per_epoch=1,
        )


def test_mapping_descriptors_accept_source_as_domain_alias():
    sampler = DomainPhaseBatchSampler(
        [{"source": "r60", "episode_id": "episode", "phase": "free"}],
        domain_quotas={"r60": 1},
        phase_quotas={"r60": {"free": 1}},
        batches_per_epoch=1,
    )

    assert list(sampler) == [[0]]


def test_resume_rejects_a_different_descriptor_catalog():
    first = DomainPhaseBatchSampler(
        [SampleDescriptor("r2", "episode_a", "contact")],
        domain_quotas={"r2": 1},
        phase_quotas={"r2": {"contact": 1}},
        batches_per_epoch=1,
    )
    second = DomainPhaseBatchSampler(
        [SampleDescriptor("r2", "episode_b", "contact")],
        domain_quotas={"r2": 1},
        phase_quotas={"r2": {"contact": 1}},
        batches_per_epoch=1,
    )

    with pytest.raises(ValueError, match="descriptor_signature is incompatible"):
        second.load_state_dict(first.state_dict())
