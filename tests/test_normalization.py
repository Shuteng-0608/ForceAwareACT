import torch

from force_aware_act.data.normalization import (
    compute_normalization_stats_from_batches,
    denormalize_tensor,
    normalize_tensor,
)


def test_compute_normalization_stats_shapes():
    batch = {
        "qpos": torch.randn(2, 7),
        "action_chunk": torch.randn(2, 5, 7),
        "force_window": torch.randn(2, 4, 6),
        "future_force_chunk": torch.randn(2, 5, 6),
    }

    stats = compute_normalization_stats_from_batches([batch])

    assert stats["qpos_mean"].shape == (7,)
    assert stats["qpos_std"].shape == (7,)
    assert stats["action_mean"].shape == (7,)
    assert stats["action_std"].shape == (7,)
    assert stats["force_mean"].shape == (6,)
    assert stats["force_std"].shape == (6,)


def test_normalize_then_denormalize_recovers_values():
    x = torch.randn(2, 5, 7)
    mean = torch.randn(7)
    std = torch.rand(7) + 0.5

    normalized = normalize_tensor(x, mean, std)
    recovered = denormalize_tensor(normalized, mean, std)

    torch.testing.assert_close(recovered, x)


def test_std_is_never_below_eps():
    eps = 1.0e-3
    batch = {
        "qpos": torch.ones(2, 7),
        "action_chunk": torch.ones(2, 5, 7),
        "force_window": torch.ones(2, 4, 6),
        "future_force_chunk": torch.ones(2, 5, 6),
    }

    stats = compute_normalization_stats_from_batches([batch], eps=eps)

    assert torch.all(stats["qpos_std"] >= eps)
    assert torch.all(stats["action_std"] >= eps)
    assert torch.all(stats["force_std"] >= eps)
