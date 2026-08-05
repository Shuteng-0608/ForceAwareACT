import numpy as np
import pytest

from scripts.audit_rollout_force_usage import build_force_variants


def test_force_interventions_preserve_explicit_temporal_meanings():
    history = np.arange(36, dtype=np.float32).reshape(6, 6)

    variants = build_force_variants(
        history,
        window_len=4,
        delay_steps=2,
    )

    np.testing.assert_array_equal(variants["actual"], history[-4:])
    np.testing.assert_array_equal(
        variants["zero_physical_wrench"],
        np.zeros((4, 6), dtype=np.float32),
    )
    np.testing.assert_array_equal(
        variants["reversed_time"],
        history[-4:][::-1],
    )
    np.testing.assert_array_equal(variants["delayed_2_steps"], history[:4])


@pytest.mark.parametrize("window_len, delay_steps", ((0, 1), (4, 0)))
def test_force_interventions_reject_nonpositive_lengths(window_len, delay_steps):
    with pytest.raises(ValueError, match="must be positive"):
        build_force_variants(
            np.zeros((3, 6), dtype=np.float32),
            window_len=window_len,
            delay_steps=delay_steps,
        )
