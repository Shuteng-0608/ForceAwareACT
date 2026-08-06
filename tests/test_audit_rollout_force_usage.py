import numpy as np
import pytest

from scripts.audit_rollout_force_usage import (
    _native_window_summary,
    build_force_variants,
    build_native_rate_force_variants,
)


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


def test_native_rate_interventions_keep_timestamps_implicit_and_delay_samples():
    history = np.arange(30, dtype=np.float32).reshape(5, 6)

    variants = build_native_rate_force_variants(
        history,
        delay_force_samples=2,
    )

    assert set(variants) == {
        "actual",
        "zero_physical_wrench",
        "reversed_time",
        "delayed_2_force_samples",
    }
    np.testing.assert_array_equal(variants["actual"], history)
    np.testing.assert_array_equal(
        variants["zero_physical_wrench"], np.zeros_like(history)
    )
    np.testing.assert_array_equal(variants["reversed_time"], history[::-1])
    expected_delayed = np.vstack((history[0], history[0], history[:-2]))
    np.testing.assert_array_equal(
        variants["delayed_2_force_samples"], expected_delayed
    )


def test_native_rate_interventions_reject_nonpositive_delay():
    with pytest.raises(ValueError, match="must be positive"):
        build_native_rate_force_variants(
            np.zeros((3, 6), dtype=np.float32),
            delay_force_samples=0,
        )


def test_native_window_summary_reports_rate_span_and_causality():
    summary = _native_window_summary(
        np.asarray((1.000, 1.002, 1.004), dtype=np.float64),
        state_timestamp=1.005,
    )

    assert summary["valid_sample_count"] == 3
    assert summary["window_span_seconds"] == pytest.approx(0.004)
    assert summary["latest_sample_age_seconds"] == pytest.approx(0.001)
    assert summary["median_sample_period_seconds"] == pytest.approx(0.002)
    assert summary["observed_sample_rate_hz"] == pytest.approx(500.0)
    assert summary["strictly_causal"] is True
