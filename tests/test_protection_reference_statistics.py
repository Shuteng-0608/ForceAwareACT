"""Numerical statistics checks only; no hardware or protection fault injection."""
import numpy as np
import pytest
from scripts.analyze_protection_references import hysteresis_windows, window_rate, ceil_step


def test_hysteresis_duration_retains_band_but_ends_before_release():
    t=np.arange(6,dtype=np.int64)*10_000_000
    assert hysteresis_windows(t,[0.,3.,1.5,2.2,1.,3.],2.,1.)==[(1,3),(5,5)]


def test_missing_time_splits_observed_intervals():
    t=np.array([0,10,40,50],dtype=np.int64)*1_000_000
    assert hysteresis_windows(t,[3.,3.,3.,3.],2.,1.)==[(0,1),(2,3)]
    with pytest.raises(ValueError):hysteresis_windows([0,0],[3.,3.],2.,1.)


def test_backward_secants_preserve_rate_and_exclude_gaps():
    t=np.array([0,10,20,60,70,80],dtype=np.int64)*1_000_000
    x=np.repeat((t/1e9)[:,None],7,axis=1)*2
    end,rates=window_rate(t,x,20_000_000)
    assert end.tolist()==[2,5]
    np.testing.assert_allclose(rates,2.)


def test_reference_rounding_never_rounds_down():
    np.testing.assert_allclose(ceil_step([1.,1.01,.31],[.5,.5,.1]),[1.,1.5,.4])
