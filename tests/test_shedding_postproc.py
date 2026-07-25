"""Unit tests for diffsim.postproc.shedding helpers.

Deterministic, CPU-only, fast (< 1 second).  No CUDA required.

Covers:
  - strouhal() recovers a synthetic sine frequency to < 2% relative error
  - time_avg_cd() recovers a known constant mean to < 1e-10 absolute error
  - time_avg_cd() with explicit t_start keyword
  - strouhal() raises ValueError when tail is too short
  - time_avg_cd() raises ValueError when window is empty
"""
import numpy as np
import pytest

from diffsim.postproc.shedding import strouhal, time_avg_cd


def test_strouhal_recovers_synthetic_frequency():
    """FFT peak from a pure sine recovers the known frequency to < 2%."""
    # 4001 points over [0, 1.0] => dt = 0.00025, Nyquist = 2000 Hz
    t = np.linspace(0, 1.0, 4001)
    f0 = 10.0           # known frequency (Hz)
    signal = np.sin(2 * np.pi * f0 * t)

    # With U=1 and L=1, St = freq * L / U = freq = f0
    St_hat, freq_hat = strouhal(t, signal, U=1.0, L=1.0)

    assert abs(St_hat - f0) / f0 < 0.02, (
        f"Strouhal {St_hat:.4f} deviates > 2% from expected {f0}"
    )
    assert abs(freq_hat - f0) / f0 < 0.02, (
        f"freq {freq_hat:.4f} deviates > 2% from expected {f0}"
    )


def test_time_avg_cd_recovers_known_mean():
    """Constant Cd array: mean recovers to < 1e-10 absolute error."""
    cd = np.ones(100) * 3.37
    t_cd = np.linspace(0, 10.0, 100)
    mean_cd = time_avg_cd(t_cd, cd)
    assert abs(mean_cd - 3.37) < 1e-10, (
        f"time_avg_cd returned {mean_cd}, expected 3.37"
    )


def test_time_avg_cd_with_t_start():
    """Explicit t_start: only values at or after t_start are averaged."""
    t = np.linspace(0, 10.0, 101)
    # First half = 0, second half = 1 (starting from t=5)
    cd = np.where(t >= 5.0, 1.0, 0.0)
    mean_cd = time_avg_cd(t, cd, t_start=5.0)
    assert abs(mean_cd - 1.0) < 1e-10, (
        f"time_avg_cd with t_start=5.0 returned {mean_cd}, expected 1.0"
    )


def test_strouhal_raises_on_short_series():
    """strouhal raises ValueError if fewer than 4 tail points."""
    t = np.linspace(0, 0.001, 6)    # only 3 points in tail
    signal = np.sin(t)
    with pytest.raises(ValueError, match="tail"):
        strouhal(t, signal, U=1.0, L=1.0)


def test_time_avg_cd_raises_on_empty_window():
    """time_avg_cd raises ValueError if t_start is beyond the series."""
    t = np.linspace(0, 1.0, 10)
    cd = np.ones(10)
    with pytest.raises(ValueError):
        time_avg_cd(t, cd, t_start=100.0)
