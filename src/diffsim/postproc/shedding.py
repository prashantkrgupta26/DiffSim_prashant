"""Post-processing helpers for vortex-shedding force and frequency extraction.

Provides two public functions:
  - time_avg_cd : post-transient mean drag coefficient
  - strouhal    : Strouhal number from FFT of the lift (or any periodic signal)
"""
import numpy as np


def time_avg_cd(t, cd, t_start=None):
    """Time-averaged drag coefficient over the post-transient window.

    Parameters
    ----------
    t : array_like, shape (N,)
        Time points (uniform or non-uniform).
    cd : array_like, shape (N,)
        Drag coefficient history.
    t_start : float or None
        Start of the averaging window.  If None (default), use the second
        half of the series (post-transient default: t >= t[N//2]).

    Returns
    -------
    float
        Mean Cd over the selected window.

    Raises
    ------
    ValueError
        If the selected window is empty.
    """
    t = np.asarray(t, dtype=float)
    cd = np.asarray(cd, dtype=float)
    if t_start is None:
        n_half = len(t) // 2
        mask = np.zeros(len(t), dtype=bool)
        mask[n_half:] = True
    else:
        mask = t >= t_start
    if not mask.any():
        raise ValueError(
            f"time_avg_cd: window is empty (t_start={t_start}, "
            f"t range [{t[0]}, {t[-1]}])"
        )
    return float(np.mean(cd[mask]))


def strouhal(t, signal, U, L):
    """Strouhal number from FFT of the detrended post-transient signal tail.

    Uses the second half of the series as the "tail" (same post-transient
    convention as time_avg_cd).  Detrends by subtracting the tail mean,
    then finds the dominant non-DC FFT peak.

    Parameters
    ----------
    t : array_like, shape (N,)
        Uniformly-spaced time points.
    signal : array_like, shape (N,)
        Periodic signal (e.g. lift coefficient Cl history).
    U : float
        Reference velocity (same units as L/t).
    L : float
        Reference length scale (same physical units as domain geometry).

    Returns
    -------
    St : float
        Strouhal number: St = freq * L / U.
    freq : float
        Dominant shedding frequency (Hz or 1/[time unit]).

    Raises
    ------
    ValueError
        If fewer than 4 time points are in the tail.
    """
    t = np.asarray(t, dtype=float)
    signal = np.asarray(signal, dtype=float)
    N = len(t)
    n_half = N // 2
    tail = signal[n_half:]
    N_tail = len(tail)
    if N_tail < 4:
        raise ValueError(
            f"strouhal: tail has only {N_tail} point(s); need at least 4. "
            f"Run more time steps."
        )
    # Detrend: subtract tail mean
    tail = tail - tail.mean()
    # Uniform dt assumption
    dt = (t[-1] - t[0]) / (len(t) - 1)
    # FFT
    fft_vals = np.fft.rfft(tail)
    magnitudes = np.abs(fft_vals)
    # Skip DC bin (index 0); find dominant peak among bins 1..
    peak_bin = int(np.argmax(magnitudes[1:])) + 1   # +1 to offset the skip
    freq = peak_bin * (1.0 / (dt * N_tail))
    St = freq * L / U
    return float(St), float(freq)
