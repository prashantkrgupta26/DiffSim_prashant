"""Stochastic diagnostics — ensemble aggregation, bootstrap CIs, and
first-passage event detection.

Noise-driven runs (FDT thermal noise, nucleation) must be reported as
ENSEMBLES with uncertainty, not single seeds (spec P8/P9): nucleation
probability, induction-time distribution, nuclei density, crystallinity
distribution, each with a mean and a 95% confidence interval. Units are
inherited from the per-seed statistics the caller supplies.
"""
from __future__ import annotations

import numpy as np


def ensemble_aggregate(samples, axis=0):
    """Mean, standard deviation and standard error across an ensemble.

    Parameters
    ----------
    samples : array_like
        Per-seed values. With ``axis=0`` each row is one seed; the aggregate is
        taken over seeds, so a time series per seed aggregates elementwise.
    axis : int
        Ensemble axis.

    Returns
    -------
    dict
        ``mean``, ``sd`` (sample standard deviation, ddof=1), ``sem``
        (standard error of the mean), and ``n`` (ensemble size). Same units as
        ``samples``.
    """
    x = np.asarray(samples, dtype=float)
    n = x.shape[axis]
    mean = np.mean(x, axis=axis)
    sd = np.std(x, axis=axis, ddof=1) if n > 1 else np.zeros_like(mean)
    sem = sd / np.sqrt(n) if n > 1 else np.zeros_like(mean)
    return {"mean": mean, "sd": sd, "sem": sem, "n": int(n)}


def bootstrap_ci(samples, statistic=np.mean, confidence=0.95,
                 n_boot=10000, seed=0):
    """Bootstrap confidence interval for a statistic of a 1-D sample.

    Resamples with replacement ``n_boot`` times and takes the percentile
    interval. Robust for small ``n`` and non-Gaussian statistics.

    Parameters
    ----------
    samples : array_like
        1-D sample of per-seed values.
    statistic : callable
        Reduces a 1-D array to a scalar (default: mean).
    confidence : float
        Coverage (0.95 -> 95% CI).
    n_boot : int
        Number of bootstrap resamples.
    seed : int
        RNG seed for reproducibility of the CI itself.

    Returns
    -------
    dict
        ``estimate`` (statistic on the data), ``low``, ``high`` (CI bounds),
        ``confidence``. Same units as the statistic.
    """
    x = np.asarray(samples, dtype=float).ravel()
    n = x.size
    if n == 0:
        raise ValueError("empty sample")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot = np.array([statistic(x[i]) for i in idx])
    alpha = 1.0 - confidence
    low = float(np.percentile(boot, 100 * alpha / 2))
    high = float(np.percentile(boot, 100 * (1 - alpha / 2)))
    return {"estimate": float(statistic(x)), "low": low, "high": high,
            "confidence": float(confidence)}


def first_passage(series, threshold, times=None, direction="up"):
    """Time at which a signal first crosses a threshold (event detection).

    Used for induction / first-passage times: the step at which crystallinity
    (or an embryo radius) first reaches a detection level.

    Parameters
    ----------
    series : array_like
        Signal per step.
    threshold : float
        Crossing level.
    times : array_like, optional
        Times per step; default is the integer step index.
    direction : {"up", "down"}
        Crossing sense.

    Returns
    -------
    float or None
        Linearly interpolated crossing time, or ``None`` if never crossed.
        Units: those of ``times`` (steps by default).
    """
    s = np.asarray(series, dtype=float).ravel()
    if times is None:
        times = np.arange(s.size, dtype=float)
    t = np.asarray(times, dtype=float).ravel()
    if direction == "up":
        hit = np.where(s >= threshold)[0]
    else:
        hit = np.where(s <= threshold)[0]
    if hit.size == 0:
        return None
    k = int(hit[0])
    if k == 0:
        return float(t[0])
    # linear interpolation between k-1 and k
    s0, s1 = s[k - 1], s[k]
    if s1 == s0:
        return float(t[k])
    frac = (threshold - s0) / (s1 - s0)
    return float(t[k - 1] + frac * (t[k] - t[k - 1]))


def event_probability(events):
    """Fraction of ensemble members in which an event occurred.

    Parameters
    ----------
    events : array_like of bool
        One boolean per seed (e.g. "nucleated within the horizon").

    Returns
    -------
    dict
        ``p`` (probability), ``n`` (ensemble size), and a Wald 95% ``low``/
        ``high`` band clamped to ``[0, 1]``. Dimensionless.
    """
    e = np.asarray(events, dtype=bool).ravel()
    n = e.size
    if n == 0:
        raise ValueError("empty event list")
    p = float(np.mean(e))
    se = np.sqrt(p * (1 - p) / n)
    return {"p": p, "n": int(n),
            "low": max(0.0, p - 1.96 * se),
            "high": min(1.0, p + 1.96 * se)}


__all__ = [
    "ensemble_aggregate", "bootstrap_ci", "first_passage", "event_probability",
]
