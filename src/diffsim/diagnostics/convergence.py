"""Convergence diagnostics — observed order of accuracy and Richardson
extrapolation.

Verification (code verification, spec C1) asks: does the discretization error
shrink at the theoretical rate as the mesh/timestep is refined? These helpers
turn a refinement study (errors vs h or dt) into an observed order and a
Richardson-extrapolated reference. Errors are whatever norm the caller
measures (L2/H1 of ``c`` and ``mu``, mass, residual); refinement parameters
share the caller's units.
"""
from __future__ import annotations

import numpy as np


def observed_order(h, errors):
    """Least-squares observed order of accuracy ``p`` from ``error ~ C h^p``.

    Fits ``log(error) = log(C) + p log(h)``.

    Parameters
    ----------
    h : array_like
        Refinement parameter per level (grid spacing or timestep), > 0.
    errors : array_like
        Error norm per level, > 0, same length as ``h``.

    Returns
    -------
    float
        Observed order ``p`` (dimensionless). For a second-order scheme with a
        clean asymptotic range this is ~2.
    """
    h = np.asarray(h, dtype=float).ravel()
    e = np.asarray(errors, dtype=float).ravel()
    if h.shape != e.shape or h.size < 2:
        raise ValueError("need >=2 matched (h, error) points")
    if np.any(h <= 0) or np.any(e <= 0):
        raise ValueError("h and errors must be strictly positive")
    p = np.polyfit(np.log(h), np.log(e), 1)[0]
    return float(p)


def pairwise_orders(h, errors):
    """Local observed order between successive refinement levels.

    ``p_k = log(e_k / e_{k+1}) / log(h_k / h_{k+1})``. Returns an array of
    length ``len(h)-1``; a flat sequence indicates the asymptotic range has
    been reached (dimensionless).
    """
    h = np.asarray(h, dtype=float).ravel()
    e = np.asarray(errors, dtype=float).ravel()
    return np.log(e[:-1] / e[1:]) / np.log(h[:-1] / h[1:])


def richardson_extrapolation(f_coarse, f_fine, refinement_ratio, order):
    """Richardson-extrapolated value from two grids of known order.

    ``f_exact ~ f_fine + (f_fine - f_coarse) / (r^p - 1)``.

    Parameters
    ----------
    f_coarse, f_fine : float or ndarray
        Solution functional on the coarse and fine grids.
    refinement_ratio : float
        ``r = h_coarse / h_fine`` (> 1).
    order : float
        Assumed order ``p`` of the scheme.

    Returns
    -------
    Extrapolated estimate, same shape/units as the inputs.
    """
    r = float(refinement_ratio)
    p = float(order)
    denom = r ** p - 1.0
    if denom == 0:
        raise ValueError("refinement_ratio**order must differ from 1")
    return f_fine + (np.asarray(f_fine) - np.asarray(f_coarse)) / denom


def richardson_error_estimate(f_coarse, f_fine, refinement_ratio, order):
    """Estimated discretization error of the FINE solution via Richardson.

    ``|f_fine - f_exact| ~ |f_fine - f_coarse| / (r^p - 1)``. Same units as the
    functional. A cheap a-posteriori error bar for a verified reference (spec
    C1: "verify the reference — halve ref dt, Richardson").
    """
    r = float(refinement_ratio)
    p = float(order)
    denom = r ** p - 1.0
    return float(np.abs(np.asarray(f_fine) - np.asarray(f_coarse)) / abs(denom))


def gci(f_coarse, f_fine, refinement_ratio, order, safety=1.25):
    """Grid Convergence Index (Roache) — a conservative relative error band.

    ``GCI = safety * |(f_fine - f_coarse)/f_fine| / (r^p - 1)``. Dimensionless
    fraction. Reported as a percentage when multiplied by 100.
    """
    r = float(refinement_ratio)
    p = float(order)
    rel = abs((f_fine - f_coarse) / f_fine) if f_fine != 0 else abs(f_fine - f_coarse)
    return float(safety * rel / (r ** p - 1.0))


__all__ = [
    "observed_order", "pairwise_orders", "richardson_extrapolation",
    "richardson_error_estimate", "gci",
]
