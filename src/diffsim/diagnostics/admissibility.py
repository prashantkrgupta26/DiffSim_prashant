"""Admissibility diagnostics — bound and simplex constraints on order fields.

Phase-field composition variables must stay physically admissible:
Flory-Huggins fractions in ``(0, 1)``, a ternary blend on the Gibbs simplex
``sum_i phi_i = 1`` with ``phi_i >= 0``. The tutorials must *report* how far
the discrete solution wanders and how much projection/clipping was applied
(spec P1/P4: report ``phi_min/max``, ``projected_dofs``, ``max_correction``),
rather than silently clip. These are dimensionless.
"""
from __future__ import annotations

import numpy as np


def field_bounds(field):
    """Min and max of a field. Returns ``(min, max)`` (dimensionless)."""
    f = np.asarray(field, dtype=float)
    return (float(f.min()), float(f.max()))


def bound_violation(field, lo=0.0, hi=1.0):
    """How far a field exceeds ``[lo, hi]``.

    Returns
    -------
    dict
        ``min``, ``max`` of the field; ``below`` = ``max(0, lo - min)``,
        ``above`` = ``max(0, max - hi)`` (magnitudes of the worst excursions);
        ``violating_fraction`` = fraction of entries outside ``[lo, hi]``.
    """
    f = np.asarray(field, dtype=float)
    fmin, fmax = float(f.min()), float(f.max())
    below = max(0.0, lo - fmin)
    above = max(0.0, fmax - hi)
    frac = float(np.mean((f < lo) | (f > hi)))
    return {"min": fmin, "max": fmax, "below": below, "above": above,
            "violating_fraction": frac}


def simplex_residual(fields):
    """Deviation from the Gibbs simplex ``sum_i phi_i = 1``.

    Parameters
    ----------
    fields : sequence of array_like
        Component fields ``phi_i`` (same shape). For a binary system pass the
        one independent field plus its complement, or the full N components.

    Returns
    -------
    dict
        ``max_abs`` and ``rms`` of ``|sum_i phi_i - 1|`` over all points
        (dimensionless). Zero to round-off means the fields lie on the simplex.
    """
    arrs = [np.asarray(f, dtype=float) for f in fields]
    s = sum(arrs)
    resid = np.abs(s - 1.0)
    return {"max_abs": float(resid.max()), "rms": float(np.sqrt(np.mean(resid ** 2)))}


def clipped_fraction(field, lo=0.0, hi=1.0):
    """Fraction of entries that a clip to ``[lo, hi]`` would move.

    Dimensionless in ``[0, 1]``. This is the "projected/clipped fraction" the
    tutorials must report when they enforce admissibility.
    """
    f = np.asarray(field, dtype=float)
    return float(np.mean((f < lo) | (f > hi)))


def projection_report(field, lo=0.0, hi=1.0):
    """Full report for a clip-to-box admissibility projection.

    Returns
    -------
    dict
        ``phi_min``, ``phi_max`` (pre-projection bounds), ``projected_dofs``
        (count of entries moved), ``clipped_fraction`` (dimensionless), and
        ``max_correction`` (largest single-entry displacement).
    """
    f = np.asarray(field, dtype=float)
    clipped = np.clip(f, lo, hi)
    moved = clipped != f
    corr = np.abs(clipped - f)
    return {
        "phi_min": float(f.min()),
        "phi_max": float(f.max()),
        "projected_dofs": int(np.count_nonzero(moved)),
        "clipped_fraction": float(np.mean(moved)),
        "max_correction": float(corr.max()) if corr.size else 0.0,
    }


__all__ = [
    "field_bounds", "bound_violation", "simplex_residual", "clipped_fraction",
    "projection_report",
]
