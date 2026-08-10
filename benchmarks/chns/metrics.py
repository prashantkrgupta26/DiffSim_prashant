"""benchmarks/chns/metrics.py — Hysing et al. 2009 benchmark metrics.

References:
    Hysing, S. et al. (2009). Quantitative benchmark computations of two-dimensional
    bubble dynamics. Int. J. Numer. Meth. Fluids, 60(11), 1259-1288.

Metrics:
    centroid_y    — phi-weighted mean y-coordinate (bubble/drop centroid)
    rise_velocity — finite-difference derivative of centroid_y series
    circularity   — 2*sqrt(pi*A)/P using diffuse-interface perimeter integral
"""

from __future__ import annotations
import numpy as np


def centroid_y(phi: np.ndarray, coords: np.ndarray) -> float:
    """Return the phi-weighted mean y-coordinate over the phi > 0 mask.

    Args:
        phi   : [n] phase field values (phi = +1 inside bubble, -1 outside)
        coords: [n, dim] node coordinates (y is column index 1)

    Returns:
        float: phi-weighted centroid y-coordinate; nan if mask is empty.
    """
    mask = phi > 0.0
    if not mask.any():
        return float("nan")
    w = phi[mask]
    return float(np.average(coords[mask, 1], weights=w))


def rise_velocity(centroid_series: np.ndarray, dt: float) -> np.ndarray:
    """Finite-difference derivative of centroid_y time series.

    Uses central differences for interior points, first-order one-sided
    differences at the endpoints.

    Args:
        centroid_series: [T] array of centroid_y values at uniform time steps.
        dt             : time step between consecutive entries.

    Returns:
        [T] array of rise velocities (d centroid_y / dt).
    """
    series = np.asarray(centroid_series, dtype=float)
    if series.ndim != 1 or series.size < 2:
        raise ValueError("centroid_series must be a 1-D array with >= 2 elements")
    vel = np.empty_like(series)
    # Central difference for interior
    vel[1:-1] = (series[2:] - series[:-2]) / (2.0 * dt)
    # One-sided at endpoints
    vel[0] = (series[1] - series[0]) / dt
    vel[-1] = (series[-1] - series[-2]) / dt
    return vel


def circularity(phi: np.ndarray, coords: np.ndarray, h: float) -> float:
    """Compute circularity from the diffuse phase field on a regular grid.

    Circularity = (perimeter of equal-area circle) / (measured perimeter)
               = 2 * sqrt(pi * A) / P

    Area:
        A = count(phi > 0) * h^2   (cell-count area, phi > 0 mask)

    Perimeter (diffuse-interface form, Hysing et al.):
        P = sum(|grad phi|) * h^2 / 2
        The 1/2 factor accounts for the tanh profile spanning [-1, 1]
        (the integral of |tanh'| over R equals 2, so the diffuse perimeter
        integral sums to 2 * physical perimeter; dividing by 2 recovers P).

    Gradient is computed with numpy.gradient on a reconstructed 2-D grid.
    Coords must lie on a uniform structured mesh; this function infers the
    grid by sorting unique x- and y-values.

    Args:
        phi   : [n] phase field values
        coords: [n, 2] node coordinates
        h     : grid spacing (uniform, isotropic)

    Returns:
        float: circularity in (0, 1] for convex shapes; > 1 is unphysical
               (indicates numerical error or non-convex interface).
    """
    phi = np.asarray(phi, dtype=float)
    coords = np.asarray(coords, dtype=float)

    # --- Area: cell-count * h^2 ---
    area = float(np.sum(phi > 0.0)) * h ** 2

    if area == 0.0:
        return float("nan")

    # --- Perimeter: diffuse-interface |grad phi| integral ---
    # Reconstruct 2-D grid from scattered (x, y, phi) data
    xs = np.unique(np.round(coords[:, 0], decimals=12))
    ys = np.unique(np.round(coords[:, 1], decimals=12))
    nx, ny = len(xs), len(ys)

    # Map each point to (ix, iy)
    ix = np.searchsorted(xs, np.round(coords[:, 0], decimals=12))
    iy = np.searchsorted(ys, np.round(coords[:, 1], decimals=12))

    phi_grid = np.zeros((nx, ny))
    phi_grid[ix, iy] = phi

    # Central-difference gradient on the 2-D grid
    dphi_dx, dphi_dy = np.gradient(phi_grid, h, h)
    grad_mag = np.sqrt(dphi_dx ** 2 + dphi_dy ** 2)

    # Diffuse-interface perimeter (factor 1/2 for tanh [-1,1] profile)
    perimeter = np.sum(grad_mag) * h ** 2 / 2.0

    if perimeter == 0.0:
        return float("nan")

    return 2.0 * np.sqrt(np.pi * area) / perimeter
