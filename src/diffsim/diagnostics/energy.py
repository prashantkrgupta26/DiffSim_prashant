"""Free-energy diagnostics — the Ginzburg-Landau energy budget and its split.

The total free energy of the coupled Cahn-Hilliard / Allen-Cahn film is

    F = F_bulk + F_grad + F_wall + F_cryst + F_coupling

with (per unit volume, integrated by quadrature)

    f_bulk     homogeneous mixing energy      f(phi)            [energy/vol]
    f_grad     composition gradient penalty   (kappa/2)|grad phi|^2
    f_wall     substrate wall energy          surface integral  [energy/area]
    f_cryst    crystalline order energy       g(psi) + (eps^2/2)|grad psi|^2
    f_coupling CH<->AC coupling energy        h(phi, psi)

For a Lyapunov (gradient-flow) system F can only DECREASE in the continuous
problem; the *discrete* energy may show a bounded start-up transient (e.g. a
one-step ``mu_init="consistent"`` correction). This module reports the split
and the stepwise increments so a student can separate "continuous Lyapunov"
from "discrete energy stability" from "one monotone run" (spec P1).

Units
-----
Every ``F_*`` is an integrated energy: ``[energy density] * length**dim`` for
volume terms, ``[energy density] * length**(dim-1)`` for the wall term.
"""
from __future__ import annotations

import numpy as np


def integrate_density(density_gp, weights):
    """``INT rho dV`` for an energy density sampled at quadrature points.

    Parameters
    ----------
    density_gp : array_like
        Energy density at quadrature points. Units: ``energy / length**dim``.
    weights : array_like
        Quadrature weights (``length**dim``).

    Returns
    -------
    float
        Integrated energy. Units: ``energy``.
    """
    d = np.asarray(density_gp, dtype=float).ravel()
    w = np.asarray(weights, dtype=float).ravel()
    if d.shape != w.shape:
        raise ValueError("density/weights length mismatch")
    return float(np.dot(w, d))


def gradient_energy(grad_phi_gp, weights, kappa):
    """Interfacial (gradient) energy ``INT (kappa/2)|grad phi|^2 dV``.

    Parameters
    ----------
    grad_phi_gp : array_like, shape (npts, dim)
        Gradient of the field at quadrature points.
    weights : array_like, shape (npts,)
        Quadrature weights (``length**dim``).
    kappa : float
        Gradient-energy coefficient. Units set the interface width
        ``ell ~ sqrt(kappa / W)`` (spec P1: ~sqrt(kappa/W), NOT sqrt(kappa)).

    Returns
    -------
    float
        Non-negative interfacial energy. Units: ``energy``.
    """
    g = np.asarray(grad_phi_gp, dtype=float)
    if g.ndim == 1:
        g = g[:, None]
    dens = 0.5 * float(kappa) * np.sum(g * g, axis=1)
    return integrate_density(dens, weights)


def energy_split(densities, weights):
    """Assemble a labelled energy budget from named density arrays.

    Parameters
    ----------
    densities : dict[str, array_like]
        Named energy densities at quadrature points, e.g.
        ``{"bulk": ..., "grad": ..., "wall": ..., "cryst": ..., "coupling": ...}``.
        Boundary/wall terms must be passed already restricted to their own
        (surface) quadrature; pass matching ``weights`` via a nested dict, or
        pre-integrate and use :func:`assemble_total`.
    weights : array_like
        Quadrature weights shared by the volume densities.

    Returns
    -------
    dict[str, float]
        Each named integrated energy plus ``"total"`` = their sum.
    """
    out = {}
    for name, dens in densities.items():
        out[name] = integrate_density(dens, weights)
    out["total"] = float(sum(out.values()))
    return out


def assemble_total(**components):
    """Sum pre-integrated energy components into a budget dict with ``total``.

    Example
    -------
    >>> assemble_total(bulk=1.0, grad=0.5)["total"]
    1.5
    """
    out = {k: float(v) for k, v in components.items()}
    out["total"] = float(sum(out.values()))
    return out


def stepwise_increment(energy_series):
    """Per-step change ``F[k] - F[k-1]`` of an energy trajectory.

    Returns
    -------
    numpy.ndarray
        Length ``len(series)-1``. For a discretely energy-stable scheme these
        are all ``<= 0`` (to round-off); the **largest positive** increment is
        the honest measure of any violation (spec P1).
    """
    f = np.asarray(energy_series, dtype=float).ravel()
    return np.diff(f)


def largest_positive_increment(energy_series):
    """Max stepwise energy increase (``0.0`` if the run is monotone).

    Units: ``energy``. A positive value is the size of the worst violation of
    discrete energy decrease — report it rather than a boolean.
    """
    inc = stepwise_increment(energy_series)
    if inc.size == 0:
        return 0.0
    return float(max(0.0, inc.max()))


def is_monotone_decreasing(energy_series, skip=1, atol=1e-9):
    """Whether ``F`` decreases monotonically after ``skip`` start-up steps.

    Parameters
    ----------
    skip : int
        Number of leading steps to ignore (the ``mu_init`` start-up transient).
    atol : float
        Absolute slack: increments up to ``atol`` count as non-increasing.
    """
    f = np.asarray(energy_series, dtype=float).ravel()
    if f.size <= skip + 1:
        return True
    return bool(np.all(np.diff(f[skip:]) <= atol))


__all__ = [
    "integrate_density", "gradient_energy", "energy_split", "assemble_total",
    "stepwise_increment", "largest_positive_increment", "is_monotone_decreasing",
]
