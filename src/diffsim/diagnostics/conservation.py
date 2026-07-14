"""Conservation diagnostics — mass, component content, boundary flux, and
moving-domain balances.

All integrals are computed by **quadrature** (weighted sums over Gauss
points), never by nodal averaging, so the numbers match what the assembly
actually integrates (spec P3: "QUADRATURE mass, not nodal-mean-vs-nominal").

Units convention
----------------
Fields are dimensionless order parameters unless stated; quadrature weights
``w`` carry the physical volume element (length**dim), so every integral
below has units of ``[field] * length**dim`` (a *content*, e.g. an amount of
material) unless divided by a volume to give a mean. Callers are responsible
for supplying weights in consistent physical units.
"""
from __future__ import annotations

import numpy as np


def quadrature_mass(field_gp, weights):
    """Total content ``INT phi dV`` by Gauss-point quadrature.

    Parameters
    ----------
    field_gp : array_like
        Field values sampled at quadrature points (flattened over all
        elements/points). Units: ``[field]``.
    weights : array_like
        Quadrature weights including the geometric Jacobian, same length as
        ``field_gp``. Units: ``length**dim``.

    Returns
    -------
    float
        ``INT field dV``. Units: ``[field] * length**dim``.
    """
    field_gp = np.asarray(field_gp, dtype=float).ravel()
    weights = np.asarray(weights, dtype=float).ravel()
    if field_gp.shape != weights.shape:
        raise ValueError(
            f"field/weights length mismatch: {field_gp.shape} vs "
            f"{weights.shape}")
    return float(np.dot(weights, field_gp))


def domain_volume(weights):
    """Total quadrature volume ``INT dV = sum(weights)``. Units: ``length**dim``."""
    return float(np.sum(np.asarray(weights, dtype=float)))


def mean_field(field_gp, weights):
    """Volume-weighted mean ``INT phi dV / INT dV``. Units: ``[field]``."""
    vol = domain_volume(weights)
    if vol == 0.0:
        raise ValueError("zero quadrature volume")
    return quadrature_mass(field_gp, weights) / vol


def component_content(fields_gp, weights):
    """Per-component content for a multi-field (ternary/N-ary) system.

    Parameters
    ----------
    fields_gp : sequence of array_like
        One field array per component, each sampled at quadrature points.
    weights : array_like
        Quadrature weights (length**dim).

    Returns
    -------
    numpy.ndarray
        ``INT phi_i dV`` for each component ``i``. Units: ``[field]*length**dim``.
    """
    return np.array([quadrature_mass(f, weights) for f in fields_gp])


def mass_drift(mass_series):
    """Absolute change of a conserved quantity over a trajectory.

    Parameters
    ----------
    mass_series : array_like
        Time series of a content (e.g. ``INT phi dV`` per step).

    Returns
    -------
    float
        ``|mass[-1] - mass[0]|``. For conserved (Model-B / no-flux) dynamics
        this should sit at solver/round-off tolerance, NOT bitwise zero.
    """
    m = np.asarray(mass_series, dtype=float).ravel()
    if m.size == 0:
        raise ValueError("empty mass series")
    return float(abs(m[-1] - m[0]))


def relative_mass_drift(mass_series):
    """``|m[-1]-m[0]| / max(|m[0]|, eps)`` — dimensionless drift fraction."""
    m = np.asarray(mass_series, dtype=float).ravel()
    denom = max(abs(m[0]), 1e-300)
    return mass_drift(m) / denom


def boundary_flux(flux_normal_gp, weights_boundary):
    """Net flux through a boundary ``-INT (J . n) dS`` by surface quadrature.

    The sign follows the divergence theorem: for ``phi_t = -div J`` the rate
    of change of interior content equals ``-INT J.n dS`` (outflow decreases
    content). Supply the *outward* normal component ``J.n``.

    Parameters
    ----------
    flux_normal_gp : array_like
        Outward normal flux component ``J . n`` at boundary quadrature points.
        Units: ``[field] * length / time``.
    weights_boundary : array_like
        Surface quadrature weights. Units: ``length**(dim-1)``.

    Returns
    -------
    float
        ``-INT J.n dS`` = d/dt(content). Units: ``[field]*length**dim / time``.
    """
    fn = np.asarray(flux_normal_gp, dtype=float).ravel()
    wb = np.asarray(weights_boundary, dtype=float).ravel()
    if fn.shape != wb.shape:
        raise ValueError("flux/weights length mismatch")
    return -float(np.dot(wb, fn))


def moving_domain_balance(content_series, height_series, flux_series=None,
                          times=None):
    """Residual of a moving-frame (shrinking-film) content balance.

    For an evaporating film of instantaneous height ``h(t)`` the scaled
    content ``h(t) * INT_domain phi dV_hat`` (integral over the fixed unit
    domain) changes only through boundary transfer. This returns the balance
    residual

        R(t_k) = [C(t_k) - C(t_0)] - integral_of_flux,

    where ``C = height * content``. With ``flux_series=None`` it returns the
    raw change ``C(t_k) - C(t_0)`` so a caller can compare against an
    independently measured loss.

    Parameters
    ----------
    content_series : array_like
        ``INT phi dV_hat`` on the fixed reference domain, per step.
    height_series : array_like
        Physical film height ``h(t)`` per step (same length). Units: ``length``.
    flux_series : array_like, optional
        Instantaneous loss rate ``dC/dt`` (e.g. evaporative flux), per step.
    times : array_like, optional
        Times for trapezoidal integration of ``flux_series``.

    Returns
    -------
    numpy.ndarray
        Residual per step (units of ``C``). Should stay near zero (transfer
        tolerance) for a correctly closed balance.
    """
    c = np.asarray(content_series, dtype=float).ravel()
    h = np.asarray(height_series, dtype=float).ravel()
    if c.shape != h.shape:
        raise ValueError("content/height length mismatch")
    scaled = h * c
    change = scaled - scaled[0]
    if flux_series is None:
        return change
    f = np.asarray(flux_series, dtype=float).ravel()
    if times is None:
        times = np.arange(f.size, dtype=float)
    t = np.asarray(times, dtype=float).ravel()
    integ = np.concatenate([[0.0], np.cumsum(
        0.5 * (f[1:] + f[:-1]) * np.diff(t))])
    return change - integ


def transfer_mass_change(content_before, content_after):
    """Content change attributable to a discrete transfer/remesh operation.

    Used to verify a conservative field transfer (AMR coarsen/refine, BDF
    history remap): a mass-conservative transfer has
    ``transfer_mass_change ~ 0`` to interpolation tolerance.

    Returns ``content_after - content_before`` (signed). Units: content.
    """
    return float(np.asarray(content_after) - np.asarray(content_before))


__all__ = [
    "quadrature_mass", "domain_volume", "mean_field", "component_content",
    "mass_drift", "relative_mass_drift", "boundary_flux",
    "moving_domain_balance", "transfer_mass_change",
]
