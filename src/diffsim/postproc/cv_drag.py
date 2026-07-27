"""Control-volume drag observable (2-D) — spec 2026-07-26-leakdrag-discriminator.

Instantaneous box momentum balance from NODAL fields, SBM-independent:
    Cd = -∮ [ u_x (u·n) + p n_x - nu du_x/dn ] dS / (0.5 U² L_ref)
Faces are axis-aligned lines sampled at nodal locations (trapezoid at the
local node pitch); viscous normal derivative by one-sided difference to the
parallel node line one local pitch inside the box. rho = 1 (code units).
"""
import numpy as np

_TOLF = 0.25  # face-matching tolerance as a fraction of the local pitch


def _line_nodes(coords, axis, value, lo, hi, tol):
    """Sorted node indices on the line coords[axis]==value within [lo,hi]."""
    on = np.abs(coords[:, axis] - value) < tol
    span = (coords[:, 1 - axis] >= lo - tol) & (coords[:, 1 - axis] <= hi + tol)
    idx = np.where(on & span)[0]
    order = np.argsort(coords[idx, 1 - axis])
    return idx[order]


def _face_integral(coords, u, p, nu, axis, value, lo, hi, n_sign):
    """∫ [u_x(u·n) + p n_x - nu du_x/dn] ds over one axis-aligned face.

    axis=0: vertical face x=value, n=(n_sign,0), integrate over y in [lo,hi].
    axis=1: horizontal face y=value, n=(0,n_sign), integrate over x.
    """
    # local pitch from the nearest node spacing on the face
    probe = _line_nodes(coords, axis, value, lo, hi, tol=1e-6)
    if len(probe) < 2:
        raise ValueError(f"no node line at axis{axis}={value} — box edges must "
                         "lie on mesh node lines")
    s = coords[probe, 1 - axis]
    h = np.min(np.diff(s))
    tol = _TOLF * h
    idx = _line_nodes(coords, axis, value, lo, hi, tol)
    s = coords[idx, 1 - axis]
    ux = u[idx, 0]
    un = u[idx, axis] * n_sign          # u·n on this face
    pn = p[idx] * (n_sign if axis == 0 else 0.0)   # p n_x
    # du_x/dn: one-sided toward the inside line at value - n_sign*h
    inner = _line_nodes(coords, axis, value - n_sign * h, lo, hi, tol)
    ux_in = np.interp(s, coords[inner, 1 - axis], u[inner, 0])
    duxdn = (ux - ux_in) / h
    integrand = ux * un + pn - nu * duxdn
    return np.trapezoid(integrand, s)


def cv_drag_box(coords, u, p, box, nu, U_inf=1.0, L_ref=1.0 / 16.0):
    """Instantaneous CV drag coefficient for rectangular `box`=(x0,x1,y0,y1)."""
    x0, x1, y0, y1 = box
    total = 0.0
    total += _face_integral(coords, u, p, nu, 0, x1, y0, y1, +1)  # right
    total += _face_integral(coords, u, p, nu, 0, x0, y0, y1, -1)  # left
    total += _face_integral(coords, u, p, nu, 1, y1, x0, x1, +1)  # top
    total += _face_integral(coords, u, p, nu, 1, y0, x0, x1, -1)  # bottom
    return -total / (0.5 * U_inf ** 2 * L_ref)
