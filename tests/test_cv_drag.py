"""Analytic gates for the control-volume drag observable (spec 2026-07-26)."""
import os, sys
import numpy as np
import pytest
sys.path.insert(0, os.path.dirname(__file__))

from diffsim.octree.build import build_uniform
from diffsim.mesh.nodes import build_mesh
from diffsim.postproc.cv_drag import cv_drag_box, _face_integral

BOX = (0.25, 0.75, 0.25, 0.75)

def _mesh_coords(level=5):
    mesh = build_mesh(build_uniform(level, dim=2), p=1)
    return np.asarray(mesh.node_coords)

def test_uniform_freestream_zero_drag():
    """u=(U,0), p=0: every flux contribution cancels; Cd must be ~0."""
    coords = _mesh_coords()
    u = np.zeros((len(coords), 2)); u[:, 0] = 1.0
    p = np.zeros(len(coords))
    cd = cv_drag_box(coords, u, p, BOX, nu=2.5e-4)
    assert abs(cd) < 1e-10, f"uniform flow gave Cd={cd}"

def test_analytic_momentum_flux():
    """u=(x,-y) (div-free), p=0, nu=0 on box (x0,x1,y0,y1):
      ∮u_x(u·n)dS = x1²Δy − x0²Δy − Δ(x²)/2·y1 + Δ(x²)/2·y0
    with Δy=y1−y0, Δ(x²)=x1²−x0². Cd = −that / (0.5·U²·L_ref)."""
    coords = _mesh_coords()
    u = np.stack([coords[:, 0], -coords[:, 1]], axis=1)
    p = np.zeros(len(coords))
    x0, x1, y0, y1 = BOX
    dy, dx2 = (y1 - y0), (x1**2 - x0**2)
    flux = x1**2 * dy - x0**2 * dy - 0.5 * dx2 * y1 + 0.5 * dx2 * y0
    expect = -flux / (0.5 * 1.0**2 * (1.0 / 16.0))
    cd = cv_drag_box(coords, u, p, BOX, nu=0.0)
    assert np.isclose(cd, expect, rtol=1e-6), f"{cd} vs {expect}"

def test_pressure_term():
    """u=0, p=x, nu=0: ∮p·n_x dS = (x1−x0)·... → p(x1)Δy − p(x0)Δy = Δx·Δy? No:
    right face contributes +p(x1)Δy = x1Δy, left −x0Δy → net (x1−x0)Δy."""
    coords = _mesh_coords()
    u = np.zeros((len(coords), 2))
    p = coords[:, 0].copy()
    x0, x1, y0, y1 = BOX
    expect = -((x1 - x0) * (y1 - y0)) / (0.5 * (1.0 / 16.0))
    cd = cv_drag_box(coords, u, p, BOX, nu=0.0)
    assert np.isclose(cd, expect, rtol=1e-6), f"{cd} vs {expect}"

def test_viscous_term_all_faces_dense_reference():
    """u_x = x*y, u_y = 0, p = 0, nu = 0.01: the viscous normal derivative
    is nonzero and VARYING on all four faces (du_x/dn = y*n_sign on vertical
    faces, x*n_sign on horizontal). u_x is bilinear, so the implementation's
    one-sided nodal difference is EXACT. Reference: genuine 2001-point
    trapezoid quadrature per face of the analytic integrand
    [u_x*(u.n) + p*n_x - nu*du_x/dn], summed, negated, normalized —
    re-implementing the INTEGRAL definition, never calling cv_drag code."""
    coords = _mesh_coords()
    u = np.zeros((len(coords), 2))
    u[:, 0] = coords[:, 0] * coords[:, 1]
    p = np.zeros(len(coords))
    nu = 0.01
    x0, x1, y0, y1 = BOX

    def ref_face(axis, value, lo, hi, n_sign):
        """Trapezoid quadrature of integrand [u_x*(u·n) + p*n_x - nu*du_x/dn]
        along a face. axis=0 for vertical (x=value), axis=1 for horizontal (y=value).
        """
        s = np.linspace(lo, hi, 2001)
        if axis == 0:   # vertical face x=value: u.n = u_x*n_sign, du_x/dn = y*n_sign
            ux = value * s          # u_x = x*y with x=value, y=s
            un = ux * n_sign        # u_y=0; u.n = u_x*n_sign
            pn = 0.0 * s            # p=0
            duxdn = s * n_sign      # d(x*y)/dx = y, outward = *n_sign
        else:           # horizontal face y=value: u.n = u_y*n_sign = 0, du_x/dn = x*n_sign
            ux = s * value          # u_x = x*y with x=s, y=value
            un = 0.0 * s            # u_y=0; u.n = 0
            pn = 0.0 * s            # p=0
            duxdn = s * n_sign      # d(x*y)/dy = x, outward = *n_sign
        return np.trapezoid(ux * un + pn - nu * duxdn, s)

    total = (ref_face(0, x1, y0, y1, +1) + ref_face(0, x0, y0, y1, -1)
             + ref_face(1, y1, x0, x1, +1) + ref_face(1, y0, x0, x1, -1))
    expect = -total / (0.5 * 1.0**2 * (1.0 / 16.0))
    cd = cv_drag_box(coords, u, p, BOX, nu=nu)
    assert np.isclose(cd, expect, rtol=1e-3, atol=1e-3), f"{cd} vs {expect}"

def test_missing_interior_line_raises():
    """Interior node line missing at distance h inside face edge: detect
    when CV box edge sits outside the mesh such that inner line doesn't exist."""
    coords = _mesh_coords()
    u = np.zeros((len(coords), 2))
    p = np.zeros(len(coords))

    # Call _face_integral with a value that forces inner line search outside mesh
    # Find the mesh bounds
    x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
    y_min, y_max = coords[:, 1].min(), coords[:, 1].max()

    # axis=0 (vertical face), value slightly outside, n_sign=-1 so inner is
    # at value - (-1)*h = value + h — if value is already at/past boundary,
    # inner line is unreachable
    with pytest.raises(ValueError, match="interior node line"):
        _face_integral(coords, u, p, nu=0.01, axis=0, value=x_max,
                       lo=y_min, hi=y_max, n_sign=-1)

