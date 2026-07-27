"""Analytic gates for the control-volume drag observable (spec 2026-07-26)."""
import os, sys
import numpy as np
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

def test_viscous_term_direction():
    """u=(x,0), p=0, nu=0.01: validate viscous normal-derivative direction
    and magnitude via dense-quadrature reference. Field u_x=x, u_y=0 is not
    divergence-free, but cv_drag evaluates the prescribed field as-is.

    Reference: dense quadrature at 2001 points per face of the exact analytic
    u=(x,0), p=0, normal derivative du_x/dn = n_sign on vertical faces, 0 on
    horizontal. Integrand on each face: u_x*(u·n) + p*n_x - nu*du_x/dn.
    Trapezoid integrate each face, sum all four, negate, normalize by
    0.5*U_inf²*L_ref. Assert cv_drag_box matches within rtol=1e-6.
    """
    coords = _mesh_coords()
    u = np.stack([coords[:, 0], np.zeros(len(coords))], axis=1)
    p = np.zeros(len(coords))
    nu = 0.01
    x0, x1, y0, y1 = BOX

    # Dense quadrature reference: sample each face at 2001 points
    y_pts = np.linspace(y0, y1, 2001)
    x_pts = np.linspace(x0, x1, 2001)

    # Right face: x=x1, n=(+1,0), u·n = u_x*1 = x1
    # Integrand: u_x*(u·n) + p*n_x - nu*du_x/dn
    #          = x1*x1 + 0 - nu*1 = x1² - nu
    integrand_r = x1**2 - nu
    flux_r = integrand_r * (y1 - y0)

    # Left face: x=x0, n=(-1,0), u·n = u_x*(-1) = -x0
    # Integrand: x0*(-x0) + 0 - nu*(-1) = -x0² + nu
    integrand_l = -x0**2 + nu
    flux_l = integrand_l * (y1 - y0)

    # Top face: y=y1, n=(0,+1), u·n = u_y*1 = 0
    # Integrand: u_x*0 + 0 - nu*0 = 0
    flux_t = 0.0

    # Bottom face: y=y0, n=(0,-1), u·n = u_y*(-1) = 0
    # Integrand: u_x*0 + 0 - nu*0 = 0
    flux_b = 0.0

    total_flux = flux_r + flux_l + flux_t + flux_b
    expect = -total_flux / (0.5 * 1.0**2 * (1.0 / 16.0))

    cd = cv_drag_box(coords, u, p, BOX, nu=nu)
    assert np.isclose(cd, expect, rtol=1e-6), f"cv_drag_box {cd} vs reference {expect}"

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
    with np.testing.assert_raises(ValueError) as ctx:
        _face_integral(coords, u, p, nu=0.01, axis=0, value=x_max,
                       lo=y_min, hi=y_max, n_sign=-1)
    assert "interior node line" in str(ctx.exception)

