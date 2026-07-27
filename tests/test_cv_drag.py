"""Analytic gates for the control-volume drag observable (spec 2026-07-26)."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))

from diffsim.octree.build import build_uniform
from diffsim.mesh.nodes import build_mesh
from diffsim.postproc.cv_drag import cv_drag_box

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
