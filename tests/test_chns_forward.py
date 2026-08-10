"""tests/test_chns_forward.py — Task 5 (SP-0): CHNSDiscrete numpy mirror.

TDD gates (task-5-brief.md):
  1. test_chns_discrete_jacobian_vs_fd — analytic J @ v vs central FD of the
     coupled residual, rel error < 1e-6 on a level-3 2-D mesh.
  2. test_chns_discrete_stationary_drop — tanh disk, u0=0, gravity OFF:
     after 5 steps max|u| < 1e-3 and mass |sum(phi)-sum(phi0)| flat.
  3. test_chns_discrete_bubble_rises — ratio 10, gravity ON: light bubble
     (phi=-1) centroid_y strictly increases over the last 5 steps.

benchmarks/ is not on sys.path in tests — mirror the bootstrap from
tests/test_chns_scaffold.py.
"""
import os as _os
import sys as _sys

import numpy as np
import pytest

_BENCH_DIR = _os.path.normpath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "benchmarks")
)
if _BENCH_DIR not in _sys.path:
    _sys.path.insert(0, _BENCH_DIR)
_REPO_DIR = _os.path.normpath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..")
)
if _REPO_DIR not in _sys.path:
    _sys.path.insert(0, _REPO_DIR)

from chns.cases import BUBBLE_RISE_RE35_WE10  # noqa: E402
from chns import metrics  # noqa: E402


# ---------------------------------------------------------------------------
# Mesh helper (uniform p=1, constraints.T == identity)
# ---------------------------------------------------------------------------
def _make_dm(level, dim=2):
    import warp as wp
    wp.init()
    from diffsim.octree.build import build_uniform
    from diffsim.mesh.nodes import build_mesh
    from diffsim.mesh.constraints import build_constraints
    from diffsim.mesh.basis import basis_tables
    from diffsim.assembly.operators import DeviceMesh
    from diffsim import default_device

    tree = build_uniform(level, dim=dim)
    mesh = build_mesh(tree, p=1)
    cons = build_constraints(mesh)
    device = default_device()
    dm = DeviceMesh.from_mesh(mesh, cons, basis_tables(1, dim=dim), device)
    return dm, mesh, cons


# ---------------------------------------------------------------------------
# Step 1: analytic Jacobian vs central finite differences
# ---------------------------------------------------------------------------
def test_chns_discrete_jacobian_vs_fd():
    from diffsim.adjoint.chns import CHNSDiscrete

    dm, mesh, cons = _make_dm(level=3, dim=2)
    op = CHNSDiscrete(level=3, dim=2, case=BUBBLE_RISE_RE35_WE10, dt=1e-2, dm=dm)

    rng = np.random.default_rng(0)
    coords = mesh.node_coords
    # smooth admissible phi in [-0.9, 0.9]
    phi = 0.9 * np.tanh(
        (np.sqrt((coords[:, 0] - 0.5) ** 2 + (coords[:, 1] - 0.5) ** 2) - 0.25)
        / (2.0 * op.h)
    )
    n = op.nn
    dim = op.dim
    x = np.zeros(op.ndof)
    # small velocities, admissible phi, zero p, mu from a small field
    for d in range(dim):
        x[d::op.blk] = 1e-3 * rng.standard_normal(n)
    x[dim::op.blk] = 0.0                       # p
    x[dim + 1::op.blk] = phi                   # phi
    x[dim + 2::op.blk] = 1e-2 * rng.standard_normal(n)  # mu

    # history: previous state = current phi, u=0 (only needed by residual)
    op.set_history(u_n=np.zeros((n, dim)), phi_n=phi.copy())

    R0 = op.residual(x)
    J = op.jacobian(x)

    eps = 1e-6
    for _ in range(3):
        v = rng.standard_normal(op.ndof)
        v /= np.linalg.norm(v)
        Rp = op.residual(x + eps * v)
        Rm = op.residual(x - eps * v)
        fd = (Rp - Rm) / (2.0 * eps)
        Jv = J @ v
        rel = np.linalg.norm(Jv - fd) / max(np.linalg.norm(Jv), 1e-30)
        assert rel < 1e-6, f"Jacobian vs FD rel error {rel:.3e} >= 1e-6"


# ---------------------------------------------------------------------------
# Step 2: stationary drop (gravity off) — parasitic bound + mass flat
# ---------------------------------------------------------------------------
def test_chns_discrete_stationary_drop():
    from diffsim.adjoint.chns import CHNSDiscrete

    level = 5
    dm, mesh, cons = _make_dm(level=level, dim=2)
    op = CHNSDiscrete(level=level, dim=2, case=BUBBLE_RISE_RE35_WE10,
                      dt=1e-3, dm=dm, gravity=False, Cn_override="2h")

    coords = mesh.node_coords
    r = np.sqrt((coords[:, 0] - 0.5) ** 2 + (coords[:, 1] - 0.5) ** 2)
    phi0 = -np.tanh((r - 0.25) / (op.Cn * np.sqrt(2.0)))
    op.set_initial(phi0)

    M = op.lumped_mass()
    mass0 = float(M @ phi0)

    for _ in range(5):
        op.step()

    umax = float(np.abs(op.u).max())
    mass1 = float(M @ op.phi)
    assert umax < 1e-3, f"parasitic current max|u|={umax:.3e} >= 1e-3"
    assert abs(mass1 - mass0) < 1e-12 * op.nn, (
        f"mass drift {abs(mass1 - mass0):.3e} >= 1e-12*n"
    )


# ---------------------------------------------------------------------------
# Step 3: bubble rise — light bubble centroid strictly increases
# ---------------------------------------------------------------------------
def test_chns_discrete_bubble_rises():
    from diffsim.adjoint.chns import CHNSDiscrete

    level = 5
    dm, mesh, cons = _make_dm(level=level, dim=2)
    op = CHNSDiscrete(level=level, dim=2, case=BUBBLE_RISE_RE35_WE10,
                      dt=BUBBLE_RISE_RE35_WE10.dt0, dm=dm, gravity=True,
                      Cn_override="2h")

    coords = mesh.node_coords
    r = np.sqrt((coords[:, 0] - 0.5) ** 2 + (coords[:, 1] - 0.35) ** 2)
    phi0 = -np.tanh((r - 0.2) / (op.Cn * np.sqrt(2.0)))  # phi=-1 light bubble
    op.set_initial(phi0)

    cy = []
    for _ in range(10):
        op.step()
        cy.append(metrics.centroid_y(-op.phi, coords))  # bubble = phi<0
    cy = np.asarray(cy)
    last = cy[-5:]
    assert np.all(np.diff(last) > 0), (
        f"bubble centroid_y not strictly increasing over last 5 steps: {last}"
    )


# ---------------------------------------------------------------------------
# Task 6: staggered CH -> NS projection prototype smoke
# ---------------------------------------------------------------------------
def test_staggered_bubble_smoke():
    """CHNSStaggeredStepper, BUBBLE_RISE_RE35_WE10 at level 5 (32x32),
    rho_ratio pinned to 10 for the smoke (the case value is already 10;
    pinned via dataclasses.replace so a future case edit cannot silently
    change this gate), 20 steps at case dt0:
      - no NaN in phi/u/p at any snapshot,
      - |mass drift| printed and < 1e-6 (staggered is NOT exactly
        conservative: convective-form CH advection + the split — the
        loose bound is the gate, the actual value is spike evidence),
      - light bubble (phi=-1) centroid strictly rising over the last 5
        steps.
    Cn_override="2h" mirrors the CHNSDiscrete twin tests above (case
    Cn=0.01 is below the level-5 h=1/32; the sibling monolithic gates
    use the same 2h resolvability convention)."""
    from dataclasses import replace
    from diffsim.steppers.chns import CHNSStaggeredStepper

    level = 5
    dm, mesh, cons = _make_dm(level=level, dim=2)
    case = replace(BUBBLE_RISE_RE35_WE10, rho_ratio=10.0)
    st = CHNSStaggeredStepper(dm, case, dt=case.dt0, Cn_override="2h")

    xc = mesh.node_coords[cons.free_nodes]
    r = np.sqrt((xc[:, 0] - 0.5) ** 2 + (xc[:, 1] - 0.35) ** 2)
    phi0 = np.tanh((r - 0.2) / (st.Cn * np.sqrt(2.0)))  # phi=-1 light bubble
    st.set_initial(phi0)

    mass0 = st.mass_phi()
    snaps = st.march(t_end=20 * case.dt0)
    assert len(snaps) == 20

    cy = []
    for s in snaps:
        assert np.isfinite(s["phi"]).all(), f"NaN phi at t={s['t']}"
        assert np.isfinite(s["u"]).all(), f"NaN u at t={s['t']}"
        assert np.isfinite(s["p"]).all(), f"NaN p at t={s['t']}"
        cy.append(metrics.centroid_y(-s["phi"], xc))  # bubble = phi<0

    drift = abs(st.mass_phi() - mass0)
    wall = float(np.mean([s["wall_per_step"] for s in snaps]))
    print(f"[staggered smoke] |mass drift| = {drift:.3e} (bound 1e-6), "
          f"wall/step = {wall:.3f} s, "
          f"newton_iters(last) = {st.last_newton_iters}, "
          f"clamped(last) = {st.last_clamped}")
    assert drift < 1e-6, f"|mass drift| {drift:.3e} >= 1e-6"

    last = np.asarray(cy[-5:])
    assert np.all(np.diff(last) > 0), (
        f"bubble centroid_y not strictly rising over last 5 steps: {last}"
    )
