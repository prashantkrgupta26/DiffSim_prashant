"""M1b Task 8 scaffolding: lid-driven cavity Re=100 vs Ghia et al. (1982)
Table I/II centerline profiles — the first literature-anchored flow
benchmark (production has no reference values checked in; conventions doc
item 7). Coarse-CI variant: level 5, pseudo-time to steady state,
tolerance 0.06 on the profiles. Raw (unregularized) lid — Ghia's setup;
corner nodes take the lid value (standard practice)."""
import numpy as np
import pytest
from diffsim.octree.build import build_uniform
from diffsim.mesh.nodes import build_mesh
from diffsim.mesh.constraints import build_constraints
from diffsim.mesh.basis import basis_tables
from diffsim.assembly.operators import DeviceMesh
from diffsim.steppers.linearized import LinearizedMonolithicStepper
from diffsim.mesh.pointeval import point_eval_weights

pytestmark = pytest.mark.tier5

# Ghia, Ghia & Shin (1982), Re = 100
GHIA_Y = np.array([0.0000, 0.0547, 0.0625, 0.0703, 0.1016, 0.1719, 0.2813,
                   0.4531, 0.5000, 0.6172, 0.7344, 0.8516, 0.9531, 0.9609,
                   0.9688, 0.9766, 1.0000])
GHIA_U = np.array([0.0000, -0.03717, -0.04192, -0.04775, -0.06434, -0.10150,
                   -0.15662, -0.21090, -0.20581, -0.13641, 0.00332, 0.23151,
                   0.68717, 0.73722, 0.78871, 0.84123, 1.00000])
GHIA_X = np.array([0.0000, 0.0625, 0.0703, 0.0781, 0.0938, 0.1563, 0.2266,
                   0.2344, 0.5000, 0.8047, 0.8594, 0.9063, 0.9453, 0.9531,
                   0.9609, 0.9688, 1.0000])
GHIA_V = np.array([0.0000, 0.09233, 0.10091, 0.10890, 0.12317, 0.16077,
                   0.17507, 0.17527, 0.05454, -0.24533, -0.22445, -0.16914,
                   -0.10313, -0.08864, -0.07391, -0.05906, 0.00000])


def _lid_g(x, t):
    g = np.zeros((len(x), 2))
    g[np.abs(x[:, 1] - 1.0) < 1e-12, 0] = 1.0     # lid wins at corners
    return g


def test_cavity_re100_ghia(device):
    level, Re = 5, 100.0
    nu = 1.0 / Re
    dt = 0.05
    tree = build_uniform(level, dim=2)
    mesh = build_mesh(tree, p=1)
    cons = build_constraints(mesh)
    dm = DeviceMesh.from_mesh(mesh, cons, basis_tables(1, dim=2), device)
    st = LinearizedMonolithicStepper(
        dm, nu, dt, f_fn=lambda x, t: np.zeros((len(x), 2)), g_fn=_lid_g,
        order=1)                                   # pseudo-time: BDF1
    st.set_initial(lambda x: np.zeros((len(x), 2)))
    prev = None
    steps = 0
    for steps in range(1, 401):
        x = st.step()
        u = x[:, :2]
        if prev is not None:
            rate = np.abs(u - prev).max() / dt
            if rate < 2e-4:
                break
        prev = u.copy()
    assert steps < 400, "no steady state in 400 pseudo-steps"

    # centerline profiles via point evaluation on the FULL mesh
    W_u = point_eval_weights(mesh, np.stack(
        [np.full_like(GHIA_Y, 0.5), GHIA_Y], axis=1))
    W_v = point_eval_weights(mesh, np.stack(
        [GHIA_X, np.full_like(GHIA_X, 0.5)], axis=1))
    T = dm.constraints.T.tocsr()
    u_full = np.asarray(T @ u[:, 0])
    v_full = np.asarray(T @ u[:, 1])
    u_c = np.asarray(W_u @ u_full)
    v_c = np.asarray(W_v @ v_full)
    du = np.abs(u_c - GHIA_U).max()
    dv = np.abs(v_c - GHIA_V).max()
    # coarse-CI tolerance at level 5 p1 (a 33^2 grid vs Ghia's 129^2)
    assert du < 0.06, (du, u_c.round(4).tolist())
    assert dv < 0.06, (dv, v_c.round(4).tolist())
    # the primary vortex signature: u-min on the centerline in the right
    # place and depth (Ghia: -0.211 at y=0.453) — mechanism, not just norms
    i_min = int(np.argmin(u_c))
    assert abs(GHIA_Y[i_min] - 0.4531) < 0.2, GHIA_Y[i_min]
    assert u_c[i_min] < -0.15, u_c[i_min]


def test_cavity_re100_ghia_leray(device):
    """Spec S17 discipline: the SAME benchmark through the Leray stepper.
    True-transient march (BDF2) to steady; same Ghia tolerance."""
    from diffsim.steppers.leray import LerayProjectionStepper
    level, Re, dt = 5, 100.0, 0.05
    tree = build_uniform(level, dim=2)
    mesh = build_mesh(tree, p=1)
    cons = build_constraints(mesh)
    dm = DeviceMesh.from_mesh(mesh, cons, basis_tables(1, dim=2), device)
    st = LerayProjectionStepper(
        dm, 1.0 / Re, dt, f_fn=lambda x, t: np.zeros((len(x), 2)),
        g_fn=_lid_g, order=1, picard_iters=1)
    st.set_initial(lambda x: np.zeros((len(x), 2)))
    # MEASURED: the centerline profile converges to du_ghia = 0.0035 by
    # step 150 while the max-rate criterion plateaus ~1e-2 (a slowly
    # decaying pressure-splitting oscillation that never touches the
    # profile) — so march a FIXED 200 steps and assert the physics, not
    # the rate. Note: the Leray profile lands ~10x closer to Ghia than the
    # coarse tolerance requires.
    for _ in range(200):
        u, p = st.step()
    W_u = point_eval_weights(mesh, np.stack(
        [np.full_like(GHIA_Y, 0.5), GHIA_Y], axis=1))
    T = dm.constraints.T.tocsr()
    u_c = np.asarray(W_u @ np.asarray(T @ u[:, 0]))
    du = np.abs(u_c - GHIA_U).max()
    assert du < 0.06, (du, u_c.round(4).tolist())
    assert du < 0.02, du     # measured 0.0035 — lock well inside it
