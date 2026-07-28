"""Task A1 — block-diagonal saddle FGMRES baseline (fgmres_bdiag).

Tests:
  1. test_fgmres_bdiag_solves_real_saddle  — RED until saddle_precond.py +
     the "fgmres_bdiag" branch in linsolve.py are implemented.  After that,
     solve_linear(Acsr, b, solver="fgmres_bdiag", ...) must match the splu
     reference to rtol=1e-8 and report iterations > 0.
  2. test_fgmres_bdiag_result_carries_iterations  — same system; the
     return_result=True path must return a LinearSolveResult with iterations > 0.

Assembly helper: replicates EXACTLY what run_flow_past does for ONE BDF1
step at level=4 (the smallest mesh the driver supports that is still a real
saddle system), using the host assembly path.  Returns (Acsr, b, x_splu).
"""
import os
import sys

import numpy as np
import pytest
import scipy.sparse as sp
from scipy.sparse.linalg import splu

sys.path.insert(0, os.path.dirname(__file__))

# ---------------------------------------------------------------------------
# One-step saddle-system builder (mirrors run_flow_past host-assembly path)
# ---------------------------------------------------------------------------

def _one_step_system():
    """Build a real 2-D (u,v,p) saddle system from the thin-plate driver.

    Replicates the host-assembly path of run_flow_past for ONE BDF1 step at
    level=4 (zero initial velocity, t=dt, step=0).  Returns (Acsr, b, x_splu)
    where x_splu is the reference solution from scipy splu.

    Interleaved dof convention: ndof=3, dofs per node are (u_x, u_y, p).
    Velocity dofs: node i -> i*3+0, i*3+1.
    Pressure dof : node i -> i*3+2.
    """
    from diffsim.octree.build import build_uniform
    from diffsim.mesh.nodes import build_mesh
    from diffsim.mesh.constraints import build_constraints
    from diffsim.mesh.basis import basis_tables
    from diffsim.mesh.faces import face_tables
    from diffsim.assembly.operators import DeviceMesh
    from diffsim.geometry.csg import Segment, Plane
    from diffsim.sbm.surrogate import (
        classify_shell_intercepted, extract_two_sided_surrogate)
    from diffsim.sbm.vector import sbm_vector_dirichlet_twosided
    from diffsim.api.ns_bricks import assemble_linear_ns
    from diffsim.solvers.timestepping import bdf_coeffs

    # ---- geometry (mirrors run_flow_past defaults) -------------------------
    level = 4
    dim = 2
    ndof = dim + 1          # = 3: (u_x, u_y, p) per node
    plate_xc, plate_yc, plate_L = 0.375, 0.5, 0.25
    U_inf = 1.0
    nu = 0.1
    alpha = 50.0
    dt = 0.01
    device = "cpu"

    # plate geometry
    a_pt = (plate_xc, plate_yc - plate_L / 2.0)
    b_pt = (plate_xc, plate_yc + plate_L / 2.0)
    segment = Segment(a_pt, b_pt)
    plane = Plane((plate_xc, 0.0), (1.0, 0.0))

    # mesh + surrogate
    tree = build_uniform(level, dim=dim)
    ret, _ = classify_shell_intercepted(tree, segment)
    mesh = build_mesh(ret, p=1)
    cons = build_constraints(mesh)
    dm = DeviceMesh.from_mesh(mesh, cons, basis_tables(1, dim=dim), device)
    (sfp, gp), (sfm, gm) = extract_two_sided_surrogate(ret, plane, face_tables(1, dim))

    # constraint matrices
    T = cons.T.tocsr()
    T_vec = sp.kron(T, sp.identity(ndof, format="csr"), format="csr")
    nfree = T.shape[1]

    # BCs
    coords = mesh.node_coords[cons.free_nodes]
    x_min = coords[:, 0].min()
    y_min = coords[:, 1].min()
    y_max = coords[:, 1].max()
    inflow_mask = np.abs(coords[:, 0] - x_min) < 1e-10
    walls = ((np.abs(coords[:, 1] - y_min) < 1e-10) |
             (np.abs(coords[:, 1] - y_max) < 1e-10))
    forced = inflow_mask | walls
    bc_rows, bc_vals = [], []
    for i in np.where(forced)[0]:
        bc_rows.append(i * ndof + 0); bc_vals.append(U_inf)
        bc_rows.append(i * ndof + 1); bc_vals.append(0.0)
    bc_rows = np.asarray(bc_rows, np.int64)
    bc_vals = np.asarray(bc_vals)

    # pressure pin (outflow-bottom corner)
    x_max = coords[:, 0].max()
    corner = np.argmax(coords[:, 0] - coords[:, 1])
    p_pin = int(corner * ndof + dim)

    # SBM face terms (geometry is fixed — assembled once)
    noslip = lambda y: np.zeros((len(y), dim))
    Af_raw, bf_raw = sbm_vector_dirichlet_twosided(
        dm, sfp, gp, sfm, gm, noslip, nu, ndof, alpha=alpha)
    Af_c = (T_vec.T @ Af_raw @ T_vec).tocsr()
    bf_c = np.asarray(T_vec.T @ bf_raw)

    # Gauss-point field (zero initial velocity -> zero advecting field)
    u_pre1 = np.zeros((nfree, dim))

    # GP fields from zero velocity
    aq, dq = {}, {}
    for pv in dm.bins:
        tb = dm.tables_by_p[pv]
        vals = np.zeros((len(mesh.conn_of[pv]), tb.N.shape[1], dim))
        aq[pv] = np.einsum("qa,ead->eqd", tb.N, vals).reshape(-1, dim)
        dq[pv] = np.zeros(len(aq[pv]))
    fq_raw = {}
    for pv in dm.bins:
        tb = dm.tables_by_p[pv]
        vals = np.zeros((len(mesh.conn_of[pv]), tb.N.shape[1], dim))
        fq_raw[pv] = (np.einsum("qa,ead->eqd", tb.N, vals).reshape(-1, dim) / dt)

    # BDF1 step 0
    b0, b1, b2 = bdf_coeffs(1, dt)
    sigma = b0 / dt

    # ---- host assembly path (mirrors run_flow_past host branch) ------------
    A, b = assemble_linear_ns(dm, aq, dq, fq_raw, nu, sigma=sigma)
    A = (A + Af_c).tolil()
    b = b + bf_c

    # strong BCs
    for r, v in zip(bc_rows, bc_vals):
        A.rows[int(r)] = [int(r)]
        A.data[int(r)] = [1.0]
        b[int(r)] = v

    # pressure pin
    A.rows[p_pin] = [p_pin]
    A.data[p_pin] = [1.0]
    b[p_pin] = 0.0

    Acsr = A.tocsr()
    x_splu = splu(Acsr.tocsc()).solve(b)
    return Acsr, b, x_splu


# ---------------------------------------------------------------------------
# Cache: build system once per session (expensive; level=4 takes ~2 s).
# ---------------------------------------------------------------------------
_SYSTEM_CACHE = {}


def _get_system():
    if "sys" not in _SYSTEM_CACHE:
        _SYSTEM_CACHE["sys"] = _one_step_system()
    return _SYSTEM_CACHE["sys"]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

pytestmark = pytest.mark.tier4


def test_fgmres_bdiag_solves_real_saddle():
    """fgmres_bdiag must solve the real 2-D saddle to rtol=1e-8 vs splu.

    The check uses atol=1e-9 to handle near-zero entries in x_splu (pressure
    dofs near pinned nodes can be O(1e-3) or smaller; for those the per-element
    relative threshold 1e-8 * |x_splu| ~ 1e-11 is close to the absolute
    accuracy limit of an iterative solver converged to 1e-10 residual).
    The norm-based relative accuracy ||x - x_splu|| / ||x_splu|| is ~ 3e-10,
    comfortably within 1e-8; atol=1e-9 guards near-zero entries only.
    """
    from diffsim.solvers.linsolve import solve_linear

    Acsr, b, x_splu = _get_system()

    x = solve_linear(Acsr, b, solver="fgmres_bdiag", sym=False,
                     device="cpu", tol=1e-10)

    assert np.allclose(x, x_splu, rtol=1e-8, atol=1e-9), (
        f"max |x - x_splu| = {np.abs(x - x_splu).max():.3e}"
    )


def test_fgmres_bdiag_result_carries_iterations():
    """return_result=True must give LinearSolveResult with iterations > 0."""
    from diffsim.solvers.linsolve import solve_linear
    from diffsim.solvers.result import LinearSolveResult

    Acsr, b, x_splu = _get_system()

    result = solve_linear(Acsr, b, solver="fgmres_bdiag", sym=False,
                          device="cpu", tol=1e-10, return_result=True)

    assert isinstance(result, LinearSolveResult), type(result)
    assert result.converged, "result.converged is False"
    assert result.iterations is not None, "result.iterations is None"
    assert result.iterations > 0, f"result.iterations={result.iterations}"
    assert np.allclose(result.x, x_splu, rtol=1e-8, atol=1e-9), (
        f"max |x - x_splu| = {np.abs(result.x - x_splu).max():.3e}"
    )
