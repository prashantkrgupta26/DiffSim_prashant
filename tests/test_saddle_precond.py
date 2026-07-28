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

def _one_step_system(return_meta=False):
    """Build a real 2-D (u,v,p) saddle system from the thin-plate driver.

    Replicates the host-assembly path of run_flow_past for ONE BDF1 step at
    level=4 (zero initial velocity, t=dt, step=0).  Returns (Acsr, b, x_splu)
    where x_splu is the reference solution from scipy splu.

    Interleaved dof convention: ndof=3, dofs per node are (u_x, u_y, p).
    Velocity dofs: node i -> i*3+0, i*3+1.
    Pressure dof : node i -> i*3+2.

    ``return_meta=True`` additionally returns the ``dm`` DeviceMesh, the
    scalar physical parameters ``(nu, sigma)`` and the pressure-pin dof
    ``p_pin`` — everything Task A3's ``build_pcd_meta`` needs to assemble the
    pressure-space PCD operators on the SAME octree/constraints as the saddle.
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
    if return_meta:
        return Acsr, b, x_splu, dm, nu, sigma, p_pin
    return Acsr, b, x_splu


# ---------------------------------------------------------------------------
# Cache: build system once per session (expensive; level=4 takes ~2 s).
# ---------------------------------------------------------------------------
_SYSTEM_CACHE = {}


def _get_system():
    if "sys" not in _SYSTEM_CACHE:
        _SYSTEM_CACHE["sys"] = _one_step_system()
    return _SYSTEM_CACHE["sys"]


def _get_system_with_meta():
    """Same real level-4 saddle as _get_system, but also carrying the pieces
    Task A3 needs to build the PCD pressure-space operators (dm, nu, sigma,
    p_pin).  Cached separately (the extra return is only needed by the PCD
    tests)."""
    if "sys_meta" not in _SYSTEM_CACHE:
        _SYSTEM_CACHE["sys_meta"] = _one_step_system(return_meta=True)
    return _SYSTEM_CACHE["sys_meta"]


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


# ---------------------------------------------------------------------------
# Task A3: PCD (pressure convection-diffusion) Schur preconditioner
# ---------------------------------------------------------------------------

def _pcd_cache(dm, nu, sigma, p_pin):
    """Build the ('pcd_meta', key) cache the fgmres_pcd backend reads (the
    least-invasive plumbing: the backend cannot see dm, so the caller builds
    meta once and passes it through the existing solve_linear cache)."""
    from diffsim.solvers.saddle_precond import build_pcd_meta
    meta = build_pcd_meta(dm, nu, sigma, p_pin=p_pin)
    return {("pcd_meta", "A3"): meta}


def _pcd_accurate(x, x_splu):
    """PCD accuracy gate vs splu.  The MEANINGFUL metric is the norm-relative
    error ||x - x_splu|| / ||x_splu|| < 1e-8 (measured ~8e-10; residual ~7e-11).

    Unlike the bdiag path (an EXACT diagonal apply), PCD's preconditioner uses
    truncated Krylov inner solves, so it is a flexible/inexact operator whose
    per-ELEMENT solution-error floor on tiny (O(1e-5)) velocity/pressure
    components is ~1e-8 absolute — larger than bdiag's near-zero floor.  So the
    per-element guard uses atol=5e-8 (vs bdiag's 1e-9): it guards near-zero
    entries at PCD's real accuracy floor, NOT a looser correctness bar (the
    norm-relative 1e-8 assertion below is the correctness gate)."""
    nrel = np.linalg.norm(x - x_splu) / np.linalg.norm(x_splu)
    assert nrel < 1e-8, f"norm-relative ||x-x_splu||/||x_splu|| = {nrel:.3e}"
    assert np.allclose(x, x_splu, rtol=1e-8, atol=5e-8), (
        f"max |x - x_splu| = {np.abs(x - x_splu).max():.3e}"
    )


def test_fgmres_pcd_solves_real_saddle():
    """fgmres_pcd must solve the real 2-D saddle to rtol=1e-8 vs splu."""
    from diffsim.solvers.linsolve import solve_linear

    Acsr, b, x_splu, dm, nu, sigma, p_pin = _get_system_with_meta()
    cache = _pcd_cache(dm, nu, sigma, p_pin)

    x = solve_linear(Acsr, b, solver="fgmres_pcd", sym=False,
                     device="cpu", tol=1e-10, cache=cache, cache_key="A3")

    _pcd_accurate(x, x_splu)


def test_fgmres_pcd_result_carries_iterations():
    """return_result=True must give LinearSolveResult with iterations > 0."""
    from diffsim.solvers.linsolve import solve_linear
    from diffsim.solvers.result import LinearSolveResult

    Acsr, b, x_splu, dm, nu, sigma, p_pin = _get_system_with_meta()
    cache = _pcd_cache(dm, nu, sigma, p_pin)

    result = solve_linear(Acsr, b, solver="fgmres_pcd", sym=False,
                          device="cpu", tol=1e-10, cache=cache,
                          cache_key="A3", return_result=True)

    assert isinstance(result, LinearSolveResult), type(result)
    assert result.converged, "result.converged is False"
    assert result.iterations is not None, "result.iterations is None"
    assert result.iterations > 0, f"result.iterations={result.iterations}"
    _pcd_accurate(result.x, x_splu)


def test_fgmres_pcd_beats_bdiag():
    """The POINT of PCD: fewer outer iterations than block-diagonal Jacobi on
    the SAME saddle.  This is the campaign-relevant claim — if PCD does NOT
    beat bdiag we FAIL loudly with both counts printed (a finding, not a
    silent pass).  Iteration counts here are the module sentinel _LAST_ITERS
    (total inner FGMRES iterations), read via return_result."""
    from diffsim.solvers.linsolve import solve_linear

    Acsr, b, x_splu, dm, nu, sigma, p_pin = _get_system_with_meta()
    cache = _pcd_cache(dm, nu, sigma, p_pin)

    r_pcd = solve_linear(Acsr, b, solver="fgmres_pcd", sym=False,
                         device="cpu", tol=1e-10, cache=cache,
                         cache_key="A3", return_result=True)
    r_bd = solve_linear(Acsr, b, solver="fgmres_bdiag", sym=False,
                        device="cpu", tol=1e-10, return_result=True)

    it_pcd, it_bd = r_pcd.iterations, r_bd.iterations
    print(f"\n[A3] PCD vs bdiag total inner FGMRES iterations: "
          f"pcd={it_pcd} bdiag={it_bd}")

    # both must be correct solutions first (PCD at its inexact-inner accuracy
    # floor via _pcd_accurate; bdiag is an exact diagonal apply -> tight)
    _pcd_accurate(r_pcd.x, x_splu)
    assert np.allclose(r_bd.x, x_splu, rtol=1e-8, atol=1e-9)

    assert it_pcd < it_bd, (
        f"PCD did NOT beat bdiag: pcd={it_pcd} bdiag={it_bd} "
        f"(the point of PCD is fewer iterations than block-Jacobi)"
    )
