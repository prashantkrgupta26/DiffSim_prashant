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


def _get_system_with_meta(inner=None):
    """Same real level-4 saddle as _get_system, but also carrying the pieces
    Task A3 needs to build the PCD pressure-space operators (dm, nu, sigma,
    p_pin).  Cached separately (the extra return is only needed by the PCD
    tests).

    Default (``inner=None``): returns the 7-tuple
    ``(Acsr, b, x_splu, dm, nu, sigma, p_pin)`` — the historical signature used
    by every existing PCD test.

    ``inner="jacobi"|"amgx"`` (Task T2): returns the 4-tuple
    ``(Acsr, b, x_splu, cache)`` where ``cache`` already holds the
    ``("pcd_meta", "k")`` entry built via ``build_pcd_meta(..., inner=inner)``,
    ready to hand straight to ``solve_linear(..., cache_key="k")``.  This is the
    routing-test shape from the T2 brief."""
    if "sys_meta" not in _SYSTEM_CACHE:
        _SYSTEM_CACHE["sys_meta"] = _one_step_system(return_meta=True)
    Acsr, b, x_splu, dm, nu, sigma, p_pin = _SYSTEM_CACHE["sys_meta"]
    if inner is None:
        return Acsr, b, x_splu, dm, nu, sigma, p_pin
    from diffsim.solvers.saddle_precond import build_pcd_meta
    meta = build_pcd_meta(dm, nu, sigma, p_pin=p_pin, inner=inner)
    cache = {("pcd_meta", "k"): meta}
    return Acsr, b, x_splu, cache


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


# ---------------------------------------------------------------------------
# Task T1: PCD inner-solve telemetry
# ---------------------------------------------------------------------------

def test_fgmres_pcd_inner_stats():
    """return_result=True must populate r.inner_stats with per-block
    telemetry (applies, iters_total, cap_hits, max_exit_relres) for each
    of the three PCD inner-solve blocks (F, Ap, Mp)."""
    from diffsim.solvers.linsolve import solve_linear

    from diffsim.solvers.saddle_precond import build_pcd_meta

    Acsr, b, x_splu, dm, nu, sigma, p_pin = _get_system_with_meta()
    cache = {("pcd_meta", "T1"): build_pcd_meta(dm, nu, sigma, p_pin=p_pin)}

    r = solve_linear(Acsr, b, solver="fgmres_pcd", sym=False, device="cpu",
                     tol=1e-10, cache=cache, cache_key="T1", return_result=True)
    st = r.inner_stats
    assert st is not None, "r.inner_stats is None — telemetry not wired up"
    for blk in ("F", "Ap", "Mp"):
        assert blk in st, f"block '{blk}' missing from inner_stats"
        assert st[blk]["applies"] > 0, (
            f"{blk}: applies={st[blk]['applies']}, expected > 0")
        assert st[blk]["iters_total"] > 0, (
            f"{blk}: iters_total={st[blk]['iters_total']}, expected > 0")
        assert st[blk]["cap_hits"] >= 0, (
            f"{blk}: cap_hits={st[blk]['cap_hits']}, expected >= 0")
        assert 0.0 <= st[blk]["max_exit_relres"] < float("inf"), (
            f"{blk}: max_exit_relres={st[blk]['max_exit_relres']}, "
            f"expected finite non-negative")


# ---------------------------------------------------------------------------
# Task T2: AMGX velocity-inner PCD (pcd_inner="amgx", extracted block only)
# ---------------------------------------------------------------------------

def _assert_pcd_accuracy(x, x_splu):
    """Alias for the T2 brief's routing test (same gates as _pcd_accurate)."""
    _pcd_accurate(x, x_splu)


def test_fgmres_pcd_amgx_routing(monkeypatch):
    """ROUTING proof (CPU/CI, no pyamgx): with inner="amgx", make_pcd_apply
    must extract the velocity sub-CSR F = A[u_ids][:,u_ids] ONCE and hand THAT
    (never the raw saddle) to amgx_solve.  A scipy-splu stand-in monkeypatched
    over saddle_precond.amgx_solve proves (a) the F it receives is the
    (n_u, n_u) velocity block and (b) the outer solve still hits the PCD
    accuracy gates.  The HARD documented constraint (AMGX must never see the
    raw saddle) is enforced here by asserting the received shape."""
    from diffsim.solvers.linsolve import solve_linear

    calls = {}

    def fake_amgx(F, rhs, sym=False, tol=0.0, maxiter=0, **kw):
        calls["shape"] = F.shape
        calls["n"] = calls.get("n", 0) + 1
        calls["nnz"] = int(F.nnz)
        calls["sym"] = sym
        import scipy.sparse.linalg as sla
        return sla.spsolve(F.tocsc(), rhs)

    monkeypatch.setattr("diffsim.solvers.saddle_precond.amgx_solve", fake_amgx)

    Acsr, b, x_splu, cache = _get_system_with_meta(inner="amgx")
    r = solve_linear(Acsr, b, solver="fgmres_pcd", sym=False, device="cpu",
                     tol=1e-10, cache=cache, cache_key="k", return_result=True)

    n_u = (Acsr.shape[0] // 3) * 2          # 2-D: velocity dofs
    assert calls["shape"] == (n_u, n_u), (
        f"amgx_solve got shape {calls['shape']}, expected velocity block "
        f"({n_u}, {n_u}) — NOT the raw saddle {Acsr.shape}")
    assert calls["nnz"] == int(Acsr[
        (np.arange(Acsr.shape[0] // 3)[:, None] * 3
         + np.arange(2)[None, :]).ravel()][:, (
            np.arange(Acsr.shape[0] // 3)[:, None] * 3
            + np.arange(2)[None, :]).ravel()].nnz), (
        "F nnz mismatch — the extracted block is not the velocity sub-CSR")
    assert calls["sym"] is False, "F block is nonsymmetric — sym must be False"
    assert calls["n"] >= 1
    _assert_pcd_accuracy(r.x, x_splu)


def test_build_pcd_meta_inner_default_is_jacobi():
    """The inner kwarg defaults to 'jacobi' (today's behavior bit-for-bit)."""
    from diffsim.solvers.saddle_precond import build_pcd_meta

    _, _, _, dm, nu, sigma, p_pin = _get_system_with_meta()
    meta = build_pcd_meta(dm, nu, sigma, p_pin=p_pin)
    assert meta["inner"] == "jacobi"
    meta_a = build_pcd_meta(dm, nu, sigma, p_pin=p_pin, inner="amgx")
    assert meta_a["inner"] == "amgx"


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("pyamgx") is None,
    reason="pyamgx not installed (GPU-only); real-AMGX check runs on the box")
def test_fgmres_pcd_amgx_real():  # needs pyamgx (GPU box); PCD apply itself is host-side
    """Real-AMGX correctness on GPU (skipped on CPU-only CI; T4's box run
    executes this).  With inner='amgx' the F-inner runs the actual AMGX
    BiCGStab+classical-AMG solve on the extracted velocity block; the outer
    FGMRES must still reach the PCD accuracy gates vs splu."""
    from diffsim.solvers.linsolve import solve_linear

    Acsr, b, x_splu, cache = _get_system_with_meta(inner="amgx")
    r = solve_linear(Acsr, b, solver="fgmres_pcd", sym=False, device="cpu",
                     tol=1e-10, cache=cache, cache_key="k", return_result=True)
    _assert_pcd_accuracy(r.x, x_splu)
    # F-inner telemetry must be populated from AMGX last_solve_stats
    assert r.inner_stats["F"]["applies"] > 0


# ---------------------------------------------------------------------------
# Task T5: AMG-on-Ap inner for PCD (ap_inner="amgx")
# ---------------------------------------------------------------------------

def _get_system_with_ap_inner(ap_inner):
    """Same real level-4 saddle as _get_system_with_meta, but with ap_inner
    set in the pcd_meta.  Returns (Acsr, b, x_splu, cache) where cache
    already holds the ("pcd_meta", "k_ap") entry."""
    if "sys_meta" not in _SYSTEM_CACHE:
        _SYSTEM_CACHE["sys_meta"] = _one_step_system(return_meta=True)
    Acsr, b, x_splu, dm, nu, sigma, p_pin = _SYSTEM_CACHE["sys_meta"]
    from diffsim.solvers.saddle_precond import build_pcd_meta
    meta = build_pcd_meta(dm, nu, sigma, p_pin=p_pin, ap_inner=ap_inner)
    cache = {("pcd_meta", "k_ap"): meta}
    return Acsr, b, x_splu, cache


def test_build_pcd_meta_ap_inner_default_is_jacobi():
    """ap_inner defaults to 'jacobi' (bit-for-bit unchanged); accepts 'amgx'."""
    from diffsim.solvers.saddle_precond import build_pcd_meta

    _, _, _, dm, nu, sigma, p_pin = _get_system_with_meta()
    # default — ap_inner absent
    meta = build_pcd_meta(dm, nu, sigma, p_pin=p_pin)
    assert meta["ap_inner"] == "jacobi", (
        f"default ap_inner={meta['ap_inner']!r}, expected 'jacobi'")
    # explicit jacobi
    meta_j = build_pcd_meta(dm, nu, sigma, p_pin=p_pin, ap_inner="jacobi")
    assert meta_j["ap_inner"] == "jacobi"
    # amgx
    meta_a = build_pcd_meta(dm, nu, sigma, p_pin=p_pin, ap_inner="amgx")
    assert meta_a["ap_inner"] == "amgx"
    # invalid
    with pytest.raises(ValueError, match="ap_inner"):
        build_pcd_meta(dm, nu, sigma, p_pin=p_pin, ap_inner="bad")


def test_fgmres_pcd_ap_amgx_routing(monkeypatch):
    """ROUTING proof (CPU/CI, no pyamgx): with ap_inner="amgx", make_pcd_apply
    must pass the SCALAR pinned Ap (shape (n_p, n_p)) to amgx_solve with
    sym=True.  A scipy-spsolve stand-in monkeypatched over
    saddle_precond.amgx_solve proves: (a) the Ap it receives is the scalar
    pressure block with shape (n_nodes, n_nodes); (b) sym=True; (c) the outer
    solve still hits the PCD accuracy gates.  The HARD documented constraint
    (AMGX must never see the raw saddle) is enforced by asserting shape."""
    from diffsim.solvers.linsolve import solve_linear

    ap_calls = {}

    def fake_amgx(M, rhs, sym=False, tol=0.0, maxiter=0, **kw):
        ap_calls.setdefault("shapes", []).append(M.shape)
        ap_calls["sym"] = sym
        ap_calls["n"] = ap_calls.get("n", 0) + 1
        import scipy.sparse.linalg as sla
        return sla.spsolve(M.tocsc(), rhs)

    monkeypatch.setattr("diffsim.solvers.saddle_precond.amgx_solve", fake_amgx)

    Acsr, b, x_splu, cache = _get_system_with_ap_inner("amgx")
    r = solve_linear(Acsr, b, solver="fgmres_pcd", sym=False, device="cpu",
                     tol=1e-10, cache=cache, cache_key="k_ap",
                     return_result=True)

    n_nodes = Acsr.shape[0] // 3   # 2-D: (u_x, u_y, p) per node
    # Every call must be to the scalar Ap (n_nodes × n_nodes), not the saddle
    for shape in ap_calls.get("shapes", []):
        assert shape == (n_nodes, n_nodes), (
            f"amgx_solve received shape {shape}, expected Ap ({n_nodes}, {n_nodes})"
            f" — NOT the raw saddle {Acsr.shape}")
    assert ap_calls.get("sym") is True, "Ap block is SPD — sym must be True"
    assert ap_calls.get("n", 0) >= 1, "amgx_solve was never called for Ap"
    _assert_pcd_accuracy(r.x, x_splu)


def test_fgmres_pcd_ap_jacobi_default_parity():
    """ap_inner='jacobi' (explicit) must produce bit-for-bit identical results
    to the default (ap_inner omitted), including inner_stats for Ap block.
    This guards the default-parity contract (T5 must not change the jacobi path).
    """
    from diffsim.solvers.saddle_precond import build_pcd_meta
    from diffsim.solvers.linsolve import solve_linear

    _, _, x_splu, dm, nu, sigma, p_pin = _get_system_with_meta()
    Acsr, b, _ = _get_system()

    # Default path (no ap_inner kwarg)
    meta_def = build_pcd_meta(dm, nu, sigma, p_pin=p_pin)
    cache_def = {("pcd_meta", "def"): meta_def}
    r_def = solve_linear(Acsr, b, solver="fgmres_pcd", sym=False, device="cpu",
                         tol=1e-10, cache=cache_def, cache_key="def",
                         return_result=True)

    # Explicit jacobi
    meta_jac = build_pcd_meta(dm, nu, sigma, p_pin=p_pin, ap_inner="jacobi")
    cache_jac = {("pcd_meta", "jac"): meta_jac}
    r_jac = solve_linear(Acsr, b, solver="fgmres_pcd", sym=False, device="cpu",
                         tol=1e-10, cache=cache_jac, cache_key="jac",
                         return_result=True)

    # Both must be accurate
    _assert_pcd_accuracy(r_def.x, x_splu)
    _assert_pcd_accuracy(r_jac.x, x_splu)
    # Bit-for-bit identical iteration counts (same code path)
    assert r_def.iterations == r_jac.iterations, (
        f"Default vs explicit jacobi iterations differ: "
        f"{r_def.iterations} vs {r_jac.iterations}")
    # Ap inner stats must be present and positive on both
    assert r_def.inner_stats["Ap"]["applies"] > 0
    assert r_jac.inner_stats["Ap"]["applies"] > 0
    assert r_def.inner_stats["Ap"]["iters_total"] > 0
    assert r_jac.inner_stats["Ap"]["iters_total"] > 0


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("pyamgx") is None,
    reason="pyamgx not installed (GPU-only); real AMG-on-Ap check runs on the box")
def test_fgmres_pcd_ap_amgx_real():
    """Real-AMGX Ap-inner correctness on GPU (skipped on CPU-only CI; T5's box
    run executes this).  With ap_inner='amgx' the Ap-inner runs the actual
    AMGX PCG+classical-AMG solve on the pinned scalar Ap; the outer FGMRES
    must still reach the PCD accuracy gates vs splu.  Ap iters/apply should
    collapse from ~283 (jacobi) to O(10) (AMG)."""
    from diffsim.solvers.linsolve import solve_linear

    Acsr, b, x_splu, cache = _get_system_with_ap_inner("amgx")
    r = solve_linear(Acsr, b, solver="fgmres_pcd", sym=False, device="cpu",
                     tol=1e-10, cache=cache, cache_key="k_ap",
                     return_result=True)
    _assert_pcd_accuracy(r.x, x_splu)
    # Ap-inner telemetry must be populated from AMGX last_solve_stats
    assert r.inner_stats["Ap"]["applies"] > 0
