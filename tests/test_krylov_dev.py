"""M1b Task 1 gates: single-sync device CG — parity with the host solver,
strict sync accounting, and a wall-clock sanity margin (m1a finding 3)."""
import time

import numpy as np
import pytest
import warp as wp
from diffsim.solvers.krylov import cg
from diffsim.solvers.krylov_dev import cg_dev, SyncCounter
from diffsim.assembly.operators import CSROperator
import scipy.sparse as sp

pytestmark = pytest.mark.tier3


def _spd_csr(n, seed=3):
    rng = np.random.default_rng(seed)
    # SPD banded system: 1D Laplacian + random positive diagonal
    main = 2.0 + rng.uniform(0.5, 1.5, n)
    A = sp.diags([-np.ones(n - 1), main, -np.ones(n - 1)], [-1, 0, 1]).tocsr()
    return A


def test_cg_dev_matches_host(device):
    n = 4000
    A = _spd_csr(n)
    op = CSROperator(A, device)
    rng = np.random.default_rng(5)
    b = rng.standard_normal(n)
    x_h, info_h = cg(op, b, tol=1e-12, maxiter=8000,
                     diag=np.asarray(A.diagonal()))
    x_d, info_d = cg_dev(op, b, tol=1e-12, maxiter=8000,
                         diag=np.asarray(A.diagonal()), check_every=10)
    assert info_d["converged"] and info_h["converged"]
    # both solve to the same tolerance: compare against the direct solution
    from scipy.sparse.linalg import spsolve
    x_ref = spsolve(A, b)
    for x in (x_h, x_d):
        assert np.abs(x - x_ref).max() < 1e-8 * np.abs(x_ref).max()


def test_cg_dev_sync_accounting(device):
    n = 2000
    A = _spd_csr(n)
    op = CSROperator(A, device)
    b = np.ones(n)
    sc = SyncCounter()
    x, info = cg_dev(op, b, tol=1e-11, maxiter=4000,
                     diag=np.asarray(A.diagonal()), check_every=25,
                     sync_counter=sc)
    assert info["converged"]
    # syncs = 1 (bnorm) + ceil(iters/check_every) — nothing per-iteration
    import math
    assert sc.count == 1 + math.ceil(info["iters"] / 25), (
        sc.count, info["iters"])


def test_cg_dev_per_iteration_cheaper(device):
    """Structural claim: the fused path's per-iteration cost is below the
    host-sync path's AT MEANINGFUL SIZE. (Care point: a 27-iteration 15 ms
    microbench showed near-parity — idle-queue WSL2 syncs are ~0.1 ms; the
    measured 10 ms pathology appears under queued load, m1a finding 3. The
    absolute ratios are recorded in m1b findings, not hard-asserted.)"""
    if device == "cpu":
        pytest.skip("latency comparison is a GPU statement")
    n = 200000
    rng = np.random.default_rng(7)
    main = 2.0 + 0.05 * rng.uniform(0.5, 1.5, n)
    A = sp.diags([-np.ones(n - 1), main, -np.ones(n - 1)], [-1, 0, 1]).tocsr()
    op = CSROperator(A, device)
    b = np.ones(n)
    diag = np.asarray(A.diagonal())
    cg(op, b, tol=1e-4, maxiter=30, diag=diag)          # warm
    cg_dev(op, b, tol=1e-4, maxiter=30, diag=diag)      # warm
    t0 = time.time()
    _, ih = cg(op, b, tol=1e-9, maxiter=1500, diag=diag)
    t_host = time.time() - t0
    t0 = time.time()
    _, idv = cg_dev(op, b, tol=1e-9, maxiter=1500, diag=diag, check_every=20)
    t_dev = time.time() - t0
    assert ih["converged"] and idv["converged"]
    per_h = t_host / ih["iters"]
    per_d = t_dev / idv["iters"]
    print(f"per-iter host {per_h*1e3:.3f} ms vs dev {per_d*1e3:.3f} ms "
          f"(iters {ih['iters']}/{idv['iters']})")
    assert per_d < per_h, (per_d, per_h)


def test_bicgstab_dev_matches_host_nonsymmetric(device):
    from diffsim.solvers.krylov import bicgstab
    from diffsim.solvers.krylov_dev import bicgstab_dev
    n = 3000
    rng = np.random.default_rng(9)
    main = 3.0 + rng.uniform(0, 1, n)
    A = sp.diags([-1.2 * np.ones(n - 1), main, -0.8 * np.ones(n - 1)],
                 [-1, 0, 1]).tocsr()                      # nonsymmetric
    op = CSROperator(A, device)
    b = rng.standard_normal(n)
    diag = np.asarray(A.diagonal())
    x_h, ih = bicgstab(op, b, tol=1e-12, maxiter=6000, diag=diag)
    sc = SyncCounter()
    x_d, idv = bicgstab_dev(op, b, tol=1e-12, maxiter=6000, diag=diag,
                            check_every=25, sync_counter=sc)
    assert ih["converged"] and idv["converged"]
    from scipy.sparse.linalg import spsolve
    x_ref = spsolve(A, b)
    for x in (x_h, x_d):
        assert np.abs(x - x_ref).max() < 1e-8 * np.abs(x_ref).max()
    import math
    assert sc.count == 1 + math.ceil(idv["iters"] / 25)


def test_bicgstab_dev_breakdown_guard(device):
    from diffsim.solvers.krylov_dev import bicgstab_dev
    A = sp.csr_matrix(np.array([[0.0, 1.0], [-1.0, 0.0]]))  # skew: rhat.Ab=0
    op = CSROperator(A, device)
    x, info = bicgstab_dev(op, np.array([1.0, 1.0]), maxiter=50,
                           check_every=5, max_restarts=3)
    assert info["converged"] is False
    # with restart-on-breakdown semantics: either the restart budget is
    # exhausted (breakdown reported) or maxiter hits first with restarts>0
    assert info.get("breakdown") == "rhat_v" or info.get("restarts", 0) > 0, info


# ---------------------------------------------------------------------------
# Task #40: fused / graph-captured inner loop — parity with the legacy path
# ---------------------------------------------------------------------------

def _nonsym_csr(n, seed=9):
    rng = np.random.default_rng(seed)
    main = 3.0 + rng.uniform(0, 1, n)
    return sp.diags([-1.2 * np.ones(n - 1), main, -0.8 * np.ones(n - 1)],
                    [-1, 0, 1]).tocsr()


def _max_rel(a, b):
    return np.abs(a - b).max() / max(np.abs(b).max(), 1e-300)


def test_cg_fused_and_graph_match_legacy(device):
    """G1 contract: fused ("fused") and captured ("graph") CG produce the
    SAME iterates as the legacy loop ("off") — bit-equal on the serial
    CPU backend, few-ULP under CUDA atomic scheduling; identical
    iteration counts always (same batch boundaries, same checks)."""
    from diffsim.solvers.krylov_dev import cg_dev
    n = 4000
    A = _spd_csr(n)
    op = CSROperator(A, device)
    b = np.random.default_rng(5).standard_normal(n)
    kw = dict(tol=1e-12, maxiter=8000, diag=np.asarray(A.diagonal()),
              check_every=10)
    x_off, i_off = cg_dev(op, b, graph="off", **kw)
    x_f, i_f = cg_dev(op, b, graph="fused", **kw)
    x_g, i_g = cg_dev(op, b, graph="graph", **kw)
    assert i_off["converged"] and i_f["converged"] and i_g["converged"]
    assert i_off["iters"] == i_f["iters"] == i_g["iters"]
    assert i_g.get("graph") is True     # capture actually engaged
    if device == "cpu":
        assert np.array_equal(x_f, x_off)   # serial backend: bit-equal
        assert np.array_equal(x_g, x_f)     # replay == recorded launches
    else:
        assert _max_rel(x_f, x_off) < 1e-12
        assert _max_rel(x_g, x_f) < 1e-12


def test_bicgstab_fused_and_graph_match_legacy(device):
    from diffsim.solvers.krylov_dev import bicgstab_dev
    n = 3000
    A = _nonsym_csr(n)
    op = CSROperator(A, device)
    b = np.random.default_rng(11).standard_normal(n)
    kw = dict(tol=1e-12, maxiter=6000, diag=np.asarray(A.diagonal()),
              check_every=25)
    x_off, i_off = bicgstab_dev(op, b, graph="off", **kw)
    x_f, i_f = bicgstab_dev(op, b, graph="fused", **kw)
    x_g, i_g = bicgstab_dev(op, b, graph="graph", **kw)
    assert i_off["converged"] and i_f["converged"] and i_g["converged"]
    assert i_off["iters"] == i_f["iters"] == i_g["iters"]
    assert i_g.get("graph") is True
    if device == "cpu":
        assert np.array_equal(x_f, x_off)
        assert np.array_equal(x_g, x_f)
    else:
        assert _max_rel(x_f, x_off) < 1e-12
        assert _max_rel(x_g, x_f) < 1e-12


def test_fused_partial_tail_batch(device):
    """maxiter not a multiple of check_every: the captured path replays
    full batches and runs the tail on fused launches — iterate-identical
    to the legacy loop (same it counts, same x)."""
    from diffsim.solvers.krylov_dev import cg_dev
    n = 2000
    A = _spd_csr(n)
    op = CSROperator(A, device)
    b = np.ones(n)
    kw = dict(tol=1e-30, atol=0.0, maxiter=33, diag=np.asarray(A.diagonal()),
              check_every=10)                 # unconverged by construction
    x_off, i_off = cg_dev(op, b, graph="off", **kw)
    x_g, i_g = cg_dev(op, b, graph="graph", **kw)
    assert i_off["iters"] == i_g["iters"] == 33
    assert not i_off["converged"] and not i_g["converged"]
    if device == "cpu":
        assert np.array_equal(x_g, x_off)
    else:
        assert _max_rel(x_g, x_off) < 1e-12


def test_fused_sync_accounting(device):
    """The fused path preserves the single-sync contract: 1 bnorm sync +
    ceil(iters / check_every) periodic checks, nothing per iteration."""
    from diffsim.solvers.krylov_dev import cg_dev
    import math
    n = 2000
    A = _spd_csr(n)
    op = CSROperator(A, device)
    sc = SyncCounter()
    x, info = cg_dev(op, np.ones(n), tol=1e-11, maxiter=4000,
                     diag=np.asarray(A.diagonal()), check_every=25,
                     sync_counter=sc, graph="fused")
    assert info["converged"]
    assert sc.count == 1 + math.ceil(info["iters"] / 25), (
        sc.count, info["iters"])


def test_workspace_and_graph_reuse(device):
    """The workspace (and captured graph) are cached per operator-buffer
    key: a second solve on the same operator replays the SAME graph (no
    re-capture), and a different operator gets a different workspace."""
    from diffsim.solvers import krylov_dev as kd
    n = 1500
    A = _spd_csr(n)
    op = CSROperator(A, device)
    diag = np.asarray(A.diagonal())
    kw = dict(tol=1e-11, maxiter=4000, diag=diag, check_every=10,
              graph="graph")
    kd.cg_dev(op, np.ones(n), **kw)
    key = ("cg", str(device), n, 10, kd._op_key(op))
    ws = kd._WS_CACHE[key]
    g1 = ws.graph
    assert g1 is not None
    x2, i2 = kd.cg_dev(op, np.arange(n, dtype=float), **kw)
    assert kd._WS_CACHE[key] is ws and ws.graph is g1   # no re-capture
    assert i2["converged"]
    op2 = CSROperator(_spd_csr(n, seed=8), device)
    kd.cg_dev(op2, np.ones(n), **kw)
    assert kd._WS_CACHE[("cg", str(device), n, 10, kd._op_key(op2))] is not ws


def test_bicgstab_fused_breakdown_restart_parity(device):
    """Breakdown/restart semantics survive fusion: the device 'first'
    flag (scal[11]) makes post-restart batches graph-uniform.  The skew
    system rhat.(A rhat) = 0 trips the rhat_v guard in both paths."""
    from diffsim.solvers.krylov_dev import bicgstab_dev
    A = sp.csr_matrix(np.array([[0.0, 1.0], [-1.0, 0.0]]))
    b = np.array([1.0, 1.0])
    diag = np.ones(2)                   # unit Jacobi: same math as None
    x_off, i_off = bicgstab_dev(CSROperator(A, device), b, maxiter=50,
                                check_every=5, max_restarts=3, diag=diag,
                                graph="off")
    x_f, i_f = bicgstab_dev(CSROperator(A, device), b, maxiter=50,
                            check_every=5, max_restarts=3, diag=diag,
                            graph="graph")
    assert i_off["converged"] is False and i_f["converged"] is False
    assert i_off.get("breakdown") == i_f.get("breakdown")
    assert i_off.get("restarts") == i_f.get("restarts")
    assert i_off["iters"] == i_f["iters"]


def test_matrix_free_op_falls_back_to_legacy(device):
    """Regression (G2 catch): cg_dev/bicgstab_dev accept ANY object with
    .matvec/.n_free/.device — a matrix-free op without CSR device
    buffers must take the legacy loop under every knob value, not crash
    in the workspace keying."""
    from diffsim.solvers.krylov_dev import cg_dev
    n = 1000
    A = _spd_csr(n)
    inner = CSROperator(A, device)

    class Wrapped:                      # operator protocol, no _dev/_spmv
        device = inner.device
        n_free = inner.n_free

        def matvec(self, x, y):
            inner.matvec(x, y)

    b = np.ones(n)
    diag = np.asarray(A.diagonal())
    for mode in ("auto", "graph", "fused", "off"):
        x, info = cg_dev(Wrapped(), b, tol=1e-11, maxiter=4000,
                         diag=diag, check_every=10, graph=mode)
        assert info["converged"]
        assert "graph" not in info      # legacy loop path taken
