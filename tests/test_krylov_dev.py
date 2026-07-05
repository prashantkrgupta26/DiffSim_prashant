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
                           check_every=5)
    assert info["converged"] is False
    assert info.get("breakdown") == "rhat_v"
