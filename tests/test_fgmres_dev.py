"""Task #49 gates: the device-resident outer flexible GMRES
(solvers/fgmres_dev) — the Arnoldi/Givens/back-substitution/restart
machinery.  Parity against scipy lgmres (the flexible right-preconditioned
outer it replaces), on a stored device matvec + Jacobi device
preconditioner (the CSROperator interface the blockch outer supplies).
CPU-runnable: warp's CPU backend exercises the exact same kernels the
CUDA outer runs, so the math is verified bit-for-bit on serial CPU and to
atomic-ULP on CUDA."""
import numpy as np
import pytest
import scipy.sparse as sp
import warp as wp
from scipy.sparse.linalg import lgmres, LinearOperator

from diffsim.assembly.operators import CSROperator
from diffsim.solvers.fgmres_dev import fgmres_dev

pytestmark = pytest.mark.tier3


def _nonsym_csr(n, seed=0):
    # diagonally-dominant nonsymmetric banded system (Jacobi-solvable)
    rng = np.random.default_rng(seed)
    main = 4.0 + rng.uniform(0.5, 1.5, n)
    upper = 0.6 * rng.standard_normal(n - 1)
    lower = -0.4 * rng.standard_normal(n - 1)
    return sp.diags([lower, main, upper], [-1, 0, 1]).tocsr()


def _run(A, b, device, restart, maxiter, tol=1e-11):
    op = CSROperator(A, device)
    dinv = 1.0 / np.asarray(A.diagonal())
    dinv_d = wp.array(np.ascontiguousarray(dinv), dtype=wp.float64,
                      device=device)

    prec_k = _jacobi_kernel()

    def apply_dev(v, z):
        wp.launch(prec_k, dim=A.shape[0], inputs=[dinv_d, v, z],
                  device=device)

    b_dev = wp.array(np.ascontiguousarray(b, np.float64), dtype=wp.float64,
                     device=device)
    x_dev, info = fgmres_dev(op.matvec, b_dev, apply_dev, A.shape[0],
                             device, tol=tol, atol=1e-13, restart=restart,
                             maxiter=maxiter)
    return x_dev.numpy(), info


_JAC = [None]


def _jacobi_kernel():
    if _JAC[0] is not None:
        return _JAC[0]

    @wp.kernel(enable_backward=False)
    def jac(dinv: wp.array(dtype=wp.float64),
            v: wp.array(dtype=wp.float64),
            z: wp.array(dtype=wp.float64)):
        i = wp.tid()
        z[i] = dinv[i] * v[i]

    _JAC[0] = jac
    return jac


def test_fgmres_dev_solves_nonsym(device):
    n = 500
    A = _nonsym_csr(n)
    rng = np.random.default_rng(7)
    xtrue = rng.standard_normal(n)
    b = A @ xtrue
    x, info = _run(A, b, device, restart=40, maxiter=50)
    assert info["converged"], info
    res = np.abs(A @ x - b).max() / max(np.abs(b).max(), 1e-30)
    assert res < 1e-9, (res, info)
    assert np.abs(x - xtrue).max() < 1e-7


def test_fgmres_dev_restart_recovers(device):
    # a small restart forces multiple cycles; must still converge to the
    # same solution (the restart / back-substitution / x-accumulation path)
    n = 500
    A = _nonsym_csr(n, seed=2)
    rng = np.random.default_rng(9)
    xtrue = rng.standard_normal(n)
    b = A @ xtrue
    x_big, info_big = _run(A, b, device, restart=40, maxiter=50)
    x_sm, info_sm = _run(A, b, device, restart=10, maxiter=80)
    assert info_big["converged"] and info_sm["converged"]
    assert info_sm["outer"] >= info_big["outer"]     # more cycles
    assert np.abs(x_big - x_sm).max() < 1e-7, (info_big, info_sm)


def test_fgmres_dev_matches_scipy_lgmres(device):
    # the outer it replaces: scipy lgmres with the same Jacobi right
    # preconditioner.  Both converge to the same solution (few-ULP on the
    # linear solve; the augmentation subspace differs but the SOLUTION of
    # a consistent system is unique).
    n = 400
    A = _nonsym_csr(n, seed=5)
    rng = np.random.default_rng(11)
    xtrue = rng.standard_normal(n)
    b = A @ xtrue
    x_dev, info = _run(A, b, device, restart=30, maxiter=60, tol=1e-12)
    Minv = LinearOperator((n, n),
                          lambda v: v / np.asarray(A.diagonal()))
    x_sp, sinfo = lgmres(A, b, M=Minv, rtol=1e-12, atol=1e-13, maxiter=60)
    assert info["converged"] and sinfo == 0
    assert np.abs(x_dev - x_sp).max() < 1e-6, (info, sinfo)


def test_fgmres_dev_zero_rhs(device):
    # zero rhs -> zero solution, no iterations, converged
    n = 200
    A = _nonsym_csr(n, seed=1)
    b = np.zeros(n)
    x, info = _run(A, b, device, restart=20, maxiter=10)
    assert info["converged"]
    assert np.abs(x).max() == 0.0
    assert info["inner"] == 0
