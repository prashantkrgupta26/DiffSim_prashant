"""Tests for dist_cg.py — serial (N=1) correctness.

All tests run on CPU with SerialComm (no halo, no NCCL, no warp).

Test matrix: 2-D 5-point Laplacian on an m×m grid with homogeneous
Dirichlet BCs (the standard -del^2 u = f problem).  The assembled matrix
is symmetric positive definite and gives a well-conditioned test bench for
Jacobi-preconditioned CG.

Covered:
  - Jacobi preconditioned PCG converges; ||x - x_true|| / ||x_true|| < 1e-8
  - Iteration count matches scipy.sparse.linalg.cg within ±1
  - Determinism: two runs give identical iters and resid_history
  - Unpreconditioned (identity) PCG converges to the same solution
  - N=1 degenerate path: SerialComm.allreduce_sum returns x; exchange_halo is a no-op
"""
import numpy as np
import pytest
import scipy.sparse
import scipy.sparse.linalg

from diffsim.solvers.dist_cg import SerialComm, pcg, Comm


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def laplacian_2d(m: int) -> scipy.sparse.csr_matrix:
    """Return the 2-D 5-point Laplacian on an m×m grid as a CSR matrix.

    Homogeneous Dirichlet BCs; size = m*m x m*m.  The diagonal is 4 and
    off-diagonals are -1 (standard finite-difference -del^2 with h=1/(m+1)).
    This matrix is SPD.
    """
    n = m * m
    diagonals = [
        4.0 * np.ones(n),           # main diagonal
        -np.ones(n - 1),            # sub-diagonal (horizontal)
        -np.ones(n - 1),            # super-diagonal (horizontal)
        -np.ones(n - m),            # lower m-diagonal (vertical)
        -np.ones(n - m),            # upper m-diagonal (vertical)
    ]
    # Zero out the coupling that crosses row boundaries
    horiz_mask = np.ones(n - 1, dtype=bool)
    horiz_mask[m - 1::m] = False    # last node in each row

    A = scipy.sparse.diags(
        [diagonals[0],
         diagonals[1] * horiz_mask,
         diagonals[2] * horiz_mask,
         diagonals[3],
         diagonals[4]],
        offsets=[0, -1, 1, -m, m],
        format="csr",
    )
    return A.astype(np.float64)


def jacobi_precond(A: scipy.sparse.csr_matrix):
    """Return a Jacobi preconditioner callable: z = r / diag(A)."""
    diag_inv = 1.0 / np.array(A.diagonal(), dtype=np.float64)

    def precond(r: np.ndarray) -> np.ndarray:
        return diag_inv * r

    return precond


def identity_precond(r: np.ndarray) -> np.ndarray:
    """Identity preconditioner: z = r."""
    return r.copy()


def make_spmv_serial(A: scipy.sparse.csr_matrix, comm: SerialComm):
    """Return a serial spmv callable: calls exchange_halo (no-op) then A@p."""
    def spmv(p: np.ndarray) -> np.ndarray:
        # CONTRACT: call exchange_halo BEFORE the local matvec.
        # At N=1 this is a no-op; at N>1 it would fill ghost entries.
        comm.exchange_halo(p)
        return np.asarray(A @ p, dtype=np.float64)

    return spmv


def scipy_cg_iters(A, b, rtol, M=None):
    """Run scipy CG and return (x, num_iters).

    Uses ``rtol=`` (scipy >= 1.12) for the relative tolerance; falls back to
    the old ``tol=`` keyword for older installs.
    """
    iters = [0]

    def callback(_):
        iters[0] += 1

    try:
        x, info = scipy.sparse.linalg.cg(
            A, b, rtol=rtol, M=M, callback=callback
        )
    except TypeError:
        # Older scipy (< 1.12) uses `tol` instead of `rtol`
        x, info = scipy.sparse.linalg.cg(
            A, b, tol=rtol, M=M, callback=callback  # type: ignore[call-arg]
        )
    return x, iters[0]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(params=[8, 16])
def laplacian_system(request):
    """(A, b, x_true, m) for each grid size m."""
    m = request.param
    rng = np.random.default_rng(42)
    A = laplacian_2d(m)
    x_true = rng.standard_normal(m * m)
    b = np.asarray(A @ x_true, dtype=np.float64)
    return A, b, x_true, m


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSerialComm:
    """Unit tests for the SerialComm protocol implementation."""

    def test_allreduce_returns_x(self):
        comm = SerialComm()
        for v in [0.0, 1.0, -3.14, 1e-15]:
            assert comm.allreduce_sum(v) == float(v)

    def test_exchange_halo_noop(self):
        comm = SerialComm()
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        before = arr.copy()
        comm.exchange_halo(arr)
        np.testing.assert_array_equal(arr, before)

    def test_serial_comm_satisfies_protocol(self):
        assert isinstance(SerialComm(), Comm)


class TestPCGConvergence:
    """PCG convergence and accuracy tests with SerialComm."""

    def test_jacobi_converges_accuracy(self, laplacian_system):
        """Jacobi-preconditioned PCG: ||x - x_true|| / ||x_true|| < 1e-8."""
        A, b, x_true, m = laplacian_system
        comm = SerialComm()
        spmv = make_spmv_serial(A, comm)
        precond = jacobi_precond(A)
        rtol = 1e-10

        x, info = pcg(spmv, precond, b, comm, rtol=rtol)

        assert info["converged"], (
            f"PCG did not converge (m={m}): iters={info['iters']}, "
            f"last resid={info['resid_history'][-1]:.2e}"
        )

        rel_err = np.linalg.norm(x - x_true) / np.linalg.norm(x_true)
        assert rel_err < 1e-8, (
            f"Solution accuracy too low (m={m}): rel_err={rel_err:.2e}"
        )

    def test_jacobi_iters_match_scipy_within_1(self, laplacian_system):
        """Iteration count is within ±1 of scipy CG with same Jacobi precond."""
        A, b, x_true, m = laplacian_system
        comm = SerialComm()
        spmv = make_spmv_serial(A, comm)
        precond = jacobi_precond(A)
        rtol = 1e-10

        _, info = pcg(spmv, precond, b, comm, rtol=rtol)

        # scipy CG with the same Jacobi M and tolerance
        diag_inv = 1.0 / np.array(A.diagonal(), dtype=np.float64)
        M_lin = scipy.sparse.linalg.LinearOperator(
            shape=A.shape, matvec=lambda r: diag_inv * r, dtype=np.float64
        )
        _, scipy_iters = scipy_cg_iters(A, b, rtol=rtol, M=M_lin)

        our_iters = info["iters"]
        assert abs(our_iters - scipy_iters) <= 1, (
            f"Iteration count mismatch (m={m}): ours={our_iters}, "
            f"scipy={scipy_iters} (diff={abs(our_iters - scipy_iters)})"
        )

    def test_unpreconditioned_converges(self, laplacian_system):
        """Identity (no) preconditioner converges to the same solution."""
        A, b, x_true, m = laplacian_system
        comm = SerialComm()
        spmv = make_spmv_serial(A, comm)
        rtol = 1e-10

        x, info = pcg(spmv, identity_precond, b, comm, rtol=rtol)

        assert info["converged"], (
            f"Unpreconditioned PCG did not converge (m={m}): "
            f"iters={info['iters']}"
        )

        rel_err = np.linalg.norm(x - x_true) / np.linalg.norm(x_true)
        assert rel_err < 1e-8, (
            f"Unpreconditioned PCG solution accuracy too low (m={m}): "
            f"rel_err={rel_err:.2e}"
        )

    def test_jacobi_and_identity_same_solution(self, laplacian_system):
        """Jacobi and identity preconditioners converge to the same x."""
        A, b, x_true, m = laplacian_system
        comm = SerialComm()
        spmv = make_spmv_serial(A, comm)
        rtol = 1e-10

        x_jac, info_j = pcg(spmv, jacobi_precond(A), b, comm, rtol=rtol)
        x_id, info_i = pcg(spmv, identity_precond, b, comm, rtol=rtol)

        assert info_j["converged"] and info_i["converged"]

        # Solutions should agree to near-rtol accuracy
        np.testing.assert_allclose(x_jac, x_id, rtol=1e-7, atol=1e-10,
                                   err_msg=f"Jacobi and identity solutions differ (m={m})")


class TestPCGDeterminism:
    """Two runs with identical inputs produce identical output."""

    def test_determinism_iters(self, laplacian_system):
        """Two Jacobi-preconditioned PCG runs give identical iters."""
        A, b, x_true, m = laplacian_system
        comm = SerialComm()
        spmv = make_spmv_serial(A, comm)
        precond = jacobi_precond(A)
        rtol = 1e-10

        _, info1 = pcg(spmv, precond, b, comm, rtol=rtol)
        _, info2 = pcg(spmv, precond, b, comm, rtol=rtol)

        assert info1["iters"] == info2["iters"], (
            f"Non-deterministic iters (m={m}): {info1['iters']} != {info2['iters']}"
        )

    def test_determinism_resid_history(self, laplacian_system):
        """Two Jacobi-preconditioned PCG runs give identical resid_history."""
        A, b, x_true, m = laplacian_system
        comm = SerialComm()
        spmv = make_spmv_serial(A, comm)
        precond = jacobi_precond(A)
        rtol = 1e-10

        _, info1 = pcg(spmv, precond, b, comm, rtol=rtol)
        _, info2 = pcg(spmv, precond, b, comm, rtol=rtol)

        assert len(info1["resid_history"]) == len(info2["resid_history"]), (
            f"Resid history length differs (m={m})"
        )
        np.testing.assert_array_equal(
            info1["resid_history"], info2["resid_history"],
            err_msg=f"Resid history not deterministic (m={m})"
        )


class TestPCGEdgeCases:
    """Edge-case and degenerate-path tests."""

    def test_zero_rhs_converges_immediately(self):
        """b=0 → solution is x=0 with 0 iterations."""
        m = 4
        A = laplacian_2d(m)
        b = np.zeros(m * m, dtype=np.float64)
        comm = SerialComm()
        spmv = make_spmv_serial(A, comm)
        precond = jacobi_precond(A)

        x, info = pcg(spmv, precond, b, comm)

        assert info["converged"]
        np.testing.assert_array_equal(x, np.zeros(m * m, dtype=np.float64))

    def test_initial_guess_x0(self, laplacian_system):
        """PCG with x0 close to solution converges faster than from 0."""
        A, b, x_true, m = laplacian_system
        comm = SerialComm()
        spmv = make_spmv_serial(A, comm)
        precond = jacobi_precond(A)
        rtol = 1e-10

        # Warm start: x0 = x_true + small noise
        rng = np.random.default_rng(7)
        x0 = x_true + rng.standard_normal(m * m) * 1e-3

        _, info_warm = pcg(spmv, precond, b, comm, rtol=rtol, x0=x0)
        _, info_cold = pcg(spmv, precond, b, comm, rtol=rtol)

        assert info_warm["converged"], "Warm-start PCG did not converge"
        # Warm start should need no more iterations than cold start
        assert info_warm["iters"] <= info_cold["iters"], (
            f"Warm start used more iters ({info_warm['iters']}) "
            f"than cold start ({info_cold['iters']}) (m={m})"
        )

    def test_n1_degenerate_matches_serial_numpy(self):
        """N=1 PCG with SerialComm matches a direct numpy solve."""
        m = 6
        A = laplacian_2d(m)
        rng = np.random.default_rng(99)
        x_true = rng.standard_normal(m * m)
        b = np.asarray(A @ x_true, dtype=np.float64)

        comm = SerialComm()
        spmv = make_spmv_serial(A, comm)
        precond = jacobi_precond(A)

        x_pcg, info = pcg(spmv, precond, b, comm, rtol=1e-10)

        assert info["converged"]
        rel_err = np.linalg.norm(x_pcg - x_true) / np.linalg.norm(x_true)
        assert rel_err < 1e-8, f"N=1 degenerate path: rel_err={rel_err:.2e}"

    def test_info_keys(self, laplacian_system):
        """info dict must have iters, resid_history, converged."""
        A, b, x_true, m = laplacian_system
        comm = SerialComm()
        spmv = make_spmv_serial(A, comm)
        precond = jacobi_precond(A)

        _, info = pcg(spmv, precond, b, comm)

        assert "iters" in info
        assert "resid_history" in info
        assert "converged" in info
        assert isinstance(info["iters"], int)
        assert isinstance(info["resid_history"], list)
        assert isinstance(info["converged"], bool)

    def test_maxit_exhaustion_not_converged(self):
        """PCG with maxit too small must report converged=False and iters==maxit."""
        m = 16
        A = laplacian_2d(m)
        rng = np.random.default_rng(42)
        x_true = rng.standard_normal(m * m)
        b = np.asarray(A @ x_true, dtype=np.float64)

        comm = SerialComm()
        spmv = make_spmv_serial(A, comm)
        precond = jacobi_precond(A)

        # maxit=2 is far too few for the m=16 Laplacian to converge to rtol=1e-10
        maxit = 2
        _, info = pcg(spmv, precond, b, comm, rtol=1e-10, maxit=maxit)

        assert info["converged"] is False, (
            f"Expected converged=False with maxit={maxit}, "
            f"got converged={info['converged']}"
        )
        assert info["iters"] == maxit, (
            f"Expected iters=={maxit}, got iters={info['iters']}"
        )
        assert len(info["resid_history"]) >= 1, (
            f"Expected resid_history to be populated, got {info['resid_history']}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
