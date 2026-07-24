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


class TestPCGTorchAgnosticism:
    """Prove pcg() works with torch CPU tensors (no GPU needed).

    Task 3a requirement: pcg() must accept torch tensors in addition to numpy
    arrays so that TorchDistComm (Task 3b, GPU) can drop in without touching
    the CG core.  We verify:
      - SerialComm + small SPD matmul spmv + Jacobi precond converges.
      - The torch-tensor result matches the numpy result (same relative accuracy).
    """

    @pytest.fixture(autouse=True)
    def skip_if_no_torch(self):
        """Skip the whole class if torch is not installed."""
        pytest.importorskip("torch", reason="torch not installed")

    def _make_torch_system(self, m: int):
        """Return (A_torch, diag_inv_torch, b_torch, x_true_torch) on CPU."""
        import torch

        A_sp = laplacian_2d(m)
        rng = np.random.default_rng(1234)
        x_true_np = rng.standard_normal(m * m)
        b_np = np.asarray(A_sp @ x_true_np, dtype=np.float64)

        # Dense torch version (small test matrix)
        A_dense = torch.tensor(A_sp.toarray(), dtype=torch.float64)
        b_torch = torch.tensor(b_np, dtype=torch.float64)
        x_true_torch = torch.tensor(x_true_np, dtype=torch.float64)
        diag_inv = torch.tensor(1.0 / A_sp.diagonal(), dtype=torch.float64)

        return A_dense, diag_inv, b_torch, x_true_torch

    def test_torch_cpu_tensors_converge(self):
        """pcg() with torch CPU tensors + SerialComm converges to rel_err < 1e-8."""
        import torch

        m = 8
        A_dense, diag_inv, b_torch, x_true_torch = self._make_torch_system(m)

        comm = SerialComm()

        def spmv_torch(p):
            comm.exchange_halo(p)          # no-op for SerialComm
            return A_dense @ p             # torch matmul, stays fp64

        def jacobi_torch(r):
            return diag_inv * r

        x_torch, info = pcg(spmv_torch, jacobi_torch, b_torch, comm, rtol=1e-10)

        assert info["converged"], (
            f"torch-tensor PCG did not converge: iters={info['iters']}, "
            f"last_resid={info['resid_history'][-1]:.2e}"
        )
        assert isinstance(x_torch, torch.Tensor), "x_local should be a torch.Tensor"

        rel_err = float(
            torch.linalg.norm(x_torch - x_true_torch)
            / torch.linalg.norm(x_true_torch)
        )
        assert rel_err < 1e-8, (
            f"torch-tensor PCG solution accuracy too low: rel_err={rel_err:.2e}"
        )

    def test_torch_matches_numpy_result(self):
        """pcg() with torch tensors gives the same solution as with numpy arrays."""
        import torch

        m = 8
        A_sp = laplacian_2d(m)
        A_dense, diag_inv_torch, b_torch, _ = self._make_torch_system(m)

        b_np = b_torch.numpy()
        diag_inv_np = diag_inv_torch.numpy()

        comm = SerialComm()

        # numpy run
        def spmv_np(p):
            comm.exchange_halo(p)
            return np.asarray(A_sp @ p, dtype=np.float64)

        def jacobi_np(r):
            return diag_inv_np * r

        x_np, info_np = pcg(spmv_np, jacobi_np, b_np, comm, rtol=1e-10)

        # torch run
        def spmv_torch(p):
            comm.exchange_halo(p)
            return A_dense @ p

        def jacobi_torch(r):
            return diag_inv_torch * r

        x_torch, info_torch = pcg(spmv_torch, jacobi_torch, b_torch, comm, rtol=1e-10)

        assert info_np["converged"] and info_torch["converged"]

        # Solutions should agree to near-machine-epsilon
        x_torch_np = x_torch.numpy()
        np.testing.assert_allclose(
            x_torch_np, x_np, rtol=1e-10, atol=1e-12,
            err_msg="torch and numpy PCG solutions diverge"
        )


class TestRandomRHS3DPoisson:
    """N=1 (SerialComm) 3-D Poisson with random RHS exercises many CG iterations.

    The manufactured-solution RHS is a single eigenmode of the 7-point Laplacian,
    so MMS CG converges in ~1 iteration.  A fixed-seed random RHS populates the
    Krylov space more broadly and forces the iterative distributed loop to run
    many steps, which is the coverage gap closed by --rhs random.

    Gate: iters > 5  AND  rel_err < 1e-8 (dist-CG vs scipy spsolve).
    """

    @pytest.fixture(autouse=True)
    def skip_if_no_torch(self):
        pytest.importorskip("torch", reason="torch not installed")

    @pytest.fixture(autouse=True)
    def skip_if_no_diffsim(self):
        pytest.importorskip("diffsim", reason="diffsim not installed")

    @pytest.fixture(params=[(8, 4, 4), (12, 6, 6)])
    def dims(self, request):
        return request.param

    def test_random_rhs_multi_iter_and_accuracy(self, dims):
        """random RHS: iters > 5 AND rel_err < 1e-8 vs scipy spsolve."""
        import torch

        # Import the driver helpers — they live in scripts/, not a package.
        import importlib.util
        import pathlib

        driver_path = pathlib.Path(__file__).parent.parent / "scripts" / "nccl_cg_proof.py"
        spec = importlib.util.spec_from_file_location("nccl_cg_proof", driver_path)
        drv = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(drv)

        from diffsim.mesh.partition import slab_partition
        from diffsim.solvers.dist_cg import SerialComm, pcg
        from diffsim.solvers.dist_cg_torch import TorchDistComm

        device = torch.device("cpu")

        # N=1 partition (world_size=1, rank=0) — uses the serial halo path
        all_parts = slab_partition(dims, 1)
        part = all_parts[0]
        n_owned = len(part.owned)
        n_ghost = len(part.ghost)

        # Build local system with random RHS
        A_local_csr, b_local, diag_local, _ = drv.assemble_local_poisson(
            part, dims, device, rhs_mode="random"
        )

        # At N=1 we need a TorchDistComm or SerialComm-compatible object.
        # TorchDistComm requires an active process group; use SerialComm-like
        # wrapper that exercises exchange_halo (no-op) and allreduce (identity).
        serial_comm = SerialComm()

        # Wrap in a duck-typed adapter: pcg expects comm.allreduce_sum and
        # comm.exchange_halo; SerialComm provides both on numpy/torch tensors.
        padded = torch.zeros(n_owned + n_ghost, dtype=torch.float64, device=device)

        def spmv(p_owned):
            padded[:n_owned].copy_(p_owned)
            serial_comm.exchange_halo(padded)   # no-op at N=1
            p_np = padded.cpu().numpy()
            Ap_np = np.asarray(A_local_csr @ p_np, dtype=np.float64)
            return torch.tensor(Ap_np, dtype=torch.float64, device=device)

        def precond(r):
            return r / diag_local

        x_local, info = pcg(spmv, precond, b_local, serial_comm, rtol=1e-10)

        assert info["converged"], (
            f"random-RHS PCG did not converge (dims={dims}): "
            f"iters={info['iters']}, last_resid={info['resid_history'][-1]:.2e}"
        )
        assert info["iters"] > 5, (
            f"Expected iters > 5 for random RHS (dims={dims}), got {info['iters']}. "
            "If iters==1 the random RHS is still a near-eigenmode — check _random_rhs_full."
        )

        # Reference solve: same random RHS via assemble_global_poisson
        A_ref, b_ref, _ = drv.assemble_global_poisson(dims, rhs_mode="random")
        x_ref = scipy.sparse.linalg.spsolve(A_ref, b_ref)

        # Gather local solution (N=1 → owned is all nodes, already in gid order)
        x_dist = x_local.cpu().numpy()

        # Re-order: x_dist[loc_row] is for global gid = part.owned[loc_row]
        x_global = np.zeros(int(np.prod(dims)), dtype=np.float64)
        x_global[part.owned] = x_dist

        err = np.abs(x_global - x_ref)
        rel_err = float(np.max(err) / (np.max(np.abs(x_ref)) + 1e-300))

        assert rel_err < 1e-8, (
            f"random-RHS accuracy too low (dims={dims}): rel_err={rel_err:.3e}. "
            "Likely the distributed b and serial b disagree — check rhs_full indexing."
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
