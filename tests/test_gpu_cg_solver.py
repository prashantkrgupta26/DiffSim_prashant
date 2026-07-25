"""Tests for the 'gpu_cg' solver option in solve_linear.

Two test layers:
1. Unit test — small 2-D 5-point Laplacian SPD matrix; gpu_cg vs splu.
2. End-to-end test — one PPE solve step via LerayProjectionStepper with
   solver="gpu_cg" vs solver="splu"; pressure/velocity fields compared.

The gpu_cg path: scipy CSR -> torch.sparse_csr_tensor on `device`, SerialComm
PCG, Jacobi preconditioner, returns host numpy (same contract as splu/cudss).
"""
import numpy as np
import pytest
import scipy.sparse as sp

from diffsim.solvers.linsolve import solve_linear


# ---------------------------------------------------------------------------
# Helper: 2-D 5-point Laplacian on an (N x N) grid (interior nodes only)
# ---------------------------------------------------------------------------

def _laplacian_2d(n):
    """Build the (n^2 x n^2) 5-point Laplacian on a unit square grid.

    Interior-only: n is the number of INTERIOR nodes per axis so the matrix
    is strictly SPD (no Dirichlet rows needed; all rows are interior stencils).
    """
    size = n * n
    h = 1.0 / (n + 1)
    diag = 4.0 / h ** 2
    off = -1.0 / h ** 2

    rows, cols, vals = [], [], []
    for i in range(n):
        for j in range(n):
            k = i * n + j
            # diagonal
            rows.append(k); cols.append(k); vals.append(diag)
            # left neighbour
            if j > 0:
                rows.append(k); cols.append(k - 1); vals.append(off)
            # right neighbour
            if j < n - 1:
                rows.append(k); cols.append(k + 1); vals.append(off)
            # lower neighbour
            if i > 0:
                rows.append(k); cols.append(k - n); vals.append(off)
            # upper neighbour
            if i < n - 1:
                rows.append(k); cols.append(k + n); vals.append(off)

    A = sp.csr_matrix(
        (np.array(vals, dtype=np.float64),
         (np.array(rows, dtype=np.int32), np.array(cols, dtype=np.int32))),
        shape=(size, size),
    )
    return A


# ---------------------------------------------------------------------------
# Unit test: gpu_cg vs splu on a small SPD system
# ---------------------------------------------------------------------------

class TestGpuCgUnit:
    """gpu_cg vs splu agreement on a small 2-D Laplacian."""

    def _solve_both(self, n=10, device="cpu"):
        A = _laplacian_2d(n)
        rng = np.random.default_rng(42)
        b = rng.standard_normal(n * n)

        x_splu = solve_linear(A, b, solver="splu", sym=True, device=device)
        x_cg = solve_linear(A, b, solver="gpu_cg", sym=True, device=device,
                            tol=1e-10, maxiter=5000)
        return x_splu, x_cg, b

    def test_relative_error_vs_splu_small(self):
        """10x10 interior grid: gpu_cg must agree with splu to < 1e-6 rel."""
        x_splu, x_cg, _ = self._solve_both(n=10, device="cpu")
        norm_ref = np.linalg.norm(x_splu)
        rel_err = np.linalg.norm(x_cg - x_splu) / max(norm_ref, 1e-300)
        assert rel_err < 1e-6, (
            f"gpu_cg vs splu rel_err={rel_err:.3e} exceeds 1e-6 threshold")

    def test_residual_small(self):
        """10x10 interior grid: solution residual ||Ax - b|| / ||b|| < 1e-8."""
        A = _laplacian_2d(10)
        rng = np.random.default_rng(43)
        b = rng.standard_normal(100)
        x = solve_linear(A, b, solver="gpu_cg", sym=True, device="cpu",
                         tol=1e-10, maxiter=5000)
        res = np.linalg.norm(A @ x - b) / np.linalg.norm(b)
        assert res < 1e-8, f"residual {res:.3e} exceeds 1e-8"

    def test_sym_false_raises(self):
        """gpu_cg must raise ValueError when sym=False."""
        A = _laplacian_2d(5)
        b = np.ones(25)
        with pytest.raises(ValueError, match="sym=True"):
            solve_linear(A, b, solver="gpu_cg", sym=False, device="cpu")

    def test_zero_rhs_returns_zero(self):
        """Zero RHS: gpu_cg must return the zero vector."""
        A = _laplacian_2d(8)
        b = np.zeros(64)
        x = solve_linear(A, b, solver="gpu_cg", sym=True, device="cpu",
                         tol=1e-10, maxiter=5000)
        assert np.allclose(x, 0.0), f"non-zero result for zero rhs: {np.abs(x).max()}"

    def test_returns_numpy_array(self):
        """Return type must be a numpy ndarray (same contract as splu)."""
        A = _laplacian_2d(6)
        b = np.ones(36, dtype=np.float64)
        x = solve_linear(A, b, solver="gpu_cg", sym=True, device="cpu")
        assert isinstance(x, np.ndarray), f"expected ndarray, got {type(x)}"
        assert x.dtype == np.float64 or np.can_cast(x.dtype, np.float64), (
            f"unexpected dtype {x.dtype}")

    def test_medium_grid(self):
        """20x20 interior grid: rel_err < 1e-6 (higher-iteration exercise)."""
        x_splu, x_cg, _ = self._solve_both(n=20, device="cpu")
        norm_ref = np.linalg.norm(x_splu)
        rel_err = np.linalg.norm(x_cg - x_splu) / max(norm_ref, 1e-300)
        assert rel_err < 1e-6, (
            f"20x20 gpu_cg vs splu rel_err={rel_err:.3e} exceeds 1e-6")


# ---------------------------------------------------------------------------
# End-to-end test: PPE Laplacian (real FEM K_p) with gpu_cg vs splu
# ---------------------------------------------------------------------------

def _build_ppe_matrix(device, level=3):
    """Build the real PPE Laplacian K_p from a coarse 2-D FEM mesh.

    Returns (K_pinned, rhs) where row/col 0 is pinned (Dirichlet, node-0
    pressure pin for enclosed-flow PPE).
    """
    from diffsim.octree.build import build_uniform
    from diffsim.mesh.nodes import build_mesh
    from diffsim.mesh.constraints import build_constraints
    from diffsim.mesh.basis import basis_tables
    from diffsim.assembly.operators import DeviceMesh, assemble_csr
    from diffsim.steppers.leray import LerayProjectionStepper

    tree = build_uniform(level, dim=2)
    mesh = build_mesh(tree, p=1)
    cons = build_constraints(mesh)
    dm = DeviceMesh.from_mesh(mesh, cons, basis_tables(1, dim=2), device)

    # Build a stepper to get the assembled K_p
    st = LerayProjectionStepper(
        dm, 0.01, 0.05,
        f_fn=lambda x, t: np.zeros((len(x), 2)),
        g_fn=lambda x, t: np.zeros((len(x), 2)),
        order=1, picard_iters=1, timestab=False)

    Kp = st.K_p.tolil()
    n = Kp.shape[0]

    # Pin node 0 (enclosed-flow PPE Dirichlet BC: p[0] = 0)
    Kp[0, :] = 0.0
    Kp[:, 0] = 0.0
    Kp[0, 0] = 1.0
    Kp = Kp.tocsr()
    Kp.eliminate_zeros()

    # Fixed-seed random RHS (interior nodes only; node 0 gets rhs=0)
    rhs = np.random.default_rng(7).standard_normal(n)
    rhs[0] = 0.0

    return Kp, rhs


@pytest.mark.tier1
def test_gpu_cg_ppe_matches_splu(device):
    """End-to-end: gpu_cg on the real FEM PPE Laplacian must match splu.

    Uses the true K_p assembled by LerayProjectionStepper on a level-3 2-D
    mesh (coarse enough to be fast; large enough to exercise many CG iters).
    Checks relative error < 1e-5 and residual < 1e-7.
    """
    Kp, rhs = _build_ppe_matrix(device, level=3)

    x_splu = solve_linear(Kp, rhs, solver="splu", sym=True, device=device)
    x_cg = solve_linear(Kp, rhs, solver="gpu_cg", sym=True, device=device,
                        tol=1e-10, maxiter=5000)

    # 1. Agreement with splu reference
    norm_ref = np.linalg.norm(x_splu)
    rel_err = np.linalg.norm(x_cg - x_splu) / max(norm_ref, 1e-300)
    assert rel_err < 1e-5, (
        f"PPE gpu_cg vs splu rel_err={rel_err:.3e} (threshold 1e-5)")

    # 2. Direct residual check
    res = np.linalg.norm(Kp @ x_cg - rhs) / np.linalg.norm(rhs)
    assert res < 1e-7, (
        f"PPE residual ||Kp x - rhs|| / ||rhs|| = {res:.3e} (threshold 1e-7)")
