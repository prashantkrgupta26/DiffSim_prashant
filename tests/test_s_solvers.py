import numpy as np
import pytest
import warp as wp
from diffsim.solvers.krylov import cg, bicgstab

pytestmark = pytest.mark.tier3

class DenseOp:
    """Test operator wrapping a dense SPD/nonsym matrix on device via numpy."""
    def __init__(self, A, device):
        self.A, self.device, self.n_free = A, device, A.shape[0]
    def matvec(self, x, y):
        r = self.A @ x.numpy()
        wp.copy(y, wp.array(r, dtype=wp.float64, device=self.device))

def _spd(n, rng):
    Q = rng.standard_normal((n, n))
    return Q @ Q.T + n * np.eye(n)

def test_S2_cg_manufactured_system(device):
    rng = np.random.default_rng(7)
    A = _spd(80, rng)
    x_exact = rng.standard_normal(80)
    op = DenseOp(A, device)
    x, info = cg(op, A @ x_exact, tol=1e-12)
    assert info["converged"]
    assert np.linalg.norm(x - x_exact) / np.linalg.norm(x_exact) < 1e-8

def test_S2_bicgstab_nonsymmetric(device):
    rng = np.random.default_rng(8)
    A = _spd(60, rng) + 0.3 * rng.standard_normal((60, 60))
    x_exact = rng.standard_normal(60)
    op = DenseOp(A, device)
    x, info = bicgstab(op, A @ x_exact, tol=1e-12)
    assert info["converged"]
    assert np.linalg.norm(x - x_exact) / np.linalg.norm(x_exact) < 1e-7

def test_S1_reported_residual_matches_recomputed(device):
    rng = np.random.default_rng(9)
    A = _spd(50, rng)
    b = rng.standard_normal(50)
    op = DenseOp(A, device)
    x, info = cg(op, b, tol=1e-10)
    relres = np.linalg.norm(b - A @ x) / np.linalg.norm(b)
    assert abs(relres - info["relres"]) < 1e-9
