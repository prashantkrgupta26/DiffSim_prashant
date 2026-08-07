"""Pluggable linear-solver backends for the M-component adjoint engine.

The rung-1 engine (adjoint/multiphase.py) solves two square sparse systems per
committed step: the forward Newton update J dx = -R and the adjoint J^T lam =
rhs.  Rung 2 keeps the verified numpy assembly and swaps ONLY these solves onto
a CUDA device via cuDSS, so 256x128 (~131k DOF, past scipy splu's practical CPU
ceiling) becomes reachable while the three-way-verified adjoint math is
untouched.

ScipyBackend is the CPU default (bit-identical to the pre-backend splu path);
CudssBackend runs factor+solve on a CUDA device through nvmath's cuDSS
DirectSolver, mirroring physics/multiphase.MultiPhaseStepper._solve.
"""
from scipy.sparse.linalg import splu


class LinearBackend:
    """Solve A x = b and A^T x = b for a square scipy-sparse A."""

    def solve(self, A, b):
        raise NotImplementedError

    def solve_T(self, A, b):
        raise NotImplementedError


class ScipyBackend(LinearBackend):
    """Host splu — the rung-1 default; preserves exact pre-backend behavior."""

    def solve(self, A, b):
        return splu(A.tocsc()).solve(b)

    def solve_T(self, A, b):
        return splu(A.T.tocsc()).solve(b)


def scipy_to_torch_csr(A, device, torch):
    """scipy sparse -> torch.sparse_csr_tensor (int64 indices, float64 vals) on
    ``device``.  Mirrors the CSR handoff production uses for cuDSS zero-copy
    (physics/multiphase.MultiPhaseStepper.device_csr)."""
    A = A.tocsr()
    crow = torch.as_tensor(A.indptr.astype("int64"), device=device)
    col = torch.as_tensor(A.indices.astype("int64"), device=device)
    val = torch.as_tensor(A.data.astype("float64"), device=device)
    return torch.sparse_csr_tensor(crow, col, val, size=A.shape, device=device)
