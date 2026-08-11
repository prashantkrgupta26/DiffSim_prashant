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
import numpy as np
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


class CudssBackend(LinearBackend):
    """Factor+solve on a CUDA device via nvmath cuDSS DirectSolver.

    Plans + factorizes a fresh DirectSolver on every solve (freeing the prior
    one): the host-numpy assembly hands a new device CSR buffer each call, and
    one instance solves both the forward Jacobian and its transpose, so cuDSS
    operand-reuse (reset_operands) does not apply — see _solve_csr.  Uses
    DirectSolverOptions(blocking=True) to suppress the mt-planning layer thread-
    team leak (measured 2026-07-10: 18.6k threads -> libgomp failure).
    Assembly stays host-numpy; only the solve is on device.

    Construction is host-safe (torch imported lazily) so the engine + daisy-morph
    wiring unit-test on a CPU box; the DirectSolver path needs a real GPU.
    """

    def __init__(self, device="cuda:0"):
        self.device_str = str(device)
        self._torch = None
        self._device = None
        self._solver = None      # cuDSS DirectSolver (rebuilt per solve)

    def _lazy(self):
        if self._torch is None:
            import torch
            self._torch = torch
            self._device = torch.device(self.device_str)
        return self._torch

    def _solve_csr(self, A, b):
        torch = self._lazy()
        from nvmath.sparse.advanced import DirectSolver, DirectSolverOptions
        At = scipy_to_torch_csr(A, self._device, torch)
        bt = torch.as_tensor(b.astype("float64"), device=self._device)
        # Plan + factorize FRESH every solve.  Two reasons cuDSS operand-reuse
        # (reset_operands) is invalid for this backend:
        #  (1) scipy_to_torch_csr allocates a NEW device buffer each call, but
        #      reset_operands assumes the operand values are updated IN PLACE in
        #      the planned buffers -- passing new buffers invalidates the plan
        #      ("different buffers ... requires calling plan() and factorize()
        #      again", measured on gpubox 2026-08-07).
        #  (2) one backend instance solves BOTH the forward Jacobian J (solve)
        #      and the adjoint J^T (solve_T) during a single gradient: same nnz
        #      but DIFFERENT sparsity pattern, so an nnz-only reuse guard would
        #      reuse a plan across incompatible patterns.
        # Free the prior solver first to avoid leaking device buffers.  Options:
        # DirectSolverOptions(blocking=True) mirrors production and suppresses
        # the mt-planning gomp thread-team leak (physics/multiphase.py:1966-1968,
        # measured 2026-07-10).
        if self._solver is not None:
            try:
                self._solver.free()
            except Exception:
                pass
            self._solver = None
        self._solver = DirectSolver(
            At, bt, options=DirectSolverOptions(blocking=True))
        self._solver.plan()
        self._solver.factorize()
        return np.asarray(self._solver.solve().cpu())

    def solve(self, A, b):
        return self._solve_csr(A.tocsr(), b)

    def solve_T(self, A, b):
        return self._solve_csr(A.T.tocsr(), b)
