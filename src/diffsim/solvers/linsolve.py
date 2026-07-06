"""Unified linear-solve dispatch for the steppers (M1b GPU-residency work).

Backends:
  "splu"  — scipy SuperLU on the host (prototype default; unbeatable small,
            pays factorization on EVERY new matrix).
  "fused" — the single-sync device Krylov (solvers/krylov_dev): BiCGStab for
            nonsymmetric systems, CG for SPD; Jacobi preconditioned; the
            iterations live on the GPU, one scalar readback per
            check_every iterations (m1a finding 3 / P2 measurements).
  "amgx"  — NVIDIA AMGX (algebraic multigrid) through pyamgx, when built:
            AMG-preconditioned Krylov — flattens the O(h^-1) Jacobi
            iteration growth (P2 Explore (b)). Falls back with a clear
            error if pyamgx is unavailable.
  "cudss" — NVIDIA cuDSS (GPU direct sparse solver) via the official
            nvmath-python package: factorization AND triangular solves on
            the device — the GPU analogue of splu, right fit for the
            per-step-new-matrix stepper pattern (spec S5.4's "cuDSS
            analogue" made literal).

The dispatch is deliberately per-solve and stateless except for optional
operator caching by `cache` (a dict the caller owns): constant matrices
(mass, PPE stiffness) upload/factorize once.
"""
import numpy as np


def solve_linear(A, b, solver="splu", sym=False, tol=1e-10, maxiter=40000,
                 device="cuda:0", cache=None, cache_key=None):
    """Solve A x = b (scipy CSR A, host b). Returns host x.

    sym=True routes to CG/SPD paths. cache/cache_key: reuse device uploads
    or factorizations for constant matrices across steps."""
    A = A.tocsr()
    if cache is not None and cache_key is not None:
        # cheap staleness guard (evaluation solver-review item): cached
        # factorizations are for CONSTANT matrices — catch reuse of a key
        # after the matrix changed shape/pattern (values are the caller's
        # contract; a full value check would defeat the cache's purpose)
        fp = (A.shape, A.nnz, str(A.dtype))
        old = cache.get(("fingerprint", cache_key))
        if old is None:
            cache[("fingerprint", cache_key)] = fp
        elif old != fp:
            raise ValueError(
                f"solve_linear cache_key={cache_key!r} reused with a "
                f"different matrix (was {old}, now {fp}) — cached "
                f"factorizations are for constant matrices")
    if solver == "splu":
        from scipy.sparse.linalg import splu
        if cache is not None and cache_key is not None:
            lu = cache.get(("splu", cache_key))
            if lu is None:
                lu = splu(A.tocsc())
                cache[("splu", cache_key)] = lu
            return lu.solve(b)
        return splu(A.tocsc()).solve(b)

    if solver == "fused":
        from ..assembly.operators import CSROperator
        from .krylov_dev import cg_dev, bicgstab_dev
        if cache is not None and cache_key is not None:
            op = cache.get(("fusedop", cache_key))
            if op is None:
                op = CSROperator(A, device)
                cache[("fusedop", cache_key)] = op
        else:
            op = CSROperator(A, device)
        diag = np.asarray(A.diagonal())
        krylov = cg_dev if sym else bicgstab_dev
        x, info = krylov(op, b, tol=tol, atol=1e-13, maxiter=maxiter,
                         diag=diag, check_every=100)
        if not info.get("converged"):
            raise RuntimeError(f"fused solve failed: {info}")
        return x

    if solver == "amgx":
        from .amgx import amgx_solve
        return amgx_solve(A, b, sym=sym, tol=tol, maxiter=maxiter,
                          cache=cache, cache_key=cache_key)

    if solver == "cudss":
        # constant-matrix reuse: keep the factorized DirectSolver per key
        from nvmath.sparse.advanced import DirectSolver, direct_solver
        if cache is not None and cache_key is not None:
            slv = cache.get(("cudss", cache_key))
            if slv is None:
                slv = DirectSolver(A, np.ascontiguousarray(b, np.float64))
                slv.plan()
                slv.factorize()
                cache[("cudss", cache_key)] = slv
            slv.reset_operands(b=np.ascontiguousarray(b, np.float64))
            return np.asarray(slv.solve())
        return np.asarray(direct_solver(
            A, np.ascontiguousarray(b, np.float64)))

    raise ValueError(f"unknown solver '{solver}'")
