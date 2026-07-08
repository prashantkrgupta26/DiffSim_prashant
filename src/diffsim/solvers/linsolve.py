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

_CUDSS_OPTS = ...          # lazily built by cudss_options()


def cudss_options():
    """DirectSolverOptions with multithreaded host planning
    (libcudss_mtlayer_gomp) when the layer ships with nvmath — measured
    NECESSARY for per-step refactorization loops (M3 bunny v2: a
    single-threaded plan() was a 3.5 h mostly-idle stall). Shared by
    solve_linear and the device-resident stepper/film solve paths."""
    global _CUDSS_OPTS
    if _CUDSS_OPTS is ...:
        from nvmath.sparse.advanced import DirectSolverOptions
        import glob as _glob
        mt = _glob.glob(
            "/home/bglab/Baskar/DiffSim/.venv/lib/python3.12/"
            "site-packages/nvidia/cu12/lib/libcudss_mtlayer_gomp.so*")
        _CUDSS_OPTS = (DirectSolverOptions(multithreading_lib=mt[0])
                       if mt else None)
    return _CUDSS_OPTS


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

    if solver == "blocktri":
        # task-#6 production recipe (findings 8f-i): FGMRES + block-
        # triangular preconditioner with EXACT F (cuDSS velocity block)
        # + diagC Schur — 2-3 iterations at sigma=0 where AMG diverged.
        # kwargs via cache: caller stores {"ndof": d+1} under
        # ("blocktri_meta", cache_key).
        import scipy.sparse as _sp
        from scipy.sparse.linalg import LinearOperator, lgmres
        meta = (cache or {}).get(("blocktri_meta", cache_key), {})
        ndof = meta.get("ndof", 3)
        n = A.shape[0] // ndof
        dim = ndof - 1
        u_ids = (np.arange(n)[:, None] * ndof
                 + np.arange(dim)[None, :]).ravel()
        p_ids = np.arange(n) * ndof + dim
        F = A[u_ids][:, u_ids].tocsr()
        G = A[u_ids][:, p_ids].tocsr()
        Cd = np.asarray(A[p_ids][:, p_ids].diagonal())
        Cd[Cd == 0] = 1.0
        from nvmath.sparse.advanced import DirectSolver
        slvF = DirectSolver(F, np.zeros(F.shape[0]))
        slvF.plan()
        slvF.factorize()

        def apply(r):
            r_u, r_p = r[u_ids], r[p_ids]
            z_p = r_p / np.abs(Cd)
            slvF.reset_operands(b=np.ascontiguousarray(r_u - G @ z_p))
            z_u = np.asarray(slvF.solve())
            z = np.empty_like(r)
            z[u_ids] = z_u
            z[p_ids] = z_p
            return z

        it = [0]
        x, info = lgmres(A, b, M=LinearOperator(A.shape, apply),
                         rtol=tol, atol=1e-13, maxiter=100,
                         callback=lambda _: it.__setitem__(0, it[0] + 1))
        if info != 0:
            raise RuntimeError(f"blocktri FGMRES not converged: {info}")
        return x

    if solver == "cudss":
        # constant-matrix reuse: keep the factorized DirectSolver per key
        from nvmath.sparse.advanced import DirectSolver, direct_solver
        _opts = cudss_options()
        if cache is not None and cache_key is not None:
            slv = cache.get(("cudss", cache_key))
            if slv is None:
                slv = DirectSolver(A, np.ascontiguousarray(b, np.float64),
                                   options=_opts)
                slv.plan()
                slv.factorize()
                cache[("cudss", cache_key)] = slv
            slv.reset_operands(b=np.ascontiguousarray(b, np.float64))
            return np.asarray(slv.solve())
        return np.asarray(direct_solver(
            A, np.ascontiguousarray(b, np.float64),
            options=_opts))

    raise ValueError(f"unknown solver '{solver}'")
