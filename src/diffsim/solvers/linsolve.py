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
        import os as _os
        try:
            import nvidia
            _roots = list(nvidia.__path__)
        except ImportError:
            _roots = []
        mt = []
        for _r in _roots:
            mt = _glob.glob(_os.path.join(
                _r, "cu12", "lib", "libcudss_mtlayer_gomp.so*"))
            if mt:
                break
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

    if solver == "blockch":
        # Two-factor Schur block preconditioner for the mixed CH Newton
        # system  J = [[sigma M, m K], [-(f'' M + kap K), M]]  (node-major
        # 2-dof (c, mu)), Pearson-Wathen-matching class. Schur onto c:
        #     S = sigma M + m K M^{-1} (F + kap K),   F = Int f'' N N,
        # approximated by the NONSYMMETRIC two-factor form
        #     S~ = W1 M^{-1} W2
        #     W1 = sqrt(sigma) M + sqrt(m kap) K
        #     W2 = W1 + (m/sqrt(sigma)) F        (F SIGNED, not clipped)
        # Everything is recovered from J's own constrained blocks:
        # K = J_cmu/m, F = -J_muc - kap K.
        # MEASURED DESIGN LAWS (2026-07-09, three dumped offender
        # systems: FH quench sigma=500 with f'' to 9984; poly sigma=50
        # biharmonic-dominated; poly sigma=50 Newton-transient with
        # f'' to 4100):
        # (1) the classical single-factor form (F dropped, the proven
        #     [1/2,1]-bound preconditioner) DIVERGES for the logarithmic
        #     potential even with exact mass solves;
        # (2) dropping W1's sqrt(m kap)K from the second factor DIVERGES
        #     in the biharmonic-dominated regime;
        # (3) CLIPPING F to its positive part DIVERGES on large-
        #     curvature Newton transients (200 its where signed F takes
        #     88) — carry the sign, solve W2 with GMRES not CG;
        # (4) lumped mass in the pre/back-substitution DIVERGES at large
        #     f'' (the lumping error is amplified by F): consistent-mass
        #     CG solves are load-bearing.
        # Signed-F verdict across the three offenders: 32 / 53 / 88 its
        # (1-4 typical away from pathologies). The exact-Schur fallback
        # (inner GMRES on S, W-preconditioned) is 2-3 outer its
        # everywhere at ~5-10x the cost per apply — the recorded escape
        # hatch if a regime defeats the two-factor form.
        # (The published OSC pipeline, Bergermann et al. CiCP 2023,
        # sidesteps all of this by going semi-implicit AND replacing the
        # log with a polynomial; this form keeps the fully-implicit
        # Newton on the true regularized log.)
        # Apply, given r = (r_c, r_mu):
        #     a    = W1^{-1} (r_c - J_cmu M^{-1} r_mu)
        #     z_c  = W2^{-1} (M a)
        #     z_mu = M^{-1} (r_mu - J_muc z_c)
        # W1: Jacobi-CG (SPD); W2: Jacobi-GMRES (indefinite where f''<0);
        # M: Jacobi-CG. Outer: lgmres. meta via cache:
        # {"sigma","m","kappa"} under ("blockch_meta", cache_key) —
        # refresh when sigma (dt/BDF) changes.
        from scipy.sparse.linalg import (LinearOperator, cg as _cg,
                                         gmres as _gmres,
                                         lgmres as _lgmres)
        meta = (cache or {}).get(("blockch_meta", cache_key))
        if meta is None:
            raise ValueError("blockch requires ('blockch_meta', cache_key) "
                             "= {'sigma','m','kappa'} in cache")
        sig, mmo, kap = meta["sigma"], meta["m"], meta["kappa"]
        n = A.shape[0] // 2
        ci = np.arange(n) * 2
        mi = ci + 1
        Acm = A[ci][:, mi].tocsr()
        Amc = A[mi][:, ci].tocsr()
        Amm = A[mi][:, mi].tocsr()          # the (constrained) mass matrix
        Acc = A[ci][:, ci].tocsr()          # sigma * mass (+ Dirichlet rows)
        K = (Acm / mmo).tocsr()
        F = (-Amc - kap * K).tocsr()        # signed curvature (law 3)
        W1 = ((np.sqrt(sig) / sig) * Acc + np.sqrt(mmo * kap) * K).tocsr()
        W2 = (W1 + (mmo / np.sqrt(sig)) * F).tocsr()
        inner_it = [0]

        def _jacobi(W):
            d = W.diagonal().copy()
            d[d == 0] = 1.0
            return LinearOperator(W.shape, lambda v: v / d)

        _cb = lambda *_: inner_it.__setitem__(0, inner_it[0] + 1)
        if meta.get("inners") == "device":
            # DEVICE inners: the fused single-sync Krylov stack. W1/M are
            # SPD -> cg_dev; W2 is indefinite where f'' < 0 -> bicgstab_dev.
            # Matrices ride to the device once per Newton solve.
            from ..assembly.operators import CSROperator
            from .krylov_dev import cg_dev, bicgstab_dev
            opM = CSROperator(Amm, device)
            opW1 = CSROperator(W1, device)
            opW2 = CSROperator(W2, device)
            dgM, dg1, dg2 = (np.asarray(X.diagonal()).copy()
                             for X in (Amm, W1, W2))
            for dg in (dgM, dg1, dg2):
                dg[dg == 0] = 1.0
            dg2 = np.abs(dg2)               # Jacobi sign-guard (indefinite)

            def _dev(op, y, dg, rtol, krylov, label):
                x_, info = krylov(op, y, tol=rtol, atol=1e-13,
                                  maxiter=4000, diag=dg, check_every=50)
                if not info.get("converged"):
                    raise RuntimeError(f"blockch {label} device solve: "
                                       f"{info}")
                inner_it[0] += info.get("iters", 0)
                return x_

            _msolve = lambda y: _dev(opM, y, dgM, 1e-10, cg_dev, "mass")
            _w1solve = lambda y: _dev(opW1, y, dg1, 1e-8, cg_dev, "W1")
            _w2solve = lambda y: _dev(opW2, y, dg2, 1e-8, bicgstab_dev,
                                      "W2")
        else:
            MjM, Mj1, Mj2 = _jacobi(Amm), _jacobi(W1), _jacobi(W2)

            def _msolve(y):
                z, info = _cg(Amm, y, M=MjM, rtol=1e-10, atol=0.0,
                              maxiter=1000, callback=_cb)
                if info != 0:
                    raise RuntimeError(
                        f"blockch mass CG not converged: {info}")
                return z

            def _w1solve(y):
                z, info = _cg(W1, y, M=Mj1, rtol=1e-8, atol=0.0,
                              maxiter=3000, callback=_cb)
                if info != 0:
                    raise RuntimeError(
                        f"blockch W1 CG not converged: {info}")
                return z

            def _w2solve(y):
                z, info = _gmres(W2, y, M=Mj2, rtol=1e-8, atol=0.0,
                                 maxiter=3000, restart=100, callback=_cb,
                                 callback_type="legacy")
                if info != 0:
                    raise RuntimeError(
                        f"blockch W2 GMRES not converged: {info}")
                return z

        def apply(r):
            rc, rm = r[ci], r[mi]
            a = _w1solve(rc - Acm @ _msolve(rm))
            zc = _w2solve(Amm @ a)
            zm = _msolve(rm - Amc @ zc)
            z = np.empty_like(r)
            z[ci] = zc
            z[mi] = zm
            return z

        it = [0]
        x, info = _lgmres(A, b, M=LinearOperator(A.shape, apply),
                          rtol=tol, atol=1e-13, maxiter=100,
                          callback=lambda _: it.__setitem__(0, it[0] + 1))
        if info != 0:
            # ESCALATE to the exact-Schur fallback: inner GMRES on the
            # true S = J_cc + m K M^{-1} H (matrix-free), preconditioned
            # by the W1 square-root form — measured 2-3 outer its on
            # every offender the two-factor form has ever lost.
            H = (-Amc).tocsr()
            Sc = LinearOperator((n, n), lambda v:
                                Acc @ v + mmo * (K @ _msolve(H @ v)))

            def _schur(y):
                zz, sinfo = _gmres(Sc, y,
                                   M=LinearOperator(
                                       (n, n),
                                       lambda v: _w1solve(Amm @ _w1solve(v))),
                                   rtol=1e-8, atol=0.0, maxiter=800,
                                   restart=160, callback=_cb,
                                   callback_type="legacy")
                if sinfo != 0:
                    raise RuntimeError(
                        f"blockch fallback Schur GMRES: {sinfo}")
                return zz

            def apply_fb(r):
                rc, rm = r[ci], r[mi]
                zc = _schur(rc - Acm @ _msolve(rm))
                zm = _msolve(rm - Amc @ zc)
                z = np.empty_like(r)
                z[ci] = zc
                z[mi] = zm
                return z

            it[0] = 0
            x, info = _lgmres(A, b, M=LinearOperator(A.shape, apply_fb),
                              rtol=tol, atol=1e-13, maxiter=40,
                              callback=lambda _:
                              it.__setitem__(0, it[0] + 1))
            if info != 0:
                raise RuntimeError(
                    f"blockch fallback FGMRES not converged: {info}")
            it[0] += 1000        # mark fallback path in the iters record
        if cache is not None:
            cache[("blockch_iters", cache_key)] = (it[0], inner_it[0])
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
