"""P2-R2b.1 diagnostic — Schur-approximation quality on the REAL monolithic
sphere saddle, with EXACT F and Schur inner solves.

With exact F, block-upper-triangular right-preconditioned FGMRES convergence
is governed PURELY by how well S~ approximates the true Schur
S = C - D F^{-1} G. This isolates the Schur formula from AMG/inner-solve
quality — the discriminating experiment for the L5 stagnation cliff.

Modes compared (schur_mode values of BlockAMGPreconditioner):
  cc      : sigma*Kp^{-1} + nu*Mp^{-1}   (Cahouet-Chabard, default)
  pspg_c  : C^{-1}                        (PSPG pressure block)
  diag_f  : (C - D diag(F)^{-1} G)^{-1}   (SIMPLE-style; SEES the SBM
            Nitsche penalty in F's near-surface diagonal, which CC's
            F~sigma*M assumption misses — a subspace growing ~h^-2)

Env: LEVEL (default 4), EXACT ("splu" default — Mac; "cudss" — gpubox, for
L5+ where host splu stalls), DEVICE (fixture device; default cpu for splu,
cuda:0 for cudss), ALPHA (100), STEPS (3), MODES ("cc,pspg_c,diag_f").

Measured (Mac, splu, ALPHA=100): L3 step1 cc=17 pspg_c=17 diag_f=11;
L4 step1 cc=21 pspg_c=23 diag_f=11; L4 step3 cc=28 pspg_c=32 diag_f=18.
diag_f is the best AND the most mesh-stable.
"""
import os
import sys
import time

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu

sys.path.insert(0, os.path.dirname(__file__))

from diffsim.solvers.block_precond import _fgmres  # noqa: E402
import p2r0_task10_sphere_derisk as derisk  # noqa: E402

LEVEL = int(os.environ.get("LEVEL", "4"))
EXACT = os.environ.get("EXACT", "splu")
DEVICE = os.environ.get("DEVICE", "cpu" if EXACT == "splu" else "cuda:0")
ALPHA = float(os.environ.get("ALPHA", "100"))
STEPS = int(os.environ.get("STEPS", "3"))
MODES = os.environ.get("MODES", "cc,pspg_c,diag_f").split(",")
RESTART = int(os.environ.get("RESTART", "300"))
MAXITER = int(os.environ.get("MAXITER", "3"))


def _direct(tag, cache):
    """Persistent exact solver factory: splu (host) or cuDSS (GPU)."""
    if EXACT == "splu":
        def make(M):
            lu = splu(sp.csc_matrix(M))
            return lambda r: lu.solve(r)
        return make
    from diffsim.solvers.linsolve import solve_linear

    def make(M, _tag=[0]):
        M = M.tocsr()
        key = f"{tag}-{id(M)}"

        def solve(r):
            return solve_linear(M, np.asarray(r, np.float64), solver="cudss",
                                sym=False, device=DEVICE, cache=cache,
                                cache_key=key)
        return solve
    return make


captured = []
_march_cache = {}


def _spy(A, b, **kw):
    cache = kw.get("cache") or {}
    meta = None
    for (t, _k), v in cache.items():
        if t == "blockamgx_meta":
            meta = v
    captured.append(dict(A=A.copy(), b=b.copy(), meta=meta))
    step_key = f"march-{len(captured)}"
    if EXACT == "splu":
        return splu(sp.csc_matrix(A)).solve(b)
    from diffsim.solvers.linsolve import solve_linear
    return solve_linear(A.tocsr(), b, solver="cudss", sym=False,
                        device=DEVICE, cache=_march_cache,
                        cache_key=step_key)


def run_case(cap, mode):
    A, b, meta = cap["A"].tocsr(), cap["b"], cap["meta"]
    n_nodes, ndof = meta["n_nodes"], meta["ndof"]
    dim = ndof - 1
    idx = np.arange(n_nodes * ndof).reshape(n_nodes, ndof)
    u_ids, p_ids = idx[:, :dim].ravel(), idx[:, dim].ravel()
    F = A[u_ids][:, u_ids].tocsr()
    G = A[u_ids][:, p_ids].tocsr()
    D = A[p_ids][:, u_ids].tocsr()
    C = A[p_ids][:, p_ids].tocsr()
    cache = {}
    make = _direct(mode, cache)
    f_inv = make(F)
    sigma, nu = meta["sigma"], meta["nu"]
    if mode == "cc":
        kp_inv = make(sp.csr_matrix(meta["Kp"]))
        Mp = meta["Mp_diag"]
        s_inv = lambda r: sigma * kp_inv(r) + nu * (r / Mp)  # noqa: E731
    elif mode == "pspg_c":
        s_inv = make(C)
    elif mode == "diag_f":
        S = (C - D @ sp.diags(1.0 / F.diagonal()) @ G).tocsr()
        s_inv = make(S)
    else:
        raise ValueError(mode)

    def apply_M(r):
        z_p = s_inv(r[p_ids])
        z_u = f_inv(r[u_ids] - G @ z_p)
        z = np.empty_like(r)
        z[u_ids], z[p_ids] = z_u, z_p
        return z

    x, it, rel = _fgmres(A, b, apply_M, tol=1e-9, restart=RESTART,
                         maxiter=MAXITER)
    return it, rel


def main():
    t0 = time.time()
    fx = derisk.build_sphere_3d(DEVICE, LEVEL, 100.0)
    print(f"[schurq] L{LEVEL} exact={EXACT}: n_free={len(fx['coords'])} "
          f"n_dof={len(fx['coords']) * fx['ndof']} "
          f"sf_faces={int(fx['sf'].elem.size)} ({time.time()-t0:.0f}s)",
          flush=True)
    derisk.solve_linear = _spy
    derisk.monolithic_cd(fx, alpha=ALPHA, dt=0.05, max_steps=STEPS,
                         rate_tol=1e-12, solver="blockamgx")
    print(f"[schurq] captured {len(captured)} systems", flush=True)
    for step in sorted({0, len(captured) - 1}):
        cap = captured[step]
        print(f"[schurq] --- step {step + 1} (of {len(captured)}) "
              f"n={cap['A'].shape[0]} ---", flush=True)
        for mode in MODES:
            t0 = time.time()
            it, rel = run_case(cap, mode)
            print(f"[schurq]   exact-F + {mode:8s}: iters={it:4d} "
                  f"rel={rel:.3e} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
