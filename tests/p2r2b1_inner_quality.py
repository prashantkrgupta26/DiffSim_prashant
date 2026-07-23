"""P2-R2b.1 diagnostic — what does AMGX ACTUALLY achieve on the real
monolithic blocks? (gpubox only; needs AMGX.)

The L5 exact-F Schur experiment (p2r2b1_schur_quality.py) showed outer
FGMRES needs only 9-36 iterations with EXACT inner solves, yet the AMGX
march grinds — so an inner solve is not delivering its nominal tolerance.
This probes each block directly: build _AMGXCycle(block) at a given
(iters, tol), solve against random RHS, and report the ACHIEVED relative
residual ||M z - b|| / ||b|| (plus AMGX's reported iteration count).

Env: LEVEL (5), DEVICE (cuda:0), ALPHA (100), ITERS (100), TOL (1e-6).
Blocks probed: F (velocity, interleaved 3-dof, SBM penalty), S (diag_f
Schur), C (PSPG p-p), Kp (pressure stiffness).
"""
import os
import sys
import time

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, os.path.dirname(__file__))

from diffsim.solvers.block_precond import _AMGXCycle  # noqa: E402
from diffsim.solvers.linsolve import solve_linear  # noqa: E402
import p2r0_task10_sphere_derisk as derisk  # noqa: E402

LEVEL = int(os.environ.get("LEVEL", "5"))
DEVICE = os.environ.get("DEVICE", "cuda:0")
ALPHA = float(os.environ.get("ALPHA", "100"))
ITERS = int(os.environ.get("ITERS", "100"))
TOL = float(os.environ.get("TOL", "1e-6"))

captured = []
_march_cache = {}


def _spy(A, b, **kw):
    captured.append(dict(A=A.copy(), b=b.copy()))
    return solve_linear(A.tocsr(), b, solver="cudss", sym=False,
                        device=DEVICE, cache=_march_cache,
                        cache_key=f"march-{len(captured)}")


def probe(name, M, sym, iters=ITERS, tol=TOL):
    M = M.tocsr()
    rng = np.random.default_rng(0)
    b = rng.standard_normal(M.shape[0])
    t0 = time.time()
    cyc = _AMGXCycle(M, sym=sym, cycles=iters, tol=tol)
    t_setup = time.time() - t0
    t0 = time.time()
    z = cyc.solve(b)
    t_solve = time.time() - t0
    rel = float(np.linalg.norm(M @ z - b) / np.linalg.norm(b))
    try:
        it = int(cyc.slv.iterations_number)
        st = cyc.slv.status
    except Exception:  # noqa: BLE001
        it, st = -1, "?"
    print(f"[innerq] {name:10s} n={M.shape[0]:7d} sym={int(sym)} "
          f"target tol={tol:.0e} iters<={iters}: ACHIEVED rel={rel:.3e} "
          f"amgx_iters={it} status={st} "
          f"(setup {t_setup:.1f}s solve {t_solve:.1f}s)", flush=True)
    cyc.destroy()
    return rel


def main():
    fx = derisk.build_sphere_3d(DEVICE, LEVEL, 100.0)
    print(f"[innerq] L{LEVEL}: n_free={len(fx['coords'])} "
          f"sf_faces={int(fx['sf'].elem.size)}", flush=True)
    derisk.solve_linear = _spy
    derisk.monolithic_cd(fx, alpha=ALPHA, dt=0.05, max_steps=1,
                         rate_tol=1e-12, solver="blockamgx")
    A = captured[0]["A"].tocsr()
    meta_nodes = len(fx["coords"])
    ndof = fx["ndof"]
    dim = ndof - 1
    idx = np.arange(meta_nodes * ndof).reshape(meta_nodes, ndof)
    u_ids, p_ids = idx[:, :dim].ravel(), idx[:, dim].ravel()
    F = A[u_ids][:, u_ids].tocsr()
    G = A[u_ids][:, p_ids].tocsr()
    D = A[p_ids][:, u_ids].tocsr()
    C = A[p_ids][:, p_ids].tocsr()
    S = (C - D @ sp.diags(1.0 / F.diagonal()) @ G).tocsr()

    probe("F(bicgstab)", F, sym=False)
    probe("S(bicgstab)", S, sym=False)
    probe("S(pcg)", S, sym=True)
    probe("C(pcg)", C, sym=True)


if __name__ == "__main__":
    main()
