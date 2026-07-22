"""P2-R2b.1 Task 1 — BlockAMGPreconditioner AMGX de-risk probe (gpubox helper,
NOT an in-CI gate).

Front-loads THE critical risk for R2b.1: `BlockAMGPreconditioner` builds TWO
`_AMGXCycle` objects (one for the velocity block F, one for the pressure
stiffness Kp), and each `_AMGXCycle.__init__` creates its OWN
`pyamgx.Resources()`. Per `src/diffsim/solvers/amgx.py` (line ~82, LIFETIME
RULE) multiple live Resources sets in one process may SEGFAULT inside AMGX.
This preconditioner has NEVER run end-to-end, so before wiring it into the 3-D
march we must confirm on gpubox that:

  STEP A  — it even INSTANTIATES (2 coexisting Resources) without a segfault.
  STEP B  — `solve_block_preconditioned` CONVERGES on a real (small) monolithic
            SBM-NS saddle, matching a splu direct solve to rel-err < 1e-6, and
            with a sane outer-FGMRES iteration count.

If STEP A segfaults the whole process, THAT is the finding (no traceback will
print — the process dies): the remediation is to refactor
`BlockAMGPreconditioner` to create ONE shared `pyamgx.Resources` used by both
`_AMGXCycle`s, mirroring `amgx_solve`'s singleton.

Run on gpubox (needs AMGX/GPU):
    .venv/bin/python tests/p2r2b1_blockamgx_derisk.py

This module CANNOT run on the Mac (no pyamgx); it exits early with a note.
"""
import os
import sys
import time

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu

sys.path.insert(0, os.path.dirname(__file__))

from diffsim.api.ns_bricks import assemble_linear_ns
from diffsim.physics.poisson import gauss_points
from diffsim.sbm.vector import sbm_vector_dirichlet
from diffsim.steppers.leray import LerayProjectionStepper

# reuse the exact 3-D sphere fixture + monolithic assembly conventions.
from p2r0_task10_sphere_derisk import build_sphere_3d, qref, R, CTR, U_IN  # noqa: E402


def build_saddle(fx, alpha, dt):
    """Assemble ONE monolithic SBM-NS saddle (first Picard step, zero initial
    velocity) the SAME way monolithic_cd does, plus the pressure stiffness Kp
    and mass-diagonal Mp_diag for the block preconditioner.

    Returns (A, b, meta) where meta is the ('blockamgx_meta', key) contract."""
    dm, mesh, cons = fx["dm"], fx["mesh"], fx["cons"]
    sf, geo = fx["sf"], fx["geo"]
    nu, ndof, dim = fx["nu"], fx["ndof"], fx["dim"]
    coords = fx["coords"]
    T = cons.T.tocsr()
    T_vec = sp.kron(T, sp.identity(ndof, format="csr"), format="csr")
    nfree = T.shape[1]
    gauss_points(mesh, dm.tables_by_p)  # warm/validate quadrature (as mono)
    strong = np.where(fx["strong_mask"])[0]
    g_strong = fx["u_inf"][strong]

    # SBM Dirichlet block (Nitsche) — constrained, exactly as monolithic_cd.
    Af, bf = sbm_vector_dirichlet(
        dm, sf, geo, lambda y: np.zeros((len(y), dim)), nu, ndof, alpha=alpha)
    Af_c = (T_vec.T @ Af @ T_vec).tocsr()
    bf_c = np.asarray(T_vec.T @ bf)

    # first step: zero initial velocity -> aq/dq/fq all zero. Per-bin GP
    # arrays have ne*nqp rows (matching assemble_linear_ns' expectation).
    aq, dq, fq = {}, {}, {}
    for pv, bd in dm.bins.items():
        nrows = len(bd["eids"]) * bd["nqp"]
        aq[pv] = np.zeros((nrows, dim))
        dq[pv] = np.zeros(nrows)
        fq[pv] = np.zeros((nrows, dim))
    sigma = 1.0 / dt
    A, b = assemble_linear_ns(dm, aq, dq, fq, nu, sigma=sigma)
    A = (A + Af_c).tolil()
    b = np.asarray(b) + bf_c

    # strong-Dirichlet identity rows (velocity BC) — track their monolithic ids
    dir_rows = []
    for k, i in enumerate(strong):
        for c in range(dim):
            r = int(i * ndof + c)
            A.rows[r] = [r]
            A.data[r] = [1.0]
            b[r] = g_strong[k, c]
            dir_rows.append(r)
    # pressure pin (single node) — an identity row on the pressure DOF.
    pin = int(np.argmax(coords.sum(1))) * ndof + dim
    A.rows[pin] = [pin]
    A.data[pin] = [1.0]
    b[pin] = 0.0
    A = A.tocsr()
    dir_rows = np.asarray(dir_rows, dtype=np.int64)

    # pressure stiffness Kp (the Leray PPE operator) + mass diagonal, reused
    # from LerayProjectionStepper on the SAME device mesh (K_p = T^T K T,
    # M = T^T M T on free nodes). Pin the same pressure node in Kp so its
    # Cahouet-Chabard inverse is nonsingular.
    stp = LerayProjectionStepper(
        dm, nu, dt,
        lambda x, t: np.zeros((len(x), dim)),      # f_fn (body force)
        lambda x, t: np.zeros((len(x), dim)),      # g_fn (Dirichlet data)
        solver="splu")
    Kp = stp.K_p.tolil()
    p_pin = int(np.argmax(coords.sum(1)))         # pressure-node index (scalar)
    Kp.rows[p_pin] = [p_pin]
    Kp.data[p_pin] = [1.0]
    Kp = Kp.tocsr()
    Mp_diag = np.asarray(stp.M.diagonal())
    Mp_diag[p_pin] = 1.0

    meta = dict(n_nodes=nfree, ndof=ndof, Kp=Kp, Mp_diag=Mp_diag,
                sigma=sigma, nu=nu, dir_rows=dir_rows)
    return A, b, meta


def main():
    try:
        import pyamgx  # noqa: F401
    except Exception as e:  # pragma: no cover - Mac path
        print(f"[r2b1-derisk] pyamgx not importable ({e}); this probe needs "
              f"AMGX/GPU (run on gpubox). SKIPPING.", flush=True)
        return

    level = int(os.environ.get("LEVEL", "3"))
    Re = float(os.environ.get("RE", "100"))
    dt = float(os.environ.get("DT", "0.05"))
    alpha = float(os.environ.get("ALPHA", "20"))
    device = "cuda:0" if os.environ.get("DIFFSIM_CUDA") else "cpu"

    print(f"[r2b1-derisk] build_sphere_3d level={level} Re={Re} dt={dt} "
          f"alpha={alpha} device={device}", flush=True)
    t0 = time.time()
    fx = build_sphere_3d(device, level, Re)
    print(f"[r2b1-derisk] fixture: n_free={len(fx['coords'])} "
          f"sf.elem={fx['sf'].elem.size} ({time.time()-t0:.1f}s)", flush=True)

    t0 = time.time()
    A, b, meta = build_saddle(fx, alpha, dt)
    print(f"[r2b1-derisk] saddle: A={A.shape} nnz={A.nnz} "
          f"n_nodes={meta['n_nodes']} ndof={meta['ndof']} "
          f"dir_rows={meta['dir_rows'].size} ({time.time()-t0:.1f}s)",
          flush=True)

    # ---------- STEP A: 2-Resources instantiation / segfault check ----------
    from diffsim.solvers.block_precond import (
        BlockAMGPreconditioner, solve_block_preconditioned)
    print("[r2b1-derisk] STEP A: instantiate BlockAMGPreconditioner "
          "(creates 2 AMGX Resources) ...", flush=True)
    try:
        t0 = time.time()
        pre = BlockAMGPreconditioner(
            A, meta["n_nodes"], meta["ndof"], meta["Kp"], meta["Mp_diag"],
            meta["sigma"], meta["nu"], dir_rows=meta["dir_rows"])
        print(f"[r2b1-derisk] STEP A INSTANTIATE: OK "
              f"({time.time()-t0:.1f}s) — 2 coexisting Resources survived",
              flush=True)
    except Exception as e:
        print(f"[r2b1-derisk] STEP A INSTANTIATE: FAILED with {type(e).__name__}"
              f": {e}", flush=True)
        print("[r2b1-derisk] verdict = instantiate_failed", flush=True)
        return

    # ---------- STEP B: convergence vs splu ----------
    print("[r2b1-derisk] STEP B: solve_block_preconditioned vs splu ...",
          flush=True)
    t0 = time.time()
    x_direct = splu(A.tocsc()).solve(b)
    t_splu = time.time() - t0

    t0 = time.time()
    try:
        x_block, outer = solve_block_preconditioned(
            A, b, pre, tol=1e-8, maxiter=200)
    except Exception as e:
        print(f"[r2b1-derisk] STEP B SOLVE: FAILED with {type(e).__name__}"
              f": {e}", flush=True)
        print("[r2b1-derisk] verdict = solve_failed", flush=True)
        return
    t_block = time.time() - t0

    rel = np.linalg.norm(x_block - x_direct) / np.linalg.norm(x_direct)
    print(f"[r2b1-derisk] STEP B: rel-err={rel:.3e} outer_fgmres_iters={outer} "
          f"(splu {t_splu:.2f}s, block {t_block:.2f}s)", flush=True)
    gate = rel < 1e-6
    print(f"[r2b1-derisk] GATE rel-err<1e-6: {'PASS' if gate else 'FAIL'} "
          f"(rel={rel:.3e})", flush=True)
    print(f"[r2b1-derisk] verdict = "
          f"{'converged' if gate else 'not_converged'}", flush=True)


if __name__ == "__main__":
    main()
