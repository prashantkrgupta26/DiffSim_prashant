"""P2-R2b.1 diagnostic — RACE of AMGX strategies for the F (velocity) block.

p2r2b1_inner_quality.py measured: AMGX PBICGSTAB + scalar classical AMG
achieves only ~6.6e-2 on the L5 F block after 100 iterations (the S/C
pressure solves are fine). This races candidate fixes on the REAL F:

  scalar-classical : current config (baseline)
  equilibrated     : symmetric diagonal scaling D^-1/2 F D^-1/2, same config
                     (kills the 1.0-identity-row vs ~6e-4-interior-row scale
                     spread that can poison classical interpolation)
  jacobi-only      : PBICGSTAB + block-Jacobi (no AMG at all; F is
                     sigma*M-dominated, so plain Jacobi may be enough)
  eq+jacobi        : equilibrated + jacobi-only
  bsr-aggregation  : BSR(3x3) upload + AGGREGATION AMG (component-aware
                     coarsening for the interleaved vector block)

Env: LEVEL (5), DEVICE (cuda:0), ALPHA (100), ITERS (100), TOL (1e-6).
gpubox only (needs AMGX; march step uses cuDSS).
"""
import json
import os
import sys
import time

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, os.path.dirname(__file__))

from diffsim.solvers.amgx import _ensure_init  # noqa: E402
from diffsim.solvers.linsolve import solve_linear  # noqa: E402
import p2r0_task10_sphere_derisk as derisk  # noqa: E402

LEVEL = int(os.environ.get("LEVEL", "5"))
DEVICE = os.environ.get("DEVICE", "cuda:0")
ALPHA = float(os.environ.get("ALPHA", "100"))
ITERS = int(os.environ.get("ITERS", "100"))
TOL = float(os.environ.get("TOL", "1e-6"))

HERE = os.path.join(os.path.dirname(__file__), "..", "src", "diffsim",
                    "solvers", "amgx_configs")

captured = []
_march_cache = {}


def _spy(A, b, **kw):
    captured.append(dict(A=A.copy(), b=b.copy()))
    return solve_linear(A.tocsr(), b, solver="cudss", sym=False,
                        device=DEVICE, cache=_march_cache,
                        cache_key=f"march-{len(captured)}")


def base_cfg():
    with open(os.path.join(HERE, "PBICGSTAB_CLASSICAL_JACOBI.json")) as fh:
        cfg = json.load(fh)
    cfg["solver"]["max_iters"] = ITERS
    cfg["solver"]["tolerance"] = TOL
    cfg["solver"]["monitor_residual"] = 1
    cfg["solver"]["print_solve_stats"] = 0
    cfg["solver"]["obtain_timings"] = 0
    cfg["verbosity_level"] = 1
    return cfg


def jacobi_cfg():
    cfg = base_cfg()
    cfg["solver"]["preconditioner"] = {
        "solver": "BLOCK_JACOBI", "scope": "jacobi",
        "monitor_residual": 0, "print_solve_stats": 0}
    return cfg


def aggregation_cfg():
    cfg = base_cfg()
    cfg["solver"]["preconditioner"] = {
        "solver": "AMG", "algorithm": "AGGREGATION", "selector": "SIZE_2",
        "smoother": {"solver": "BLOCK_JACOBI", "scope": "jacobi",
                     "monitor_residual": 0, "print_solve_stats": 0},
        "presweeps": 1, "postsweeps": 1, "max_iters": 1, "cycle": "V",
        "max_levels": 50, "coarse_solver": "DENSE_LU_SOLVER",
        "scope": "amg", "monitor_residual": 0, "print_solve_stats": 0}
    return cfg


def run_amgx(name, cfg_dict, F, b, bsr=False):
    import pyamgx
    t0 = time.time()
    try:
        cfg = pyamgx.Config().create(json.dumps(cfg_dict))
        rsc = pyamgx.Resources().create_simple(cfg)
        M = pyamgx.Matrix().create(rsc)
        X = pyamgx.Vector().create(rsc)
        B = pyamgx.Vector().create(rsc)
        slv = pyamgx.Solver().create(rsc, cfg)
        if bsr:
            Fb = sp.bsr_matrix(F.tocsr(), blocksize=(3, 3))
            Fb.sort_indices()
            M.upload(Fb.indptr.astype(np.int32),
                     Fb.indices.astype(np.int32),
                     np.ascontiguousarray(Fb.data.ravel(), np.float64),
                     block_dims=[3, 3])
        else:
            Fc = F.tocsr()
            Fc.sort_indices()
            M.upload_CSR(Fc)
        slv.setup(M)
        t_setup = time.time() - t0
        x = np.zeros(F.shape[0])
        B.upload(np.ascontiguousarray(b, np.float64))
        X.upload(x)
        t0 = time.time()
        slv.solve(B, X, zero_initial_guess=True)
        t_solve = time.time() - t0
        X.download(x)
        rel = float(np.linalg.norm(F @ x - b) / np.linalg.norm(b))
        try:
            it, st = int(slv.iterations_number), slv.status
        except Exception:  # noqa: BLE001
            it, st = -1, "?"
        print(f"[fconfig] {name:18s}: ACHIEVED rel={rel:.3e} iters={it} "
              f"status={st} (setup {t_setup:.1f}s solve {t_solve:.1f}s)",
              flush=True)
        for obj in (slv, B, X, M, rsc, cfg):
            try:
                obj.destroy()
            except Exception:  # noqa: BLE001
                pass
    except Exception as e:  # noqa: BLE001
        print(f"[fconfig] {name:18s}: FAILED to run: {e}", flush=True)


def main():
    _ensure_init()
    fx = derisk.build_sphere_3d(DEVICE, LEVEL, 100.0)
    print(f"[fconfig] L{LEVEL}: n_free={len(fx['coords'])} "
          f"sf_faces={int(fx['sf'].elem.size)}", flush=True)
    derisk.solve_linear = _spy
    derisk.monolithic_cd(fx, alpha=ALPHA, dt=0.05, max_steps=1,
                         rate_tol=1e-12, solver="blockamgx")
    A = captured[0]["A"].tocsr()
    n_nodes, ndof = len(fx["coords"]), fx["ndof"]
    dim = ndof - 1
    idx = np.arange(n_nodes * ndof).reshape(n_nodes, ndof)
    u_ids = idx[:, :dim].ravel()
    F = A[u_ids][:, u_ids].tocsr()
    rng = np.random.default_rng(0)
    b = rng.standard_normal(F.shape[0])

    dg = np.abs(F.diagonal())
    print(f"[fconfig] F diag scale: min={dg.min():.3e} max={dg.max():.3e} "
          f"(spread {dg.max()/dg.min():.1e})", flush=True)
    Ds = sp.diags(1.0 / np.sqrt(dg))
    F_eq = (Ds @ F @ Ds).tocsr()

    # equilibrated variants: solve F_eq xh = Ds b (rel measured on F_eq —
    # the scaled system's convergence is what the preconditioner would see).
    b_eq = np.asarray(Ds @ b)

    run_amgx("scalar-classical", base_cfg(), F, b)
    run_amgx("equilibrated", base_cfg(), F_eq, b_eq)
    run_amgx("jacobi-only", jacobi_cfg(), F, b)
    run_amgx("eq+jacobi", jacobi_cfg(), F_eq, b_eq)
    run_amgx("bsr-aggregation", aggregation_cfg(), F, b, bsr=True)


if __name__ == "__main__":
    main()
