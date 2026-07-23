"""P2-R2b.1 diagnostic — F-block AMGX race, round 2 (subprocess-isolated).

Round 1 (p2r2b1_f_configs.py, L5): scalar-classical 6.6e-2; equilibrated
1.6e-1 (worse); jacobi-only diverges; BSR+aggregation SEGFAULTS. So the
interleaved full F resists the quick fixes.

Round 2 tests the STRUCTURAL hypothesis: under Picard (newton=0) the only
cross-component u-u coupling in F is the SBM Nitsche block — each diagonal
component block F_cc is a SCALAR convection-diffusion-reaction operator
(classical AMG's home turf). If AMGX solves F00 well, a component-decoupled
F~ = blockdiag(F00,F11,F22) preconditioner (penalty coupling left to the
outer FGMRES) gives a fully-iterative F path for L6.

Variants (each in its own SUBPROCESS so an AMGX abort can't kill the race):
  f00-classical    : scalar classical AMG on the component-0 block
  fdiag-combined   : blockdiag(F00,F11,F22) solves vs the FULL F residual
                     (preconditioner quality incl. the dropped penalty)
  dilu-full        : full F, classical AMG + MULTICOLOR_DILU smoother
  d1-noaggr-full   : full F, classical AMG, D1 interpolator, no aggressive
  sweeps3-full     : full F, classical AMG, 3+3 Jacobi sweeps
  bsr-aggregation  : BSR(3x3) + aggregation retry (isolated this time)

Env: LEVEL (5), DEVICE (cuda:0), ALPHA (100), ITERS (100), TOL (1e-6),
VARIANTS (comma list; default all). gpubox only.
"""
import json
import os
import subprocess
import sys
import time

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, os.path.dirname(__file__))

LEVEL = int(os.environ.get("LEVEL", "5"))
DEVICE = os.environ.get("DEVICE", "cuda:0")
ALPHA = float(os.environ.get("ALPHA", "100"))
ITERS = int(os.environ.get("ITERS", "100"))
TOL = float(os.environ.get("TOL", "1e-6"))
ALL_VARIANTS = ("f00-classical", "fdiag-combined", "dilu-full",
                "d1-noaggr-full", "sweeps3-full", "bsr-aggregation")

HERE = os.path.join(os.path.dirname(__file__), "..", "src", "diffsim",
                    "solvers", "amgx_configs")
NPZ = os.environ.get("F_NPZ", "/tmp/p2r2b1_F_l%d.npz" % LEVEL)


def base_cfg(iters=None, tol=None):
    with open(os.path.join(HERE, "PBICGSTAB_CLASSICAL_JACOBI.json")) as fh:
        cfg = json.load(fh)
    cfg["solver"]["max_iters"] = iters or ITERS
    cfg["solver"]["tolerance"] = TOL if tol is None else tol
    cfg["solver"]["monitor_residual"] = 1
    cfg["solver"]["print_solve_stats"] = 0
    cfg["solver"]["obtain_timings"] = 0
    cfg["verbosity_level"] = 1
    return cfg


def cfg_for(variant):
    cfg = base_cfg()
    pre = cfg["solver"]["preconditioner"]
    if variant == "dilu-full":
        pre["smoother"] = {"solver": "MULTICOLOR_DILU", "scope": "jacobi",
                           "monitor_residual": 0, "print_solve_stats": 0}
    elif variant == "d1-noaggr-full":
        pre["interpolator"] = "D1"
        pre.pop("aggressive_levels", None)
    elif variant == "sweeps3-full":
        pre["presweeps"] = 3
        pre["postsweeps"] = 3
    elif variant == "bsr-aggregation":
        cfg["solver"]["preconditioner"] = {
            "solver": "AMG", "algorithm": "AGGREGATION", "selector": "SIZE_2",
            "smoother": {"solver": "BLOCK_JACOBI", "scope": "jacobi",
                         "monitor_residual": 0, "print_solve_stats": 0},
            "presweeps": 1, "postsweeps": 1, "max_iters": 1, "cycle": "V",
            "max_levels": 50, "coarse_solver": "DENSE_LU_SOLVER",
            "scope": "amg", "monitor_residual": 0, "print_solve_stats": 0}
    return cfg


def amgx_solve(cfg_dict, M, b, bsr=False):
    from diffsim.solvers.amgx import _ensure_init
    _ensure_init()
    import pyamgx
    cfg = pyamgx.Config().create(json.dumps(cfg_dict))
    rsc = pyamgx.Resources().create_simple(cfg)
    Mx = pyamgx.Matrix().create(rsc)
    X = pyamgx.Vector().create(rsc)
    B = pyamgx.Vector().create(rsc)
    slv = pyamgx.Solver().create(rsc, cfg)
    if bsr:
        Mb = sp.bsr_matrix(M.tocsr(), blocksize=(3, 3))
        Mb.sort_indices()
        Mx.upload(Mb.indptr.astype(np.int32), Mb.indices.astype(np.int32),
                  np.ascontiguousarray(Mb.data.ravel(), np.float64),
                  block_dims=[3, 3])
    else:
        Mc = M.tocsr()
        Mc.sort_indices()
        Mx.upload_CSR(Mc)
    slv.setup(Mx)
    x = np.zeros(M.shape[0])
    B.upload(np.ascontiguousarray(b, np.float64))
    X.upload(x)
    t0 = time.time()
    slv.solve(B, X, zero_initial_guess=True)
    t = time.time() - t0
    X.download(x)
    try:
        it, st = int(slv.iterations_number), slv.status
    except Exception:  # noqa: BLE001
        it, st = -1, "?"
    return x, it, st, t


def run_variant(variant):
    """Executed IN THE SUBPROCESS: load F from NPZ, run, print result."""
    d = np.load(NPZ)
    F = sp.csr_matrix((d["data"], d["indices"], d["indptr"]),
                      shape=tuple(d["shape"]))
    rng = np.random.default_rng(0)
    b = rng.standard_normal(F.shape[0])
    n = F.shape[0]
    n_nodes = n // 3
    idx = np.arange(n).reshape(n_nodes, 3)
    if variant == "f00-classical":
        ids = idx[:, 0].ravel()
        F00 = F[ids][:, ids].tocsr()
        b0 = b[ids]
        x, it, st, t = amgx_solve(base_cfg(), F00, b0)
        rel = float(np.linalg.norm(F00 @ x - b0) / np.linalg.norm(b0))
        print(f"[fconfig2] {variant:16s}: ACHIEVED rel={rel:.3e} iters={it} "
              f"status={st} ({t:.1f}s) [scalar n={F00.shape[0]}]", flush=True)
        return
    if variant == "fdiag-combined":
        x = np.empty(n)
        its = []
        t_tot = 0.0
        for c in range(3):
            ids = idx[:, c].ravel()
            Fc = F[ids][:, ids].tocsr()
            xc, it, st, t = amgx_solve(base_cfg(), Fc, b[ids])
            x[ids] = xc
            its.append(it)
            t_tot += t
        rel = float(np.linalg.norm(F @ x - b) / np.linalg.norm(b))
        print(f"[fconfig2] {variant:16s}: FULL-F rel={rel:.3e} "
              f"(per-comp iters={its}) ({t_tot:.1f}s)", flush=True)
        return
    bsr = variant == "bsr-aggregation"
    x, it, st, t = amgx_solve(cfg_for(variant), F, b, bsr=bsr)
    rel = float(np.linalg.norm(F @ x - b) / np.linalg.norm(b))
    print(f"[fconfig2] {variant:16s}: ACHIEVED rel={rel:.3e} iters={it} "
          f"status={st} ({t:.1f}s)", flush=True)


def main():
    if os.environ.get("FCONFIG2_VARIANT"):
        run_variant(os.environ["FCONFIG2_VARIANT"])
        return
    # parent: build the L5 step-1 F once, save to NPZ, then fan out.
    from diffsim.solvers.linsolve import solve_linear
    import p2r0_task10_sphere_derisk as derisk

    captured = {}
    march_cache = {}

    def _spy(A, b, **kw):
        captured["A"] = A.copy()
        return solve_linear(A.tocsr(), b, solver="cudss", sym=False,
                            device=DEVICE, cache=march_cache,
                            cache_key="march")

    fx = derisk.build_sphere_3d(DEVICE, LEVEL, 100.0)
    print(f"[fconfig2] L{LEVEL}: n_free={len(fx['coords'])} "
          f"sf_faces={int(fx['sf'].elem.size)}", flush=True)
    derisk.solve_linear = _spy
    derisk.monolithic_cd(fx, alpha=ALPHA, dt=0.05, max_steps=1,
                         rate_tol=1e-12, solver="blockamgx")
    A = captured["A"].tocsr()
    n_nodes, ndof = len(fx["coords"]), fx["ndof"]
    dim = ndof - 1
    idx = np.arange(n_nodes * ndof).reshape(n_nodes, ndof)
    u_ids = idx[:, :dim].ravel()
    F = A[u_ids][:, u_ids].tocsr()
    np.savez(NPZ, data=F.data, indices=F.indices, indptr=F.indptr,
             shape=np.array(F.shape))
    variants = os.environ.get("VARIANTS", ",".join(ALL_VARIANTS)).split(",")
    env = dict(os.environ)
    for v in variants:
        env["FCONFIG2_VARIANT"] = v
        r = subprocess.run([sys.executable, os.path.abspath(__file__)],
                           env=env, capture_output=True, text=True,
                           timeout=1200)
        out = (r.stdout or "").strip()
        if out:
            print(out, flush=True)
        if r.returncode != 0:
            tail = (r.stderr or "").strip().splitlines()[-3:]
            print(f"[fconfig2] {v:16s}: SUBPROCESS rc={r.returncode} "
                  f"{' | '.join(tail)}", flush=True)


if __name__ == "__main__":
    main()
