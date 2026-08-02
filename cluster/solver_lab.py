"""Offline solver lab — run candidate saddle solvers on dumped snapshots.

Rung 1 of the solver-escalation campaign (spec 2026-08-01).  A snapshot is a
self-contained system dumped by run_truck(dump_system_steps=...): the saddle
CSR A and rhs b at a march step, plus the PCD pressure operators (Mp, Ap)
and the knob record of the leg that produced it.

Configs:
  bdiag      — fgmres_bdiag at the leg's knobs (control A: must reproduce
               the leg's outcome; a captured miss-system must MISS)
  pcd-jacobi — fgmres_pcd, Jacobi inners (control B: the slow fallback)
  pcd-amgx   — fgmres_pcd, AMGX F-inner + AMG-on-Ap (the candidate;
               requires CUDA + pyamgx)

Key mapping (snapshot npz -> pcd_meta dict -> consumer):
  npz["p_pin"]       -> meta["p_pin_local"]  -> make_pcd_apply reads
                                                meta.get("p_pin_local");
                        stored as int(pcd_meta["p_pin_local"]) by Task 1,
                        i.e. p_pin_global // ndof (pressure-LOCAL index,
                        valid for indexing the nfree x nfree Ap).
  build_pcd_meta returns: ndof, dim, Mp, Ap, sigma, nu, p_pin_local,
                          inner, ap_inner — no "p_pin" key; do NOT set it.

Usage:
  python cluster/solver_lab.py SNAP.npz --config pcd-amgx \\
      [--device cuda:0] [--restart 60] [--maxiter 3000] [--tol 5e-4]
Prints one JSON row per run (append to a .jsonl for sweeps).
"""
import argparse
import json
import sys
import time

import numpy as np
import scipy.sparse as sp


def _load(npz_path):
    d = np.load(npz_path)
    A = sp.csr_matrix((d["A_data"], d["A_indices"], d["A_indptr"]),
                      shape=tuple(d["A_shape"]))
    Mp = sp.csr_matrix((d["Mp_data"], d["Mp_indices"], d["Mp_indptr"]),
                       shape=tuple(d["Mp_shape"]))
    Ap = sp.csr_matrix((d["Ap_data"], d["Ap_indices"], d["Ap_indptr"]),
                       shape=tuple(d["Ap_shape"]))
    # npz key "p_pin" stores build_pcd_meta's "p_pin_local" — the
    # PRESSURE-LOCAL pinned index (p_pin_global // ndof), valid for
    # indexing the (nfree x nfree) Ap.  Task 1 dumps:
    #   p_pin=int(pcd_meta["p_pin_local"])
    # so d["p_pin"] is already the local index; it is placed under the
    # "p_pin_local" key in the meta dict (what make_pcd_apply reads).
    # F2: warm-start slots (optional; empty sentinel when absent from cache).
    x_prev = np.asarray(d["x_prev"]) if "x_prev" in d else np.zeros(0)
    x_prev2 = np.asarray(d["x_prev2"]) if "x_prev2" in d else np.zeros(0)
    meta = dict(ndof=int(d["ndof"]), nfree=int(d["nfree"]),
                tol=float(d["tol"]), sigma=float(d["sigma"]),
                nu=float(d["nu"]), dt=float(d["dt"]), step=int(d["step"]),
                p_pin_local=int(d["p_pin"]),
                bd=json.loads(str(d["bd_json"])),
                x_prev=x_prev, x_prev2=x_prev2)
    return A, np.asarray(d["b"], np.float64), Mp, Ap, meta


def _build_cache(config, Mp, Ap, meta, inner=None, ap_inner=None):
    """Reconstruct the solver-cache entries solve_linear expects, without a
    mesh: bdiag uses the knob dict; pcd uses a hand-built pcd_meta (the
    same keys build_pcd_meta returns — see saddle_precond.py:~514).

    build_pcd_meta return keys: ndof, dim, Mp, Ap, sigma, nu,
                                p_pin_local, inner, ap_inner.
    No "p_pin" key in the return dict; make_pcd_apply reads p_pin_local.

    F2: when config=="bdiag" and the snapshot carries non-empty x_prev,
    seed the bdiag warm-start cache so the lab reproduces the leg's
    extrapolated x0 rather than running cold.
    """
    cache = {"ndof": meta["ndof"]}
    if config == "bdiag":
        bd = {"ndof": meta["ndof"]}
        for k in ("saddle_x0", "saddle_restart", "saddle_equilibrate",
                  "saddle_min_work"):
            if k in meta["bd"]:
                bd[k] = meta["bd"][k]
        cache[("blocktri_meta", "lab")] = bd
        # Seed warm-start history from the snapshot so Control A can replay
        # the leg's extrapolated x0 (F2: snapshot warm-start fidelity).
        _xp = meta.get("x_prev", np.zeros(0))
        _xp2 = meta.get("x_prev2", np.zeros(0))
        if _xp.size:
            cache[("bdiag_x_prev", "lab")] = _xp.copy()
            if _xp2.size:
                cache[("bdiag_x_prev2", "lab")] = _xp2.copy()
    else:
        # Default inner/ap_inner from config; CLI overrides via --inner/--ap-inner
        # (F5: expose sweep axes).
        _inner = inner if inner is not None else (
            "amgx" if config == "pcd-amgx" else "jacobi")
        _ap_inner = ap_inner if ap_inner is not None else (
            "amgx" if config == "pcd-amgx" else "jacobi")
        # Mirror the exact key set build_pcd_meta returns (saddle_precond.py
        # ~514).  make_pcd_apply reads: ndof, dim, Mp, Ap, sigma, nu,
        # p_pin_local (via meta.get("p_pin_local")), inner, ap_inner.
        cache[("pcd_meta", "lab")] = {
            "ndof": meta["ndof"],
            "dim": meta["ndof"] - 1,   # ndof = dim+1 (velocity comps + pressure)
            "Mp": Mp,
            "Ap": Ap,
            "sigma": meta["sigma"],
            "nu": meta["nu"],
            "p_pin_local": meta["p_pin_local"],
            "inner": _inner,
            "ap_inner": _ap_inner,
        }
    return cache


def run_config(npz_path, config, *, device="cpu", restart=None,
               maxiter=None, tol=None, inner=None, ap_inner=None):
    """Run one solver config on a dumped snapshot.

    Parameters
    ----------
    npz_path : str or Path
        Path to the ``sys_stepNNNN.npz`` file written by Task 1.
    config : {"bdiag", "pcd-jacobi", "pcd-amgx"}
        Which solver configuration to exercise.
    device : str
        Warp device string, e.g. "cpu" or "cuda:0".
    restart : int or None
        Override outer FGMRES restart for bdiag (saddle_restart in bd meta).
        fgmres_pcd uses a fixed restart=60 hardcoded in linsolve — there is
        no restart kwarg on solve_linear for the pcd path; pass --maxiter to
        vary the cycle budget (cycles = min(200, maxiter//60)) instead.
    maxiter : int or None
        Override total inner iteration budget passed to solve_linear.
    tol : float or None
        Convergence tolerance (relative residual).  None or 0.0 -> snapshot
        tol (F8: guard --tol 0.0 treated as unset).
    inner : str or None
        Override the F-block inner-solve backend for pcd configs ("jacobi" or
        "amgx").  None -> config default (pcd-amgx: "amgx"; pcd-jacobi:
        "jacobi").  Ignored for bdiag.
    ap_inner : str or None
        Override the Ap-block inner-solve backend for pcd configs.  Same
        semantics as ``inner``.  Ignored for bdiag.

    Returns
    -------
    dict
        config, snapshot, step, n, tol, converged, relres, iters (total inner
        iterations, renamed from "outer" — F6), inner_stats, wall_s, device.
        On miss: relres parsed from the ConvergenceError message (F3), and
        error key with the truncated exception string.
    """
    import re as _re
    from diffsim.solvers import linsolve
    A, b, Mp, Ap, meta = _load(npz_path)
    cache = _build_cache(config, Mp, Ap, meta,
                         inner=inner, ap_inner=ap_inner)
    if restart is not None and config == "bdiag":
        cache[("blocktri_meta", "lab")]["saddle_restart"] = int(restart)
    elif restart is not None:
        # pcd path: pcd_restart meta knob (linsolve default 60); smaller
        # restart shrinks the V+Z Krylov store — the OOM lever alongside
        # an AMGX F-hierarchy (sesc Rung 1).
        cache[("pcd_meta", "lab")]["pcd_restart"] = int(restart)
    solver = "fgmres_bdiag" if config == "bdiag" else "fgmres_pcd"
    # F8: treat tol=0.0 as unset (float(0.0 or meta["tol"]) would pick meta
    # tol, but explicit 0.0 from argparse would be falsy — guard explicitly).
    _tol = (float(tol) if (tol is not None and float(tol) != 0.0)
            else float(meta["tol"]))
    kw = dict(solver=solver, sym=False, tol=_tol,
              device=device, cache=cache, cache_key="lab")
    if maxiter is not None:
        kw["maxiter"] = int(maxiter)

    # Clear the sentinels before the solve so we report fresh values.
    linsolve._LAST_ITERS[0] = None
    linsolve._LAST_INNER_STATS[0] = None

    t0 = time.time()
    converged = True
    relres = float("nan")
    error = None
    try:
        x = linsolve.solve_linear(A, b, **kw)
        relres = float(np.linalg.norm(b - A @ x) / np.linalg.norm(b))
    except Exception as e:
        if type(e).__name__ not in ("ConvergenceError",):
            raise
        converged = False
        error = str(e)[:200]
        # F3: parse relres from the ConvergenceError message so miss rows
        # carry the achieved residual (linsolve format: "relres=<float>").
        _m = _re.search(r"relres=([0-9.e+\-inf]+)", str(e), _re.IGNORECASE)
        if _m:
            try:
                relres = float(_m.group(1))
            except ValueError:
                pass
    wall = time.time() - t0

    # _LAST_ITERS[0] is set only on convergence (after the raise in the
    # ConvergenceError branch); on miss it stays None.
    # F6: key renamed "iters" (it holds total inner iterations, not restart
    # cycles — the "outer" label was misleading for Rung-1 analysis tables).
    raw_iters = linsolve._LAST_ITERS[0]
    iters = int(raw_iters) if raw_iters is not None else -1

    inner_stats = (linsolve._LAST_INNER_STATS[0]
                   if solver == "fgmres_pcd" else None)

    row = dict(config=config, snapshot=str(npz_path), step=meta["step"],
               n=A.shape[0], tol=kw["tol"], converged=converged,
               relres=relres, iters=iters,
               inner_stats=inner_stats, wall_s=round(wall, 3), device=device)
    if error is not None:
        row["error"] = error
    return row


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Offline solver lab — run a candidate saddle solver on a "
                    "dumped snapshot; prints one JSON row.")
    ap.add_argument("snapshot")
    ap.add_argument("--config", required=True,
                    choices=("bdiag", "pcd-jacobi", "pcd-amgx"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--restart", type=int, default=None,
                    help="Override outer FGMRES restart (bdiag only; "
                         "fgmres_pcd restart is fixed at 60 in linsolve — "
                         "use --maxiter to vary cycle budget for pcd configs)")
    ap.add_argument("--maxiter", type=int, default=None)
    ap.add_argument("--tol", type=float, default=None,
                    help="Convergence tolerance; 0.0 treated as unset "
                         "(falls back to snapshot tol)")
    ap.add_argument("--inner", default=None,
                    choices=("jacobi", "amgx"),
                    help="Override F-block inner-solve backend for pcd configs "
                         "(F5: sweep axis; ignored for bdiag)")
    ap.add_argument("--ap-inner", default=None,
                    choices=("jacobi", "amgx"),
                    help="Override Ap-block inner-solve backend for pcd "
                         "configs (F5: sweep axis; ignored for bdiag)")
    a = ap.parse_args(argv)
    row = run_config(a.snapshot, a.config, device=a.device,
                     restart=a.restart, maxiter=a.maxiter, tol=a.tol,
                     inner=a.inner, ap_inner=a.ap_inner)
    print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
