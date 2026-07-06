"""AMGX backend (optional): NVIDIA algebraic multigrid through pyamgx.

Build once (see benchmarks/README.md):
    git clone --recursive https://github.com/NVIDIA/AMGX && cd AMGX
    cmake -B build -DCMAKE_CUDA_ARCHITECTURES=<arch> && make -C build -j amgxsh
    AMGX_DIR=$PWD pip install pyamgx  (from github.com/shwina/pyamgx)

Configs: classical AMG-preconditioned Krylov — PCG for SPD (the PPE),
AMG-preconditioned BiCGStab for the nonsymmetric stabilized NS blocks.
Solver objects are cached per (cache_key, config) and re-setup with new
matrix values each step (AMGX's replace-coefficients path when the sparsity
is unchanged)."""
import atexit

import numpy as np

_ctx = {"initialized": False}


def _preload_libamgx():
    """Locate libamgxsh.so without requiring LD_LIBRARY_PATH: the repo's
    extern/ staging dir first (see benchmarks/README for the build), then
    the system loader."""
    import ctypes
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    for cand in (os.path.join(here, "..", "..", "..", "extern",
                              "libamgxsh.so"),
                 "libamgxsh.so"):
        try:
            ctypes.CDLL(cand, mode=ctypes.RTLD_GLOBAL)
            return
        except OSError:
            continue


def _ensure_init():
    _preload_libamgx()
    import pyamgx
    if not _ctx["initialized"]:
        pyamgx.initialize()
        _ctx["initialized"] = True
        # NO atexit finalize, deliberately: in a process that also holds
        # warp/torch CUDA state, atexit teardown order is arbitrary; once
        # another library has torn the context down, AMGX's finalize-time
        # pool audit SIGABRTs the process (measured: pytest exits 134
        # AFTER a fully green summary). The OS reclaims everything at
        # process death; finalize is optional at exit.


def _config(sym: bool, tol: float, maxiter: int = 2000):
    """Load AMGX's own validated example configs (bundled with the AMGX
    source; copies staged in extern/amgx_configs) and override tolerance.
    Hand-rolled config dicts are a trap: AMGX rejects malformed configs at
    Solver.create ("Incorrect parameters") or, worse, aborts the process.
    SPD  -> PCG + classical-AMG V-cycle (Jacobi smoother)
    nonsym -> BiCGStab + classical-AMG preconditioner (Jacobi smoother)"""
    import json
    import os
    import pyamgx
    here = os.path.join(os.path.dirname(__file__), "amgx_configs")
    fname = ("PCG_CLASSICAL_V_JACOBI.json" if sym
             else "PBICGSTAB_CLASSICAL_JACOBI.json")
    with open(os.path.join(here, fname)) as fh:
        cfg = json.load(fh)
    cfg["solver"]["tolerance"] = tol
    cfg["solver"]["max_iters"] = int(maxiter)
    cfg["solver"]["monitor_residual"] = 1
    cfg["solver"].setdefault("convergence", "RELATIVE_INI_CORE")
    cfg["solver"]["print_solve_stats"] = 0
    cfg["verbosity_level"] = 1              # errors only; kills pool spam
    return pyamgx.Config().create(json.dumps(cfg))


def amgx_solve(A, b, sym=False, tol=1e-10, maxiter=2000,
               cache=None, cache_key=None):
    """Solve on the GPU via AMGX. A: scipy CSR (FP64), b: host vector.

    LIFETIME RULE (measured the hard way): AMGX objects are process-global
    SINGLETONS here — one (config, resources, matrix, vectors, solver) set
    per symmetry class, created once and reused for every solve. Multiple
    live Resources sets in one process (e.g. per-stepper caches) segfault
    inside AMGX. The caller-provided cache is therefore ignored for AMGX;
    sparsity/value changes are handled by re-upload + re-setup per call."""
    _ensure_init()
    import pyamgx
    A = A.tocsr()
    A.sort_indices()
    # tol/maxiter are BAKED into the AMGX config at solver creation, so
    # they are part of the singleton key: a later call with different
    # settings gets a matching solver instead of silently inheriting the
    # first call's (evaluation solver-review item, CONFIRMED). Resources
    # remain shared per the lifetime rule via the first-created state.
    key = ("singleton", bool(sym), float(tol), int(maxiter))
    state = _ctx.get(key)
    if state is None:
        cfg = _config(sym, tol, maxiter)
        rsc = pyamgx.Resources().create_simple(cfg)
        state = {"cfg": cfg, "rsc": rsc,
                 "M": pyamgx.Matrix().create(rsc),
                 "X": pyamgx.Vector().create(rsc),
                 "B": pyamgx.Vector().create(rsc),
                 "slv": pyamgx.Solver().create(rsc, cfg)}
        _ctx[key] = state

    M, X, B, slv = state["M"], state["X"], state["B"], state["slv"]
    M.upload_CSR(A)
    slv.setup(M)
    B.upload(np.ascontiguousarray(b, np.float64))
    x = np.zeros_like(b)
    X.upload(x)
    slv.solve(B, X)
    if slv.status not in ("success",):
        raise RuntimeError(f"AMGX solve status: {slv.status}")
    X.download(x)
    return x
