"""AMGX backend (optional): NVIDIA algebraic multigrid through pyamgx.

Build once (see benchmarks/README.md):
    git clone --recursive https://github.com/NVIDIA/AMGX && cd AMGX
    cmake -B build -DCMAKE_CUDA_ARCHITECTURES=<arch> && make -C build -j amgxsh
    AMGX_DIR=$PWD pip install pyamgx  (from github.com/shwina/pyamgx)

Configs: classical AMG-preconditioned Krylov — PCG for SPD (the PPE),
AMG-preconditioned BiCGStab for the nonsymmetric stabilized NS blocks.
Solver objects are cached per (cache_key, config) and re-setup with new
matrix values each step (AMGX's replace-coefficients path when the sparsity
is unchanged).

SETUP-REUSE (task W1, 2026-07-24): the AMG hierarchy (galerkin coarse
operators + interpolators) is the expensive part of ``slv.setup(M)``. When the
CSR SPARSITY is unchanged across solves — the standard projection PPE path,
where ``K_p`` is a CONSTANT operator (only the RHS changes step to step) — we
skip the re-upload + re-setup entirely and only push the new *values* via
``M.replace_coefficients(data)``, reusing the already-built hierarchy. Guarded
by a sparsity fingerprint (shape, nnz, indptr, indices); any change falls back
to the full upload_CSR + setup path. Bit-for-bit for a truly constant matrix
(same values -> identical solve); controlled by ``AMGX_SETUP_REUSE`` (default
on). The last solve's iteration count / residual are exposed via
``last_solve_stats()`` for the scaling harness."""
import atexit
import os

import numpy as np

_ctx = {"initialized": False}
# Per-symmetry-class last-solve telemetry (iterations, residual, timings),
# populated by amgx_solve for the scaling harness.
_last_stats = {}

# Default-on setup-reuse: skip slv.setup(M) when the CSR sparsity is unchanged
# and only replace the coefficient VALUES. Set AMGX_SETUP_REUSE=0 to force the
# legacy re-setup-every-call behaviour (A/B measurement).
_SETUP_REUSE = os.environ.get("AMGX_SETUP_REUSE", "1") not in ("0", "false",
                                                               "False", "")


def last_solve_stats():
    """Telemetry from the most recent amgx_solve, keyed by the singleton key.
    Each value is a dict: iterations, residual, setup_reused (bool),
    t_setup_s, t_solve_s, nnz, n."""
    return dict(_last_stats)


def _sparsity_fingerprint(A):
    """Cheap-but-safe CSR sparsity fingerprint: shape, nnz, and hashes of the
    indptr/indices arrays. Two matrices with the same fingerprint share a
    sparsity pattern, so the AMG hierarchy can be reused via
    replace_coefficients. Uses the raw array bytes' hash (fast, no allocation
    beyond the digest) — collisions are astronomically unlikely and a false
    match only reuses a valid hierarchy for a same-shape/nnz matrix."""
    return (A.shape, int(A.nnz),
            hash(A.indptr.tobytes()), hash(A.indices.tobytes()))


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
    value changes are pushed via replace_coefficients when the sparsity is
    unchanged (skips the AMG re-setup), else a full re-upload + re-setup."""
    import time
    _ensure_init()
    import pyamgx
    A = A.tocsr()
    A.sort_indices()
    # tol/maxiter are BAKED into the AMGX config at solver creation, so
    # they are part of the singleton key: a later call with different
    # settings gets a matching solver instead of silently inheriting the
    # first call's (evaluation solver-review item, CONFIRMED). Resources
    # remain shared per the lifetime rule via the first-created state.
    # CAVEAT (T2 review): callers sharing a key share the singleton matrix
    # object — two DIFFERENT simultaneously-live matrices with identical
    # (sym, tol, maxiter) would thrash the fingerprint check (results stay
    # correct, but every alternation forces a full re-setup). Keep
    # tol/maxiter distinct per live matrix (PCD F-inner: 1e-4/50;
    # direct "amgx" solves: typically 1e-10/2000).
    key = ("singleton", bool(sym), float(tol), int(maxiter))
    state = _ctx.get(key)
    if state is None:
        cfg = _config(sym, tol, maxiter)
        rsc = pyamgx.Resources().create_simple(cfg)
        state = {"cfg": cfg, "rsc": rsc,
                 "M": pyamgx.Matrix().create(rsc),
                 "X": pyamgx.Vector().create(rsc),
                 "B": pyamgx.Vector().create(rsc),
                 "slv": pyamgx.Solver().create(rsc, cfg),
                 "fp": None}
        _ctx[key] = state

    M, X, B, slv = state["M"], state["X"], state["B"], state["slv"]
    # ---- IDENTITY FAST PATH (sesc Rung 1): within one outer solve the
    # F-inner passes the SAME csr object with the SAME data buffer on every
    # apply.  At 0.6B nnz the sparsity fingerprint alone costs a 2.5 GB
    # tobytes copy per call and the values re-upload another ~5 GB — pure
    # waste when literally nothing changed.  Key on (id(A), id(A.data)):
    # any new object or rebound data buffer falls through to the fingerprint
    # path; in-place mutation of the SAME data buffer is outside this
    # module's contract (callers construct fresh CSRs per assembly).
    _ident = (id(A), id(A.data))
    t0 = time.perf_counter()
    if _SETUP_REUSE and state.get("ident") == _ident:
        reuse = True  # hierarchy AND values already uploaded — nothing to do
    else:
        # ---- SETUP-REUSE: fingerprint the sparsity; when unchanged, only
        # push the new coefficient VALUES and reuse the built AMG hierarchy
        # (skip the expensive setup). Otherwise re-upload + re-setup. ----
        fp = _sparsity_fingerprint(A)
        reuse = _SETUP_REUSE and state["fp"] is not None and state["fp"] == fp
        if reuse:
            # Same pattern: only the values changed. AMGX expects the CSR
            # VALUES array in the same (sorted) order originally uploaded.
            M.replace_coefficients(np.ascontiguousarray(A.data, np.float64))
            # No slv.setup(): the hierarchy from the first setup is reused.
        else:
            M.upload_CSR(A)
            slv.setup(M)
            state["fp"] = fp
        state["ident"] = _ident
    t_setup = time.perf_counter() - t0
    B.upload(np.ascontiguousarray(b, np.float64))
    x = np.zeros_like(b)
    X.upload(x)
    t1 = time.perf_counter()
    slv.solve(B, X)
    t_solve = time.perf_counter() - t1
    # "not_converged" means maxiter was hit — accept the truncated iterate
    # (it is used as a smoother in the PCD F-inner; the outer FGMRES carries
    # the remaining residual, same as the Jacobi-CG cap_hits convention).
    # Any other non-success status (e.g. "numerical_issues", "crashed") still
    # raises so genuine AMGX failures surface.
    if slv.status not in ("success", "not_converged"):
        raise RuntimeError(f"AMGX solve status: {slv.status}")
    X.download(x)
    # telemetry for the scaling harness (best-effort; ignore if unavailable)
    try:
        iters = int(slv.iterations_number)
    except Exception:
        iters = None
    # get_residual() needs store_res_history in the config (off by default to
    # avoid per-iteration overhead + console spam); skip it rather than trip
    # AMGX's "Residual history was not recorded" exception every solve.
    res = None
    _last_stats[key] = {"iterations": iters, "residual": res,
                        "setup_reused": bool(reuse), "t_setup_s": t_setup,
                        "t_solve_s": t_solve, "nnz": int(A.nnz),
                        "n": int(A.shape[0])}
    return x
