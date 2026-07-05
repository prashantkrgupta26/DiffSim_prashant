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


def _ensure_init():
    import pyamgx
    if not _ctx["initialized"]:
        pyamgx.initialize()
        _ctx["initialized"] = True
        atexit.register(pyamgx.finalize)


def _config(sym: bool, tol: float):
    import pyamgx
    if sym:
        cfg = {
            "config_version": 2,
            "solver": {
                "solver": "PCG",
                "preconditioner": {"solver": "AMG", "algorithm": "CLASSICAL",
                                   "max_levels": 20, "cycle": "V",
                                   "smoother": "BLOCK_JACOBI",
                                   "presweeps": 1, "postsweeps": 1},
                "max_iters": 1000, "tolerance": tol,
                "convergence": "RELATIVE_INI_CORE", "monitor_residual": 1,
            },
        }
    else:
        cfg = {
            "config_version": 2,
            "solver": {
                "solver": "PBICGSTAB",
                "preconditioner": {"solver": "AMG", "algorithm": "CLASSICAL",
                                   "max_levels": 20, "cycle": "V",
                                   "smoother": "BLOCK_JACOBI",
                                   "presweeps": 1, "postsweeps": 1},
                "max_iters": 2000, "tolerance": tol,
                "convergence": "RELATIVE_INI_CORE", "monitor_residual": 1,
            },
        }
    import json
    return pyamgx.Config().create(json.dumps(cfg))


def amgx_solve(A, b, sym=False, tol=1e-10, cache=None, cache_key=None):
    """Solve on the GPU via AMGX. A: scipy CSR (FP64), b: host vector."""
    import pyamgx
    _ensure_init()
    A = A.tocsr()
    A.sort_indices()
    key = ("amgx", cache_key, sym) if cache_key is not None else None
    state = cache.get(key) if (cache is not None and key) else None
    if state is None:
        cfg = _config(sym, tol)
        rsc = pyamgx.Resources().create_simple(cfg)
        M = pyamgx.Matrix().create(rsc)
        X = pyamgx.Vector().create(rsc)
        B = pyamgx.Vector().create(rsc)
        slv = pyamgx.Solver().create(rsc, cfg)
        state = {"cfg": cfg, "rsc": rsc, "M": M, "X": X, "B": B, "slv": slv,
                 "setup_nnz": -1}
        if cache is not None and key:
            cache[key] = state
    M, X, B, slv = state["M"], state["X"], state["B"], state["slv"]
    M.upload_CSR(A)
    slv.setup(M)
    B.upload(np.ascontiguousarray(b, np.float64))
    x = np.zeros_like(b)
    X.upload(x)
    slv.solve(B, X)
    status = slv.status
    if status not in ("success",):
        raise RuntimeError(f"AMGX solve status: {status}")
    X.download(x)
    return x
