"""OrgElMorph environment doctor — the one command a new student runs first.

Probes the full stack the course depends on and prints a report ending in
EXACTLY one of:

    READY: full CUDA         everything works, cuDSS available (every chapter)
    READY: reduced (no cuDSS)   CUDA works but no cuDSS; cuDSS chapters fall
                                back to splu/matrix-free (documented, slower)
    NOT READY: <specific fix>   what to install/fix, one actionable line

It does NOT stop at reporting versions: it compiles+runs a tiny Warp kernel and
assembles+solves a small FE system, because "imports succeed" is not the same
as "the device toolchain works". Exit code: 0 when READY (full or reduced),
1 when NOT READY.
"""
from __future__ import annotations

import importlib
import platform
import sys


def _v(mod_name, attr="__version__"):
    try:
        mod = importlib.import_module(mod_name)
        return getattr(mod, attr, "(installed)")
    except Exception as exc:
        return f"MISSING ({type(exc).__name__})"


def _line(label, value, ok=None):
    mark = "  " if ok is None else ("OK" if ok else "!!")
    print(f"  [{mark}] {label:<26} {value}")


def probe():
    """Run all probes; return ``(verdict, fix)`` where verdict is one of
    'full', 'reduced', 'notready' and fix is a one-line remedy (or "")."""
    print("=" * 66)
    print("OrgElMorph doctor — environment probe")
    print("=" * 66)

    # --- interpreter + core libs -----------------------------------------
    print("\nInterpreter & core libraries")
    _line("python", platform.python_version(),
          sys.version_info[:2] >= (3, 10))
    _line("platform", platform.platform())
    numpy_v = _v("numpy")
    scipy_v = _v("scipy")
    _line("numpy", numpy_v, not numpy_v.startswith("MISSING"))
    _line("scipy", scipy_v, not scipy_v.startswith("MISSING"))
    if numpy_v.startswith("MISSING") or scipy_v.startswith("MISSING"):
        return "notready", "pip install -e . (numpy/scipy missing)"

    # --- diffsim ----------------------------------------------------------
    print("\ndiffsim package")
    try:
        import diffsim
        _line("diffsim", diffsim.__version__, True)
    except Exception as exc:
        _line("diffsim", f"import failed: {exc}", False)
        return "notready", "pip install -e .  (diffsim not importable)"
    try:
        from diffsim.diagnostics import provenance
        meta = provenance.collect_environment()
        _line("diffsim commit", f"{str(meta.get('diffsim_commit'))[:12]} "
              f"(dirty={meta.get('git_dirty')})")
    except Exception as exc:
        _line("diffsim.diagnostics", f"import failed: {exc}", False)
        return "notready", "diffsim.diagnostics import failed — reinstall"

    # --- warp -------------------------------------------------------------
    print("\nWarp (kernel codegen)")
    warp_v = _v("warp")
    _line("warp", warp_v, not warp_v.startswith("MISSING"))
    if warp_v.startswith("MISSING"):
        return "notready", "pip install warp-lang>=1.7  (Warp missing)"

    # --- torch + CUDA -----------------------------------------------------
    print("\nPyTorch & CUDA")
    cuda_ok = False
    try:
        import torch
        _line("torch", torch.__version__, True)
        cuda_ok = torch.cuda.is_available()
        _line("cuda available", str(cuda_ok), cuda_ok)
        if cuda_ok:
            _line("cuda runtime", str(torch.version.cuda))
            _line("gpu", torch.cuda.get_device_name(0))
            gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            _line("gpu memory", f"{gb:.1f} GB")
            _line("gpu count", str(torch.cuda.device_count()))
    except Exception as exc:
        _line("torch", f"import failed: {exc}", False)
        return "notready", "pip install torch  (PyTorch missing/broken)"

    # --- FP64 support -----------------------------------------------------
    if cuda_ok:
        try:
            import torch
            x = torch.ones(4, dtype=torch.float64, device="cuda")
            _line("fp64 device tensor", str(float((x * 2).sum())), True)
        except Exception as exc:
            _line("fp64 device tensor", f"failed: {exc}", False)

    # --- cuDSS / nvmath ---------------------------------------------------
    print("\nDirect/iterative solver backends")
    cudss_ok = False
    try:
        import nvmath  # noqa: F401
        import nvmath.sparse.advanced  # noqa: F401
        _line("nvmath (cuDSS)", _v("nvmath"), True)
        cudss_ok = cuda_ok
    except Exception as exc:
        _line("nvmath (cuDSS)", f"unavailable ({type(exc).__name__})", False)
    try:
        import pyamgx  # noqa: F401
        _line("pyamgx (AMGX)", "available")
    except Exception:
        _line("pyamgx (AMGX)", "not built (optional)")
    _line("scipy SuperLU (splu)", "always available (host fallback)", True)

    # --- live Warp kernel compile+run ------------------------------------
    print("\nLive smoke: Warp kernel compile + run")
    warp_run_ok = _warp_smoke()
    _line("warp kernel add", "compiled + ran" if warp_run_ok else "FAILED",
          warp_run_ok)

    # --- live 8x8 assemble + solve ---------------------------------------
    print("\nLive smoke: 8x8 assemble + solve")
    solve_ok, detail = _assemble_solve_smoke()
    _line("8x8 CH assemble+solve", detail, solve_ok)

    # --- verdict ----------------------------------------------------------
    if not cuda_ok:
        return "notready", ("no CUDA device — the tutorials need an NVIDIA "
                            "GPU; check `nvidia-smi` and the CUDA driver")
    if not warp_run_ok:
        return "notready", "Warp kernel failed to run — check CUDA toolkit/driver"
    if not solve_ok:
        return "notready", "assemble+solve smoke failed — see the error above"
    if cudss_ok:
        return "full", ""
    return "reduced", ""


def _warp_smoke():
    """Compile and launch a trivial Warp kernel; return True on success."""
    try:
        import warp as wp
        wp.init()
        dev = "cuda:0" if wp.get_cuda_device_count() else "cpu"

        @wp.kernel
        def _add(a: wp.array(dtype=wp.float64),
                 b: wp.array(dtype=wp.float64),
                 out: wp.array(dtype=wp.float64)):
            i = wp.tid()
            out[i] = a[i] + b[i]

        import numpy as np
        a = wp.array(np.ones(8), dtype=wp.float64, device=dev)
        b = wp.array(2.0 * np.ones(8), dtype=wp.float64, device=dev)
        out = wp.zeros(8, dtype=wp.float64, device=dev)
        wp.launch(_add, dim=8, inputs=[a, b, out], device=dev)
        wp.synchronize()
        return bool(np.allclose(out.numpy(), 3.0))
    except Exception as exc:
        print(f"        warp smoke error: {type(exc).__name__}: {exc}")
        return False


def _assemble_solve_smoke():
    """Build a tiny 8x8 uniform mesh, one Cahn-Hilliard step, check finite.

    Returns ``(ok, detail)``. Uses the same production bricks the tutorials
    call, so it exercises the real assembly + solve path end-to-end.
    """
    try:
        import numpy as np
        from diffsim.octree.build import build_uniform
        from diffsim.mesh.nodes import build_mesh
        from diffsim.mesh.constraints import build_constraints
        from diffsim.mesh.basis import basis_tables
        from diffsim.assembly.operators import DeviceMesh
        from diffsim.physics.cahn_hilliard import CahnHilliardStepper

        import torch
        dev = "cuda:0" if torch.cuda.is_available() else "cpu"
        tree = build_uniform(3, dim=2)        # 8x8
        mesh = build_mesh(tree, p=1)
        cons = build_constraints(mesh)
        dm = DeviceMesh.from_mesh(mesh, cons, basis_tables(1, dim=2), dev)
        st = CahnHilliardStepper(dm, 1.0, 5e-4, 0.02, order=1, energy="poly")
        rng = np.random.default_rng(0)
        st.set_initial(lambda x: 0.05 * rng.standard_normal(len(x)),
                       mu_init="consistent")
        c, mu = st.step()
        c = np.asarray(c)
        finite = bool(np.all(np.isfinite(c)))
        return finite, (f"stepped, c in [{c.min():.3f},{c.max():.3f}], "
                        f"finite={finite}")
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def main():
    verdict, fix = probe()
    print("\n" + "=" * 66)
    if verdict == "full":
        print("READY: full CUDA")
        code = 0
    elif verdict == "reduced":
        print("READY: reduced (no cuDSS)")
        print("  cuDSS chapters (C4 solver ecosystem, large runs) fall back to")
        print("  splu / matrix-free — correct but slower. See README.md.")
        code = 0
    else:
        print(f"NOT READY: {fix}")
        code = 1
    print("=" * 66)
    return code


if __name__ == "__main__":
    sys.exit(main())
