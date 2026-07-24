"""PPE end-to-end DEVICE-path driver (task: l8-hero de-risk).

The 100M-hero PPE pipeline, staged and measured, WITHOUT the Leray stepper's
separate single-threaded momentum-predictor host assembly (a distinct wall out
of scope for the PPE de-risk).  The pipeline this exercises IS the hero path:

    mesh build (host, one-time)
        -> DeviceScalarPoissonAssembler symbolic (one-time)
        -> fill()  (device scatter, PER STEP)
        -> AMGX PPE solve (setup one-time w/ reuse, solve PER STEP)

For each requested level it reports the six stage timings + peak GPU mem, and
optionally verifies the device-assembled + AMGX-solved PPE against a host
splu reference (parity) at small levels.  A ceiling (OOM / warp 2^31 / mesh
build) is caught per level and reported as the wall — the key de-risk number.

Run on the box (DiffSim-l8 lane):
    LD_LIBRARY_PATH=/usr/lib/wsl/lib:$HOME/AMGX/build \
    PYTHONPATH=/home/bglab/Baskar/DiffSim-l8/src CUDA_VISIBLE_DEVICES=1 \
      python tests/ppe_e2e_device.py --levels 6 7 8 --steps 4
    # parity:
      python tests/ppe_e2e_device.py --parity --levels 4 5 6
"""
import argparse
import gc
import sys
import time

import numpy as np
import scipy.sparse as sp

from diffsim.octree.build import build_uniform
from diffsim.mesh.nodes import build_mesh
from diffsim.mesh.constraints import build_constraints
from diffsim.mesh.basis import basis_tables
from diffsim.assembly.operators import DeviceMesh, DeviceScalarPoissonAssembler
from diffsim.solvers.linsolve import solve_linear
from diffsim.solvers.amgx import last_solve_stats


def gpu_mem_mb(device):
    try:
        import torch
        idx = 0
        if isinstance(device, str) and ":" in device:
            idx = int(device.split(":")[1])
        torch.cuda.synchronize(idx)
        return torch.cuda.max_memory_allocated(idx) / (1024 ** 2)
    except Exception:
        return float("nan")


def reset_peak():
    try:
        import torch
        torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


def wp_mem_mb(device):
    """Warp's own device allocation high-water (torch peak misses warp
    allocations, which is MOST of the K_p vals/indices).  Sum current +
    report the mempool high-water if available."""
    try:
        import warp as wp
        dev = wp.get_device(device if ":" in str(device) else "cuda:0")
        # bytes currently allocated by warp on this device
        return dev.total_memory / (1024 ** 2), \
            (dev.total_memory - dev.free_memory) / (1024 ** 2)
    except Exception:
        return float("nan"), float("nan")


def build_mesh_stage(level, device):
    t0 = time.perf_counter()
    tree = build_uniform(level, dim=3)
    mesh = build_mesh(tree, p=1)
    cons = build_constraints(mesh)
    dm = DeviceMesh.from_mesh(mesh, cons, basis_tables(1, dim=3), device)
    return dm, time.perf_counter() - t0


def pin_row0(Kp):
    """Pin free-node 0 (enclosed-flow PPE pin) directly on CSR arrays."""
    Kp = Kp.tocsr().copy()
    r0s, r0e = Kp.indptr[0], Kp.indptr[1]
    Kp.data[r0s:r0e] = 0.0
    row0_cols = Kp.indices[r0s:r0e]
    diag_pos = np.where(row0_cols == 0)[0]
    if diag_pos.size:
        Kp.data[r0s + diag_pos[0]] = 1.0
    Kp.eliminate_zeros()
    return Kp


def run_level(level, device, steps, solver="amgx"):
    reset_peak()
    stages = {}

    # STAGE 1: mesh build (host, one-time)
    dm, t_mesh = build_mesh_stage(level, device)
    stages["mesh_build_s"] = t_mesh

    # STAGE 2: device symbolic (one-time)
    t0 = time.perf_counter()
    asm = DeviceScalarPoissonAssembler(dm)
    stages["dev_symbolic_s"] = time.perf_counter() - t0
    n = asm.Nfull
    nnz = asm.nnz

    # STAGE 3: device fill (per-step) — time each, take median
    fill_t = []
    for _ in range(steps + 1):        # +1 warm
        t0 = time.perf_counter()
        asm.fill()
        try:
            import warp as wp
            wp.synchronize_device(dm.device)
        except Exception:
            pass
        fill_t.append(time.perf_counter() - t0)
    stages["dev_fill_med_s"] = float(np.median(fill_t[1:]))

    # download the assembled CSR once, pin row0
    Kp = asm.to_csr()
    Kp = pin_row0(Kp)

    coords = dm.mesh.node_coords[dm.constraints.free_nodes] \
        if hasattr(dm.constraints, "free_nodes") else dm.mesh.node_coords
    # manufactured smooth x_true consistent with the pin
    m = Kp.shape[0]
    xc = dm.mesh.node_coords
    if xc.shape[0] != m:
        xc = xc[:m]
    x_true = (np.cos(np.pi * xc[:, 0]) * np.cos(np.pi * xc[:, 1])
              * np.cos(np.pi * xc[:, 2]))
    x_true = np.ascontiguousarray(x_true, np.float64)
    x_true[0] = 0.0
    b = np.asarray(Kp @ x_true)
    b[0] = 0.0

    # STAGE 4+5: AMGX setup (warm) + per-step solve, reuse fast path
    cache = {}
    t0 = time.perf_counter()
    x = solve_linear(Kp, b, solver=solver, sym=True, device=device,
                     cache=cache, cache_key="ppe")     # warm: full setup
    stages["amgx_setup_warm_s"] = time.perf_counter() - t0

    solve_t = []
    reused = iters = None
    for _ in range(steps):
        t0 = time.perf_counter()
        x = solve_linear(Kp, b, solver=solver, sym=True, device=device,
                         cache=cache, cache_key="ppe")
        solve_t.append(time.perf_counter() - t0)
        if solver == "amgx":
            st = last_solve_stats()
            for k, v in st.items():
                if k[1] is True:
                    iters, reused = v["iterations"], v["setup_reused"]
                    break
    stages["amgx_solve_med_s"] = float(np.median(solve_t))

    err = float(np.abs(x - x_true).max())
    torch_mem = gpu_mem_mb(device)
    wp_tot, wp_used = wp_mem_mb(device)

    return {
        "level": level, "n": int(n), "nnz": int(nnz),
        "amg_iters": iters, "setup_reused": reused,
        "err_vs_true": err,
        "torch_peak_mb": torch_mem, "gpu_used_mb": wp_used,
        **stages,
    }


def parity_level(level, device, steps=2, tol=1e-6):
    """Device-assembled + AMGX-solved PPE vs host-splu on the SAME K_p."""
    dm, _ = build_mesh_stage(level, device)
    asm = DeviceScalarPoissonAssembler(dm)
    asm.fill()
    Kp = pin_row0(asm.to_csr())
    m = Kp.shape[0]
    xc = dm.mesh.node_coords[:m]
    x_true = np.ascontiguousarray(
        np.cos(np.pi * xc[:, 0]) * np.cos(np.pi * xc[:, 1])
        * np.cos(np.pi * xc[:, 2]), np.float64)
    x_true[0] = 0.0
    b = np.asarray(Kp @ x_true); b[0] = 0.0

    x_splu = solve_linear(Kp, b, solver="splu", sym=True, device=device,
                          cache={}, cache_key="p")
    x_amgx = solve_linear(Kp, b, solver="amgx", sym=True, device=device,
                          cache={}, cache_key="p")
    scale = max(np.abs(x_splu).max(), 1e-30)
    d = np.abs(x_amgx - x_splu).max()
    rel = d / scale
    return d, rel, rel < tol


def fmt(r):
    return (
        f"  L{r['level']:<2d} n={r['n']:>13,d} nnz={r['nnz']:>14,d}  "
        f"mesh={r['mesh_build_s']:8.2f}s sym={r['dev_symbolic_s']:7.2f}s "
        f"fill={r['dev_fill_med_s']*1e3:8.1f}ms "
        f"setup={r['amgx_setup_warm_s']:7.2f}s "
        f"solve={r['amgx_solve_med_s']*1e3:8.1f}ms "
        f"iters={str(r['amg_iters']):>4s} reuse={str(r['setup_reused']):>5s} "
        f"err={r['err_vs_true']:.1e} gpu_used={r['gpu_used_mb']:.0f}MiB")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", type=int, nargs="+", default=[6, 7, 8])
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--solver", default="amgx")
    ap.add_argument("--parity", action="store_true")
    ap.add_argument("--parity-tol", type=float, default=1e-6)
    args = ap.parse_args(argv)

    if args.parity:
        ok = True
        print("# PPE PARITY: device-assembled K_p, AMGX vs host-splu solve")
        for L in args.levels:
            try:
                d, rel, good = parity_level(L, args.device, tol=args.parity_tol)
                ok = ok and good
                print(f"  L{L}: |x_amgx-x_splu|={d:.3e} rel={rel:.3e} -> "
                      f"{'PASS' if good else 'FAIL'}", flush=True)
                gc.collect()
            except Exception as e:  # noqa: BLE001
                import traceback
                print(f"  L{L}: ERROR {type(e).__name__}: {e}", flush=True)
                traceback.print_exc()
                ok = False
                break
        print(f"\nPPE PARITY: {'PASS' if ok else 'FAIL'} (tol={args.parity_tol})")
        return 0 if ok else 1

    print(f"# PPE END-TO-END DEVICE PATH — steps={args.steps} "
          f"device={args.device} solver={args.solver}")
    print(f"# stages: mesh(1x) sym(1x) fill(/step) amgx_setup(1x) "
          f"amgx_solve(/step)")
    rows = []
    for L in args.levels:
        try:
            r = run_level(L, args.device, args.steps, args.solver)
            rows.append(r)
            print(fmt(r), flush=True)
            gc.collect()
        except Exception as e:  # noqa: BLE001 — the ceiling is the datapoint
            import traceback
            print(f"  L{L}: CEILING/ERROR: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
            print(f"  -> single-GPU PPE ceiling bites at L{L}", flush=True)
            break
    print("\n# STAGE TABLE")
    for r in rows:
        print(fmt(r))
    return 0


if __name__ == "__main__":
    sys.exit(main())
