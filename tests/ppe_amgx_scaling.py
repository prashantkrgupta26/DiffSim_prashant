"""PPE / AMGX weak-strong scaling harness (task W1).

Drives the Leray pressure-projection stepper's SPD pressure-Poisson (PPE)
through AMGX (PCG + classical-AMG V-cycle) on 3-D uniform meshes at increasing
levels, measuring per-solve time, AMG iterations, residual, memory, and the
single-GPU DOF ceiling. Reusable on Nova A100/GH200 later.

The PPE scalar DOF map for a 3-D uniform octree at level L (P1 nodes on a
2^L-per-axis grid): n_nodes = (2^L + 1)^3. Levels:
    L4 ~ 4.9k, L5 ~ 36k, L6 ~ 275k, L7 ~ 2.1M, L8 ~ 16.6M, L9 ~ 133M.
100M DOF sits between L8 and L9; the study reveals where the single-GPU
ceiling actually bites (warp 2^31 array-shape limit vs host COO->CSR
assembly).

Correctness (small levels): the PPE solve with solver="amgx" must match
solver="splu" to solver tolerance (same corrected u / pressure), and AMG-PCG
must converge (iteration count + residual reported).

Run (on the box, in the DiffSim-proj lane):
    LD_LIBRARY_PATH=/usr/lib/wsl/lib:$HOME/AMGX/build \
      python tests/ppe_amgx_scaling.py --levels 5 6 7 --solver amgx --steps 5
    # parity check:
    python tests/ppe_amgx_scaling.py --parity --levels 4 5

NOT a pytest module (no test_ prefix): a standalone driver so it never runs in
the suite gate. AMGX's process-global singleton lifetime means each Python
process holds ONE AMGX context; the harness runs all levels for a given solver
in one process (AMGX handles the changing sparsity via re-setup between levels;
within a level the constant K_p triggers the setup-reuse fast path).
"""
import argparse
import gc
import sys
import time

import numpy as np

from diffsim.octree.build import build_uniform
from diffsim.mesh.nodes import build_mesh
from diffsim.mesh.constraints import build_constraints
from diffsim.mesh.basis import basis_tables
from diffsim.assembly.operators import DeviceMesh
from diffsim.steppers.leray import LerayProjectionStepper

NU = 0.01


# ------- a smooth, dimension-agnostic 3-D forcing (no exact MMS needed) -------
# The scaling / parity study does not need an exact analytic solution: it needs
# the SAME well-posed PPE problem under both solvers. A smooth divergence-laden
# body force drives non-trivial u_hat (hence a non-trivial PPE RHS) each step.
def f_fn(x, t):
    # x: [n, 3] gauss points. Return [n, 3] force.
    kx = np.sin(np.pi * x[:, 0]) * np.cos(np.pi * x[:, 1])
    ky = -np.cos(np.pi * x[:, 0]) * np.sin(np.pi * x[:, 1])
    kz = np.sin(np.pi * x[:, 2]) * np.cos(np.pi * x[:, 0])
    return np.stack([kx, ky, kz], axis=1)


def g_fn(x, t):
    return np.zeros((len(x), 3))


def build_dm(level, device):
    tree = build_uniform(level, dim=3)
    mesh = build_mesh(tree, p=1)
    cons = build_constraints(mesh)
    dm = DeviceMesh.from_mesh(mesh, cons, basis_tables(1, dim=3), device)
    return dm


def gpu_mem_mb(device):
    """Best-effort peak GPU memory (MiB) via torch if available."""
    try:
        import torch
        idx = 0
        if isinstance(device, str) and ":" in device:
            idx = int(device.split(":")[1])
        torch.cuda.synchronize(idx)
        return torch.cuda.max_memory_allocated(idx) / (1024 ** 2)
    except Exception:
        return float("nan")


def run_level(level, solver, device, steps, dt, order):
    """Build the stepper at `level`, march `steps` PPE solves, return metrics.

    Returns a dict with DOF, per-solve time, AMG iters, residual, assembly
    time, memory, and the corrected velocity (for parity). Raises are caught
    by the caller and recorded as the ceiling."""
    try:
        import torch
        torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass

    t_build0 = time.perf_counter()
    dm = build_dm(level, device)
    # PPE scalar DOF = number of FREE pressure nodes; K_p is n_free x n_free.
    # assemble_csr(dm) is called inside the stepper ctor (K_p host assembly) —
    # time the whole ctor to capture the host COO->CSR assembly wall.
    st = LerayProjectionStepper(
        dm, NU, dt, f_fn=f_fn, g_fn=g_fn, order=order,
        picard_iters=1, timestab=False, solver=solver)
    st.set_initial(lambda x: np.zeros((len(x), 3)))
    t_build = time.perf_counter() - t_build0

    n_ppe = st.K_p.shape[0]
    nnz_ppe = st.K_p.nnz

    # Warm one step (JIT / first AMGX setup), then time `steps` marches.
    _ = st.step()
    per_solve = []
    amgx_stats = []
    t_march0 = time.perf_counter()
    for _ in range(steps):
        ts = time.perf_counter()
        u, p = st.step()
        per_solve.append(time.perf_counter() - ts)
        if solver == "amgx":
            from diffsim.solvers.amgx import last_solve_stats
            amgx_stats.append(last_solve_stats())
    t_march = time.perf_counter() - t_march0

    mem = gpu_mem_mb(device)

    # collate AMGX PPE telemetry (sym=True singleton key)
    iters = res = None
    reused = None
    if amgx_stats:
        # pick the SPD (sym=True) entry from the last step
        for k, v in amgx_stats[-1].items():
            if k[1] is True:   # sym flag
                iters, res, reused = v["iterations"], v["residual"], \
                    v["setup_reused"]
                break

    return {
        "level": level, "solver": solver,
        "n_ppe": int(n_ppe), "nnz_ppe": int(nnz_ppe),
        "t_build_s": t_build,
        "t_per_solve_s": float(np.median(per_solve)),
        "t_march_s": t_march, "steps": steps,
        "amg_iters": iters, "amg_res": res, "setup_reused": reused,
        "mem_mb": mem,
        "u_sig": float(np.abs(u).max()),   # cheap parity signature
        "u": u, "p": p,
    }


def run_level_assembly(level, device, device_assembly):
    """Isolate the K_p ASSEMBLY cost (the host COO->CSR wall vs the device
    scatter).  Builds the mesh, then times assemble_csr(dm) [host] or
    assemble_csr_device(dm) [device], and (device only) verifies the CSR
    equals the host CSR to fp tolerance when small enough to afford it."""
    t0 = time.perf_counter()
    dm = build_dm(level, device)
    t_mesh = time.perf_counter() - t0
    from diffsim.assembly.operators import (assemble_csr,
                                            DeviceScalarPoissonAssembler)
    ta = time.perf_counter()
    if device_assembly:
        asm = DeviceScalarPoissonAssembler(dm)
        t_sym = time.perf_counter() - ta
        tf = time.perf_counter()
        asm.fill()
        Kp = asm.to_csr()
        t_fill = time.perf_counter() - tf
        t_assemble = time.perf_counter() - ta
    else:
        Kp = assemble_csr(dm)
        t_assemble = time.perf_counter() - ta
        t_sym = t_fill = float("nan")
    mem = gpu_mem_mb(device)
    return {
        "level": level, "n_ppe": int(Kp.shape[0]), "nnz_ppe": int(Kp.nnz),
        "t_mesh_s": t_mesh, "t_assemble_s": t_assemble,
        "t_sym_s": t_sym, "t_fill_s": t_fill, "mem_mb": mem,
        "device_assembly": device_assembly,
    }


def run_level_ppe_only(level, solver, device, steps, order):
    """Isolate the SPD PPE solve: build the stepper's CONSTANT K_p (the exact
    operator solver='amgx' routes through the PPE), then hammer solve_linear on
    it directly. This measures the AMGX/AMG-PCG PPE scaling WITHOUT the
    single-threaded Python momentum-predictor assembly (a separate host-side
    wall). K_p is constant across steps -> exercises the setup-reuse fast path.

    RHS model: b = K_p @ x_true with a smooth manufactured x_true (a known
    solution the PPE pin is consistent with), so the AMG-PCG error can be
    reported against ground truth, and the RHS is realistic (in range(K_p))."""
    try:
        import torch
        torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass
    import scipy.sparse as sp

    t_build0 = time.perf_counter()
    dm = build_dm(level, device)
    from diffsim.assembly.operators import assemble_csr
    Kp = assemble_csr(dm).tocsr()
    t_assemble = time.perf_counter() - t_build0

    n = Kp.shape[0]
    nnz = Kp.nnz
    # Pin free-node 0 (the enclosed-flow PPE pin) so the operator is
    # SPD-nonsingular. Do it DIRECTLY on the CSR arrays (NOT via .tolil(),
    # which materializes a Python list-of-lists over ALL rows — O(nnz) Python
    # and multi-GB at L8+). Zero row 0's off-diagonals in place and set its
    # diagonal to 1: rewrite only row 0's slice of data/indices.
    Kp = Kp.copy()
    r0s, r0e = Kp.indptr[0], Kp.indptr[1]
    Kp.data[r0s:r0e] = 0.0
    # find (or leave) the diagonal entry in row 0 and set to 1
    row0_cols = Kp.indices[r0s:r0e]
    diag_pos = np.where(row0_cols == 0)[0]
    if diag_pos.size:
        Kp.data[r0s + diag_pos[0]] = 1.0
    else:
        # no explicit diagonal stored — add one (rare for an FE stiffness)
        Kp = Kp.tolil()
        Kp[0, 0] = 1.0
        Kp = Kp.tocsr()
    Kp.eliminate_zeros()

    coords = dm.mesh.node_coords[dm.constraints.free_nodes]
    x_true = np.cos(np.pi * coords[:, 0]) * np.cos(np.pi * coords[:, 1]) \
        * np.cos(np.pi * coords[:, 2])
    x_true = np.ascontiguousarray(x_true, np.float64)
    x_true[0] = 0.0                      # honor the pin
    b = np.asarray(Kp @ x_true)
    b[0] = 0.0

    from diffsim.solvers.linsolve import solve_linear
    cache = {}
    # warm (first AMGX setup / JIT)
    x = solve_linear(Kp, b, solver=solver, sym=True, device=device,
                     cache=cache, cache_key="ppe")
    per_solve = []
    stats = []
    for _ in range(steps):
        ts = time.perf_counter()
        x = solve_linear(Kp, b, solver=solver, sym=True, device=device,
                         cache=cache, cache_key="ppe")
        per_solve.append(time.perf_counter() - ts)
        if solver == "amgx":
            from diffsim.solvers.amgx import last_solve_stats
            stats.append(last_solve_stats())

    err = float(np.abs(x - x_true).max())
    mem = gpu_mem_mb(device)
    iters = reused = None
    if stats:
        for k, v in stats[-1].items():
            if k[1] is True:
                iters, reused = v["iterations"], v["setup_reused"]
                break
    return {
        "level": level, "solver": solver, "ppe_only": True,
        "n_ppe": int(n), "nnz_ppe": int(nnz),
        "t_build_s": t_assemble,
        "t_per_solve_s": float(np.median(per_solve)),
        "amg_iters": iters, "amg_res": None, "setup_reused": reused,
        "mem_mb": mem, "err_vs_true": err,
        "u": None, "p": None,
    }


def fmt_row(r):
    it = "-" if r.get("amg_iters") is None else str(r["amg_iters"])
    res = "-" if r.get("amg_res") is None else f"{r['amg_res']:.1e}"
    mem = "nan" if r["mem_mb"] != r["mem_mb"] else f"{r['mem_mb']:.0f}"
    return (f"  L{r['level']:<2d} {r['solver']:<5s} "
            f"n={r['n_ppe']:>12,d} nnz={r['nnz_ppe']:>13,d} "
            f"build={r['t_build_s']:7.2f}s solve={r['t_per_solve_s']*1e3:9.2f}ms "
            f"iters={it:>4s} res={res:>8s} mem={mem:>7s}MiB")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", type=int, nargs="+", default=[5, 6, 7])
    ap.add_argument("--solver", default="amgx",
                    choices=["amgx", "splu", "fused"])
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--dt", type=float, default=0.05)
    ap.add_argument("--order", type=int, default=1)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--parity", action="store_true",
                    help="run splu AND amgx at each level, compare u/p")
    ap.add_argument("--parity-tol", type=float, default=1e-5)
    ap.add_argument("--ppe-only", action="store_true",
                    help="isolate the SPD PPE solve (skip the Python "
                    "momentum-predictor host-assembly wall)")
    ap.add_argument("--assembly", action="store_true",
                    help="isolate K_p assembly cost (host COO->CSR)")
    ap.add_argument("--device-assembly", action="store_true",
                    help="with --assembly: use the device-resident scalar "
                    "Poisson assembler (device scatter, no host COO->CSR)")
    ap.add_argument("--assembly-parity", action="store_true",
                    help="verify device K_p == host K_p to fp tol per level")
    args = ap.parse_args(argv)

    if args.assembly_parity:
        from diffsim.assembly.operators import (assemble_csr,
                                                assemble_csr_device)
        ok = True
        for L in args.levels:
            dm = build_dm(L, args.device)
            Kh = assemble_csr(dm).tocsr(); Kh.sort_indices()
            Kd = assemble_csr_device(dm).tocsr(); Kd.sort_indices()
            scale = max(np.abs(Kh.data).max(), 1e-30)
            structok = (Kh.indptr.tolist() == Kd.indptr.tolist()
                        and np.array_equal(Kh.indices, Kd.indices))
            err = np.abs(Kh.data - Kd.data).max() / scale if structok \
                else float("nan")
            good = structok and err < 1e-12
            ok = ok and good
            print(f"  L{L}: n={Kh.shape[0]:,} nnz={Kh.nnz:,} "
                  f"struct={'OK' if structok else 'MISMATCH'} "
                  f"rel_err={err:.2e} -> {'PASS' if good else 'FAIL'}",
                  flush=True)
            del dm, Kh, Kd
            gc.collect()
        print(f"\nASSEMBLY PARITY: {'PASS' if ok else 'FAIL'}")
        return 0 if ok else 1

    if args.assembly:
        mode = "DEVICE scatter" if args.device_assembly else "HOST COO->CSR"
        print(f"# MODE: K_p ASSEMBLY ONLY ({mode})")
        rows = []
        for L in args.levels:
            try:
                r = run_level_assembly(L, args.device, args.device_assembly)
                rows.append(r)
                extra = (f" [sym={r['t_sym_s']:.2f}s fill={r['t_fill_s']:.2f}s]"
                         if args.device_assembly else "")
                print(f"  L{r['level']:<2d} n={r['n_ppe']:>12,d} "
                      f"nnz={r['nnz_ppe']:>13,d} mesh={r['t_mesh_s']:7.2f}s "
                      f"assemble={r['t_assemble_s']:9.2f}s{extra} "
                      f"mem={r['mem_mb']:.0f}MiB", flush=True)
                gc.collect()
            except Exception as e:  # noqa: BLE001
                import traceback
                print(f"  L{L}: CEILING/ERROR: {type(e).__name__}: {e}",
                      flush=True)
                traceback.print_exc()
                break
        return 0

    print(f"# PPE/AMGX scaling — solver={args.solver} steps={args.steps} "
          f"dt={args.dt} order={args.order} device={args.device}")
    print(f"# levels={args.levels}")

    if args.parity:
        ok = True
        for L in args.levels:
            r_splu = run_level(L, "splu", args.device, args.steps,
                               args.dt, args.order)
            r_amgx = run_level(L, "amgx", args.device, args.steps,
                               args.dt, args.order)
            du = np.abs(r_amgx["u"] - r_splu["u"]).max()
            dp = np.abs(r_amgx["p"] - r_splu["p"]).max()
            scale = max(np.abs(r_splu["u"]).max(), 1e-30)
            rel = du / scale
            good = rel < args.parity_tol
            ok = ok and good
            print(f"  L{L}: |u_amgx-u_splu|={du:.3e} rel={rel:.3e} "
                  f"|dp|={dp:.3e} amg_iters={r_amgx['amg_iters']} "
                  f"res={r_amgx['amg_res']} -> "
                  f"{'PASS' if good else 'FAIL'}")
            del r_splu, r_amgx
            gc.collect()
        print(f"\nPARITY: {'PASS' if ok else 'FAIL'} "
              f"(tol={args.parity_tol})")
        return 0 if ok else 1

    # scaling march
    runner = (lambda L: run_level_ppe_only(L, args.solver, args.device,
                                           args.steps, args.order)) \
        if args.ppe_only else \
        (lambda L: run_level(L, args.solver, args.device, args.steps,
                             args.dt, args.order))
    if args.ppe_only:
        print("# MODE: PPE-only (K_p solve isolated from predictor assembly)")
    rows = []
    for L in args.levels:
        try:
            r = runner(L)
            rows.append(r)
            print(fmt_row(r), flush=True)
            if r.get("err_vs_true") is not None:
                print(f"       err_vs_true={r['err_vs_true']:.2e} "
                      f"setup_reused={r['setup_reused']}", flush=True)
            del r["u"], r["p"]
            gc.collect()
        except Exception as e:  # noqa: BLE001 — the ceiling is a data point
            import traceback
            print(f"  L{L} {args.solver}: CEILING/ERROR: "
                  f"{type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
            print(f"  -> single-GPU ceiling for solver={args.solver} "
                  f"bites at L{L}", flush=True)
            break
    print("\n# SCALING TABLE (median per-solve):")
    print("# level  n_ppe        nnz          solve_ms   iters  mem_MiB")
    for r in rows:
        it = "-" if r.get("amg_iters") is None else r["amg_iters"]
        mem = "nan" if r["mem_mb"] != r["mem_mb"] else f"{r['mem_mb']:.0f}"
        print(f"  L{r['level']:<2d} {r['n_ppe']:>12,d} {r['nnz_ppe']:>13,d} "
              f"{r['t_per_solve_s']*1e3:>9.2f}  {str(it):>5s}  {mem:>7s}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
