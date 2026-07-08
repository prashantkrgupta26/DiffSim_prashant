"""M1d D3 EXIT TRACE (formal closure): one transient NS step at 2-D L8
and 3-D L6 (uniform lid-driven cavity), use_device_assembly=True —
per-stage wall times proving the per-step hot loop is device-resident
and host work is ORCHESTRATION-ONLY.

Solvers: cudss (device-resident zero-copy CSR, plan once + per-step
refactorize) at 2-D L8 and 3-D L5; at 3-D L6 (1.098M dofs, nnz 115M)
the cuDSS factorization exceeds the 48 GB card (ALLOC_FAILED, default
AND hybrid memory mode — measured 2026-07-08; the GH200 capacity
question made concrete), so L6 runs the fused device BiCGStab over the
same zero-copy CSR (values never leave the GPU).

Stage classes (recorded by LinearizedMonolithicStepper.trace):
  host:* — per-step HOST COMPUTE, identified explicitly: the g_fn
           strong-row callback + strong-value concat (f_fn=None here:
           zero body force evaluates nothing), history rotate.
           D3 target: SUM < 5% of step wall time.
  dev:*  — device work (GP-field kernels incl. the once-per-step node
           uploads, BDF extrapolation axpbys, slot-map assembly +
           strong rows, solve incl. the solution download).
Epoch setup (one-time: mesh/DeviceMesh, symbolic slot maps + strong-row
plan, kernel compile + solver plan inside the first step) is reported
separately — it is NOT per-step work.

Run: python benchmarks/m1d_closure_trace.py [--quick]
     (--quick: 2-D L6 / 3-D L4 smoke of the harness)
"""
import sys
import time

import numpy as np


def run_case(dim, level, solver, warm_steps=3):
    from diffsim.octree.build import build_uniform
    from diffsim.mesh.nodes import build_mesh
    from diffsim.mesh.constraints import build_constraints
    from diffsim.mesh.basis import basis_tables
    from diffsim.assembly.operators import DeviceMesh
    from diffsim.steppers.linearized import LinearizedMonolithicStepper

    def lid(x, t):
        g = np.zeros((len(x), dim))
        g[np.abs(x[:, 1] - 1.0) < 1e-12, 0] = 1.0
        return g

    print(f"\n=== {dim}-D L{level} uniform cavity, "
          f"use_device_assembly=True + {solver} ===", flush=True)
    t0 = time.perf_counter()
    tree = build_uniform(level, dim=dim)
    mesh = build_mesh(tree, p=1)
    cons = build_constraints(mesh)
    dm = DeviceMesh.from_mesh(mesh, cons, basis_tables(1, dim=dim),
                              "cuda:0")
    t_mesh = time.perf_counter() - t0
    ndofs = dm.n_free * (dim + 1)
    print(f"elements {len(tree)}  nodes {dm.n_nodes}  dofs {ndofs}",
          flush=True)

    t0 = time.perf_counter()
    st = LinearizedMonolithicStepper(
        dm, 0.01, 0.05, f_fn=None,           # zero body force
        g_fn=lid, order=2, use_device_assembly=True, solver=solver)
    t_sym = time.perf_counter() - t0          # slot maps + strong plan
    st.set_initial(lambda x: np.zeros((len(x), dim)))

    t0 = time.perf_counter()
    st.step()                                  # compile + solver plan
    t_first = time.perf_counter() - t0
    for _ in range(warm_steps - 1):
        st.step()                              # reach full BDF2+finescale

    st.trace = {}
    t0 = time.perf_counter()
    st.step()                                  # the traced step
    total = time.perf_counter() - t0
    tr = dict(st.trace)
    st.trace = None

    print(f"epoch setup (one-time): mesh+DeviceMesh {t_mesh:.2f} s | "
          f"symbolic slot maps + strong-row plan {t_sym:.2f} s | "
          f"first step (kernel compile + solver plan + step) "
          f"{t_first:.2f} s")
    print(f"\ntraced steady-state step: {total*1e3:.1f} ms total "
          f"(trace sync boundaries included)")
    print(f"{'stage':<52} {'ms':>10} {'%':>6}")
    for k in sorted(tr, key=tr.get, reverse=True):
        print(f"{k:<52} {tr[k]*1e3:>10.2f} {100*tr[k]/total:>5.1f}%")
    host = sum(v for k, v in tr.items() if k.startswith("host:"))
    dev = sum(v for k, v in tr.items() if k.startswith(("dev:",
                                                        "solve:")))
    orch = total - host - dev
    print(f"{'-'*70}")
    print(f"{'per-step HOST COMPUTE (sum of host:*)':<52} "
          f"{host*1e3:>10.2f} {100*host/total:>5.1f}%")
    print(f"{'device work (launched, incl. solve + x download)':<52} "
          f"{dev*1e3:>10.2f} {100*dev/total:>5.1f}%")
    print(f"{'untimed orchestration (python glue)':<52} "
          f"{orch*1e3:>10.2f} {100*orch/total:>5.1f}%")
    verdict = "PASS" if host / total < 0.05 else "FAIL"
    print(f"D3 gate: per-step host compute {100*host/total:.2f}% "
          f"of step time (< 5%) -> {verdict}", flush=True)
    return dict(dim=dim, level=level, ndofs=ndofs, total=total,
                host=host, trace=tr, t_sym=t_sym, t_first=t_first)


if __name__ == "__main__":
    quick = "--quick" in sys.argv
    cases = ([(2, 6, "cudss"), (3, 4, "cudss")] if quick else
             [(2, 8, "cudss"), (3, 5, "cudss"), (3, 6, "fused")])
    out = [run_case(*c) for c in cases]
    print("""
HONEST NOTES (measured):
- Remaining per-step host work: the g_fn strong-row callback + the
  strong-value concat (sub-ms), the history rotate (O(N) numpy copies),
  and — when a body force is supplied — the f_fn user callback at GPs
  (f_fn=None removes it; with a callback it measured ~2-7 ms at 2-D L8
  on this WSL2 box, allocation-dominated). BDF extrapolation/history
  combination now runs as device axpbys.
- Per-step transfers: history node vectors UP (u1, u2[, u3, p1, p2],
  each n_free*dim FP64), solution DOWN once — the output/checkpoint
  transfers the GPU-only convention allows.
- 3-D L6 cuDSS: factorization ALLOC_FAILED on 48 GB (default and
  hybrid memory mode; plan() succeeds in ~5 s) -> the direct-solver
  capacity ceiling on this card sits between 3-D L5 (144k dofs, fits
  easily) and L6 (1.098M dofs). This is exactly the pure-device vs
  coherent CAPACITY question the Nova GH200 stage prices (D5 verdict);
  the fused device BiCGStab (zero-copy CSR operator) carries L6 today.
""")
