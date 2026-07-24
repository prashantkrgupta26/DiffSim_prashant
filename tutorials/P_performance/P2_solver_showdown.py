"""P2 — Solver showdown: direct vs iterative vs single-sync iterative.

LEARNING OUTCOME. You can choose a linear solver from MEASUREMENT, not
folklore: direct (splu), host-orchestrated Krylov, and the fused
device-resident Krylov — and you understand the GPU-era twist the 1-D
document's direct-vs-iterative section could not show: on a GPU the enemy
is not flops but SYNCHRONIZATION. You count the syncs.

BACKGROUND. Three ways to solve the same SPD Poisson system:
  1. splu           — O(n^1.5) flops (2-D), unbeatable small, memory-bound
                      large, refactorize every time the matrix changes.
  2. host CG        — O(n) per iteration x O(h^-1) iterations (Jacobi),
                      but EVERY dot product ships a scalar to the host and
                      stalls the pipeline (~6 syncs / iteration).
  3. fused CG (diffsim.solvers.krylov_dev) — same math, all reduction
                      scalars stay on the device; ONE readback per
                      check_every iterations. m1b findings 2: measured
                      4.6x per-iteration at 200k DOFs.

EXPECTED RESULTS (measured, RTX 6000 Ada, level 8, n = 66049):
    splu     0.51 s   (and 0.51 s again on ANY new matrix)
    host CG  0.53 s   597 iters   ~3582 syncs    0.889 ms/iter
    fused CG 0.16 s   600 iters      13 syncs    0.262 ms/iter  (3.4x)
Numbers move with hardware; the RANKING and the sync counts must not.

Run:  python tutorials/P_performance/P2_solver_showdown.py
"""
import os
import sys
import time

import numpy as np
from scipy.sparse.linalg import splu

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import _viz as _viz  # guarded viz helper (no-ops when [viz] not installed)

from diffsim.octree.build import build_uniform
from diffsim.mesh.nodes import build_mesh
from diffsim.mesh.constraints import build_constraints
from diffsim.mesh.basis import basis_tables
from diffsim.assembly.operators import DeviceMesh, assemble_csr, CSROperator
from diffsim.solvers.krylov import cg as cg_host
from diffsim.solvers.krylov_dev import cg_dev, SyncCounter
from diffsim import default_device

DEVICE = default_device()


def build(level):
    tree = build_uniform(level, dim=2)
    mesh = build_mesh(tree, p=1)
    cons = build_constraints(mesh)
    dm = DeviceMesh.from_mesh(mesh, cons, basis_tables(1, dim=2), DEVICE)
    A = assemble_csr(dm)
    # make it SPD (pin the constant-vector nullspace of pure-Neumann K)
    A = A + 1e-3 * __import__("scipy.sparse", fromlist=["identity"]).identity(
        A.shape[0], format="csr")
    return A


def main(level=8):
    """Run the solver showdown and emit a timing comparison figure."""
    A = build(level)
    n = A.shape[0]
    rng = np.random.default_rng(1)
    b = rng.standard_normal(n)
    diag = np.asarray(A.diagonal())
    op = CSROperator(A, DEVICE)
    # warm everything (JIT + BLAS init) before timing — P1's rule (ii)
    cg_host(op, b, tol=1e-4, maxiter=20, diag=diag)
    cg_dev(op, b, tol=1e-4, maxiter=20, diag=diag)

    t0 = time.perf_counter()
    lu = splu(A.tocsc())
    x_d = lu.solve(b)
    t_direct = time.perf_counter() - t0

    t0 = time.perf_counter()
    x_h, info_h = cg_host(op, b, tol=1e-10, maxiter=20000, diag=diag)
    t_host = time.perf_counter() - t0

    sc = SyncCounter()
    t0 = time.perf_counter()
    x_f, info_f = cg_dev(op, b, tol=1e-10, maxiter=20000, diag=diag,
                         check_every=50, sync_counter=sc)
    t_fused = time.perf_counter() - t0

    ref = np.linalg.norm(x_d)
    print(f"n = {n}")
    print(f"{'solver':>10} {'time':>9} {'iters':>7} {'syncs':>7} "
          f"{'|x-x_direct|':>13}")
    print(f"{'splu':>10} {t_direct:>9.3f} {'—':>7} {'—':>7} {'0':>13}")
    print(f"{'host CG':>10} {t_host:>9.3f} {info_h['iters']:>7} "
          f"{'~' + str(6 * info_h['iters']):>7} "
          f"{np.linalg.norm(x_h - x_d) / ref:>13.2e}")
    print(f"{'fused CG':>10} {t_fused:>9.3f} {info_f['iters']:>7} "
          f"{sc.count:>7} {np.linalg.norm(x_f - x_d) / ref:>13.2e}")
    print(f"\nper-iteration: host {1e3 * t_host / info_h['iters']:.3f} ms   "
          f"fused {1e3 * t_fused / info_f['iters']:.3f} ms")
    # --- viz (additive; no-ops on base venv) ---
    _viz.history(
        __file__,
        ["splu", "host CG", "fused CG"],
        {"time (s)": [t_direct, t_host, t_fused]},
        "solver_times",
        xlabel="solver", ylabel="wall time (s)",
    )
    return t_direct, t_host, t_fused


if __name__ == "__main__":
    main()
    print("""
EXPLORE
  (a) The break-even question that decides track-D's architecture: splu
      wins ONE solve — but a Navier-Stokes step builds a NEW matrix every
      time. Time (factorize+solve) vs (fused CG from scratch) at levels
      7, 8, 9. Where is the crossover, and which side are real flow runs
      on?
  (b) Iteration counts grow like O(h^-1) with Jacobi (condition number
      O(h^-2), CG needs sqrt of it). Verify the growth across levels
      6..9. What preconditioner would flatten it? (This is the open
      m1b task: the production answer is multigrid or ASM.)
  (c) Raise check_every from 50 to 5000. Time changes little, but the
      solver may overshoot the tolerance by many iterations — quantify
      the waste. Choose a principled check_every from your measured
      iteration count.
  (d) Read the two kernels behind _dot_dev (krylov_dev.py) and count
      launches per CG iteration (~12). At ~5 us/launch, what is the
      launch floor per iteration, and at what n does real work exceed it?
      Compare with your fused-CG per-iteration time at level 6 vs 9.
""")
