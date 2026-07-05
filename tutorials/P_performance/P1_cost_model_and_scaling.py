"""P1 — The cost model: per-stage scaling, measured exponents, memory.

LEARNING OUTCOME. You can do for this 2-D/GPU stack exactly what
FEM_computational_cost.pdf does for the 1-D CPU solver: predict each
stage's complexity exponent analytically, MEASURE it from wall-clock
scaling, identify the bottleneck and the memory wall, and explain every
disagreement between prediction and measurement. (Read that document
first — this chapter is its hands-on continuation.)

BACKGROUND — the analytical table you are testing (n = number of DOFs):
    stage                          predicted time     predicted memory
    octree + mesh + constraints    O(n)               O(n)
    element kernels (GPU)          O(n) work          O(n) (+ constant
                                   ... but see (ii)!    per-variant JIT)
    CSR triplets -> matrix (host)  O(n)               O(n) (~9n nnz p1 2-D)
    splu factorization (2-D)       O(n^1.5)           O(n log n) fill
    triangular solves              O(n log n)         —
GPU-specific effects the 1-D document could not show you:
  (i) LAUNCH FLOOR: below ~1e5 elements a kernel launch costs more than
      the work it does — small-n timings measure overhead, not algorithm.
 (ii) JIT COMPILE: the FIRST run of each kernel variant pays seconds-to-
      minutes of compilation (disk-cached afterwards). Never time a cold
      cache. We warm everything before measuring.
(iii) HOST<->DEVICE SYNC: every .numpy() readback stalls the GPU pipeline
      (m1a findings 3 measured ~10 ms under load on WSL2).

EXPECTED RESULTS (measured, levels 5-8) — a story in two acts:
    ACT 1 (before 2026-07-05): constraints measured O(n) as predicted BUT
        with a giant Python-loop constant — 222 s at level 8, 500x the
        factorization. An O(n) stage with a bad constant beat the O(n^1.5)
        solve at every reachable size: the INVERSE of the 1-D document's
        bottleneck story. Both bottleneck species are real.
    ACT 2 (after m1b findings 9 — the per-point lookup was vectorized into
        per-level sorted-key searchsorted sweeps): constraints at level 8
        now 0.71 s (313x), and the crown RETURNS to the textbook holder:
        factorize 1.16 s > constraints 0.71 s ~ mesh 0.56 s >> assemble
        0.07 s. Measure your own table; small-n exponents remain noisy
        (launch floor, rule (i)).
    The meta-lesson outranks both acts: the ranking of stages is an
    EMPIRICAL, VERSIONED fact about the code — remeasure after every
    optimization, because your bottleneck story WILL go stale.

Run:  python tutorials/P_performance/P1_cost_model_and_scaling.py
"""
import time

import numpy as np
from scipy.sparse.linalg import splu

from diffsim.octree.build import build_uniform
from diffsim.mesh.nodes import build_mesh
from diffsim.mesh.constraints import build_constraints
from diffsim.mesh.basis import basis_tables
from diffsim.assembly.operators import DeviceMesh, assemble_csr

DEVICE = "cuda:0"


def stages(level):
    t = {}
    t0 = time.perf_counter()
    tree = build_uniform(level, dim=2)
    mesh = build_mesh(tree, p=1)
    t["mesh"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    cons = build_constraints(mesh)
    t["constraints"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    dm = DeviceMesh.from_mesh(mesh, cons, basis_tables(1, dim=2), DEVICE)
    A = assemble_csr(dm)
    t["assemble"] = time.perf_counter() - t0
    n = A.shape[0]
    rng = np.random.default_rng(0)
    b = rng.standard_normal(n)
    t0 = time.perf_counter()
    lu = splu(A.tocsc())
    t["factorize"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    lu.solve(b)
    t["backsolve"] = time.perf_counter() - t0
    return n, t, A.nnz


if __name__ == "__main__":
    stages(4)                                    # WARM the JIT cache first!
    levels = (5, 6, 7, 8)
    rows = [stages(lv) for lv in levels]
    keys = ["mesh", "constraints", "assemble", "factorize", "backsolve"]
    print(f"{'level':>6} {'n':>8} {'nnz':>9} "
          + " ".join(f"{k:>12}" for k in keys))
    for lv, (n, t, nnz) in zip(levels, rows):
        print(f"{lv:>6} {n:>8} {nnz:>9} "
              + " ".join(f"{t[k]:>12.4f}" for k in keys))
    print("\nmeasured exponents (log2 time ratio / log2 n ratio):")
    for k in keys:
        exps = []
        for i in range(len(rows) - 1):
            n0, t0, _ = rows[i]
            n1, t1, _ = rows[i + 1]
            exps.append(np.log(t1[k] / t0[k]) / np.log(n1 / n0))
        print(f"  {k:>12}: " + "  ".join(f"{e:5.2f}" for e in exps))
    # the memory-cliff estimate, GPU edition
    import shutil
    n8 = rows[-1][0]
    print(f"""
memory model at level 8 (n = {n8}):
  sparse CSR:  ~{rows[-1][2] * 16 / 1e6:.0f} MB   (16 B/nnz)
  DENSE would be 8 n^2 = {8 * n8 ** 2 / 1e9:.1f} GB  <- the cliff the 1-D
  document computes; here it arrives at level ~8 already. Sparse formats
  are not an optimization, they are admission to the building.
""")
    print("""
EXPLORE
  (a) Add p=2 columns to the table. Which stages' constants grow by the
      DOF ratio (~2.25x in 2-D) and which by the kernel-work ratio (~5x)?
  (b) Repeat at dim=3, levels 3..5. Find the level where factorize
      overtakes assemble — it comes MUCH sooner (fill-in scales worse in
      3-D: O(n^2) time, O(n^{4/3}) memory). This measured fact is the
      entire justification for P2.
  (c) Time the FIRST-EVER run of a new kernel variant (delete
      ~/.cache/warp -- careful -- or change a kernel constant) and compare
      against the warm run. Report the JIT tax for the p1 2-D variant.
  (d) Reproduce the 1-D document's flop-ledger discipline for ONE element
      matrix: count the operations in the p1 2-D stiffness kernel
      (read make_poisson_element_matrices in assembly/operators.py) and
      compare flops/second against your GPU's FP64 peak. What fraction do
      you reach, and is the kernel compute- or bandwidth-bound?
""")
