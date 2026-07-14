"""OrgElMorph course - Computational C4: the solver ecosystem.

Importable core for the solver-choice concept.  Every implicit step of
this simulator ends in a linear solve, and there is no single best
solver.  This module measures the choice where it is cheap to measure --
a small 2-D Cahn-Hilliard Jacobian, direct CPU (scipy splu) vs direct
GPU (cuDSS) -- and pairs it with the MEASURED 3-D scaling tables from the
device-assembly and blockch development notes, which are far too slow to
re-run here.

The teaching goal is a decision, not a number: given a problem, WHICH
solver, and WHY.

  DIRECT (LU factorization: splu on CPU, cuDSS on GPU).  Robust, no
  tuning, exact to round-off.  Cost and memory grow superlinearly with
  the bandwidth of the matrix -- cheap in 2-D, brutal in 3-D (the fill-in
  of the factors explodes).  Measured below: cuDSS overtakes splu as the
  2-D problem grows, and cuDSS remains the 2-D and small-3-D workhorse.

  BLOCK PRECONDITIONER (blockch / blockch_dev).  An iterative outer
  solver (FGMRES) wrapped around a block factorization that never forms
  the full LU.  LATENCY-bound at small sizes (so it LOSES in 2-D by
  1-2 orders of magnitude -- cited), but its memory and per-solve cost
  scale far better, so it WINS in 3-D past the point where cuDSS's fill
  hits the card's memory ceiling.

  MATRIX-FREE.  Never stores the global matrix at all -- applies the
  Jacobian action J.v batch-wise through the element kernels.  The only
  survivor once even the sparse matrix overflows int32 indexing or the
  card's memory (the 256x256x128 class).  Cited, not implemented here.

The 2-D benchmark isolates the LINEAR SOLVE (factorize + solve on the
real CH Jacobian), not the whole step -- the step is host-assembly-bound
in this brick, which would swamp the solver signal.
"""
import time

import numpy as np
import scipy.sparse.linalg as spla

import diffsim.solvers.linsolve as _linsolve
from diffsim.octree.build import build_uniform
from diffsim.mesh.nodes import build_mesh
from diffsim.mesh.constraints import build_constraints
from diffsim.mesh.basis import basis_tables
from diffsim.assembly.operators import DeviceMesh
from diffsim.physics.cahn_hilliard import CahnHilliardStepper


# --- the measured 3-D scaling story (cited, NOT re-run) --------------
# Source: docs/dev/2026-07-13-m5-device-assembly.md (D3 table) and
# docs/dev/2026-07-13-blockch-mpf.md (B2/B3/B4/B5).  RTX 6000 Ada 48 GB,
# S3b film physics.  s/call = per linear-solve (factorize/apply).  These
# are the numbers that decide the 3-D solver choice; they take minutes
# per step to reproduce, so we cite the dev-note measurements.
CITED_3D = [
    # (case, dofs, cuDSS s/call, blockch_dev s/call, note)
    ("3d_l5 (32^3)",      202_752, 6.95, 1.44, "blockch 4.8x on solve"),
    ("3d_slab64 (64x64x16)", 417_792, 17.7, 2.10,
     "blockch 8.4x on solve; 2.6x on step"),
    ("3d_slab64z32",      811_008, None, 2.16,
     "cuDSS CEILINGS (48 GB, >16 min factorization); blockch marches"),
]
CITED_FACTS = {
    "cudss_ceiling_dofs": 811_008,      # 48 GB factorization wall
    "blockch_slab64_speedup": 8.4,      # solve, slab64
    "masked_cudss_slab64_scall": 3.70,  # blockmask fixes 3-D fill: 17.7->3.70
    "amgx_verdict": "no",               # AMG adds nothing at production dt
    "matrixfree_dofs": 6_389_760,       # 128x128x64 runs on ONE 48 GB card
}


def build_dm(level, device="cuda:0"):
    tree = build_uniform(level, dim=2)
    mesh = build_mesh(tree, p=1)
    cons = build_constraints(mesh)
    dm = DeviceMesh.from_mesh(mesh, cons, basis_tables(1, dim=2), device)
    return dm, mesh, cons


def capture_ch_system(level, device="cuda:0"):
    """Assemble ONE real Cahn-Hilliard Newton system (the mixed (c, mu)
    saddle Jacobian A and residual b) at a seeded spinodal iterate, by
    intercepting the stepper's linear solve.  Returns (A_csr, b, dofs,
    nnz) -- the exact operator the production solver would factor."""
    dm, mesh, cons = build_dm(level, device)
    st = CahnHilliardStepper(dm, 1.0, 5e-4, 0.02, order=1,
                             linsolver="cudss")
    rng = np.random.default_rng(3)
    st.set_initial(lambda x: 0.05 * rng.standard_normal(len(x)),
                   mu_init="consistent")
    cap = {}
    orig = _linsolve.solve_linear

    def spy(A, b, **kw):
        if "A" not in cap:
            cap["A"], cap["b"] = A.copy(), b.copy()
        return orig(A, b, **kw)

    _linsolve.solve_linear = spy
    try:
        st.step()                     # first Newton iterate -> spy fires
    finally:
        _linsolve.solve_linear = orig
    A = cap["A"].tocsr()
    return A, cap["b"], dm.n_nodes * 2, int(A.nnz)


def _time(fn, reps=10):
    fn()                              # warm up (compile / plan / cache)
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    return (time.perf_counter() - t0) / reps


def benchmark_solvers_2d(levels=(5, 6, 7), device="cuda:0"):
    """Time direct CPU (splu) vs direct GPU (cuDSS) on the real CH
    Jacobian at each mesh level -- factorize + solve, the fair
    comparison (both refactor each call).  Returns per-level records."""
    orig = _linsolve.solve_linear
    recs = []
    for lv in levels:
        A, b, dofs, nnz = capture_ch_system(lv, device)
        Ac = A.tocsc()
        t_splu = _time(lambda: spla.splu(Ac).solve(b))
        t_cudss = _time(lambda: orig(A, b, solver="cudss", tol=1e-10,
                                     cache={}, cache_key="k",
                                     device=device))
        recs.append(dict(level=lv, dofs=dofs, nnz=nnz,
                         splu_ms=t_splu * 1e3, cudss_ms=t_cudss * 1e3,
                         speedup=t_splu / t_cudss))
    return recs


def choose_solver(dim, dofs):
    """The decision this concept teaches, as code.  Thresholds are the
    measured crossovers (2-D: cuDSS overtakes splu ~1e4 dofs; 3-D: cuDSS
    ceiling ~8e5 dofs on a 48 GB card; matrix-free beyond int32/memory)."""
    if dim == 2:
        return "splu" if dofs < 1e4 else "cudss"
    # 3-D
    if dofs < 5e5:
        return "cudss (or masked cuDSS)"
    if dofs < 1.5e6:
        return "blockch_dev"
    return "matrix-free (or multi-GPU)"
