"""OrgElMorph course - Computational C4: the solver ecosystem.

Importable core for the solver-choice concept.  Every implicit step ends in a
linear solve of the mixed (c, mu) Cahn-Hilliard Jacobian -- an INDEFINITE
saddle matrix -- and there is no single best solver.  This module measures the
choice where it is cheap (a small 2-D CH Jacobian) with BOTH speed AND
CORRECTNESS, and pairs it with the MEASURED 3-D scaling tables from the
device-assembly and blockch dev notes.

THE HEADLINE FINDING (measured): cuDSS DIVERGES on the polynomial CH saddle.
Marching the real 20-step spinodal (level 5, dt 0.02, mu_init=consistent)
with the production stepper:

    poly  splu : c in [-1.03, 1.01]   ok
    poly  cudss: c in [-552,  542 ]   DIVERGED (field blows up ~500x)
    fh    splu : c in [ 0.06, 0.94]   ok
    fh    cudss: c in [0.001, 0.999]  ok

WHY.  The mixed (c, mu) CH Jacobian is a SADDLE, and for the polynomial well
f(c)=1/4(c^2-1)^2 the bulk curvature f''(c)=3c^2-1 goes NEGATIVE across the
spinodal band |c|<1/sqrt(3) -- so the block is genuinely INDEFINITE there.
cuDSS factorizes WITHOUT partial pivoting, which an indefinite system needs,
so its solution is wrong (a small per-solve residual does NOT imply a small
ERROR when the matrix is ill-conditioned), and the error COMPOUNDS over the
Newton/time iterations until the field blows up.  Flory-Huggins survives
because its entropic curvature f''=A(1/c + 1/(1-c)) >= 4A stays POSITIVE and
better-conditioned -- but you cannot rely on that in general.

THE TRAP this chapter teaches: a residual captured at ONE Newton iterate is
DECEPTIVELY small for cuDSS on the poly saddle (~1e-13) even though the 20-
step march diverges.  You must measure the marched SOLUTION (c.min/max), not
a one-shot residual.

THE CORRECT RECIPE, therefore:
  * SMALL: scipy `splu` (CPU direct, PIVOTED) -- safe on the indefinite saddle.
  * AT SCALE: `blockch` / `blockch_dev` -- an FGMRES around a block
    preconditioner that splits the saddle into SPD sub-solves (mass / W1 / W2
    blocks), so no monolithic indefinite factorization is ever formed.
  * NOT cuDSS on the raw CH block.  (cuDSS is fine on the well-conditioned SPD
    sub-blocks a preconditioner factors, and on block-masked 3-D film
    patterns -- but not on the raw indefinite CH saddle.)
The `--solver auto` default of splu for CH is correct.

The teaching goal is a decision, not a number: given a problem, WHICH solver,
WHY, and PROVED correct by the marched solution.
"""
import platform
import statistics
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
# S3b film physics.  s/call = per linear-solve.  These take minutes per step
# to reproduce, so we cite the dev-note measurements.  NOTE: the cuDSS column
# is the BLOCK-MASKED film pattern (cuDSS is viable there); it is NOT cuDSS on
# the raw CH saddle, which the small 2-D correctness test below shows fails.
CITED_3D = [
    ("3d_l5 (32^3)",      202_752, 6.95, 1.44, "blockch 4.8x on solve"),
    ("3d_slab64 (64x64x16)", 417_792, 17.7, 2.10,
     "blockch 8.4x on solve; 2.6x on step"),
    ("3d_slab64z32",      811_008, None, 2.16,
     "cuDSS CEILINGS (48 GB, >16 min factorization); blockch marches"),
]
CITED_FACTS = {
    "cudss_ceiling_dofs": 811_008,      # 48 GB factorization wall (masked)
    "blockch_slab64_speedup": 8.4,      # solve, slab64
    "masked_cudss_slab64_scall": 3.70,  # blockmask fixes 3-D fill: 17.7->3.70
    "amgx_verdict": "no",               # AMG adds nothing at production dt
    "matrixfree_dofs": 6_389_760,       # 128x128x64 runs on ONE 48 GB card
}


def cudss_available():
    """Graceful capability probe: True iff nvmath's cuDSS DirectSolver imports."""
    try:
        from nvmath.sparse.advanced import DirectSolver  # noqa: F401
        return True
    except Exception:
        return False


def benchmark_provenance():
    """Versioned benchmark context so a timing is reproducible/attributable:
    commit, machine, gpu, cuda, library versions, and the run knobs."""
    prov = {"host": platform.node(), "python": platform.python_version()}
    try:
        import subprocess
        prov["diffsim_commit"] = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        prov["diffsim_commit"] = "unknown"
    for mod in ("numpy", "scipy", "warp", "nvmath", "torch"):
        try:
            prov[mod] = __import__(mod).__version__
        except Exception:
            prov[mod] = "absent"
    try:
        import warp as wp
        dev = wp.get_device("cuda:0")
        prov["gpu"] = str(dev.name)
    except Exception:
        prov["gpu"] = "cpu-only"
    prov["cudss_available"] = cudss_available()
    return prov


def build_dm(level, device="cuda:0"):
    tree = build_uniform(level, dim=2)
    mesh = build_mesh(tree, p=1)
    cons = build_constraints(mesh)
    dm = DeviceMesh.from_mesh(mesh, cons, basis_tables(1, dim=2), device)
    return dm, mesh, cons


def capture_ch_system(level, device="cuda:0", warmup=4, energy="poly"):
    """Assemble ONE real Cahn-Hilliard Newton system (the mixed (c, mu)
    saddle Jacobian A and residual b) at a seeded spinodal iterate.

    Uses the stepper's SUPPORTED capture API (`capture_system=True` ->
    `st.last_system`) -- NOT a monkeypatch of `solve_linear`.  Warms up a few
    steps into the quench so the residual b is NON-TRIVIAL (a consistent-mu
    step-0 iterate has b ~ machine zero, which makes a relative-residual
    correctness test degenerate).  Returns (A_csr, b, dofs, nnz)."""
    dm, mesh, cons = build_dm(level, device)
    st = CahnHilliardStepper(dm, 1.0, 5e-4, 0.02, order=1,
                             linsolver="splu", capture_system=True,
                             energy=energy)
    rng = np.random.default_rng(3)
    m0 = 0.5 if energy == "fh" else 0.0
    st.set_initial(lambda x: m0 + 0.05 * rng.standard_normal(len(x)),
                   mu_init="consistent")
    for _ in range(max(1, warmup)):
        st.step()                              # assembles + stores last_system
    A, b = st.last_system
    A = A.tocsr()
    return A, b, dm.n_nodes * 2, int(A.nnz)


def march_divergence(level=5, steps=20, dt=0.02, device="cuda:0",
                     energies=("poly", "fh"), solvers=("splu", "cudss")):
    """THE headline correctness test: march the real spinodal with the
    production stepper under each (energy, solver) and report the final field
    range c.min/max.  A blown-up range means the solver DIVERGED -- the true
    test of a solver on the indefinite CH saddle (a one-iterate residual can
    be deceptively small; see solver_correctness).  Measured: cuDSS diverges
    on the polynomial well (indefinite f''<0 in the spinodal band, no
    pivoting) but survives Flory-Huggins; splu is safe on both."""
    rows = []
    for energy in energies:
        m0 = 0.5 if energy == "fh" else 0.0
        for solver in solvers:
            if solver == "cudss" and not cudss_available():
                rows.append(dict(energy=energy, solver=solver, cmin=None,
                                 cmax=None, diverged=None,
                                 note="cuDSS unavailable"))
                continue
            dm, mesh, cons = build_dm(level, device)
            st = CahnHilliardStepper(dm, 1.0, 5e-4, dt, order=1,
                                     linsolver=solver, energy=energy,
                                     fh_A=1.0, fh_B=3.0)
            rng = np.random.default_rng(3)
            st.set_initial(lambda x: m0 + 0.05 * rng.standard_normal(len(x)),
                           mu_init="consistent")
            c = st.hist[0]
            try:
                for _ in range(steps):
                    c, _ = st.step()
                cmin, cmax = float(c.min()), float(c.max())
            except Exception as e:
                cmin = cmax = float("nan")
            # physical range: poly ~[-1,1], FH ~(0,1); >2x outside => diverged
            hi_ref = 1.2 if energy == "poly" else 1.05
            lo_ref = -1.2 if energy == "poly" else -0.05
            diverged = not (lo_ref < cmin and cmax < hi_ref)
            rows.append(dict(energy=energy, solver=solver, cmin=cmin,
                             cmax=cmax, diverged=diverged))
    return dict(level=level, steps=steps, dt=dt, rows=rows)


def _residual(A, b, x):
    """Relative residual ||A x - b|| / ||b|| -- the correctness that a
    fast-but-wrong solver fails."""
    nb = float(np.linalg.norm(b))
    return float(np.linalg.norm(A @ x - b) / (nb if nb > 0 else 1.0))


def _solve_with(A, b, solver, device):
    """Solve A x = b with one backend; return (x, iters, converged, reason).
    Never raises -- a diverged/erroring backend is reported, not fatal (that
    is the point of the correctness check)."""
    try:
        if solver == "splu":
            x = spla.splu(A.tocsc()).solve(b)
            return x, 1, True, "direct"
        res = _linsolve.solve_linear(A, b, solver=solver, tol=1e-10,
                                     cache={}, cache_key="k", device=device,
                                     return_result=True)
        return np.asarray(res.x), res.iterations, res.converged, res.reason
    except Exception as e:
        return None, None, False, f"{type(e).__name__}: {e}"[:80]


def solver_correctness(level=6, device="cuda:0",
                       solvers=("splu", "cudss", "blockch")):
    """Solve ONE captured (poly) CH saddle with several backends and report
    the relative residual.  THE TRAP: cuDSS's one-iterate residual here is
    deceptively SMALL (~1e-13) even though the 20-step march with cuDSS
    DIVERGES (see march_divergence) -- a small residual does not imply a
    small error on an indefinite, unpivoted system.  This function exists to
    show the deception; the marched solution is the real test."""
    A, b, dofs, nnz = capture_ch_system(level, device)
    x_ref = spla.splu(A.tocsc()).solve(b)          # trusted
    rows = []
    for s in solvers:
        if s == "cudss" and not cudss_available():
            rows.append(dict(solver=s, residual=None, iters=None,
                             converged=False, diff=None,
                             note="cuDSS unavailable (nvmath not installed)"))
            continue
        # blockch needs its meta in the cache
        if s in ("blockch", "blockch_dev"):
            x, it, conv, reason = _solve_blockch(A, b, device)
        else:
            x, it, conv, reason = _solve_with(A, b, s, device)
        if x is None:
            rows.append(dict(solver=s, residual=None, iters=it,
                             converged=conv, diff=None, note=reason))
            continue
        rows.append(dict(solver=s, residual=_residual(A, b, x), iters=it,
                         converged=conv,
                         diff=float(np.linalg.norm(x - x_ref)
                                    / max(np.linalg.norm(x_ref), 1e-30)),
                         note=reason))
    return dict(level=level, dofs=dofs, nnz=nnz, rows=rows)


def _solve_blockch(A, b, device, dev_inners=False):
    """blockch needs {'sigma','m','kappa'} meta for the CH block recipe."""
    cache = {}
    solver = "blockch"
    meta = {"sigma": 1.0 / 0.02, "m": 1.0, "kappa": 5e-4}
    if dev_inners:
        meta["inners"] = "device"
    cache[("blockch_meta", "k")] = meta
    try:
        res = _linsolve.solve_linear(A, b, solver=solver, tol=1e-10,
                                     cache=cache, cache_key="k",
                                     device=device, return_result=True)
        rec = cache.get(("blockch_iters", "k"))
        it = rec[0] if rec else res.iterations
        return np.asarray(res.x), it, True, "FGMRES+block precond"
    except Exception as e:
        return None, None, False, f"{type(e).__name__}: {e}"[:80]


def _timed(fn, reps=7, sync=True):
    """CUDA-synced timing: warm-up (cold time recorded once), then `reps`
    warm calls -> median + IQR (seconds).  Returns (cold, median, iqr)."""
    import warp as wp
    t0 = time.perf_counter()
    fn()
    if sync:
        wp.synchronize()
    cold = time.perf_counter() - t0
    samples = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        if sync:
            wp.synchronize()
        samples.append(time.perf_counter() - t0)
    samples.sort()
    med = statistics.median(samples)
    q1 = samples[len(samples) // 4]
    q3 = samples[(3 * len(samples)) // 4]
    return cold, med, q3 - q1


def benchmark_solvers_2d(levels=(5, 6, 7), device="cuda:0"):
    """Time direct CPU (splu) vs direct GPU (cuDSS) on the real CH Jacobian at
    each mesh level -- factorize + solve, cold + warm, CUDA-synced, median +
    IQR -- AND report the residual of each (correctness beside speed)."""
    have_cudss = cudss_available()
    recs = []
    for lv in levels:
        A, b, dofs, nnz = capture_ch_system(lv, device)
        Ac = A.tocsc()
        x_ref = spla.splu(Ac).solve(b)
        _, splu_med, splu_iqr = _timed(lambda: spla.splu(Ac).solve(b))
        rec = dict(level=lv, dofs=dofs, nnz=nnz,
                   splu_ms=splu_med * 1e3, splu_iqr_ms=splu_iqr * 1e3,
                   splu_resid=_residual(A, b, x_ref))
        if have_cudss:
            def cu():
                return _linsolve.solve_linear(A, b, solver="cudss", tol=1e-10,
                                              cache={}, cache_key="k",
                                              device=device)
            cold, med, iqr = _timed(cu)
            x_cu = cu()
            rec.update(cudss_ms=med * 1e3, cudss_iqr_ms=iqr * 1e3,
                       cudss_cold_ms=cold * 1e3,
                       cudss_resid=_residual(A, b, x_cu),
                       speedup=splu_med / med)
        else:
            rec.update(cudss_ms=None, cudss_resid=None, speedup=None)
        recs.append(rec)
    return recs


# --- measured decision support (replaces choose_solver(dim, dofs)) ---

def recommend_solver(dofs, dim=2, nnz=None, indefinite=True, precision="fp64",
                     device_mem_gb=48.0, reuse_factorization=False,
                     newton_tol=1e-10, dt=0.02, cudss=None):
    """Measured decision SUPPORT (not a one-liner): weigh problem size,
    sparsity, block structure, precision, memory, reuse, tolerance, dt and
    hardware, and return a RECOMMENDATION WITH A RATIONALE.

    The thresholds are the measured crossovers of this chapter and the cited
    3-D dev-note tables; the rationale names WHY, so a reader can re-derive the
    choice when the hardware or problem changes."""
    if cudss is None:
        cudss = cudss_available()
    est_nnz = nnz if nnz is not None else int(dofs * (7 if dim == 2 else 15))
    reasons = []
    # 0. the indefinite (c,mu) saddle EXCLUDES a non-pivoting GPU direct solve
    # on the raw block: cuDSS DIVERGES on the polynomial CH saddle (measured
    # 20-step march blows the field up ~500x; f''<0 in the spinodal band, no
    # partial pivoting).  splu (pivoted) is safe; blockch splits into SPD
    # sub-solves.
    if indefinite:
        reasons.append("indefinite (c,mu) saddle: cuDSS DIVERGES on the raw "
                       "poly CH block (measured; no pivoting) -- exclude it; "
                       "use pivoted splu (small) or blockch (SPD sub-solves)")
    # 1. small: CPU direct, PIVOTED -> safe on the saddle
    if dofs < 2e4:
        pick = "splu"
        reasons.append(f"{dofs:.0f} dofs is small: pivoted CPU splu is safe on "
                       "the indefinite saddle and cheaper than GPU launch")
    elif dim == 2 and dofs < 5e5:
        pick = "splu (small/mid) or blockch (large 2-D)"
        reasons.append("2-D at this size: still the indefinite CH saddle, so a "
                       "PIVOTED direct solve (splu) or the CH-aware blockch "
                       "preconditioner -- NOT cuDSS on the raw block")
    else:
        # 3-D or large 2-D: MEMORY decides, and the cuDSS factorization
        # ceiling is a MEASURED empirical fact (~8e5 dofs on 48 GB, dev
        # notes), not a clean analytic fill formula -- 3-D LU fill grows far
        # faster than nnz.  Use the measured ceiling.
        cudss_ceiling = 8e5 * (device_mem_gb / 48.0)
        if dim == 3 and dofs < 5e5:
            pick = "masked cuDSS (block-masked pattern) or blockch_dev"
            reasons.append(f"3-D, ~{dofs:.0f} dofs: below the measured cuDSS "
                           f"factorization ceiling (~{cudss_ceiling:.0e} dofs "
                           f"on {device_mem_gb:.0f} GB), a block-masked cuDSS "
                           "pattern is viable; blockch_dev is the robust "
                           "default")
        elif dofs < 1.5e6:
            pick = "blockch_dev"
            reasons.append(f"3-D at ~{dofs:.0f} dofs is at/above the measured "
                           f"cuDSS ceiling (~{cudss_ceiling:.0e} dofs on "
                           f"{device_mem_gb:.0f} GB; a >16 min factorization "
                           "that never returns); blockch_dev's block "
                           "preconditioner never forms the LU")
        else:
            pick = "matrix-free (blockch_dev with matrix-free outer, or "
            pick += "multi-GPU)"
            reasons.append(f"~{dofs:.0f} dofs: even the sparse matrix/int32 "
                           "indexing or the card memory is exceeded -- only a "
                           "matrix-free apply survives")
    if reuse_factorization:
        reasons.append("factorization is reused across solves (frozen "
                       "Jacobian): a one-time direct factor amortizes -- "
                       "prefer splu/cuDSS on the reusable sub-block")
    if precision == "fp32":
        reasons.append("fp32 halves memory and bandwidth but the CH condition "
                       "number may demand fp64 for the tolerance -- verify the "
                       "residual before trusting fp32")
    if not cudss:
        reasons.append("cuDSS unavailable here: fall back to splu (small) / "
                       "blockch_dev (scale)")
    return dict(dofs=dofs, dim=dim, nnz=est_nnz, recommendation=pick,
                rationale=reasons)
