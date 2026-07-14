"""OrgElMorph course - Computational C4: the solver ecosystem.

Importable core for the solver-choice concept.  Every implicit step ends in a
linear solve of the mixed (c, mu) Cahn-Hilliard Jacobian -- an INDEFINITE
saddle matrix -- and there is no single best solver.  This module measures the
choice where it is cheap (a small 2-D CH Jacobian) with BOTH speed AND
CORRECTNESS, and pairs it with the MEASURED 3-D scaling tables from the
device-assembly and blockch dev notes.

CORRECTNESS FIRST, THEN SPEED.  A fast solver that returns a wrong answer is
not fast -- so this chapter reports the RELATIVE RESIDUAL ||Ax-b||/||b|| of
every solver, not just its timing.  A non-pivoting GPU direct solver on an
INDEFINITE saddle is a real theoretical RISK, so you must VERIFY it rather
than trust it.  Measured here (2-D, poly and Flory-Huggins, sigma up to 500):
scipy `splu`, cuDSS, and the CH-specific `blockch` preconditioner ALL reach
tiny residuals (~1e-11 to 1e-13) -- the verification PASSES for each.  So in
2-D correctness does not decide the choice; SPEED and, at scale, MEMORY do.

THE REAL 3-D WALL IS MEMORY, NOT ACCURACY.  A direct factorization's fill-in
explodes in 3-D; cuDSS ceilings on a 48 GB card around ~8e5 dofs (a >16 min
factorization that never returns -- cited dev notes), NOT because it is
inaccurate but because the factors do not fit.  That is where `blockch` /
`blockch_dev` (an FGMRES around a block preconditioner that never forms the
full LU) and, beyond it, matrix-free take over.  A block-masked cuDSS pattern
extends the direct reach in 3-D.  (Historical note: the two-factor blockch
PRECONDITIONER was itself hard to design -- several naive block
preconditioners DIVERGE on the stiff FH log potential; see the design-laws in
`src/diffsim/solvers/linsolve.py`.  That is a statement about approximate
preconditioners, not about a pivoted direct solve.)

The teaching goal is a decision, not a number: given a problem, WHICH solver,
WHY, and PROVED correct by its residual.
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
    """Solve the SAME captured CH saddle with several backends and report the
    relative residual, iteration count, and difference from the trusted splu
    solution.  This is the VERIFICATION step: on the 2-D CH saddle every
    backend (pivoted CPU direct splu, GPU direct cuDSS, and the CH-specific
    blockch preconditioner) reaches a tiny residual -- correctness is proved,
    not assumed.  (The point is the habit of checking; a non-pivoting direct
    solve on an indefinite saddle is a risk you must verify.)"""
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
    # 0. always VERIFY: a non-pivoting direct solve on an indefinite saddle is
    # a risk -- check the residual (measured fine in 2-D for splu AND cuDSS)
    if indefinite:
        reasons.append("indefinite (c,mu) saddle: verify the residual of any "
                       "non-pivoting direct solve (measured accurate in 2-D "
                       "for both splu and cuDSS; do not merely assume it)")
    # 1. small: CPU direct wins (no launch/transfer overhead, pivoted, exact)
    if dofs < 2e4:
        pick = "splu"
        reasons.append(f"{dofs:.0f} dofs is small: CPU splu's factorization is "
                       "cheaper than GPU launch+transfer, and it is pivoted "
                       "and exact")
    elif dim == 2 and dofs < 5e5:
        pick = "cuDSS (GPU direct; splu if no GPU)"
        reasons.append("2-D at this size: fill-in is modest, so a GPU direct "
                       "solve (cuDSS) is fast and -- verified -- accurate; "
                       "splu is the CPU fallback")
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
