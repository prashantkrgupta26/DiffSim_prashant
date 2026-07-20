"""Task #36 (8j) G2: the FP64-iterative-refinement LOOP logic, unit-tested
on CPU with fake fp32/fp64 callbacks (no cuDSS — the cuDSS-backed G1/G2/G5
gates are CUDA-only and live in the GPU lanes).

These pin the loop's contract: the residual rides the FP64 matvec; the
correction rides the (fp32) factor_solve; the refinement count is returned
and the >max_iter case is flagged non-converged (the caller's FAIL policy).
"""
import numpy as np
import pytest

from diffsim.solvers.iterative_refinement import fp64_iterative_refinement


def _fp32_factor_solve(A, r):
    """A fake 'fp32 factorization' solve: round the operator AND the rhs to
    fp32, solve, promote — mimics the accuracy loss of an fp32 factor so IR
    has real work to do (a few sweeps to recover fp64)."""
    A32 = A.astype(np.float32).astype(np.float64)
    r32 = r.astype(np.float32).astype(np.float64)
    return np.linalg.solve(A32, r32)


def test_ir_recovers_fp64_from_fp32_factor():
    """An fp32-factored solve of a mildly ill-conditioned system is
    refined back to fp64: rel residual <= 1e-12 in a handful of sweeps."""
    rng = np.random.default_rng(0)
    n = 60
    A = rng.standard_normal((n, n))
    A = A + n * np.eye(n)                      # well-ish conditioned, cond~50
    x_true = rng.standard_normal(n)
    b = A @ x_true

    matvec = lambda x: A @ x                    # FP64 shadow matvec
    solve = lambda r: _fp32_factor_solve(A, r)  # fp32 factor solve
    x, info = fp64_iterative_refinement(matvec, solve, b, tol=1e-12,
                                        max_iter=10)
    assert info["converged"], info
    assert info["rel_resid"] <= 1e-12, info
    # §8j: a few sweeps (not zero — the fp32 factor is genuinely lossy;
    # not many — a converged Newton step is well-conditioned)
    assert 1 <= info["refinements"] <= 8, info
    assert np.linalg.norm(x - x_true) / np.linalg.norm(x_true) <= 1e-10


def test_ir_residual_uses_fp64_matvec_not_fp32():
    """The stopping residual MUST be computed with the FP64 matvec (against
    the fp64 stored values), NOT the fp32 factor.  We inject a matvec that
    is the EXACT fp64 operator and a factor_solve for a PERTURBED operator;
    if the loop (wrongly) judged convergence via the fp32 factor it would
    stop early with a large true residual.  Here it must keep refining to
    the true fp64 residual (or exhaust max_iter), and rel_resid reflects
    the FP64 operator."""
    rng = np.random.default_rng(1)
    n = 40
    A = rng.standard_normal((n, n)) + n * np.eye(n)
    x_true = rng.standard_normal(n)
    b = A @ x_true
    matvec = lambda x: A @ x                    # the TRUE fp64 operator
    solve = lambda r: _fp32_factor_solve(A, r)
    x, info = fp64_iterative_refinement(matvec, solve, b, tol=1e-13,
                                        max_iter=10)
    # rel_resid is measured against A (the fp64 matvec): a genuine fp64
    # residual, consistent with x being fp64-accurate.
    true_rel = np.linalg.norm(b - A @ x) / np.linalg.norm(b)
    assert abs(true_rel - info["rel_resid"]) <= 1e-15 + 1e-3 * info["rel_resid"]


def test_ir_nonconvergence_flagged_over_max_iter():
    """A near-singular system the fp32 factor cannot recover: the loop
    exhausts max_iter and returns converged=False with the count == max_iter
    — the >10 FAIL signal the caller (G2) surfaces (this helper does not
    raise; it reports)."""
    n = 30
    rng = np.random.default_rng(2)
    A = rng.standard_normal((n, n))
    # make it very ill-conditioned so fp32 correction stalls
    U, _, Vt = np.linalg.svd(A)
    s = np.logspace(0, -9, n)                   # cond ~ 1e9
    A = (U * s) @ Vt
    x_true = rng.standard_normal(n)
    b = A @ x_true
    matvec = lambda x: A @ x
    solve = lambda r: _fp32_factor_solve(A, r)
    x, info = fp64_iterative_refinement(matvec, solve, b, tol=1e-14,
                                        max_iter=4)
    assert not info["converged"], info
    assert info["refinements"] == 4, info


def test_ir_zero_rhs_is_exact():
    """A zero rhs solves exactly to zero with no refinement (the frozen /
    strong-row degenerate case the blockch stack also special-cases)."""
    matvec = lambda x: x
    solve = lambda r: r
    x, info = fp64_iterative_refinement(matvec, solve, np.zeros(10))
    assert np.array_equal(x, np.zeros(10))
    assert info["refinements"] == 0 and info["converged"]


def test_ir_already_converged_zero_refinements():
    """If the initial fp32 solve already meets tol (a well-conditioned,
    modestly-scaled system), refinements == 0."""
    n = 20
    A = np.eye(n) * 2.0                         # trivial, fp32-exact
    b = np.arange(1.0, n + 1.0)
    matvec = lambda x: A @ x
    solve = lambda r: _fp32_factor_solve(A, r)
    x, info = fp64_iterative_refinement(matvec, solve, b, tol=1e-10)
    assert info["converged"] and info["refinements"] == 0, info
