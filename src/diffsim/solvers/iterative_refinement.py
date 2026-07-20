"""Task #36 (8j activation): FP64 iterative refinement over an fp32 factor.

The mechanism recorded in docs/dev/m1a-deferred-findings.md §8j: factor the
matrix in fp32 (half the factor's bytes/bandwidth — the first-order lever at
hero scale where the cuDSS refactorize is memory-system-bound), then recover
FP64 accuracy with a few refinement sweeps whose residual is computed in
FP64.  §8j measured 2-8 sweeps to ~1e-13 across 2-D L7-L8 / 3-D L4-L5; the
refinement count is the transferable datum (logged per solve).

CONTRACT (brief scope 2):
  - the fp32 factor solves the CORRECTION equation A dx = r each sweep;
  - the residual r = b - A x is computed in FP64 against the fp64-assembled
    RHS and an fp64 SHADOW matvec of A x (NOT the fp32 stored values);
  - converge to ||r|| / ||b|| <= tol (default 1e-12) or max_iter (default
    10) sweeps.  >10 on a converged-Newton step is a conditioning FAIL for
    the caller to surface (G2) — this helper returns the count, it does not
    itself raise on non-convergence (the caller owns the policy).

The loop is deliberately backend-agnostic: it takes an fp32 `factor_solve`
callable (r_fp64 -> correction, internally fp32) and an fp64 `matvec`
(x_fp64 -> A@x_fp64).  cuDSS wiring lives in the consumers (wodo_film
_solve_device, exciton _cudss_dev_solve); this module is the pure numeric
loop, unit-testable on CPU with fake callbacks (no cuDSS).
"""
import numpy as np


def fp64_iterative_refinement(matvec, factor_solve, b, *, tol=1e-12,
                              max_iter=10, x0=None):
    """FP64 iterative refinement of an fp32-factored solve of A x = b.

    Parameters
    ----------
    matvec : callable(np.ndarray[f64]) -> np.ndarray[f64]
        The FP64 SHADOW matvec x -> A @ x (against the fp64 stored values,
        NOT the fp32 factor).  This is the residual's A-apply.
    factor_solve : callable(np.ndarray[f64]) -> np.ndarray[f64]
        Solve A y = r with the fp32 factorization; may round r to fp32
        internally and promote the result — that is the whole point.  The
        FIRST call (r == b) is the initial solve.
    b : np.ndarray
        FP64 right-hand side.
    tol : float
        Relative residual stopping tolerance ||r|| / ||b|| (default 1e-12).
    max_iter : int
        Maximum refinement sweeps AFTER the initial solve (default 10).
    x0 : np.ndarray | None
        Optional initial iterate.  None -> the fp32 initial solve of b.

    Returns
    -------
    (x, info) : (np.ndarray[f64], dict)
        x is the refined FP64 solution; info carries:
          refinements : int   sweeps taken (0 == initial solve already at
                              tol; the §8j datum is this + a note)
          converged   : bool  ||r||/||b|| <= tol reached within max_iter
          rel_resid   : float final ||r||/||b||
          resid_hist  : list  ||r||/||b|| after each sweep (initial first)
    """
    b = np.ascontiguousarray(b, np.float64)
    bnorm = float(np.linalg.norm(b))
    if bnorm == 0.0:
        # exact: A x = 0 -> x = 0 (a zero rhs never needs refinement).
        z = np.zeros_like(b)
        return z, dict(refinements=0, converged=True, rel_resid=0.0,
                       resid_hist=[0.0])

    if x0 is None:
        x = np.ascontiguousarray(factor_solve(b), np.float64)
    else:
        x = np.ascontiguousarray(x0, np.float64).copy()

    r = b - matvec(x)
    rel = float(np.linalg.norm(r)) / bnorm
    hist = [rel]
    n = 0
    while rel > tol and n < max_iter:
        dx = np.ascontiguousarray(factor_solve(r), np.float64)
        x = x + dx
        r = b - matvec(x)
        rel = float(np.linalg.norm(r)) / bnorm
        hist.append(rel)
        n += 1

    return x, dict(refinements=n, converged=(rel <= tol), rel_resid=rel,
                   resid_hist=hist)
