"""Task #36 (8j): FP64 iterative refinement over an fp32 factor.

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

Task #43 (device-resident buffers): #36's loop marshalled through host
numpy every sweep (a `.cpu()` + `from_numpy(...).to(dev)` round-trip per
matvec AND per correction solve, plus a fresh `wp.array`/`wp.zeros` per
matvec).  That per-sweep host↔device traffic is a marshalling cost unrelated
to the fp32-factor thesis — on the GH200 it would sink the fp32 datum for the
wrong reason.  The loop now accepts an optional `backend` (an ``IRBackend``)
that keeps the residual `r`, the correction `dx`, the iterate `x`, and the
shadow matvec output DEVICE-RESIDENT, allocated once and reused across
sweeps.  ``backend=None`` (the default) is the original pure-numpy path —
BIT-IDENTICAL, so the CPU unit tests are unchanged.  The device backend
performs the SAME arithmetic (residual in fp64 against the fp64 shadow
operator, correction in fp32) — only the buffer marshalling moves onto the
device.  See ``WarpIRBackend`` in the cuDSS consumers.
"""
import numpy as np


class _NumpyIRBackend:
    """Reference (host) backend: the original numpy semantics, factored out
    so the loop body is identical whether it runs on host or device buffers.

    A backend owns the persistent work buffers (``r``, ``dx``, the shadow
    matvec output) and the vector primitives (subtract-into, axpy-into,
    norm).  ``matvec``/``factor_solve`` are the caller's fp64/fp32 callables;
    for the numpy backend they take and return host numpy (the #36 contract),
    so this backend simply forwards through them.  The device backends keep
    everything resident and pass device buffers straight through."""

    def __init__(self, matvec, factor_solve):
        self._matvec = matvec
        self._factor_solve = factor_solve

    # --- buffer lifecycle -------------------------------------------------
    def as_rhs(self, b):
        """Coerce/adopt the fp64 rhs into the backend's buffer space."""
        return np.ascontiguousarray(b, np.float64)

    def norm(self, v):
        return float(np.linalg.norm(v))

    def zeros_like(self, v):
        return np.zeros_like(v)

    # --- the two caller callables, in buffer space ------------------------
    def initial(self, b):
        """The x0 == None initial fp32 solve (r == b).  On host this is just
        ``factor_solve`` — the device backends land it in the iterate buffer
        rather than the shared correction buffer (buffer-aliasing safety)."""
        return self.factor_solve(b)

    def set_iterate(self, x0):
        """Adopt a provided initial iterate x0 into the iterate buffer."""
        return np.ascontiguousarray(x0, np.float64).copy()

    def factor_solve(self, r):
        """fp32 correction solve of A y = r; returns an fp64 buffer."""
        return np.ascontiguousarray(self._factor_solve(r), np.float64)

    def matvec(self, x):
        """fp64 shadow matvec A @ x; returns an fp64 buffer."""
        return np.ascontiguousarray(self._matvec(x), np.float64)

    # --- vector arithmetic ------------------------------------------------
    def residual(self, b, ax, out=None):
        """r = b - A x  (out is ignored on host; a fresh array is fine)."""
        return b - ax

    def axpy(self, x, dx, out=None):
        """x <- x + dx  (out ignored on host)."""
        return x + dx

    def to_host(self, x):
        return np.ascontiguousarray(x, np.float64)


def fp64_iterative_refinement(matvec, factor_solve, b, *, tol=1e-12,
                              max_iter=10, x0=None, backend=None):
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
    backend : IRBackend | None
        Optional device-resident backend (Task #43).  ``None`` uses the
        reference host (numpy) backend — bit-identical to the #36 loop and
        the mode the CPU unit tests exercise.  A device backend keeps ``r``,
        the correction and the shadow matvec output resident (``matvec`` /
        ``factor_solve`` then operate device-to-device); the numerics are
        unchanged (fp64 residual against the fp64 shadow operator, fp32
        correction).  ``matvec``/``factor_solve`` are ignored when a backend
        is supplied — the backend owns them.

    Returns
    -------
    (x, info) : (np.ndarray[f64], dict)
        x is the refined FP64 solution (always returned on the HOST); info
        carries:
          refinements : int   sweeps taken (0 == initial solve already at
                              tol; the §8j datum is this + a note)
          converged   : bool  ||r||/||b|| <= tol reached within max_iter
          rel_resid   : float final ||r||/||b||
          resid_hist  : list  ||r||/||b|| after each sweep (initial first)
    """
    be = backend if backend is not None else _NumpyIRBackend(matvec,
                                                             factor_solve)
    b = be.as_rhs(b)
    bnorm = be.norm(b)
    if bnorm == 0.0:
        # exact: A x = 0 -> x = 0 (a zero rhs never needs refinement).
        z = be.zeros_like(b)
        return be.to_host(z), dict(refinements=0, converged=True,
                                   rel_resid=0.0, resid_hist=[0.0])

    if x0 is None:
        x = be.initial(b)
    else:
        x = be.set_iterate(x0)

    r = be.residual(b, be.matvec(x))
    rel = be.norm(r) / bnorm
    hist = [rel]
    n = 0
    while rel > tol and n < max_iter:
        dx = be.factor_solve(r)
        x = be.axpy(x, dx)
        r = be.residual(b, be.matvec(x))
        rel = be.norm(r) / bnorm
        hist.append(rel)
        n += 1

    return be.to_host(x), dict(refinements=n, converged=(rel <= tol),
                               rel_resid=rel, resid_hist=hist)
