"""Distributed Preconditioned Conjugate Gradient (PCG) solver.

Pure numpy fp64 implementation with injection-based comm, spmv, and
preconditioner protocols.  CPU-runnable with SerialComm (N=1); drop-in
replaceable with NCCL-backed variants in Task 3 without touching pcg().

Design principles
-----------------
- No torch / warp imports — stays CPU-runnable.
- fp64 throughout (explicit dtype=np.float64).
- Injection: comm, spmv, precond are passed in; pcg() has no opinions about
  the physical mesh, matrix format, or MPI/NCCL library.
- N=1 degenerate path via SerialComm equals a serial CG solve exactly.

Allreduce count per iteration
------------------------------
This implementation uses **3 allreduces per iteration**:

  1. ``rz  = allreduce_sum(r_local @ z_local)``   — for alpha numerator / beta
  2. ``pAp = allreduce_sum(p_local @ Ap_local)``   — for alpha denominator
  3. ``rr  = allreduce_sum(r_local @ r_local)``     — for convergence check

Rationale: keeping rr as a separate reduction makes the convergence check
exact (no stale residual from a prior iteration) and adds only one extra
allreduce relative to the theoretical 2-per-iter minimum.  The rz and pAp
dots are always needed for the PCG recurrence; rr is needed for the
convergence test.  With check_every > 1 optimisations, rr could be folded
into a dual-dot with rz (slot 0 and slot 1 of the same allreduce), reducing
to 2 allreduces/iter while retaining exact convergence checks — that
optimisation is deferred to Task 3 where the GPU dual-dot kernel is wired in.
"""
from __future__ import annotations

import math
from typing import Protocol, runtime_checkable

import numpy as np


# ---------------------------------------------------------------------------
# Comm protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class Comm(Protocol):
    """Protocol for inter-rank communication in distributed PCG.

    Implementors must be safe to call from the pcg() loop.

    Two methods are required:

    ``allreduce_sum(x: float) -> float``
        All-reduce a scalar by summation across all ranks and return the
        global result to every rank.

    ``exchange_halo(local_vec: np.ndarray) -> None``
        Fill the ghost entries (the ``[n_owned:]`` tail) of *local_vec*
        in-place from the owning ranks.  The owned entries (``[:n_owned]``)
        must already hold the current values before this call.  Ghost
        entries are undefined on entry and must be fully populated on exit.
        At N=1 there are no ghosts and this is a no-op.
    """

    def allreduce_sum(self, x: float) -> float:
        """Return global sum of x across all ranks."""
        ...

    def exchange_halo(self, local_vec: np.ndarray) -> None:
        """Fill ghost entries of local_vec in-place from owning ranks."""
        ...


class SerialComm:
    """Serial (N=1) communicator.  No ghosts; allreduce is identity.

    At N=1 there is exactly one rank that owns all DOFs and no ghost
    entries exist, so:
      - ``allreduce_sum`` returns *x* unchanged.
      - ``exchange_halo`` is a no-op (local_vec has no ghost tail).
    """

    def allreduce_sum(self, x: float) -> float:  # noqa: D401
        """Return x (single-rank global sum is x itself)."""
        return float(x)

    def exchange_halo(self, local_vec: np.ndarray) -> None:
        """No-op: no ghost entries at N=1."""
        return


# ---------------------------------------------------------------------------
# PCG core
# ---------------------------------------------------------------------------

def pcg(
    spmv,
    precond,
    b_local: np.ndarray,
    comm: Comm,
    *,
    rtol: float = 1e-10,
    maxit: int | None = None,
    x0: np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    """Distributed Preconditioned Conjugate Gradient solver (fp64).

    Solves A x = b where A is a distributed SPD matrix.  All vectors are
    numpy fp64 arrays of length ``n_owned`` (the number of DOFs owned by this
    rank, NOT including ghosts).

    Parameters
    ----------
    spmv : callable
        ``spmv(p_local: np.ndarray) -> np.ndarray``

        **Contract (caller's responsibility):**
        pcg() passes *p_local* as an owned-length vector (length ``n_owned``).
        The spmv wrapper (Task 3) is responsible for maintaining its own
        ghost-padded buffer of length ``n_owned + n_ghost``, copying the owned
        entries in, calling ``comm.exchange_halo`` to refresh the ghost tail,
        and then performing the local matrix-vector product.  The return value
        is the local A*p restricted to owned rows only (length ``n_owned``).
        pcg() never builds or sees a ghost-padded buffer.

        SerialComm (N=1) example::

            def spmv_serial(p):
                # No halo needed (SerialComm.exchange_halo is a no-op)
                comm.exchange_halo(p)
                return A @ p

        Multi-rank example (Task 3)::

            def spmv_dist(p_with_ghost):
                comm.exchange_halo(p_with_ghost)   # fills p_with_ghost[n_owned:]
                return A_local @ p_with_ghost       # A_local has columns for owned+ghost

    precond : callable
        ``precond(r_owned: np.ndarray) -> np.ndarray``
        Apply the preconditioner M^{-1} on owned rows.  Returns ``z_owned``
        of length ``n_owned``.  For Jacobi: ``z = r / diag(A_owned)``.
        For the identity: ``z = r.copy()``.

    b_local : np.ndarray, shape (n_owned,), dtype float64
        Local RHS vector (owned rows only).

    comm : Comm
        Communicator implementing ``allreduce_sum`` and ``exchange_halo``.

    rtol : float, default 1e-10
        Relative tolerance.  Converges when
        ``sqrt(<r,r>_global) / sqrt(<b,b>_global) < rtol``.

    maxit : int or None
        Maximum iterations.  Defaults to ``max(200, 2*n_owned)`` when None.

    x0 : np.ndarray or None
        Initial guess (owned rows, length n_owned).  Zero vector if None.

    Returns
    -------
    x_local : np.ndarray, shape (n_owned,), dtype float64
        Solution on owned rows.

    info : dict
        - ``iters``  (int)   : iterations performed
        - ``resid_history`` (list[float]) : relative residual at each iteration
          (``sqrt(<r,r>_global) / sqrt(<b,b>_global)``)
        - ``converged`` (bool) : True if rtol was achieved within maxit

    Allreduce count per iteration
    ------------------------------
    3 allreduces/iter:
      1. rz  = allreduce(r·z)       for alpha numerator and beta denominator
      2. pAp = allreduce(p·Ap)      for alpha denominator
      3. rr  = allreduce(r_new·r_new) for convergence check (exact per iter)

    Notes
    -----
    pcg() does **not** call ``comm.exchange_halo`` directly — that is entirely
    delegated to ``spmv``.  pcg() is halo-transparent: it only sees
    owned-length vectors and operates on them with numpy dot products followed
    by allreduce_sum.
    """
    b_local = np.asarray(b_local, dtype=np.float64)
    n_owned = b_local.shape[0]

    if maxit is None:
        maxit = max(200, 2 * n_owned)

    # ---- global ||b|| (needed for relative convergence criterion) ----------
    bb_local = float(np.dot(b_local, b_local))
    bb_global = comm.allreduce_sum(bb_local)
    bnorm_global = math.sqrt(max(bb_global, 0.0))
    # Guard: if b==0 the solution is 0 and we are already converged.
    if bnorm_global == 0.0:
        x_local = np.zeros(n_owned, dtype=np.float64)
        return x_local, {
            "iters": 0,
            "resid_history": [0.0],
            "converged": True,
        }

    # ---- initialise --------------------------------------------------------
    if x0 is not None:
        x_local = np.array(x0, dtype=np.float64, copy=True)
    else:
        x_local = np.zeros(n_owned, dtype=np.float64)

    # r = b - A x_0.  For x_0 = 0 this is just b.
    if x0 is not None:
        Ax0 = spmv(x_local.copy())          # spmv may need the ghost buffer
        r_owned = b_local - Ax0
    else:
        r_owned = b_local.copy()

    z_owned = precond(r_owned)              # z = M^{-1} r
    p_owned = z_owned.copy()               # p = z

    # rz = <r, z>_global
    rz_local = float(np.dot(r_owned, z_owned))
    rz = comm.allreduce_sum(rz_local)

    # initial ||r||_global for resid_history
    rr_local = float(np.dot(r_owned, r_owned))
    rr_global = comm.allreduce_sum(rr_local)
    rnorm = math.sqrt(max(rr_global, 0.0))
    rel_resid = rnorm / bnorm_global

    resid_history: list[float] = []

    # ---- main CG loop -------------------------------------------------------
    converged = False
    it = 0
    for it in range(1, maxit + 1):
        # Allreduce 2: pAp
        # pcg passes p_owned (length n_owned) to spmv.  The spmv wrapper (Task 3)
        # owns the ghost-padded buffer and calls comm.exchange_halo internally.
        Ap_owned = spmv(p_owned)            # spmv handles halo exchange internally
        pAp_local = float(np.dot(p_owned, Ap_owned))
        pAp = comm.allreduce_sum(pAp_local)

        if pAp <= 0.0:
            # A is not (semi-)positive definite in p's direction — bail out
            break

        alpha = rz / pAp

        # x = x + alpha p
        x_local += alpha * p_owned

        # r = r - alpha Ap
        r_owned -= alpha * Ap_owned

        # z = M^{-1} r
        z_owned = precond(r_owned)

        # Allreduce 1: rz_new = <r_new, z_new>_global
        rz_new_local = float(np.dot(r_owned, z_owned))
        rz_new = comm.allreduce_sum(rz_new_local)

        # Allreduce 3: rr = <r_new, r_new>_global (convergence check)
        rr_local = float(np.dot(r_owned, r_owned))
        rr_global = comm.allreduce_sum(rr_local)

        rnorm = math.sqrt(max(rr_global, 0.0))
        rel_resid = rnorm / bnorm_global
        resid_history.append(rel_resid)

        if rel_resid < rtol:
            converged = True
            break

        # beta = rz_new / rz_old
        beta = rz_new / rz if rz != 0.0 else 0.0

        # p = z + beta p
        p_owned = z_owned + beta * p_owned

        rz = rz_new

    return x_local, {
        "iters": it,
        "resid_history": resid_history,
        "converged": converged,
    }
