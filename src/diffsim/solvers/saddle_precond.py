"""Task A1 — block-diagonal (Jacobi-by-block) saddle preconditioner.

Produces ``make_bdiag_apply(A, ndof, device) -> apply_dev``, the
block-diagonal preconditioner closure for the monolithic (u, p) saddle system
assembled by ``assemble_linear_ns``:

  Interleaved dof convention (ndof = dim+1 per node):
    velocity dofs: node i -> i*ndof + 0, ..., i*ndof + (dim-1)
    pressure dof : node i -> i*ndof + dim

  Preconditioner M^{-1}:
    velocity block   : Jacobi (1/d_u, element-wise).
    pressure block   : Jacobi with a safe floor — the PSPG-stabilized p-p
                       diagonal is nonzero but can be small, so we use
                           d_p_safe = max(|d_p|, eps_p * max_d)
                       where max_d = max(|d|) over ALL dofs and
                       eps_p = 1e-12 (documented floor, see below).
                       This prevents near-zero pressure pivots from amplifying
                       residual noise without discarding real PSPG entries.

``apply_dev(v_in, z_out)``: Warp device closure (v_in, z_out are wp.array
  of dtype float64 on `device`) that computes z = M^{-1} v in place.

Design notes
------------
- The Jacobi diagonal is extracted ONCE at construction time; per-apply is a
  single element-wise multiply (one Warp kernel launch).
- The eps_p floor is relative to max|d| so it scales with the system size and
  conditioning — it is nonzero only when |d_p| is truly negligible (< 1e-12
  of the largest diagonal entry), which never happens in practice for
  PSPG-stabilized equal-order elements at reasonable Re and mesh size.
- The apply_dev closure captures the device diagonal array; it is safe to call
  from fgmres_dev's inner loop without extra host contact.
"""
import numpy as np
import warp as wp

from ..assembly.operators import _kernel_cache
from ..assembly.femelm import FEMElm, fe_N  # noqa: F401 (used by MassBrick)
from ..api.equation import CEquation, assemble_brick_csr


def amgx_solve(F, rhs, sym=False, tol=1e-10, maxiter=2000, **kw):
    """Lazy module-level shim for the AMGX velocity-inner (Task T2).

    Defined here as a thin wrapper — NOT a top-level ``from .amgx import
    amgx_solve`` — so that importing ``saddle_precond`` on a CPU-only / CI box
    never touches ``pyamgx`` (GPU-only).  pyamgx is imported ONLY when this
    shim is actually CALLED, i.e. only on the ``inner="amgx"`` code path.

    Being a real module attribute is what lets the CPU routing test
    monkeypatch ``diffsim.solvers.saddle_precond.amgx_solve`` with a scipy-splu
    stand-in to prove the F extraction without any GPU.  The HARD documented
    constraint (AMGX only ever sees the EXTRACTED velocity block, never the raw
    saddle) is enforced by the caller in ``make_pcd_apply``, which passes the
    sub-CSR ``F`` here."""
    from .amgx import amgx_solve as _real_amgx_solve
    return _real_amgx_solve(F, rhs, sym=sym, tol=tol, maxiter=maxiter, **kw)


def _bdiag_kernel():
    """Element-wise z[i] = dinv[i] * v[i] (reuse the Jacobi kernel pattern
    from test_fgmres_dev.py; cached so it is compiled only once)."""
    k = _kernel_cache.get(("saddle_bdiag_apply",))
    if k is not None:
        return k

    @wp.kernel(module="unique", enable_backward=False)
    def bdiag_apply(dinv: wp.array(dtype=wp.float64),
                    v: wp.array(dtype=wp.float64),
                    z: wp.array(dtype=wp.float64)):
        i = wp.tid()
        z[i] = dinv[i] * v[i]

    _kernel_cache[("saddle_bdiag_apply",)] = bdiag_apply
    return bdiag_apply


def make_bdiag_apply(A, ndof, device):
    """Block-diagonal (Jacobi-by-block) preconditioner for the (u, p) saddle.

    Parameters
    ----------
    A : scipy CSR matrix
        The monolithic saddle system (shape N x N, N = n_nodes * ndof).
    ndof : int
        Dofs per node (dim+1 for a dim-dimensional problem).
        Velocity dofs are 0 .. dim-1 within each node block;
        pressure dof is dim (the last one).
    device : str
        Warp device string (e.g. "cpu" or "cuda:0").

    Returns
    -------
    apply_dev : callable
        ``apply_dev(v_in_wp, z_out_wp)`` — both wp.array(dtype=float64) on
        `device`.  Computes z = M^{-1} v (block-diagonal Jacobi).
    """
    A = A.tocsr()
    N = A.shape[0]
    diag = np.asarray(A.diagonal()).copy()   # shape (N,), host numpy

    # Guard: N must be divisible by ndof (N = n_nodes * ndof)
    assert N % ndof == 0, f"N={N} not divisible by ndof={ndof}"

    # Identify pressure dofs: node i -> i*ndof + (ndof-1)  (= i*ndof + dim)
    # All dofs not at offset (ndof-1) within a node block are velocity dofs.
    p_mask = np.zeros(N, dtype=bool)
    p_mask[np.arange(N // ndof) * ndof + (ndof - 1)] = True

    # Velocity block: plain Jacobi.  Guard zeros (can arise on Dirichlet rows
    # where the row is replaced by [0,…,1,…,0]; those diagonal entries are 1.0
    # after surgery, so the guard is defensive only).
    dinv = np.empty(N, dtype=np.float64)
    d_u = diag[~p_mask].copy()
    d_u[d_u == 0.0] = 1.0
    dinv[~p_mask] = 1.0 / d_u

    # Pressure block: abs-value floor at eps_p * max|d| (whole system).
    # The PSPG stabilization makes the p-p diagonal nonzero; the floor
    # catches pathological near-zero entries without destroying real entries.
    max_d = float(np.abs(diag).max()) if N > 0 else 1.0
    eps_p = 1e-12
    d_p = diag[p_mask].copy()
    d_p_safe = np.maximum(np.abs(d_p), eps_p * max_d)
    dinv[p_mask] = 1.0 / d_p_safe

    # Upload the inverse diagonal to the device (once).
    dinv_d = wp.array(np.ascontiguousarray(dinv), dtype=wp.float64, device=device)

    kernel = _bdiag_kernel()

    def apply_dev(v_in, z_out):
        """z_out = M^{-1} v_in  (element-wise diagonal scaling)."""
        wp.launch(kernel, dim=N, inputs=[dinv_d, v_in, z_out], device=device)

    return apply_dev


# ===========================================================================
# Task A3 — PCD (pressure convection-diffusion) Schur preconditioner
# ===========================================================================
#
# Derivation (interleaved (u, p) VMS/PSPG saddle, ndof = dim+1, pressure at
# node offset `dim`).  Permuted to block form the assembled matrix reads
#
#       [ F   G ] [u]   [f]
#       [ D   C ] [p] = [g]
#
#   F : velocity convection-diffusion-reaction (sigma*M_u + nu*K_u + adv + SBM)
#   G : pressure gradient block (u-rows, p-cols); IBP form  -(div w) p
#   D : divergence block (p-rows, u-cols);  q (div u) + PSPG
#   C : pressure-pressure block = +tauM (grad q . grad p)  (SPD-like; ns_bricks
#       line ~443 assembles it with a PLUS sign, a scaled Laplacian).
#
# The exact Schur complement onto pressure is  S = C - D F^{-1} G.  In the
# time-dependent regime F ~ sigma*M_u (mass-dominated at production dt), the
# Cahouet-Chabard / PCD approximation of S^{-1} is
#
#       S^{-1} ~ sigma * Ap^{-1} + nu * Mp^{-1}                          (CC)
#
# with Ap the pressure-space STIFFNESS (scalar Laplacian INT grad q.grad p)
# and Mp the pressure-space MASS (INT q p) on the SAME scalar Q1 space (same
# octree nodes, same hanging-node constraints as the saddle).  This is exactly
# the factored PCD form  S^{-1} ~ Mp^{-1} Fp Ap^{-1}  with the convection-free
# transient  Fp = sigma*Mp + nu*Ap:
#
#       Mp^{-1} (sigma Mp + nu Ap) Ap^{-1} = sigma Ap^{-1} + nu Mp^{-1}
#
# so the additive (CC) and factored (Mp^-1 Fp Ap^-1) forms are algebraically
# identical here; we implement the additive form (a) because it is symmetric
# in the two SPD inner solves so no operator-order / transpose ambiguity can
# creep in under right-preconditioning, and (b) it matches the proven
# block_precond.BlockAMGPreconditioner Cahouet-Chabard branch bit-for-bit.
# We OMIT the pressure convection term N_p(a) (the "Cahouet-Chabard-like"
# transient form) — honest omission: at production dt the sigma*Mp term
# dominates Fp, and the docs' physical-time finding is that block
# preconditioning works without it.  N_p is a documented follow-on if the
# GPU ladder (A5) demands it.
#
# The block preconditioner is UPPER-triangular (same as block_precond):
#
#       z_p = S^{-1} r_p
#       z_u = F^{-1} (r_u - G z_p)
#
# Orientation under fgmres_dev (RIGHT preconditioning): fgmres applies
# z = M^{-1} v and forms A z; the additive S^{-1} is self-transpose in its two
# SPD factors, so there is no transpose to get wrong.  The measured iteration
# count (pcd < bdiag) arbitrates the derivation.
#
# Plumbing (VERIFIED least-invasive route): the linsolve backend cannot see
# `dm`, so the caller builds the pressure operators ONCE per mesh via
# build_pcd_meta(dm, nu, sigma) and passes the resulting `meta` dict through
# the EXISTING solve_linear cache under the key ("pcd_meta", cache_key) — the
# same mechanism blockch/blockamgx use for their meta.  No new argument to
# solve_linear; no dm dependency in the backend.


class _ScalarMassBrick(CEquation):
    r"""Scalar consistent-mass brick  m(w, u) = INT w u dV  (ndof=1).

    Used to assemble the pressure-space mass Mp on the scalar Q1 space via the
    standard Integrands API (assemble_brick_csr), so Mp inherits the SAME
    hanging-node constraint condensation (T^T M T) that the saddle used."""
    ndof = 1

    @staticmethod
    @wp.func
    def Integrands_Ae(fe: FEMElm,
                      Ntab: wp.array2d(dtype=wp.float64),
                      dNtab: wp.array3d(dtype=wp.float64),
                      detJxW: wp.float64, dscale: wp.float64,
                      nbf: wp.int32, dim: wp.int32, ndof: wp.int32,
                      Ae: wp.array3d(dtype=wp.float64), e: wp.int32):
        # M^e_{ab} += N_a N_b * |J| w_q
        for a in range(nbf):
            for b in range(nbf):
                Ae[e, ndof * a, ndof * b] += (fe_N(Ntab, fe, a)
                                              * fe_N(Ntab, fe, b) * detJxW)


def build_pcd_meta(dm, nu, sigma, p_pin=None, inner="jacobi",
                   ap_inner="jacobi"):
    """Assemble the pressure-space PCD operators ONCE per mesh from ``dm``.

    Returns ``meta = {"ndof", "dim", "Mp", "Ap", "sigma", "nu", "p_pin",
    "inner", "ap_inner"}``
    where

      Mp : scalar pressure MASS   INT q p dV   (constrained T^T M T)
      Ap : scalar pressure STIFFNESS INT grad q . grad p dV (constrained
           T^T K T) — the pressure Laplacian for the Cahouet-Chabard Schur.

    Both are assembled on the SAME scalar Q1 space as the saddle's pressure
    dofs (same octree nodes, same ``dm.constraints``), so the constraint
    condensation matches the saddle bit-for-bit (assemble_brick_csr applies
    ``T^T (.) T`` with ndof=1 — the scalar analogue of the saddle's ndof=3
    ``T_vec``).  CONSTRAINT-HANDLING DECISION: reuse assemble_brick_csr's
    condensation (the exact path the driver uses for scalar bricks); no
    separate hanging-node logic.

    ``inner`` (Task T2): the F-block (velocity) inner-solve backend.
    ``"jacobi"`` (default) = today's Jacobi-CG on the velocity block, BIT-FOR-BIT
    unchanged.  ``"amgx"`` routes the F-inner through AMGX algebraic multigrid
    on the EXTRACTED velocity sub-CSR (never the raw saddle — documented
    divergence).  Only the F block changes; the Ap and Mp inners are Jacobi-CG
    in BOTH modes (they are the SPD pressure-space solves and are unaffected by
    this task by design).  ``inner`` is a cheap flag on the meta dict; the
    per-mesh operator assembly (Mp, Ap) is identical either way.

    ``ap_inner`` (Task T5): the Ap-block (pressure Laplacian) inner-solve
    backend.  ``"jacobi"`` (default) = today's Jacobi-CG on Ap, BIT-FOR-BIT
    unchanged.  ``"amgx"`` routes the Ap-inner through AMGX PCG+classical-AMG
    on the ALREADY-PINNED scalar Ap (sym=True — Ap is SPD after the pin; the
    constant-pressure nullspace is removed by the same pin the saddle uses).
    Ap sparsity is CONSTANT across steps (build_pcd_meta builds Ap once per
    mesh; the AMG hierarchy is built exactly once per run and reused via the
    setup-reuse path in amgx.py).  The AMGX singleton key (True, 1e-4, 200)
    is DISTINCT from the F-inner key (False, 1e-4, 50) — no singleton thrash
    even when both are live simultaneously (per the CAVEAT in amgx.py).
    Stats feed the T1 schema for the Ap block (iters from last_solve_stats,
    cap_hits=0 convention matching the F-amgx path).

    ``p_pin`` (optional): the monolithic pressure-pin dof index.  CONSTRAINT
    on Ap (MEASURED, root-caused): the pressure STIFFNESS Ap is a pure NEUMANN
    Laplacian — SINGULAR by the constant-pressure nullspace.  Jacobi-CG on a
    singular Ap BREAKS DOWN (rho/pTAp -> 0/0, NaN/Inf) and poisons the whole
    preconditioner (measured: outer FGMRES blows up to relres ~1e140 / stalls
    at 1e-6).  So when ``p_pin`` is given we PIN Ap at the same pressure node
    the saddle pins: replace row & column ``p_pin_local`` with identity,
    turning the singular Neumann Laplacian into an SPD Neumann->Dirichlet
    operator (the standard Cahouet-Chabard treatment).  This is consistent
    with the saddle (which pins the SAME pressure dof to an identity row) and
    makes the inner Jacobi-CG well-posed.  Mp (mass) is SPD already and is not
    pinned.  With the pin, exact-inner PCD converges in 2 outer iters vs bdiag
    4 on this level-4 saddle (measured).
    """
    from ..api import PoissonBrick
    dim = dm.dim
    ndof = dim + 1
    Mp = assemble_brick_csr(dm, _ScalarMassBrick).tocsr()
    Ap = assemble_brick_csr(dm, PoissonBrick).tocsr()
    # local pressure-node index of the global pin dof (pin dof = node*ndof+dim)
    p_pin_local = None if p_pin is None else int(p_pin) // ndof
    if p_pin_local is not None:
        Ap = _pin_symmetric(Ap, p_pin_local)
    if inner not in ("jacobi", "amgx"):
        raise ValueError(
            f"build_pcd_meta: inner must be 'jacobi' or 'amgx', got {inner!r}")
    if ap_inner not in ("jacobi", "amgx"):
        raise ValueError(
            f"build_pcd_meta: ap_inner must be 'jacobi' or 'amgx', got {ap_inner!r}")
    return {"ndof": ndof, "dim": dim, "Mp": Mp, "Ap": Ap,
            "sigma": float(sigma), "nu": float(nu),
            "p_pin_local": p_pin_local, "inner": inner, "ap_inner": ap_inner}


def _pin_symmetric(M, i):
    """Return a copy of sparse ``M`` with row & column ``i`` replaced by the
    identity (M[i,i]=1, all other entries in row/col i zeroed) — turns a
    singular Neumann Laplacian into an SPD Neumann->Dirichlet operator, the
    standard Cahouet-Chabard pin.  Symmetric (row AND col) so the pinned Ap
    stays SPD for Jacobi-CG.  Vectorized: mask out any nnz in row i or col i,
    then add the (i,i) identity — O(nnz), no Python row scan (scale-safe)."""
    import scipy.sparse as sp
    M = M.tocoo(copy=True)
    keep = (M.row != i) & (M.col != i)
    row = np.concatenate([M.row[keep], [i]])
    col = np.concatenate([M.col[keep], [i]])
    data = np.concatenate([M.data[keep], [1.0]])
    return sp.csr_matrix((data, (row, col)), shape=M.shape)


def make_pcd_apply(A, meta, device, stats=None):
    """PCD Schur preconditioner apply for the interleaved (u, p) saddle.

    Implements  P^{-1} = upper-block-triangular with
        z_p = S^{-1} r_p,  S^{-1} ~ sigma Ap^{-1} + nu Mp^{-1}   (Cahouet-
                                                                 Chabard / PCD)
        z_u = F^{-1} (r_u - G z_p)
    where F, G are the velocity/gradient blocks extracted from A by the
    interleaved component mask (pressure at offset `dim`), and Ap/Mp are the
    pressure-space operators in ``meta`` (build_pcd_meta).

    Inner solves (PRECONDITIONER strength — the outer FGMRES is the true gate):
    Jacobi-CG on F (velocity block), on Ap and Mp (both SPD).  cg_dev on
    `device`.  Inner tol is _INNER_TOL (see constant below for rationale).

    Parameters
    ----------
    stats : dict or None
        T1 telemetry accumulator.  When a dict is provided each apply
        accumulates records under keys ``"F"``, ``"Ap"``, ``"Mp"``::

            {blk: {"applies": int, "iters_total": int,
                   "cap_hits": int, "max_exit_relres": float}}

        ``applies``        — number of times this block's inner CG was called
        ``iters_total``    — cumulative CG iteration count across all applies
        ``cap_hits``       — applies where iters reached ``_INNER_MAX``
        ``max_exit_relres``— maximum exit relative residual across all applies

        ``stats=None`` (the default) is byte-identical to the old signature.

    Returns ``apply_dev(v_in_wp, z_out_wp)`` — device-in/device-out closure
    (wp.array float64).  The block gather/scatter is by component mask (the
    interleaved layout is NOT 2x2-block-partitioned), mirroring
    make_bdiag_apply; the inner solves run through the device Krylov stack, so
    the closure round-trips r/z to host once per apply (CPU-correct; the fully
    device-resident apply is a later-task concern, matching bdiag's staging)."""
    from ..assembly.operators import CSROperator
    from .krylov_dev import cg_dev

    A = A.tocsr()
    N = A.shape[0]
    ndof = meta["ndof"]
    dim = meta["dim"]
    assert N % ndof == 0, f"N={N} not divisible by ndof={ndof}"
    n = N // ndof

    # component masks (interleaved): velocity dofs 0..dim-1, pressure dof dim
    node = np.arange(n)
    u_ids = (node[:, None] * ndof + np.arange(dim)[None, :]).ravel()
    p_ids = node * ndof + dim

    F = A[u_ids][:, u_ids].tocsr()          # velocity block
    G = A[u_ids][:, p_ids].tocsr()          # pressure-gradient block

    Mp = meta["Mp"].tocsr()
    Ap = meta["Ap"].tocsr()
    sigma, nu = meta["sigma"], meta["nu"]
    p_pin_local = meta.get("p_pin_local")
    # Task T2: F-block (velocity) inner-solve backend.  "jacobi" (default) =
    # Jacobi-CG through the device Krylov stack, bit-for-bit today's behavior.
    # "amgx" = AMGX algebraic multigrid on the EXTRACTED velocity sub-CSR F
    # (already extracted ONCE above — never the raw saddle).
    inner = meta.get("inner", "jacobi")
    # Task T5: Ap-block (pressure Laplacian) inner-solve backend.  "jacobi"
    # (default) = Jacobi-CG on the pinned Ap, BIT-FOR-BIT today's behavior.
    # "amgx" = AMGX PCG+classical-AMG on the ALREADY-PINNED scalar Ap (sym=True
    # — Ap is SPD after the standard Cahouet-Chabard pin; the singleton key
    # (True, 1e-4, 200) is distinct from the F-inner key (False, 1e-4, 50)).
    ap_inner = meta.get("ap_inner", "jacobi")

    # device operators + Jacobi diagonals (guard zeros / sign for robustness)
    opF = CSROperator(F, device)
    opAp = CSROperator(Ap, device)
    opMp = CSROperator(Mp, device)

    def _jac(M):
        d = np.abs(np.asarray(M.diagonal()).copy())
        d[d == 0.0] = 1.0
        return d

    dF, dAp, dMp = _jac(F), _jac(Ap), _jac(Mp)

    # Inner tolerance for the three Jacobi-CG solves (F, Ap, Mp) per apply.
    #
    # fgmres_dev is a FLEXIBLE outer (it stores the preconditioned Z-basis, not
    # just the Krylov basis), so call-varying / inexact preconditioners are
    # mathematically sound — FGMRES convergence theory guarantees this.
    #
    # Measured on the level-4 test saddle (fgmres_dev outer, tol 1e-8):
    #   inner 1e-2  ->  21 total inner FGMRES iters,  rel-err 2.6e-10 vs splu
    #   inner 1e-4  ->  20 total inner FGMRES iters,  rel-err 2.3e-10 vs splu
    #   inner 1e-8  ->  19 total inner FGMRES iters,  rel-err 7.7e-10 vs splu
    # All three converge cleanly; the extra outer iterations from looser inners
    # are negligible.  1e-4 is chosen as the default: near-minimal outer
    # iteration count with margin; the A5 GPU ladder may tune this further.
    #
    # NOTE: an earlier implementation comment claimed that inner tol 1e-2
    # "DIVERGES the outer."  That observation was an artifact of a HOST scipy
    # lgmres probe used during development.  scipy lgmres is NOT robustly
    # flexible with call-varying operators (it reuses the Krylov basis across
    # restarts without re-preconditioning), so inexact inners genuinely caused
    # issues there.  That finding does NOT transfer to the shipped fgmres_dev
    # backend, which IS robustly flexible.  The 1e-8 value that followed from
    # that probe was over-tightened and needlessly expensive at 9.24M-DOF scale.
    _INNER_TOL, _INNER_MAX = 1e-4, 500
    # Task T2: AMGX F-inner budget.  A small max_iters keeps the F-inner a cheap
    # SMOOTHER (the flexible outer FGMRES carries the remaining residual), which
    # is the whole point of replacing Jacobi-CG with algebraic multigrid: AMG
    # gets close in a handful of BiCGStab+V-cycle iterations instead of the
    # hundreds Jacobi-CG needs at scale.  _INNER_TOL is shared with the Jacobi
    # path so the F-inner target is identical across backends.
    _AMGX_MAX = 50

    def _cg(op, y, diag, tol, mx, blk=None):
        """Inner Jacobi-CG solve.

        blk : str or None
            When ``stats`` (outer closure variable) is a dict and ``blk`` is
            one of "F"/"Ap"/"Mp", accumulate per-apply telemetry:
            iters, cap-hit flag, exit relres.  Zero-rhs short-circuit is
            NOT counted as an apply (it produces an exact zero solution with
            no CG work).
        """
        if not np.any(y):
            return np.zeros_like(y)
        x_, info = cg_dev(op, y, tol=tol, atol=1e-30, maxiter=mx, diag=diag,
                          check_every=10)
        # accept the truncated iterate even if the cap is hit (a smoother, not
        # an exact solve) — the outer FGMRES carries the remaining residual.
        if stats is not None and blk is not None:
            rec = stats[blk]
            it = info.get("iters", 0)
            rr = info.get("relres", 0.0)
            rec["applies"] += 1
            rec["iters_total"] += it
            if it >= mx:
                rec["cap_hits"] += 1
            if rr > rec["max_exit_relres"]:
                rec["max_exit_relres"] = float(rr)
        return x_

    def _amgx_F(y):
        """AMGX F-inner: BiCGStab + classical-AMG on the EXTRACTED velocity
        block F (sym=False — the convection makes F nonsymmetric).  ``F`` was
        extracted ONCE at make_pcd_apply construction (each outer solve
        builds a fresh closure over its own F; the AMGX singleton state
        persists globally across closures — same sparsity => values-only
        refresh, see the key CAVEAT in amgx.py); AMGX's own
        setup-reuse (see amgx.py) skips the AMG-hierarchy rebuild when the
        sparsity is unchanged across applies/steps, so per-apply work is
        solve-only.  Stats fed in the SAME T1 schema: iters from
        last_solve_stats(), cap_hits=0 convention (AMGX truncates internally
        and we accept the iterate; no repo-side cap), exit relres best-effort
        (AMGX residual history is off by default -> 0.0)."""
        if not np.any(y):
            return np.zeros_like(y)
        # module-level shim (monkeypatchable on CPU; imports pyamgx lazily on
        # the GPU path only) — passed the sub-CSR F, NEVER the raw saddle.
        x_ = amgx_solve(F, y, sym=False, tol=_INNER_TOL, maxiter=_AMGX_MAX)
        if stats is not None:
            rec = stats["F"]
            rec["applies"] += 1
            it, rr = 0, 0.0
            try:
                from .amgx import last_solve_stats
                key = ("singleton", False, float(_INNER_TOL), int(_AMGX_MAX))
                s = last_solve_stats().get(key)
                if s is not None:
                    it = int(s.get("iterations") or 0)
                    rr = float(s.get("residual") or 0.0)
            except Exception:
                pass
            rec["iters_total"] += it
            # cap_hits=0 convention: AMGX truncates at max_iters internally and
            # we accept the iterate; no separate repo-side cap counting.
            if rr > rec["max_exit_relres"]:
                rec["max_exit_relres"] = rr
        return x_

    # Task T5: AMGX Ap-inner budget.  200 iterations is generous for a PCG
    # + classical-AMG on the SCALAR pressure Laplacian (SPD, M-matrix-like
    # after the Cahouet-Chabard pin) — expected O(10) iterations.
    # Key (True, 1e-4, 200) is DISTINCT from the F-inner key (False, 1e-4, 50).
    _AMGX_AP_MAX = 200

    def _amgx_Ap(y):
        """AMGX Ap-inner: PCG + classical-AMG on the ALREADY-PINNED scalar Ap
        (sym=True — the pin makes Ap SPD; classical-AMG is the canonical Ap
        solver).  Ap is CONSTANT across steps (built once per mesh in
        build_pcd_meta; the AMG hierarchy is built exactly once per run and
        reused via amgx.py's setup-reuse).  Stats fed in the T1 schema:
        iters from last_solve_stats() keyed by (True, 1e-4, 200),
        cap_hits=0 convention (AMGX truncates internally; we accept the
        iterate as a smoother; the outer FGMRES carries the remaining
        residual)."""
        if not np.any(y):
            return np.zeros_like(y)
        # module-level shim (monkeypatchable on CPU; imports pyamgx lazily
        # on the GPU path only) — passed the SCALAR pinned Ap, NEVER the
        # raw saddle.  Shape: (n_pressure_nodes, n_pressure_nodes).
        x_ = amgx_solve(Ap, y, sym=True, tol=_INNER_TOL, maxiter=_AMGX_AP_MAX)
        if stats is not None:
            rec = stats["Ap"]
            rec["applies"] += 1
            it, rr = 0, 0.0
            try:
                from .amgx import last_solve_stats
                key = ("singleton", True, float(_INNER_TOL), int(_AMGX_AP_MAX))
                s = last_solve_stats().get(key)
                if s is not None:
                    it = int(s.get("iterations") or 0)
                    rr = float(s.get("residual") or 0.0)
            except Exception:
                pass
            rec["iters_total"] += it
            # cap_hits=0 convention (same as F-amgx path).
            if rr > rec["max_exit_relres"]:
                rec["max_exit_relres"] = rr
        return x_

    def _solve_F(y):
        """Dispatch the F-block (velocity) inner solve by the meta ``inner``
        flag: Jacobi-CG (default) or AMGX on the extracted velocity block."""
        if inner == "amgx":
            return _amgx_F(y)
        return _cg(opF, y, dF, _INNER_TOL, _INNER_MAX, blk="F")

    def _solve_Ap(y):
        """Dispatch the Ap-block (pressure Laplacian) inner solve by the meta
        ``ap_inner`` flag: Jacobi-CG (default) or AMGX on the pinned Ap."""
        if ap_inner == "amgx":
            return _amgx_Ap(y)
        return _cg(opAp, y, dAp, _INNER_TOL, _INNER_MAX, blk="Ap")

    def _apply_host(r):
        r_u = r[u_ids]
        r_p = r[p_ids]
        # Schur:  z_p = sigma Ap^{-1} r_p + nu Mp^{-1} r_p   (Cahouet-Chabard).
        # Ap in meta is already PINNED (SPD) when p_pin was supplied, so the
        # inner solve on Ap is well-posed (a singular Neumann Ap breaks CG).
        z_p = (sigma * _solve_Ap(r_p)
               + nu * _cg(opMp, r_p, dMp, _INNER_TOL, _INNER_MAX, blk="Mp"))
        if p_pin_local is not None:
            # the saddle pins this pressure dof to an identity row; make the
            # preconditioner respect it (pass the residual straight through)
            z_p[p_pin_local] = r_p[p_pin_local]
        # velocity: z_u = F^{-1} (r_u - G z_p) — Jacobi-CG or AMGX per meta
        r_u_corr = r_u - G @ z_p
        z_u = _solve_F(r_u_corr)
        z = np.empty_like(r)
        z[u_ids] = z_u
        z[p_ids] = z_p
        return z

    def apply_dev(v_in, z_out):
        r = v_in.numpy()
        z = _apply_host(r)
        wp.copy(z_out, wp.array(np.ascontiguousarray(z, np.float64),
                                dtype=wp.float64, device=device))

    return apply_dev
