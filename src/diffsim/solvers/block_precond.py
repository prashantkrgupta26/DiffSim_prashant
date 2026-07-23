"""EXPERIMENTAL — block preconditioner for the monolithic stabilized (u,p)
NS system (m1b findings 8e item (i); STATUS in findings 8f: v1 measured
NOT yet effective — kept as the harness for the preconditioner study, not
wired into the linsolve dispatch).

The assembled block, in node-major interleaved DOFs, is permuted to

        [ F   G ] [u]   [f]
        [ D   C ] [p] = [g]

(F: velocity convection-diffusion-reaction; G: gradient+SUPG; D:
divergence+PSPG; C: PSPG pressure Laplacian). We apply the standard block
UPPER-triangular right preconditioner

        P^{-1} r = [ z_u ]   with  z_p = S~^{-1} r_p
                   [ z_p ]         z_u = F~^{-1} (r_u - G z_p)

- F~^{-1}: ONE AMGX classical-AMG V-cycle on F (not a solve!). At stepping
  sigma = b0/dt, F is mass-dominated — AMG's best case; a single cycle is
  the textbook preconditioner application.
- S~^{-1}: Cahouet-Chabard for the time-dependent Schur complement:
      S~^{-1} = sigma * K_p^{-1} + nu * M_p^{-1}
  with K_p the pressure stiffness (the Leray PPE operator, reused) and M_p
  the pressure mass (Jacobi-inverted: diagonal is spectrally fine here).
  K_p^{-1} via AMGX PCG at loose tolerance (1e-3, few V-cycles).

Outer Krylov: scipy's flexible GMRES/BiCGStab via LinearOperator — HOST-
orchestrated deliberately for v1: with a strong preconditioner the outer
iteration count is O(10-50), so the per-iteration host sync that killed
plain Krylov (thousands of iterations) is amortized away. The fully
device-resident outer loop is the follow-on once counts are confirmed.
"""
import os

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator


class BlockAMGPreconditioner:
    def __init__(self, A, n_nodes, ndof, Kp, Mp_diag, sigma, nu,
                 dir_rows=None, f_iters=2, f_tol=1e-2, kp_iters=8,
                 kp_tol=1e-3, f_cycles=1, kp_cycles=3,
                 schur_mode="cahouet_chabard", f_solver="amgx"):
        """A: assembled monolithic CSR (interleaved node-major DOFs).
        n_nodes: FREE nodes; ndof = dim+1. Kp: pressure stiffness on the
        same free nodes (with its own pinned row handled by caller);
        Mp_diag: pressure mass diagonal. dir_rows: strong-Dirichlet row ids
        of the monolithic system (identity rows — kept in F).

        Inner-solve STRENGTHS are tunable (defaults reproduce the original
        hardcoded behavior bit-for-bit): `f_iters`/`f_tol` govern the AMG
        solve of the velocity block F in apply(); `kp_iters`/`kp_tol` govern
        the Schur pressure-stiffness solve; `f_cycles`/`kp_cycles` set the
        AMGX max_iters (V-cycle count) of the persistent F / Kp cycles.

        `schur_mode` selects the Schur-complement approximation S~^{-1}:
        - "cahouet_chabard" (DEFAULT, bit-for-bit unchanged): the time-
          dependent Cahouet-Chabard S~^{-1} = sigma*Kp^{-1} + nu*Mp^{-1}.
          Assumes an inf-sup-STABLE saddle (no pressure-pressure block).
        - "pspg_c": S~^{-1} ≈ C^{-1}, where C = A[p_ids][:, p_ids] is the
          assembled pressure-pressure block of the monolithic matrix. For a
          PSPG-stabilized EQUAL-ORDER (P1-P1) discretization this block is
          real and nonzero (∫ τ ∇q·∇p + pressure/PSPG terms) and DOMINATES
          the true Schur S = C - D F^{-1} G — Cahouet-Chabard is WRONG here.
          Reuses the kp_iters/kp_tol/kp_cycles knobs for the C-AMG solve.
        - "diag_f": S~ = C - D diag(F)^{-1} G, assembled algebraically from
          the monolithic blocks (SIMPLE-style Schur; the same approximation
          the proven `blocktri` solver uses). Unlike Cahouet-Chabard, this
          SEES the SBM Nitsche penalty sitting in F's near-surface diagonal
          (where F is penalty-dominated, not sigma*M-dominated, so the CC
          formula is locally wrong — a subspace that GROWS ~h^-2 with
          refinement, i.e. exactly a scale-onset failure). Measured on the
          real L3/L4 sphere saddles with exact-F: diag_f 11-18 outer iters
          vs cc 17-28, and mesh-stable. Reuses the kp knobs.

        `f_solver` selects the velocity-block solve:
        - "amgx" (default): AMGX Krylov+AMG cycle (knobs above). MEASURED
          CAVEAT (L5): scalar classical AMG on the interleaved 3-dof F
          reaches only ~6.6e-2 in 100 BiCGStab iterations — the reason the
          march ground even after the FGMRES fix.
        - "cudss": EXACT F via a cuDSS factorization (nvmath DirectSolver).
          Reproduces the exact-F outer counts (diag_f: 9-20 at L5). Memory
          is the F factorization only (~3/4 of the monolithic direct solve
          that OOMs at L6 — so this is an L5-class option; at L6 try amgx
          with a better F config first)."""
        self.n, self.ndof = n_nodes, ndof
        dim = ndof - 1
        # interleaved -> blocked permutation
        idx = np.arange(n_nodes * ndof).reshape(n_nodes, ndof)
        self.u_ids = idx[:, :dim].ravel()
        self.p_ids = idx[:, dim].ravel()
        Ac = A.tocsr()
        self.F = Ac[self.u_ids][:, self.u_ids].tocsr()
        self.G = Ac[self.u_ids][:, self.p_ids].tocsr()
        self.sigma, self.nu = sigma, nu
        # dir_rows: strong-Dirichlet monolithic row ids (identity rows). They
        # live in the velocity block; a single AMG V-cycle on F does NOT
        # reproduce their exact identity action, so we enforce z_u = r_u on
        # these rows AFTER the cycle (the preconditioner must respect the
        # BC exactly, else the outer FGMRES sees a spurious BC residual).
        self._u_dir = None
        if dir_rows is not None:
            dir_rows = np.asarray(dir_rows).ravel()
            if dir_rows.size:
                # map monolithic dir rows -> position within u_ids (velocity
                # block). rows that are pressure DOFs are ignored (pins are
                # handled by the Schur block / caller).
                pos = np.full(n_nodes * ndof, -1, dtype=np.int64)
                pos[self.u_ids] = np.arange(self.u_ids.size)
                loc = pos[dir_rows]
                self._u_dir = loc[loc >= 0]
        self.Kp = Kp.tocsr()
        self.Mp_diag = np.asarray(Mp_diag)
        # inner-solve tuning knobs (defaults == original hardcoded behavior)
        self.f_iters, self.f_tol = int(f_iters), float(f_tol)
        self.kp_iters, self.kp_tol = int(kp_iters), float(kp_tol)
        self.f_cycles, self.kp_cycles = int(f_cycles), int(kp_cycles)
        self.schur_mode = str(schur_mode)
        # HONEST knob wiring (R2b.1 cliff fix): the AMGX inner solver runs
        # `*_iters` Krylov iterations (early-exit at relative `*_tol`), each
        # preconditioned by `*_cycles` AMG V-cycle(s). Previously the
        # `*_iters`/`*_tol` knobs were silently DISCARDED by
        # `_AMGXCycle.solve(b, **_ignored)` — every "strengthen the inner
        # solve" experiment was a no-op. Inner solves are inexact/nonlinear,
        # which is exactly why the outer Krylov must be FGMRES (see
        # `solve_block_preconditioned`).
        self.f_solver = str(f_solver)
        self._amg_F = None
        self._f_direct = None
        if self.f_solver == "cudss":
            # EXACT F: cuDSS factorization, persistent for this matrix.
            from nvmath.sparse.advanced import DirectSolver
            from .linsolve import cudss_options
            self._f_direct = DirectSolver(
                self.F, np.zeros(self.F.shape[0]), options=cudss_options())
            self._f_direct.plan()
            self._f_direct.factorize()
        elif self.f_solver == "amgx":
            self._amg_F = _AMGXCycle(self.F, sym=False, cycles=self.f_iters,
                                     tol=self.f_tol,
                                     pre_cycles=self.f_cycles)
        else:
            raise ValueError(f"unknown f_solver {self.f_solver!r}; "
                             "expected 'amgx' or 'cudss'")
        # Build ONLY the Schur operator the chosen mode needs, so pspg_c and
        # cahouet_chabard each use exactly 2 AMGX Resources (F + one Schur):
        # no extra Resource -> no new segfault risk.
        self._amg_Kp = None
        self._amg_C = None
        self._amg_S = None
        if self.schur_mode == "cahouet_chabard":
            self._amg_Kp = _AMGXCycle(self.Kp, sym=True, cycles=self.kp_iters,
                                      tol=self.kp_tol,
                                      pre_cycles=self.kp_cycles)
        elif self.schur_mode == "pspg_c":
            # C is the pressure-pressure block of the monolithic A — the
            # assembled PSPG pressure operator (∫ τ ∇q·∇p + PSPG/mass terms)
            # whose inverse approximates the Schur for equal-order P1-P1.
            # NOTE: A carries one PINNED pressure row/col (identity). C
            # inherits that identity row — KEEP it; a single identity row is
            # fine for AMG and it fixes the pressure nullspace. C is SPD-like,
            # so use the symmetric (PCG) AMGX cycle.
            self.C = Ac[self.p_ids][:, self.p_ids].tocsr()
            self._amg_C = _AMGXCycle(self.C, sym=True, cycles=self.kp_iters,
                                     tol=self.kp_tol,
                                     pre_cycles=self.kp_cycles)
        elif self.schur_mode == "diag_f":
            # SIMPLE-style algebraic Schur: S~ = C - D diag(F)^{-1} G. The
            # pinned pressure row of A gives C an identity row and a ZERO D
            # row there, so S~ inherits the pin's identity row (nullspace
            # fixed). Mildly nonsymmetric at convective steps (D != -G^T
            # once PSPG/SUPG convection enters) — use the BiCGStab cycle.
            D = Ac[self.p_ids][:, self.u_ids].tocsr()
            C = Ac[self.p_ids][:, self.p_ids].tocsr()
            self.S = (C - D @ sp.diags(1.0 / self.F.diagonal()) @ self.G
                      ).tocsr()
            self._amg_S = _AMGXCycle(self.S, sym=False, cycles=self.kp_iters,
                                     tol=self.kp_tol,
                                     pre_cycles=self.kp_cycles)
        else:
            raise ValueError(
                f"unknown schur_mode {self.schur_mode!r}; expected "
                "'cahouet_chabard', 'pspg_c' or 'diag_f'")

    def apply(self, r):
        r = np.asarray(r)
        r_u, r_p = r[self.u_ids], r[self.p_ids]
        if self.schur_mode == "pspg_c":
            # Schur: S~^{-1} ≈ C^{-1} (PSPG pressure block of monolithic A).
            z_p = self._amg_C.solve(r_p)
        elif self.schur_mode == "diag_f":
            # Schur: SIMPLE-style S~ = C - D diag(F)^{-1} G (penalty-aware).
            z_p = self._amg_S.solve(r_p)
        else:
            # Schur: Cahouet-Chabard
            z_p = (self.sigma * self._amg_Kp.solve(r_p)
                   + self.nu * (r_p / self.Mp_diag))
        # velocity: inner solve on the corrected residual
        r_u_corr = r_u - self.G @ z_p
        if self._f_direct is not None:
            self._f_direct.reset_operands(
                b=np.ascontiguousarray(r_u_corr, np.float64))
            z_u = np.asarray(self._f_direct.solve())
        else:
            z_u = self._amg_F.solve(r_u_corr)
        if self._u_dir is not None:
            # identity action on strong-Dirichlet rows (F rows are identity)
            z_u[self._u_dir] = r_u_corr[self._u_dir]
        z = np.empty_like(r)
        z[self.u_ids], z[self.p_ids] = z_u, z_p
        return z

    def as_linear_operator(self):
        n = self.n * self.ndof
        return LinearOperator((n, n), matvec=self.apply)

    def destroy(self):
        """Free GPU-side state (AMGX objects + cuDSS factorization). The
        march rebuilds the preconditioner EVERY Picard step (A changes);
        without an explicit teardown 80 steps would leak 80 AMGX Resources
        + cuDSS factors. linsolve's blockamgx branch calls this on the
        previous instance before building the new one."""
        for cyc in (self._amg_F, self._amg_Kp, self._amg_C, self._amg_S):
            # getattr-guarded: test stubs (_ExactCycleStub) have no destroy
            _d = getattr(cyc, "destroy", None)
            if _d is not None:
                _d()
        self._amg_F = self._amg_Kp = self._amg_C = self._amg_S = None
        if self._f_direct is not None:
            try:
                self._f_direct.free()
            except Exception:  # noqa: BLE001 — teardown is best-effort
                pass
            self._f_direct = None


class _AMGXCycle:
    """A persistent AMGX solver used as an INNER solver of the block
    preconditioner: setup once per matrix, apply a HARD-CAPPED number of
    Krylov iterations per call (early-exit at relative `tol`), each
    preconditioned by `pre_cycles` AMG V-cycle(s). CARE POINT (measured):
    reusing a solver-grade config (max_iters=2000, tol 1e-4) turns every
    preconditioner application into a near-full solve — the first probe
    ground for 19+ minutes against cuDSS's 147 ms.

    NOTE (R2b.1 cliff fix): the inner solver is a Krylov method (PBICGSTAB /
    PCG), whose action is NONLINEAR in b — so the resulting preconditioner
    is a VARYING operator and the outer Krylov MUST be flexible (FGMRES).
    Feeding this into scipy's standard `gmres` (non-flexible) is invalid and
    was the L5 stagnation cliff."""

    def __init__(self, A, sym, cycles=1, tol=0.0, pre_cycles=1):
        from .amgx import _ensure_init
        _ensure_init()
        import json
        import pyamgx
        here = os.path.join(os.path.dirname(__file__), "amgx_configs")
        fname = ("PCG_CLASSICAL_V_JACOBI.json" if sym
                 else "PBICGSTAB_CLASSICAL_JACOBI.json")
        with open(os.path.join(here, fname)) as fh:
            cfg = json.load(fh)
        cfg["solver"]["max_iters"] = int(cycles)
        # early-exit at RELATIVE_INI `tol` (0.0 = never; run all iterations)
        cfg["solver"]["tolerance"] = float(tol)
        cfg["solver"]["monitor_residual"] = 1
        # QUIET: no per-solve stats/timing spam (the outer FGMRES logging is
        # the readable signal). Grid stats print once at setup — keep them:
        # they answer the AMG-coarsening-quality question (handoff H3).
        cfg["solver"]["print_solve_stats"] = 0
        cfg["solver"]["obtain_timings"] = 0
        cfg["solver"]["preconditioner"]["max_iters"] = int(pre_cycles)
        cfg["verbosity_level"] = 1
        self.cfg = pyamgx.Config().create(json.dumps(cfg))
        self.rsc = pyamgx.Resources().create_simple(self.cfg)
        self.M = pyamgx.Matrix().create(self.rsc)
        self.X = pyamgx.Vector().create(self.rsc)
        self.B = pyamgx.Vector().create(self.rsc)
        self.slv = pyamgx.Solver().create(self.rsc, self.cfg)
        A = A.tocsr()
        A.sort_indices()
        self.M.upload_CSR(A)
        self.slv.setup(self.M)
        self._x = np.zeros(A.shape[0])
        self._tag = "sym" if sym else "nonsym"
        self._calls = 0
        # BLOCKAMGX_INNER_LOG=N -> log the achieved inner iterations/status
        # every N-th call (0/absent = off). The H3 diagnostic: is AMGX
        # actually reaching its tolerance on the SBM-structured blocks?
        self._log_every = int(os.environ.get("BLOCKAMGX_INNER_LOG", "0"))

    def solve(self, b, **_ignored):
        self.B.upload(np.ascontiguousarray(b, np.float64))
        self._x[:] = 0.0
        self.X.upload(self._x)
        self.slv.solve(self.B, self.X, zero_initial_guess=True)
        self.X.download(self._x)
        self._calls += 1
        if self._log_every and self._calls % self._log_every == 1:
            try:
                it = int(self.slv.iterations_number)
                st = self.slv.status
                res = None
                try:
                    res = float(self.slv.get_residual(it - 1)) if it > 0 \
                        else None
                except Exception:  # noqa: BLE001 — optional
                    pass
                res_s = f"{res:.3e}" if res is not None else "n/a"
                print(f"[blockamgx]     inner[{self._tag} n={len(b)}] "
                      f"call={self._calls} iters={it} resid={res_s} "
                      f"status={st}", flush=True)
            except Exception as e:  # noqa: BLE001 — telemetry must not kill
                print(f"[blockamgx]     inner[{self._tag}] telemetry "
                      f"unavailable: {e}", flush=True)
                self._log_every = 0
        return self._x.copy()

    def destroy(self):
        """Tear down the AMGX objects (reverse creation order). Lets probe
        scripts build cycles sequentially without accumulating Resources."""
        for obj in (self.slv, self.B, self.X, self.M, self.rsc, self.cfg):
            try:
                obj.destroy()
            except Exception:  # noqa: BLE001 — teardown is best-effort
                pass


def _fgmres(A, b, apply_M, tol=1e-9, atol=1e-13, restart=50, maxiter=200,
            log=None, on_cycle=None):
    """Right-preconditioned FLEXIBLE GMRES (FGMRES, Saad 1993) with restart.

    Returns (x, total_inner_iters, rel_resid). `apply_M` may be ANY
    (nonlinear / iteration-varying) approximate solve — the preconditioned
    basis vectors Z_j = M(V_j) are stored explicitly and the solution update
    is x += Z y, which is exactly what makes FGMRES correct where standard
    preconditioned GMRES silently breaks. `maxiter` counts RESTART CYCLES
    (scipy-gmres semantics, so the existing GMRES_MAXITER knob keeps its
    meaning); `log(it, rel)` is called once per inner iteration with the
    TRUE residual norm estimate (right preconditioning ⇒ the Arnoldi
    residual IS the true residual ||b - A x||, no M-norm distortion)."""
    A = A.tocsr()
    n = b.shape[0]
    x = np.zeros(n)
    b_nrm = float(np.linalg.norm(b))
    if b_nrm == 0.0:
        return x, 0, 0.0
    target = max(tol * b_nrm, atol)
    it_total = 0
    res = b_nrm
    for _outer in range(int(maxiter)):
        r = b - A @ x
        beta = float(np.linalg.norm(r))
        if on_cycle is not None:
            on_cycle(_outer, r)
        if beta <= target:
            return x, it_total, beta / b_nrm
        m = int(restart)
        V = np.empty((m + 1, n))
        Z = np.empty((m, n))
        H = np.zeros((m + 1, m))
        cs, sn = np.zeros(m), np.zeros(m)
        g = np.zeros(m + 1)
        g[0] = beta
        V[0] = r / beta
        k_used = 0
        breakdown = False
        for k in range(m):
            Z[k] = apply_M(V[k])
            w = A @ Z[k]
            # modified Gram-Schmidt + one re-orthogonalization pass (cheap
            # insurance against loss of orthogonality on hard systems)
            for j in range(k + 1):
                H[j, k] = float(w @ V[j])
                w -= H[j, k] * V[j]
            for j in range(k + 1):
                c = float(w @ V[j])
                H[j, k] += c
                w -= c * V[j]
            H[k + 1, k] = float(np.linalg.norm(w))
            if H[k + 1, k] > 1e-14 * beta:
                V[k + 1] = w / H[k + 1, k]
            else:
                breakdown = True      # lucky/unlucky breakdown: solve & exit
            # Givens rotations -> triangular H, running residual estimate
            for j in range(k):
                t = cs[j] * H[j, k] + sn[j] * H[j + 1, k]
                H[j + 1, k] = -sn[j] * H[j, k] + cs[j] * H[j + 1, k]
                H[j, k] = t
            d = float(np.hypot(H[k, k], H[k + 1, k]))
            if d == 0.0:
                breakdown = True
                k_used = k + 1
                break
            cs[k], sn[k] = H[k, k] / d, H[k + 1, k] / d
            H[k, k] = d
            H[k + 1, k] = 0.0
            g[k + 1] = -sn[k] * g[k]
            g[k] = cs[k] * g[k]
            res = abs(float(g[k + 1]))
            it_total += 1
            k_used = k + 1
            if log is not None:
                log(it_total, res / b_nrm)
            if res <= target or breakdown:
                break
        # solution update from the flexible basis: x += Z[:k] @ y
        y = np.zeros(k_used)
        for i in range(k_used - 1, -1, -1):
            y[i] = (g[i] - H[i, i + 1:k_used] @ y[i + 1:k_used]) / H[i, i]
        x = x + Z[:k_used].T @ y
        if res <= target or breakdown:
            break
    r_true = float(np.linalg.norm(b - A @ x))
    return x, it_total, r_true / b_nrm


def solve_block_preconditioned(A, b, pre: BlockAMGPreconditioner,
                               tol=1e-9, maxiter=200, restart=50):
    """Right-preconditioned FLEXIBLE GMRES on the monolithic system. Returns
    (x, iters). Raises on non-convergence.

    R2b.1 L5-cliff ROOT CAUSE + FIX: this used to call scipy's `gmres`,
    which is STANDARD (non-flexible) GMRES and requires the preconditioner
    to be a fixed linear operator. Ours is not — the inner F/Schur solves
    are truncated Krylov iterations (PBICGSTAB / PCG), a nonlinear,
    call-to-call-varying operator. On the tiny L3 saddle the inner solves
    are effectively exact, the operator is effectively constant, and GMRES
    converges; at L5 they are genuinely inexact, the Arnoldi recurrence no
    longer holds, and the outer iteration stagnates — a structural cliff
    immune to inner-solve strengthening, exactly as observed. FGMRES stores
    the preconditioned basis explicitly and is provably correct for any
    varying M.

    Instrumentation (handoff H0), on by default, silence with
    BLOCKAMGX_QUIET=1: per-iteration TRUE relative residual (throttled),
    plus a one-shot probe of a single preconditioner application
    (rel ||b - A M(b)||/||b|| — "does one apply() reduce anything?").
    BLOCKAMGX_IDENTITY=1 swaps M for the identity (handoff H1 A/B test)."""
    quiet = bool(os.environ.get("BLOCKAMGX_QUIET"))
    identity = bool(os.environ.get("BLOCKAMGX_IDENTITY"))
    A = A.tocsr()
    apply_M = (lambda v: v.copy()) if identity else pre.apply

    if not quiet and not getattr(pre, "_probe_done", False):
        pre._probe_done = True
        b_nrm = float(np.linalg.norm(b))
        z = apply_M(b)
        r1 = float(np.linalg.norm(b - A @ z)) / max(b_nrm, 1e-300)
        print(f"[blockamgx] n={A.shape[0]} probe: single-apply rel resid "
              f"||b-A·M(b)||/||b|| = {r1:.3e} "
              f"(1.0 ⇒ preconditioner does nothing; "
              f"identity={'ON' if identity else 'off'})", flush=True)

    last = [0.0]

    def _log(it, rel):
        last[0] = rel
        if quiet:
            return
        if it <= 10 or it % 10 == 0:
            print(f"[blockamgx]   fgmres it={it:4d} rel={rel:.3e}",
                  flush=True)

    b_nrm = float(np.linalg.norm(b))

    def _split(tag, r):
        # WHERE does the residual live? u-interior vs u-Dirichlet vs pressure
        # — discriminates F-block vs BC-handling vs Schur as the slow subspace.
        if quiet:
            return
        r_u = r[pre.u_ids]
        r_p = r[pre.p_ids]
        n_u = float(np.linalg.norm(r_u))
        n_p = float(np.linalg.norm(r_p))
        n_dir = (float(np.linalg.norm(r_u[pre._u_dir]))
                 if pre._u_dir is not None else 0.0)
        n_int = float(np.sqrt(max(n_u ** 2 - n_dir ** 2, 0.0)))
        print(f"[blockamgx]   {tag}: |r|/|b| split  u_int={n_int / b_nrm:.3e} "
              f"u_dir={n_dir / b_nrm:.3e} p={n_p / b_nrm:.3e}", flush=True)

    x, iters, rel = _fgmres(A, b, apply_M, tol=tol, atol=1e-13,
                            restart=restart, maxiter=maxiter, log=_log,
                            on_cycle=lambda o, r: _split(f"cycle{o}", r))
    if not quiet:
        print(f"[blockamgx] fgmres done: iters={iters} true rel={rel:.3e} "
              f"(tol={tol:.1e})", flush=True)
    if rel > tol:
        raise RuntimeError(
            f"block-preconditioned FGMRES failed: rel={rel:.3e} > tol={tol:.1e} "
            f"after {iters} iterations")
    return x, iters
