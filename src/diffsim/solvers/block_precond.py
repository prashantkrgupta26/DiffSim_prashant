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
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator, gmres


class BlockAMGPreconditioner:
    def __init__(self, A, n_nodes, ndof, Kp, Mp_diag, sigma, nu,
                 dir_rows=None):
        """A: assembled monolithic CSR (interleaved node-major DOFs).
        n_nodes: FREE nodes; ndof = dim+1. Kp: pressure stiffness on the
        same free nodes (with its own pinned row handled by caller);
        Mp_diag: pressure mass diagonal. dir_rows: strong-Dirichlet row ids
        of the monolithic system (identity rows — kept in F)."""
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
        self._amg_F = _AMGXCycle(self.F, sym=False, cycles=1)
        self._amg_Kp = _AMGXCycle(self.Kp, sym=True, cycles=3)

    def apply(self, r):
        r = np.asarray(r)
        r_u, r_p = r[self.u_ids], r[self.p_ids]
        # Schur: Cahouet-Chabard
        z_p = (self.sigma * self._amg_Kp.solve(r_p, tol=1e-3, iters=8)
               + self.nu * (r_p / self.Mp_diag))
        # velocity: one AMG cycle on the corrected residual
        r_u_corr = r_u - self.G @ z_p
        z_u = self._amg_F.solve(r_u_corr, tol=1e-2, iters=2)
        if self._u_dir is not None:
            # identity action on strong-Dirichlet rows (F rows are identity)
            z_u[self._u_dir] = r_u_corr[self._u_dir]
        z = np.empty_like(r)
        z[self.u_ids], z[self.p_ids] = z_u, z_p
        return z

    def as_linear_operator(self):
        n = self.n * self.ndof
        return LinearOperator((n, n), matvec=self.apply)


class _AMGXCycle:
    """A persistent AMGX solver used as a PRECONDITIONER: setup once per
    matrix, apply a HARD-CAPPED number of cycles per call. CARE POINT
    (measured): reusing a solver-grade config (max_iters=2000, tol 1e-4)
    turns every preconditioner application into a near-full solve — the
    first probe ground for 19+ minutes against cuDSS's 147 ms. A
    preconditioner must be a fixed, cheap operator."""

    def __init__(self, A, sym, cycles=1):
        from .amgx import _ensure_init
        _ensure_init()
        import json
        import os
        import pyamgx
        here = os.path.join(os.path.dirname(__file__), "amgx_configs")
        fname = ("PCG_CLASSICAL_V_JACOBI.json" if sym
                 else "PBICGSTAB_CLASSICAL_JACOBI.json")
        with open(os.path.join(here, fname)) as fh:
            cfg = json.load(fh)
        cfg["solver"]["max_iters"] = int(cycles)
        cfg["solver"]["tolerance"] = 0.0        # never early-exit
        cfg["solver"]["monitor_residual"] = 1
        cfg["solver"]["print_solve_stats"] = 0
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

    def solve(self, b, **_ignored):
        self.B.upload(np.ascontiguousarray(b, np.float64))
        self._x[:] = 0.0
        self.X.upload(self._x)
        self.slv.solve(self.B, self.X, zero_initial_guess=True)
        self.X.download(self._x)
        return self._x.copy()


def solve_block_preconditioned(A, b, pre: BlockAMGPreconditioner,
                               tol=1e-9, maxiter=200):
    """Right-preconditioned GMRES on the monolithic system. Returns
    (x, iters). Raises on non-convergence."""
    it_count = [0]

    def _cb(_):
        it_count[0] += 1
    x, info = gmres(A.tocsr(), b, M=pre.as_linear_operator(),
                    rtol=tol, atol=1e-13, maxiter=maxiter, restart=50,
                    callback=_cb, callback_type="pr_norm")
    if info != 0:
        raise RuntimeError(f"block-preconditioned GMRES failed: info={info} "
                           f"after {it_count[0]} iterations")
    return x, it_count[0]
