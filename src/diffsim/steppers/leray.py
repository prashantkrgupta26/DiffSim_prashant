"""Leray pressure-projection stepper (M1b Task 6) — Algorithm 1 of the
Helmholtz-Leray VMS draft (THE key document, user directive; conventions doc
'NS pressure-projection' section). Scope: this stepper only.

Per step tn -> tn+1 (p* = extrapolated pressure, order 1: p* = p_hat^n):

1. MOMENTUM PREDICTOR — NONLINEAR: the draft solves Newton with the
   fine-scale closure recomputed at every iterate (no lagging). v1 delta
   (documented): PICARD iterations (default 2) — each pass re-assembles the
   monolithic block at a = u-current-iterate with the PRESSURE DOFS PINNED
   to p*'s nodal values, so momentum rows see grad p* through the coupling
   and tau/fine-scale terms are recomputed consistently per iterate.
   Newton's quadratic tail matters for stiff steps, not for the order gate;
   upgrade tracked in m1b findings.
2. PPE (SPD, tau_m fine-scale RHS, NO tauC — draft Remark 2.3):
       (grad p_hat, grad q) = sigma (u_hat - tau_m r_m(u_hat, p*), grad q)
   r_m the explicit strong momentum residual at (u_hat, p*); h-based tau_m
   (Eq. 45). Enclosed flow: p_hat pinned (the draft's q = 0 on Gamma_D^p has
   no boundary here).
3. VELOCITY UPDATE (fine scale retained): consistent-mass L2 projection
       u^{n+1} = u_hat - (1/sigma)(grad p_hat - grad p*)
4. p* <- p_hat (first-order/incremental — the draft's choice).
"""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu

from ..api.ns_bricks import assemble_linear_ns
from ..assembly.operators import assemble_csr
from ..physics.poisson import gauss_points
from ..physics.vms import tau_hbased_host
from ..solvers.timestepping import bdf_coeffs, bdf_order_now, History


class LerayProjectionStepper:
    def __init__(self, dm, nu, dt, f_fn, g_fn, order=2, picard_iters=2,
                 solver="splu",
                 timestab=True, ppe_finescale=False, predictor="picard"):
        self.dm, self.nu, self.dt, self.order = dm, nu, dt, order
        self.picard_iters = picard_iters
        # 'picard' (v1) or 'newton' (the draft's Algorithm 1): Newton adds
        # the (du.grad)a cross-block in the momentum operator and folds the
        # (a.grad)a RHS partner into f_eff (which hands it SUPG/PSPG
        # consistency for free). SUPG linearization stays Picard-level —
        # the standard, documented delta from exact Newton.
        self.predictor = predictor
        self.timestab = timestab
        # v1 STABILITY FINDING (measured blowup ~7e5 on the vortex MMS):
        # sigma*tau_m ~ 0.8 at gate dt's, so treating tau_m r_m EXPLICITLY
        # in the PPE RHS (r_m contains sigma*u_hat and grad p*) is far
        # outside stability. The draft keeps the tau_m grad(p_hat) piece
        # IMPLICIT (a (1+sigma tau_m)-weighted PPE operator). v1 defaults to
        # the classic incremental projection (flux = sigma u_hat); the
        # fine-scale PPE term stays behind this flag until the implicit
        # weighting lands (m1b findings TODO, benchmark task).
        self.ppe_finescale = ppe_finescale
        # linear-solve backend for ALL three sub-solves (predictor: nonsym;
        # PPE + mass updates: SPD): "splu" | "fused" | "amgx"
        self.solver = solver
        self._solver_cache = {}
        self.f_fn, self.g_fn = f_fn, g_fn
        self.ndof = dm.dim + 1
        self.hist = History()          # velocity-only history [n_free*dim]
        self.t = 0.0
        mesh = dm.mesh
        self.free_coords = mesh.node_coords[dm.constraints.free_nodes]
        self.n_free = len(self.free_coords)
        bdry = mesh.boundary_nodes[dm.constraints.free_nodes]
        self.dir_nodes = np.where(bdry)[0]
        self.p_star = np.zeros(self.n_free)
        self.xq = gauss_points(mesh, dm.tables_by_p)
        # scalar stiffness (PPE) + scalar consistent mass (update solves)
        self.K_p = assemble_csr(dm)
        self._K_p_lu = None
        self.M = self._mass_matrix()
        self._M_lu = None            # mass solves go through solve_linear

    # ---------------- helpers ----------------
    def _mass_matrix(self):
        dm = self.dm
        rows, cols, vals = [], [], []
        for pv, b in dm.bins.items():
            tb = dm.tables_by_p[pv]
            h = dm.mesh.tree.h()[dm.mesh.bins[pv]]
            jac = (h / 2.0) ** dm.dim
            Me = np.einsum("qa,qb,q->ab", tb.N, tb.N, tb.w)
            conn = dm.mesh.conn_of[pv].astype(np.int64)
            ne, nbf = conn.shape
            Mee = Me[None, :, :] * jac[:, None, None]
            rows.append(np.repeat(conn, nbf, axis=1).ravel())
            cols.append(np.tile(conn, (1, nbf)).ravel())
            vals.append(Mee.ravel())
        Nn = dm.n_nodes
        M = sp.coo_matrix((np.concatenate(vals),
                           (np.concatenate(rows), np.concatenate(cols))),
                          shape=(Nn, Nn)).tocsr()
        T = dm.constraints.T.tocsr()
        return (T.T @ M @ T).tocsr()

    def _weighted_stiffness(self, w_gp_by_bin):
        """K_w[a,b] = int w(x) grad N_a . grad N_b — per-GP weights (the
        implicit fine-scale PPE operator, weight 1/sigma + tau_m)."""
        dm = self.dm
        rows, cols, vals = [], [], []
        for pv, b in dm.bins.items():
            tb = dm.tables_by_p[pv]
            h = dm.mesh.tree.h()[dm.mesh.bins[pv]]
            jac = (h / 2.0) ** dm.dim
            dsc2 = (2.0 / h) ** 2
            ne = len(h)
            wq = w_gp_by_bin[pv].reshape(ne, tb.nqp)
            Ke = np.einsum("qad,qbd,q,eq,e->eab", tb.dN, tb.dN, tb.w, wq,
                           jac * dsc2)
            conn = dm.mesh.conn_of[pv].astype(np.int64)
            nbf = conn.shape[1]
            rows.append(np.repeat(conn, nbf, axis=1).ravel())
            cols.append(np.tile(conn, (1, nbf)).ravel())
            vals.append(Ke.ravel())
        Nn = dm.n_nodes
        K = sp.coo_matrix((np.concatenate(vals),
                           (np.concatenate(rows), np.concatenate(cols))),
                          shape=(Nn, Nn)).tocsr()
        T = dm.constraints.T.tocsr()
        return (T.T @ K @ T).tocsr()

    def _gp_vals(self, node_scalar_or_vec, grad=False):
        """Node field (n_free[, k]) -> per-bin GP values (and gradients)."""
        dm = self.dm
        full = np.asarray(dm.constraints.T @ node_scalar_or_vec)
        out, gout = {}, {}
        for pv, b in dm.bins.items():
            tb = dm.tables_by_p[pv]
            conn = dm.mesh.conn_of[pv]
            vals = full[conn]
            out[pv] = np.einsum("qa,ea...->eq...", tb.N, vals).reshape(
                (-1,) + full.shape[1:])
            if grad:
                h = dm.mesh.tree.h()[dm.mesh.bins[pv]]
                g = np.einsum("qad,ea...->eqd...", tb.dN, vals)
                if full.ndim > 1:
                    # star-in-subscript is 3.11+ (PEP 646); build the index
                    # tuple explicitly so requires-python >=3.10 holds
                    idx = (slice(None), None, None) + (None,) * (full.ndim - 1)
                    g = g * (2.0 / h)[idx]
                else:
                    g = g * (2.0 / h)[:, None, None]
                gout[pv] = g.reshape((-1, dm.dim) + full.shape[1:])
        return (out, gout) if grad else out

    def set_initial(self, u0_fn):
        self.hist.rotate(u0_fn(self.free_coords).ravel())

    def _uvec(self, flat):
        return flat.reshape(self.n_free, self.dm.dim)

    def _predictor_setup(self, t_new):
        """Common BDF/history setup for the predictor; returns
        (b0, b1, b2, sigma, u1, u2, fq_base, gvals)."""
        dm = self.dm
        o = bdf_order_now(t_new, self.dt, self.order,
                          have_history=self.hist.have(2))
        b0, b1, b2 = bdf_coeffs(o, self.dt)
        sigma = b0 / self.dt
        u1 = self._uvec(self.hist.pre1)
        u2 = self._uvec(self.hist.pre2) if self.hist.have(2) else None
        h_node = (b1 * u1 + (b2 * u2 if (b2 != 0.0 and u2 is not None)
                             else 0.0)) / self.dt
        hq = self._gp_vals(h_node)
        fq_base = {pv: self.f_fn(self.xq[pv], t_new) - hq[pv]
                   for pv in self.xq}
        gvals = self.g_fn(self.free_coords[self.dir_nodes], t_new)
        return b0, b1, b2, sigma, u1, u2, fq_base, gvals

    def _predict(self, t_new=None, extra_block=None, sbm_nodes=None,
                 return_matrix=False):
        """Momentum predictor (Algorithm 1 Step 1) as a standalone hook.

        The composed volumetric-SBM stepper injects the shifted-Nitsche
        vector Dirichlet block via ``extra_block=(A_sbm_c, b_sbm_c)`` (both
        CONSTRAINED, free-node-major ``n_free*ndof``): the block is added to
        the assembled momentum system before the strong-row overwrite, and
        the ``sbm_nodes`` (free-node indices governed WEAKLY by SBM) skip the
        strong box-Dirichlet overwrite so the immersed body stays weak.

        Keeps the body-fitted path bit-identical when ``extra_block is None``
        and ``sbm_nodes is None``.

        Returns ``uhat`` (or the assembled csr matrix when
        ``return_matrix`` is True — for composition tests). Does NOT advance
        the history or ``p_star``.
        """
        dm = self.dm
        dim = dm.dim
        ndof = self.ndof
        if t_new is None:
            t_new = self.t + self.dt
        b0, b1, b2, sigma, u1, u2, fq_base, gvals = \
            self._predictor_setup(t_new)
        p_node_full = self.p_star
        strong_skip = (set() if sbm_nodes is None
                       else set(int(i) for i in np.asarray(sbm_nodes)))

        a_node = 2.0 * u1 - u2 if u2 is not None else u1.copy()
        uhat = None
        A_out = None
        self.predictor_diffs = []
        prev_iter = None
        for _ in range(self.picard_iters):
            aq_d = self._gp_vals(a_node, grad=True)
            aq, gaq = aq_d
            gaq_flat = {pv: gaq[pv].reshape(-1, dim, dim).transpose(0, 2, 1)
                        for pv in aq}     # [g, i, j] = da_i/dx_j
            dq = {pv: np.einsum("gdd->g", gaq[pv].reshape(-1, dim, dim))
                  for pv in aq}
            newton = self.predictor == "newton"
            if newton:
                conv_a = {pv: np.einsum("gj,gij->gi", aq[pv],
                                        gaq_flat[pv]) for pv in aq}
                fq_it = {pv: fq_base[pv] + conv_a[pv] for pv in aq}
            else:
                fq_it = fq_base
            A, b = assemble_linear_ns(
                dm, aq, dq, fq_it, self.nu, sigma=sigma,
                sig2tau=((2.0 * sigma) ** 2 if self.timestab else 0.0),
                gaq_by_bin=(gaq_flat if newton else None), newton=newton)
            # SBM face block: add the constrained shifted-Nitsche vector
            # Dirichlet into the momentum system BEFORE strong-row overwrite.
            if extra_block is not None:
                A_sbm_c, b_sbm_c = extra_block
                A = (A + A_sbm_c)
                b = b + np.asarray(b_sbm_c)
            A = A.tolil()
            for k, i in enumerate(self.dir_nodes):
                if int(i) in strong_skip:      # SBM-governed: stays weak
                    continue
                for c in range(dim):
                    r = i * ndof + c
                    A.rows[r] = [int(r)]
                    A.data[r] = [1.0]
                    b[r] = gvals[k, c]
            for i in range(self.n_free):           # pin ALL pressure DOFs
                r = i * ndof + dim
                A.rows[r] = [int(r)]
                A.data[r] = [1.0]
                b[r] = p_node_full[i]
            Acsr = A.tocsr()
            from ..solvers.linsolve import solve_linear
            x = solve_linear(Acsr, b, solver=self.solver, sym=False,
                             device=self.dm.device,
                             cache=self._solver_cache)
            uhat = x.reshape(self.n_free, ndof)[:, :dim]
            A_out = Acsr
            if prev_iter is not None:
                self.predictor_diffs.append(
                    float(np.abs(uhat - prev_iter).max()))
            prev_iter = uhat
            a_node = uhat
        return A_out if return_matrix else uhat

    # ---------------- the step ----------------
    def step(self, extra_block=None, sbm_nodes=None):
        dm = self.dm
        dim = dm.dim
        ndof = self.ndof
        t_new = self.t + self.dt
        b0, b1, b2, sigma, u1, u2, fq_base, gvals = \
            self._predictor_setup(t_new)

        # ---- Step 1: nonlinear predictor (Picard over the full block with
        # pressure DOFS PINNED to p*) ----
        uhat = self._predict(t_new=t_new, extra_block=extra_block,
                             sbm_nodes=sbm_nodes)
        # ---- Step 2: PPE with tau_m fine-scale RHS ----
        uq, guq = self._gp_vals(uhat, grad=True)
        pq_g = self._gp_vals(self.p_star, grad=True)[1]
        w_gp = {}
        rhs = np.zeros(dm.n_nodes)
        for pv, b_ in dm.bins.items():
            tb = dm.tables_by_p[pv]
            h = dm.mesh.tree.h()[dm.mesh.bins[pv]]
            nqp = tb.nqp
            he = np.repeat(h, nqp)
            aqv = uq[pv]
            umag = np.sqrt((aqv ** 2).sum(1))
            taum = tau_hbased_host(umag, he, self.nu,
                                   dt=(self.dt / b0 if self.timestab
                                       else None), dim=dim)
            # explicit strong residual r_m = sigma(u_hat) + a.grad u_hat
            #   + grad p* - f  (BDF history already inside fq_base's -h term)
            agu = np.einsum("gd,gdc->gc",
                            aqv, guq[pv].reshape(-1, dim, dim))
            if self.ppe_finescale:
                # IMPLICIT fine scale (draft-faithful; findings 5b): the
                # phi-part of r_m moves to the LHS -> weight (1/sigma+tau_m)
                # on the stiffness; RHS flux = u_hat - tau_m r_m_expl
                # (coefficients bounded: (1 - sigma tau_m) u_hat - ...).
                r_m = (sigma * aqv + agu + pq_g[pv].reshape(-1, dim)
                       - fq_base[pv])
                flux = aqv - taum[:, None] * r_m
                w_gp[pv] = 1.0 / sigma + taum
            else:
                flux = sigma * aqv
            # b_q = int grad(N_q) . flux
            conn = dm.mesh.conn_of[pv]
            jac = (h / 2.0) ** dim
            w = tb.w
            ne = len(h)
            fl = flux.reshape(ne, nqp, dim)
            dsc = (2.0 / h)
            be = np.einsum("qad,eqd,q,e->ea", tb.dN, fl, w,
                           jac * dsc)
            np.add.at(rhs, conn.ravel(), be.ravel())
        rhs_free = np.asarray(dm.constraints.T.T @ rhs)
        if self.ppe_finescale:
            Kp = self._weighted_stiffness(w_gp).tolil()   # per-step tau_m
            Kp.rows[0] = [0]
            Kp.data[0] = [1.0]
            rhs_free[0] = 0.0
            from ..solvers.linsolve import solve_linear
            phi = solve_linear(Kp.tocsr(), rhs_free, solver=self.solver,
                               sym=True, device=self.dm.device,
                               cache=self._solver_cache)
        else:
            Kp = self.K_p.tolil()
            Kp.rows[0] = [0]
            Kp.data[0] = [1.0]
            rhs_free[0] = 0.0                      # pin (enclosed flow)
            from ..solvers.linsolve import solve_linear
            phi = solve_linear(Kp.tocsr(), rhs_free, solver=self.solver,
                               sym=True, device=self.dm.device,
                               cache=self._solver_cache, cache_key="ppe")
        # The PPE unknown is the pressure INCREMENT phi = p_hat - p*: the
        # RHS carries only (u_hat, grad q)-type terms, so the solution IS
        # the increment. (Treating it as the total pressure double-counts
        # p* every step — measured compounding blowup ~7e5.)
        p_hat = self.p_star + phi
        # ---- Step 3: velocity update u = u_hat - (1/sigma) grad(phi) ----
        dphi_g = self._gp_vals(phi, grad=True)[1]
        u_new = np.empty_like(uhat)
        for c in range(dim):
            rhs_c = np.zeros(dm.n_nodes)
            for pv, b_ in dm.bins.items():
                tb = dm.tables_by_p[pv]
                h = dm.mesh.tree.h()[dm.mesh.bins[pv]]
                nqp = tb.nqp
                ne = len(h)
                jac = (h / 2.0) ** dim
                integ = (uq[pv].reshape(ne, nqp, dim)[:, :, c]
                         - dphi_g[pv].reshape(ne, nqp, dim)[:, :, c] / sigma)
                be = np.einsum("qa,eq,q,e->ea", tb.N, integ, tb.w, jac)
                np.add.at(rhs_c, dm.mesh.conn_of[pv].ravel(), be.ravel())
            from ..solvers.linsolve import solve_linear
            u_new[:, c] = solve_linear(
                self.M, np.asarray(dm.constraints.T.T @ rhs_c),
                solver=self.solver, sym=True, device=dm.device,
                cache=self._solver_cache, cache_key="mass")
        # strong Dirichlet on the updated field (draft: trace preserved).
        # SBM-governed nodes (weak immersed body) are NOT strong-overwritten —
        # their trace comes from the projection (Task 3 surrogate-consistent
        # correction refines this); pass sbm_nodes to skip the overwrite.
        if sbm_nodes is None:
            u_new[self.dir_nodes] = gvals
        else:
            skip = set(int(i) for i in np.asarray(sbm_nodes))
            keep = [k for k, i in enumerate(self.dir_nodes)
                    if int(i) not in skip]
            if keep:
                keep = np.asarray(keep)
                u_new[self.dir_nodes[keep]] = gvals[keep]
        # ---- Step 4 ----
        self.p_star = p_hat
        self.hist.rotate(u_new.ravel(), dt=self.dt)
        self.t = t_new
        return u_new, p_hat

    def divergence_l2(self):
        dm = self.dm
        _, g = self._gp_vals(self._uvec(self.hist.pre1), grad=True)
        tot, vol = 0.0, 0.0
        for pv in g:
            tb = dm.tables_by_p[pv]
            h = dm.mesh.tree.h()[dm.mesh.bins[pv]]
            jac = (h / 2.0) ** dm.dim
            dv = np.einsum("gdd->g", g[pv].reshape(-1, dm.dim, dm.dim))
            w = np.tile(tb.w, len(h)) * np.repeat(jac, tb.nqp)
            tot += (dv ** 2 * w).sum()
            vol += w.sum()
        return np.sqrt(tot / vol)
