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
                 timestab=True, ppe_finescale=False, predictor="picard",
                 velocity_update="consistent", graddiv_scale=1.0,
                 pressure_update="standard", ppe_fine_scale=False):
        self.dm, self.nu, self.dt, self.order = dm, nu, dt, order
        self.picard_iters = picard_iters
        # P2-R0 velocity-update EXPERIMENT knob (default "consistent" =
        # unchanged behaviour). Controls Step 3 (the velocity update) and,
        # for "graddiv", an EXTRA grad-div penalty in the predictor:
        #   "consistent" — consistent-mass L2 re-projection (Algorithm 1).
        #   "lumped"      — row-sum (lumped) diagonal mass in Step 3, so the
        #                   update collocates to the nodal
        #                   u = u_hat - (1/sigma) grad(phi), which better
        #                   preserves the discrete pointwise divergence
        #                   relation than the consistent-mass smear.
        #   "graddiv"     — consistent-mass update PLUS a graddiv_scale-times
        #                   tau_C (div w, div u) grad-div (LSIC) penalty added
        #                   to the predictor system, which damps the predicted
        #                   (hence corrected) pointwise divergence directly.
        if velocity_update not in ("consistent", "lumped", "graddiv"):
            raise ValueError(
                f"velocity_update must be consistent|lumped|graddiv, "
                f"got {velocity_update!r}")
        self.velocity_update = velocity_update
        self.graddiv_scale = float(graddiv_scale)
        # P2-R2a pressure-treatment enum (default "standard" = unchanged classic
        # incremental p_hat = p* + phi). Folds Taly's pressure_extrap_c into one
        # enum (see the plan's interaction table): standard/rotational are
        # pressure-extrap order 1 (p*=p^n, BDF2-compatible); chorin is order 0
        # (p*=0, 1st-order). We never use 2nd-order pressure extrapolation.
        #   "standard"   — classic incremental: p_hat = p* + phi (Algorithm 1).
        #   "rotational" — Timmermans consistent-incremental:
        #                    p_hat = p* + phi - nu * q,  M_p q = B^T u_hat.
        #   "chorin"     — non-incremental confirmation: p* reset to 0 each step
        #                  (no accumulation); p_hat = phi. 1st-order in time.
        if pressure_update not in ("standard", "rotational", "chorin"):
            raise ValueError(
                f"pressure_update must be standard|rotational|chorin, "
                f"got {pressure_update!r}")
        self.pressure_update = pressure_update
        # P2-R2a VMS fine-scale-consistency lever (default False = unchanged).
        # NOTE: spelled with an underscore to distinguish from the PRE-EXISTING
        # self.ppe_finescale flag (leray.py, dt=Δt/b0 τ_m bug, audit §5) which
        # R2a leaves untouched. When True: the fine-scale velocity -tau_M r_m
        # enters BOTH the PPE source (sigma (grad q, -tau_M r_m)) AND the
        # velocity update (u = u_hat - tau_M r_m - (1/sigma)(grad p_hat -
        # grad p*)) — matching Taly Proj_Linear_PPE/VUE_Integrands. tau_M here
        # is computed with the CORRECT dt (NOT dt/b0).
        self.ppe_fine_scale = bool(ppe_fine_scale)
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
        # Step-3 update operator: consistent M (default), or the row-sum
        # (lumped) diagonal of M — a diagonal solve that nodally collocates
        # u = u_hat - (1/sigma) grad(phi) (P2-R0 velocity-update experiment).
        if self.velocity_update == "lumped":
            row_sum = np.asarray(self.M.sum(axis=1)).ravel()
            self.M_lumped = sp.diags(row_sum, format="csr")
        else:
            self.M_lumped = None
        # Extra grad-div (LSIC) block for the "graddiv" variant — a scaled
        # vector-Laplacian-of-divergence penalty tau_C (div w, div u) added to
        # the predictor momentum system. Zero unless graddiv_scale > 0 and the
        # variant is selected; assembled once (tau_C frozen at the steady,
        # velocity-independent value, |u|-part dropped for a cached operator).
        self._graddiv_block = None
        if self.velocity_update == "graddiv" and self.graddiv_scale != 0.0:
            self._graddiv_block = self._graddiv_matrix()

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

    def _graddiv_matrix(self):
        """Extra grad-div (LSIC) penalty block for the predictor (P2-R0
        velocity-update experiment, "graddiv"). Assembles the CONSTRAINED,
        free-node-major, VECTOR (ndof=dim+1) operator

            G[a i, b j] = scale * tau_C * int (dN_a/dx_i)(dN_b/dx_j) dV

        added to the momentum block so the predictor damps ||div u_hat||
        pointwise. tau_C is the metric-form grad-div parameter frozen at the
        steady limit (|u|-independent so the block caches; the transient
        (2 b0/dt)^2 term is dropped from tau_M here). The pressure rows/cols
        (component ``dim``) are left zero, so this is purely a momentum-side
        stabilization. Returns a CSR of shape (n_free*ndof, n_free*ndof)."""
        dm = self.dm
        dim = dm.dim
        ndof = self.ndof
        # steady tau_M = 1/sqrt(CI_F nu^2 G:G), tau_C = 1/(tau_M g.g), with
        # G:G = dim (2/h)^4, g.g = dim (2/h)^2 on axis-aligned cubes (vms.py).
        from ..physics.vms import CI_F
        rows, cols, vals = [], [], []
        for pv, b in dm.bins.items():
            tb = dm.tables_by_p[pv]
            h = dm.mesh.tree.h()[dm.mesh.bins[pv]]
            jac = (h / 2.0) ** dim
            dsc = (2.0 / h)
            ne = len(h)
            GG = dim * (2.0 / h) ** 4
            gg = dim * (2.0 / h) ** 2
            tauM = 1.0 / np.sqrt(CI_F * self.nu ** 2 * GG)
            tauC = 1.0 / (tauM * gg)                      # [ne]
            coef = self.graddiv_scale * tauC              # [ne]
            # per-element grad-div: Ge[e, a, i, b, j]
            #   = coef[e] * (dN_a/dx_i)(dN_b/dx_j) * w * jac
            # dN scaled to physical by dsc; note (dsc*dsc) folded into GdG.
            GdG = np.einsum("qad,qbc,q->abdc", tb.dN, tb.dN, tb.w)  # ref
            conn = dm.mesh.conn_of[pv].astype(np.int64)
            nbf = conn.shape[1]
            scale_e = coef * (dsc ** 2) * jac             # [ne]
            for a in range(nbf):
                for bcol in range(nbf):
                    for i in range(dim):
                        for j in range(dim):
                            r = conn[:, a] * ndof + i
                            c = conn[:, bcol] * ndof + j
                            v = GdG[a, bcol, i, j] * scale_e
                            rows.append(r)
                            cols.append(c)
                            vals.append(v)
        Nn = dm.n_nodes * ndof
        G = sp.coo_matrix((np.concatenate(vals),
                           (np.concatenate(rows), np.concatenate(cols))),
                          shape=(Nn, Nn)).tocsr()
        T = dm.constraints.T.tocsr()
        T_vec = sp.kron(T, sp.identity(ndof, format="csr"), format="csr")
        return (T_vec.T @ G @ T_vec).tocsr()

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

    def _weak_div_free(self, u_free):
        """Assemble the free-node weak divergence B^T u_hat (scalar, n_free)
        from a free-node velocity field u_free [n_free, dim].

        Used by the rotational pressure update when ppe_fine_scale=True (the
        rhs_free already carries the fine-scale source and cannot be reused as
        sigma * B^T u_hat). Assembles int grad(N_a) . u dV in the scalar free-
        node space, then pins free-node 0 to zero (same convention as the PPE).
        """
        dm = self.dm
        dim = dm.dim
        uq = self._gp_vals(u_free)
        rhs = np.zeros(dm.n_nodes)
        for pv, b_ in dm.bins.items():
            tb = dm.tables_by_p[pv]
            h = dm.mesh.tree.h()[dm.mesh.bins[pv]]
            nqp = tb.nqp
            ne = len(h)
            jac = (h / 2.0) ** dim
            dsc = (2.0 / h)
            fl = uq[pv].reshape(ne, nqp, dim)
            be = np.einsum("qad,eqd,q,e->ea", tb.dN, fl, tb.w, jac * dsc)
            np.add.at(rhs, dm.mesh.conn_of[pv].ravel(), be.ravel())
        bt_free = np.asarray(dm.constraints.T.T @ rhs)
        bt_free[0] = 0.0  # pin free-node 0 (same convention as PPE)
        return bt_free

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
            # P2-R0 "graddiv" variant: add the cached extra grad-div (LSIC)
            # penalty to the momentum system (RHS unchanged — homogeneous
            # penalty). No-op for the other variants (block is None).
            if self._graddiv_block is not None:
                A = (A + self._graddiv_block)
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
    def step(self, extra_block=None, sbm_nodes=None, ppe_surrogate_flux=None):
        """One projection step.

        ``ppe_surrogate_flux`` is the SURROGATE-CONSISTENT PPE boundary hook
        (P2-R0 Task 3). The surrogate-consistent boundary condition on the
        pressure-Poisson increment ``phi`` at the immersed body is a
        HOMOGENEOUS Neumann condition ``grad(phi).n_hat = 0`` (Suresh
        pressure-projection SBM paper, Eq. 5 + Remark 3.9): this is exactly
        the NATURAL boundary condition of the divergence-form PPE RHS
        ``(sigma u_hat, grad q)`` on the surrogate faces (they carry no strong
        constraint and are not pinned), so the DEFAULT ``None`` already
        imposes it and PROVES the blockage/no-penetration of the SBM
        predictor is preserved by the projection: since
        ``u = u_hat - (1/sigma) grad(phi)`` and ``grad(phi).n_hat = 0`` at the
        surrogate, ``u.n_hat = u_hat.n_hat`` there (Remark 3.9). Choosing a
        non-zero surrogate flux (or a ``phi``-Dirichlet pin) instead lets the
        correction push mass through the body and is REJECTED by the paper;
        the hook exists so that a wrong/omitted BC can be injected as a
        planted-break to prove the homogeneous-Neumann choice is
        load-bearing. When callable, ``ppe_surrogate_flux(uhat)`` returns a
        FULL node-major (``dm.n_nodes``) scalar added to the PPE RHS before
        the constraint reduction.
        """
        dm = self.dm
        dim = dm.dim
        ndof = self.ndof
        t_new = self.t + self.dt
        # P2-R2a: Chorin (non-incremental) confirmation mode — zero the
        # accumulated pressure each step so the predictor never sees a
        # compounding grad p* and p_hat = phi (pressure-extrap order 0).
        if self.pressure_update == "chorin":
            self.p_star = np.zeros(self.n_free)
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
        fs_vel = {}   # P2-R2a: cached -tau_M r_m per bin for velocity update
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
            if self.ppe_fine_scale:
                # NEW VMS-consistent PPE source (Taly Proj_Linear_PPE_Integrands
                # line ~294): flux = sigma*(u_hat - tau_M R). tau_M uses the
                # CORRECT dt (self.dt), NOT self.dt/b0 (the old ppe_finescale
                # bug). R is the coarse momentum residual already assembled here
                #   R = sigma*u_hat + a.grad u_hat + grad(p*) - f
                # where the pressure term is grad(p_star) — the LAGGED pressure
                # p^n (Taly NL integrands use vpre1.gradp, NOT p^{n+1}, NOT the
                # extrapolated p*). In our INCREMENTAL setting p_star IS the
                # lagged pressure the PPE re-solves against, so pq_g (=grad p*)
                # is exactly Taly's lagged-pressure gradient — REUSE it, do NOT
                # recompute a second, subtly-different residual.
                taum_fs = tau_hbased_host(umag, he, self.nu,
                                          dt=(self.dt if self.timestab else None),
                                          dim=dim)
                r_m = (sigma * aqv + agu + pq_g[pv].reshape(-1, dim)
                       - fq_base[pv])
                fs_vel[pv] = taum_fs[:, None] * r_m  # -tau_M r_m (stashed)
                flux = sigma * (aqv - fs_vel[pv])
            elif self.ppe_finescale:      # PRE-EXISTING flag, untouched
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
        # Surrogate-consistent PPE boundary hook (Task 3): default None keeps
        # the homogeneous-Neumann natural BC at the surrogate (Suresh Eq. 5 /
        # Remark 3.9). A non-None flux is the paper-rejected non-homogeneous
        # choice, used only as a planted-break to prove the BC is load-bearing.
        if ppe_surrogate_flux is not None:
            rhs = rhs + np.asarray(ppe_surrogate_flux(uhat))
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
        # ---- pressure update (P2-R2a pressure_update enum) ----
        if self.pressure_update == "chorin":
            p_hat = phi.copy()             # p* zeroed at top of step(); no accum.
        elif self.pressure_update == "rotational":
            # Timmermans: p_hat = p* + phi - nu * q, M_p q = B^T u_hat.
            if self.ppe_fine_scale:
                # rhs_free carries the fine-scale term, so recompute B^T u_hat
                # directly from u_hat (weak divergence, pinned at free-node 0).
                bt_uhat = self._weak_div_free(uhat)
            else:
                bt_uhat = rhs_free / sigma     # rhs_free = sigma * B^T u_hat
            from ..solvers.linsolve import solve_linear
            q = solve_linear(self.M, bt_uhat, solver=self.solver, sym=True,
                             device=self.dm.device, cache=self._solver_cache,
                             cache_key="mass")   # SAME key as velocity update
            p_hat = self.p_star + phi - self.nu * q
        else:                                    # "standard" (default)
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
                if self.ppe_fine_scale:
                    # -tau_M r_m fine-scale velocity (Taly VUE line ~230);
                    # fs_vel[pv] = tau_M r_m cached from the PPE loop — the
                    # SAME fine-scale velocity as the PPE source, ensuring
                    # VMS consistency between the PPE source and velocity update.
                    integ = integ - fs_vel[pv].reshape(ne, nqp, dim)[:, :, c]
                be = np.einsum("qa,eq,q,e->ea", tb.N, integ, tb.w, jac)
                np.add.at(rhs_c, dm.mesh.conn_of[pv].ravel(), be.ravel())
            from ..solvers.linsolve import solve_linear
            # P2-R0 velocity-update experiment: "lumped" uses the row-sum
            # diagonal mass (a nodal-collocation update); default consistent M.
            if self.velocity_update == "lumped":
                rhs_free = np.asarray(dm.constraints.T.T @ rhs_c)
                u_new[:, c] = rhs_free / self.M_lumped.diagonal()
            else:
                u_new[:, c] = solve_linear(
                    self.M, np.asarray(dm.constraints.T.T @ rhs_c),
                    solver=self.solver, sym=True, device=dm.device,
                    cache=self._solver_cache, cache_key="mass")
        # strong Dirichlet on the updated field (draft: trace preserved).
        # SURROGATE-CONSISTENT CORRECTION (Task 3): SBM-governed nodes (the
        # weak immersed body) are NOT strong-overwritten by the box trace —
        # their corrected velocity IS the L2 projection u = u_hat -
        # (1/sigma) grad(phi) (Suresh Eq. 6). With homogeneous Neumann on phi
        # at the surrogate (the default PPE BC above), grad(phi).n_hat = 0
        # there, so the projection preserves the SBM predictor's shifted
        # no-penetration u.n_hat ~ 0 (Remark 3.9) instead of stamping the box
        # inflow onto the body (which would leak flow through it). Pass
        # sbm_nodes to skip the box overwrite on exactly those nodes.
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
