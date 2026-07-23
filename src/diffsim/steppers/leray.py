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
                 pressure_update="standard", ppe_fine_scale=False,
                 pressure_outflow_nodes=None,
                 inner_iterate=False, inner_max=8, inner_tol=1e-6,
                 inner_relax=1.0, inner_accel="none", inner_anderson_m=3,
                 consistent_ppe=False, consistent_projection=False,
                 backflow_beta=None):
        # ---- CONSISTENT-PROJECTION MODE (the 2026-07-23 fix, changes #1-#4) ----
        # ONE mode that turns on the coherent VMS-stabilized Helmholtz-Leray set
        # (ns_projection_vms_paper Eq 44a-c / Algorithm 1, exact discrete forms):
        #   #1 PSPG-consistent PPE operator — the elementwise -sigma(tau_m r_m,
        #      grad q) term makes the split's discrete incompressibility identical
        #      to the monolithic PSPG continuity block (Eq 47-49), so the
        #      monolithic steady state is a FIXED POINT of the split.
        #   #2 fine-scale u' = -tau_m r_m in BOTH the PPE source AND the L2
        #      correction, with the CONSISTENT mass (D = G^T) — Eq 44b/44c.
        #   BOTH #1 and #2 are exactly the PRE-EXISTING ``ppe_fine_scale=True``
        #      path (flux = sigma(u_hat + u'), correction subtracts the SAME u'),
        #      so the mode turns that flag on (NOT the divergent GᵀM⁻¹G
        #      ``consistent_ppe`` path — Task 4 found that diverges).
        #   #4 rotational-incremental pressure update p = p* + phi - nu div(u_hat)
        #      (Timmermans 1996; valid for constant nu) — the mode's default.
        #   #3 disjoint outflow BCs (Baskar's rule / Eq 67d-67e) is imposed by the
        #      CALLER supplying ``pressure_outflow_nodes`` (PPE Dirichlet p'=0 on
        #      the pressure CORRECTION at outflow, removing the node-0 gauge) while
        #      the predictor already leaves outflow velocity free (the natural
        #      traction-free do-nothing, n.grad u = 0). The mode ASSERTS the caller
        #      passed outflow nodes for an open flow (else it is the enclosed-cavity
        #      path, node-0 pin, which is left intact for rung 0).
        # Default False => bit-for-bit unchanged. When True it OVERRIDES
        # ppe_fine_scale and pressure_update to the coherent set (a later explicit
        # kwarg cannot silently half-enable it).
        # ---- BACKFLOW STABILIZATION (change #6, 2026-07-23) ----
        # Velocity-based directional-do-nothing outflow stabilization
        # (Bazilevs et al. CMAME 2009; Esmaily-Moghadam et al. Comput. Mech.
        # 2011; = Braack-Mucha directional-do-nothing). Adds
        #     - beta * rho * int_{Gamma_out} (u.n)_- (u . v) dGamma
        # to the momentum predictor's outflow traction, ACTIVE ONLY where the
        # flow reverses through the open outlet (u.n < 0). The open "do-nothing"
        # outflow leaves the convective energy flux unbounded when the wake
        # pushes fluid back in (Re=100 blows up ~step 700); this term restores a
        # coercive (positive-definite) contribution exactly there and is ~0
        # (benign) with no backflow. ``backflow_beta`` is the knob:
        #   None (default) -> 0.0 in the base scheme (OFF, bit-for-bit),
        #                     0.5 in the consistent-projection scheme (the
        #                     analysis-backed value; beta=1 is the robust upper
        #                     choice). An explicit float overrides either.
        # The predictor's outflow-face set is discovered from the mesh the first
        # time step() runs (cached), so ``pressure_outflow_nodes`` (the disjoint
        # PPE Dirichlet) and this velocity-side term stay on the SAME open
        # outlet without new caller wiring.
        if backflow_beta is None:
            self.backflow_beta = 0.5 if consistent_projection else 0.0
        else:
            self.backflow_beta = float(backflow_beta)
        self._backflow_faces = None       # lazily discovered outflow face set
        self.consistent_projection = bool(consistent_projection)
        if self.consistent_projection:
            ppe_fine_scale = True
            if pressure_update == "standard":
                pressure_update = "rotational"
            if consistent_ppe:
                raise ValueError(
                    "consistent_projection uses the PSPG-stabilized PPE "
                    "(ppe_fine_scale); do NOT combine with the divergent "
                    "consistent_ppe=True (GᵀM⁻¹G) path.")
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
        # P2-R2a OUTFLOW-BC lever (Task 3b, Baskar outflow-BC route; default
        # None = UNCHANGED enclosed-flow node-0 pin). The current PPE pins a
        # single arbitrary FREE node (node 0) as "enclosed flow", which leaves
        # the outflow pressure floating for an EXTERNAL flow with a free
        # outflow face -> the incremental p* drifts (bake-off 556166d: no
        # stable config). Taly ns_vms instead imposes a physical Dirichlet
        # pressure BC on the outlet nodes (NS_VMS_Proj_PPE.h::fillEssBC).
        # When ``pressure_outflow_nodes`` is a non-empty array of FREE-node
        # indices, the PPE applies Dirichlet p=0 on ALL those rows (zero row,
        # unit diagonal, zero rhs) INSTEAD of the single node-0 pin, EVERYWHERE
        # the code pins the pressure/PPE (PPE solve, weak-div-free helper, and
        # the rotational B^T u helper) so the projection/consistency paths stay
        # mutually consistent. ``None`` keeps the exact node-0 pin (bit-for-bit).
        if pressure_outflow_nodes is None:
            self.pressure_outflow_nodes = None
        else:
            pon = np.unique(np.asarray(pressure_outflow_nodes, dtype=np.int64))
            self.pressure_outflow_nodes = pon if pon.size else None
        # ---- STABILIZED INNER predictor<->PPE iteration (ladder Task 4) ----
        # Default OFF: single-pass (one predictor -> PPE -> correct), BIT-FOR-BIT
        # identical to the classic incremental projection. When on, the within-
        # step lagged-pressure split is driven to its predictor<->PPE fixed point
        # BEFORE the velocity correction, so p* is self-consistent with u_hat at
        # the end of the step (the from-rest weak-fixed-point remedy of
        # docs/dev/2026-07-23-projection-sbm-weak-fixed-point-verdict.md, path a).
        # The naive nu-loop is an unstable accelerant (verdict exp. 2, rho climbs
        # past 1); this iteration is STABILIZED by:
        #   inner_relax (omega in (0,1]): damped update p* <- p* + omega*(p_hat-p*)
        #   inner_accel ("none"|"anderson"): Anderson mixing (history m) of the
        #                fixed-point map g(p*) = p_hat(p*) on the residual r=p_hat-p*
        #   divergence guard: if the inner residual ||p_hat - p*|| rises for 2
        #                consecutive iters, break to the LAST BOUNDED iterate.
        self.inner_iterate = bool(inner_iterate)
        self.inner_max = int(inner_max)
        self.inner_tol = float(inner_tol)
        omega = float(inner_relax)
        if not (0.0 < omega <= 1.0):
            raise ValueError(
                f"inner_relax must be in (0, 1], got {inner_relax!r}")
        self.inner_relax = omega
        if inner_accel not in ("none", "anderson"):
            raise ValueError(
                f"inner_accel must be none|anderson, got {inner_accel!r}")
        self.inner_accel = inner_accel
        self.inner_anderson_m = int(inner_anderson_m)
        # per-step inner diagnostics (populated by step() when inner_iterate)
        self.inner_iters = 0
        self.inner_res_hist = []
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
        # ---- CONSISTENT PPE operator L = G^T M^-1 G (ladder Task 4, Part 2) ----
        # Default False keeps the FE-Laplacian K_p (unchanged). The classic
        # incremental projection solves the PPE with K_p, but the velocity
        # correction u = u_hat - (1/sigma) M^-1 G phi projects with the DISCRETE
        # operator G^T M^-1 G. K_p != G^T M^-1 G (measured ~67% relative
        # difference on the rung-A mesh), so the PPE does NOT map the corrected
        # field onto the discretely divergence-free space (B^T u -> 0): a seeded
        # monolithic field is NOT preserved and the split fixed point differs
        # from the monolithic (docs/dev/2026-07-23 verdict exp. 4). When True,
        # the PPE uses the CONSISTENT L so the projection is a true discrete
        # projection (idempotent: B^T u_corr -> machine zero). This is the
        # operator-consistency test the rung-A Part-2 diagnosis calls for.
        self.consistent_ppe = bool(consistent_ppe)
        self._L_ppe = self._consistent_ppe_operator() if self.consistent_ppe \
            else None
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

    def _backflow_block(self, a_node):
        """Constrained node-major outflow backflow block (#6) at advecting
        field ``a_node`` [n_free, dim]. Discovers the outflow face set once
        (cached) and delegates to ns_bricks.assemble_backflow_block. Returns a
        zero CSR when ``backflow_beta == 0`` (never called in that case)."""
        from ..api.ns_bricks import assemble_backflow_block, outflow_faces
        if self._backflow_faces is None:
            self._backflow_faces = outflow_faces(self.dm.mesh)
        return assemble_backflow_block(
            self.dm, a_node, self.backflow_beta, self.ndof,
            faces=self._backflow_faces)

    def _gradient_operator(self):
        """Discrete gradient G: scalar free-node pressure -> per-component
        free-node vectors, with block c the CSR of
            G_c[a, b] = int N_a (dN_b/dx_c) dV
        (the (test N, trial grad-N) coupling). Returns a list of ``dim`` CSR
        blocks in the CONSTRAINED scalar free-node space (n_free x n_free).

        This is the SAME operator the velocity correction applies: the
        correction RHS is int N (u_hat - (1/sigma) grad(phi)) so the phi-part is
        -(1/sigma) G phi. Building it explicitly lets the PPE use the CONSISTENT
        L = G^T M^-1 G (see _consistent_ppe_operator)."""
        dm = self.dm
        dim = dm.dim
        blocks = []
        T = dm.constraints.T.tocsr()
        for c in range(dim):
            rows, cols, vals = [], [], []
            for pv, b in dm.bins.items():
                tb = dm.tables_by_p[pv]
                h = dm.mesh.tree.h()[dm.mesh.bins[pv]]
                jac = (h / 2.0) ** dim
                dsc = (2.0 / h)
                # Ge[e, a, b] = int N_a (dN_b/dx_c) = sum_q N_a dN_b_c w * jac*dsc
                Ge = np.einsum("qa,qbc,q->abc", tb.N, tb.dN, tb.w)[:, :, c]
                conn = dm.mesh.conn_of[pv].astype(np.int64)
                nbf = conn.shape[1]
                Gee = Ge[None, :, :] * (jac * dsc)[:, None, None]
                rows.append(np.repeat(conn, nbf, axis=1).ravel())
                cols.append(np.tile(conn, (1, nbf)).ravel())
                vals.append(Gee.ravel())
            Nn = dm.n_nodes
            Gc = sp.coo_matrix((np.concatenate(vals),
                                (np.concatenate(rows), np.concatenate(cols))),
                               shape=(Nn, Nn)).tocsr()
            blocks.append((T.T @ Gc @ T).tocsr())
        return blocks

    def _consistent_ppe_operator(self):
        """L = sum_c G_c^T M^-1 G_c — the CONSISTENT PPE operator (Part 2).

        G_c^T[a,b] = int (dN_a/dx_c) N_b = the discrete divergence B^T's c-block;
        M the scalar consistent mass. L is the operator for which the classic
        consistent-mass velocity correction is a TRUE discrete projection
        (B^T u_corr = 0 at the fixed point). Assembled once (factorize M, apply
        M^-1 to each G_c column). Small 2-D rung-A meshes only."""
        from scipy.sparse.linalg import splu as _splu
        G = self._gradient_operator()          # dim blocks, each G_c
        Minv = _splu(self.M.tocsc())
        L = None
        for Gc in G:
            Gd = Gc.toarray()
            MiG = Minv.solve(Gd)               # M^-1 G_c (dense n_free x n_free)
            Lc = Gc.T @ MiG                     # G_c^T M^-1 G_c
            L = Lc if L is None else L + Lc
        return sp.csr_matrix(L)

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

    def _pin_rows(self):
        """Free-node rows the PPE/consistency paths pin the pressure at.

        Default (``pressure_outflow_nodes is None``) — the single enclosed-flow
        node-0 pin (unchanged). Otherwise — the outflow Dirichlet nodes (Taly
        physical outlet pressure BC). ALL pin sites (the PPE solve, the weak-
        div-free helper, the rotational B^T u helper) route through here so
        they pin the SAME rows and stay consistent."""
        if self.pressure_outflow_nodes is None:
            return (0,)
        return self.pressure_outflow_nodes

    def _apply_pin_lil(self, Kp_lil, rhs_free):
        """Apply the pressure pin (Dirichlet p=0) to a LIL PPE operator +
        free-space rhs, honoring ``pressure_outflow_nodes``. Zeros each pinned
        row to a unit diagonal and zeros the matching rhs entry."""
        for r in self._pin_rows():
            r = int(r)
            Kp_lil.rows[r] = [r]
            Kp_lil.data[r] = [1.0]
            rhs_free[r] = 0.0
        return Kp_lil, rhs_free

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
        # pin the SAME rows as the PPE (free-node 0, or the outflow Dirichlet
        # nodes when pressure_outflow_nodes is set) so the rotational B^T u
        # helper stays consistent with the PPE pin convention.
        for r in self._pin_rows():
            bt_free[int(r)] = 0.0
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
            # Backflow stabilization (#6): add the outflow directional-do-nothing
            # block, linearized (Picard) at the current advecting iterate
            # ``a_node``. beta=0 => a structural zero (bit-for-bit OFF).
            if self.backflow_beta != 0.0:
                A = (A + self._backflow_block(a_node))
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

        # ---- Steps 1-2 (+ pressure update): the predictor<->PPE PASS ----
        # Single-pass (default) runs this exactly once against the lagged p*;
        # the STABILIZED inner iteration (Task 4) drives p* to the within-step
        # predictor<->PPE fixed point before the correction (see _inner_solve).
        if not self.inner_iterate:
            self.inner_iters = 1
            self.inner_res_hist = []
            uhat, phi, p_hat, uq, fs_vel, sigma = self._projection_pass(
                t_new, extra_block, sbm_nodes, ppe_surrogate_flux)
        else:
            uhat, phi, p_hat, uq, fs_vel, sigma = self._inner_solve(
                t_new, extra_block, sbm_nodes, ppe_surrogate_flux)
        # ---- Step 3: velocity correction (uses the final pass) ----
        return self._correct_and_finish(
            uhat, phi, p_hat, uq, fs_vel, sigma, dim, gvals, sbm_nodes, t_new)

    def _projection_pass(self, t_new, extra_block, sbm_nodes,
                         ppe_surrogate_flux):
        """One predictor -> PPE -> pressure-update PASS against the CURRENT
        ``self.p_star``. Returns ``(uhat, phi, p_hat, uq, fs_vel, sigma)``: the
        predicted velocity, the pressure increment ``phi``, the updated
        pressure ``p_hat = p* + phi`` (per ``pressure_update``), the GP velocity
        values + cached fine-scale velocity for the correction, and ``sigma``.

        Does NOT mutate ``self.p_star`` or the history — pure w.r.t. the working
        pressure, so the inner iteration can call it repeatedly. Single-pass
        default: called once, bit-for-bit identical to the classic split."""
        dm = self.dm
        dim = dm.dim
        # ---- Step 1: nonlinear predictor (Picard over the full block with
        # pressure DOFS PINNED to p*) ----
        uhat = self._predict(t_new=t_new, extra_block=extra_block,
                             sbm_nodes=sbm_nodes)
        # ---- Step 2: PPE with tau_m fine-scale RHS ----
        uq, guq = self._gp_vals(uhat, grad=True)
        pq_g = self._gp_vals(self.p_star, grad=True)[1]
        b0, _b1, _b2, sigma, _u1, _u2, fq_base, _gvals = \
            self._predictor_setup(t_new)
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
                if self.consistent_projection:
                    # CONSISTENT-PROJECTION PPE source (the 2026-07-23 fix, #1).
                    # Per ns_projection Eq 44b + Remark 2.2/Eq 45, the coarse
                    # divergence stays COLLOCATED with the test q (NOT integrated
                    # by parts) so it matches the monolithic PSPG continuity row
                    #   (q, div u_h) + tau_m (grad q, r_m) = 0
                    # term-for-term; ONLY the fine scale u' = -tau_m r_m is taken
                    # by parts:  -sigma(div u_h, q)_h - sigma(tau_m r_m, grad q)_h.
                    # The by-parts of the WHOLE flux (old ppe_fine_scale path)
                    # instead put the coarse part by parts too, fabricating a
                    # boundary term sigma(u_h.n, q)_Gamma that is NONZERO at an
                    # OPEN outflow -> a spurious phi that corrupts a seeded
                    # monolithic field (div 1.22->6.0). Assembling the coarse
                    # divergence collocated closes that gap: the monolithic steady
                    # state becomes a fixed point (phi -> 0).  ``flux`` here is
                    # the ONLY by-parts (grad-q) piece (the fine scale); the
                    # coarse divergence is added collocated below.
                    flux = -sigma * fs_vel[pv]            # -sigma tau_m r_m
                    div_uh = np.einsum(
                        "gdd->g", guq[pv].reshape(-1, dim, dim))
                    conn0 = dm.mesh.conn_of[pv]
                    jac0 = (h / 2.0) ** dim
                    ne0 = len(h)
                    # -sigma int N_a (div u_h) : collocated coarse divergence.
                    be0 = np.einsum(
                        "qa,eq,q,e->ea", tb.N,
                        div_uh.reshape(ne0, nqp), tb.w, -sigma * jac0)
                    np.add.at(rhs, conn0.ravel(), be0.ravel())
                else:
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
            # pin: node-0 (enclosed flow) OR the outflow Dirichlet nodes.
            Kp, rhs_free = self._apply_pin_lil(Kp, rhs_free)
            from ..solvers.linsolve import solve_linear
            phi = solve_linear(Kp.tocsr(), rhs_free, solver=self.solver,
                               sym=True, device=self.dm.device,
                               cache=self._solver_cache)
        else:
            # CONSISTENT PPE (Part 2): L = G^T M^-1 G instead of the FE
            # Laplacian K_p, so the classic consistent-mass correction is a true
            # discrete projection. Default: K_p (unchanged).
            Kp = (self._L_ppe if self.consistent_ppe else self.K_p).tolil()
            # pin (enclosed flow: node 0) OR the outflow Dirichlet nodes.
            Kp, rhs_free = self._apply_pin_lil(Kp, rhs_free)
            from ..solvers.linsolve import solve_linear
            # NOTE: the "ppe" cache_key must NOT be reused across different pin
            # patterns; the outflow pin changes the factorized operator, so key
            # it only for the default node-0 pin (unchanged fast path).
            cache_key = ("ppe" if (self.pressure_outflow_nodes is None
                                   and not self.consistent_ppe) else None)
            phi = solve_linear(Kp.tocsr(), rhs_free, solver=self.solver,
                               sym=True, device=self.dm.device,
                               cache=self._solver_cache, cache_key=cache_key)
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
        return uhat, phi, p_hat, uq, fs_vel, sigma

    def _inner_solve(self, t_new, extra_block, sbm_nodes, ppe_surrogate_flux):
        """STABILIZED inner predictor<->PPE iteration (Task 4).

        Drives ``self.p_star`` to the within-step fixed point of the map
        ``g(p*) = p_hat(p*)`` (predict u_hat at p*, PPE -> phi -> p_hat) BEFORE
        the velocity correction, so p* is self-consistent with u_hat at the end
        of the step. The naive fixed-point recursion (``p* <- g(p*)``) is the
        unstable nu-loop (verdict exp. 2); this is stabilized by damped
        relaxation and optional Anderson mixing on the residual r = g(p*) - p*,
        plus a divergence guard that returns the LAST BOUNDED iterate if the
        residual rises for two consecutive iterations.

        Returns the final ``(uhat, phi, p_hat, uq, fs_vel, sigma)`` bundle for
        the correction (using the LAST pass's phi = p_hat - p*, so at the fixed
        point phi -> 0 and the corrected field -> the divergence-free predictor).
        """
        p0 = self.p_star.copy()
        res_hist = []
        gk_list = []          # g(p*) iterates (for Anderson)
        pk_list = []          # p* iterates    (for Anderson)
        last_bundle = None
        best = None           # (res, p_star_in, bundle) — last bounded iterate
        rise = 0
        prev_res = np.inf
        m = max(1, self.inner_anderson_m)
        for it in range(self.inner_max):
            bundle = self._projection_pass(
                t_new, extra_block, sbm_nodes, ppe_surrogate_flux)
            p_hat = bundle[2]
            r = p_hat - self.p_star                       # fixed-point residual
            res = float(np.linalg.norm(r))
            res_hist.append(res)
            bounded = np.isfinite(res)
            if bounded and (best is None or res <= best[0]):
                best = (res, self.p_star.copy(), bundle)
            last_bundle = bundle
            self.inner_iters = it + 1
            if bounded and res < self.inner_tol:
                break
            # divergence guard: residual rose two iters in a row -> bail to the
            # last bounded (best) iterate; the split is locally non-contractive.
            if (not bounded) or res > prev_res:
                rise += 1
            else:
                rise = 0
            if rise >= 2 or not bounded:
                if best is not None:
                    # recompute the pass AT the best p* so phi/uhat are the
                    # bounded ones handed to the correction.
                    self.p_star = best[1]
                    last_bundle = best[2]
                break
            prev_res = res
            # ---- pressure update: relaxation, optionally Anderson-mixed ----
            if self.inner_accel == "anderson":
                gk_list.append(p_hat.copy())
                pk_list.append(self.p_star.copy())
                p_next = self._anderson_step(pk_list, gk_list, m,
                                             self.inner_relax)
            else:
                # damped relaxation: p* <- p* + omega (p_hat - p*)
                p_next = self.p_star + self.inner_relax * r
            self.p_star = p_next
        else:
            # exhausted inner_max without hitting tol — keep the best bounded.
            if best is not None and not np.isfinite(res_hist[-1]):
                self.p_star = best[1]
                last_bundle = best[2]
        self.inner_res_hist = res_hist
        # restore p_star to p0 so _correct_and_finish sets it to p_hat cleanly
        # (the returned p_hat is the fixed-point pressure; the correction's phi
        # is that pass's increment relative to the p* it was solved at).
        return last_bundle

    @staticmethod
    def _anderson_step(pk_list, gk_list, m, beta):
        """Anderson-accelerated update for the fixed-point map g. Given the
        histories of iterates p_k and their images g_k = g(p_k), builds the
        residuals f_k = g_k - p_k and returns the type-II Anderson mixing

            p_{k+1} = (1-beta) * (sum alpha_i p_i) + beta * (sum alpha_i g_i)

        over the last ``m+1`` samples, alpha the least-squares coefficients
        minimizing ||sum alpha_i f_i|| with sum alpha_i = 1 (solved via the
        differenced-residual normal equations). Falls back to damped relaxation
        when the history is too short or the LS system is singular."""
        p = np.asarray(pk_list)
        g = np.asarray(gk_list)
        f = g - p                                    # residuals
        k = len(f)
        if k < 2:
            # not enough history: plain damped relaxation.
            return p[-1] + beta * f[-1]
        mm = min(m, k - 1)
        F = f[-1] - f[-(mm + 1):-1]                  # [mm, n] differences
        F = F.T                                      # [n, mm]
        fk = f[-1]                                   # [n]
        try:
            gamma, *_ = np.linalg.lstsq(F, fk, rcond=None)
        except np.linalg.LinAlgError:
            return p[-1] + beta * f[-1]
        if not np.all(np.isfinite(gamma)):
            return p[-1] + beta * f[-1]
        pw = p[-1] - (p[-(mm + 1):-1] - p[-1]).T @ gamma
        gw = g[-1] - (g[-(mm + 1):-1] - g[-1]).T @ gamma
        return (1.0 - beta) * pw + beta * gw

    def _correct_and_finish(self, uhat, phi, p_hat, uq, fs_vel, sigma,
                            dim, gvals, sbm_nodes, t_new):
        """Step 3 (velocity correction) + Step 4 (commit p* and history)."""
        dm = self.dm
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
