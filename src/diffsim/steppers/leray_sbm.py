"""Volumetric-SBM composed Leray projection stepper (P2-R0, Tasks 2-4).

## P2-R0 SCALING-PATHWAY DECLARATION (Task 11, mandatory)

Recorded 2026-07-21.  Full table in spec §P2-R0 Scaling-pathway declaration.

DEVICE STATUS: R0 is HOST/splu-reference only.  The device port is the R2
prerequisite (resolution 3 + Task-9/10 findings).  Do NOT read this module as
device-ready.

STAGE RESIDENCY (summary):
- Octree / surrogate extraction / distance:  host, per-epoch  (static geometry)
- SBM face-block assembly:                  host, once-per-epoch  (cached)
- Predictor sub-solve (nonsymmetric Oseen): host (R0) → device FGMRES #49 (R2)
- PPE sub-solve (SPD Laplacian):            host (R0) → AMGX AMG/CG (R2)
- Correction sub-solve (mass matrix):       host (R0) → device (R2)

THE SCALABLE LEVER: the SPD pressure-Poisson (PPE) admits AMG/CG directly
(AMGX); this is the primary reason projection was chosen over the monolithic
saddle for the P2 hero path.  R2's device-PPE task is the highest-leverage
scaling item.  cuDSS cannot reach the 100M-DOF hero (#43); iterative device
solve (device-FGMRES #49 / AMGX) is the mandated path.

NO NEW NNZ SPACE: the SBM face block scatters ONLY into existing NS node-pair
slots (verified by tests/test_p2r0_scaling.py::test_no_new_nnz_space).  The
ChunkedCSR (#38) + fp32-IR (#36) + multi-GPU (S4) 100M-DOF budget is not
inflated by the SBM composition.

R0 OPEN ITEMS (deferred to R2, not hidden):
- 3-D pressure-coupling stability (principled penalty α~Pe·p² or implicit PPE).
- G4/G5 Cd/Strouhal literature convergence (requires device-AMG mesh scales).



`LeraySBMStepper` COMPOSES `LerayProjectionStepper` (the audited VMS-Helmholtz-
Leray projection stepper) with an immersed-geometry oracle — it does NOT fork
the base stepper. It runs the per-epoch surrogate pipeline
(`classify_lambda` -> `extract_surrogate` -> `GeometryData.evaluate`), builds
the geometry-only shifted-Nitsche vector Dirichlet face block ONCE
(`sbm_vector_dirichlet`, no-slip body), and injects it into the base stepper's
predictor sub-solve via the `extra_block`/`sbm_nodes` hook added to
`leray.py::LerayProjectionStepper._predict`.

END-TO-END DRIVER (Task 4): a caller supplies ONLY an immersed-geometry oracle
+ box boundary data (`u_inf`, `strong_mask`) and gets a full BDF2
projection+SBM march with the base stepper's ergonomics — `set_initial`,
`step() -> (u_new, p_hat)`, `divergence_l2()` — plus the `surrogate_traction`
observable. The BDF1->BDF2 bootstrap is handled by the base `History`
(`bdf_order_now` gates BDF1 until `t >= 1.5 dt` and 2 history slots exist).
Backflow (inflow) stabilization on the surrogate faces is wired PER STEP: the
advecting field `a_face` is the current velocity `u^n` sampled at the
surrogate-face GPs, and its `-beta (a.n)_- N N` contribution is added to the
cached geometry block (which stays untouched — the Task-2 caching invariant).

Pipeline per step: predictor (SBM Dirichlet + backflow, box strong) -> PPE with
surrogate-consistent homogeneous-Neumann BC (Task 3) -> L2 correction leaving
the SBM trace to the projection (`sbm_nodes` skip the box overwrite) -> BDF
history rotate.

CONVERGENCE CONTROL: `picard_iters` is threaded to the base predictor and is
the exposed knob for driving the predictor<->PPE coupling (Tasks 6/9 use it to
probe whether more Picard reduces the immersed-projection divergence — an open
convergence item; on the coarse Re20 fixture div stays O(10), not the tight
body-fitted target).

The immersed body is enforced WEAKLY (SBM, on the surrogate faces); the box
inflow/walls are enforced STRONGLY from the driver-supplied `strong_mask` +
`u_inf` (the M1b-test convention). `geo.n` is domain-outward; drag orientation
uses `n_hat = -geo.n` inside `surrogate_traction`.
"""
import numpy as np
import scipy.sparse as sp

from ..mesh.faces import face_tables
from ..sbm.surrogate import (classify_lambda, extract_surrogate, GeometryData)
from ..sbm.vector import (sbm_vector_dirichlet, sbm_vector_dirichlet_twosided,
                          surrogate_traction)
from ..solvers.timestepping import bdf_coeffs, bdf_order_now
from .leray import LerayProjectionStepper


class LeraySBMStepper:
    def __init__(self, oracle, dm, nu, dt, f_fn, *, u_inf, strong_mask,
                 lam=0.5, domain="outside", order=2, picard_iters=2,
                 solver="splu", ppe_finescale=False, alpha=10.0,
                 beta_backflow=1.0, velocity_update="consistent",
                 graddiv_scale=1.0, pressure_update="standard",
                 ppe_fine_scale=False, pressure_outflow_nodes=None,
                 sbm_pressure_coupling=False,
                 inner_iterate=False, inner_max=8, inner_tol=1e-6,
                 inner_relax=1.0, inner_accel="none", inner_anderson_m=3,
                 consistent_ppe=False, consistent_projection=False):
        self.oracle = oracle
        self.dm = dm
        self.nu = nu
        self.dt = dt
        self.dim = dm.dim
        self.ndof = dm.dim + 1
        self.domain = domain
        self.lam = lam
        self.alpha = alpha
        self.beta_backflow = beta_backflow
        # P2-R2a stage 1: SBM velocity<->pressure coupling (T3 + T6). Default
        # False reproduces the R0 velocity-velocity-diagonal path bit-for-bit.
        self.sbm_pressure_coupling = bool(sbm_pressure_coupling)

        # --- box strong Dirichlet (inflow/walls), driver-supplied ---
        u_inf = np.asarray(u_inf, dtype=np.float64)
        strong_mask = np.asarray(strong_mask, dtype=bool)
        self.strong_mask = strong_mask
        self.u_inf = u_inf
        self._strong_nodes = np.where(strong_mask)[0]

        # --- per-epoch surrogate pipeline (mirrors test_cylinder.py) ---
        # dm already carries the retained tree/mesh; classify from the SAME
        # base tree the mesh was built on so the surrogate matches the dm.
        tree = dm.mesh.tree
        ret, _frac = classify_lambda(tree, oracle, lam, domain=domain)
        self.sf = extract_surrogate(ret)
        self.geo = GeometryData.evaluate(
            oracle, ret, self.sf, face_tables(1, self.dim), domain=domain)

        # --- internal base stepper for the volume machinery ---
        # g_fn returns the STRONG box-Dirichlet trace (u_inf) at dir_nodes;
        # we override dir_nodes to exactly the strong-mask nodes so the box
        # is strong and the immersed body is left weak (SBM).
        base = LerayProjectionStepper(
            dm, nu, dt, f_fn, self._g_box, order=order,
            picard_iters=picard_iters, solver=solver,
            ppe_finescale=ppe_finescale,
            velocity_update=velocity_update, graddiv_scale=graddiv_scale,
            pressure_update=pressure_update, ppe_fine_scale=ppe_fine_scale,
            pressure_outflow_nodes=pressure_outflow_nodes,
            inner_iterate=inner_iterate, inner_max=inner_max,
            inner_tol=inner_tol, inner_relax=inner_relax,
            inner_accel=inner_accel, inner_anderson_m=inner_anderson_m,
            consistent_ppe=consistent_ppe,
            consistent_projection=consistent_projection)
        base.dir_nodes = self._strong_nodes
        self.base = base
        self.n_free = base.n_free

        # --- geometry-only SBM face block, assembled ONCE ---
        T = dm.constraints.T.tocsr()
        self._T_vec = sp.kron(T, sp.identity(self.ndof, format="csr"),
                              format="csr")
        self._g_body = lambda y: np.zeros((len(y), self.dim))  # no-slip
        Af, bf = sbm_vector_dirichlet(
            dm, self.sf, self.geo, self._g_body, nu, self.ndof, alpha=alpha,
            a_face=None, beta_backflow=beta_backflow)
        self.Af_c = (self._T_vec.T @ Af @ self._T_vec).tocsr()
        self.bf_c = np.asarray(self._T_vec.T @ bf)

        # --- surrogate-face GP evaluation cache (for the per-step backflow
        # advecting field a_face on sf; matches the geo.n GP layout (fi, q)) ---
        self._bf_setup()

        # --- SBM-governed free nodes: surrogate-face nodes that must NOT get
        # the strong box-row overwrite (weak body). Computed from the face
        # block's nonzero velocity rows mapped to free-node indices. ---
        self._sbm_nodes = self._compute_sbm_nodes()

    # ---- driver-supplied box Dirichlet ----
    def _g_box(self, coords_at_dir, t):
        # dir_nodes == strong_nodes, so return the u_inf rows in that order.
        return self.u_inf[self._strong_nodes]

    def _compute_sbm_nodes(self):
        dm = self.dm
        mesh = dm.mesh
        # global node ids touched by surrogate faces
        conn = mesh.conn_of[1][np.searchsorted(mesh.bins[1], self.sf.elem)]
        glob = np.unique(conn.ravel())
        # free_nodes is an ARRAY OF NODE IDS (not a bool mask)
        free_idx = dm.constraints.free_nodes
        free_of = np.full(dm.n_nodes, -1, dtype=np.int64)
        free_of[free_idx] = np.arange(len(free_idx))
        fn = free_of[glob]
        return fn[fn >= 0]

    # ---- per-step backflow (inflow-stabilization) advecting field ----
    def _bf_setup(self):
        """Cache the surrogate-face GP tables so the per-step backflow
        advecting field ``a_face`` (the CURRENT velocity sampled at the
        surrogate-face GPs, ordered ``(fi, q)`` to match ``geo.n``) can be
        rebuilt cheaply each step. Backflow adds ``-beta (a.n)_- N N`` on the
        surrogate faces (inflow through an outflow face is penalized), which
        suppresses spurious inflow through the immersed body during the
        transient — the same production ``inflow_g < 0`` branch that
        ``sbm_vector_dirichlet`` assembles from ``a_face``."""
        dm = self.dm
        mesh = dm.mesh
        sf = self.sf
        self._bf_pv = int(np.unique(np.asarray(mesh.p_elem)[sf.elem])[0])
        self._bf_ftab = face_tables(self._bf_pv, self.dim)
        self._bf_conn = mesh.conn_of[self._bf_pv][
            np.searchsorted(mesh.bins[self._bf_pv], sf.elem)]     # [Nf, nbf]

    def _t3_pressure_rhs(self):
        """T3 pressure-consistency, lagged, on the PREDICTOR RHS.

        Monolithic ([Mono] NSEquation.h:237-238) is the LHS block
        ``Ae(vel_j, pres) += N_a N_b normal(j)`` -> weak form
        ``+int_Gamma~ p N_a normal_j`` with ``normal = geo.n`` (domain-outward,
        INTO the body). In the projection factoring the pressure is LAGGED to
        the known ``p*`` and moved to the RHS. Following the task spec's
        surrogate convention ``n_tilde = -geo.n`` (matching
        ``surrogate_traction``'s ``+p*geo.n`` drag integrand, which is the
        SAME sign as the monolithic ``+p normal`` momentum loading), the
        predictor RHS force per velocity component c and test node a is

            b[node_a, c] += sum_gp w_gp * p*_gp * n_tilde_c(gp) * N_a(gp),
            n_tilde = -geo.n,  w_gp = ftab.w[q] * jacS * geo.corr.

        This lands in the FULL node-major (unconstrained) RHS; the caller
        constraint-reduces it with ``_T_vec.T``. Returns a full node-major
        [n_nodes*ndof] vector (velocity rows only; pressure rows zero).
        Assembled EACH STEP because it depends on the lagged p*.
        """
        dm = self.dm
        mesh = dm.mesh
        dim = self.dim
        sf, geo = self.sf, self.geo
        ftab = self._bf_ftab
        nqf = ftab.nqf
        conn = self._bf_conn
        h = mesh.tree.h()[sf.elem]
        jacS = (h / 2.0) ** (dim - 1)
        # lagged pressure p* (free-node scalar) -> full node-major scalar
        p_full = np.asarray(dm.constraints.T @ self.base.p_star)   # [n_nodes]
        b = np.zeros(dm.n_nodes * self.ndof)
        bv = b.reshape(dm.n_nodes, self.ndof)
        for fi in range(len(sf.elem)):
            f = int(sf.face[fi])
            pn = p_full[conn[fi]]                                  # [nbf]
            for q in range(nqf):
                w = ftab.w[q] * jacS[fi] * geo.corr[fi * nqf + q]
                n_hat = -geo.n[fi * nqf + q]                       # n_tilde
                Nq = ftab.N[f][q]                                  # [nbf]
                pq = Nq @ pn                                       # p*_gp
                # b[node_a, c] += w * pq * n_hat_c * N_a
                for c in range(dim):
                    bv[conn[fi], c] += w * pq * n_hat[c] * Nq
        return b

    def _t6_nopenetration_flux(self, uhat):
        """T6 shifted no-penetration flux for the PPE RHS.

        Monolithic ([Mono] NSEquation.h:222-223) is the continuity-row block
        ``Ae(pres, vel_j) += -N_a (N_b + gradDELUdotd(j)) normal(j)`` -> weak
        form ``-int_Gamma~ q (n . Su)`` with ``normal = geo.n`` and the shifted
        trial ``Su = u + grad(u).d`` (first-order shift; p1 faces). In the
        PROJECTION factoring this continuity constraint is enforced through the
        PPE surrogate-flux hook: the PPE source must SUBTRACT the surrogate
        shifted-normal velocity of the predictor so the corrected field carries
        no penetration through the body.

        The base PPE builds its source as ``int grad(N).flux`` with
        ``flux = sigma u_hat`` (divergence form). Its natural (homogeneous
        Neumann) surrogate BC leaves ``n.u_tilde`` untouched. The R0
        planted-break ``_ppe_break_flux`` ADDS ``+N_a sigma (u_hat.geo.n) corr``
        (pushing mass THROUGH the body). T6 is the opposite sign of that: it
        REMOVES the shifted-normal flux so the PPE enforces no-penetration.
        Using ``n_tilde = -geo.n`` and the shifted normal velocity
        ``n_tilde . u_tilde``, ``u_tilde = u_hat + grad(u_hat).d``:

            rhs[node_a] += sum_gp N_a(gp) * sigma * (n_tilde . u_tilde)(gp)
                           * w_gp,   w_gp = ftab.w[q] * jacS * geo.corr.

        (Sign: this is the negative of the R0 break-flux with the same
        ``n_tilde = -geo.n``, i.e. it drives ``n.u -> 0`` at the surrogate
        rather than pushing flow through it — the continuity-row role of T6.)
        Returns a FULL node-major [n_nodes] scalar; the base ``step`` adds it
        to the PPE RHS before the constraint reduction.
        """
        dm = self.dm
        mesh = dm.mesh
        dim = self.dim
        sf, geo = self.sf, self.geo
        ftab = self._bf_ftab
        nqf = ftab.nqf
        conn = self._bf_conn
        h = mesh.tree.h()[sf.elem]
        jacS = (h / 2.0) ** (dim - 1)
        dscale = 2.0 / h
        b0, _b1, _b2 = bdf_coeffs(
            bdf_order_now(self.base.t + self.dt, self.dt, self.base.order,
                          have_history=self.base.hist.have(2)), self.dt)
        sigma = b0 / self.dt
        u_full = np.asarray(dm.constraints.T @ uhat)              # [n_nodes, dim]
        dvec = geo.d.reshape(len(sf.elem), nqf, dim)
        rhs = np.zeros(dm.n_nodes)
        for fi in range(len(sf.elem)):
            f = int(sf.face[fi])
            un = u_full[conn[fi]]                                 # [nbf, dim]
            for q in range(nqf):
                w = ftab.w[q] * jacS[fi] * geo.corr[fi * nqf + q]
                n_hat = -geo.n[fi * nqf + q]                      # n_tilde
                Nq = ftab.N[f][q]                                 # [nbf]
                # shifted trial velocity u_tilde = u_hat + grad(u_hat).d
                gradu = (ftab.dN[f][q] * dscale[fi]).T @ un       # [dim, dim]
                u_gp = Nq @ un                                    # [dim]
                u_tilde = u_gp + gradu @ dvec[fi, q]              # grad(u).d
                un_tilde = u_tilde @ n_hat                        # n_tilde . u_tilde
                rhs[conn[fi]] += Nq * (sigma * w * un_tilde)
        return rhs

    def _a_face(self, u_free):
        """Advecting field at the surrogate-face GPs from a free-node velocity
        ``u_free`` [n_free, dim], flattened ``(fi, q)`` -> [ne_f*nqf, dim]."""
        dm = self.dm
        u_full = np.asarray(dm.constraints.T @ u_free)            # [n_nodes, dim]
        ftab = self._bf_ftab
        nqf = ftab.nqf
        conn = self._bf_conn
        sf = self.sf
        af = np.empty((len(sf.elem) * nqf, self.dim))
        for fi in range(len(sf.elem)):
            f = int(sf.face[fi])
            un = u_full[conn[fi]]                                 # [nbf, dim]
            for q in range(nqf):
                af[fi * nqf + q] = ftab.N[f][q] @ un
        return af

    def _backflow_block(self, u_free):
        """Constrained free-node-major backflow matrix addition assembled from
        the current advecting field on the surrogate faces. Zero advecting
        field -> zero block (so BDF1 step-0 from rest adds nothing). The
        geometry-only base block ``Af_c`` stays cached and untouched (Task-2
        caching invariant); this is the velocity-dependent increment added to
        it per step."""
        if self.beta_backflow == 0.0:
            return None
        a_face = self._a_face(u_free)
        if not np.any(a_face):
            return None
        # Reuse sbm_vector_dirichlet's backflow assembly by differencing the
        # a_face block against the geometry-only block (both share the same
        # consistency/penalty terms; the difference is exactly the backflow Ab).
        Af_bf, _ = sbm_vector_dirichlet(
            self.dm, self.sf, self.geo, self._g_body, self.nu, self.ndof,
            alpha=self.alpha, a_face=a_face, beta_backflow=self.beta_backflow)
        Ab = (self._T_vec.T @ Af_bf @ self._T_vec).tocsr() - self.Af_c
        return Ab

    def _extra_block(self, u_free):
        """Predictor SBM extra-block = cached geometry block + per-step
        backflow increment (+ per-step T3 pressure-consistency RHS when
        ``sbm_pressure_coupling`` is on)."""
        Ab = self._backflow_block(u_free)
        A = self.Af_c if Ab is None else (self.Af_c + Ab).tocsr()
        b = self.bf_c
        if self.sbm_pressure_coupling:
            # T3: +int_Gamma~ p* n_tilde N_a on the momentum RHS (lagged p*).
            # Full node-major -> constrained free-node-major via _T_vec.T.
            b = b + np.asarray(self._T_vec.T @ self._t3_pressure_rhs())
        return (A, b)

    # ---- public ergonomics (delegate to the base) ----
    def set_initial(self, u0_fn):
        self.base.set_initial(u0_fn)

    def divergence_l2(self):
        return self.base.divergence_l2()

    @property
    def t(self):
        return self.base.t

    def _current_a_free(self):
        """Free-node velocity that drives the per-step backflow advecting
        field: the current state ``u^n`` (``hist.pre1``), or zeros before
        ``set_initial`` (step 0 from rest -> no backflow)."""
        if self.base.hist.pre1 is None:
            return np.zeros((self.n_free, self.dim))
        return self.base._uvec(self.base.hist.pre1)

    # ---- predictor hook (Task 2): SBM block into the momentum sub-solve ----
    def _predict(self, return_matrix=False):
        return self.base._predict(
            extra_block=self._extra_block(self._current_a_free()),
            sbm_nodes=self._sbm_nodes, return_matrix=return_matrix)

    def step(self, surrogate_consistent=True):
        """One projection step with the surrogate-consistent PPE + correction
        (P2-R0 Task 3).

        The surrogate-consistent boundary condition on the pressure-Poisson
        increment at the immersed body is HOMOGENEOUS Neumann
        ``grad(phi).n_hat = 0`` (Suresh pressure-projection SBM paper, Eq. 5 +
        Remark 3.9) — already the natural BC of the base PPE RHS on the
        surrogate faces — and the correction leaves the SBM-governed velocity
        trace to the L2 projection (``sbm_nodes`` skip the box overwrite). The
        two together preserve the SBM predictor's shifted no-penetration
        (blockage) through the projection.

        ``surrogate_consistent=False`` is the PLANTED BREAK: it injects the
        paper-REJECTED non-homogeneous surrogate pressure flux
        ``oint_sf sigma (u_hat.n_hat) q dS~`` into the PPE RHS, deviating from
        homogeneous Neumann. This lets the correction push mass through the
        body, so the no-penetration metric degrades — proving the
        homogeneous-Neumann BC is load-bearing, not decorative.
        """
        if not surrogate_consistent:
            flux = self._ppe_break_flux          # planted-break (R0)
        elif self.sbm_pressure_coupling:
            # T6 shifted no-penetration: the surrogate-consistent PPE flux is
            # NOT homogeneous (zero) — it removes the predictor's shifted
            # normal velocity n_tilde . u_tilde so the corrected field carries
            # no penetration through the body.
            flux = self._t6_nopenetration_flux
        else:
            flux = None                          # R0 homogeneous-Neumann
        return self.base.step(
            extra_block=self._extra_block(self._current_a_free()),
            sbm_nodes=self._sbm_nodes, ppe_surrogate_flux=flux)

    # ---- surrogate-face PPE flux (planted-break only) ----
    def _ppe_break_flux(self, uhat):
        """Non-homogeneous surrogate pressure flux (Suresh Remark 3.9 REJECTS
        this): oint_sf N_a (sigma u_hat.n_hat) corr dS~ over the surrogate
        faces, scattered to full node-major DOFs. Only used by the
        planted-break leg to deviate from homogeneous Neumann. Mirrors the
        surrogate_traction face-loop (geo.n domain-outward; area-corrected via
        geo.corr)."""
        dm = self.dm
        mesh = dm.mesh
        dim = self.dim
        sf, geo = self.sf, self.geo
        pv = int(np.unique(np.asarray(mesh.p_elem)[sf.elem])[0])
        ftab = face_tables(pv, dim)
        nqf = ftab.nqf
        conn = mesh.conn_of[pv][np.searchsorted(mesh.bins[pv], sf.elem)]
        h = mesh.tree.h()[sf.elem]
        jacS = (h / 2.0) ** (dim - 1)
        b0, _b1, _b2 = bdf_coeffs(
            bdf_order_now(self.base.t + self.dt, self.dt, self.base.order,
                          have_history=self.base.hist.have(2)), self.dt)
        sigma = b0 / self.dt
        # map free-node uhat -> full node-major velocity
        u_full = np.asarray(dm.constraints.T @ uhat)      # [n_nodes, dim]
        rhs = np.zeros(dm.n_nodes)
        for fi in range(len(sf.elem)):
            f = int(sf.face[fi])
            un = u_full[conn[fi]]                          # [nbf, dim]
            for q in range(nqf):
                w = ftab.w[q] * jacS[fi] * geo.corr[fi * nqf + q]
                n = geo.n[fi * nqf + q]
                uq_n = (ftab.N[f][q] @ un) @ n
                rhs[conn[fi]] += ftab.N[f][q] * (sigma * w * uq_n)
        return rhs

    # ---- observable ----
    def surrogate_traction(self, x_full=None):
        if x_full is None:
            # current state: (u^n, p_hat) node-major free vector -> full
            u = self.base._uvec(self.base.hist.pre1)
            xfree = np.zeros(self.n_free * self.ndof)
            xv = xfree.reshape(self.n_free, self.ndof)
            xv[:, :self.dim] = u
            xv[:, self.dim] = self.base.p_star
            x_full = np.asarray(self._T_vec @ xfree)
        return surrogate_traction(self.dm, self.sf, self.geo, x_full,
                                  self.nu, self.ndof)

    # ---- no-penetration (blockage) observable ----
    def surrogate_normal_flux(self, u_free=None):
        """Area-averaged and net normal velocity at the surrogate boundary —
        the physical no-penetration / blockage metric (P2-R0 Task 3 gate).

        Returns ``(mean_un, net_flux, area)`` where
        ``net_flux = oint_sf (u.n_hat) corr dS~`` (area-corrected to the true
        boundary) and ``mean_un = net_flux / area``. Blockage is preserved
        when ``|mean_un| << U_in``. This is a GENUINE physical check computed
        directly from the corrected velocity field (a ``surrogate_traction``-
        style face loop), INDEPENDENT of the PPE/correction BC assembly it
        verifies. ``geo.n`` is domain-outward; the normal-velocity sign is
        immaterial to the blockage magnitude.
        """
        dm = self.dm
        mesh = dm.mesh
        dim = self.dim
        sf, geo = self.sf, self.geo
        if u_free is None:
            u_free = self.base._uvec(self.base.hist.pre1)
        u_full = np.asarray(dm.constraints.T @ u_free)      # [n_nodes, dim]
        pv = int(np.unique(np.asarray(mesh.p_elem)[sf.elem])[0])
        ftab = face_tables(pv, dim)
        nqf = ftab.nqf
        conn = mesh.conn_of[pv][np.searchsorted(mesh.bins[pv], sf.elem)]
        h = mesh.tree.h()[sf.elem]
        jacS = (h / 2.0) ** (dim - 1)
        net, area = 0.0, 0.0
        for fi in range(len(sf.elem)):
            f = int(sf.face[fi])
            un = u_full[conn[fi]]
            for q in range(nqf):
                w = ftab.w[q] * jacS[fi] * geo.corr[fi * nqf + q]
                n = geo.n[fi * nqf + q]
                net += w * ((ftab.N[f][q] @ un) @ n)
                area += w
        return net / area, net, area


class LeraySBMShellStepper:
    """Two-sided co-dim-1 thin-shell SBM projection stepper.

    Mirrors ``LeraySBMStepper`` but drives the immersed body with the
    TWO-SIDED shifted-Nitsche assembly (``sbm_vector_dirichlet_twosided``,
    both Γ̃+ and Γ̃−) instead of the one-sided ``sbm_vector_dirichlet``.

    The caller pre-builds the two-sided surrogate from
    ``extract_two_sided_surrogate`` and passes the four geometry objects
    ``(sf_plus, geo_plus, sf_minus, geo_minus)`` directly — the per-epoch
    pipeline (classify + extract) has already run. The PPE sub-solve uses
    ``ppe_solver=`` (default ``"gpu_cg"``, the SPD Jacobi-CG on device)
    which is the scalable lever for L9-near-plate scale (~186k nodes).

    Unlike ``LeraySBMStepper``, there are NO ``lam`` / ``domain`` parameters:
    the shell surrogate geometry is pre-built by the caller (there is no
    single domain side for a co-dim-1 shell — both sides are fluid, oriented
    per-face in ``mode="shell"``), so classification/extraction do not run
    here.

    3-D DRAG STATUS (2026-07-25, outflow-BC p′-scheme fix): with the
    incremental van-Kan p′-scheme outflow BCs wired in — the WHOLE outflow-face
    p′=0 Dirichlet (``pressure_outflow_nodes`` = the outlet plane node set,
    NOT a single corner pin) + ``consistent_projection=True`` +
    ``inner_iterate=True`` + ``rotational_pin_wall=True`` — the 3-D Cd is now
    POSITIVE and NON-DIVERGING, tracking the monolithic startup transient in
    sign and shape. See ``tests/p2r1c_thin_plate_flow_3d_projection.py`` (the
    driver wires exactly this set) and
    ``docs/dev/2026-07-25-p2r1c-projection-shell-gpu-cg-runbook.md``.

    ⚠️ RESIDUAL GAP: the split's CONVERGED fixed-point Cd is still ~40% below
    the monolithic on the same mesh (plate-surface pressure jump right-signed
    but too small); this does NOT close under harder inner iteration — a
    structural difference between the split's fixed point and the monolithic
    SBM-NS saddle for the two-sided immersed shell (the deeper
    ``p2-r2a-monolithic-pivot`` defect, NOT the outflow-BC wiring gap this fix
    closed). The monolithic SBM-NS saddle
    (``tests/p2r1c_thin_plate_flow_3d.py``) remains the quantitatively faithful
    3-D engine; this stepper delivers the correct-signed, SCALABLE (gpu_cg PPE)
    drag whose magnitude is ~40% low.

    PREDICTOR sub-solve: still ``solver=`` (default ``"splu"``,
    nonsymmetric Oseen, not SPD; ``gpu_cg`` is SPD-only). At L4-L6 smoke
    scales splu is fine. For near-L9 scales the predictor becomes the
    bottleneck — FGMRES (device #49 or AMGX) is the follow-on work.

    PPE sub-solve: ``ppe_solver="gpu_cg"`` routes through
    ``solve_linear(..., sym=True)`` -> ``dist_cg.pcg + SerialComm``
    (torch.sparse_csr_tensor CG on device, Jacobi preconditioned). No
    factorization; no matrix caching. The convergence info is logged when
    ``verbose=True``.

    Two-sided coupling in the PPE: the surrogate-consistent PPE flux
    (T6 or homogeneous Neumann) is applied on BOTH sides by summing the
    two side contributions (same loop structure as the one-sided T6 but
    iterating over both ``(sf_plus, geo_plus)`` and ``(sf_minus, geo_minus)``).
    The homogeneous-Neumann default leaves the natural BC on both sides
    — identical to the one-sided case.

    All one-sided public methods and behaviours are preserved:
    - ``set_initial``, ``step``, ``divergence_l2``, ``t``.
    - ``surrogate_traction``: sums Fp + Fm (both sides).
    - ``surrogate_normal_flux``: sums net flux over both sides.
    """

    def __init__(self, sf_plus, geo_plus, sf_minus, geo_minus,
                 dm, nu, dt, f_fn, *, u_inf, strong_mask,
                 order=2, picard_iters=2,
                 solver="splu", ppe_solver="gpu_cg",
                 ppe_finescale=False, alpha=10.0,
                 beta_backflow=1.0, velocity_update="consistent",
                 graddiv_scale=1.0, pressure_update="standard",
                 ppe_fine_scale=False, pressure_outflow_nodes=None,
                 sbm_pressure_coupling=False,
                 inner_iterate=False, inner_max=8, inner_tol=1e-6,
                 inner_relax=1.0, inner_accel="none", inner_anderson_m=3,
                 consistent_ppe=False, consistent_projection=False,
                 rotational_pin_wall=False,
                 device_assembly=False,
                 verbose=False):
        # ``rotational_pin_wall`` (FN4, 2026-07-23) — pin the rotational
        # -nu*q correction to 0 on the IMMERSED SHELL (sbm) nodes. At a weak
        # Nitsche wall the predictor carries O(1) penetration divergence in the
        # wall cells, so the Timmermans -nu*q term writes a refinement-growing
        # pressure error onto the plate nodes the drag integral reads; pinning
        # q=0 there is the documented immersed-body drag fix (leray.py FN4
        # note). The base ``step`` already pins q on ``sbm_nodes`` (the shell
        # face nodes this stepper passes) when the flag is on. Only meaningful
        # with a rotational pressure update (``consistent_projection`` or
        # ``pressure_update="rotational"``). Default False => bit-for-bit.
        self.sf_plus = sf_plus
        self.geo_plus = geo_plus
        self.sf_minus = sf_minus
        self.geo_minus = geo_minus
        self.dm = dm
        self.nu = nu
        self.dt = dt
        self.dim = dm.dim
        self.ndof = dm.dim + 1
        self.alpha = alpha
        self.beta_backflow = beta_backflow
        self.sbm_pressure_coupling = bool(sbm_pressure_coupling)
        self.verbose = verbose

        u_inf = np.asarray(u_inf, dtype=np.float64)
        strong_mask = np.asarray(strong_mask, dtype=bool)
        self.strong_mask = strong_mask
        self.u_inf = u_inf
        self._strong_nodes = np.where(strong_mask)[0]

        # Base projection stepper — predictor uses solver, PPE uses ppe_solver.
        base = LerayProjectionStepper(
            dm, nu, dt, f_fn, self._g_box, order=order,
            picard_iters=picard_iters, solver=solver, ppe_solver=ppe_solver,
            ppe_finescale=ppe_finescale,
            velocity_update=velocity_update, graddiv_scale=graddiv_scale,
            pressure_update=pressure_update, ppe_fine_scale=ppe_fine_scale,
            pressure_outflow_nodes=pressure_outflow_nodes,
            inner_iterate=inner_iterate, inner_max=inner_max,
            inner_tol=inner_tol, inner_relax=inner_relax,
            inner_accel=inner_accel, inner_anderson_m=inner_anderson_m,
            consistent_ppe=consistent_ppe,
            consistent_projection=consistent_projection,
            rotational_pin_wall=rotational_pin_wall,
            device_assembly=device_assembly)
        base.dir_nodes = self._strong_nodes
        self.base = base
        self.n_free = base.n_free

        # Constraint prolongation matrices
        T = dm.constraints.T.tocsr()
        self._T_vec = sp.kron(T, sp.identity(self.ndof, format="csr"),
                              format="csr")
        self._g_body = lambda y: np.zeros((len(y), self.dim))  # no-slip

        # TWO-SIDED geometry block assembled ONCE (sum of + and - sides)
        Af_raw, bf_raw = sbm_vector_dirichlet_twosided(
            dm, sf_plus, geo_plus, sf_minus, geo_minus,
            self._g_body, nu, self.ndof, alpha=alpha,
            a_face_plus=None, a_face_minus=None,
            beta_backflow=beta_backflow)
        self.Af_c = (self._T_vec.T @ Af_raw @ self._T_vec).tocsr()
        self.bf_c = np.asarray(self._T_vec.T @ bf_raw)

        # Per-step backflow GP tables for BOTH sides
        self._bf_setup_twosided()

        # SBM-governed free nodes: union of both sides' surrogate-face nodes
        self._sbm_nodes = self._compute_sbm_nodes_twosided()

    def _g_box(self, coords_at_dir, t):
        return self.u_inf[self._strong_nodes]

    def _compute_sbm_nodes_twosided(self):
        """Union of surrogate-face nodes from both shell sides."""
        dm = self.dm
        mesh = dm.mesh

        def _face_nodes(sf):
            if sf.elem.size == 0:
                return np.array([], dtype=np.int64)
            conn = mesh.conn_of[1][np.searchsorted(mesh.bins[1], sf.elem)]
            return np.unique(conn.ravel())

        glob_p = _face_nodes(self.sf_plus)
        glob_m = _face_nodes(self.sf_minus)
        glob = np.unique(np.concatenate([glob_p, glob_m]))
        # Map to free-node indices
        free_idx = dm.constraints.free_nodes
        free_of = np.full(dm.n_nodes, -1, dtype=np.int64)
        free_of[free_idx] = np.arange(len(free_idx))
        fn = free_of[glob]
        return fn[fn >= 0]

    def _bf_setup_twosided(self):
        """Cache surrogate-face GP tables for BOTH sides for backflow."""
        dm = self.dm
        mesh = dm.mesh

        def _setup_side(sf):
            if sf.elem.size == 0:
                return 1, face_tables(1, self.dim), np.empty((0,), dtype=np.int64)
            pv = int(np.unique(np.asarray(mesh.p_elem)[sf.elem])[0])
            ftab = face_tables(pv, self.dim)
            conn = mesh.conn_of[pv][np.searchsorted(mesh.bins[pv], sf.elem)]
            return pv, ftab, conn

        pv_p, ftab_p, conn_p = _setup_side(self.sf_plus)
        _pv_m, _ftab_m, conn_m = _setup_side(self.sf_minus)
        # Both sides share the same polynomial order on octree P1 meshes
        self._bf_pv = pv_p
        self._bf_ftab = ftab_p
        self._bf_conn_p = conn_p
        self._bf_conn_m = conn_m

    def _a_face_side(self, u_free, sf, conn):
        """Advecting field at the surrogate-face GPs for one side."""
        dm = self.dm
        u_full = np.asarray(dm.constraints.T @ u_free)
        ftab = self._bf_ftab
        nqf = ftab.nqf
        af = np.empty((len(sf.elem) * nqf, self.dim))
        for fi in range(len(sf.elem)):
            f = int(sf.face[fi])
            un = u_full[conn[fi]]
            for q in range(nqf):
                af[fi * nqf + q] = ftab.N[f][q] @ un
        return af

    def _backflow_block_twosided(self, u_free):
        """Per-step backflow matrix increment for the two-sided shell.

        Calls ``sbm_vector_dirichlet_twosided`` with the current advecting
        field on each side and differences against the cached geometry-only
        block (the Task-2 caching invariant from the one-sided path).
        """
        if self.beta_backflow == 0.0:
            return None
        af_p = self._a_face_side(u_free, self.sf_plus, self._bf_conn_p)
        af_m = self._a_face_side(u_free, self.sf_minus, self._bf_conn_m)
        if not np.any(af_p) and not np.any(af_m):
            return None
        Af_bf_raw, _ = sbm_vector_dirichlet_twosided(
            self.dm, self.sf_plus, self.geo_plus,
            self.sf_minus, self.geo_minus,
            self._g_body, self.nu, self.ndof, alpha=self.alpha,
            a_face_plus=af_p, a_face_minus=af_m,
            beta_backflow=self.beta_backflow)
        Ab = ((self._T_vec.T @ Af_bf_raw @ self._T_vec).tocsr()
              - self.Af_c)
        return Ab

    def _extra_block(self, u_free):
        """Predictor SBM extra-block: cached geometry block + backflow."""
        Ab = self._backflow_block_twosided(u_free)
        A = self.Af_c if Ab is None else (self.Af_c + Ab).tocsr()
        b = self.bf_c
        return (A, b)

    def _t6_nopenetration_flux_twosided(self, uhat):
        """T6 surrogate-consistent PPE flux summed over BOTH shell sides.

        Mirrors ``LeraySBMStepper._t6_nopenetration_flux`` but iterates
        over (sf_plus, geo_plus) then (sf_minus, geo_minus) and sums the
        two contributions. The sign convention is identical: n_hat = -geo.n
        (outward from the fluid, INTO the shell) for each side, and the
        shifted normal velocity n_hat . u_tilde drives the continuity
        constraint.
        """
        dm = self.dm
        mesh = dm.mesh
        dim = self.dim
        ftab = self._bf_ftab
        nqf = ftab.nqf
        h_p = mesh.tree.h()[self.sf_plus.elem]
        h_m = mesh.tree.h()[self.sf_minus.elem]
        dscale_p = 2.0 / h_p
        dscale_m = 2.0 / h_m
        from ..solvers.timestepping import bdf_coeffs, bdf_order_now
        b0, _b1, _b2 = bdf_coeffs(
            bdf_order_now(self.base.t + self.dt, self.dt, self.base.order,
                          have_history=self.base.hist.have(2)), self.dt)
        sigma = b0 / self.dt
        u_full = np.asarray(dm.constraints.T @ uhat)
        rhs = np.zeros(dm.n_nodes)

        def _side(sf, geo, conn, h, dscale):
            jacS = (h / 2.0) ** (dim - 1)
            dvec = geo.d.reshape(len(sf.elem), nqf, dim)
            for fi in range(len(sf.elem)):
                f = int(sf.face[fi])
                un = u_full[conn[fi]]
                for q in range(nqf):
                    w = ftab.w[q] * jacS[fi] * geo.corr[fi * nqf + q]
                    n_hat = -geo.n[fi * nqf + q]
                    Nq = ftab.N[f][q]
                    gradu = (ftab.dN[f][q] * dscale[fi]).T @ un
                    u_gp = Nq @ un
                    u_tilde = u_gp + gradu @ dvec[fi, q]
                    un_tilde = u_tilde @ n_hat
                    rhs[conn[fi]] += Nq * (sigma * w * un_tilde)

        _side(self.sf_plus, self.geo_plus, self._bf_conn_p, h_p, dscale_p)
        _side(self.sf_minus, self.geo_minus, self._bf_conn_m, h_m, dscale_m)
        return rhs

    # ---- public ergonomics ----
    def set_initial(self, u0_fn):
        self.base.set_initial(u0_fn)

    def divergence_l2(self):
        return self.base.divergence_l2()

    @property
    def t(self):
        return self.base.t

    def _current_a_free(self):
        if self.base.hist.pre1 is None:
            return np.zeros((self.n_free, self.dim))
        return self.base._uvec(self.base.hist.pre1)

    def _predict(self, return_matrix=False):
        return self.base._predict(
            extra_block=self._extra_block(self._current_a_free()),
            sbm_nodes=self._sbm_nodes, return_matrix=return_matrix)

    def step(self, surrogate_consistent=True):
        """One projection step with the two-sided shell PPE + correction.

        PPE surrogate BC: same homogeneous-Neumann convention as the
        one-sided stepper (Suresh Remark 3.9). When ``sbm_pressure_coupling``
        is True, T6 is applied on BOTH shell sides (see
        ``_t6_nopenetration_flux_twosided``).
        """
        if not surrogate_consistent:
            flux = self._ppe_break_flux_twosided
        elif self.sbm_pressure_coupling:
            flux = self._t6_nopenetration_flux_twosided
        else:
            flux = None  # homogeneous Neumann (correct default)
        return self.base.step(
            extra_block=self._extra_block(self._current_a_free()),
            sbm_nodes=self._sbm_nodes, ppe_surrogate_flux=flux)

    def _ppe_break_flux_twosided(self, uhat):
        """Planted-break flux summed over both shell sides (anti-vacuity)."""
        dm = self.dm
        mesh = dm.mesh
        dim = self.dim
        ftab = self._bf_ftab
        nqf = ftab.nqf
        from ..solvers.timestepping import bdf_coeffs, bdf_order_now
        b0, _b1, _b2 = bdf_coeffs(
            bdf_order_now(self.base.t + self.dt, self.dt, self.base.order,
                          have_history=self.base.hist.have(2)), self.dt)
        sigma = b0 / self.dt
        u_full = np.asarray(dm.constraints.T @ uhat)
        rhs = np.zeros(dm.n_nodes)

        def _side(sf, geo, conn):
            h = mesh.tree.h()[sf.elem]
            jacS = (h / 2.0) ** (dim - 1)
            for fi in range(len(sf.elem)):
                f = int(sf.face[fi])
                un = u_full[conn[fi]]
                for q in range(nqf):
                    w = ftab.w[q] * jacS[fi] * geo.corr[fi * nqf + q]
                    n = geo.n[fi * nqf + q]
                    uq_n = (ftab.N[f][q] @ un) @ n
                    rhs[conn[fi]] += ftab.N[f][q] * (sigma * w * uq_n)

        _side(self.sf_plus, self.geo_plus, self._bf_conn_p)
        _side(self.sf_minus, self.geo_minus, self._bf_conn_m)
        return rhs

    def surrogate_traction(self):
        """Two-sided traction: Fp + Fm."""
        if self.base.hist.pre1 is None:
            return np.zeros(self.dim)
        u = self.base._uvec(self.base.hist.pre1)
        xfree = np.zeros(self.n_free * self.ndof)
        xv = xfree.reshape(self.n_free, self.ndof)
        xv[:, :self.dim] = u
        xv[:, self.dim] = self.base.p_star
        x_full = np.asarray(self._T_vec @ xfree)
        Fp = surrogate_traction(self.dm, self.sf_plus, self.geo_plus,
                                x_full, self.nu, self.ndof)
        Fm = surrogate_traction(self.dm, self.sf_minus, self.geo_minus,
                                x_full, self.nu, self.ndof)
        return Fp + Fm

    def surrogate_normal_flux(self, u_free=None):
        """Area-averaged normal flux summed over both shell sides."""
        dm = self.dm
        mesh = dm.mesh
        dim = self.dim
        ftab = self._bf_ftab
        nqf = ftab.nqf
        if u_free is None:
            u_free = self.base._uvec(self.base.hist.pre1)
        u_full = np.asarray(dm.constraints.T @ u_free)

        def _flux(sf, geo, conn):
            h = mesh.tree.h()[sf.elem]
            jacS = (h / 2.0) ** (dim - 1)
            net, area = 0.0, 0.0
            for fi in range(len(sf.elem)):
                f = int(sf.face[fi])
                un = u_full[conn[fi]]
                for q in range(nqf):
                    w = ftab.w[q] * jacS[fi] * geo.corr[fi * nqf + q]
                    n = geo.n[fi * nqf + q]
                    net += w * ((ftab.N[f][q] @ un) @ n)
                    area += w
            return net, area

        net_p, area_p = _flux(self.sf_plus, self.geo_plus, self._bf_conn_p)
        net_m, area_m = _flux(self.sf_minus, self.geo_minus, self._bf_conn_m)
        net = net_p + net_m
        area = area_p + area_m
        mean_un = net / area if area > 0 else 0.0
        return mean_un, net, area
