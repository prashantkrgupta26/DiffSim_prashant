"""Volumetric-SBM composed Leray projection stepper (P2-R0, Task 2).

`LeraySBMStepper` COMPOSES `LerayProjectionStepper` (the audited VMS-Helmholtz-
Leray projection stepper) with an immersed-geometry oracle — it does NOT fork
the base stepper. It runs the per-epoch surrogate pipeline
(`classify_lambda` -> `extract_surrogate` -> `GeometryData.evaluate`), builds
the geometry-only shifted-Nitsche vector Dirichlet face block ONCE
(`sbm_vector_dirichlet`, no-slip body), and injects it into the base stepper's
predictor sub-solve via the `extra_block`/`sbm_nodes` hook added to
`leray.py::LerayProjectionStepper._predict`.

R0 scope (Task 2): the SBM block is wired into the PREDICTOR only. The
surrogate-consistent PPE + correction boundary (no-penetration preservation)
land in Task 3; the finalized end-to-end driver ergonomics land in Task 4.

The immersed body is enforced WEAKLY (SBM, on the surrogate faces); the box
inflow/walls are enforced STRONGLY from the driver-supplied `strong_mask` +
`u_inf` (the M1b-test convention). `geo.n` is domain-outward; drag orientation
uses `n_hat = -geo.n` inside `surrogate_traction`.
"""
import numpy as np
import scipy.sparse as sp

from ..mesh.faces import face_tables
from ..sbm.surrogate import (classify_lambda, extract_surrogate, GeometryData)
from ..sbm.vector import sbm_vector_dirichlet, surrogate_traction
from ..solvers.timestepping import bdf_coeffs, bdf_order_now
from .leray import LerayProjectionStepper


class LeraySBMStepper:
    def __init__(self, oracle, dm, nu, dt, f_fn, *, u_inf, strong_mask,
                 lam=0.5, domain="outside", order=2, picard_iters=2,
                 solver="splu", ppe_finescale=False, alpha=10.0,
                 beta_backflow=1.0):
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
            ppe_finescale=ppe_finescale)
        base.dir_nodes = self._strong_nodes
        self.base = base
        self.n_free = base.n_free

        # --- geometry-only SBM face block, assembled ONCE ---
        T = dm.constraints.T.tocsr()
        self._T_vec = sp.kron(T, sp.identity(self.ndof, format="csr"),
                              format="csr")
        g_body = lambda y: np.zeros((len(y), self.dim))  # no-slip
        Af, bf = sbm_vector_dirichlet(
            dm, self.sf, self.geo, g_body, nu, self.ndof, alpha=alpha,
            a_face=None, beta_backflow=beta_backflow)
        self.Af_c = (self._T_vec.T @ Af @ self._T_vec).tocsr()
        self.bf_c = np.asarray(self._T_vec.T @ bf)

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

    # ---- public ergonomics (delegate to the base) ----
    def set_initial(self, u0_fn):
        self.base.set_initial(u0_fn)

    def divergence_l2(self):
        return self.base.divergence_l2()

    @property
    def t(self):
        return self.base.t

    # ---- predictor hook (Task 2): SBM block into the momentum sub-solve ----
    def _predict(self, return_matrix=False):
        return self.base._predict(
            extra_block=(self.Af_c, self.bf_c), sbm_nodes=self._sbm_nodes,
            return_matrix=return_matrix)

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
        flux = None if surrogate_consistent else self._ppe_break_flux
        return self.base.step(extra_block=(self.Af_c, self.bf_c),
                              sbm_nodes=self._sbm_nodes,
                              ppe_surrogate_flux=flux)

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
