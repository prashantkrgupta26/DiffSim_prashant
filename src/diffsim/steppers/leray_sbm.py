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

    def step(self):
        return self.base.step(extra_block=(self.Af_c, self.bf_c),
                              sbm_nodes=self._sbm_nodes)

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
