"""Linearized monolithic NS stepper (M1b Task 5b) — the Flow-Bench variant
(production-code-conventions.md items 1-6): ONE linear (u,p) solve per step.

Per step tn -> tn+1:
1. BDF order via the t < 1.5dt bootstrap; sigma = b0/dt.
2. Advecting field a = extrapolation (order 1/2) of the previous velocities,
   evaluated at Gauss points; div a computed CONSISTENTLY from the discrete
   field (dN contraction). v1 NOTE: the production fine-scale correction
   (u_pre - tauM res_M_pre before extrapolating) is deferred to the
   benchmark task — it sharpens constants, not convergence order; recorded
   as a TODO tied to conventions item 1.
3. RHS f_eff = f(t_{n+1}) - (b1 u^n + b2 u^{n-1})/dt — the history term is
   part of the linearized strong residual, so it flows through the SUPG/PSPG
   consistency terms too (the be kernel takes the combined field).
4. Assemble (tau frozen at a), apply strong velocity Dirichlet + pressure
   pin, solve, rotate history.

Solver: scipy splu (prototype scale; the matrix-free + fused-Krylov path is
wired at the benchmark task where sizes demand it — Task 1 built it).
"""
import numpy as np
from scipy.sparse.linalg import splu

from ..api.ns_bricks import assemble_linear_ns
from ..physics.poisson import gauss_points
from ..solvers.timestepping import (bdf_coeffs, bdf_order_now,
                                    extrapolate_velocity, History)


class LinearizedMonolithicStepper:
    def __init__(self, dm, nu, dt, f_fn, g_fn, order=2, p_pin_value_fn=None,
                 timestab=True, s_skew=0.5):
        """f_fn(x, t) -> [N, dim] body force; g_fn(x, t) -> [N, dim] boundary
        velocity; p_pin_value_fn(x0, t) -> pin value (default 0)."""
        self.dm, self.nu, self.dt, self.order = dm, nu, dt, order
        # production timeStab toggle: with it ON tau carries (2 b0/dt)^2 and
        # the stabilized SPATIAL problem varies with dt — measured to floor
        # dt-ladder studies at ~2e-3 (temporal-order gates run it OFF)
        self.timestab = timestab
        self.s_skew = s_skew
        self.f_fn, self.g_fn = f_fn, g_fn
        self.p_pin_value_fn = p_pin_value_fn or (lambda x0, t: 0.0)
        self.ndof = dm.dim + 1
        self.hist = History()
        self.t = 0.0
        mesh = dm.mesh
        self.free_coords = mesh.node_coords[dm.constraints.free_nodes]
        self.n_free = len(self.free_coords)
        bdry = mesh.boundary_nodes[dm.constraints.free_nodes]
        rows = []
        for i in np.where(bdry)[0]:
            rows += [i * self.ndof + c for c in range(dm.dim)]
        self.dir_rows = np.array(rows, np.int64)
        self.dir_nodes = np.where(bdry)[0]
        self.pin_row = 0 * self.ndof + dm.dim
        self.xq = gauss_points(mesh, dm.tables_by_p)

    def set_initial(self, u0_fn):
        u = np.zeros((self.n_free, self.ndof))
        u[:, :self.dm.dim] = u0_fn(self.free_coords)
        self.hist.rotate(u.ravel())

    def _node_field(self, flat):
        return flat.reshape(self.n_free, self.ndof)[:, :self.dm.dim]

    def _gp_eval(self, node_vals):
        """Interp node velocity [n_free, dim] -> per-bin GP values + div."""
        dm = self.dm
        full = np.asarray(dm.constraints.T @ node_vals)   # per component
        aq, dq = {}, {}
        for pv, b in dm.bins.items():
            tb = dm.tables_by_p[pv]
            conn = dm.mesh.conn_of[pv]
            vals = full[conn]                              # [ne, nbf, dim]
            aq[pv] = np.einsum("qa,ead->eqd", tb.N, vals).reshape(
                -1, dm.dim)
            h = dm.mesh.tree.h()[dm.mesh.bins[pv]]
            # consistent div: sum_{a,d} dN[q,a,d] * dscale_e * vals[e,a,d]
            dq[pv] = (np.einsum("qad,ead->eq", tb.dN, vals)
                      * (2.0 / h)[:, None]).reshape(-1)
        return aq, dq

    def step(self):
        dm = self.dm
        t_new = self.t + self.dt
        o = bdf_order_now(t_new, self.dt, self.order,
                          have_history=self.hist.have(2))
        b0, b1, b2 = bdf_coeffs(o, self.dt)
        sigma = b0 / self.dt
        u1 = self._node_field(self.hist.pre1)
        u2 = (self._node_field(self.hist.pre2)
              if self.hist.have(2) else None)
        a_node = extrapolate_velocity(min(o, 2 if u2 is not None else 1),
                                      u1, u2)
        aq, dq = self._gp_eval(a_node)
        # history contribution at GPs (part of the strong residual)
        h_node = b1 * u1 + (b2 * u2 if (b2 != 0.0 and u2 is not None)
                            else 0.0)
        hq, _ = self._gp_eval(np.asarray(h_node) / self.dt)
        fq = {pv: self.f_fn(self.xq[pv], t_new) - hq[pv] for pv in self.xq}
        A, b = assemble_linear_ns(dm, aq, dq, fq, self.nu, sigma=sigma,
                                  sig2tau=((2.0 * sigma) ** 2
                                           if self.timestab else 0.0),
                                  s_skew=self.s_skew)
        A = A.tolil()
        gvals = self.g_fn(self.free_coords[self.dir_nodes], t_new)
        for k, r in enumerate(self.dir_rows):
            A.rows[r] = [int(r)]
            A.data[r] = [1.0]
            b[r] = gvals[k // dm.dim, k % dm.dim]
        A.rows[self.pin_row] = [self.pin_row]
        A.data[self.pin_row] = [1.0]
        b[self.pin_row] = self.p_pin_value_fn(self.free_coords[0], t_new)
        x = splu(A.tocsr().tocsc()).solve(b)
        self.hist.rotate(x, dt=self.dt)
        self.t = t_new
        return x.reshape(self.n_free, self.ndof)

    def divergence_l2(self):
        """||div u_h||_L2 sentinel on the current field."""
        _, dq = self._gp_eval(self._node_field(self.hist.pre1))
        dm = self.dm
        tot, vol = 0.0, 0.0
        for pv, b in dm.bins.items():
            tb = dm.tables_by_p[pv]
            h = dm.mesh.tree.h()[dm.mesh.bins[pv]]
            jac = (h / 2.0) ** dm.dim
            w = np.tile(tb.w, len(h)) * np.repeat(jac, tb.nqp)
            tot += (dq[pv] ** 2 * w).sum()
            vol += w.sum()
        return np.sqrt(tot / vol)
