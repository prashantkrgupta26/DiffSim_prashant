"""Linearized monolithic NS stepper (M1b Task 5b) — the Flow-Bench variant
(production-code-conventions.md items 1-6): ONE linear (u,p) solve per step.

Per step tn -> tn+1:
1. BDF order via the t < 1.5dt bootstrap; sigma = b0/dt.
2. Advecting field a = extrapolation (order 1/2) of the previous velocities,
   evaluated at Gauss points; div a computed CONSISTENTLY from the discrete
   field (dN contraction), with the production FINE-SCALE CORRECTION
   (conventions item 1): a = extrapolation of (u_pre - tauM res_M_pre),
   residual explicit from history with the BE time term (item 5);
   finescale_extrap=False recovers the plain extrapolation.
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
                 use_device_assembly=False,
                 timestab=True, s_skew=0.5, finescale_extrap=True,
                 solver="splu"):
        """f_fn(x, t) -> [N, dim] body force; g_fn(x, t) -> [N, dim] boundary
        velocity; p_pin_value_fn(x0, t) -> pin value (default 0)."""
        self.dm, self.nu, self.dt, self.order = dm, nu, dt, order
        # production timeStab toggle: with it ON tau carries (2 b0/dt)^2 and
        # the stabilized SPATIAL problem varies with dt — measured to floor
        # dt-ladder studies at ~2e-3 (temporal-order gates run it OFF)
        self.timestab = timestab
        self.s_skew = s_skew
        # production conventions item 1: the advecting field is the
        # extrapolation of the FINE-SCALE-CORRECTED velocity
        # u_corr = u_pre - tauM res_M_pre with the residual fully explicit
        # (BE time term from PRE levels, conventions item 5). Correction is
        # a Gauss-point field (the fine scale lives at GPs); the discrete
        # divergence for the s-skew term stays that of the COARSE
        # extrapolated field (standard VMS practice).
        self.finescale_extrap = finescale_extrap
        # linear-solve backend: "splu" (host direct) | "fused" (device
        # single-sync BiCGStab) | "amgx" (AMG-preconditioned, GPU)
        self.solver = solver
        self._solver_cache = {}
        self.f_fn, self.g_fn = f_fn, g_fn
        self.p_pin_value_fn = p_pin_value_fn or (lambda x0, t: 0.0)
        self.ndof = dm.dim + 1
        self.hist = History()
        # M1d: device-side assembly + strong rows (4-40x measured); the
        # symbolic pattern + strong-row plan build once here
        self._dev_asm = None
        if use_device_assembly:
            from ..assembly.device_assembly import DeviceNSAssembler
            self._dev_asm = DeviceNSAssembler(dm)
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
        if self._dev_asm is not None:
            self._dev_asm.set_strong_rows(
                np.concatenate([self.dir_rows, [self.pin_row]]))
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

    def _corrected_gp(self, u_a, u_b, p_a, t_a, sig2tau):
        """GP field of (u_a - tauM res_M) with res_M = (u_a-u_b)/dt (BE)
        + u_a.grad(u_a) + grad(p_a) - f(t_a); nu-lap dropped at p1."""
        dm = self.dm
        aq_a, _ = self._gp_eval(u_a)
        aq_b, _ = self._gp_eval(u_b)
        from ..physics.vms import tau_metric_host
        out = {}
        for pv, b in dm.bins.items():
            tb = dm.tables_by_p[pv]
            h = dm.mesh.tree.h()[dm.mesh.bins[pv]]
            nqp = tb.nqp
            he = np.repeat(h, nqp)
            conn = dm.mesh.conn_of[pv]
            full_u = np.asarray(dm.constraints.T @ u_a)
            full_p = np.asarray(dm.constraints.T @ p_a)
            gu = (np.einsum("qad,ea...->eqd...", tb.dN, full_u[conn])
                  * (2.0 / h)[:, None, None, None]).reshape(
                -1, dm.dim, dm.dim)
            gp_ = (np.einsum("qad,ea->eqd", tb.dN, full_p[conn])
                   * (2.0 / h)[:, None, None]).reshape(-1, dm.dim)
            agu = np.einsum("gd,gdc->gc", aq_a[pv], gu)
            res = ((aq_a[pv] - aq_b[pv]) / self.dt + agu + gp_
                   - self.f_fn(self.xq[pv], t_a))
            umag = np.sqrt((aq_a[pv] ** 2).sum(1))
            tau, _ = tau_metric_host(umag, he, self.nu, dt=None, dim=dm.dim)
            if sig2tau > 0.0:
                tau = 1.0 / np.sqrt(sig2tau + 1.0 / tau ** 2)
            out[pv] = aq_a[pv] - tau[:, None] * res
        return out

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
        if self.finescale_extrap and self.hist.have(2):
            sig2tau = (2.0 * sigma) ** 2 if self.timestab else 0.0
            p1_ = self.hist.pre1.reshape(self.n_free, self.ndof)[:, dm.dim]
            c_n = self._corrected_gp(u1, u2, p1_, self.t, sig2tau)
            if self.hist.have(3):
                u3 = self._node_field(self.hist.pre3)
                p2_ = self.hist.pre2.reshape(self.n_free,
                                             self.ndof)[:, dm.dim]
                c_m = self._corrected_gp(u2, u3, p2_, self.t - self.dt,
                                         sig2tau)
                aq = {pv: 2.0 * c_n[pv] - c_m[pv] for pv in aq}
            else:
                aq = c_n
        # history contribution at GPs (part of the strong residual)
        h_node = b1 * u1 + (b2 * u2 if (b2 != 0.0 and u2 is not None)
                            else 0.0)
        hq, _ = self._gp_eval(np.asarray(h_node) / self.dt)
        fq = {pv: self.f_fn(self.xq[pv], t_new) - hq[pv] for pv in self.xq}
        gvals = self.g_fn(self.free_coords[self.dir_nodes], t_new)
        if self._dev_asm is not None:
            sbv = np.concatenate(
                [gvals.reshape(-1),
                 [self.p_pin_value_fn(self.free_coords[0], t_new)]])
            A, b = self._dev_asm.assemble(
                aq, dq, fq, self.nu, sigma,
                sig2tau=((2.0 * sigma) ** 2 if self.timestab else 0.0),
                s_skew=self.s_skew, strong_b_vals=sbv)
            A = A.tocsr()
        else:
            A, b = assemble_linear_ns(dm, aq, dq, fq, self.nu, sigma=sigma,
                                      sig2tau=((2.0 * sigma) ** 2
                                               if self.timestab else 0.0),
                                      s_skew=self.s_skew)
            A = A.tolil()
            for k, r in enumerate(self.dir_rows):
                A.rows[r] = [int(r)]
                A.data[r] = [1.0]
                b[r] = gvals[k // dm.dim, k % dm.dim]
            A.rows[self.pin_row] = [self.pin_row]
            A.data[self.pin_row] = [1.0]
            b[self.pin_row] = self.p_pin_value_fn(self.free_coords[0],
                                                  t_new)
            A = A.tocsr()
        from ..solvers.linsolve import solve_linear
        x = solve_linear(A, b, solver=self.solver, sym=False,
                         device=self.dm.device, cache=self._solver_cache)
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
