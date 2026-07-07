"""M1c: checkpointed transient adjoint for the linearized s=1/2 stepper.

For J = sum_n j(x_n) over an N-step BDF run, the reverse sweep solves

    A_n^T lam_n = dj/dx_n  -  sum_{m>n} (dR_m/dx_n)^T lam_m

where x_n enters step m through (i) the extrapolated advecting field
a_m = c1 u_{m-1} + c2 u_{m-2} (volume A(aq,dq) + load tau(aq) chains, both
TAPED) and (ii) the BDF history load fq_m = f - (b1 u_{m-1} + b2 u_{m-2})/dt
(taped load fq chain). Every cotangent is exact — no FD anywhere.

    dJ/dnu = sum_n [ vol_dnu_n + load_dnu_n ]

Gate scope: finescale_extrap=False; per-step BDF order recorded (step 1
is BDF1/extrap-1). Checkpoint-everything memory model (2-D scale)."""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu


def _dsolve(A_csc, b):
    """cuDSS-first direct solve (M2-D: L5-3D makes splu ~600s/solve);
    exact like splu — transient gates unchanged."""
    try:
        from ..solvers.linsolve import solve_linear
        return solve_linear(A_csc.tocsr(), b, solver="cudss")
    except Exception:
        return splu(A_csc).solve(b)

from .ns_adjoint import ns_volume_cotangents, ns_load_cotangents
from .ns_shape import _gp_field_transpose


def _interp_N_transpose(dm, cot_gp_by_bin, ndof):
    """Transpose of velocity GP interpolation (N only) -> full node-major."""
    dim = dm.dim
    out = np.zeros(dm.n_nodes * ndof)
    ov = out.reshape(dm.n_nodes, ndof)
    for pv in dm.bins:
        tb = dm.tables_by_p[pv]
        conn = dm.mesh.conn_of[pv]
        ne, nqp = len(conn), tb.nqp
        cot = cot_gp_by_bin[pv].reshape(ne, nqp, dim)
        contrib = np.einsum("qa,eqd->ead", tb.N, cot)
        np.add.at(ov[:, :dim], conn.reshape(-1), contrib.reshape(-1, dim))
    return out


class TransientAdjoint:
    """Record an N-step linearized run, then reverse-sweep dJ/dnu."""

    def __init__(self, st):
        self.st = st
        self.steps = []          # per-step frozen record

    def run(self, n_steps):
        st = self.st
        from ..solvers.timestepping import bdf_order_now, bdf_coeffs
        from ..steppers.linearized import extrapolate_velocity
        xs = []
        for _ in range(n_steps):
            t_new = st.t + st.dt
            o = bdf_order_now(t_new, st.dt, st.order,
                              have_history=st.hist.have(2))
            b0, b1, b2 = bdf_coeffs(o, st.dt)
            u1 = st._node_field(st.hist.pre1)
            u2 = (st._node_field(st.hist.pre2) if st.hist.have(2) else None)
            eo = min(o, 2 if u2 is not None else 1)
            c1, c2 = (2.0, -1.0) if eo == 2 else (1.0, 0.0)
            a_node = extrapolate_velocity(eo, u1, u2)
            aq, dq = st._gp_eval(a_node)
            h_node = b1 * u1 + (b2 * u2 if (b2 != 0.0 and u2 is not None)
                                else 0.0)
            hq, _ = st._gp_eval(np.asarray(h_node) / st.dt)
            fq = {pv: st.f_fn(st.xq[pv], t_new) - hq[pv] for pv in st.xq}
            x = st.step()                              # the actual solve
            rec = dict(aq=aq, dq=dq, fq=fq, sigma=b0 / st.dt,
                       b1=b1, b2=b2, c1=c1, c2=c2,
                       x_free=x.reshape(-1).copy())
            self.steps.append(rec)
            xs.append(rec["x_free"])
        return xs

    def _rebuild_A(self, rec):
        st = self.st
        dm = st.dm
        from ..api.ns_bricks import assemble_linear_ns
        A, _ = assemble_linear_ns(
            dm, rec["aq"], rec["dq"], rec["fq"], st.nu, sigma=rec["sigma"],
            sig2tau=((2.0 * rec["sigma"]) ** 2 if st.timestab else 0.0),
            s_skew=st.s_skew)
        A = A.tolil()
        for r in st.dir_rows:
            A.rows[r] = [int(r)]
            A.data[r] = [1.0]
        A.rows[st.pin_row] = [st.pin_row]
        A.data[st.pin_row] = [1.0]
        return A.tocsr()

    def nu_gradient(self, dJdx_list):
        """dJ/dnu; dJdx_list[n] = dJ/d(x_free of step n) (flat)."""
        st = self.st
        dm = st.dm
        dim = dm.dim
        ndof = st.ndof
        N = len(self.steps)
        T = dm.constraints.T.tocsr()
        T_vec = sp.kron(T, sp.identity(ndof, format="csr"), format="csr")

        # pending[k] = list of node-major FULL cotangent vectors addressed
        # to step k's x (from later steps' chains), FREE layout
        pending = [[] for _ in range(N)]
        dnu_total = 0.0
        for n in range(N - 1, -1, -1):
            rec = self.steps[n]
            rhs = np.asarray(dJdx_list[n], np.float64).copy()
            for extra in pending[n]:
                rhs += extra
            A = self._rebuild_A(rec)
            lam = _dsolve(A.tocsc().T, rhs)
            lam[st.dir_rows] = 0.0
            lam[st.pin_row] = 0.0
            lam_full = np.asarray(T_vec @ lam)
            x_full = np.asarray(T_vec @ rec["x_free"])

            # taped sweeps for THIS step
            (aqb_v, dnu_v) = ns_volume_cotangents(
                dm, rec["aq"], rec["dq"], st.nu, rec["sigma"], st.s_skew,
                x_full, lam_full, timestab=st.timestab,
                tau_frozen=False)
            aqb_l, fqb_l, dnu_l = ns_load_cotangents(
                dm, rec["aq"], rec["fq"], st.nu, rec["sigma"], lam_full,
                timestab=st.timestab)
            dnu_total += dnu_v + dnu_l

            # chains to EARLIER steps (x_{n-1}, x_{n-2}). Algebra:
            # lam^T dR/daq = lam^T d(Ax)/daq - lam^T db/daq
            #             = (-aqb_v) - aqb_l          [returns' conventions]
            # rhs_k += -(dR/dx_k)^T lam
            #        = c_k * interp^T( -(lam^T dR/daq) ) + ...
            #        = c_k * interp^T( aqb_v + aqb_l )  -- so G uses the
            # RAW returned cotangents summed (their built-in signs already
            # encode the double negation); dq analogously (volume only).
            G = _gp_field_transpose(
                dm, {pv: aqb_v[pv][0] + aqb_l[pv] for pv in dm.bins},
                {pv: aqb_v[pv][1] for pv in dm.bins}, ndof)
            # fq chain: lam^T dR/dfq = -fq_bar; dfq/dx_k = -(b_k/dt) interp
            # -> rhs_k += -(b_k/dt) * interp^T(fq_bar)
            Ffq = _interp_N_transpose(
                dm, {pv: fqb_l[pv] for pv in dm.bins}, ndof)
            for k_off, c_ext, b_hist in ((1, rec["c1"], rec["b1"]),
                                         (2, rec["c2"], rec["b2"])):
                kn = n - k_off
                if kn < 0:
                    continue
                contrib_full = (c_ext * G
                                - (b_hist / st.dt) * Ffq)
                pending[kn].append(np.asarray(T_vec.T @ contrib_full))
        return dnu_total


class TransientShapeAdjoint(TransientAdjoint):
    """Transient adjoint with an immersed SBM obstacle: the face block is
    CONSTANT across the run (geometry frozen per epoch), each reverse-step
    lambda contributes face d-cotangents, accumulated over the sweep and
    composed through the oracle's torch chain once.

    Scope (gate 1): state-only QoIs (no explicit geometry term in J —
    drag-weighted objectives add their traction_torch chain at the hero).
    Forward runner: strouhal-pattern composition (assemble_linear_ns +
    constant face block + strong rows) rather than the plain stepper."""

    def __init__(self, dm, sf, geo, oracle, nu, dt, alpha_face,
                 strong_nodes, g_strong, order=2):
        self.dm = dm
        self.sf = sf
        self.geo = geo
        self.oracle = oracle
        self.nu = nu
        self.dt = dt
        self.alpha_face = alpha_face
        self.order = order
        self.steps = []
        dim = dm.dim
        self.ndof = dim + 1
        T = dm.constraints.T.tocsr()
        self.T_vec = sp.kron(T, sp.identity(self.ndof, format="csr"),
                             format="csr")
        self.n_free = T.shape[1]
        self.strong_nodes = strong_nodes
        self.g_strong = g_strong
        from .vector import sbm_vector_dirichlet
        Af, bf = sbm_vector_dirichlet(
            dm, sf, geo, lambda y: np.zeros((len(y), dim)), nu, self.ndof,
            alpha=alpha_face)
        self.Af_c = (self.T_vec.T @ Af @ self.T_vec).tocsr()
        self.bf_c = np.asarray(self.T_vec.T @ bf)
        coords = dm.mesh.node_coords[dm.constraints.free_nodes]
        self.pin = int(np.argmax(coords[:, 0] + coords[:, 1]))
        self._x_hist = []

    def _gp_field(self, node_vec):
        dm = self.dm
        dim = dm.dim
        T = dm.constraints.T.tocsr()
        full = np.asarray(T @ node_vec)
        aq, dq = {}, {}
        for pv in dm.bins:
            tb = dm.tables_by_p[pv]
            conn = dm.mesh.conn_of[pv]
            vals = full[conn]
            aq[pv] = np.einsum("qa,ead->eqd", tb.N, vals).reshape(-1, dim)
            h = dm.mesh.tree.h()[dm.mesh.bins[pv]]
            dq[pv] = (np.einsum("qad,ead->eq", tb.dN, vals)
                      * (2.0 / h)[:, None]).reshape(-1)
        return aq, dq

    def _assemble_step(self, aq, dq, fq, sigma):
        from ..api.ns_bricks import assemble_linear_ns
        dm = self.dm
        dim = dm.dim
        ndof = self.ndof
        A, b = assemble_linear_ns(dm, aq, dq, fq, self.nu, sigma=sigma,
                                  sig2tau=(2.0 * sigma) ** 2)
        A = (A + self.Af_c).tolil()
        b = b + self.bf_c
        for k_, i in enumerate(self.strong_nodes):
            for c in range(dim):
                r_ = i * ndof + c
                A.rows[r_] = [int(r_)]
                A.data[r_] = [1.0]
                b[r_] = self.g_strong[k_, c]
        rp = self.pin * ndof + dim
        A.rows[rp] = [rp]
        A.data[rp] = [1.0]
        b[rp] = 0.0
        return A.tocsr(), b

    def run(self, n_steps):
        from scipy.sparse.linalg import splu
        from ..solvers.timestepping import bdf_order_now, bdf_coeffs
        dm = self.dm
        dim = dm.dim
        ndof = self.ndof
        x = np.zeros(self.n_free * ndof)
        x1 = x2 = None
        t = 0.0
        xs = []
        for n in range(n_steps):
            t += self.dt
            o = bdf_order_now(t, self.dt, self.order,
                              have_history=x2 is not None)
            b0, b1, b2 = bdf_coeffs(o, self.dt)
            sigma = b0 / self.dt
            u1 = (x1 if x1 is not None else x).reshape(
                self.n_free, ndof)[:, :dim]
            u2 = (x2.reshape(self.n_free, ndof)[:, :dim]
                  if x2 is not None else None)
            eo = 2 if (o >= 2 and u2 is not None) else 1
            c1, c2 = (2.0, -1.0) if eo == 2 else (1.0, 0.0)
            a_node = c1 * u1 + (c2 * u2 if u2 is not None else 0.0)
            aq, dq = self._gp_field(a_node)
            h_node = b1 * u1 + (b2 * u2 if (b2 != 0.0 and u2 is not None)
                                else 0.0)
            hq, _ = self._gp_field(np.asarray(h_node) / self.dt)
            fq = {pv: -hq[pv] for pv in hq}
            A, b = self._assemble_step(aq, dq, fq, sigma)
            x2 = x1
            x1 = _dsolve(A.tocsc(), b)
            rec = dict(aq=aq, dq=dq, fq=fq, sigma=sigma, b1=b1, b2=b2,
                       c1=c1, c2=c2, x_free=x1.copy())
            self.steps.append(rec)
            xs.append(x1.copy())
        return xs

    def _rebuild_A(self, rec):
        A, _ = self._assemble_step(rec["aq"], rec["dq"], rec["fq"],
                                   rec["sigma"])
        return A

    def shape_gradient(self, dJdx_list):
        """Accumulate face d-cotangents over the reverse sweep; compose
        through the oracle torch chain. Returns via oracle.params[i].grad."""
        import torch
        import warp as wp
        from scipy.sparse.linalg import splu
        from .ns_shape import face_dbar_sweep
        from .adjoint import distance_torch
        dm = self.dm
        dim = dm.dim
        ndof = self.ndof
        N = len(self.steps)
        strong_rows = np.concatenate(
            [np.array([i * ndof + c for i in self.strong_nodes
                       for c in range(dim)], np.int64),
             np.array([self.pin * ndof + dim], np.int64)])
        pending = [[] for _ in range(N)]
        dbar_total = np.zeros_like(self.geo.d)
        from .ns_adjoint import ns_volume_cotangents, ns_load_cotangents
        from .ns_shape import _gp_field_transpose
        from .transient_adjoint import _interp_N_transpose
        for n in range(N - 1, -1, -1):
            rec = self.steps[n]
            rhs = np.asarray(dJdx_list[n], np.float64).copy()
            for extra in pending[n]:
                rhs += extra
            A = self._rebuild_A(rec)
            lam = _dsolve(A.tocsc().T, rhs)
            lam[strong_rows] = 0.0
            lam_full = np.asarray(self.T_vec @ lam)
            x_full = np.asarray(self.T_vec @ rec["x_free"])
            lam_d = wp.array(lam_full, dtype=wp.float64, device=dm.device)
            x_d = wp.array(np.ascontiguousarray(x_full), dtype=wp.float64,
                           device=dm.device)
            # face d-cotangents for THIS step (accumulate)
            dbar_total += face_dbar_sweep(dm, self.sf, self.geo, x_d,
                                          lam_d, self.nu, self.alpha_face,
                                          ndof)
            # volume chains to earlier steps (transient-proven algebra)
            aqb_v, _ = ns_volume_cotangents(
                dm, rec["aq"], rec["dq"], self.nu, rec["sigma"], 0.5,
                x_full, lam_full, timestab=True, tau_frozen=False)
            aqb_l, fqb_l, _ = ns_load_cotangents(
                dm, rec["aq"], rec["fq"], self.nu, rec["sigma"], lam_full,
                timestab=True)
            G = _gp_field_transpose(
                dm, {pv: aqb_v[pv][0] + aqb_l[pv] for pv in dm.bins},
                {pv: aqb_v[pv][1] for pv in dm.bins}, ndof)
            Ffq = _interp_N_transpose(
                dm, {pv: fqb_l[pv] for pv in dm.bins}, ndof)
            for k_off, c_ext, b_hist in ((1, rec["c1"], rec["b1"]),
                                         (2, rec["c2"], rec["b2"])):
                kn = n - k_off
                if kn < 0:
                    continue
                contrib_full = c_ext * G - (b_hist / self.dt) * Ffq
                pending[kn].append(np.asarray(self.T_vec.T @ contrib_full))
        # torch composition: theta -> d (no-slip: gbar chain vanishes;
        # state-only QoI: no explicit geometry term)
        for p_ in self.oracle.params:
            p_.requires_grad_(True)
            if p_.grad is not None:
                p_.grad = None
        own = getattr(type(self.oracle), "distance_torch", None)
        d_t, n_t, ok = (self.oracle.distance_torch(self.geo.xq) if own
                        else distance_torch(self.oracle, self.geo.xq))
        loss = -(d_t * torch.tensor(dbar_total,
                                    dtype=torch.float64)).sum()
        loss.backward()
