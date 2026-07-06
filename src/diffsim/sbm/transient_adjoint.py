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
            lam = splu(A.tocsc().T).solve(rhs)
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
