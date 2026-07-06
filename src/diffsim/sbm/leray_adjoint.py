"""M1c: staged adjoint for ONE Leray projection step (production stepper
#2), first gate = d(QoI)/d(nu) with the step's inputs (history, p_star)
FROZEN — the same partial the FD control reruns.

Scope (gate #1): picard_iters=1 (advecting field = extrapolated history,
frozen), ppe_finescale=False (PPE flux = sigma*u_hat: nu-free). The step
is then a linear three-stage chain

    A(aq_hist, nu) x = b(nu)          (predictor; pressure pinned to p*)
    Kp phi = rhs(uhat)                (constant SPD; pin row 0)
    M u_c  = P_c(uhat, phi)           (mass solves; strong dirs overwrite)

and the adjoint is the reversed transposed chain. nu enters ONLY the
predictor: (dA/dnu) x through the TAPED volume kernel (exact, incl. tau's
nu-dependence — consistency-gated against assemble_linear_ns), and db/dnu
via central FD on the assembled load at frozen aq (interim until the RHS
kernel is taped; the load's nu-dependence is the SUPG tau(nu) weight).
"""
import numpy as np
import scipy.sparse as sp

from ..api.ns_bricks import assemble_linear_ns
from .ns_adjoint import ns_volume_cotangents


def _interp_transpose_scalar(st, cot_gp_by_bin):
    """Transpose of _gp_vals for a scalar field: GP cotangents -> free."""
    dm = st.dm
    out = np.zeros(dm.n_nodes)
    for pv in dm.bins:
        tb = dm.tables_by_p[pv]
        conn = dm.mesh.conn_of[pv]
        ne, nqp = len(conn), tb.nqp
        cot = cot_gp_by_bin[pv].reshape(ne, nqp)
        be = np.einsum("qa,eq->ea", tb.N, cot)
        np.add.at(out, conn.ravel(), be.ravel())
    return np.asarray(dm.constraints.T.T @ out)


def _grad_interp_transpose_scalar(st, cot_gp_by_bin):
    """Transpose of the gradient interpolation for a scalar field."""
    dm = st.dm
    dim = dm.dim
    out = np.zeros(dm.n_nodes)
    for pv in dm.bins:
        tb = dm.tables_by_p[pv]
        conn = dm.mesh.conn_of[pv]
        h = dm.mesh.tree.h()[dm.mesh.bins[pv]]
        ne, nqp = len(conn), tb.nqp
        cot = cot_gp_by_bin[pv].reshape(ne, nqp, dim)
        be = np.einsum("qad,eqd->ea", tb.dN, cot) * (2.0 / h)[:, None]
        np.add.at(out, conn.ravel(), be.ravel())
    return np.asarray(dm.constraints.T.T @ out)


def leray_step_nu_gradient(st, dJdu_new, dJdp=None):
    """dJ/dnu for the LAST executed step of LerayProjectionStepper st
    (requires: picard_iters=1, ppe_finescale=False, and st must have been
    stepped so its per-step operators can be rebuilt from the SAME frozen
    inputs). dJdu_new: [n_free, dim]. Returns float dJ/dnu.

    Rebuilds the step's operators from the stepper's rotated history
    (hist.pre2 is the pre-step state u1 after rotation etc.) — so call it
    IMMEDIATELY after step() with the pre-step history you kept."""
    raise NotImplementedError("use LerayStepAdjoint (explicit-state API)")


class LerayStepAdjoint:
    """Explicit-state single-step adjoint: capture the step's frozen
    inputs BEFORE calling step(), then compute gradients after."""

    def __init__(self, st):
        self.st = st
        # capture frozen inputs (pre-step)
        self.u1 = st._uvec(st.hist.pre1)
        self.u2 = st._uvec(st.hist.pre2) if st.hist.have(2) else None
        self.p_star = st.p_star.copy()
        self.t_new = st.t + st.dt

    def _predictor_chain(self, nu):
        """Replicate the stepper's Picard loop from the frozen inputs,
        recording every iterate: list of dicts(A, b, x, aq, dq, fq)."""
        st = self.st
        dm = st.dm
        dim = dm.dim
        ndof = st.ndof
        from scipy.sparse.linalg import splu
        from ..solvers.timestepping import bdf_order_now, bdf_coeffs
        o = bdf_order_now(self.t_new, st.dt, st.order,
                          have_history=self.u2 is not None)
        b0, b1, b2 = bdf_coeffs(o, st.dt)
        sigma = b0 / st.dt
        h_node = (b1 * self.u1 + (b2 * self.u2 if (b2 != 0.0 and
                                                   self.u2 is not None)
                                  else 0.0)) / st.dt
        hq = st._gp_vals(h_node)
        fq = {pv: st.f_fn(st.xq[pv], self.t_new) - hq[pv] for pv in st.xq}
        gvals = st.g_fn(st.free_coords[st.dir_nodes], self.t_new)
        a_node = (2.0 * self.u1 - self.u2 if self.u2 is not None
                  else self.u1.copy())
        chain = []
        for _ in range(st.picard_iters):
            aq, gaq = st._gp_vals(a_node, grad=True)
            dq = {pv: np.einsum("gdd->g",
                                gaq[pv].reshape(-1, dim, dim)) for pv in aq}
            A, b = assemble_linear_ns(
                dm, aq, dq, fq, nu, sigma=sigma,
                sig2tau=((2.0 * sigma) ** 2 if st.timestab else 0.0))
            A = A.tolil()
            for k, i in enumerate(st.dir_nodes):
                for c in range(dim):
                    r = i * ndof + c
                    A.rows[r] = [int(r)]
                    A.data[r] = [1.0]
                    b[r] = gvals[k, c]
            for i in range(st.n_free):
                r = i * ndof + dim
                A.rows[r] = [int(r)]
                A.data[r] = [1.0]
                b[r] = self.p_star[i]
            A = A.tocsr()
            x = splu(A.tocsc()).solve(b)
            chain.append(dict(A=A, x=x, aq=aq, dq=dq, fq=fq, sigma=sigma))
            a_node = x.reshape(st.n_free, ndof)[:, :dim]
        return chain

    def nu_gradient(self, dJdu_new):
        """dJ/dnu; dJdu_new [n_free, dim]. Supports picard_iters >= 1
        (recorded-iterate composition; inter-iterate chain uses the
        transient-chain-proven convention at tau_frozen=False)."""
        st = self.st
        dm = st.dm
        dim = dm.dim
        ndof = st.ndof
        assert not st.ppe_finescale, "finescale branch: item (c)"
        from scipy.sparse.linalg import splu
        from .ns_adjoint import ns_load_cotangents
        from .ns_shape import _gp_field_transpose
        chain = self._predictor_chain(st.nu)
        last = chain[-1]
        aq, sigma = last["aq"], last["sigma"]
        uhat = last["x"].reshape(st.n_free, ndof)[:, :dim]

        # ---- stage 3 transpose: mass updates + strong overwrite --------
        lam_u = np.asarray(dJdu_new, np.float64).copy()
        lam_u[st.dir_nodes] = 0.0
        M_lu = splu(st.M.tocsc())
        uq_cot = {pv: np.zeros((len(aq[pv]), dim)) for pv in aq}
        phi_cot_gp = {pv: np.zeros((len(aq[pv]), dim)) for pv in aq}
        for c in range(dim):
            lam_Mc = M_lu.solve(lam_u[:, c])
            lam_full = np.asarray(dm.constraints.T @ lam_Mc)
            for pv in dm.bins:
                tb = dm.tables_by_p[pv]
                conn = dm.mesh.conn_of[pv]
                h = dm.mesh.tree.h()[dm.mesh.bins[pv]]
                jac = (h / 2.0) ** dim
                lam_e = lam_full[conn]
                w_e = np.einsum("qa,ea,q,e->eq", tb.N, lam_e, tb.w, jac)
                uq_cot[pv][:, c] += w_e.reshape(-1)
                phi_cot_gp[pv][:, c] += -w_e.reshape(-1) / sigma

        # ---- stage 2 transpose: PPE -------------------------------------
        phi_cot = _grad_interp_transpose_scalar(st, phi_cot_gp)
        Kp = st.K_p.tolil()
        Kp.rows[0] = [0]
        Kp.data[0] = [1.0]
        phi_cot[0] = 0.0
        lam_p = splu(Kp.tocsr().tocsc().T).solve(phi_cot)
        lam_p[0] = 0.0
        lam_p_full = np.asarray(dm.constraints.T @ lam_p)
        for pv in dm.bins:
            tb = dm.tables_by_p[pv]
            conn = dm.mesh.conn_of[pv]
            h = dm.mesh.tree.h()[dm.mesh.bins[pv]]
            jac = (h / 2.0) ** dim
            dsc = 2.0 / h
            lam_e = lam_p_full[conn]
            w_ed = np.einsum("qad,ea,q,e->eqd", tb.dN, lam_e, tb.w,
                             jac * dsc)
            uq_cot[pv] += sigma * w_ed.reshape(-1, dim)

        # uq cotangent -> uhat (final iterate x) rhs
        uhat_cot = np.zeros((st.n_free, dim))
        for c in range(dim):
            uhat_cot[:, c] = _interp_transpose_scalar(
                st, {pv: uq_cot[pv][:, c] for pv in aq})
        rhs_adj = np.zeros(st.n_free * ndof)
        for c in range(dim):
            rhs_adj[c::ndof] = uhat_cot[:, c]

        # ---- reversed sweep over Picard iterates ------------------------
        T = dm.constraints.T.tocsr()
        T_vec = sp.kron(T, sp.identity(ndof, format="csr"), format="csr")
        dnu_total = 0.0
        for it in range(len(chain) - 1, -1, -1):
            rec = chain[it]
            lam_x = splu(rec["A"].tocsc().T).solve(rhs_adj)
            for i in st.dir_nodes:
                for c in range(dim):
                    lam_x[i * ndof + c] = 0.0
            lam_x[dim::ndof] = 0.0
            x_full = np.asarray(T_vec @ rec["x"])
            lam_full = np.asarray(T_vec @ lam_x)
            aqb_v, dnu_v = ns_volume_cotangents(
                dm, rec["aq"], rec["dq"], st.nu, rec["sigma"], 0.5,
                x_full, lam_full, timestab=st.timestab,
                tau_frozen=False)
            aqb_l, fqb_l, dnu_l = ns_load_cotangents(
                dm, rec["aq"], rec["fq"], st.nu, rec["sigma"], lam_full,
                timestab=st.timestab)
            dnu_total += dnu_v + dnu_l
            if it > 0:
                # inter-iterate chain: a_node(it) = uhat(it-1); the
                # transient-proven convention: rhs += interp^T(aqb_v.aq
                # + aqb_l, aqb_v.dq)
                G = _gp_field_transpose(
                    dm, {pv: aqb_v[pv][0] + aqb_l[pv] for pv in dm.bins},
                    {pv: aqb_v[pv][1] for pv in dm.bins}, ndof)
                rhs_adj = np.asarray(T_vec.T @ G)
        return dnu_total
