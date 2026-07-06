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
        dnu_stage2 = 0.0
        if not st.ppe_finescale:
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
        else:
            # FINESCALE branch (item c): flux = u - tau(u) r_m(u),
            # Kp(w) phi = rhs(flux), w = 1/sigma + tau. All chains
            # ANALYTIC (tau = 1/sqrt(sig2 + 4|u|^2/h^2 + cCI nu^2/h^4)).
            from ..physics.vms import tau_hbased_host, CI_F
            uq2, guq2 = st._gp_vals(uhat, grad=True)
            pq_g = st._gp_vals(self.p_star, grad=True)[1]
            fq = chain[-1]["fq"]
            c2CI = CI_F * 16.0 * dim
            per = {}
            w_gp = {}
            for pv in dm.bins:
                h = dm.mesh.tree.h()[dm.mesh.bins[pv]]
                nqp = dm.tables_by_p[pv].nqp
                he = np.repeat(h, nqp)
                u = uq2[pv]
                gu = guq2[pv].reshape(-1, dim, dim)   # [g, dx, comp] — the
                # stepper's own convention (its agu einsum 'gd,gdc->gc')
                umag = np.sqrt((u ** 2).sum(1))
                dtarg = (st.dt / (chain[-1]["sigma"] * st.dt)
                         if st.timestab else None)
                # NOTE b0/dt = sigma -> dt/b0 = 1/sigma
                tau = tau_hbased_host(umag, he, st.nu,
                                      dt=(1.0 / sigma if st.timestab
                                          else None), dim=dim)
                agu = np.einsum("gd,gdc->gc", u, gu)      # (u.grad)u_c
                r_m = (sigma * u + agu + pq_g[pv].reshape(-1, dim)
                       - fq[pv])
                dtau_du = -(tau ** 3)[:, None] * 4.0 * u / (he ** 2)[:, None]
                dtau_dnu = -(tau ** 3) * c2CI * st.nu / he ** 4
                per[pv] = dict(u=u, gu=gu, tau=tau, r_m=r_m,
                               dtau_du=dtau_du, dtau_dnu=dtau_dnu, he=he)
                w_gp[pv] = 1.0 / sigma + tau
            Kpw = st._weighted_stiffness(w_gp).tolil()
            Kpw.rows[0] = [0]
            Kpw.data[0] = [1.0]
            phi_cot[0] = 0.0
            lam_p = splu(Kpw.tocsr().tocsc().T).solve(phi_cot)
            lam_p[0] = 0.0
            lam_p_full = np.asarray(dm.constraints.T @ lam_p)
            # phi itself (needed for the w-chain): recompute stage 2 fwd
            rhs2 = np.zeros(dm.n_nodes)
            for pv in dm.bins:
                tb = dm.tables_by_p[pv]
                conn = dm.mesh.conn_of[pv]
                h = dm.mesh.tree.h()[dm.mesh.bins[pv]]
                ne, nqp = len(conn), tb.nqp
                jac = (h / 2.0) ** dim
                flux = (per[pv]["u"]
                        - per[pv]["tau"][:, None] * per[pv]["r_m"])
                fl = flux.reshape(ne, nqp, dim)
                be = np.einsum("qad,eqd,q,e->ea", tb.dN, fl, tb.w,
                               jac * (2.0 / h))
                np.add.at(rhs2, conn.ravel(), be.ravel())
            rhs2f = np.asarray(dm.constraints.T.T @ rhs2)
            rhs2f[0] = 0.0
            phi = splu(Kpw.tocsr().tocsc()).solve(rhs2f)
            phi_full = np.asarray(dm.constraints.T @ phi)
            for pv in dm.bins:
                tb = dm.tables_by_p[pv]
                conn = dm.mesh.conn_of[pv]
                h = dm.mesh.tree.h()[dm.mesh.bins[pv]]
                ne, nqp = len(conn), tb.nqp
                jac = (h / 2.0) ** dim
                dsc = 2.0 / h
                lam_e = lam_p_full[conn]
                # flux cotangent (from rhs2 assembly, transposed)
                fcot = np.einsum("qad,ea,q,e->eqd", tb.dN, lam_e, tb.w,
                                 jac * dsc).reshape(-1, dim)
                # w-chain: -lam_p^T dKp(w)/dw_g phi = -(grad lam.grad phi)_g
                gl = np.einsum("qad,ea->eqd", tb.dN,
                               lam_e) * dsc[:, None, None]
                gp_ = np.einsum("qad,ea->eqd", tb.dN,
                                phi_full[conn]) * dsc[:, None, None]
                wq = np.einsum("q,e->eq", tb.w, jac)
                wcot = -(gl * gp_).sum(-1) * wq          # [ne, nqp]
                wcot = wcot.reshape(-1)
                P = per[pv]
                # flux -> u chains
                # value: delta - dtau/du_e r_m_d - tau (sigma delta + gu[d,e])
                fd_rm = np.einsum("gd,ge->gde", P["r_m"], P["dtau_du"])
                # value-channel of the advective term: d(agu_c)/du_e = gu[g,e,c]
                tau_gu = P["tau"][:, None, None] * (
                    sigma * np.eye(dim)[None, :, :]
                    + P["gu"].transpose(0, 2, 1))   # -> [g, c(comp), e]
                dflux = (np.eye(dim)[None, :, :] - fd_rm - tau_gu)
                uq_cot[pv] += np.einsum("gd,gde->ge", fcot, dflux)
                # grad channel: dflux_d/d(gu[d,e-slot]) = -tau*u_e on the
                # advective term (u.grad)u_d -> cotangent to gu[d,e]
                # d(agu_c)/d(gu[g,e,c]) = u_e -> gcot[g, e(dx), c(comp)]
                gcot = -np.einsum("gc,g,ge->gec", fcot, P["tau"], P["u"])
                gc = gcot.reshape(ne, nqp, dim, dim)   # [e, q, dx, comp]
                add = np.einsum("qad,eqdc->eac", tb.dN,
                                gc) * dsc[:, None, None]
                tmp = np.zeros((dm.n_nodes, dim))
                np.add.at(tmp, conn.reshape(-1), add.reshape(-1, dim))
                for c in range(dim):
                    uhat_extra = np.asarray(
                        dm.constraints.T.T @ tmp[:, c])
                    # accumulate later via uq_cot pathway equivalent:
                    # store directly into a node-level bucket
                    if "node_extra" not in per[pv]:
                        per[pv]["node_extra"] = np.zeros((st.n_free, dim))
                    per[pv]["node_extra"][:, c] += uhat_extra
                # w-chain to u and nu
                uq_cot[pv] += wcot[:, None] * P["dtau_du"]
                dnu_stage2 += float((wcot * P["dtau_dnu"]).sum())
                # flux tau(nu) chain
                dnu_stage2 += float(np.einsum(
                    "gd,g,gd->", fcot, -P["dtau_dnu"], P["r_m"]))

        # uq cotangent -> uhat (final iterate x) rhs
        uhat_cot = np.zeros((st.n_free, dim))
        for c in range(dim):
            uhat_cot[:, c] = _interp_transpose_scalar(
                st, {pv: uq_cot[pv][:, c] for pv in aq})
        if st.ppe_finescale:
            for pv in dm.bins:
                extra = per[pv].get("node_extra")
                if extra is not None:
                    uhat_cot += extra
        rhs_adj = np.zeros(st.n_free * ndof)
        for c in range(dim):
            rhs_adj[c::ndof] = uhat_cot[:, c]

        # ---- reversed sweep over Picard iterates ------------------------
        T = dm.constraints.T.tocsr()
        T_vec = sp.kron(T, sp.identity(ndof, format="csr"), format="csr")
        dnu_total = dnu_stage2
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
