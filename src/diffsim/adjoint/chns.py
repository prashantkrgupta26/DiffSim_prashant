r"""CHNSDiscrete — reference-grade numpy mirror of the coupled Cahn-Hilliard /
Navier-Stokes (CHNS) BDF1 monolithic solve (SP-0 Task 5).

This class is *ground truth*: a later Warp kernel is parity-checked against it,
so the discretization contract below is BINDING — the kernel must reproduce
every term exactly.  Correctness beats speed throughout; the element loops are
plain vectorised numpy with an analytically-assembled Jacobian (FD-verified).

================================================================================
DOF LAYOUT
================================================================================
Node-major, block ``blk = dim + 3`` per node, ordered

    (u_0, .., u_{dim-1},  p,  phi,  mu)

so the global dof of (node a, field f) is ``a*blk + f`` with

    f = 0 .. dim-1  -> velocity component
    f = dim         -> pressure p
    f = dim + 1     -> phase field phi
    f = dim + 2     -> chemical potential mu

The state vector ``x`` is length ``ndof = blk * nn``.

================================================================================
NON-DIMENSIONAL PARAMETERS (Khanwale form; from the CHNSCase)
================================================================================
Re, We, Cn, Pe, Fr, rho_ratio (= rho_h/rho_l), eta_ratio (= eta_h/eta_l).

Density / viscosity normalisation (documented contract):
    rho_h = 1,           rho_l = 1 / rho_ratio      (heavy phi=+1, light phi=-1)
    eta_h = 1,           eta_l = 1 / eta_ratio
Local rho(phi), eta(phi) come from ``physics.chns.mix_props`` with these
endpoints, evaluated at ``phi^{n+1}`` (fully implicit).  mix_props linearly
interpolates rho = a_rho*phi + b_rho with a=(hi-lo)/2, b=(hi+lo)/2 and floors
each property at 1e-3*lo; the number of GP-clamps is surfaced per step.

================================================================================
DISCRETIZATION CONTRACT (BDF1, Q1 equal-order, PSPG + SUPG)
================================================================================
Time: BDF1.  History u_n, phi_n from the previously committed step.

Momentum residual (test w, component i), CONVECTIVE (non-skew) form:
    R^u_i = INT w [ rho (u-u_n)/dt + rho (u.grad)u + J.grad u ]              (Galerkin)
          + INT (2 eta / Re) D(u):D(w)      with D = 1/2 (grad u + grad u^T)
          - INT (div w) p
          - INT w . f_cap
          - INT w . f_grav
          + INT tau_m (u.grad w) . r_mom_strong                             (SUPG)
  where
    f_cap  = (Cn*We)^{-1} mu grad phi                       (capillary_gp)
    f_grav = rho g_hat / Fr^2,   g_hat = (0,-1) in 2-D (last axis in 3-D)
    J      = -((rho_h - rho_l)/2) * (1/Pe) * M * grad mu    (AGG mass flux, M=1)
    r_mom_strong = rho (u-u_n)/dt + rho (u.grad)u + J.grad u + grad p
                   - f_cap - f_grav
  The p1 Laplacian of u is identically zero, so the viscous strong term is
  omitted from r_mom_strong (bit-identical to ns_bricks make_linear_ns_Ae).
  Viscous term is integrated in the SYMMETRIC form (2 eta/Re) D(u):D(w).

Continuity residual (test q):
    R^p = INT q (div u)  +  INT tau_m grad q . r_mom_strong                 (PSPG)
  Pressure nullspace for enclosed flow: pin p at node 0 (strong row).

CH phi residual (test psi):
    R^phi = INT psi [ (phi - phi_n)/dt + u.grad phi ]
          + (1/Pe) INT grad psi . (M grad mu)               (M = 1 constant)

CH mu residual (test chi):
    R^mu = INT chi (mu - f'(phi)) - Cn^2 INT grad chi . grad phi
    f'(phi) = phi^3 - phi          (double-well f = 1/4 (phi^2-1)^2)

tau_m: physics.chns.tau_m_gp with LOCAL rho/eta per Gauss point (contrast-aware
stabilization), house constants Ci=(4, 36), c2CI = 36*16*dim.

Momentum convection: plain convective form rho*(u.grad)u — NOT the spec's skew-symmetric
s=1/2 form (controller ruling: convective is the parity target for the spike phase; skew
revisit deferred to Task 9 build-out). The Task 7 kernel must match the convective form.

mix_props clamp Jacobian convention: drho/deta = interpolation slope where unclamped,
exactly 0 where clamped (piecewise-constant at the clamp boundary) — kernel must match.

History commit rule: after each converged step, u_n and phi_n <- converged state; p and mu
carry no history (BDF1 on u,phi only).

SUPG/PSPG choice (parity contract, per the escalation note): the SUPG/PSPG
weighting acts on the FULL momentum strong residual r_mom_strong INCLUDING the
pressure gradient, capillary and gravity forces — this mirrors make_linear_ns_Ae
(where resu carries sigma*u + conv - nu*lap and the pressure-gradient / body
forces enter through the same tau_m*agw weighting) generalised to the coupled
system.  The Jacobian below differentiates SUPG/PSPG at the Newton (full) level
w.r.t. the primary unknowns u, p, phi, mu.

Newton: full coupled residual; Jacobian assembled analytically per element;
scipy.sparse splu solve.  Guards: mix_props clamp counter surfaced per step;
raise RuntimeError if clamped > 0.01*n_gp (spec §6) or Newton fails to converge.

Boundary conditions: no-slip u=0 on all boundary nodes (strong rows); natural
no-flux for phi/mu.  Sufficient for stationary-drop and closed-box bubble-rise.

Gate scope: uniform mesh (constraints.T == identity), 2-D and 3-D generic.
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu

from ..physics.chns import mix_props, capillary_gp, tau_m_gp


class CHNSDiscrete:
    """Coupled BDF1 CHNS residual/Jacobian on a uniform Q1 mesh (numpy mirror).

    Parameters
    ----------
    level, dim : int
        Refinement level / spatial dimension (documentary; the mesh comes from
        ``dm``).
    case : CHNSCase
        Supplies Re, We, Cn, Pe, Fr, rho_ratio, eta_ratio.
    dt : float
        Fixed BDF1 time step.
    dm : DeviceMesh
        Uniform Q1 mesh (constraints.T must be identity).  Required.
    gravity : bool
        If False, the gravity body force is zeroed (stationary-drop test).
    Cn_override : {None, "2h", float}
        Optional Cn override.  "2h" sets Cn = 2*h (parasitic-current bound for
        the coarse static-drop gate); a float sets Cn directly.
    newton_tol, newton_max : Newton controls.
    """

    def __init__(self, level, dim, case, dt, dm, gravity=True,
                 Cn_override=None, newton_tol=1e-10, newton_max=30):
        T = dm.constraints.T
        assert T.shape[0] == T.shape[1] and (
            abs(T - sp.eye(T.shape[0])).nnz == 0), \
            "CHNSDiscrete assumes constraints.T == identity (uniform mesh)"
        assert int(dim) == int(dm.dim), (
            f"dim arg ({dim}) disagrees with dm.dim ({dm.dim})")
        self.dm = dm
        self.case = case
        self.level = int(level)
        self.dim = int(dm.dim)
        self.dt = float(dt)
        self.gravity = bool(gravity)
        self.newton_tol = float(newton_tol)
        self.newton_max = int(newton_max)

        # --- non-dim parameters -------------------------------------------
        self.Re = float(case.Re)
        self.We = float(case.We)
        self.Pe = float(case.Pe)
        self.Fr = float(case.Fr)
        self.rho_ratio = float(case.rho_ratio)
        self.eta_ratio = float(case.eta_ratio)
        # normalisation: heavy phase = 1, light phase = 1/ratio
        self.rho_h, self.rho_l = 1.0, 1.0 / self.rho_ratio
        self.eta_h, self.eta_l = 1.0, 1.0 / self.eta_ratio
        # AGG flux prefactor: -((rho_h - rho_l)/2)*(1/Pe)*M, M=1
        self.agg = -0.5 * (self.rho_h - self.rho_l) / self.Pe

        self.blk = self.dim + 3
        self.nn = dm.n_nodes
        self.ndof = self.blk * self.nn

        # --- element scaffolding (mirrors MultiCHDiscrete) -----------------
        self.bins = []
        hmin = np.inf
        for pv, b in dm.bins.items():
            tb = dm.tables_by_p[pv]
            conn = dm.mesh.conn_of[pv].astype(np.int64)
            h = np.asarray(dm.mesh.tree.h()[dm.mesh.bins[pv]], np.float64)
            N = np.asarray(tb.N, np.float64)          # [nqp, nbf]
            dN = np.asarray(tb.dN, np.float64)        # [nqp, nbf, dim]
            w = np.asarray(tb.w, np.float64)          # [nqp]
            ne, nbf = conn.shape
            dscale = 2.0 / h                          # [ne] physical grad scale
            jac = (h / 2.0) ** self.dim
            dJxW = w[None, :] * jac[:, None]          # [ne, nqp]
            gdof = (conn[:, :, None] * self.blk
                    + np.arange(self.blk)[None, None, :]).reshape(
                        ne, self.blk * nbf)
            self.bins.append(dict(conn=conn, N=N, dN=dN, w=w, h=h, ne=ne,
                                  nbf=nbf, nqp=N.shape[0], dscale=dscale,
                                  dJxW=dJxW, gdof=gdof))
            hmin = min(hmin, float(h.min()))
        self.h = hmin

        # --- Cn resolution --------------------------------------------------
        if Cn_override is None:
            self.Cn = float(case.Cn)
        elif Cn_override == "2h":
            self.Cn = 2.0 * self.h
        else:
            self.Cn = float(Cn_override)

        # --- boundary (no-slip velocity) nodes -----------------------------
        self.bnd = np.asarray(dm.mesh.boundary_nodes, bool)
        self.coords = np.asarray(dm.mesh.node_coords, np.float64)

        # --- state ----------------------------------------------------------
        self.u = np.zeros((self.nn, self.dim))
        self.p = np.zeros(self.nn)
        self.phi = np.zeros(self.nn)
        self.mu = np.zeros(self.nn)
        self.u_n = np.zeros((self.nn, self.dim))
        self.phi_n = np.zeros(self.nn)
        self.t = 0.0
        self._lumped = None

    # ------------------------------------------------------------------
    # state <-> packed vector
    # ------------------------------------------------------------------
    def pack(self):
        x = np.zeros(self.ndof)
        blk, dim = self.blk, self.dim
        for d in range(dim):
            x[d::blk] = self.u[:, d]
        x[dim::blk] = self.p
        x[dim + 1::blk] = self.phi
        x[dim + 2::blk] = self.mu
        return x

    def unpack(self, x):
        blk, dim = self.blk, self.dim
        u = np.stack([x[d::blk] for d in range(dim)], axis=1)
        p = x[dim::blk]
        phi = x[dim + 1::blk]
        mu = x[dim + 2::blk]
        return u, p, phi, mu

    def set_history(self, u_n, phi_n):
        self.u_n = np.asarray(u_n, np.float64).reshape(self.nn, self.dim).copy()
        self.phi_n = np.asarray(phi_n, np.float64).copy()

    def set_initial(self, phi0, u0=None):
        self.phi = np.asarray(phi0, np.float64).copy()
        self.mu = np.zeros(self.nn)
        self.p = np.zeros(self.nn)
        self.u = (np.zeros((self.nn, self.dim)) if u0 is None
                  else np.asarray(u0, np.float64).reshape(self.nn, self.dim))
        self.u_n = self.u.copy()
        self.phi_n = self.phi.copy()
        self.t = 0.0

    # ------------------------------------------------------------------
    # lumped mass integral (mass diagnostic)
    # ------------------------------------------------------------------
    def lumped_mass(self):
        """Row-sum-lumped mass vector m so that ``m @ field`` = INT field."""
        if self._lumped is not None:
            return self._lumped
        m = np.zeros(self.nn)
        for B in self.bins:
            # consistent mass row sums = INT N_a  (partition of unity)
            Ma = np.einsum("eq,qa->ea", B["dJxW"], B["N"])
            np.add.at(m, B["conn"].ravel(), Ma.ravel())
        self._lumped = m
        return m

    # ------------------------------------------------------------------
    # interpolation of a nodal field to GPs (value + physical gradient)
    # ------------------------------------------------------------------
    def _interp(self, field, B):
        vals = field[B["conn"]]                        # [ne, nbf]
        v = np.einsum("qa,ea->eq", B["N"], vals)       # [ne, nqp]
        g = np.einsum("qad,ea,e->eqd", B["dN"], vals, B["dscale"])
        return v, g

    def _interp_vec(self, field, B):
        # field [nn, dim] -> value [ne,nqp,dim], grad [ne,nqp,dim(comp),dim(sp)]
        vals = field[B["conn"]]                        # [ne, nbf, dim]
        v = np.einsum("qa,ead->eqd", B["N"], vals)
        g = np.einsum("qas,ead,e->eqds", B["dN"], vals, B["dscale"])
        return v, g

    # ==================================================================
    # residual + jacobian (assembled together for efficiency / consistency)
    # ==================================================================
    def _assemble(self, x, want_jac):
        u, p, phi, mu = self.unpack(x)
        blk, dim, dt = self.blk, self.dim, self.dt
        Re, We, Cn, Pe, Fr = self.Re, self.We, self.Cn, self.Pe, self.Fr
        cw_inv = 1.0 / (Cn * We)
        R = np.zeros(self.ndof)
        rows, cols, vals = [], [], []
        n_clamp = 0
        n_gp = 0

        # gravity unit vector g_hat = (0,...,-1) (last axis)
        ghat = np.zeros(dim)
        if self.gravity:
            ghat[-1] = -1.0
        grav_scale = 1.0 / Fr ** 2

        for B in self.bins:
            dJxW, N, dN, dscale = B["dJxW"], B["N"], B["dN"], B["dscale"]
            ne, nbf, nqp = B["ne"], B["nbf"], B["nqp"]
            h_e = B["h"]
            n_gp += ne * nqp

            u_gp, gu = self._interp_vec(u, B)          # [e,q,d], [e,q,d,s]
            un_gp, _ = self._interp_vec(self.u_n, B)
            p_gp, gp = self._interp(p, B)
            phi_gp, gphi = self._interp(phi, B)
            phin_gp, _ = self._interp(self.phi_n, B)
            mu_gp, gmu = self._interp(mu, B)

            # --- local rho, eta at GPs (implicit phi) ----------------------
            rho_gp, eta_gp, nc = mix_props(
                phi_gp.ravel(), self.rho_h, self.rho_l,
                self.eta_h, self.eta_l)
            n_clamp += nc
            rho_gp = rho_gp.reshape(ne, nqp)
            eta_gp = eta_gp.reshape(ne, nqp)
            # d rho/d phi, d eta/d phi (linear interp; 0 where clamped)
            a_rho = 0.5 * (self.rho_h - self.rho_l)
            a_eta = 0.5 * (self.eta_h - self.eta_l)
            rho_raw = a_rho * phi_gp + 0.5 * (self.rho_h + self.rho_l)
            eta_raw = a_eta * phi_gp + 0.5 * (self.eta_h + self.eta_l)
            drho = np.where(rho_raw >= 1e-3 * self.rho_l, a_rho, 0.0)
            deta = np.where(eta_raw >= 1e-3 * self.eta_l, a_eta, 0.0)

            # --- tau_m (per GP, local rho/eta) -----------------------------
            tau = tau_m_gp(u_gp.reshape(-1, dim), rho_gp.ravel(),
                           eta_gp.ravel(), float(h_e[0]), dt, Re
                           ).reshape(ne, nqp)
            # --- analytic tau derivatives (Newton-consistent; NOT frozen) --
            # tau = 1/(denom*rho), denom = sqrt(A + Bu + C),
            #   A = 4/dt^2, Bu = 4|u|^2/h^2, C = c2CI*nu^2/h^4, nu = eta/(rho Re)
            h_ = float(h_e[0])
            c2CI = 36.0 * 16.0 * dim
            nu_loc = eta_gp / (rho_gp * Re)
            A_t = 4.0 / dt ** 2
            Bu_t = 4.0 * np.sum(u_gp ** 2, axis=-1) / h_ ** 2      # [e,q]
            C_t = c2CI * nu_loc ** 2 / h_ ** 4
            denom = np.sqrt(A_t + Bu_t + C_t)                      # [e,q]
            # d denom / d rho, d eta (via C only)
            dC_deta = 2.0 * c2CI * eta_gp / (h_ ** 4 * rho_gp ** 2 * Re ** 2)
            dC_drho = -2.0 * c2CI * eta_gp ** 2 / (h_ ** 4 * rho_gp ** 3
                                                   * Re ** 2)
            ddenom_dphi = ((dC_deta * deta + dC_drho * drho)
                           / (2.0 * denom))
            # tau = 1/(denom rho): dtau/dphi = -tau*ddenom_dphi/denom
            #                                  - tau*drho/rho
            dtau_dphi = -tau * (ddenom_dphi / denom + drho / rho_gp)  # [e,q]
            # dtau/du_k (via Bu): dBu/du_k = 8 u_k/h^2 ->
            #   ddenom/du_k = (8 u_k/h^2)/(2 denom); dtau/du_k = -tau*ddenom/denom
            dtau_du = (-tau[..., None]
                       * (4.0 * u_gp / h_ ** 2)[...]
                       / denom[..., None] ** 2)                    # [e,q,k]

            # --- AGG flux J = agg * grad mu --------------------------------
            J_gp = self.agg * gmu                      # [e,q,s]

            # --- convective derivatives ------------------------------------
            # (u.grad)u : [e,q,comp]  = sum_s u_s d u_comp/d x_s
            ugradu = np.einsum("eqs,eqds->eqd", u_gp, gu)
            # (J.grad)u : [e,q,comp]
            Jgradu = np.einsum("eqs,eqds->eqd", J_gp, gu)
            # u.grad phi : [e,q]
            ugradphi = np.einsum("eqs,eqs->eq", u_gp, gphi)

            # --- capillary + gravity body forces [e,q,comp] ----------------
            fcap = cw_inv * mu_gp[..., None] * gphi     # [e,q,s]
            fgrav = (rho_gp[..., None] * grav_scale) * ghat[None, None, :]

            # --- momentum strong residual (p1: no viscous lap) -------------
            # r_mom = rho(u-u_n)/dt + rho ugradu + Jgradu + grad p - fcap - fgrav
            r_mom = (rho_gp[..., None] * (u_gp - un_gp) / dt
                     + rho_gp[..., None] * ugradu
                     + Jgradu + gp - fcap - fgrav)      # [e,q,comp]

            # ---------------- RESIDUAL ROWS --------------------------------
            # momentum galerkin: INT N_a * [rho(u-un)/dt + rho ugradu + Jgradu]
            mom_body = (rho_gp[..., None] * (u_gp - un_gp) / dt
                        + rho_gp[..., None] * ugradu + Jgradu
                        - fcap - fgrav)                 # [e,q,comp]
            # INT N_a * mom_body  -> [e,a,comp]
            Rmom = np.einsum("eq,qa,eqd->ead", dJxW, N, mom_body)
            # viscous (2 eta/Re) D(u):D(w):  eta/Re (grad u + grad u^T):grad w
            # grad w for test a comp i is dN_a in direction; assemble per a,comp
            # symgrad_u = grad u + grad u^T : [e,q,comp,s]
            symgu = gu + np.transpose(gu, (0, 1, 3, 2))
            # dN physical [e,q,a,s]
            dNp = np.einsum("qas,e->eqas", dN, dscale)
            # visc contribution to R for (a,comp d): INT (eta/Re) symgu[:,:,d,s] dN_a,s
            Rvisc = (1.0 / Re) * np.einsum(
                "eq,eq,eqds,eqas->ead", dJxW, eta_gp, symgu, dNp)
            # pressure: -INT (div w) p ; div w for comp d is dN_a,d
            Rpres = -np.einsum("eq,eq,eqad->ead", dJxW, p_gp, dNp)
            # SUPG: INT tau (u.grad w) . r_mom ; u.grad w_a = sum_s u_s dN_a,s
            ugw = np.einsum("eqs,eqas->eqa", u_gp, dNp)      # [e,q,a]
            Rsupg = np.einsum("eq,eq,eqa,eqd->ead", dJxW, tau, ugw, r_mom)
            Ru = Rmom + Rvisc + Rpres + Rsupg               # [e,a,comp]
            for d in range(dim):
                np.add.at(R, B["gdof"][:, d::blk].ravel(),
                          Ru[:, :, d].ravel())

            # continuity: INT q div u + PSPG INT tau grad q . r_mom
            divu = np.einsum("eqdd->eq", gu)                # [e,q]
            Rcont = np.einsum("eq,qa,eq->ea", dJxW, N, divu)
            Rpspg = np.einsum("eq,eq,eqas,eqs->ea", dJxW, tau, dNp, r_mom)
            Rp = Rcont + Rpspg
            np.add.at(R, B["gdof"][:, dim::blk].ravel(), Rp.ravel())

            # CH phi (CONSERVATIVE advection, spec §1 div(u phi) in strong
            # Galerkin form u.grad phi + phi div u): summed over test functions
            # (partition of unity) INT div(u phi) = 0 under no-flux BCs, giving
            # machine-exact lumped-mass conservation.  "Non-skew" per the brief
            # = not the skew average; this is the divergence (conservative) form.
            ch_body = (phi_gp - phin_gp) / dt + ugradphi + phi_gp * divu
            Rphi = (np.einsum("eq,qa,eq->ea", dJxW, N, ch_body)
                    + (1.0 / Pe) * np.einsum("eq,eqas,eqs->ea",
                                             dJxW, dNp, gmu))
            np.add.at(R, B["gdof"][:, dim + 1::blk].ravel(), Rphi.ravel())

            # CH mu: INT chi(mu - f'(phi)) - Cn^2 INT gchi . grad phi
            fp = phi_gp ** 3 - phi_gp                       # f'(phi)
            Rmu = (np.einsum("eq,qa,eq->ea", dJxW, N, mu_gp - fp)
                   - Cn ** 2 * np.einsum("eq,eqas,eqs->ea", dJxW, dNp, gphi))
            np.add.at(R, B["gdof"][:, dim + 2::blk].ravel(), Rmu.ravel())

            if not want_jac:
                continue

            # ================= ELEMENT JACOBIAN =========================
            Ae = np.zeros((ne, blk * nbf, blk * nbf))

            # Precompute scalar shape products
            # NNab = INT N_a N_b [e,a,b]
            NNab = np.einsum("eq,qa,qb->eab", dJxW, N, N)
            # For gradient couplings we build per-term einsums below.

            # helper scatter: add contribution block for (row field fr, col
            # field fc) given per-(e,a,b) matrix
            def addblk(fr, fc, mat):
                Ae[:, fr::blk, fc::blk] += mat

            #### --- Momentum row derivatives -------------------------------
            # d r_mom / d(unknowns) pieces reused by galerkin, SUPG, PSPG.
            # We differentiate the *strong* residual r_mom and mom_body wrt
            # nodal dofs, expressed at GPs, then contract with the appropriate
            # test weightings.

            # Shorthands
            # dN_b physical: dNp[:,:,b,s]
            # --- du_j (velocity col, component jc) -----------------------
            # d(rho(u-un)/dt)/du = rho/dt * delta ; body comp d wrt u_jc:
            #   transient: (rho/dt) N_b delta_{d,jc}
            #   convection rho ugradu comp d = rho sum_s u_s du_d/dx_s
            #     d/du_jc(node b): rho [ N_b du_d/dx_jc + delta_{d,jc} sum_s u_s dN_b,s ]
            #   Jgradu comp d = sum_s J_s du_d/dx_s ; d/du_jc = delta_{d,jc} sum_s J_s dN_b,s
            # We assemble momentum body Jacobian wrt u.
            # term A: transient+conv-second (delta_{d,jc}) coefficient c1[e,q,b]
            #   c1 = (rho/dt) N_b + rho (u.grad N_b) + (J.grad N_b)
            uGNb = np.einsum("eqs,eqbs->eqb", u_gp, dNp)     # u.grad N_b
            JGNb = np.einsum("eqs,eqbs->eqb", J_gp, dNp)     # J.grad N_b
            c1 = (rho_gp[..., None] * (N[None] / dt)         # (rho/dt) N_b
                  + rho_gp[..., None] * uGNb + JGNb)          # [e,q,b]
            # galerkin transient+conv (diagonal in comp): INT N_a * c1 -> a,b
            GAL_diag = np.einsum("eq,qa,eqb->eab", dJxW, N, c1)
            for d in range(dim):
                addblk(d, d, GAL_diag)
            # conv-first term: rho N_b du_d/dx_jc (couples comp d to comp jc)
            # coefficient m[e,q,b,d,jc] = rho N_b gu[e,q,d,jc]; contract INT N_a
            for d in range(dim):
                for jc in range(dim):
                    # rho * N_b * gu_{d,jc}
                    mm = np.einsum("eq,qb,eq->eqb", rho_gp, N, gu[:, :, d, jc])
                    blk_ab = np.einsum("eq,qa,eqb->eab", dJxW, N, mm)
                    addblk(d, jc, blk_ab)

            # --- viscous (2 eta/Re) D(u):D(w), linear in u.  Row (a,d) col
            # (b,jc):  (1/Re) INT eta [ delta_{d,jc} gNa.gNb + gNa_jc gNb_d ]
            visc_gg = (1.0 / Re) * np.einsum(
                "eq,eq,eqas,eqbs->eab", dJxW, eta_gp, dNp, dNp)  # gNa.gNb
            for d in range(dim):
                addblk(d, d, visc_gg)                            # delta_{d,jc}
                for jc in range(dim):
                    m_t = (1.0 / Re) * np.einsum(
                        "eq,eq,eqa,eqb->eab", dJxW, eta_gp,
                        dNp[:, :, :, jc], dNp[:, :, :, d])
                    addblk(d, jc, m_t)

            # --- momentum wrt p: galerkin body has no p; strong r_mom has
            #   +grad p.  Only through SUPG (below).  Pressure term Rpres =
            #   -INT (div w) p -> d/dp_b = -INT dN_a,d N_b
            for d in range(dim):
                blk_ab = -np.einsum("eq,eqa,qb->eab", dJxW, dNp[:, :, :, d], N)
                addblk(d, dim, blk_ab)

            # --- momentum wrt phi: rho depends on phi (drho), fcap on phi
            #   (grad phi), fgrav on phi (rho).  Also viscous eta(phi).
            # d(mom_body)/dphi_b comp d:
            #   drho/dphi N_b * [(u-un)/dt + ugradu]_d      (transient+conv)
            #   - d(fcap)/dphi : fcap = cw_inv mu grad phi ; wrt phi_b:
            #       cw_inv mu dN_b,d  -> body has -fcap so derivative -cw_inv mu dN_b,d
            #   - d(fgrav)/dphi: -(drho N_b) grav_scale ghat_d
            drho_Nb = np.einsum("eq,qb->eqb", drho, N)       # drho * N_b
            accel_d = ((u_gp - un_gp) / dt + ugradu)         # [e,q,d]
            for d in range(dim):
                # drho part (transient+conv) : INT N_a * drho N_b accel_d
                m1 = np.einsum("eq,qa,eqb,eq->eab", dJxW, N, drho_Nb,
                               accel_d[:, :, d])
                # -fcap part: body = -fcap ; d/dphi_b = -cw_inv mu dN_b,d
                m2 = -np.einsum("eq,qa,eq,eqb->eab", dJxW, N, cw_inv * mu_gp,
                                dNp[:, :, :, d])
                # -fgrav part: -(drho N_b) grav_scale ghat_d
                m3 = -np.einsum("eq,qa,eqb->eab", dJxW * grav_scale * ghat[d],
                                N, drho_Nb)
                addblk(d, dim + 1, m1 + m2 + m3)
            # viscous eta(phi): Rvisc = (1/Re) INT eta symgu:dN_a ; d/dphi_b
            #   = (1/Re) INT deta N_b symgu[:,:,d,s] dN_a,s
            for d in range(dim):
                mvis = (1.0 / Re) * np.einsum(
                    "eq,eq,qb,eqs,eqas->eab", dJxW, deta, N,
                    symgu[:, :, d, :], dNp)
                addblk(d, dim + 1, mvis)

            # --- momentum wrt mu: fcap = cw_inv mu grad phi ; body -fcap
            #   d/dmu_b = -cw_inv N_b grad phi_d ; also J = agg grad mu ->
            #   Jgradu comp d = agg sum_s (grad mu)_s du_d/dx_s ;
            #   d/dmu_b = agg (grad N_b . grad? ) -> agg sum_s dN_b,s du_d/dx_s
            for d in range(dim):
                # -fcap wrt mu
                mfc = -np.einsum("eq,qa,eq,qb->eab", dJxW, N,
                                 cw_inv * gphi[:, :, d], N)
                # Jgradu wrt mu: agg * (dN_b . row d of gu)
                gu_d = gu[:, :, d, :]                        # [e,q,s]
                mJ = self.agg * np.einsum("eq,qa,eqbs,eqs->eab",
                                          dJxW, N, dNp, gu_d)
                addblk(d, dim + 2, mfc + mJ)

            #### --- viscous already added above; --- SUPG derivatives ------
            # Rsupg = INT tau (u.grad w_a) . r_mom.
            # tau is differentiated (Newton-consistent): dtau_du and dtau_dphi
            # terms follow.  d/d(dof) therefore has three pieces:
            #   (i) d(u.grad w_a)/du * r_mom   (SUPG test depends on u)
            #   (ii) tau (u.grad w_a) . d r_mom/d(dof)
            #   (iii) d tau/d(dof) * (u.grad w_a) . r_mom   (Newton-consistent)
            # ugw[e,q,a] = u.grad w_a ; bake dJxW into the SUPG weight so all
            # blocks contracted against tau_ugw carry the quadrature measure.
            tau_ugw = (dJxW * tau)[:, :, None] * ugw         # [e,q,a]
            # d r_mom/du_jc (component d row):
            #   transient: rho/dt N_b delta ; conv-first rho N_b gu_{d,jc};
            #   conv-second rho delta u.gradN_b ; Jgradu delta J.gradN_b
            #   grad p: 0 ; fcap:0 ; fgrav:0
            # -> same c1 (diag) + conv-first (off) as galerkin body strong part
            #    r_mom diag coeff cr1 = rho/dt N_b + rho u.gradN_b + J.gradN_b
            cr1 = c1                                          # identical
            for d in range(dim):
                # (ii) diagonal in comp d,jc=d
                m_ii = np.einsum("eqa,eqb->eab", tau_ugw, cr1)
                addblk(d, d, m_ii)
                # (ii) conv-first off-diagonal comp d wrt u_jc
                for jc in range(dim):
                    mm = np.einsum("eq,qb->eqb", rho_gp, N) * gu[:, :, d, jc][:, :, None]
                    m_off = np.einsum("eqa,eqb->eab", tau_ugw, mm)
                    addblk(d, jc, m_off)
                # (i) SUPG test depends on u_jc: d(u.grad w_a)/du_jc(node b)
                #     = N_b dN_a,jc ; times tau * r_mom_d
                # (iii) tau depends on u_jc: dtau/du_jc N_b weight ugw_a r_mom_d
                for jc in range(dim):
                    m_i = np.einsum("eq,eq,qb,eqa,eq->eab", dJxW, tau, N,
                                    dNp[:, :, :, jc], r_mom[:, :, d])
                    m_iii = np.einsum("eq,eqa,eq,eq,qb->eab", dJxW, ugw,
                                      r_mom[:, :, d], dtau_du[:, :, jc], N)
                    addblk(d, jc, m_i + m_iii)
            # d r_mom/dp = grad p -> dN_b ; SUPG: tau ugw_a * dN_b,d
            for d in range(dim):
                m_p = np.einsum("eqa,eqb->eab", tau_ugw, dNp[:, :, :, d])
                addblk(d, dim, m_p)
            # d r_mom/dphi: drho*(u-un)/dt + drho ugradu (transient+conv strong)
            #   - fcap(-cw_inv mu dN_b,d) - fgrav(-drho N_b grav ghat_d)
            # SUPG also depends on phi through tau(rho(phi), eta(phi)):
            #   d/dphi_b Rsupg[a,d] += INT dJxW ugw_a r_mom_d (dtau/dphi) N_b
            dtau_Nb = np.einsum("eq,qb->eqb", dtau_dphi, N)   # [e,q,b]
            for d in range(dim):
                dr_phi = (np.einsum("eqb,eq->eqb", drho_Nb, accel_d[:, :, d])
                          - cw_inv * mu_gp[:, :, None] * dNp[:, :, :, d]
                          - grav_scale * ghat[d] * drho_Nb)
                m_phi = np.einsum("eqa,eqb->eab", tau_ugw, dr_phi)
                # tau-derivative correction (uses raw ugw, weight dJxW*r_mom_d)
                m_phi_tau = np.einsum("eq,eqa,eq,eqb->eab", dJxW, ugw,
                                      r_mom[:, :, d], dtau_Nb)
                addblk(d, dim + 1, m_phi + m_phi_tau)
            # d r_mom/dmu: -cw_inv N_b gphi_d + agg (dN_b.gu_d)
            for d in range(dim):
                dr_mu = (-cw_inv * np.einsum("eq,qb->eqb", gphi[:, :, d], N)
                         + self.agg * np.einsum("eqbs,eqs->eqb", dNp,
                                                gu[:, :, d, :]))
                m_mu = np.einsum("eqa,eqb->eab", tau_ugw, dr_mu)
                addblk(d, dim + 2, m_mu)

            #### --- Continuity row ------------------------------------------
            # Rcont = INT N_a div u -> d/du_jc = INT N_a dN_b,jc
            for jc in range(dim):
                m = np.einsum("eq,qa,eqb->eab", dJxW, N, dNp[:, :, :, jc])
                addblk(dim, jc, m)
            # PSPG: Rpspg = INT tau grad N_a . r_mom (tau frozen)
            #   d/du_jc: tau grad N_a . (d r_mom/du_jc) ; bake dJxW into weight
            tau_gNa = (dJxW * tau)[:, :, None, None] * dNp   # [e,q,a,s]
            # grad N_a . r_mom (per element-a), weighted by dJxW — reused by the
            # tau-derivative PSPG corrections.
            gNa_rmom = dJxW[:, :, None] * np.einsum(
                "eqas,eqs->eqa", dNp, r_mom)                  # [e,q,a]
            for jc in range(dim):
                # diagonal-in-comp part of d r_mom/du: r_mom comp s row
                #   = delta_{s,jc}(rho/dt N_b + rho u.gradN_b + J.gradN_b)
                #     + rho N_b gu_{s,jc}
                # contract tau grad N_a,s . that
                # component jc only: tau grad N_a,jc * cr1
                m_diag = np.einsum("eqa,eqb->eab", tau_gNa[:, :, :, jc], cr1)
                # conv-first: sum_s tau gNa_s * rho N_b gu_{s,jc}
                m_conv = np.einsum("eqas,eqb,eqs->eab", tau_gNa,
                                   np.einsum("eq,qb->eqb", rho_gp, N),
                                   gu[:, :, :, jc])
                # tau depends on u_jc: dtau/du_jc N_b (grad N_a . r_mom)
                m_tau = np.einsum("eqa,eq,qb->eab", gNa_rmom,
                                  dtau_du[:, :, jc], N)
                addblk(dim, jc, m_diag + m_conv + m_tau)
            # d/dp: tau grad N_a . grad N_b
            m_pp = np.einsum("eqas,eqbs->eab", tau_gNa, dNp)
            addblk(dim, dim, m_pp)
            # d/dphi: tau grad N_a . d r_mom/dphi
            for_phi = (np.einsum("eqb,eqd->eqbd", drho_Nb, accel_d)
                       - cw_inv * np.einsum("eq,eqbd->eqbd", mu_gp, dNp)
                       - grav_scale * np.einsum("eqb,d->eqbd", drho_Nb, ghat))
            m_cphi = np.einsum("eqas,eqbs->eab", tau_gNa, for_phi)
            # tau depends on phi: dtau/dphi N_b (grad N_a . r_mom)
            m_cphi_tau = np.einsum("eqa,eqb->eab", gNa_rmom, dtau_Nb)
            addblk(dim, dim + 1, m_cphi + m_cphi_tau)
            # d/dmu: tau grad N_a . d r_mom/dmu
            for_mu = (-cw_inv * np.einsum("eqd,qb->eqbd", gphi, N)
                      + self.agg * np.einsum("eqbs,eqds->eqbd", dNp, gu))
            m_cmu = np.einsum("eqas,eqbs->eab", tau_gNa, for_mu)
            addblk(dim, dim + 2, m_cmu)

            #### --- CH phi row (conservative advection) ---------------------
            # Rphi = INT N_a[(phi-phin)/dt + u.grad phi + phi div u]
            #        + (1/Pe) INT gNa.grad mu
            # d/du_jc: from u.grad phi -> N_b dphi/dx_jc ; from phi div u ->
            #          phi dN_b,jc  (d div u/du_jc,b = dN_b,jc)
            for jc in range(dim):
                m = (np.einsum("eq,qa,qb,eq->eab", dJxW, N, N, gphi[:, :, jc])
                     + np.einsum("eq,qa,eq,eqb->eab", dJxW, N, phi_gp,
                                 dNp[:, :, :, jc]))
                addblk(dim + 1, jc, m)
            # d/dphi: transient (1/dt)N_b + u.grad N_b + div u N_b
            cphi = (N[None] / dt) + uGNb + divu[:, :, None] * N[None]  # [e,q,b]
            m_phiphi = np.einsum("eq,qa,eqb->eab", dJxW, N, cphi)
            addblk(dim + 1, dim + 1, m_phiphi)
            # d/dmu: (1/Pe) INT gNa . grad N_b
            m_phimu = (1.0 / Pe) * np.einsum("eq,eqas,eqbs->eab",
                                             dJxW, dNp, dNp)
            addblk(dim + 1, dim + 2, m_phimu)

            #### --- CH mu row -----------------------------------------------
            # Rmu = INT N_a(mu - f'(phi)) - Cn^2 INT gNa.grad phi
            # d/dphi: -INT N_a f''(phi) N_b - Cn^2 INT gNa.grad N_b
            fpp = 3.0 * phi_gp ** 2 - 1.0                    # f''(phi)
            m_muphi = (-np.einsum("eq,qa,eq,qb->eab", dJxW, N, fpp, N)
                       - Cn ** 2 * np.einsum("eq,eqas,eqbs->eab",
                                             dJxW, dNp, dNp))
            addblk(dim + 2, dim + 1, m_muphi)
            # d/dmu: INT N_a N_b
            addblk(dim + 2, dim + 2, NNab)

            # ---- scatter Ae ----
            gdof = B["gdof"]
            rows.append(np.repeat(gdof, blk * nbf, axis=1).ravel())
            cols.append(np.tile(gdof, (1, blk * nbf)).ravel())
            vals.append(Ae.ravel())

        J = None
        if want_jac:
            J = sp.coo_matrix(
                (np.concatenate(vals),
                 (np.concatenate(rows), np.concatenate(cols))),
                shape=(self.ndof, self.ndof)).tocsr()
        self._last_clamp = n_clamp
        self._last_ngp = n_gp
        return R, J

    def residual(self, x):
        R, _ = self._assemble(x, want_jac=False)
        return self._apply_bc_res(x, R)

    def jacobian(self, x):
        _, J = self._assemble(x, want_jac=True)
        return self._apply_bc_jac(J)

    # ------------------------------------------------------------------
    # boundary conditions: no-slip velocity (strong), pressure pin at node 0
    # ------------------------------------------------------------------
    def _bc_rows(self):
        blk, dim = self.blk, self.dim
        rows = []
        for a in np.where(self.bnd)[0]:
            for d in range(dim):
                rows.append(a * blk + d)
        rows.append(0 * blk + dim)          # pressure pin at node 0
        return np.asarray(rows, np.int64)

    def _apply_bc_res(self, x, R):
        rows = self._bc_rows()
        blk, dim = self.blk, self.dim
        # velocity: enforce u=0 -> residual = x_dof (target 0)
        # pressure pin: residual = p_node0 (target 0)
        R[rows] = x[rows]
        return R

    def _apply_bc_jac(self, J):
        rows = self._bc_rows()
        J = J.tolil()
        for r in rows:
            J.rows[r] = [r]
            J.data[r] = [1.0]
        return J.tocsr()

    # ------------------------------------------------------------------
    # Newton step
    # ------------------------------------------------------------------
    def step(self, guards=True):
        x = self.pack()
        it = 0
        rnorm = np.inf
        for it in range(1, self.newton_max + 1):
            R = self.residual(x)
            rnorm = float(np.linalg.norm(R))
            if rnorm < self.newton_tol:
                break
            J = self.jacobian(x)
            dx = splu(J.tocsc()).solve(-R)
            x = x + dx
            if float(np.abs(dx).max()) < self.newton_tol:
                # recompute residual for reporting
                R = self.residual(x)
                rnorm = float(np.linalg.norm(R))
                break
        else:
            raise RuntimeError(
                f"CHNS Newton failed to converge in {self.newton_max} iters: "
                f"residual norm {rnorm:.3e}")

        clamped = int(getattr(self, "_last_clamp", 0))
        ngp = int(getattr(self, "_last_ngp", 1))
        if guards and clamped > 0.01 * ngp:
            raise RuntimeError(
                f"mix_props clamp counter {clamped} > 1% of {ngp} GPs")

        u, p, phi, mu = self.unpack(x)
        # commit history (BDF1: previous = just-converged state)
        self.u_n = u.copy()
        self.phi_n = phi.copy()
        self.u, self.p, self.phi, self.mu = u, p, phi, mu
        self.t += self.dt
        return dict(newton_iters=it, clamped=clamped, res_norm=rnorm)

    def march(self, t_end):
        out = []
        while self.t < t_end - 1e-12:
            out.append(self.step())
        return out
