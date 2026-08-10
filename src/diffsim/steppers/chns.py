r"""CHNSStaggeredStepper — SP-0 Task 6: the STAGGERED CH -> NS projection
prototype (one of the two coupling prototypes the SP-0 spike compares; the
other is Task 7's monolithic CHNSDiscrete-mirror kernel).

Thin but honest: every brick is a reviewed, existing component —

  CH solve      : physics.multiphase.MultiPhaseStepper with the SP-0
                  bulk="quartic" [-1,1] double-well mode (mu_bulk =
                  phi^3 - phi, the CHNSDiscrete contract), kappa = Cn^2,
                  onsager = 1/Pe, prescribed GP velocity via adv_gp
                  (convective form u.grad phi, solenoidal u assumed).
  NS assembly   : api.ns_bricks.assemble_linear_ns with the Task 3
                  variable-coefficient path (per-GP nu_q/rho_q/f_q).
  Projection    : LerayProjectionStepper internals BY COMPOSITION (the
                  task-brief route): its scalar consistent mass M,
                  _gp_vals, _weighted_stiffness and _weak_div_free
                  helpers drive a directly-assembled predictor + PPE +
                  velocity-update triple (the Leray step() itself is
                  constant-coefficient and resists per-step rho/eta
                  injection, so the three sub-steps are assembled here).
  Coefficients  : physics.chns.mix_props / capillary_gp with the
                  adjoint/chns.py normalisation contract rho_h = 1,
                  rho_l = 1/rho_ratio (heavy phi=+1, light phi=-1);
                  eta likewise.

PER-STEP ORDER (t_n -> t_{n+1}, BDF1 throughout)
------------------------------------------------
  (1) CH:   adv_gp <- u^n at GPs;  MultiPhaseStepper Newton solve
            -> phi^{n+1}, mu^{n+1}  (mixed form: mu is a solved DOF).
  (2) COEF: at GPs from the FRESH phi^{n+1}, mu^{n+1}:
              rho = mix_props(phi; 1, 1/rho_ratio)     [clamps counted]
              eta = mix_props(phi; 1, 1/eta_ratio)
              f   = (Cn We)^{-1} mu^{n+1} grad phi^{n+1}   (capillary)
                  + rho ghat / Fr^2,  ghat = (0, -1)       (gravity)
  (3) NS:   (a) PREDICTOR — assemble_linear_ns(sigma = 1/dt, advecting
                field a = u^n, CONVECTIVE form s_skew = 0 [matches the
                Task 5 parity ruling], nu_q = eta/Re, rho_q = rho,
                fq = rho u^n / dt  [BDF1 history], f_q = capillary +
                gravity); strong rows: no-slip u = 0 on ALL boundary
                nodes, pressure DOFS PINNED to p* nodal values (grad p*
                enters the momentum rows through the coupling — the
                Leray v1 predictor convention).  -> u_hat.
            (b) PPE — variable-density incremental projection:
                  int (1/rho) grad q . grad phi' = sigma int grad q . u_hat
                (1/rho-weighted stiffness via _weighted_stiffness; RHS
                = sigma * _weak_div_free(u_hat)); enclosed box: pinned
                at free node 0 (the Leray _pin_rows convention).
            (c) UPDATE — consistent-mass L2 correction
                  M du_c = -(1/sigma) int N (1/rho) (grad phi')_c ,
                  u^{n+1} = u_hat + du,  then re-impose u = 0 on the
                  boundary rows (strong no-slip; the L2 smear is
                  interior);  p^{n+1} = p* + phi',  p* <- p^{n+1}
                (standard incremental pressure update).

SHARED PROTOCOL (duck-typed; Task 7 implements the same names)
--------------------------------------------------------------
  .step()                      one fixed-dt step; returns the info dict
  .march(t_end, snap_every=None) -> list of snapshot dicts with keys
        t, phi, u, p, wall_per_step, newton_iters, clamped
        (snap_every=None -> a snapshot EVERY step)
  .phi  [n_free]   .u  [n_free, dim]   .p  [n_free]     (properties)
  .last_newton_iters : int = CH Newton iters + NS linear solves (the
        predictor is LINEARIZED at u^n — exactly 1 momentum solve — so
        the int is ch_iters + 1; the split detail is kept in
        .last_iters_detail = {"ch": k, "ns": 1}).
  .last_clamped : int = mix_props clamp count on the NS-COEFFICIENT
        side ONLY (the quartic CH bulk has no (0,1) clamping by
        construction — documented convention).

Ctor: CHNSStaggeredStepper(dm, case, dt, linsolver="splu",
                           Cn_override=None)
  Cn_override mirrors adjoint.chns.CHNSDiscrete: None -> case.Cn,
  "2h" -> 2 * hmin (the gate resolvability convention), float -> as-is.
"""
import time

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu

from ..api.ns_bricks import assemble_linear_ns
from ..physics.chns import mix_props, capillary_gp, make_chns_newton
from ..physics.multiphase import MultiPhaseStepper
from .leray import LerayProjectionStepper


class CHNSStaggeredStepper:
    def __init__(self, dm, case, dt, linsolver="splu", Cn_override=None):
        assert linsolver == "splu", \
            "prototype scope: splu only (spike ruling; GPU solves are " \
            "a build-out concern)"
        self.dm = dm
        self.case = case
        self.dt = float(dt)
        self.sigma = 1.0 / self.dt
        dim = dm.dim
        self.dim = dim

        # --- case parameters (Khanwale non-dimensional form) -------------
        self.Re, self.We = float(case.Re), float(case.We)
        self.Pe, self.Fr = float(case.Pe), float(case.Fr)
        hmin = float(dm.mesh.tree.h().min())
        self.h = hmin
        if Cn_override is None:
            self.Cn = float(case.Cn)
        elif Cn_override == "2h":
            self.Cn = 2.0 * hmin
        else:
            self.Cn = float(Cn_override)
        # adjoint/chns.py normalisation contract: heavy phi=+1 at 1.0
        self.rho_h, self.rho_l = 1.0, 1.0 / float(case.rho_ratio)
        self.eta_h, self.eta_l = 1.0, 1.0 / float(case.eta_ratio)
        self.ghat = np.zeros(dim)
        self.ghat[-1] = -1.0

        # --- CH brick: quartic [-1,1] double-well MultiPhaseStepper ------
        # chi_aa is a required ctor arg but UNUSED in quartic mode.
        self._ch = MultiPhaseStepper(
            dm, M=1, K=0, chi_aa=np.zeros((2, 2)),
            onsager=[[1.0 / self.Pe]], kappa=[self.Cn ** 2],
            dt=self.dt, bulk="quartic", linsolver="splu",
        )

        # --- NS bricks: Leray internals BY COMPOSITION --------------------
        # Only the constant-coefficient-free helpers are borrowed: the
        # scalar consistent mass M, _gp_vals, _weighted_stiffness,
        # _weak_div_free, dir_nodes/free bookkeeping.  Its step() is
        # never called (constant-nu; the three sub-steps live below).
        self._ns = LerayProjectionStepper(
            dm, nu=1.0 / self.Re, dt=self.dt,
            f_fn=lambda x, t: np.zeros_like(x),
            g_fn=lambda x, t: np.zeros((len(x), dim)),
            order=1, solver="splu",
        )
        self.n_free = self._ns.n_free
        self._M_lu = splu(self._ns.M.tocsc())      # update-solve factor

        # --- state ---------------------------------------------------------
        self._u = np.zeros((self.n_free, dim))
        self._p = np.zeros(self.n_free)
        self.p_star = np.zeros(self.n_free)
        self.t = 0.0
        self.last_newton_iters = 0
        self.last_iters_detail = {"ch": 0, "ns": 0}
        self.last_clamped = 0
        self.last_wall = 0.0

    # ------------------------------------------------------------------
    # protocol properties
    # ------------------------------------------------------------------
    @property
    def phi(self):
        return self._ch.phi(0)

    @property
    def u(self):
        return self._u

    @property
    def p(self):
        return self._p

    # ------------------------------------------------------------------
    def set_initial(self, phi0, u0=None):
        """phi0 [n_free] nodal phase field in [-1, 1]; u0 [n_free, dim]
        (default rest). mu seeded 0 (the MultiPhaseStepper convention)."""
        phi0 = np.asarray(phi0, float)
        self._ch.set_initial([lambda x: phi0])
        self._u = (np.zeros((self.n_free, self.dim)) if u0 is None
                   else np.asarray(u0, float).reshape(self.n_free, self.dim))
        self._p = np.zeros(self.n_free)
        self.p_star = np.zeros(self.n_free)
        self.t = 0.0

    def mass_phi(self):
        """int phi dV via the scalar consistent mass (1^T M phi;
        partition of unity). Diagnostic (the smoke's drift metric)."""
        return float(np.sum(self._ns.M @ self.phi))

    # ------------------------------------------------------------------
    def _l2_rhs(self, gp_scalar_by_bin):
        """int N_a s(x) dV -> free-node vector; s per-bin at GPs [ngp]."""
        dm = self.dm
        rhs = np.zeros(dm.n_nodes)
        for pv, b in dm.bins.items():
            tb = dm.tables_by_p[pv]
            h = dm.mesh.tree.h()[dm.mesh.bins[pv]]
            ne = len(h)
            jac = (h / 2.0) ** dm.dim
            sq = gp_scalar_by_bin[pv].reshape(ne, tb.nqp)
            be = np.einsum("qa,eq,q,e->ea", tb.N, sq, tb.w, jac)
            np.add.at(rhs, dm.mesh.conn_of[pv].ravel(), be.ravel())
        return np.asarray(dm.constraints.T.T @ rhs)

    # ------------------------------------------------------------------
    def step(self):
        """One staggered step (docstring order). Returns the info dict
        {t, newton_iters, clamped, wall_per_step}."""
        t0 = time.perf_counter()
        dm = self.dm
        dim = self.dim
        ndof = dim + 1

        # ---- (1) CH solve at the frozen u^n --------------------------
        adv = self._ns._gp_vals(self._u)          # dict pv -> [ngp, dim]
        self._ch.adv_gp = adv
        x_new, ch_iters, ok = self._ch._attempt(self.dt)
        if not ok:
            raise RuntimeError(
                f"staggered CH Newton failed at t={self.t:.6f} "
                f"(iters={ch_iters})")
        self._ch.x = x_new
        self._ch.hist = x_new.copy()
        self._ch.t += self.dt

        # ---- (2) coefficients at GPs from fresh phi/mu ----------------
        vals, grads = self._ch._pack_fields(self._ch.x)
        aq, gaq = self._ns._gp_vals(self._u, grad=True)
        div_aq, fq_hist, f_body = {}, {}, {}
        nu_q, rho_q, w_rho = {}, {}, {}
        nclamp = 0
        for pv, b in dm.bins.items():
            ne, nqp = len(b["eids"]), b["nqp"]
            phi_gp = vals[pv][:, 0]                # [ngp]
            mu_gp = vals[pv][:, 1]                 # [ngp]
            gphi_gp = grads[pv][:, 0, :]           # [ngp, dim]
            rho_gp, eta_gp, nc = mix_props(
                phi_gp, self.rho_h, self.rho_l, self.eta_h, self.eta_l)
            nclamp += int(nc)
            f_cap = capillary_gp(mu_gp, gphi_gp, self.Cn, self.We)
            f_grav = rho_gp[:, None] * self.ghat[None, :] / self.Fr ** 2
            # BDF1 history load: rho u^n / dt (the var kernel's fq is the
            # FULL momentum load; rho multiplies the time term)
            fq_hist[pv] = rho_gp[:, None] * aq[pv] * self.sigma
            f_body[pv] = (f_cap + f_grav).reshape(ne, nqp, dim)
            nu_q[pv] = (eta_gp / self.Re).reshape(ne, nqp)
            rho_q[pv] = rho_gp.reshape(ne, nqp)
            w_rho[pv] = 1.0 / rho_gp
            # div(u^n) at GPs (trace of the GP gradient)
            g = gaq[pv]                            # [ngp, dim, dim]
            div_aq[pv] = np.einsum("qdd->q", g)

        # ---- (3a) predictor ------------------------------------------
        A, bb = assemble_linear_ns(
            dm, aq, div_aq, fq_hist, nu=1.0 / self.Re,
            sigma=self.sigma, s_skew=0.0,
            nu_q_by_bin=nu_q, rho_q_by_bin=rho_q, f_q_by_bin=f_body)
        # strong rows: no-slip everywhere + pressure pinned to p*
        rows = []
        for i in self._ns.dir_nodes:
            for c in range(dim):
                rows.append(int(i) * ndof + c)
        A = A.tolil()
        for r in rows:
            A.rows[r] = [r]
            A.data[r] = [1.0]
            bb[r] = 0.0
        for i in range(self.n_free):
            r = i * ndof + dim
            A.rows[r] = [r]
            A.data[r] = [1.0]
            bb[r] = self.p_star[i]
        sol = splu(A.tocsr().tocsc()).solve(bb).reshape(self.n_free, ndof)
        uhat = sol[:, :dim]

        # ---- (3b) PPE (1/rho-weighted, node-0 pin) --------------------
        K_w = self._ns._weighted_stiffness(w_rho).tolil()
        rhs = self.sigma * self._ns._weak_div_free(uhat)
        K_w, rhs = self._ns._apply_pin_lil(K_w, rhs)
        phip = splu(K_w.tocsr().tocsc()).solve(rhs)

        # ---- (3c) velocity update + pressure --------------------------
        _, gph = self._ns._gp_vals(phip, grad=True)   # pv -> [ngp, dim]
        du = np.empty_like(uhat)
        for c in range(dim):
            rhs_c = self._l2_rhs(
                {pv: w_rho[pv] * gph[pv][:, c] for pv in gph})
            du[:, c] = -self._M_lu.solve(rhs_c) / self.sigma
        u_new = uhat + du
        u_new[self._ns.dir_nodes] = 0.0               # strong no-slip
        p_new = self.p_star + phip

        # ---- commit ---------------------------------------------------
        self._u = u_new
        self._p = p_new
        self.p_star = p_new.copy()
        self.t += self.dt
        self.last_iters_detail = {"ch": int(ch_iters), "ns": 1}
        self.last_newton_iters = int(ch_iters) + 1
        self.last_clamped = nclamp
        self.last_wall = time.perf_counter() - t0
        return {"t": self.t, "newton_iters": self.last_newton_iters,
                "clamped": self.last_clamped,
                "wall_per_step": self.last_wall}

    # ------------------------------------------------------------------
    def march(self, t_end, snap_every=None):
        """Fixed-dt march to t_end. Returns snapshot dicts (protocol
        keys); snap_every=None snapshots EVERY step, k snapshots every
        k-th step."""
        snaps = []
        k = 0
        while self.t < t_end - 1e-12:
            self.step()
            k += 1
            if snap_every is None or (k % int(snap_every) == 0):
                snaps.append({
                    "t": self.t,
                    "phi": self.phi.copy(),
                    "u": self._u.copy(),
                    "p": self._p.copy(),
                    "wall_per_step": self.last_wall,
                    "newton_iters": self.last_newton_iters,
                    "clamped": self.last_clamped,
                })
        return snaps


class CHNSMonolithicStepper:
    r"""SP-0 Task 7: the MONOLITHIC coupled (u, p, phi, mu) prototype — the
    second spike, parity-gated against ``adjoint.chns.CHNSDiscrete`` (the numpy
    mirror / ground truth) to 1e-10 on converged states.

    Fully-implicit BDF1 Newton on the coupled system, assembled by the compiled
    ``physics.chns.make_chns_newton`` Warp kernel (node-major blk = dim+3,
    fields (u_0..u_{dim-1}, p, phi, mu)); the host interpolates the state to
    Gauss points, launches the kernel for element residual (be = -R) + Jacobian
    (Ae), scatters to a scipy CSR, applies no-slip velocity rows + a node-0
    pressure pin, and solves with splu.  Every term/block mirrors the
    ``CHNSDiscrete._assemble`` contract exactly — including the DIFFERENTIATED
    (Newton-consistent) tau_m (dtau_du, dtau_dphi in-kernel), so the parity is
    against the mirror's exact Jacobian (no frozen-tau relaxation was needed).

    Shared protocol (same names as CHNSStaggeredStepper):
      .step(), .march(t_end, snap_every=None), .phi, .u, .p,
      .last_newton_iters, .last_clamped (= mix_props GP-clamp count inside the
      kernel's coefficient evaluation, computed host-side from phi at GPs — the
      same quantity the mirror surfaces per step), snapshot keys
      t, phi, u, p, wall_per_step, newton_iters, clamped.

    Ctor: CHNSMonolithicStepper(dm, case, dt, linsolver="splu",
                                Cn_override=None, gravity=True,
                                newton_tol=1e-10, newton_max=30)
      Cn_override mirrors CHNSDiscrete: None -> case.Cn, "2h" -> 2*hmin,
      float -> as-is.
    """

    def __init__(self, dm, case, dt, linsolver="splu", Cn_override=None,
                 gravity=True, newton_tol=1e-10, newton_max=30,
                 tstep="bdf1", src_fns=None, body_fn=None, interface="ch"):
        assert linsolver == "splu", \
            "prototype scope: splu only (spike ruling; GPU solves are a " \
            "build-out concern)"
        if tstep not in ("bdf1", "bdf2"):
            raise ValueError(f"tstep must be 'bdf1' or 'bdf2', got {tstep!r}")
        if interface not in ("ch", "cac"):
            raise ValueError(
                f"interface must be 'ch' or 'cac', got {interface!r}")
        self.interface = interface
        import warp as wp
        self._wp = wp
        self.tstep = tstep
        # Forcing (MMS / SP-1 deposition).  src_fns: length-blk list of per-
        # field source fns fn(xq,t)->[ngp] added to the RHS of each residual
        # row (u_0..u_{dim-1}, p, phi, mu).  The CH-phi source rides the
        # kernel's src_phi hook (SP-1 deposition channel); the u/p/mu sources
        # ride a host-side residual correction.  body_fn: fn(xq,t)->[ngp,dim]
        # for the dim momentum rows (NS body-force channel).
        self.src_fns = src_fns
        self.body_fn = body_fn
        # MMS Dirichlet hooks: when set, the strong velocity rows target
        # bc_u_fn(coords,t)[:,d] instead of 0 (the manufactured solution is
        # not no-slip-compatible); bc_phi_fn/bc_mu_fn optionally pin phi/mu on
        # boundary nodes to the manufactured field (Dirichlet CH for MMS).
        self.bc_u_fn = None
        self.bc_phi_fn = None
        self.bc_mu_fn = None
        T = dm.constraints.T
        assert T.shape[0] == T.shape[1] and (
            abs(T - sp.eye(T.shape[0])).nnz == 0), \
            "CHNSMonolithicStepper assumes constraints.T == identity (uniform)"
        self.dm = dm
        self.case = case
        self.dt = float(dt)
        self.dim = dm.dim
        dim = self.dim
        self.gravity = bool(gravity)
        self.newton_tol = float(newton_tol)
        self.newton_max = int(newton_max)

        # --- non-dim parameters (Khanwale form) --------------------------
        self.Re, self.We = float(case.Re), float(case.We)
        self.Pe, self.Fr = float(case.Pe), float(case.Fr)
        self.rho_ratio = float(case.rho_ratio)
        self.eta_ratio = float(case.eta_ratio)
        self.rho_h, self.rho_l = 1.0, 1.0 / self.rho_ratio
        self.eta_h, self.eta_l = 1.0, 1.0 / self.eta_ratio
        self.agg = -0.5 * (self.rho_h - self.rho_l) / self.Pe
        # CAC mobility gamma = 1/Pe (== CH Onsager coeff); beta frozen per
        # Newton iterate (computed host-side in _assemble for interface="cac").
        self.gamma = 1.0 / self.Pe
        self._beta_frozen = 0.0
        hmin = float(dm.mesh.tree.h().min())
        self.h = hmin
        if Cn_override is None:
            self.Cn = float(case.Cn)
        elif Cn_override == "2h":
            self.Cn = 2.0 * hmin
        else:
            self.Cn = float(Cn_override)
        self.cw_inv = 1.0 / (self.Cn * self.We)
        self.grav_scale = 1.0 / self.Fr ** 2
        self.ghat = np.zeros(dim)
        if self.gravity:
            self.ghat[-1] = -1.0

        self.blk = dim + 3
        self.nn = dm.n_nodes
        self.ndof = self.blk * self.nn

        # --- element scaffolding (mirror of CHNSDiscrete) ----------------
        self.bins = []
        for pv, b in dm.bins.items():
            tb = dm.tables_by_p[pv]
            conn = dm.mesh.conn_of[pv].astype(np.int64)
            h = np.asarray(dm.mesh.tree.h()[dm.mesh.bins[pv]], np.float64)
            N = np.asarray(tb.N, np.float64)          # [nqp, nbf]
            dN = np.asarray(tb.dN, np.float64)        # [nqp, nbf, dim]
            w = np.asarray(tb.w, np.float64)          # [nqp]
            ne, nbf = conn.shape
            nqp = N.shape[0]
            dscale = 2.0 / h
            gdof = (conn[:, :, None] * self.blk
                    + np.arange(self.blk)[None, None, :]).reshape(
                        ne, self.blk * nbf)
            self.bins.append(dict(
                pv=pv, conn=conn, N=N, dN=dN, w=w, h=h, ne=ne, nbf=nbf,
                nqp=nqp, dscale=dscale, gdof=gdof,
                # device basis tables (from DeviceMesh bin)
                conn_d=b["conn"], h_d=b["h"], N_d=b["N"], dN_d=b["dN"],
                w_d=b["w"]))

        self.bnd = np.asarray(dm.mesh.boundary_nodes, bool)
        self.coords = np.asarray(dm.mesh.node_coords, np.float64)

        # --- state --------------------------------------------------------
        self._u = np.zeros((self.nn, dim))
        self._p = np.zeros(self.nn)
        self._phi = np.zeros(self.nn)
        self._mu = np.zeros(self.nn)
        self.u_n = np.zeros((self.nn, dim))
        self.phi_n = np.zeros(self.nn)
        # A4b variable-step BDF2 history (u^{n-1}, phi^{n-1}); None => BDF1
        # bootstrap on the first step.  dt_prev = dt of the last committed step.
        self.u_nm1 = None
        self.phi_nm1 = None
        self.dt_prev = None
        self.t = 0.0
        self._t_eval = float(self.dt)     # t^{n+1} for MMS source
        self._lumped = None
        self.last_newton_iters = 0
        self.last_clamped = 0
        self.last_wall = 0.0

        # physical GP coords per bin (for MMS / SP-1 source evaluation)
        for B in self.bins:
            node_xy = self.coords[B["conn"]]              # [ne, nbf, dim]
            B["xq"] = np.einsum("qa,ead->eqd", B["N"], node_xy)  # [ne,nqp,dim]

        self._kernel = make_chns_newton(
            self.bins[0]["nbf"], self.bins[0]["nqp"], dim,
            interface=self.interface)

    # ------------------------------------------------------------------
    # protocol properties
    # ------------------------------------------------------------------
    @property
    def phi(self):
        return self._phi

    @property
    def u(self):
        return self._u

    @property
    def p(self):
        return self._p

    # ------------------------------------------------------------------
    def set_initial(self, phi0, u0=None):
        """phi0 [nn] nodal phase field in [-1, 1]; u0 [nn, dim] (rest
        default). mu seeded 0, p seeded 0 (the CHNSDiscrete convention)."""
        dim = self.dim
        self._phi = np.asarray(phi0, np.float64).copy()
        self._mu = np.zeros(self.nn)
        self._p = np.zeros(self.nn)
        self._u = (np.zeros((self.nn, dim)) if u0 is None
                   else np.asarray(u0, np.float64).reshape(self.nn, dim))
        self.u_n = self._u.copy()
        self.phi_n = self._phi.copy()
        self.t = 0.0

    def set_history(self, u_n, phi_n):
        self.u_n = np.asarray(u_n, np.float64).reshape(
            self.nn, self.dim).copy()
        self.phi_n = np.asarray(phi_n, np.float64).copy()

    # ------------------------------------------------------------------
    # pack / unpack (node-major, matches CHNSDiscrete)
    # ------------------------------------------------------------------
    def pack(self):
        x = np.zeros(self.ndof)
        blk, dim = self.blk, self.dim
        for d in range(dim):
            x[d::blk] = self._u[:, d]
        x[dim::blk] = self._p
        x[dim + 1::blk] = self._phi
        x[dim + 2::blk] = self._mu
        return x

    def unpack(self, x):
        blk, dim = self.blk, self.dim
        u = np.stack([x[d::blk] for d in range(dim)], axis=1)
        p = x[dim::blk]
        phi = x[dim + 1::blk]
        mu = x[dim + 2::blk]
        return u, p, phi, mu

    # ------------------------------------------------------------------
    def lumped_mass(self):
        if self._lumped is not None:
            return self._lumped
        m = np.zeros(self.nn)
        for B in self.bins:
            jac = (B["h"] / 2.0) ** self.dim
            dJxW = B["w"][None, :] * jac[:, None]
            Ma = np.einsum("eq,qa->ea", dJxW, B["N"])
            np.add.at(m, B["conn"].ravel(), Ma.ravel())
        self._lumped = m
        return m

    def mass_phi(self):
        """int phi dV via lumped mass (partition of unity)."""
        return float(self.lumped_mass() @ self._phi)

    def _cac_beta(self, phi):
        r"""Source-respecting CAC Lagrange multiplier from the current phi
        (a cheap GP reduction; mirrors adjoint.chns.CHNSDiscrete._cac_beta):
            beta = -( Int F'(phi)/Cn ) / ( Int sqrt(F(phi)) ),
        F = 1/4(phi^2-1)^2, F' = phi^3 - phi.  Frozen per Newton iterate."""
        Cn = self.Cn
        num = 0.0
        den = 0.0
        for B in self.bins:
            phi_gp, _ = self._interp(phi, B)              # [e, q]
            Fp = phi_gp ** 3 - phi_gp
            F = 0.25 * (phi_gp ** 2 - 1.0) ** 2
            jac = (B["h"] / 2.0) ** self.dim
            dJxW = B["w"][None, :] * jac[:, None]
            num += float(np.sum(dJxW * (Fp / Cn)))
            den += float(np.sum(dJxW * np.sqrt(np.maximum(F, 0.0))))
        den = den if abs(den) > 1e-300 else 1e-300
        return -num / den

    # ------------------------------------------------------------------
    # field interpolation to GPs (scalar + vector), matching the mirror
    # ------------------------------------------------------------------
    def _interp(self, field, B):
        vals = field[B["conn"]]                        # [ne, nbf]
        v = np.einsum("qa,ea->eq", B["N"], vals)       # [ne, nqp]
        g = np.einsum("qad,ea,e->eqd", B["dN"], vals, B["dscale"])
        return v, g

    def _interp_vec(self, field, B):
        vals = field[B["conn"]]                        # [ne, nbf, dim]
        v = np.einsum("qa,ead->eqd", B["N"], vals)
        g = np.einsum("qas,ead,e->eqds", B["dN"], vals, B["dscale"])
        return v, g

    # ------------------------------------------------------------------
    # A4b variable-step BDF2 effective time term (module docstring of
    # multiphase.py): the kernel's time term is (x - xn)/dt.  For BDF2 we
    # want sigma*x - hist with sigma = a/dt, hist = (b x^n - c x^{n-1})/dt,
    # a = (1+2r)/(1+r), b = 1+r, c = r^2/(1+r), r = dt/dt_prev.  Setting
    # dt_eff = 1/sigma and xn_eff = hist/sigma reproduces this on the SAME
    # kernel (no kernel change): (x - xn_eff)/dt_eff = sigma*x - hist.
    # First step (dt_prev None) bootstraps BDF1: dt_eff=dt, xn_eff=x^n.
    # ------------------------------------------------------------------
    def _bdf_time(self):
        if self.tstep == "bdf2" and self.dt_prev is not None \
                and self.u_nm1 is not None:
            r = self.dt / self.dt_prev
            sigma = (1.0 + 2.0 * r) / (1.0 + r) / self.dt
            bv = 1.0 + r
            cv = r * r / (1.0 + r)
            dt_eff = 1.0 / sigma
            u_n_eff = (bv * self.u_n - cv * self.u_nm1) / self.dt / sigma
            phi_n_eff = (bv * self.phi_n - cv * self.phi_nm1) / self.dt / sigma
            return dt_eff, u_n_eff, phi_n_eff
        return self.dt, self.u_n, self.phi_n

    # ------------------------------------------------------------------
    # residual + jacobian via the Warp kernel
    # ------------------------------------------------------------------
    def _assemble(self, x, want_jac):
        wp = self._wp
        u, p, phi, mu = self.unpack(x)
        blk, dim = self.blk, self.dim
        d = self.dm.device
        arr = lambda a_: wp.array(np.ascontiguousarray(a_, np.float64),
                                  dtype=wp.float64, device=d)
        ghat_d = arr(self.ghat)
        dt_eff, u_n_eff, phi_n_eff = self._bdf_time()
        # CAC: freeze beta per Newton iterate from the current iterate's phi
        # (both residual+jac calls within a Newton step see the same x -> the
        # frozen-beta contract, no dbeta/dphi block, matching the mirror).
        if self.interface == "cac":
            self._beta_frozen = self._cac_beta(phi)
        beta = self._beta_frozen
        R = np.zeros(self.ndof)
        rows, cols, valsK = [], [], []
        n_clamp = 0
        n_gp = 0
        have_src = (self.src_fns is not None) or (self.body_fn is not None)
        for B in self.bins:
            ne, nbf, nqp = B["ne"], B["nbf"], B["nqp"]
            n_gp += ne * nqp
            u_gp, gu = self._interp_vec(u, B)          # [e,q,d],[e,q,d,s]
            un_gp, _ = self._interp_vec(u_n_eff, B)
            p_gp, gp = self._interp(p, B)
            phi_gp, gphi = self._interp(phi, B)
            phin_gp, _ = self._interp(phi_n_eff, B)
            mu_gp, gmu = self._interp(mu, B)
            ngp = ne * nqp
            # clamp count (mix_props on phi at GPs) — host-side diagnostic
            _, _, nc = mix_props(phi_gp.ravel(), self.rho_h, self.rho_l,
                                 self.eta_h, self.eta_l)
            n_clamp += int(nc)

            # --- MMS / SP-1 sources at GPs (t^{n+1}) -----------------------
            src_phi = np.zeros(ngp)
            fbody = np.zeros((ngp, dim))
            xq = B["xq"].reshape(ngp, dim)
            if have_src and self.src_fns is not None \
                    and len(self.src_fns) > dim + 1 \
                    and self.src_fns[dim + 1] is not None:
                src_phi = np.asarray(
                    self.src_fns[dim + 1](xq, self._t_eval), float)
            if have_src and self.body_fn is not None:
                fbody = np.asarray(self.body_fn(xq, self._t_eval), float)
                fbody = fbody.reshape(ngp, dim)

            Ae = wp.zeros((ne, blk * nbf, blk * nbf), dtype=wp.float64,
                          device=d)
            be = wp.zeros((ne, blk * nbf), dtype=wp.float64, device=d)
            wp.launch(self._kernel, dim=ne, inputs=[
                B["conn_d"], B["h_d"], B["N_d"], B["dN_d"], B["w_d"],
                arr(u_gp.reshape(ngp, dim)),
                arr(gu.reshape(ngp, dim, dim)),
                arr(p_gp.reshape(ngp)),
                arr(gp.reshape(ngp, dim)),
                arr(phi_gp.reshape(ngp)),
                arr(gphi.reshape(ngp, dim)),
                arr(mu_gp.reshape(ngp)),
                arr(gmu.reshape(ngp, dim)),
                arr(un_gp.reshape(ngp, dim)),
                arr(phin_gp.reshape(ngp)),
                arr(src_phi.reshape(ngp)),
                arr(fbody.reshape(ngp, dim)),
                ghat_d,
                wp.float64(dt_eff), wp.float64(self.Re),
                wp.float64(self.We), wp.float64(self.Cn),
                wp.float64(self.Pe), wp.float64(self.cw_inv),
                wp.float64(self.agg), wp.float64(self.grav_scale),
                wp.float64(self.gamma), wp.float64(beta),
                wp.float64(self.rho_h), wp.float64(self.rho_l),
                wp.float64(self.eta_h), wp.float64(self.eta_l),
                Ae, be], device=d)
            beh = be.numpy()                           # be = -R
            gdof = B["gdof"]
            # residual R = -be  (kernel emits be = -residual)
            np.add.at(R, gdof.ravel(), (-beh).ravel())

            # --- host-side p/mu (and explicit u-row) src corrections -------
            # The phi-row source rides the kernel (src_phi_gp) and the momentum
            # body force rides the kernel (fbody_gp); the continuity, mu-row and
            # any explicit u-row src_fns entries ride here as -INT N_a S_f
            # (pure RHS, no Jacobian).
            if self.src_fns is not None:
                N = B["N"]
                jac = (B["h"] / 2.0) ** dim
                dJxW = B["w"][None, :] * jac[:, None]    # [ne, nqp]
                for f in range(blk):
                    if f == dim + 1:
                        continue                         # phi -> kernel hook
                    if f >= len(self.src_fns) or self.src_fns[f] is None:
                        continue
                    sf = np.asarray(self.src_fns[f](xq, self._t_eval), float)
                    if not np.any(sf):
                        continue
                    Rsrc = -np.einsum("eq,qa,eq->ea", dJxW, N,
                                      sf.reshape(ne, nqp))
                    np.add.at(R, gdof[:, f::blk].ravel(), Rsrc.ravel())

            if want_jac:
                Aeh = Ae.numpy()
                rows.append(np.repeat(gdof, blk * nbf, axis=1).ravel())
                cols.append(np.tile(gdof, (1, blk * nbf)).ravel())
                valsK.append(Aeh.ravel())
        J = None
        if want_jac:
            J = sp.coo_matrix(
                (np.concatenate(valsK),
                 (np.concatenate(rows), np.concatenate(cols))),
                shape=(self.ndof, self.ndof)).tocsr()
        self._last_clamp = n_clamp
        self._last_ngp = n_gp
        return R, J

    # ------------------------------------------------------------------
    # boundary conditions (no-slip velocity strong, node-0 pressure pin)
    # ------------------------------------------------------------------
    def _bc_rows(self):
        blk, dim = self.blk, self.dim
        rows = []
        for a in np.where(self.bnd)[0]:
            for dd in range(dim):
                rows.append(a * blk + dd)
        rows.append(0 * blk + dim)          # pressure pin at node 0
        if self.bc_phi_fn is not None:
            for a in np.where(self.bnd)[0]:
                rows.append(a * blk + dim + 1)
        if self.bc_mu_fn is not None:
            for a in np.where(self.bnd)[0]:
                rows.append(a * blk + dim + 2)
        return np.asarray(rows, np.int64)

    def _bc_target(self, x):
        """Dirichlet target vector t so the strong rows enforce x_row = t_row.
        Default: u=0 no-slip, p_node0=0.  With MMS hooks, boundary velocity /
        phi / mu target the manufactured field at t^{n+1}."""
        blk, dim = self.blk, self.dim
        rows = self._bc_rows()
        tgt = np.zeros_like(x)              # default: u=0, p_node0=0 targets
        if self.bc_u_fn is not None:
            ub = np.asarray(self.bc_u_fn(self.coords, self._t_eval), float)
            for a in np.where(self.bnd)[0]:
                for dd in range(dim):
                    tgt[a * blk + dd] = ub[a, dd]
        if self.bc_phi_fn is not None:
            pb = np.asarray(self.bc_phi_fn(self.coords, self._t_eval), float)
            for a in np.where(self.bnd)[0]:
                tgt[a * blk + dim + 1] = pb[a]
        if self.bc_mu_fn is not None:
            mb = np.asarray(self.bc_mu_fn(self.coords, self._t_eval), float)
            for a in np.where(self.bnd)[0]:
                tgt[a * blk + dim + 2] = mb[a]
        return rows, tgt

    def _apply_bc(self, x, R, J):
        rows, tgt = self._bc_target(x)
        R[rows] = x[rows] - tgt[rows]       # Newton row: x_row -> tgt_row
        J = J.tolil()
        for r in rows:
            J.rows[r] = [int(r)]
            J.data[r] = [1.0]
        return R, J.tocsr()

    # ------------------------------------------------------------------
    # one BDF1 Newton step (mirrors CHNSDiscrete.step exactly)
    # ------------------------------------------------------------------
    def step(self):
        t0 = time.perf_counter()
        self._t_eval = self.t + self.dt      # MMS source / BC at t^{n+1}
        x = self.pack()
        it = 0
        rnorm = np.inf
        for it in range(1, self.newton_max + 1):
            R, J = self._assemble(x, want_jac=True)
            R, J = self._apply_bc(x, R, J)
            rnorm = float(np.linalg.norm(R))
            if rnorm < self.newton_tol:
                break
            dx = splu(J.tocsc()).solve(-R)
            x = x + dx
            if float(np.abs(dx).max()) < self.newton_tol:
                R, _ = self._assemble(x, want_jac=False)
                rows, tgt = self._bc_target(x)
                R[rows] = x[rows] - tgt[rows]
                rnorm = float(np.linalg.norm(R))
                break
        else:
            raise RuntimeError(
                f"CHNS monolithic Newton failed to converge in "
                f"{self.newton_max} iters: residual norm {rnorm:.3e}")

        clamped = int(getattr(self, "_last_clamp", 0))
        ngp = int(getattr(self, "_last_ngp", 1))
        if clamped > 0.01 * ngp:
            raise RuntimeError(
                f"mix_props clamp counter {clamped} > 1% of {ngp} GPs")

        u, p, phi, mu = self.unpack(x)
        # BDF2 history: shift x^{n-1} <- x^n BEFORE overwriting x^n <- x^{n+1}
        self.u_nm1 = self.u_n.copy()
        self.phi_nm1 = self.phi_n.copy()
        self.dt_prev = self.dt
        self.u_n = u.copy()
        self.phi_n = phi.copy()
        self._u, self._p, self._phi, self._mu = u, p, phi, mu
        self.t += self.dt
        self.last_newton_iters = int(it)
        self.last_clamped = clamped
        self.last_wall = time.perf_counter() - t0
        return {"t": self.t, "newton_iters": self.last_newton_iters,
                "clamped": self.last_clamped,
                "wall_per_step": self.last_wall}

    # ------------------------------------------------------------------
    def energy(self):
        r"""Discrete free energy E = E_kin + E_interface + E_bulk (GP-integrated).

          E_kin       = INT (1/2) rho |u|^2                  (kinetic)
          E_interface = (1/We) INT (Cn^2/2) |grad phi|^2     (gradient / capillary)
          E_bulk      = (1/We) INT (1/4)(phi^2 - 1)^2        (double-well)

        The CH part carries the 1/We surface-tension scaling of the
        non-dimensional CHNS free energy.  For an UNFORCED closed system with
        no gravity, this quantity is non-increasing between steps (the
        structural-diagnostic gate).  Returns a dict of the three parts + total.
        """
        e_kin = e_int = e_bulk = 0.0
        for B in self.bins:
            jac = (B["h"] / 2.0) ** self.dim
            dJxW = B["w"][None, :] * jac[:, None]           # [ne, nqp]
            u_gp, _ = self._interp_vec(self._u, B)          # [e,q,d]
            phi_gp, gphi = self._interp(self._phi, B)       # [e,q], [e,q,d]
            rho_gp, _, _ = mix_props(phi_gp.ravel(), self.rho_h, self.rho_l,
                                     self.eta_h, self.eta_l)
            rho_gp = rho_gp.reshape(phi_gp.shape)
            e_kin += float(np.sum(dJxW * 0.5 * rho_gp
                                  * np.sum(u_gp ** 2, axis=-1)))
            e_int += float(np.sum(dJxW * 0.5 * self.Cn ** 2
                                  * np.sum(gphi ** 2, axis=-1))) / self.We
            e_bulk += float(np.sum(dJxW * 0.25
                                   * (phi_gp ** 2 - 1.0) ** 2)) / self.We
        return {"kinetic": e_kin, "interface": e_int, "bulk": e_bulk,
                "total": e_kin + e_int + e_bulk}

    # ------------------------------------------------------------------
    def march(self, t_end, snap_every=None):
        snaps = []
        k = 0
        while self.t < t_end - 1e-12:
            self.step()
            k += 1
            if snap_every is None or (k % int(snap_every) == 0):
                snaps.append({
                    "t": self.t,
                    "phi": self._phi.copy(),
                    "u": self._u.copy(),
                    "p": self._p.copy(),
                    "wall_per_step": self.last_wall,
                    "newton_iters": self.last_newton_iters,
                    "clamped": self.last_clamped,
                })
        return snaps


def CHNSStepper(dm, case, dt, mode="auto", tstep="bdf1", linsolver="splu",
                Cn_override=None, gravity=True, src_fns=None, body_fn=None,
                interface="ch", **kwargs):
    r"""SP-0 Task 9 facade — the ONE public entry point for CHNS forward
    marching.  Thin dispatcher over the two reviewed coupling prototypes; the
    coupling decision (docs/dev/2026-08-10-sp0-coupling-decision.md, Baskar
    2026-08-10) is baked in: ``mode="auto"`` -> MONOLITHIC (the adjoint-bearing
    primary mode).

    Parameters
    ----------
    dm, case, dt : DeviceMesh, CHNSCase, float
        Uniform Q1 mesh (constraints.T == identity), non-dim case, fixed step.
    mode : {"auto", "monolithic", "staggered"}
        "auto"/"monolithic" -> ``CHNSMonolithicStepper`` (the decided primary,
        adjoint-bearing, machine-exact mass conservation, BDF1/BDF2).
        "staggered" -> ``CHNSStaggeredStepper``, the DOCUMENTED FORWARD-ONLY
        fast mode (retained per the decision memo; 4.8x faster per step, no
        adjoint, secular mass drift).  It is BDF1-only: passing tstep="bdf2"
        with mode="staggered" raises.
    tstep : {"bdf1", "bdf2"}
        Time scheme (monolithic only).  "bdf2" is variable-step BDF2 with the
        MultiPhaseStepper A4b coefficients, BDF1-bootstrapped on the first step.
    linsolver : {"splu"}
        Prototype scope (spike ruling); GPU solves are a later build-out.
    Cn_override : {None, "2h", float}
        None -> case.Cn; "2h" -> 2*hmin (the Cn >= h resolvability convention
        the coupling-decision memo's guard finding binds benchmark configs to);
        float -> as-is.
    src_fns : list or None
        SP-1 deposition / MMS forcing hook.  Length-blk (= dim+3) list of
        per-field source functions fn(xq[ngp,dim], t) -> [ngp] added to the RHS
        of each residual row (u_0..u_{dim-1}, p, phi, mu).  The CH-phi entry
        (index dim+1) rides the monolithic kernel's src hook (the deposition
        channel); the others ride a host-side residual correction.  Static per
        step, evaluated at t^{n+1}.  (Ignored by the staggered forward mode.)
    body_fn : callable or None
        NS body-force channel fn(xq, t) -> [ngp, dim] added to the dim momentum
        rows (MMS convenience; monolithic only).
    interface : {"ch", "cac"}
        Interface model (monolithic only; the staggered fast mode is CH-only
        and raises on "cac").  "ch" (default) -> Cahn-Hilliard 4-field brick.
        "cac" -> Conservative Allen-Cahn (spec §3): source-respecting Lagrange
        multiplier, trivial-mu row, Korteweg surface tension.  See the A/B
        decision memo docs/dev/2026-08-10-sp0-interface-decision.md.

    Returns
    -------
    A stepper instance implementing the shared protocol (.step, .march, .phi,
    .u, .p, .energy [monolithic], .mass_phi, snapshot dicts).
    """
    if mode in ("auto", "monolithic"):
        return CHNSMonolithicStepper(
            dm, case, dt, linsolver=linsolver, Cn_override=Cn_override,
            gravity=gravity, tstep=tstep, src_fns=src_fns, body_fn=body_fn,
            interface=interface, **kwargs)
    if mode == "staggered":
        if interface != "ch":
            raise ValueError(
                "interface='cac' (Conservative Allen-Cahn) is wired on the "
                "monolithic mode only (the decided primary; spec §3); the "
                "staggered fast mode is CH-only.  Use mode='monolithic'.")
        if tstep != "bdf1":
            raise ValueError(
                "CHNSStaggeredStepper is BDF1-only (documented forward-only "
                f"fast mode); got tstep={tstep!r}.  Use mode='monolithic' for "
                "BDF2.")
        if src_fns is not None or body_fn is not None:
            raise ValueError(
                "src_fns/body_fn (SP-1 deposition / MMS forcing) are wired on "
                "the monolithic mode only; the staggered fast mode does not "
                "carry the source hook.")
        return CHNSStaggeredStepper(
            dm, case, dt, linsolver=linsolver, Cn_override=Cn_override,
            **kwargs)
    raise ValueError(
        f"mode must be 'auto', 'monolithic', or 'staggered', got {mode!r}")
