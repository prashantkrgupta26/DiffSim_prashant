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
from scipy.sparse.linalg import splu

from ..api.ns_bricks import assemble_linear_ns
from ..physics.chns import mix_props, capillary_gp
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
