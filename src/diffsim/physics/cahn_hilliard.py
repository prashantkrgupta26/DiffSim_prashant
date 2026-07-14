r"""M4 track (a): Cahn-Hilliard mixed (c, mu) brick — group guide Sec 2.3.

See src/diffsim/api/example_bricks.py for the Poisson walk-through that
introduces the strong -> weak -> discrete -> code recipe; this file applies the
same recipe to a conserved phase field.

-------------------------------------------------------------------------------
The Cahn-Hilliard problem, from strong form to code
-------------------------------------------------------------------------------

PHYSICS.  A conserved order parameter c (a composition) relaxes to minimize the
Ginzburg-Landau free energy  F[c] = Int [ f(c) + (kap/2) |grad c|^2 ] dV. Two
bulk free energies are supported (energy= in the stepper):

  "poly" (default): the double-well  f(c) = (1/4)(c^2 - 1)^2,
      f'(c) = c^3 - c,  f''(c) = 3c^2 - 1,  c in [-1, 1].
  "fh": the Flory-Huggins logarithmic form (Wodo & Ganapathysubramanian,
      JCP 230 (2011) 6037, Eq. 4), phi in (0, 1):
      f(phi) = A [phi ln phi + (1-phi) ln(1-phi)] + B phi (1-phi)
      f'     = A [ln phi - ln(1-phi)] + B (1 - 2 phi)
      f''    = A [1/phi + 1/(1-phi)] - 2B
      The logs enter through the C1-regularized _rlog/_rinv shared with
      the ternary brick (linear extension below 1e-4 keeps a GROWING
      restoring force at the walls — the hard-clamp lesson). Note the
      curvature split: the entropic part A(1/phi + 1/(1-phi)) >= 4A is
      strictly convex; all the destabilization is the CONSTANT -2B
      (spinodal where A(1/phi + 1/(1-phi)) < 2B).

Conserved gradient flow gives a 4th-order PDE, which we split into two
2nd-order equations by introducing the chemical potential mu = dF/dc:

        c_t = div( M grad mu )          (mass balance; M = mobility)
        mu  = f'(c) - kap div(grad c)   (chemical potential)

MIXED WEAK FORM.  Test the first equation with v, the second with q, and
integrate the divergence/Laplacian terms by parts. With natural no-flux
boundaries (grad mu . n = grad c . n = 0) the surface terms vanish:

    R_c(v)  = Int v c_t dV        + M   Int grad v . grad mu dV          = 0
    R_mu(q) = Int q mu dV - Int q f'(c) dV - kap Int grad q . grad c dV  = 0

    R_c  couples c to mu through the mobility (the conserved transport);
    R_mu  is the definition of mu, with kap grad q . grad c the interface term.

TIME + NEWTON.  BDF1/BDF2 turn c_t into (sigma c - hist), where sigma = b0/dt
and hist = sum_j b_j c_{n-j}/dt (carried in per-Gauss-point). The system is
nonlinear through f'(c); we solve it by monolithic Newton, linearizing about the
current iterate (c_k, mu_k) with f''(c) = 3 c^2 - 1. The 2x2 node block
[ dR_c/dc   dR_c/dmu ;  dR_mu/dc   dR_mu/dmu ] is, per (test a, trial b):

    dR_c/dc   = sigma  Int N_a N_b                 (the mass/time term)
    dR_c/dmu  = M      Int grad N_a . grad N_b      (mobility transport)
    dR_mu/dc  = -f''(c) Int N_a N_b - kap Int grad N_a . grad N_b
    dR_mu/dmu = Int N_a N_b

DOF LAYOUT.  Node-major 2-dof (c, mu): global block index 2*a+0 is the c row of
node a, 2*a+1 is its mu row. The kernel below writes both residual entries
(be = -R) and all four Jacobian blocks at each Gauss point.
"""
import numpy as np
import scipy.sparse as sp
import warp as wp

from ..assembly.operators import _kernel_cache


def make_ch_newton(nbf: int, nqp: int, dim: int, energy: str = "poly"):
    key = ("ch_newton", nbf, nqp, dim, energy)
    if key in _kernel_cache:
        return _kernel_cache[key]
    dim_pow = float(dim)
    fh = energy == "fh"
    if fh:
        from .ternary_ch import _rlog, _rinv

    @wp.kernel(module="unique", enable_backward=False,
               module_options=({"max_unroll": 0}
                               if (dim >= 3 or nbf > 4) else {}))
    def ch_k(conn: wp.array2d(dtype=wp.int32),
             h: wp.array(dtype=wp.float64),
             Ntab: wp.array2d(dtype=wp.float64),
             dNtab: wp.array3d(dtype=wp.float64),
             wtab: wp.array(dtype=wp.float64),
             ck: wp.array(dtype=wp.float64),      # c_k at GPs
             gck: wp.array2d(dtype=wp.float64),   # grad c_k
             muk: wp.array(dtype=wp.float64),     # mu_k
             gmuk: wp.array2d(dtype=wp.float64),  # grad mu_k
             hist: wp.array(dtype=wp.float64),    # sum ch_j c_{n-j}/dt
             fcq: wp.array(dtype=wp.float64),     # MMS source in R_c
             fmq: wp.array(dtype=wp.float64),     # MMS source in R_mu
             Mmob: wp.float64, kap: wp.float64, sigma: wp.float64,
             pA: wp.float64, pB: wp.float64,      # FH (A, B); poly ignores
             Ae: wp.array3d(dtype=wp.float64),
             be: wp.array2d(dtype=wp.float64)):
        e = wp.tid()
        he = h[e]
        jac = wp.pow(he * wp.float64(0.5), wp.float64(dim_pow))
        dscale = wp.float64(2.0) / he
        for q in range(nqp):
            dJxW = wtab[q] * jac
            gp = e * nqp + q
            c_ = ck[gp]
            if wp.static(fh):
                # f'  = A[ln c - ln(1-c)] + B(1-2c)   (regularized logs)
                # f'' = A[1/c + 1/(1-c)] - 2B
                fp = pA * (_rlog(c_) - _rlog(wp.float64(1.0) - c_)) \
                    + pB * (wp.float64(1.0) - wp.float64(2.0) * c_)
                dfdc = pA * (_rinv(c_) + _rinv(wp.float64(1.0) - c_)) \
                    - wp.float64(2.0) * pB
            else:
                fp = c_ * c_ * c_ - c_
                dfdc = wp.float64(3.0) * c_ * c_ - wp.float64(1.0)
            for a in range(nbf):
                Na = Ntab[q, a]
                # residuals (be = -r). gv_gmu = grad N_a . grad mu (the
                # mobility term of R_c); gv_gc = grad N_a . grad c (the
                # interface term of R_mu).
                gv_gmu = wp.float64(0.0)
                gv_gc = wp.float64(0.0)
                for d in range(dim):
                    gv_gmu += dNtab[q, a, d] * dscale * gmuk[gp, d]
                    gv_gc += dNtab[q, a, d] * dscale * gck[gp, d]
                # R_c = Int v c_t + M Int grad v . grad mu   (v = N_a);
                #   c_t -> sigma*c - hist (BDF); fcq is the MMS source.
                r_c = (Na * (sigma * c_ - hist[gp]) + Mmob * gv_gmu
                       - Na * fcq[gp]) * dJxW
                # R_mu = Int q mu - Int q f'(c) - kap Int grad q . grad c
                #   (q = N_a);  fp = f'(c) per the energy; fmq = MMS source.
                r_m = (Na * (muk[gp] - fp)
                       - Na * fmq[gp]) * dJxW - kap * gv_gc * dJxW
                wp.atomic_add(be, e, 2 * a + 0, -r_c)
                wp.atomic_add(be, e, 2 * a + 1, -r_m)
                for b in range(nbf):
                    Nb = Ntab[q, b]
                    # lap = grad N_a . grad N_b, shared by the transport
                    # (dR_c/dmu) and interface (dR_mu/dc) Jacobian terms.
                    lap = wp.float64(0.0)
                    for d in range(dim):
                        lap += dNtab[q, a, d] * dNtab[q, b, d] \
                            * dscale * dscale
                    # dR_c/dc = sigma N_a N_b ;  dR_c/dmu = M grad N_a . grad N_b
                    wp.atomic_add(Ae, e, 2 * a, 2 * b,
                                  sigma * Na * Nb * dJxW)
                    wp.atomic_add(Ae, e, 2 * a, 2 * b + 1,
                                  Mmob * lap * dJxW)
                    # dR_mu/dc = -f''(c) N_a N_b - kap grad N_a . grad N_b ;
                    # dR_mu/dmu = N_a N_b   (dfdc = f''(c) per the energy)
                    wp.atomic_add(Ae, e, 2 * a + 1, 2 * b,
                                  (-dfdc * Na * Nb - kap * lap) * dJxW)
                    wp.atomic_add(Ae, e, 2 * a + 1, 2 * b + 1,
                                  Na * Nb * dJxW)

    _kernel_cache[key] = ch_k
    return ch_k


def np_rlog(x, eps=1e-4):
    """numpy mirror of ternary_ch._rlog (C1 linear extension below eps),
    for energy functionals in tests/diagnostics."""
    x = np.asarray(x, dtype=np.float64)
    return np.where(x < eps, np.log(eps) + (x - eps) / eps,
                    np.log(np.maximum(x, eps)))


def fh_bulk_energy(c, A=1.0, B=3.0):
    """Flory-Huggins bulk density f(c) = A[c ln c + (1-c)ln(1-c)] + Bc(1-c)
    with the SAME regularized log the kernel uses."""
    c = np.asarray(c, dtype=np.float64)
    return A * (c * np_rlog(c) + (1.0 - c) * np_rlog(1.0 - c)) \
        + B * c * (1.0 - c)


class CahnHilliardStepper:
    def __init__(self, dm, M, kappa, dt, order=2, fc_fn=None, fm_fn=None,
                 dirichlet=None, gc_fn=None, gm_fn=None,
                 newton_tol=1e-10, newton_max=15,
                 energy="poly", fh_A=1.0, fh_B=3.0, linsolver="splu"):
        from ..physics.poisson import gauss_points
        self.dm, self.M, self.kappa, self.dt = dm, M, kappa, dt
        self.order = order
        assert energy in ("poly", "fh"), energy
        self.energy, self.fh_A, self.fh_B = energy, float(fh_A), float(fh_B)
        self.linsolver = linsolver
        self._solver_cache = {}
        z = lambda x, t: np.zeros(len(x))
        self.fc_fn, self.fm_fn = fc_fn or z, fm_fn or z
        self.dirichlet, self.gc_fn, self.gm_fn = dirichlet, gc_fn, gm_fn
        self.newton_tol, self.newton_max = newton_tol, newton_max
        self.mesh, self.cons = dm.mesh, dm.constraints
        self.Tc = self.cons.T.tocsr()
        self.T2 = sp.kron(self.Tc, sp.identity(2, format="csr"),
                          format="csr").tocsr()
        self.free_coords = self.mesh.node_coords[self.cons.free_nodes]
        self.xq = gauss_points(self.mesh, dm.tables_by_p)
        self.nfree = self.Tc.shape[1]
        self.t = 0.0
        self.dt_prev = None     # dt of the last completed step (G3)

    def set_initial(self, c0_fn, mu_init="zero"):
        """mu_init='consistent' seeds mu0 from the lumped weak potential
        mu0 = [Int N f'(c0) + kap Int grad N . grad c0] / Ml — removing the
        step-0 transient (poly gate measured E 0.25 -> 173 -> 0.61 with
        mu=0; for FH the same transient Newton-overshoots c into the walls
        where f'' hits the regularization cap 1/eps = 1e4)."""
        c0 = c0_fn(self.free_coords)
        self.hist = [c0.copy(), c0.copy()]
        self.x = np.zeros(self.nfree * 2)
        self.x[0::2] = c0
        if mu_init == "consistent":
            self.x[1::2] = self._consistent_mu(c0)
        self.t = 0.0
        self.dt_prev = None     # restart the BDF2 bootstrap (G3)
        return c0

    def _fprime_np(self, c):
        if self.energy == "fh":
            eps = 1e-4
            rlog = lambda x: np.where(x < eps,
                                      np.log(eps) + (x - eps) / eps,
                                      np.log(np.maximum(x, eps)))
            return self.fh_A * (rlog(c) - rlog(1.0 - c)) \
                + self.fh_B * (1.0 - 2.0 * c)
        return c ** 3 - c

    def _consistent_mu(self, c0):
        """Lumped L2 projection of f'(c0) - kap lap c0 (natural BCs)."""
        cv, cg = self._gp_scalar(c0)
        F_full = np.zeros(self.dm.n_nodes)
        Ml_full = np.zeros(self.dm.n_nodes)
        for pv, b in self.dm.bins.items():
            tb = self.dm.tables_by_p[pv]
            conn = self.mesh.conn_of[pv].astype(np.int64)
            ne, nbf = conn.shape
            nqp = b["nqp"]
            h = self.mesh.tree.h()[self.mesh.bins[pv]]
            jac = (h / 2.0) ** self.dm.dim
            wq = tb.w[None, :] * jac[:, None]                  # [ne,nqp]
            fp = self._fprime_np(cv[pv]).reshape(ne, nqp)
            # Int N_a f'(c0):
            ra = np.einsum("eq,qa->ea", wq * fp, tb.N)
            # + kap Int grad N_a . grad c0 (dscale on the test gradient):
            g = cg[pv].reshape(ne, nqp, self.dm.dim)
            ra += self.kappa * np.einsum(
                "eqd,qad,eq,e->ea", g, tb.dN, wq, 2.0 / h)
            np.add.at(F_full, conn.ravel(), ra.ravel())
            np.add.at(Ml_full, conn.ravel(),
                      np.einsum("eq,qa->ea", wq, tb.N).ravel())
        rhs = np.asarray(self.Tc.T @ F_full)
        ml = np.asarray(self.Tc.T @ Ml_full)
        ml[ml <= 0] = ml[ml > 0].min()
        return rhs / ml

    def _gp_scalar(self, vec):
        full = np.asarray(self.Tc @ vec)
        v, g = {}, {}
        for pv, b in self.dm.bins.items():
            tb = self.dm.tables_by_p[pv]
            conn = self.mesh.conn_of[pv]
            vals = full[conn]
            v[pv] = np.einsum("qa,ea->eq", tb.N, vals).reshape(-1)
            h = self.mesh.tree.h()[self.mesh.bins[pv]]
            g[pv] = np.stack(
                [(np.einsum("qa,ea->eq", tb.dN[:, :, d], vals)
                  * (2.0 / h)[:, None]).reshape(-1)
                 for d in range(self.dm.dim)], axis=1)
        return v, g

    def step(self):
        from scipy.sparse.linalg import splu
        d = self.dm.device
        t_new = self.t + self.dt
        # retrofit G3 (A4b pattern): VARIABLE-COEFFICIENT BDF2 — the
        # coefficients come from the ACTUAL (dt, dt_prev): r = dt/dt_p,
        # c0 = (1+2r)/(1+r), ch = [1+r, -r^2/(1+r)].  r = 1 reproduces
        # the constant-step 1.5/[2, -0.5] bit-exactly (fixed-dt
        # trajectories unchanged); under adaptive_march (dt varies,
        # incl. the accepted half-step history spacing) the scheme
        # stays consistent — the constant-coefficient form measured
        # order 0.90/0.95 on an alternating-dt sequence (retrofit
        # audit doc, G3 baseline).
        if self.order == 1 or self.dt_prev is None \
                or self.t < self.dt / 2:
            c0_, ch = 1.0, [1.0]
        else:
            rr = self.dt / self.dt_prev
            c0_ = (1.0 + 2.0 * rr) / (1.0 + rr)
            ch = [1.0 + rr, -rr * rr / (1.0 + rr)]
        sigma = c0_ / self.dt
        hist_gp = None
        for k, cc in enumerate(ch):
            v, _ = self._gp_scalar(self.hist[k])
            if hist_gp is None:
                hist_gp = {pv: (cc / self.dt) * v[pv] for pv in v}
            else:
                for pv in v:
                    hist_gp[pv] += (cc / self.dt) * v[pv]
        x = self.x.copy()
        for it in range(self.newton_max):
            cv, cg = self._gp_scalar(x[0::2])
            mv, mg = self._gp_scalar(x[1::2])
            rows, cols, vals = [], [], []
            F_full = np.zeros(self.dm.n_nodes * 2)
            for pv, b in self.dm.bins.items():
                conn = self.mesh.conn_of[pv].astype(np.int64)
                ne, nbf = conn.shape
                nqp = b["nqp"]
                arr = lambda a_: wp.array(np.ascontiguousarray(a_),
                                          dtype=wp.float64, device=d)
                Ae = wp.zeros((ne, 2 * nbf, 2 * nbf), dtype=wp.float64,
                              device=d)
                be = wp.zeros((ne, 2 * nbf), dtype=wp.float64, device=d)
                kk = make_ch_newton(nbf, nqp, self.dm.dim, self.energy)
                wp.launch(kk, dim=ne,
                          inputs=[b["conn"], b["h"], b["N"], b["dN"],
                                  b["w"], arr(cv[pv]), arr(cg[pv]),
                                  arr(mv[pv]), arr(mg[pv]),
                                  arr(hist_gp[pv]),
                                  arr(self.fc_fn(self.xq[pv], t_new)),
                                  arr(self.fm_fn(self.xq[pv], t_new)),
                                  wp.float64(self.M),
                                  wp.float64(self.kappa),
                                  wp.float64(sigma),
                                  wp.float64(self.fh_A),
                                  wp.float64(self.fh_B), Ae, be], device=d)
                Aeh, beh = Ae.numpy(), be.numpy()
                gdof = (conn[:, :, None] * 2
                        + np.arange(2)[None, None, :]).reshape(ne,
                                                               2 * nbf)
                rows.append(np.repeat(gdof, 2 * nbf, axis=1).ravel())
                cols.append(np.tile(gdof, (1, 2 * nbf)).ravel())
                vals.append(Aeh.ravel())
                np.add.at(F_full, gdof.ravel(), beh.ravel())
            K = sp.coo_matrix(
                (np.concatenate(vals),
                 (np.concatenate(rows), np.concatenate(cols))),
                shape=(self.dm.n_nodes * 2,) * 2).tocsr()
            A = (self.T2.T @ K @ self.T2).tolil()
            r = np.asarray(self.T2.T @ F_full)
            if self.dirichlet is not None:
                gcv = self.gc_fn(self.free_coords[self.dirichlet], t_new)
                gmv = self.gm_fn(self.free_coords[self.dirichlet], t_new)
                for k2, i in enumerate(self.dirichlet):
                    for c_, gval in ((0, gcv[k2]), (1, gmv[k2])):
                        rr = i * 2 + c_
                        A.rows[rr] = [int(rr)]; A.data[rr] = [1.0]
                        r[rr] = gval - x[rr]
            if self.linsolver == "splu":
                dx = splu(A.tocsr().tocsc()).solve(r)
            else:
                from ..solvers.linsolve import solve_linear
                # sigma changes with BDF startup/adaptive dt: refresh meta.
                # linsolver="blockch_dev" routes the inner Krylov solves
                # through the fused device stack (cg_dev/bicgstab_dev).
                solver = self.linsolver
                meta = {"sigma": sigma, "m": self.M, "kappa": self.kappa}
                if solver == "blockch_dev":
                    solver = "blockch"
                    meta["inners"] = "device"
                self._solver_cache[("blockch_meta", "ch")] = meta
                dx = solve_linear(A.tocsr(), r, solver=solver,
                                  tol=1e-10, cache=self._solver_cache,
                                  cache_key="ch", device=self.dm.device)
            # Newton trust clamp: a c-increment beyond 2 units is always a
            # diverging transient (poly's physical range is [-1,1], FH's
            # (0,1)); scale the WHOLE update to preserve direction.
            # Unclamped, quench-onset Newton at large dt reaches |c| ~ 37
            # (measured), where f'' = 3c^2 - 1 ~ 4100 poisons any
            # iterative Jacobian solver.
            dc_max = np.abs(dx[0::2]).max()
            if dc_max > 2.0:
                dx = dx * (2.0 / dc_max)
            x = x + dx
            if self.energy == "fh":
                # projected Newton: keep iterates physical. Converged FH
                # states sit at the binodal (>= 0.059 measured at B=3), so
                # the projection only clips transient overshoots — without
                # it the step-0 overshoot reaches the f'' regularization
                # cap (measured f'' = 9984 at Newton iterate 2) and any
                # iterative solver of the Jacobian is hostage to it.
                np.clip(x[0::2], 1e-3, 1.0 - 1e-3, out=x[0::2])
            if np.abs(dx).max() < self.newton_tol:
                break
        # Additive instrumentation (no behaviour change): expose the last
        # step's Newton convergence so tutorials/research can separate
        # ALGEBRAIC error (solver) from DISCRETIZATION error, and account
        # for real solver work (Newton iterations, adaptive-step cost).
        self.last_newton = {
            "iters": int(it + 1),
            "dx_inf": float(np.abs(dx).max()),
            "converged": bool(np.abs(dx).max() < self.newton_tol),
            "newton_tol": float(self.newton_tol),
        }
        self.x = x
        self.hist = [x[0::2].copy(), self.hist[0]]
        self.t = t_new
        self.dt_prev = self.dt      # history spacing for BDF2 (G3)
        return x[0::2], x[1::2]


def adaptive_march(stepper, t_end, tol=1e-4, dt_min=1e-5, dt_max=0.5,
                   safety=0.85, dt_cap=None, stride=2, verbose=False):
    """dt_cap (Baskar 2026-07-08): a PHYSICS upper bound on dt that the
    LTE controller cannot see — with evaporation, the constant solvent
    flux out depletes the surface cell at rate ~ k_e/h_surf, capping
    dt <= tol_phi*h_surf/(k_e*dphi) regardless of interior truncation
    error. The controller proposes, the cap disposes. During active
    evaporation the march runs AT the cap; LTE growth cashes in after
    drying (pure coarsening). stride: dof stride of the conserved field
    in stepper.x (2 for binary (c,mu), 4 for ternary)."""
    """M4: LTE-controlled adaptive time stepping (Wodo JCP 2011 class).
    Step-doubling estimator: one dt-step vs two dt/2-steps from the same
    state; LTE ~ |c1 - c2|_inf / (2^p - 1) with p the BDF order; accept
    if LTE < tol, and dt *= safety*(tol/LTE)^(1/(p+1)) (clamped [0.5,2]
    per step). dt GROWS through coarsening — the property that makes
    long phase-field horizons affordable. Returns (t_list, dt_list)."""
    import copy
    p_ord = stepper.order
    ts, dts = [], []
    while stepper.t < t_end - 1e-12:
        # G3: dt_prev joins the rewind state — rejects/replays must
        # restore the BDF2 history spacing (reject consistency)
        state = (stepper.x.copy(), [h.copy() for h in stepper.hist],
                 stepper.t, stepper.dt_prev)
        # one full step
        c1, _ = stepper.step()
        x1 = stepper.x.copy()
        # rewind; two half steps
        stepper.x, stepper.hist, stepper.t = (state[0].copy(),
                                              [h.copy() for h in state[1]],
                                              state[2])
        stepper.dt_prev = state[3]
        dt_full = stepper.dt
        stepper.dt = dt_full / 2
        stepper.step()
        c2, _ = stepper.step()
        x2 = stepper.x.copy()
        # RELATIVE L2 LTE (max-norm measured hostage to the sharpest
        # interface node: dt collapsed to the floor, 16580 steps for
        # t=1.2 — worse than fixed-step; L2 tracks the FIELD's error)
        num = float(np.linalg.norm(x1[0::stride] - x2[0::stride]))
        den = max(float(np.linalg.norm(x2[0::stride])), 1e-30)
        lte = (num / den) / (2 ** p_ord - 1)
        if lte < tol or dt_full <= dt_min * 2:
            # accept the HALF-STEP solution (more accurate); dt update
            dt_new = min(dt_max, max(
                dt_min, dt_full * min(2.0, max(
                    0.5, safety * (tol / max(lte, 1e-30))
                    ** (1.0 / (p_ord + 1))))))
            if dt_cap is not None:
                dt_new = min(dt_new, dt_cap)
            stepper.dt = dt_new
            ts.append(stepper.t)
            dts.append(dt_full)
            if verbose:
                print(f"  t={stepper.t:.3f} dt={dt_full:.4f} "
                      f"lte={lte:.2e}", flush=True)
        else:
            # reject: rewind, halve (dt_prev restored — G3)
            stepper.x, stepper.hist, stepper.t = (state[0].copy(),
                                                  [h.copy()
                                                   for h in state[1]],
                                                  state[2])
            stepper.dt_prev = state[3]
            stepper.dt = max(dt_min, dt_full / 2)
    return ts, dts
