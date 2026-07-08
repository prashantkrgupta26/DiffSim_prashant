"""M4 track (c): Wodo & Ganapathysubramanian, Comput. Mater. Sci. 55
(2012) 113-126 — evaporating ternary film (polymer/fullerene/solvent) in
the Landau-mapped frame. Adapts TernaryCHStepper (4-dof monolithic
Newton, C1-regularized FH log).

MODEL (their Eqs. 16-23, nondimensionalized):
  theta = z / h_curr(t): computational domain FIXED (strip in [0,1]^2,
  vertical axis = last axis); physical film height h_curr(t) shrinks.
  Mapped gradient: grad~ = (d/dx, (1/h_curr) d/dtheta) — implemented by
  scaling the vertical component of BOTH the test-function gradients and
  the GP field gradients by minv = 1/h_curr (so the M grad-mu divergence
  and the kappa terms pick up 1/h^2 on the vertical block:
  anisotropic lap = d2/dx2 + (1/h^2) d2/dtheta2).

  d(phi_i)/dt + K (theta/h) d(phi_i)/dtheta = grad~ . (M_ij grad~ mu_j)
  mu_i = dfFH/dphi_i - kap_i lap~ phi_i

SIGN NOTE (load-bearing; the paper's Sec. 5.2 text has a slip): with
h' = dh/dt = -k_e avg(phi_s^top) = -K (K >= 0), the mapping z = theta h
gives phi_t|_z = phi_t|_theta - theta (h'/h) phi_theta, i.e. the LHS
advection coefficient is +K theta/h (features move UP in theta as the
top sweeps down; theta' = -z h'/h^2 = +K theta/h). The top-surface
solute balance (only solvent evaporates, J_i^air = 0) at the interface
receding with h' reads J_i^diff . n = phi_i h' => M grad mu_i . n =
+K phi_i (a natural/Neumann ENRICHMENT flux). With exactly this pair,
d/dt [ h * Int phi_i dtheta ] = 0 (physical solute content conserved,
their footnote 3) — flipping either sign leaks solute at rate 2*K*Phi.

FLORY-HUGGINS with chain lengths (f = sum phi_i/N_i ln phi_i + chi
terms, phi_s = 1 - phi_1 - phi_2 eliminated; exchange potentials):
  mu_i = (1/N_i)(ln phi_i + 1) - (1/N_s)(ln phi_s + 1)
         + chi_12 phi_j + chi_is (phi_s - phi_i) - chi_js phi_j
  d(mu_i)/d(phi_i) = (1/N_i)/phi_i + (1/N_s)/phi_s - 2 chi_is
  d(mu_i)/d(phi_j) = (1/N_s)/phi_s + chi_12 - chi_1s - chi_2s
C1-regularized log kept from the base (linear extension below 1e-4).
Their b*sum(1/phi_i) simplex regularizer (b = 1e-3) is SKIPPED in v1 —
the regularized log already supplies a growing restoring force.

MOBILITY v1: constant SPD M (M12 = 0). Their composition-dependent
M_i = D(phi)/f''_ideal(phi_i) with D = sum D_i phi_i, D_p = D_f =
1e-3 D_s is v2. v1 mapping (documented, used by the Fig-3 benchmark):
freeze M at the initial composition, M0 = D(phi^0)/f''_ideal(phi^0) in
units D_s = 1, L = h0 = 1 => time unit h0^2/D_s and Biot Bi = k_e
exactly (their Eq. 33). At the 1D blend (0.2, 0.2, 0.6) this gives
M0 ~= 0.225 and an effective solvent-gradient relaxation diffusivity
M0*(d11 + d12) ~= 0.93 ~ D_s — self-consistent. v1 has no mobility
freeze-out as phi_s -> 0 (their D drops to 1e-3 D_s; noted, v2).

TIME STEPPING: BDF1 + their Appendix-A heuristic (iters < 20 =>
dt *= 1.25; no convergence in 50 (or divergence) => dt *= 0.25, retry).
Per accepted step: h_curr -= dt * K, K frozen at t_n (also frozen over
the Newton solve). No Langevin noise (CHC term) in v1 — separation is
seeded by initial-condition noise only. No SUPG on the advection term
(cell Peclet ~ K*h_el/D_eff ~ 0.05 at Bi = 10, ny = 128; noted).
"""
import numpy as np
import scipy.sparse as sp
import warp as wp

from ..assembly.operators import _kernel_cache
from ..mesh.nodes import _local_offsets
from .ternary_ch import TernaryCHStepper, _rlog, _rinv


def make_wodo_newton(nbf: int, nqp: int, dim: int):
    """tch_newton + (a) chain-length FH, (b) anisotropic Landau metric
    (vertical gradients scaled by minv = 1/h_curr), (c) mapped-frame
    advection +K theta/h d(phi)/dtheta (rows 4a+0 / 4a+2 and their
    diagonal Jacobian blocks). Vertical axis = dim-1."""
    key = ("wodo_newton", nbf, nqp, dim)
    if key in _kernel_cache:
        return _kernel_cache[key]
    dim_pow = float(dim)
    vax = dim - 1

    @wp.kernel(module="unique", enable_backward=False,
               module_options={"max_unroll": 0})
    def wodo_k(conn: wp.array2d(dtype=wp.int32),
               h: wp.array(dtype=wp.float64),
               Ntab: wp.array2d(dtype=wp.float64),
               dNtab: wp.array3d(dtype=wp.float64),
               wtab: wp.array(dtype=wp.float64),
               p1k: wp.array(dtype=wp.float64),
               gp1k: wp.array2d(dtype=wp.float64),
               m1k: wp.array(dtype=wp.float64),
               gm1k: wp.array2d(dtype=wp.float64),
               p2k: wp.array(dtype=wp.float64),
               gp2k: wp.array2d(dtype=wp.float64),
               m2k: wp.array(dtype=wp.float64),
               gm2k: wp.array2d(dtype=wp.float64),
               h1: wp.array(dtype=wp.float64),
               h2: wp.array(dtype=wp.float64),
               theta: wp.array(dtype=wp.float64),
               M11: wp.float64, M12: wp.float64, M22: wp.float64,
               c12: wp.float64, c1s: wp.float64, c2s: wp.float64,
               n1i: wp.float64, n2i: wp.float64, nsi: wp.float64,
               kap1: wp.float64, kap2: wp.float64,
               sigma: wp.float64,
               minv: wp.float64, kadv: wp.float64,
               Ae: wp.array3d(dtype=wp.float64),
               be: wp.array2d(dtype=wp.float64)):
        e = wp.tid()
        he = h[e]
        jac = wp.pow(he * wp.float64(0.5), wp.float64(dim_pow))
        dscale = wp.float64(2.0) / he
        for q in range(nqp):
            dJxW = wtab[q] * jac
            gp = e * nqp + q
            p1 = p1k[gp]
            p2 = p2k[gp]
            ps = wp.float64(1.0) - p1 - p2
            mu1b = n1i * (_rlog(p1) + wp.float64(1.0)) \
                - nsi * (_rlog(ps) + wp.float64(1.0)) \
                + c12 * p2 + c1s * (ps - p1) - c2s * p2
            mu2b = n2i * (_rlog(p2) + wp.float64(1.0)) \
                - nsi * (_rlog(ps) + wp.float64(1.0)) \
                + c12 * p1 + c2s * (ps - p2) - c1s * p1
            d11 = n1i * _rinv(p1) + nsi * _rinv(ps) - wp.float64(2.0) * c1s
            d22 = n2i * _rinv(p2) + nsi * _rinv(ps) - wp.float64(2.0) * c2s
            d12 = nsi * _rinv(ps) + c12 - c1s - c2s
            # advection coefficient on the RAW theta-derivative:
            # K * theta * (1/h_curr)
            adv = kadv * theta[gp] * minv
            for a in range(nbf):
                Na = Ntab[q, a]
                gM1 = wp.float64(0.0)
                gM2 = wp.float64(0.0)
                gP1 = wp.float64(0.0)
                gP2 = wp.float64(0.0)
                for dd in range(dim):
                    ms = wp.float64(1.0)
                    if dd == vax:
                        ms = minv
                    gNa = dNtab[q, a, dd] * dscale * ms
                    gM1 += gNa * (M11 * gm1k[gp, dd]
                                  + M12 * gm2k[gp, dd]) * ms
                    gM2 += gNa * (M12 * gm1k[gp, dd]
                                  + M22 * gm2k[gp, dd]) * ms
                    gP1 += gNa * gp1k[gp, dd] * ms
                    gP2 += gNa * gp2k[gp, dd] * ms
                r1 = (Na * (sigma * p1k[gp] - h1[gp]
                            + adv * gp1k[gp, vax]) + gM1) * dJxW
                rm1 = (Na * (m1k[gp] - mu1b)) * dJxW - kap1 * gP1 * dJxW
                r2 = (Na * (sigma * p2k[gp] - h2[gp]
                            + adv * gp2k[gp, vax]) + gM2) * dJxW
                rm2 = (Na * (m2k[gp] - mu2b)) * dJxW - kap2 * gP2 * dJxW
                wp.atomic_add(be, e, 4 * a + 0, -r1)
                wp.atomic_add(be, e, 4 * a + 1, -rm1)
                wp.atomic_add(be, e, 4 * a + 2, -r2)
                wp.atomic_add(be, e, 4 * a + 3, -rm2)
                for b in range(nbf):
                    Nb = Ntab[q, b]
                    lap = wp.float64(0.0)
                    for dd in range(dim):
                        ms = wp.float64(1.0)
                        if dd == vax:
                            ms = minv
                        lap += dNtab[q, a, dd] * dNtab[q, b, dd] \
                            * dscale * dscale * ms * ms
                    NN = Na * Nb * dJxW
                    lapw = lap * dJxW
                    advw = Na * adv * dNtab[q, b, vax] * dscale * dJxW
                    # phi1 row: d/dphi1 (dt + advection), d/dmu1, d/dmu2
                    wp.atomic_add(Ae, e, 4 * a + 0, 4 * b + 0,
                                  sigma * NN + advw)
                    wp.atomic_add(Ae, e, 4 * a + 0, 4 * b + 1,
                                  M11 * lapw)
                    wp.atomic_add(Ae, e, 4 * a + 0, 4 * b + 3,
                                  M12 * lapw)
                    # mu1 row
                    wp.atomic_add(Ae, e, 4 * a + 1, 4 * b + 0,
                                  -d11 * NN - kap1 * lapw)
                    wp.atomic_add(Ae, e, 4 * a + 1, 4 * b + 2,
                                  -d12 * NN)
                    wp.atomic_add(Ae, e, 4 * a + 1, 4 * b + 1, NN)
                    # phi2 row
                    wp.atomic_add(Ae, e, 4 * a + 2, 4 * b + 2,
                                  sigma * NN + advw)
                    wp.atomic_add(Ae, e, 4 * a + 2, 4 * b + 3,
                                  M22 * lapw)
                    wp.atomic_add(Ae, e, 4 * a + 2, 4 * b + 1,
                                  M12 * lapw)
                    # mu2 row
                    wp.atomic_add(Ae, e, 4 * a + 3, 4 * b + 2,
                                  -d22 * NN - kap2 * lapw)
                    wp.atomic_add(Ae, e, 4 * a + 3, 4 * b + 0,
                                  -d12 * NN)
                    wp.atomic_add(Ae, e, 4 * a + 3, 4 * b + 3, NN)

    _kernel_cache[key] = wodo_k
    return wodo_k


class WodoFilmStepper(TernaryCHStepper):
    """Landau-mapped evaporating-film stepper (linear elements, BDF1).

    Extra state: h_curr (physical film height, starts at 1), k_e
    (evaporation rate; Bi = k_e in units D_s = L = 1), chain lengths
    N = (N_1, N_2, N_s). K = k_e * avg(phi_s at the top node row) is
    frozen at t_n for each solve; h_curr -= dt * K on acceptance.
    """

    def __init__(self, dm, chi=(1.0, 0.3, 0.3), N=(5.0, 5.0, 1.0),
                 M=(0.225, 0.0, 0.225), kappa=(2e-4, 2e-4), k_e=1.0,
                 dt=1e-4, newton_tol=1e-9, newton_max=50):
        super().__init__(dm, chi=chi, M=M, kappa=kappa, dt=dt, order=1,
                         newton_tol=newton_tol, newton_max=newton_max)
        assert dm.mesh.p == 1, "Wodo film v1: linear elements only"
        self.N1, self.N2, self.Ns = (float(n) for n in N)
        self.k_e = float(k_e)
        self.h_curr = 1.0
        self.n_reject = 0
        # theta at GPs (computational vertical coordinate; static)
        self.theta_wp = {
            pv: wp.array(np.ascontiguousarray(self.xq[pv][:, dm.dim - 1]),
                         dtype=wp.float64, device=dm.device)
            for pv in self.xq}
        self._build_top_faces()

    # -- top-surface topology -------------------------------------------
    def _build_top_faces(self):
        coords = self.mesh.node_coords
        vax = self.dm.dim - 1
        ymax = coords[:, vax].max()
        tol = 1e-12
        self.top_nodes = np.where(coords[:, vax] > ymax - tol)[0]
        assert len(self.top_nodes) > 0
        edges, elens = [], []
        for pv, conn in self.mesh.conn_of.items():
            offs = _local_offsets(pv, self.dm.dim)
            top_loc = np.where(offs[:, vax] == pv)[0]     # p=1: 2 nodes
            nn = conn[:, top_loc]                          # [ne, 2]
            on_top = np.all(coords[nn, vax] > ymax - tol, axis=1)
            h_el = self.mesh.tree.h()[self.mesh.bins[pv]]
            for e in np.where(on_top)[0]:
                edges.append(nn[e])
                elens.append(h_el[e])
        self.top_edge_n = np.asarray(edges, np.int64)      # [nte, 2]
        self.top_edge_len = np.asarray(elens, np.float64)  # [nte]

    def _top_phis_avg(self, p1_free, p2_free):
        f1 = np.asarray(self.Tc @ p1_free)
        f2 = np.asarray(self.Tc @ p2_free)
        return float(np.mean(1.0 - f1[self.top_nodes] - f2[self.top_nodes]))

    # -- one implicit solve at frozen (h_curr, K); does NOT commit ------
    def _attempt(self, dt, K):
        from scipy.sparse.linalg import splu
        d = self.dm.device
        sigma = 1.0 / dt
        v1, _ = self._gp(self.hist[0][0])
        v2, _ = self._gp(self.hist[0][1])
        h1_gp = {pv: sigma * v1[pv] for pv in v1}
        h2_gp = {pv: sigma * v2[pv] for pv in v2}
        minv = 1.0 / self.h_curr
        coef = K * minv          # surface flux (1/h) * K, mapped measure
        n0, n1 = self.top_edge_n[:, 0], self.top_edge_n[:, 1]
        le = self.top_edge_len
        x = self.x.copy()
        for it in range(self.newton_max):
            fields = [self._gp(x[i::4]) for i in range(4)]
            rows, cols, vals = [], [], []
            F_full = np.zeros(self.dm.n_nodes * 4)
            for pv, b in self.dm.bins.items():
                conn = self.mesh.conn_of[pv].astype(np.int64)
                ne, nbf = conn.shape
                nqp = b["nqp"]
                arr = lambda a_: wp.array(np.ascontiguousarray(a_),
                                          dtype=wp.float64, device=d)
                Ae = wp.zeros((ne, 4 * nbf, 4 * nbf), dtype=wp.float64,
                              device=d)
                be = wp.zeros((ne, 4 * nbf), dtype=wp.float64, device=d)
                kk = make_wodo_newton(nbf, nqp, self.dm.dim)
                wp.launch(kk, dim=ne, inputs=[
                    b["conn"], b["h"], b["N"], b["dN"], b["w"],
                    arr(fields[0][0][pv]), arr(fields[0][1][pv]),
                    arr(fields[1][0][pv]), arr(fields[1][1][pv]),
                    arr(fields[2][0][pv]), arr(fields[2][1][pv]),
                    arr(fields[3][0][pv]), arr(fields[3][1][pv]),
                    arr(h1_gp[pv]), arr(h2_gp[pv]), self.theta_wp[pv],
                    wp.float64(self.M11), wp.float64(self.M12),
                    wp.float64(self.M22), wp.float64(self.c12),
                    wp.float64(self.c1s), wp.float64(self.c2s),
                    wp.float64(1.0 / self.N1), wp.float64(1.0 / self.N2),
                    wp.float64(1.0 / self.Ns),
                    wp.float64(self.kap1), wp.float64(self.kap2),
                    wp.float64(sigma), wp.float64(minv), wp.float64(K),
                    Ae, be], device=d)
                Aeh, beh = Ae.numpy(), be.numpy()
                gdof = (conn[:, :, None] * 4
                        + np.arange(4)[None, None, :]).reshape(ne, 4 * nbf)
                rows.append(np.repeat(gdof, 4 * nbf, axis=1).ravel())
                cols.append(np.tile(gdof, (1, 4 * nbf)).ravel())
                vals.append(Aeh.ravel())
                np.add.at(F_full, gdof.ravel(), beh.ravel())
            # -- top-surface enrichment load: R_i -= (K/h) Int w phi_i dS
            # (consistent P1 face mass matrix le/6 [[2,1],[1,2]]).
            # be = -R => F += +(K/h) Mf phi ; Jacobian dR/dphi = -(K/h) Mf.
            f1 = np.asarray(self.Tc @ x[0::4])
            f2 = np.asarray(self.Tc @ x[2::4])
            for comp, fv in ((0, f1), (2, f2)):
                np.add.at(F_full, 4 * n0 + comp,
                          coef * le / 6.0 * (2.0 * fv[n0] + fv[n1]))
                np.add.at(F_full, 4 * n1 + comp,
                          coef * le / 6.0 * (fv[n0] + 2.0 * fv[n1]))
                rows.append(np.concatenate([4 * n0 + comp, 4 * n0 + comp,
                                            4 * n1 + comp, 4 * n1 + comp]))
                cols.append(np.concatenate([4 * n0 + comp, 4 * n1 + comp,
                                            4 * n0 + comp, 4 * n1 + comp]))
                vals.append(np.concatenate([-coef * le / 3.0,
                                            -coef * le / 6.0,
                                            -coef * le / 6.0,
                                            -coef * le / 3.0]))
            Kmat = sp.coo_matrix(
                (np.concatenate(vals),
                 (np.concatenate(rows), np.concatenate(cols))),
                shape=(self.dm.n_nodes * 4,) * 2).tocsr()
            A = (self.T4.T @ Kmat @ self.T4).tocsr()
            r = np.asarray(self.T4.T @ F_full)
            dx = splu(A.tocsc()).solve(r)
            if not np.isfinite(dx).all() or np.abs(dx).max() > 1e6:
                return None, it + 1, False               # diverged
            x = x + dx
            if np.abs(dx).max() < self.newton_tol:
                return x, it + 1, True
        return x, self.newton_max, False                 # no convergence

    # -- march loop with the Appendix-A dt heuristic ---------------------
    def march(self, h_min=0.42, phis_stop=0.05, max_steps=20000,
              dh_cap=0.004, dt_min=1e-12, callback=None):
        """March until avg phi_s <= phis_stop or h_curr <= h_min.
        dh_cap bounds the per-step height decrement (the h-update is
        explicit). Returns a stop-reason string."""
        reason = "max_steps"
        for _ in range(max_steps):
            p1n, p2n = self.hist[0]
            phis_avg = float(np.mean(1.0 - np.asarray(self.Tc @ p1n)
                                     - np.asarray(self.Tc @ p2n)))
            if phis_avg <= phis_stop:
                reason = "phis_stop"
                break
            if self.h_curr <= h_min:
                reason = "h_min"
                break
            K = max(self.k_e * self._top_phis_avg(p1n, p2n), 0.0)
            dt_eff = min(self.dt, dh_cap / max(K, 1e-12))
            x_new, iters, ok = self._attempt(dt_eff, K)
            if not ok:
                self.n_reject += 1
                self.dt = dt_eff * 0.25                  # reject + retry
                if self.dt < dt_min:
                    reason = "dt_underflow"
                    break
                continue
            self.x = x_new
            self.hist = [(x_new[0::4].copy(), x_new[2::4].copy()),
                         self.hist[0]]
            self.t += dt_eff
            self.h_curr -= dt_eff * K
            if iters < 20:
                self.dt = dt_eff * 1.25
            else:
                self.dt = dt_eff
            if callback is not None:
                callback(self, K, dt_eff, iters)
        return reason
