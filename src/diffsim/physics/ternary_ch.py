"""M4 P2.5: TERNARY coupled Cahn-Hilliard (P0 memo Secs 1-3).

Fields (phi_1, mu_1, phi_2, mu_2), solvent eliminated
(phi_s = 1 - phi_1 - phi_2). Flory-Huggins exchange potentials
(N1 = N2 = 1 v1; chain lengths enter trivially later):

  mu_i^bulk = ln(phi_i/phi_s) + chi_12 phi_j + chi_is (phi_s - phi_i)
              - chi_js phi_j            (j = the other solute)
  d(mu_i)/d(phi_i) = 1/phi_i + 1/phi_s - 2 chi_is
  d(mu_i)/d(phi_j) = 1/phi_s + chi_12 - chi_1s - chi_2s   (symmetric)

Dynamics: d(phi_i)/dt = div( sum_j M_ij grad mu_j ), Onsager M SPD
(constant v1; per-GP fields = the closure interface, later).
mu_i = mu_i^bulk - kap_i lap phi_i (kap_12 cross-gradient deferred).
Log clipping at PHI_EPS (standard FH practice). Natural BCs.

WEAK FORM.  This is binary Cahn-Hilliard (cahn_hilliard.py) promoted to two
coupled composition fields. Each solute i carries a (phi_i, mu_i) pair; testing
the mass balance with v and the potential definition with q and integrating by
parts (natural no-flux, surface terms drop) gives, for i = 1, 2:

    R_{phi_i}(v) = Int v d(phi_i)/dt dV
                 + sum_j M_ij Int grad v . grad mu_j dV            = 0
    R_{mu_i}(q)  = Int q mu_i dV - Int q mu_i^bulk dV
                 - kap_i Int grad q . grad phi_i dV                = 0

The only new ingredients beyond binary CH are (i) the Onsager mobility M_ij
coupling the two potentials in the transport term, and (ii) the Flory-Huggins
bulk potential mu_i^bulk replacing the double-well f'(c). Monolithic Newton over
the 4-dof node block (phi_1, mu_1, phi_2, mu_2) uses the d(mu_i)/d(phi) entries
above. The evaporating-film driver (wodo_film.py) adds the moving top surface
and the solvent-flux boundary term on top of this.
"""
import numpy as np
import scipy.sparse as sp
import warp as wp

from ..assembly.operators import _kernel_cache

PHI_EPS = 1e-6


@wp.func
def _rlog(x: wp.float64) -> wp.float64:
    # C1-regularized log: linear extension below EPS keeps a GROWING
    # restoring force (a hard clamp froze it -> fields blew through the
    # simplex: measured phi in [-1.15, 2.03] with mass still exact)
    eps = wp.float64(1e-4)
    if x < eps:
        return wp.log(eps) + (x - eps) / eps
    return wp.log(x)


@wp.func
def _rinv(x: wp.float64) -> wp.float64:
    eps = wp.float64(1e-4)
    if x < eps:
        return wp.float64(1.0) / eps
    return wp.float64(1.0) / x


def make_tch_newton(nbf: int, nqp: int, dim: int):
    key = ("tch_newton", nbf, nqp, dim)
    if key in _kernel_cache:
        return _kernel_cache[key]
    dim_pow = float(dim)

    @wp.kernel(module="unique", enable_backward=False,
               module_options={"max_unroll": 0})
    def tch_k(conn: wp.array2d(dtype=wp.int32),
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
              h1: wp.array(dtype=wp.float64),   # hist phi1 / dt terms
              h2: wp.array(dtype=wp.float64),
              M11: wp.float64, M12: wp.float64, M22: wp.float64,
              c12: wp.float64, c1s: wp.float64, c2s: wp.float64,
              kap1: wp.float64, kap2: wp.float64,
              sigma: wp.float64,
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
            mu1b = _rlog(p1) - _rlog(ps) + c12 * p2 \
                + c1s * (ps - p1) - c2s * p2
            mu2b = _rlog(p2) - _rlog(ps) + c12 * p1 \
                + c2s * (ps - p2) - c1s * p1
            d11 = _rinv(p1) + _rinv(ps) - wp.float64(2.0) * c1s
            d22 = _rinv(p2) + _rinv(ps) - wp.float64(2.0) * c2s
            d12 = _rinv(ps) + c12 - c1s - c2s
            for a in range(nbf):
                Na = Ntab[q, a]
                gM1 = wp.float64(0.0)
                gM2 = wp.float64(0.0)
                gP1 = wp.float64(0.0)
                gP2 = wp.float64(0.0)
                for dd in range(dim):
                    gNa = dNtab[q, a, dd] * dscale
                    gM1 += gNa * (M11 * gm1k[gp, dd] + M12 * gm2k[gp, dd])
                    gM2 += gNa * (M12 * gm1k[gp, dd] + M22 * gm2k[gp, dd])
                    gP1 += gNa * gp1k[gp, dd]
                    gP2 += gNa * gp2k[gp, dd]
                # residuals (be = -r); rows: 4a+0 phi1, +1 mu1, +2 phi2,
                # +3 mu2
                r1 = (Na * (sigma * p1k[gp] - h1[gp]) + gM1) * dJxW
                rm1 = (Na * (m1k[gp] - mu1b)) * dJxW - kap1 * gP1 * dJxW
                r2 = (Na * (sigma * p2k[gp] - h2[gp]) + gM2) * dJxW
                rm2 = (Na * (m2k[gp] - mu2b)) * dJxW - kap2 * gP2 * dJxW
                wp.atomic_add(be, e, 4 * a + 0, -r1)
                wp.atomic_add(be, e, 4 * a + 1, -rm1)
                wp.atomic_add(be, e, 4 * a + 2, -r2)
                wp.atomic_add(be, e, 4 * a + 3, -rm2)
                for b in range(nbf):
                    Nb = Ntab[q, b]
                    lap = wp.float64(0.0)
                    for dd in range(dim):
                        lap += dNtab[q, a, dd] * dNtab[q, b, dd] \
                            * dscale * dscale
                    NN = Na * Nb * dJxW
                    lapw = lap * dJxW
                    # phi1 row: d/dphi1, d/dmu1, d/dmu2
                    wp.atomic_add(Ae, e, 4 * a + 0, 4 * b + 0,
                                  sigma * NN)
                    wp.atomic_add(Ae, e, 4 * a + 0, 4 * b + 1,
                                  M11 * lapw)
                    wp.atomic_add(Ae, e, 4 * a + 0, 4 * b + 3,
                                  M12 * lapw)
                    # mu1 row: d/dphi1, d/dphi2, d/dmu1
                    wp.atomic_add(Ae, e, 4 * a + 1, 4 * b + 0,
                                  -d11 * NN - kap1 * lapw)
                    wp.atomic_add(Ae, e, 4 * a + 1, 4 * b + 2,
                                  -d12 * NN)
                    wp.atomic_add(Ae, e, 4 * a + 1, 4 * b + 1, NN)
                    # phi2 row
                    wp.atomic_add(Ae, e, 4 * a + 2, 4 * b + 2,
                                  sigma * NN)
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

    _kernel_cache[key] = tch_k
    return tch_k


class TernaryCHStepper:
    """4-dof node-major (phi1, mu1, phi2, mu2); BDF1/BDF2 + Newton."""

    def __init__(self, dm, chi=(2.5, 1.0, 0.6), M=(1.0, -0.2, 1.0),
                 kappa=(1e-3, 1e-3), dt=0.01, order=2,
                 newton_tol=1e-9, newton_max=20):
        from ..physics.poisson import gauss_points
        self.dm = dm
        self.c12, self.c1s, self.c2s = chi
        self.M11, self.M12, self.M22 = M
        self.kap1, self.kap2 = kappa
        self.dt, self.order = dt, order
        self.newton_tol, self.newton_max = newton_tol, newton_max
        self.mesh, self.cons = dm.mesh, dm.constraints
        self.Tc = self.cons.T.tocsr()
        self.T4 = sp.kron(self.Tc, sp.identity(4, format="csr"),
                          format="csr").tocsr()
        self.free_coords = self.mesh.node_coords[self.cons.free_nodes]
        self.xq = gauss_points(self.mesh, dm.tables_by_p)
        self.nfree = self.Tc.shape[1]
        self.t = 0.0

    def set_initial(self, p1_fn, p2_fn):
        p1 = p1_fn(self.free_coords)
        p2 = p2_fn(self.free_coords)
        self.hist = [(p1.copy(), p2.copy()), (p1.copy(), p2.copy())]
        self.x = np.zeros(self.nfree * 4)
        self.x[0::4] = p1
        self.x[2::4] = p2
        self.t = 0.0

    def _gp(self, vec):
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
        c0_, ch = ((1.0, [1.0]) if (self.order == 1 or self.t < self.dt/2)
                   else (1.5, [2.0, -0.5]))
        sigma = c0_ / self.dt
        h1_gp = h2_gp = None
        for k, cc in enumerate(ch):
            v1, _ = self._gp(self.hist[k][0])
            v2, _ = self._gp(self.hist[k][1])
            if h1_gp is None:
                h1_gp = {pv: (cc / self.dt) * v1[pv] for pv in v1}
                h2_gp = {pv: (cc / self.dt) * v2[pv] for pv in v2}
            else:
                for pv in v1:
                    h1_gp[pv] += (cc / self.dt) * v1[pv]
                    h2_gp[pv] += (cc / self.dt) * v2[pv]
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
                kk = make_tch_newton(nbf, nqp, self.dm.dim)
                wp.launch(kk, dim=ne, inputs=[
                    b["conn"], b["h"], b["N"], b["dN"], b["w"],
                    arr(fields[0][0][pv]), arr(fields[0][1][pv]),
                    arr(fields[1][0][pv]), arr(fields[1][1][pv]),
                    arr(fields[2][0][pv]), arr(fields[2][1][pv]),
                    arr(fields[3][0][pv]), arr(fields[3][1][pv]),
                    arr(h1_gp[pv]), arr(h2_gp[pv]),
                    wp.float64(self.M11), wp.float64(self.M12),
                    wp.float64(self.M22), wp.float64(self.c12),
                    wp.float64(self.c1s), wp.float64(self.c2s),
                    wp.float64(self.kap1), wp.float64(self.kap2),
                    wp.float64(sigma), Ae, be], device=d)
                Aeh, beh = Ae.numpy(), be.numpy()
                gdof = (conn[:, :, None] * 4
                        + np.arange(4)[None, None, :]).reshape(ne,
                                                               4 * nbf)
                rows.append(np.repeat(gdof, 4 * nbf, axis=1).ravel())
                cols.append(np.tile(gdof, (1, 4 * nbf)).ravel())
                vals.append(Aeh.ravel())
                np.add.at(F_full, gdof.ravel(), beh.ravel())
            K = sp.coo_matrix(
                (np.concatenate(vals),
                 (np.concatenate(rows), np.concatenate(cols))),
                shape=(self.dm.n_nodes * 4,) * 2).tocsr()
            A = (self.T4.T @ K @ self.T4).tocsr()
            r = np.asarray(self.T4.T @ F_full)
            dx = splu(A.tocsc()).solve(r)
            x = x + dx
            if np.abs(dx).max() < self.newton_tol:
                break
        self.x = x
        self.hist = [(x[0::4].copy(), x[2::4].copy()), self.hist[0]]
        self.t += self.dt
        return x[0::4], x[2::4]
