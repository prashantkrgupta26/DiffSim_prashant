"""M4 track (a): Cahn-Hilliard mixed (c, mu) brick — group guide Sec
2.3, backward-Euler/BDF2, monolithic Newton (Suresh Jacobian pattern).

  R_c  = Int[v c_t] + M Int[grad v . grad mu]              (v test on c)
  R_mu = Int[q mu] - Int[q (c^3 - c)] - kap Int[grad q . grad c]
Node-major 2-dof layout (c, mu). Natural (no-flux) BCs default.
"""
import numpy as np
import scipy.sparse as sp
import warp as wp

from ..assembly.operators import _kernel_cache


def make_ch_newton(nbf: int, nqp: int, dim: int):
    key = ("ch_newton", nbf, nqp, dim)
    if key in _kernel_cache:
        return _kernel_cache[key]
    dim_pow = float(dim)

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
            dfdc = wp.float64(3.0) * c_ * c_ - wp.float64(1.0)
            for a in range(nbf):
                Na = Ntab[q, a]
                # residuals (be = -r)
                gv_gmu = wp.float64(0.0)
                gv_gc = wp.float64(0.0)
                for d in range(dim):
                    gv_gmu += dNtab[q, a, d] * dscale * gmuk[gp, d]
                    gv_gc += dNtab[q, a, d] * dscale * gck[gp, d]
                r_c = (Na * (sigma * c_ - hist[gp]) + Mmob * gv_gmu
                       - Na * fcq[gp]) * dJxW
                r_m = (Na * (muk[gp] - (c_ * c_ * c_ - c_))
                       - Na * fmq[gp]) * dJxW - kap * gv_gc * dJxW
                wp.atomic_add(be, e, 2 * a + 0, -r_c)
                wp.atomic_add(be, e, 2 * a + 1, -r_m)
                for b in range(nbf):
                    Nb = Ntab[q, b]
                    lap = wp.float64(0.0)
                    for d in range(dim):
                        lap += dNtab[q, a, d] * dNtab[q, b, d] \
                            * dscale * dscale
                    # dR_c/dc, dR_c/dmu
                    wp.atomic_add(Ae, e, 2 * a, 2 * b,
                                  sigma * Na * Nb * dJxW)
                    wp.atomic_add(Ae, e, 2 * a, 2 * b + 1,
                                  Mmob * lap * dJxW)
                    # dR_mu/dc, dR_mu/dmu
                    wp.atomic_add(Ae, e, 2 * a + 1, 2 * b,
                                  (-dfdc * Na * Nb - kap * lap) * dJxW)
                    wp.atomic_add(Ae, e, 2 * a + 1, 2 * b + 1,
                                  Na * Nb * dJxW)

    _kernel_cache[key] = ch_k
    return ch_k


class CahnHilliardStepper:
    def __init__(self, dm, M, kappa, dt, order=2, fc_fn=None, fm_fn=None,
                 dirichlet=None, gc_fn=None, gm_fn=None,
                 newton_tol=1e-10, newton_max=15):
        from ..physics.poisson import gauss_points
        self.dm, self.M, self.kappa, self.dt = dm, M, kappa, dt
        self.order = order
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

    def set_initial(self, c0_fn):
        c0 = c0_fn(self.free_coords)
        self.hist = [c0.copy(), c0.copy()]
        self.x = np.zeros(self.nfree * 2)
        self.x[0::2] = c0
        self.t = 0.0
        return c0

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
        c0_, ch = ((1.0, [1.0]) if (self.order == 1 or self.t < self.dt/2)
                   else (1.5, [2.0, -0.5]))
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
                kk = make_ch_newton(nbf, nqp, self.dm.dim)
                wp.launch(kk, dim=ne,
                          inputs=[b["conn"], b["h"], b["N"], b["dN"],
                                  b["w"], arr(cv[pv]), arr(cg[pv]),
                                  arr(mv[pv]), arr(mg[pv]),
                                  arr(hist_gp[pv]),
                                  arr(self.fc_fn(self.xq[pv], t_new)),
                                  arr(self.fm_fn(self.xq[pv], t_new)),
                                  wp.float64(self.M),
                                  wp.float64(self.kappa),
                                  wp.float64(sigma), Ae, be], device=d)
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
            dx = splu(A.tocsr().tocsc()).solve(r)
            x = x + dx
            if np.abs(dx).max() < self.newton_tol:
                break
        self.x = x
        self.hist = [x[0::2].copy(), self.hist[0]]
        self.t = t_new
        return x[0::2], x[1::2]
