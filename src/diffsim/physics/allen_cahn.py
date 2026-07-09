r"""M4 track (b): Allen-Cahn brick (p1 + p2) — group guide Sec 1.3.

The NON-conserved sibling of Cahn-Hilliard (see cahn_hilliard.py). Same free
energy F[c] = Int [ f(c) + (kap/2)|grad c|^2 ] dV with double-well
f(c) = (1/4)(c^2-1)^2, f'(c) = c^3 - c — but here the order parameter is NOT
conserved, so the dynamics is plain (not conserved) gradient flow:

        c_t = -M ( f'(c) - kap div(grad c) ) = -M dF/dc         (strong form)

WEAK FORM.  Test with v and integrate the Laplacian by parts (natural no-flux
boundary, so the surface term drops):

    R(v) = Int v c_t dV  +  M kap Int grad v . grad c dV  +  M Int v f'(c) dV = 0
           \___________/     \__________________________/     \______________/
             time term            interface (stiffness)         reaction

Contrast with Cahn-Hilliard: there conservation forces the 4th-order operator
and a second field mu; here f'(c) enters directly as a reaction term and a
single field c suffices. That is the whole conserved-vs-nonconserved distinction
in one line of code.

TIME + NEWTON.  BDF1/BDF2 give c_t -> (sigma c - hist), sigma = b0/dt. Newton
about c_k uses f''(c) = 3 c_k^2 - 1: the reaction term contributes
M Int v f''(c_k) N_b to the Jacobian and M Int v f'(c_k) to the residual. No
SUPG (there is no advection) — Galerkin, and lapN-free (no second-derivative
tables needed). Kernels follow the scalar_transport factory pattern.
"""
import numpy as np
import scipy.sparse as sp
import warp as wp

from ..assembly.operators import _kernel_cache


def make_ac_newton_Ae(nbf: int, nqp: int, dim: int):
    key = ("ac_newton_Ae", nbf, nqp, dim)
    if key in _kernel_cache:
        return _kernel_cache[key]
    dim_pow = float(dim)

    @wp.kernel(module="unique", enable_backward=False,
               module_options=({"max_unroll": 0}
                               if (dim >= 3 or nbf > 4) else {}))
    def ac_Ae(conn: wp.array2d(dtype=wp.int32),
              h: wp.array(dtype=wp.float64),
              Ntab: wp.array2d(dtype=wp.float64),
              dNtab: wp.array3d(dtype=wp.float64),
              wtab: wp.array(dtype=wp.float64),
              ck: wp.array(dtype=wp.float64),        # c_k at GPs
              Mmob: wp.float64, kap: wp.float64,
              sigma: wp.float64,
              Ae: wp.array3d(dtype=wp.float64)):
        e = wp.tid()
        he = h[e]
        jac = wp.pow(he * wp.float64(0.5), wp.float64(dim_pow))
        dscale = wp.float64(2.0) / he
        for q in range(nqp):
            dJxW = wtab[q] * jac
            gp = e * nqp + q
            # dfdd = M f''(c) = M (3 c_k^2 - 1) — the reaction Jacobian factor.
            dfdd = Mmob * (wp.float64(3.0) * ck[gp] * ck[gp]
                           - wp.float64(1.0))
            for a in range(nbf):                       # test function v = N_a
                Na = Ntab[q, a]
                for b in range(nbf):                   # trial (increment) N_b
                    # lap = grad N_a . grad N_b (the interface/stiffness term)
                    lap = wp.float64(0.0)
                    for d in range(dim):
                        lap += dNtab[q, a, d] * dNtab[q, b, d] \
                            * dscale * dscale
                    # dR/dc = sigma N_a N_b  (time)  + M kap grad N_a . grad N_b
                    #         (interface)  + M f''(c) N_a N_b  (reaction)
                    wp.atomic_add(
                        Ae, e, a, b,
                        (sigma * Na * Ntab[q, b] + Mmob * kap * lap
                         + dfdd * Na * Ntab[q, b]) * dJxW)

    _kernel_cache[key] = ac_Ae
    return ac_Ae


def make_ac_residual_be(nbf: int, nqp: int, dim: int):
    key = ("ac_res_be", nbf, nqp, dim)
    if key in _kernel_cache:
        return _kernel_cache[key]
    dim_pow = float(dim)

    @wp.kernel(module="unique", enable_backward=False,
               module_options=({"max_unroll": 0}
                               if (dim >= 3 or nbf > 4) else {}))
    def ac_be(conn: wp.array2d(dtype=wp.int32),
              h: wp.array(dtype=wp.float64),
              Ntab: wp.array2d(dtype=wp.float64),
              dNtab: wp.array3d(dtype=wp.float64),
              wtab: wp.array(dtype=wp.float64),
              ck: wp.array(dtype=wp.float64),        # c_k at GPs
              gck0: wp.array2d(dtype=wp.float64),    # grad c_k at GPs
              hist: wp.array(dtype=wp.float64),      # sum ch_j c_{n-j}/dt
              fq: wp.array(dtype=wp.float64),        # MMS source at GPs
              Mmob: wp.float64, kap: wp.float64,
              sigma: wp.float64,
              be: wp.array2d(dtype=wp.float64)):
        # residual r_a = Int[ Na (sigma ck - hist) + M kap gradNa.grad ck
        #                     + M Na (ck^3 - ck) - Na fq ]
        # Newton solves Ae dc = -r  (be returns -r for A dc = be)
        e = wp.tid()
        he = h[e]
        jac = wp.pow(he * wp.float64(0.5), wp.float64(dim_pow))
        dscale = wp.float64(2.0) / he
        for q in range(nqp):
            dJxW = wtab[q] * jac
            gp = e * nqp + q
            c_ = ck[gp]
            nl = Mmob * (c_ * c_ * c_ - c_)            # M f'(c) = M (c^3 - c)
            for a in range(nbf):                       # test function v = N_a
                Na = Ntab[q, a]
                # gg = grad N_a . grad c  (the interface term of the residual)
                gg = wp.float64(0.0)
                for d in range(dim):
                    gg += dNtab[q, a, d] * dscale * gck0[gp, d]
                # R = Int v c_t + M kap Int grad v . grad c + M Int v f'(c)
                #   c_t -> sigma*c - hist (BDF); fq is the MMS source.
                r_a = (Na * (sigma * c_ - hist[gp]) + Mmob * kap * gg
                       + Na * nl - Na * fq[gp]) * dJxW
                wp.atomic_add(be, e, a, -r_a)

    _kernel_cache[key] = ac_be
    return ac_be


class AllenCahnStepper:
    """BDF1/BDF2 + Newton. No-flux natural BCs (the guide's default);
    optional strong Dirichlet rows for MMS."""

    def __init__(self, dm, M, kappa, dt, order=2, f_fn=None,
                 dirichlet=None, g_fn=None, newton_tol=1e-10,
                 newton_max=12):
        from ..physics.poisson import gauss_points
        self.dm, self.M, self.kappa, self.dt = dm, M, kappa, dt
        self.order = order
        self.f_fn = f_fn or (lambda x, t: np.zeros(len(x)))
        self.dirichlet = dirichlet
        self.g_fn = g_fn
        self.newton_tol, self.newton_max = newton_tol, newton_max
        self.mesh = dm.mesh
        self.cons = dm.constraints
        self.Tc = self.cons.T.tocsr()
        self.free_coords = self.mesh.node_coords[self.cons.free_nodes]
        self.xq = gauss_points(self.mesh, dm.tables_by_p)
        self.t = 0.0
        self.hist = []

    def set_initial(self, c0_fn):
        c0 = c0_fn(self.free_coords)
        self.hist = [c0.copy(), c0.copy()]
        self.t = 0.0
        return c0

    def _gp(self, vec):
        full = np.asarray(self.Tc @ vec)
        out_v, out_g = {}, {}
        for pv, b in self.dm.bins.items():
            tb = self.dm.tables_by_p[pv]
            conn = self.mesh.conn_of[pv]
            vals = full[conn]
            out_v[pv] = np.einsum("qa,ea->eq", tb.N, vals).reshape(-1)
            h = self.mesh.tree.h()[self.mesh.bins[pv]]
            out_g[pv] = np.stack(
                [(np.einsum("qa,ea->eq", tb.dN[:, :, d], vals)
                  * (2.0 / h)[:, None]).reshape(-1)
                 for d in range(self.dm.dim)], axis=1)
        return out_v, out_g

    def step(self):
        from scipy.sparse.linalg import splu
        d = self.dm.device
        t_new = self.t + self.dt
        c0_, ch = ((1.0, [1.0]) if (self.order == 1 or self.t < self.dt/2)
                   else (1.5, [2.0, -0.5]))
        sigma = c0_ / self.dt
        hist_gp = None
        for k, cc in enumerate(ch):
            v, _ = self._gp(self.hist[k])
            if hist_gp is None:
                hist_gp = {pv: (cc / self.dt) * v[pv] for pv in v}
            else:
                for pv in v:
                    hist_gp[pv] += (cc / self.dt) * v[pv]
        c = self.hist[0].copy()
        for it in range(self.newton_max):
            cv, cg = self._gp(c)
            rows, cols, vals = [], [], []
            F_full = np.zeros(self.dm.n_nodes)
            for pv, b in self.dm.bins.items():
                conn = self.mesh.conn_of[pv].astype(np.int64)
                ne, nbf = conn.shape
                nqp = b["nqp"]
                ck = wp.array(np.ascontiguousarray(cv[pv]),
                              dtype=wp.float64, device=d)
                gck = wp.array(np.ascontiguousarray(cg[pv]),
                               dtype=wp.float64, device=d)
                hh = wp.array(np.ascontiguousarray(hist_gp[pv]),
                              dtype=wp.float64, device=d)
                ff = wp.array(np.ascontiguousarray(
                    self.f_fn(self.xq[pv], t_new)), dtype=wp.float64,
                    device=d)
                Ae = wp.zeros((ne, nbf, nbf), dtype=wp.float64, device=d)
                be = wp.zeros((ne, nbf), dtype=wp.float64, device=d)
                kA = make_ac_newton_Ae(nbf, nqp, self.dm.dim)
                kb = make_ac_residual_be(nbf, nqp, self.dm.dim)
                wp.launch(kA, dim=ne,
                          inputs=[b["conn"], b["h"], b["N"], b["dN"],
                                  b["w"], ck, wp.float64(self.M),
                                  wp.float64(self.kappa),
                                  wp.float64(sigma), Ae], device=d)
                wp.launch(kb, dim=ne,
                          inputs=[b["conn"], b["h"], b["N"], b["dN"],
                                  b["w"], ck, gck, hh, ff,
                                  wp.float64(self.M),
                                  wp.float64(self.kappa),
                                  wp.float64(sigma), be], device=d)
                Aeh, beh = Ae.numpy(), be.numpy()
                rows.append(np.repeat(conn, nbf, axis=1).ravel())
                cols.append(np.tile(conn, (1, nbf)).ravel())
                vals.append(Aeh.ravel())
                np.add.at(F_full, conn.ravel(), beh.ravel())
            K = sp.coo_matrix(
                (np.concatenate(vals),
                 (np.concatenate(rows), np.concatenate(cols))),
                shape=(self.dm.n_nodes, self.dm.n_nodes)).tocsr()
            A = (self.Tc.T @ K @ self.Tc).tolil()
            r = np.asarray(self.Tc.T @ F_full)
            if self.dirichlet is not None:
                gv = self.g_fn(self.free_coords[self.dirichlet], t_new)
                for k2, i in enumerate(self.dirichlet):
                    A.rows[i] = [int(i)]; A.data[i] = [1.0]
                    r[i] = gv[k2] - c[i]
            dc = splu(A.tocsr().tocsc()).solve(r)
            c = c + dc
            if np.abs(dc).max() < self.newton_tol:
                break
        self.hist = [c.copy(), self.hist[0]]
        self.t = t_new
        return c
