r"""Autograd twin of the discrete Cahn-Hilliard march (reference gradient for
the three-way verification gate).

A faithful torch reimplementation of adjoint/phasefield.CHDiscrete: the SAME
weak form, basis tables and dof layout, marched with torch.linalg.solve inside
a plain (unclamped) Newton loop.  Because the loop is driven to a tight
residual, autograd through the converged iterations returns the implicit-
function-theorem gradient — the independent reference that the hand IFT adjoint
must reproduce.  Interior initial data keeps the FH logs away from the walls
so no non-differentiable clamp is ever needed."""
import numpy as np
import torch

torch.set_default_dtype(torch.float64)


def _fp_poly(c, p):
    return c ** 3 - c


def _fpp_poly(c, p):
    return 3.0 * c * c - 1.0


def _fp_fh(c, p):
    A, B = p["A"], p["B"]
    return A * (torch.log(c) - torch.log(1.0 - c)) + B * (1.0 - 2.0 * c)


class CHTwin:
    def __init__(self, dm, energy="poly", dt=1e-2, order=1, device="cpu"):
        assert order in (1, 2)
        self.energy = energy
        self.dt = float(dt)
        self.order = order
        self.dev = device
        self.dim = dm.dim
        self.nn = dm.n_nodes
        self.ndof = 2 * self.nn
        t = lambda a, dt_=torch.float64: torch.tensor(
            np.asarray(a), dtype=dt_, device=device)
        self.bins = []
        for pv, b in dm.bins.items():
            tb = dm.tables_by_p[pv]
            conn = dm.mesh.conn_of[pv].astype(np.int64)
            h = np.asarray(dm.mesh.tree.h()[dm.mesh.bins[pv]], np.float64)
            ne, nbf = conn.shape
            dscale = 2.0 / h
            jac = (h / 2.0) ** self.dim
            dJxW = np.asarray(tb.w)[None, :] * jac[:, None]
            gdof = (conn[:, :, None] * 2
                    + np.arange(2)[None, None, :]).reshape(ne, 2 * nbf)
            r = np.repeat(gdof, 2 * nbf, axis=1).ravel()
            c = np.tile(gdof, (1, 2 * nbf)).ravel()
            self.bins.append(dict(
                conn=t(conn, torch.int64), N=t(tb.N), dN=t(tb.dN),
                dJxW=t(dJxW), dscale=t(dscale), ne=ne, nbf=nbf,
                nqp=np.asarray(tb.N).shape[0],
                gdof_c=t(gdof[:, 0::2].ravel(), torch.int64),
                gdof_m=t(gdof[:, 1::2].ravel(), torch.int64),
                lin=t(r * self.ndof + c, torch.int64)))

    def _interp(self, field, B):
        vals = field[B["conn"]]                            # [ne, nbf]
        c_ = torch.einsum("qa,ea->eq", B["N"], vals)
        gc = torch.einsum("qad,ea,e->eqd", B["dN"], vals, B["dscale"])
        return c_, gc

    def _assemble(self, c, mu, hist_gp, M, kap, params):
        R = torch.zeros(self.ndof, device=self.dev)
        J = torch.zeros(self.ndof * self.ndof, device=self.dev)
        fp_fn = _fp_fh if self.energy == "fh" else _fp_poly
        fpp_fn = None  # curvature via autograd of fp for FH; explicit poly
        for bi, B in enumerate(self.bins):
            c_, gc = self._interp(c, B)
            mu_, gmu = self._interp(mu, B)
            dJxW, N, dN, dscale = B["dJxW"], B["N"], B["dN"], B["dscale"]
            hg = hist_gp[bi]
            fp = fp_fn(c_, params)
            if self.energy == "fh":
                A, Bc = params["A"], params["B"]
                fpp = A * (1.0 / c_ + 1.0 / (1.0 - c_)) - 2.0 * Bc
            else:
                fpp = 3.0 * c_ * c_ - 1.0
            gN_gmu = torch.einsum("qad,e,eqd->eqa", dN, dscale, gmu)
            gN_gc = torch.einsum("qad,e,eqd->eqa", dN, dscale, gc)
            Rc = torch.einsum("eq,qa->ea", dJxW * (params["sigma"] * c_ - hg),
                              N) + M * torch.einsum("eq,eqa->ea", dJxW,
                                                    gN_gmu)
            Rm = torch.einsum("eq,qa->ea", dJxW * (mu_ - fp), N) \
                - kap * torch.einsum("eq,eqa->ea", dJxW, gN_gc)
            R = R.index_add(0, B["gdof_c"], Rc.reshape(-1))
            R = R.index_add(0, B["gdof_m"], Rm.reshape(-1))
            NN = torch.einsum("eq,qa,qb->eab", dJxW, N, N)
            LL = torch.einsum("eq,qad,qbd->eab", dJxW * dscale[:, None] ** 2,
                              dN, dN)
            FPP = torch.einsum("eq,qa,qb->eab", dJxW * fpp, N, N)
            nbf = B["nbf"]
            Ae = torch.zeros((B["ne"], 2 * nbf, 2 * nbf), device=self.dev)
            Ae[:, 0::2, 0::2] = params["sigma"] * NN
            Ae[:, 0::2, 1::2] = M * LL
            Ae[:, 1::2, 0::2] = -FPP - kap * LL
            Ae[:, 1::2, 1::2] = NN
            J = J.index_add(0, B["lin"], Ae.reshape(-1))
        return R, J.reshape(self.ndof, self.ndof)

    def march(self, c0, hist_list, M, kap, params, n_steps,
              newton_max=30, newton_tol=1e-12):
        """Return list of (c, mu) committed states with autograd graph.
        params carries energy coefficients (A, B) as leaf tensors."""
        c = c0.clone()
        mu = torch.zeros_like(c0)
        hist = [c0.clone(), c0.clone()]
        dt = self.dt
        t = 0.0
        dt_prev = None
        out = []
        for _ in range(n_steps):
            if self.order == 1 or dt_prev is None or t < dt / 2:
                c0_, ch = 1.0, [1.0]
            else:
                rr = dt / dt_prev
                c0_ = (1.0 + 2.0 * rr) / (1.0 + rr)
                ch = [1.0 + rr, -rr * rr / (1.0 + rr)]
            sigma = c0_ / dt
            p = dict(params)
            p["sigma"] = sigma
            hist_gp = None
            for k, cc in enumerate(ch):
                hi = [self._interp(hist[k], B)[0] for B in self.bins]
                if hist_gp is None:
                    hist_gp = [(cc / dt) * hi[bi] for bi in range(len(hi))]
                else:
                    for bi in range(len(hi)):
                        hist_gp[bi] = hist_gp[bi] + (cc / dt) * hi[bi]
            ck, muk = c.clone(), mu.clone()
            for it in range(newton_max):
                R, J = self._assemble(ck, muk, hist_gp, M, kap, p)
                dx = torch.linalg.solve(J, -R)
                ck = ck + dx[0::2]
                muk = muk + dx[1::2]
                if float(dx.detach().abs().max()) < newton_tol:
                    break
            c, mu = ck, muk
            out.append((c, mu))
            hist = [c, hist[0]]
            t += dt
            dt_prev = dt
        return out
