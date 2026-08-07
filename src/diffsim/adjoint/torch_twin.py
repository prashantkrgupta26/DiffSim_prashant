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
        # energy is either a string tag ("fh"/"poly") or an object exposing
        # .fp(c)/.fpp(c) (e.g. neural_energy.NeuralCHEnergy — beyond-FH head).
        obj_energy = not isinstance(self.energy, str)
        fp_fn = None if obj_energy else (_fp_fh if self.energy == "fh"
                                         else _fp_poly)
        for bi, B in enumerate(self.bins):
            c_, gc = self._interp(c, B)
            mu_, gmu = self._interp(mu, B)
            dJxW, N, dN, dscale = B["dJxW"], B["N"], B["dN"], B["dscale"]
            hg = hist_gp[bi]
            if obj_energy:
                fp = self.energy.fp(c_)
                fpp = self.energy.fpp(c_)
            else:
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


# --------------------------------------------------------------------------
# coupled Cahn-Hilliard x Allen-Cahn crystallisation twin (G2, M=K=1)
# --------------------------------------------------------------------------
def _q_t(psi):
    return psi ** 2 * (1.0 - psi) ** 2


def _p_t(psi):
    return 3.0 * psi ** 2 - 2.0 * psi ** 3


class CACHTwin:
    """Autograd twin of adjoint/crystallization.CACHDiscrete. Node block
    (phi, mu, psi); FH bulk energy for phi.  Crystallisation params
    (dsig, dh, Tm, T, eps2, L) + M, kappa, A, B as leaf tensors."""
    NF = 3

    def __init__(self, dm, dt=1e-2, order=1, device="cpu"):
        self.dt, self.order, self.dev = float(dt), order, device
        self.dim, self.nn = dm.dim, dm.n_nodes
        self.ndof = self.NF * self.nn
        t = lambda a, d=torch.float64: torch.tensor(np.asarray(a), dtype=d,
                                                    device=device)
        self.bins = []
        for pv, b in dm.bins.items():
            tb = dm.tables_by_p[pv]
            conn = dm.mesh.conn_of[pv].astype(np.int64)
            h = np.asarray(dm.mesh.tree.h()[dm.mesh.bins[pv]], np.float64)
            ne, nbf = conn.shape
            jac = (h / 2.0) ** self.dim
            dJxW = np.asarray(tb.w)[None, :] * jac[:, None]
            gdof = (conn[:, :, None] * self.NF
                    + np.arange(self.NF)[None, None, :]).reshape(
                        ne, self.NF * nbf)
            r = np.repeat(gdof, self.NF * nbf, axis=1).ravel()
            c = np.tile(gdof, (1, self.NF * nbf)).ravel()
            self.bins.append(dict(
                conn=t(conn, torch.int64), N=t(tb.N), dN=t(tb.dN),
                dJxW=t(dJxW), dscale=t(2.0 / h), ne=ne, nbf=nbf,
                gc=t(gdof[:, 0::self.NF].ravel(), torch.int64),
                gm=t(gdof[:, 1::self.NF].ravel(), torch.int64),
                gs=t(gdof[:, 2::self.NF].ravel(), torch.int64),
                lin=t(r * self.ndof + c, torch.int64)))

    def _ip(self, field, B):
        vals = field[B["conn"]]
        v = torch.einsum("qa,ea->eq", B["N"], vals)
        g = torch.einsum("qad,ea,e->eqd", B["dN"], vals, B["dscale"])
        return v, g

    def _assemble(self, x, hpg, hsg, P):
        R = torch.zeros(self.ndof, device=self.dev)
        J = torch.zeros(self.ndof * self.ndof, device=self.dev)
        drv = P["dh"] * (P["T"] / P["Tm"] - 1.0)
        for bi, B in enumerate(self.bins):
            phi_, gphi = self._ip(x[0::self.NF], B)
            mu_, gmu = self._ip(x[1::self.NF], B)
            psi_, gpsi = self._ip(x[2::self.NF], B)
            dJxW, N, dN, ds = B["dJxW"], B["N"], B["dN"], B["dscale"]
            q = _q_t(psi_)
            p = _p_t(psi_)
            fFHp = P["A"] * (torch.log(phi_) - torch.log(1.0 - phi_)) \
                + P["B"] * (1.0 - 2.0 * phi_)
            dfdphi = fFHp + q * P["dsig"] + p * drv
            # df/dpsi = phi*(q'+..); use autograd-friendly explicit derivs
            qp = 2.0 * psi_ - 6.0 * psi_ ** 2 + 4.0 * psi_ ** 3
            pp = 6.0 * psi_ - 6.0 * psi_ ** 2
            dfdpsi = phi_ * (qp * P["dsig"] + pp * drv)
            fFHpp = P["A"] * (1.0 / phi_ + 1.0 / (1.0 - phi_)) - 2.0 * P["B"]
            qpp = 2.0 - 12.0 * psi_ + 12.0 * psi_ ** 2
            ppp = 6.0 - 12.0 * psi_
            coup = qp * P["dsig"] + pp * drv
            gN_gmu = torch.einsum("qad,e,eqd->eqa", dN, ds, gmu)
            gN_gphi = torch.einsum("qad,e,eqd->eqa", dN, ds, gphi)
            gN_gpsi = torch.einsum("qad,e,eqd->eqa", dN, ds, gpsi)
            sig = P["sigma"]
            Rphi = torch.einsum("eq,qa->ea", dJxW * (sig * phi_ - hpg[bi]),
                                N) + P["M"] * torch.einsum(
                "eq,eqa->ea", dJxW, gN_gmu)
            Rmu = torch.einsum("eq,qa->ea", dJxW * (mu_ - dfdphi), N) \
                - P["kappa"] * torch.einsum("eq,eqa->ea", dJxW, gN_gphi)
            Rpsi = torch.einsum("eq,qa->ea", dJxW * (sig * psi_ - hsg[bi]),
                                N) + P["L"] * (
                torch.einsum("eq,qa->ea", dJxW * dfdpsi, N)
                + P["eps2"] * torch.einsum("eq,eqa->ea", dJxW, gN_gpsi))
            R = R.index_add(0, B["gc"], Rphi.reshape(-1))
            R = R.index_add(0, B["gm"], Rmu.reshape(-1))
            R = R.index_add(0, B["gs"], Rpsi.reshape(-1))
            NN = torch.einsum("eq,qa,qb->eab", dJxW, N, N)
            LL = torch.einsum("eq,qad,qbd->eab", dJxW * ds[:, None] ** 2,
                              dN, dN)
            WM = lambda w: torch.einsum("eq,qa,qb->eab", dJxW * w, N, N)
            nbf = B["nbf"]
            Ae = torch.zeros((B["ne"], self.NF * nbf, self.NF * nbf),
                             device=self.dev)
            NF = self.NF
            Ae[:, 0::NF, 0::NF] = sig * NN
            Ae[:, 0::NF, 1::NF] = P["M"] * LL
            Ae[:, 1::NF, 0::NF] = -WM(fFHpp) - P["kappa"] * LL
            Ae[:, 1::NF, 1::NF] = NN
            Ae[:, 1::NF, 2::NF] = -WM(coup)
            Ae[:, 2::NF, 0::NF] = P["L"] * WM(coup)
            Ae[:, 2::NF, 2::NF] = sig * NN + P["L"] * (
                WM(phi_ * (qpp * P["dsig"] + ppp * drv)) + P["eps2"] * LL)
            J = J.index_add(0, B["lin"], Ae.reshape(-1))
        return R, J.reshape(self.ndof, self.ndof)

    def march(self, phi0, psi0, P, n_steps, newton_max=40, newton_tol=1e-12,
              T_schedule=None, bT=0.0, Tref=0.0):
        NF = self.NF
        x = torch.zeros(self.ndof, device=self.dev)
        x[0::NF] = phi0
        x[2::NF] = psi0
        hist_phi = [phi0.clone(), phi0.clone()]
        hist_psi = [psi0.clone(), psi0.clone()]
        dt = self.dt
        t, dt_prev = 0.0, None
        out = []
        for istep in range(n_steps):
            if self.order == 1 or dt_prev is None or t < dt / 2:
                c0_, ch = 1.0, [1.0]
            else:
                rr = dt / dt_prev
                c0_ = (1.0 + 2.0 * rr) / (1.0 + rr)
                ch = [1.0 + rr, -rr * rr / (1.0 + rr)]
            p = dict(P)
            p["sigma"] = c0_ / dt
            if T_schedule is not None:
                Tn = T_schedule[istep]
                p["T"] = Tn
                p["B"] = P["B"] + bT * (Tn - Tref)

            def hgp(hist):
                hg = None
                for k, cc in enumerate(ch):
                    hi = [self._ip(hist[k], B)[0] for B in self.bins]
                    if hg is None:
                        hg = [(cc / dt) * hi[bi] for bi in range(len(hi))]
                    else:
                        for bi in range(len(hi)):
                            hg[bi] = hg[bi] + (cc / dt) * hi[bi]
                return hg
            hpg, hsg = hgp(hist_phi), hgp(hist_psi)
            xk = x.clone()
            for it in range(newton_max):
                R, Jm = self._assemble(xk, hpg, hsg, p)
                dx = torch.linalg.solve(Jm, -R)
                xk = xk + dx
                if float(dx.detach().abs().max()) < newton_tol:
                    break
            x = xk
            out.append(x)
            hist_phi = [x[0::NF], hist_phi[0]]
            hist_psi = [x[2::NF], hist_psi[0]]
            t += dt
            dt_prev = dt
        return out


# --------------------------------------------------------------------------
# M-component multi-Cahn-Hilliard twin (K=0) — reference for the M-generic
# adjoint (adjoint/multiphase.MultiCHAdjoint) three-way gate
# --------------------------------------------------------------------------
class MultiCHTwin:
    """Autograd twin of adjoint/multiphase.MultiCHDiscrete (K=0).  Node-major
    2M block [(phi_0,mu_0),...]; FH multi-component exchange potentials (p1
    form); M x M Onsager mobility; per-species kappa.  chi/N/onsager/kappa
    enter as leaf tensors so autograd-through-convergence returns the IFT
    gradient the hand adjoint must reproduce."""

    def __init__(self, dm, M, dt=1e-2, order=1, device="cpu"):
        assert order in (1, 2)
        self.M = int(M)
        self.blk = 2 * self.M
        self.dt = float(dt)
        self.order = order
        self.dev = device
        self.dim, self.nn = dm.dim, dm.n_nodes
        self.ndof = self.blk * self.nn
        t = lambda a, d=torch.float64: torch.tensor(np.asarray(a), dtype=d,
                                                    device=device)
        self.bins = []
        for pv, b in dm.bins.items():
            tb = dm.tables_by_p[pv]
            conn = dm.mesh.conn_of[pv].astype(np.int64)
            h = np.asarray(dm.mesh.tree.h()[dm.mesh.bins[pv]], np.float64)
            ne, nbf = conn.shape
            jac = (h / 2.0) ** self.dim
            dJxW = np.asarray(tb.w)[None, :] * jac[:, None]
            gdof = (conn[:, :, None] * self.blk
                    + np.arange(self.blk)[None, None, :]).reshape(
                        ne, self.blk * nbf)
            r = np.repeat(gdof, self.blk * nbf, axis=1).ravel()
            c = np.tile(gdof, (1, self.blk * nbf)).ravel()
            self.bins.append(dict(
                conn=t(conn, torch.int64), N=t(tb.N), dN=t(tb.dN),
                dJxW=t(dJxW), dscale=t(2.0 / h), ne=ne, nbf=nbf,
                gd=[t(gdof[:, f::self.blk].ravel(), torch.int64)
                    for f in range(self.blk)],
                lin=t(r * self.ndof + c, torch.int64)))

    def _ip(self, field, B):
        vals = field[B["conn"]]
        v = torch.einsum("qa,ea->eq", B["N"], vals)
        g = torch.einsum("qad,ea,e->eqd", B["dN"], vals, B["dscale"])
        return v, g

    def _mu_and_H(self, phi_gp, chi_t, Ninv):
        M = self.M
        ps = 1.0 - sum(phi_gp)
        phi_of = lambda l: phi_gp[l] if l < M else ps

        def S(m):
            return sum(phi_of(l) * chi_t[m, l]
                       for l in range(M + 1) if l != m)
        mus = [Ninv[i] * (torch.log(phi_gp[i]) + 1.0)
               - Ninv[M] * (torch.log(ps) + 1.0) + S(i) - S(M)
               for i in range(M)]
        H = [[Ninv[M] / ps + (chi_t[i, j] if i != j else 0.0)
              - chi_t[i, M] - chi_t[M, j] + (Ninv[i] / phi_gp[i]
                                             if i == j else 0.0)
              for j in range(M)] for i in range(M)]
        return mus, H

    def _assemble(self, x, hist_gp, chi_t, Ninv, Lam, kap, sigma):
        M, blk = self.M, self.blk
        R = torch.zeros(self.ndof, device=self.dev)
        J = torch.zeros(self.ndof * self.ndof, device=self.dev)
        for bi, B in enumerate(self.bins):
            dJxW, N, dN, ds = B["dJxW"], B["N"], B["dN"], B["dscale"]
            phi_gp, gphi, mu_gp, gmu = [], [], [], []
            for i in range(M):
                v, g = self._ip(x[2 * i::blk], B)
                phi_gp.append(v); gphi.append(g)
                v2, g2 = self._ip(x[2 * i + 1::blk], B)
                mu_gp.append(v2); gmu.append(g2)
            muref, H = self._mu_and_H(phi_gp, chi_t, Ninv)
            NN = torch.einsum("eq,qa,qb->eab", dJxW, N, N)
            LL = torch.einsum("eq,qad,qbd->eab", dJxW * ds[:, None] ** 2,
                              dN, dN)
            WM = lambda w: torch.einsum("eq,qa,qb->eab", dJxW * w, N, N)
            nbf = B["nbf"]
            Ae = torch.zeros((B["ne"], blk * nbf, blk * nbf), device=self.dev)
            for i in range(M):
                flux = sum(Lam[i, j] * gmu[j] for j in range(M))
                gN_flux = torch.einsum("qad,e,eqd->eqa", dN, ds, flux)
                Rphi = torch.einsum("eq,qa->ea",
                                    dJxW * (sigma * phi_gp[i] - hist_gp[i][bi]),
                                    N) + torch.einsum("eq,eqa->ea", dJxW,
                                                      gN_flux)
                gN_gphi = torch.einsum("qad,e,eqd->eqa", dN, ds, gphi[i])
                Rmu = torch.einsum("eq,qa->ea", dJxW * (mu_gp[i] - muref[i]),
                                   N) - kap[i] * torch.einsum(
                    "eq,eqa->ea", dJxW, gN_gphi)
                R = R.index_add(0, B["gd"][2 * i], Rphi.reshape(-1))
                R = R.index_add(0, B["gd"][2 * i + 1], Rmu.reshape(-1))
                Ae[:, 2 * i::blk, 2 * i::blk] = sigma * NN
                for j in range(M):
                    Ae[:, 2 * i::blk, 2 * j + 1::blk] = Lam[i, j] * LL
                    Ae[:, 2 * i + 1::blk, 2 * j::blk] = -WM(H[i][j])
                Ae[:, 2 * i + 1::blk, 2 * i::blk] = \
                    Ae[:, 2 * i + 1::blk, 2 * i::blk] - kap[i] * LL
                Ae[:, 2 * i + 1::blk, 2 * i + 1::blk] = NN
            J = J.index_add(0, B["lin"], Ae.reshape(-1))
        return R, J.reshape(self.ndof, self.ndof)

    def march(self, phi0_list, chi_t, N_t, Lam, kap, n_steps,
              newton_max=40, newton_tol=1e-12):
        M, blk = self.M, self.blk
        Ninv = 1.0 / N_t
        x = torch.zeros(self.ndof, device=self.dev)
        for i in range(M):
            x[2 * i::blk] = phi0_list[i]
        snap = [phi0_list[i].clone() for i in range(M)]
        hist = [snap, [p.clone() for p in snap]]
        dt = self.dt
        t, dt_prev = 0.0, None
        out = []
        for _ in range(n_steps):
            if self.order == 1 or dt_prev is None or t < dt / 2:
                c0_, ch = 1.0, [1.0]
            else:
                rr = dt / dt_prev
                c0_ = (1.0 + 2.0 * rr) / (1.0 + rr)
                ch = [1.0 + rr, -rr * rr / (1.0 + rr)]
            sigma = c0_ / dt
            hist_gp = []
            for i in range(M):
                acc = None
                for k, cc in enumerate(ch):
                    hi = [self._ip(hist[k][i], B)[0] for B in self.bins]
                    if acc is None:
                        acc = [(cc / dt) * hi[bi] for bi in range(len(hi))]
                    else:
                        for bi in range(len(hi)):
                            acc[bi] = acc[bi] + (cc / dt) * hi[bi]
                hist_gp.append(acc)
            xk = x.clone()
            for it in range(newton_max):
                R, Jm = self._assemble(xk, hist_gp, chi_t, Ninv, Lam, kap,
                                       sigma)
                dx = torch.linalg.solve(Jm, -R)
                xk = xk + dx
                if float(dx.detach().abs().max()) < newton_tol:
                    break
            x = xk
            out.append(x)
            hist = [[x[2 * i::blk] for i in range(M)], hist[0]]
            t += dt
            dt_prev = dt
        return out

    def grads(self, phi0_list, chi, N, onsager, kappa, n_steps, names,
              target):
        """Build leaf tensors for the requested params, march, backprop
        loss = 0.5 sum_i ||phi_i,N - target||^2, return {name: leaf.grad}."""
        M = self.M
        chi0 = np.asarray(chi, np.float64)
        N0 = np.asarray(N, np.float64)
        ons0 = np.asarray(onsager, np.float64)
        kap0 = [float(k) for k in kappa]
        leaves = {}
        for nm in names:
            base = (chi0 if nm.startswith("chi_") else
                    N0 if nm.startswith("N_") else
                    ons0 if nm.startswith("onsager_") else kap0)
            if nm.startswith("chi_") or nm.startswith("onsager_"):
                _, a, b = nm.split("_"); v = float(base[int(a), int(b)])
            else:
                v = float(base[int(nm.split("_")[1])])
            leaves[nm] = torch.tensor(v, dtype=torch.float64,
                                      device=self.dev, requires_grad=True)
        # assemble differentiable parameter tensors from constants + leaves
        chi_t = torch.tensor(chi0, dtype=torch.float64, device=self.dev)
        N_t = torch.tensor(N0, dtype=torch.float64, device=self.dev)
        Lam = torch.tensor(ons0, dtype=torch.float64, device=self.dev)
        kap = [torch.tensor(k, dtype=torch.float64, device=self.dev)
               for k in kap0]
        for nm, leaf in leaves.items():
            if nm.startswith("chi_"):
                _, a, b = nm.split("_"); a, b = int(a), int(b)
                E = torch.zeros(M + 1, M + 1, dtype=torch.float64,
                                device=self.dev)
                E[a, b] = 1.0; E[b, a] = 1.0
                chi_t = chi_t - chi_t * E + leaf * E   # replace entry by leaf
            elif nm.startswith("N_"):
                i = int(nm.split("_")[1])
                e = torch.zeros(M + 1, dtype=torch.float64, device=self.dev)
                e[i] = 1.0
                N_t = N_t - N_t * e + leaf * e
            elif nm.startswith("onsager_"):
                _, a, b = nm.split("_"); a, b = int(a), int(b)
                E = torch.zeros(M, M, dtype=torch.float64, device=self.dev)
                E[a, b] = 1.0
                Lam = Lam - Lam * E + leaf * E
            else:
                kap[int(nm.split("_")[1])] = leaf
        phis = [torch.tensor(np.asarray(p), dtype=torch.float64,
                             device=self.dev) for p in phi0_list]
        out = self.march(phis, chi_t, N_t, Lam, kap, n_steps)
        xN = out[-1]
        blk = self.blk
        tgt = torch.tensor(float(target), dtype=torch.float64,
                           device=self.dev)
        loss = 0.5 * sum(((xN[2 * i::blk] - tgt) ** 2).sum()
                         for i in range(M))
        loss.backward()
        return {nm: float(leaves[nm].grad) for nm in names}
