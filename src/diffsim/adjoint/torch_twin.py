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

from .neural_energy import _legendre as _legendre_torch

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


def _qp_t(psi):
    return 2.0 * psi - 6.0 * psi ** 2 + 4.0 * psi ** 3


def _qpp_t(psi):
    return 2.0 - 12.0 * psi + 12.0 * psi ** 2


def _p_t(psi):
    return 3.0 * psi ** 2 - 2.0 * psi ** 3


def _pp_t(psi):
    return 6.0 * psi - 6.0 * psi ** 2


def _ppp_t(psi):
    return 6.0 - 12.0 * psi


def _legendre2_t(u, k):
    """Second derivative d^2 P_k / du^2 of shifted-Legendre on u in [-1, 1],
    in torch (autograd not needed — used analytically in the Jacobian).
    Mirrors neural_crystal._legendre2_np. k in 0..5."""
    if k == 0 or k == 1:
        return torch.zeros_like(u)
    if k == 2:
        return 3.0 * torch.ones_like(u)
    if k == 3:
        return 15.0 * u
    if k == 4:
        return (105.0 * u * u - 15.0) / 2.0
    if k == 5:
        return (315.0 * u ** 3 - 105.0 * u) / 2.0
    raise ValueError(f"basis degree {k} not in 0..5")


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

    def _mu_and_H(self, phi_gp, chi_t, Ninv,
                  basis_leaves=None, basis_meta=None):
        """Compute FH exchange potentials and their Hessian at Gauss points.

        Optional energy correction (Task 3, M6):
          basis_leaves : dict  {(i, k): leaf_tensor}  — only the requested ones
          basis_meta   : dict  {mid, half, degrees}
        Adds sum_k gamma_{i,k} * P_k(u_i) to mus[i]
        and  sum_k gamma_{i,k} * P'_k(u_i)/half to H[i][i].
        """
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
        # --- energy correction ------------------------------------------------
        if basis_leaves and basis_meta:
            mid = basis_meta["mid"]
            half = basis_meta["half"]
            degrees = basis_meta["degrees"]
            for i in range(M):
                u = (phi_gp[i] - mid) / half
                for k in degrees:
                    key = (i, k)
                    if key not in basis_leaves:
                        continue
                    gamma_ik = basis_leaves[key]
                    pk, dpk = _legendre_torch(u, k)
                    mus[i] = mus[i] + gamma_ik * pk
                    H[i][i] = H[i][i] + gamma_ik * dpk / half
        return mus, H

    def _assemble(self, x, hist_gp, chi_t, Ninv, Lam, kap, sigma,
                  basis_leaves=None, basis_meta=None, mob_leaves=None):
        """Assemble residual R and Jacobian J.

        Optional extensions (Task 3, M6):
          basis_leaves / basis_meta — energy correction fed to _mu_and_H.
          mob_leaves  : dict {'mob_m0': tensor, 'mob_c': tensor} or None.
            When present, replaces constant Lam[i,j] with phi_diag closure
            M_ii = mob_m0*(1 + mob_c*phi_i), off-diagonal 0.
            The Jacobian includes the analytic dM_ii/dphi_i block so Newton
            converges robustly; gradient correctness comes from the exact
            residual + autograd-through-convergence.
        """
        M, blk = self.M, self.blk
        R = torch.zeros(self.ndof, device=self.dev)
        J = torch.zeros(self.ndof * self.ndof, device=self.dev)
        use_mob_leaves = mob_leaves is not None
        for bi, B in enumerate(self.bins):
            dJxW, N, dN, ds = B["dJxW"], B["N"], B["dN"], B["dscale"]
            phi_gp, gphi, mu_gp, gmu = [], [], [], []
            for i in range(M):
                v, g = self._ip(x[2 * i::blk], B)
                phi_gp.append(v); gphi.append(g)
                v2, g2 = self._ip(x[2 * i + 1::blk], B)
                mu_gp.append(v2); gmu.append(g2)
            muref, H = self._mu_and_H(phi_gp, chi_t, Ninv,
                                      basis_leaves=basis_leaves,
                                      basis_meta=basis_meta)
            NN = torch.einsum("eq,qa,qb->eab", dJxW, N, N)
            LL = torch.einsum("eq,qad,qbd->eab", dJxW * ds[:, None] ** 2,
                              dN, dN)
            WM = lambda w: torch.einsum("eq,qa,qb->eab", dJxW * w, N, N)
            nbf = B["nbf"]
            Ae = torch.zeros((B["ne"], blk * nbf, blk * nbf), device=self.dev)
            for i in range(M):
                if use_mob_leaves:
                    # phi_diag: M_ii = mob_m0*(1 + mob_c*phi_i), off-diag 0
                    mob_m0 = mob_leaves["mob_m0"]
                    mob_c = mob_leaves["mob_c"]
                    Mii = mob_m0 * (1.0 + mob_c * phi_gp[i])  # [ne, nqp]
                    # Rphi flux: ∫ M_ii ∇mu_i · ∇v dΩ  (only diagonal species)
                    flux_vec = Mii[:, :, None] * gmu[i]   # [ne, nqp, dim]
                    gN_flux = torch.einsum("qad,e,eqd->eqa", dN, ds, flux_vec)
                else:
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
                if use_mob_leaves:
                    # phi_i column → Mii-weighted Laplacian for mu_i dofs
                    Ae[:, 2 * i::blk, 2 * i + 1::blk] = torch.einsum(
                        "eq,qad,qbd->eab",
                        dJxW * Mii * ds[:, None] ** 2, dN, dN)
                    # phi_i column → extra dM_ii/dphi_i * ∇mu_i·∇N_b term
                    # dM_ii/dphi_i = mob_m0 * mob_c (scalar expression)
                    dMii_dphi = mob_m0 * mob_c
                    # ∫ dJxW * dM/dphi_i * (∇mu_i · ∇N_b) * N_a dΩ
                    # gmu_i: [ne, nqp, dim]; dN: [nqp, nbf, dim]; ds: [ne]
                    gmu_dot_dN = torch.einsum(
                        "eqd,qbd,e->eqb", gmu[i], dN, ds)  # [ne, nqp, nbf]
                    Ae[:, 2 * i::blk, 2 * i::blk] = (
                        Ae[:, 2 * i::blk, 2 * i::blk]
                        + dMii_dphi * torch.einsum(
                            "eq,qa,eqb->eab", dJxW, N, gmu_dot_dN))
                else:
                    for j in range(M):
                        Ae[:, 2 * i::blk, 2 * j + 1::blk] = Lam[i, j] * LL
                for j in range(M):
                    Ae[:, 2 * i + 1::blk, 2 * j::blk] = -WM(H[i][j])
                Ae[:, 2 * i + 1::blk, 2 * i::blk] = \
                    Ae[:, 2 * i + 1::blk, 2 * i::blk] - kap[i] * LL
                Ae[:, 2 * i + 1::blk, 2 * i + 1::blk] = NN
            J = J.index_add(0, B["lin"], Ae.reshape(-1))
        return R, J.reshape(self.ndof, self.ndof)

    def march(self, phi0_list, chi_t, N_t, Lam, kap, n_steps,
              newton_max=40, newton_tol=1e-12,
              tau=None, basis_leaves=None, basis_meta=None, mob_leaves=None):
        """March n_steps of BDF1/BDF2 with optional:
          tau           : time-scale leaf; dt_eff = tau * self.dt
          basis_leaves  : {(i,k): tensor} energy correction coefficients
          basis_meta    : {mid, half, degrees} for the shifted-Legendre basis
          mob_leaves    : {'mob_m0': tensor, 'mob_c': tensor} phi_diag closure
        When tau/basis_leaves/mob_leaves are None the march is identical to
        the original (backward-compatible, no-op for existing tests)."""
        if Lam is None and mob_leaves is None:
            raise ValueError("march needs either a constant Lam (mobility matrix) or mob_leaves (closure)")
        M, blk = self.M, self.blk
        Ninv = 1.0 / N_t
        x = torch.zeros(self.ndof, device=self.dev)
        for i in range(M):
            x[2 * i::blk] = phi0_list[i]
        snap = [phi0_list[i].clone() for i in range(M)]
        hist = [snap, [p.clone() for p in snap]]
        dt_base = self.dt
        t, dt_prev = 0.0, None
        out = []
        for _ in range(n_steps):
            # effective time step (tau leaf scales physical dt)
            dt = tau * dt_base if tau is not None else dt_base
            if self.order == 1 or dt_prev is None or t < float(dt_base) / 2:
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
                                       sigma, basis_leaves=basis_leaves,
                                       basis_meta=basis_meta,
                                       mob_leaves=mob_leaves)
                dx = torch.linalg.solve(Jm, -R)
                xk = xk + dx
                if float(dx.detach().abs().max()) < newton_tol:
                    break
            x = xk
            out.append(x)
            hist = [[x[2 * i::blk] for i in range(M)], hist[0]]
            t += float(dt_base)
            dt_prev = dt
        return out

    def grads(self, phi0_list, chi, N, onsager, kappa, n_steps, names,
              target, basis_energy=None, mob_closure=None):
        """Build leaf tensors for the requested params, march, backprop
        loss = 0.5 sum_i ||phi_i,N - target||^2, return {name: leaf.grad}.

        Extended (Task 3, M6) to support three new name families:

        ``basis_{i}_{k}`` (energy correction)
          Requires ``basis_energy`` — a BasisMultiEnergy (or object with .gamma,
          .mid, .half, .degrees).  Initial values taken from energy.gamma[(i,k)]
          (0.0 if absent).

        ``mob_{j}`` (mobility closure)
          Requires ``mob_closure`` — a MobilityClosure('phi_diag', ...) with
          .coeffs dict.  Initial values from closure.coeffs.

        ``tau`` (time-scale)
          No extra object needed; initialised to 1.0 (multiplicative identity).

        Note: when ``basis_energy`` or ``mob_closure`` are provided, their full
        correction is always applied to the march regardless of which basis/mob
        names are requested in ``names``; only the requested names receive gradients.
        """
        M = self.M
        chi0 = np.asarray(chi, np.float64)
        N0 = np.asarray(N, np.float64)
        ons0 = np.asarray(onsager, np.float64)
        kap0 = [float(k) for k in kappa]
        leaves = {}
        phi0_off = {}
        # ---- build leaves for all requested names --------------------------
        for nm in names:
            if nm.startswith("phi0_"):
                i = int(nm.split("_")[1])
                leaves[nm] = torch.tensor(0.0, dtype=torch.float64,
                                          device=self.dev, requires_grad=True)
                phi0_off[i] = leaves[nm]
            elif nm.startswith("basis_"):
                # basis_{i}_{k}: init from energy.gamma[(i,k)] or 0.0
                _, si, sk = nm.split("_")
                i_sp, k_deg = int(si), int(sk)
                v = 0.0
                if basis_energy is not None:
                    v = float(basis_energy.gamma.get((i_sp, k_deg), 0.0))
                leaves[nm] = torch.tensor(v, dtype=torch.float64,
                                          device=self.dev, requires_grad=True)
            elif nm.startswith("mob_"):
                # mob_m0 or mob_c: init from closure.coeffs
                v = 0.0
                if mob_closure is not None:
                    v = float(mob_closure.coeffs.get(nm, 0.0))
                leaves[nm] = torch.tensor(v, dtype=torch.float64,
                                          device=self.dev, requires_grad=True)
            elif nm == "tau":
                leaves[nm] = torch.tensor(1.0, dtype=torch.float64,
                                          device=self.dev, requires_grad=True)
            elif nm.startswith("chi_") or nm.startswith("onsager_"):
                base = chi0 if nm.startswith("chi_") else ons0
                _, a, b = nm.split("_")
                v = float(base[int(a), int(b)])
                leaves[nm] = torch.tensor(v, dtype=torch.float64,
                                          device=self.dev, requires_grad=True)
            elif nm.startswith("N_"):
                v = float(N0[int(nm.split("_")[1])])
                leaves[nm] = torch.tensor(v, dtype=torch.float64,
                                          device=self.dev, requires_grad=True)
            else:
                # kappa_{i}
                v = float(kap0[int(nm.split("_")[1])])
                leaves[nm] = torch.tensor(v, dtype=torch.float64,
                                          device=self.dev, requires_grad=True)
        # ---- assemble differentiable parameter tensors ---------------------
        chi_t = torch.tensor(chi0, dtype=torch.float64, device=self.dev)
        N_t = torch.tensor(N0, dtype=torch.float64, device=self.dev)
        Lam = torch.tensor(ons0, dtype=torch.float64, device=self.dev)
        kap = [torch.tensor(k, dtype=torch.float64, device=self.dev)
               for k in kap0]
        # basis leaves dict keyed by (i, k)
        b_leaves = {}   # {(i, k): leaf_tensor}
        b_meta = None
        mob_lvs = None  # {'mob_m0': tensor, 'mob_c': tensor}
        tau_leaf = None
        for nm, leaf in leaves.items():
            if nm.startswith("phi0_"):
                pass
            elif nm.startswith("basis_"):
                _, si, sk = nm.split("_")
                b_leaves[(int(si), int(sk))] = leaf
            elif nm.startswith("mob_"):
                if mob_lvs is None:
                    mob_lvs = {}
                mob_lvs[nm] = leaf
            elif nm == "tau":
                tau_leaf = leaf
            elif nm.startswith("chi_"):
                _, a, b = nm.split("_"); a, b = int(a), int(b)
                E = torch.zeros(M + 1, M + 1, dtype=torch.float64,
                                device=self.dev)
                E[a, b] = 1.0; E[b, a] = 1.0
                chi_t = chi_t - chi_t * E + leaf * E
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
        # ---- fill in constant basis coefficients not requested as leaves ---
        if basis_energy is not None:
            for (i, k), v in basis_energy.gamma.items():
                if (i, k) not in b_leaves:
                    b_leaves[(i, k)] = torch.tensor(
                        float(v), dtype=torch.float64, device=self.dev)
            b_meta = dict(mid=basis_energy.mid, half=basis_energy.half,
                          degrees=basis_energy.degrees)
        elif b_leaves:
            # no basis_energy supplied — infer meta from names with dom=(0.05,0.95)
            mid, half = 0.5, 0.45
            degrees = tuple(sorted({k for (_, k) in b_leaves}))
            b_meta = dict(mid=mid, half=half, degrees=degrees)
        # ---- if mob_closure provided, fill non-leaf mob coeffs -------------
        if mob_closure is not None:
            if mob_lvs is None:
                mob_lvs = {}
            for pname in ("mob_m0", "mob_c"):
                if pname not in mob_lvs:
                    mob_lvs[pname] = torch.tensor(
                        float(mob_closure.coeffs.get(pname, 0.0)),
                        dtype=torch.float64, device=self.dev)
        # ---- initial conditions -------------------------------------------
        phis = []
        for i in range(M):
            p = torch.tensor(np.asarray(phi0_list[i]), dtype=torch.float64,
                             device=self.dev)
            if i in phi0_off:
                p = p + phi0_off[i]
            phis.append(p)
        # ---- march and loss -----------------------------------------------
        out = self.march(phis, chi_t, N_t, Lam, kap, n_steps,
                         tau=tau_leaf,
                         basis_leaves=b_leaves if b_leaves else None,
                         basis_meta=b_meta,
                         mob_leaves=mob_lvs)
        xN = out[-1]
        blk = self.blk
        tgt = torch.tensor(float(target), dtype=torch.float64,
                           device=self.dev)
        loss = 0.5 * sum(((xN[2 * i::blk] - tgt) ** 2).sum()
                         for i in range(M))
        loss.backward()
        return {nm: float(leaves[nm].grad) for nm in names}


# --------------------------------------------------------------------------
# M-generic coupled multi-CH x multi-Allen-Cahn crystallization twin (K>=0)
# — reference for adjoint/crystallization_multi.CrystalCHAdjoint (three-way gate)
# --------------------------------------------------------------------------
class CrystalCHTwin:
    """Autograd twin of adjoint/crystallization_multi.CrystalCHDiscrete: the
    M-generic Cahn-Hilliard 2M block (a la MultiCHTwin) coupled to a K-species
    Allen-Cahn psi block (a la CACHTwin), for the AdditiveCrystalEnergy bulk

        f = f_FH(phi; chi, N)
          + sum_k phi_k [ q(psi_k) dsig_k + p(psi_k) drive_k ],
        drive_k = dh_k (T / Tm_k - 1),   k in crystallizable.

    Node-major block ``blk = 2*M + K``:
      f = 2*i     -> phi_i   (i = 0 .. M-1)
      f = 2*i + 1 -> mu_i    (i = 0 .. M-1)
      f = 2*M + j -> psi_{crystallizable[j]}   (j = 0 .. K-1)

    chi/N/onsager/kappa and the crystal params (dsig/dh/Tm/eps2/L per species)
    enter as leaf tensors so autograd-through-convergence returns the IFT
    gradient the hand adjoint (CrystalCHAdjoint) must reproduce.  Weak form,
    BDF history and Newton match CrystalCHDiscrete exactly."""

    def __init__(self, dm, M, crystallizable, dt=1e-2, order=1, device="cpu"):
        assert order in (1, 2)
        self.M = int(M)
        self.crystallizable = tuple(int(k) for k in crystallizable)
        self.K = len(self.crystallizable)
        self.blk = 2 * self.M + self.K
        self.dt = float(dt)
        self.order = order
        self.dev = device
        self.dim, self.nn = dm.dim, dm.n_nodes
        self.ndof = self.blk * self.nn
        t = lambda a, d=torch.float64: torch.tensor(np.asarray(a), dtype=d,
                                                    device=device)
        blk = self.blk
        self.bins = []
        for pv, b in dm.bins.items():
            tb = dm.tables_by_p[pv]
            conn = dm.mesh.conn_of[pv].astype(np.int64)
            h = np.asarray(dm.mesh.tree.h()[dm.mesh.bins[pv]], np.float64)
            ne, nbf = conn.shape
            jac = (h / 2.0) ** self.dim
            dJxW = np.asarray(tb.w)[None, :] * jac[:, None]
            gdof = (conn[:, :, None] * blk
                    + np.arange(blk)[None, None, :]).reshape(ne, blk * nbf)
            r = np.repeat(gdof, blk * nbf, axis=1).ravel()
            c = np.tile(gdof, (1, blk * nbf)).ravel()
            self.bins.append(dict(
                conn=t(conn, torch.int64), N=t(tb.N), dN=t(tb.dN),
                dJxW=t(dJxW), dscale=t(2.0 / h), ne=ne, nbf=nbf,
                gd=[t(gdof[:, f::blk].ravel(), torch.int64)
                    for f in range(blk)],
                lin=t(r * self.ndof + c, torch.int64)))

    def _ip(self, field, B):
        vals = field[B["conn"]]
        v = torch.einsum("qa,ea->eq", B["N"], vals)
        g = torch.einsum("qad,ea,e->eqd", B["dN"], vals, B["dscale"])
        return v, g

    def _mu_and_H(self, phi_gp, chi_t, Ninv,
                  basis_leaves=None, basis_meta=None):
        """FH exchange potentials and their phi-Hessian at Gauss points.

        Optional energy correction (BasisMultiEnergy phi-correction):
          basis_leaves : dict  {(i, k): leaf_tensor}  — only requested ones
          basis_meta   : dict  {mid, half, degrees}
        Adds sum_k gamma_{i,k} * P_k(u_i) to mus[i]
        and  sum_k gamma_{i,k} * P'_k(u_i)/half to H[i][i].
        """
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
        # --- energy correction ------------------------------------------------
        if basis_leaves and basis_meta:
            mid = basis_meta["mid"]
            half = basis_meta["half"]
            degrees = basis_meta["degrees"]
            for i in range(M):
                u = (phi_gp[i] - mid) / half
                for k in degrees:
                    key = (i, k)
                    if key not in basis_leaves:
                        continue
                    gamma_ik = basis_leaves[key]
                    pk, dpk = _legendre_torch(u, k)
                    mus[i] = mus[i] + gamma_ik * pk
                    H[i][i] = H[i][i] + gamma_ik * dpk / half
        return mus, H

    def _assemble(self, x, hpg, hsg, chi_t, Ninv, Lam, kap, eps2, L,
                  dsig, drive, sigma, cpl_leaves=None, cpl_meta=None,
                  basis_leaves=None, basis_meta=None):
        """Residual R and Jacobian J.  chi_t/Ninv are (M+1)-shaped tensors,
        Lam is M x M mobility, kap is length-M list, eps2/L/dsig/drive are
        length-K lists (per crystallizable species, in cryst order).

        Optional neural coupling extension:
          cpl_leaves : dict {(k, b): leaf_tensor}  — coupling coefficients c_{k,b}
          cpl_meta   : dict {deg_psi: tuple}        — degrees b included
        When present, ADDS to the coupling residual:
          dfdphi_k += sum_b c_{k,b} L_b(2*psi_k - 1)
          dfdpsi_k += phi_k * 2 * sum_b c_{k,b} L_b'(2*psi_k - 1)
        and the corresponding Jacobian blocks (both additive and neural coexist).

        Optional phi-basis correction:
          basis_leaves : dict {(i, k): leaf_tensor}  — BasisMultiEnergy gamma coeffs
          basis_meta   : dict {mid, half, degrees}
        When present, adds polynomial correction to muref (passed to _mu_and_H).
        """
        M, K, blk = self.M, self.K, self.blk
        R = torch.zeros(self.ndof, device=self.dev)
        J = torch.zeros(self.ndof * self.ndof, device=self.dev)
        use_cpl = cpl_leaves is not None and cpl_meta is not None
        for bi, B in enumerate(self.bins):
            dJxW, N, dN, ds = B["dJxW"], B["N"], B["dN"], B["dscale"]
            phi_gp, gphi, mu_gp, gmu = [], [], [], []
            for i in range(M):
                v, g = self._ip(x[2 * i::blk], B)
                phi_gp.append(v); gphi.append(g)
                v2, g2 = self._ip(x[2 * i + 1::blk], B)
                mu_gp.append(v2); gmu.append(g2)
            psi_gp, gpsi = [], []
            for j in range(K):
                v, g = self._ip(x[2 * M + j::blk], B)
                psi_gp.append(v); gpsi.append(g)
            muref, H = self._mu_and_H(phi_gp, chi_t, Ninv,
                                      basis_leaves=basis_leaves,
                                      basis_meta=basis_meta)
            # crystal coupling: dfdphi_k += q dsig + p drive; assemble the
            # per-crystallizable-species scalar quantities keyed by cryst idx j
            q = [_q_t(psi_gp[j]) for j in range(K)]
            p = [_p_t(psi_gp[j]) for j in range(K)]
            qp = [_qp_t(psi_gp[j]) for j in range(K)]
            pp = [_pp_t(psi_gp[j]) for j in range(K)]
            qpp = [_qpp_t(psi_gp[j]) for j in range(K)]
            ppp = [_ppp_t(psi_gp[j]) for j in range(K)]
            # coup_j = d(dfdphi_k)/dpsi_k = d(dfdpsi_k)/dphi_k  (additive)
            coup = [qp[j] * dsig[j] + pp[j] * drive[j] for j in range(K)]
            dfdphi_add = {}       # species k -> q dsig + p drive
            dfdpsi = {}           # cryst idx j -> phi_k (q' dsig + p' drive)
            for j, k in enumerate(self.crystallizable):
                dfdphi_add[k] = q[j] * dsig[j] + p[j] * drive[j]
                dfdpsi[j] = phi_gp[k] * coup[j]
            # --- neural coupling correction (optional, additive on top) ------
            # h_k(psi) = sum_b c_{k,b} L_b(u), u=2*psi-1
            # h_k'(psi) = 2 * sum_b c_{k,b} L_b'(u)
            # h_k''(psi) = 4 * sum_b c_{k,b} L_b''(u)
            cpl_hk = {}           # j-idx -> h_k at Gauss pts
            cpl_hkp = {}          # j-idx -> h_k' at Gauss pts
            cpl_hkpp = {}         # j-idx -> h_k'' at Gauss pts
            if use_cpl:
                deg_psi = cpl_meta["deg_psi"]
                for j, k in enumerate(self.crystallizable):
                    u = 2.0 * psi_gp[j] - 1.0
                    hk = torch.zeros_like(psi_gp[j])
                    hkp = torch.zeros_like(psi_gp[j])
                    hkpp = torch.zeros_like(psi_gp[j])
                    for b in deg_psi:
                        c_kb = cpl_leaves.get((k, b))
                        if c_kb is None:
                            continue
                        pk, dpk = _legendre_torch(u, b)
                        d2pk = _legendre2_t(u, b)
                        hk = hk + c_kb * pk
                        hkp = hkp + c_kb * dpk * 2.0
                        hkpp = hkpp + c_kb * d2pk * 4.0
                    cpl_hk[j] = hk
                    cpl_hkp[j] = hkp
                    cpl_hkpp[j] = hkpp
                    # add to dfdphi_k: += h_k(psi_k)
                    if k in dfdphi_add:
                        dfdphi_add[k] = dfdphi_add[k] + hk
                    else:
                        dfdphi_add[k] = hk
                    # add to dfdpsi_j: += phi_k * h_k'(psi_k)
                    dfdpsi[j] = dfdpsi[j] + phi_gp[k] * hkp
            NN = torch.einsum("eq,qa,qb->eab", dJxW, N, N)
            LL = torch.einsum("eq,qad,qbd->eab", dJxW * ds[:, None] ** 2,
                              dN, dN)
            WM = lambda w: torch.einsum("eq,qa,qb->eab", dJxW * w, N, N)
            nbf = B["nbf"]
            Ae = torch.zeros((B["ne"], blk * nbf, blk * nbf), device=self.dev)
            # --- phi/mu rows (M-CH block, plus psi coupling on mu rows) ---
            for i in range(M):
                flux = sum(Lam[i, j] * gmu[j] for j in range(M))
                gN_flux = torch.einsum("qad,e,eqd->eqa", dN, ds, flux)
                Rphi = torch.einsum(
                    "eq,qa->ea", dJxW * (sigma * phi_gp[i] - hpg[i][bi]),
                    N) + torch.einsum("eq,eqa->ea", dJxW, gN_flux)
                dfdphi_i = muref[i]
                if i in dfdphi_add:
                    dfdphi_i = dfdphi_i + dfdphi_add[i]
                gN_gphi = torch.einsum("qad,e,eqd->eqa", dN, ds, gphi[i])
                Rmu = torch.einsum("eq,qa->ea", dJxW * (mu_gp[i] - dfdphi_i),
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
                # dR_mu_i / dpsi_j : nonzero only when i is crystallizable k
                # total coupling = additive coup[j] + neural h_k'(psi_k)
                if i in self.crystallizable:
                    j = self.crystallizable.index(i)
                    total_coup_j = coup[j]
                    if use_cpl and j in cpl_hkp:
                        total_coup_j = total_coup_j + cpl_hkp[j]
                    Ae[:, 2 * i + 1::blk, 2 * M + j::blk] = -WM(total_coup_j)
            # --- psi rows (multi-Allen-Cahn block) ---
            for j, k in enumerate(self.crystallizable):
                prow = 2 * M + j
                gN_gpsi = torch.einsum("qad,e,eqd->eqa", dN, ds, gpsi[j])
                Rpsi = torch.einsum(
                    "eq,qa->ea", dJxW * (sigma * psi_gp[j] - hsg[j][bi]),
                    N) + L[j] * (
                    torch.einsum("eq,qa->ea", dJxW * dfdpsi[j], N)
                    + eps2[j] * torch.einsum("eq,eqa->ea", dJxW, gN_gpsi))
                R = R.index_add(0, B["gd"][prow], Rpsi.reshape(-1))
                # dR_psi_k / dphi_k = L_k WM(total_coup_j)  (h_k'(psi_k))
                total_coup_j = coup[j]
                if use_cpl and j in cpl_hkp:
                    total_coup_j = total_coup_j + cpl_hkp[j]
                Ae[:, prow::blk, 2 * k::blk] = L[j] * WM(total_coup_j)
                # dR_psi_k / dpsi_k = sigma NN + L_k (WM(phi_k*(q'' dsig +
                #   p''' drive + h_k''(psi_k))) + eps2_k LL)
                psi_d2 = phi_gp[k] * (qpp[j] * dsig[j] + ppp[j] * drive[j])
                if use_cpl and j in cpl_hkpp:
                    psi_d2 = psi_d2 + phi_gp[k] * cpl_hkpp[j]
                Ae[:, prow::blk, prow::blk] = sigma * NN + L[j] * (
                    WM(psi_d2) + eps2[j] * LL)
            J = J.index_add(0, B["lin"], Ae.reshape(-1))
        return R, J.reshape(self.ndof, self.ndof)

    def march(self, phi0_list, psi0_list, chi_t, N_t, Lam, kap, eps2, L,
              dsig, drive, n_steps, newton_max=40, newton_tol=1e-12,
              cpl_leaves=None, cpl_meta=None,
              basis_leaves=None, basis_meta=None):
        """March n_steps of BDF1/BDF2.  Optional neural coupling:
          cpl_leaves : dict {(k, b): tensor}   coupling coefficients c_{k,b}
          cpl_meta   : dict {deg_psi: tuple}   degrees b
        When provided, the full neural coupling is applied every step (not
        just for requested-gradient names); the additive coupling path (dsig/
        drive) coexists and is always applied from the supplied values.

        Optional phi-basis correction:
          basis_leaves : dict {(i, k): tensor}  BasisMultiEnergy gamma coeffs
          basis_meta   : dict {mid, half, degrees}
        """
        M, K, blk = self.M, self.K, self.blk
        Ninv = 1.0 / N_t
        x = torch.zeros(self.ndof, device=self.dev)
        for i in range(M):
            x[2 * i::blk] = phi0_list[i]
        for j in range(K):
            x[2 * M + j::blk] = psi0_list[j]
        phi_snap = [phi0_list[i].clone() for i in range(M)]
        psi_snap = [psi0_list[j].clone() for j in range(K)]
        hist_phi = [phi_snap, [p.clone() for p in phi_snap]]
        hist_psi = [psi_snap, [p.clone() for p in psi_snap]]
        dt = self.dt
        t, dt_prev = 0.0, None
        out = []

        def hgp(hist, nf):
            acc_all = []
            for i in range(nf):
                acc = None
                for kk, cc in enumerate(ch):
                    hi = [self._ip(hist[kk][i], B)[0] for B in self.bins]
                    if acc is None:
                        acc = [(cc / dt) * hi[bi] for bi in range(len(hi))]
                    else:
                        for bi in range(len(hi)):
                            acc[bi] = acc[bi] + (cc / dt) * hi[bi]
                acc_all.append(acc)
            return acc_all

        for _ in range(n_steps):
            if self.order == 1 or dt_prev is None or t < dt / 2:
                c0_, ch = 1.0, [1.0]
            else:
                rr = dt / dt_prev
                c0_ = (1.0 + 2.0 * rr) / (1.0 + rr)
                ch = [1.0 + rr, -rr * rr / (1.0 + rr)]
            sigma = c0_ / dt
            hpg = hgp(hist_phi, M)
            hsg = hgp(hist_psi, K)
            xk = x.clone()
            for it in range(newton_max):
                R, Jm = self._assemble(xk, hpg, hsg, chi_t, Ninv, Lam, kap,
                                       eps2, L, dsig, drive, sigma,
                                       cpl_leaves=cpl_leaves,
                                       cpl_meta=cpl_meta,
                                       basis_leaves=basis_leaves,
                                       basis_meta=basis_meta)
                dx = torch.linalg.solve(Jm, -R)
                xk = xk + dx
                if float(dx.detach().abs().max()) < newton_tol:
                    break
            x = xk
            out.append(x)
            hist_phi = [[x[2 * i::blk] for i in range(M)], hist_phi[0]]
            hist_psi = [[x[2 * M + j::blk] for j in range(K)], hist_psi[0]]
            t += dt
            dt_prev = dt
        return out

    def grads(self, phi0_list, psi0_list, energy_params, engine_params,
              n_steps, names, target, neural_energy=None):
        """Build leaf tensors for requested params, march, backprop
        loss = 0.5 sum_i ||phi_i,N - tgt||^2 + 0.5 sum_k ||psi_k,N - tgt||^2,
        return {name: leaf.grad}.

        energy_params: chi (MxM+1 sym), N (len M+1), dsig/dh/Tm (dict k->float),
        T (float).  engine_params: onsager (MxM), kappa (len M), eps2/L
        (dict k->float).  Recognised names:
          dsig_k, dh_k, Tm_k, eps2_k, L_k  (per crystallizable species k)
          chi_a_b (symmetric), N_i, onsager_a_b, kappa_i
          cpl_{k}_{b}  (neural coupling coefficients; requires neural_energy)
          basis_{i}_{k} (BasisMultiEnergy phi-correction; requires neural_energy)

        neural_energy : NeuralCrystalEnergy or None.
          When provided, its FULL coupling (all c[(k,b)]) is applied to every
          march step regardless of which cpl_* names are requested; only the
          requested names receive gradients (the non-requested c[(k,b)] are
          treated as fixed constants).
          Similarly, neural_energy.base.gamma is FULLY applied to _mu_and_H;
          only requested basis_{i}_{k} names get gradients.
        """
        M, K = self.M, self.K
        cryst = self.crystallizable
        chi0 = np.asarray(energy_params["chi"], np.float64)
        N0 = np.asarray(energy_params["N"], np.float64)
        ons0 = np.asarray(engine_params["onsager"], np.float64)
        kap0 = [float(k) for k in engine_params["kappa"]]
        dsig0 = {int(k): float(v) for k, v in energy_params["dsig"].items()}
        dh0 = {int(k): float(v) for k, v in energy_params["dh"].items()}
        Tm0 = {int(k): float(v) for k, v in energy_params["Tm"].items()}
        eps20 = {int(k): float(v) for k, v in engine_params["eps2"].items()}
        L0 = {int(k): float(v) for k, v in engine_params["L"].items()}
        T = float(energy_params["T"])

        leaves = {}
        for nm in names:
            if nm.startswith("cpl_"):
                # cpl_{k}_{b}: init from neural_energy.c[(k,b)] or 0.0
                _, sk, sb = nm.split("_")
                k_sp, b_deg = int(sk), int(sb)
                v = 0.0
                if neural_energy is not None:
                    v = float(neural_energy.c.get((k_sp, b_deg), 0.0))
            elif nm.startswith("basis_"):
                # basis_{i}_{k}: init from neural_energy.base.gamma[(i,k)] or 0.0
                _, si, sk = nm.split("_")
                i_sp, k_deg = int(si), int(sk)
                v = 0.0
                if neural_energy is not None:
                    v = float(neural_energy.base.gamma.get((i_sp, k_deg), 0.0))
            elif nm.startswith("chi_") or nm.startswith("onsager_"):
                base = chi0 if nm.startswith("chi_") else ons0
                _, a, b = nm.split("_")
                v = float(base[int(a), int(b)])
            elif nm.startswith("N_"):
                v = float(N0[int(nm.split("_")[1])])
            elif nm.startswith("kappa_"):
                v = float(kap0[int(nm.split("_")[1])])
            elif nm.startswith("dsig_"):
                v = dsig0[int(nm.split("_")[1])]
            elif nm.startswith("dh_"):
                v = dh0[int(nm.split("_")[1])]
            elif nm.startswith("Tm_"):
                v = Tm0[int(nm.split("_")[1])]
            elif nm.startswith("eps2_"):
                v = eps20[int(nm.split("_")[1])]
            elif nm.startswith("L_"):
                v = L0[int(nm.split("_")[1])]
            else:
                raise ValueError(f"unknown param name {nm!r}")
            leaves[nm] = torch.tensor(v, dtype=torch.float64,
                                      device=self.dev, requires_grad=True)

        tt = lambda a: torch.tensor(np.asarray(a), dtype=torch.float64,
                                    device=self.dev)
        # ---- differentiable param tensors (splice leaves in) ----
        chi_t = tt(chi0)
        N_t = tt(N0)
        Lam = tt(ons0)
        kap = [tt(k) for k in kap0]
        dsig_d = {k: tt(v) for k, v in dsig0.items()}
        dh_d = {k: tt(v) for k, v in dh0.items()}
        Tm_d = {k: tt(v) for k, v in Tm0.items()}
        eps2_d = {k: tt(v) for k, v in eps20.items()}
        L_d = {k: tt(v) for k, v in L0.items()}
        # ---- coupling leaf dict {(k,b): tensor} for march ------------------
        c_leaves = {}   # {(k, b): leaf or constant tensor}
        c_meta = None
        b_leaves = {}   # {(i, k): leaf or constant tensor} for basis correction
        b_meta = None
        for nm, leaf in leaves.items():
            if nm.startswith("cpl_"):
                _, sk, sb = nm.split("_")
                c_leaves[(int(sk), int(sb))] = leaf
            elif nm.startswith("basis_"):
                _, si, sk = nm.split("_")
                b_leaves[(int(si), int(sk))] = leaf
            elif nm.startswith("chi_"):
                _, a, b = nm.split("_"); a, b = int(a), int(b)
                E = torch.zeros(M + 1, M + 1, dtype=torch.float64,
                                device=self.dev)
                E[a, b] = 1.0; E[b, a] = 1.0
                chi_t = chi_t - chi_t * E + leaf * E
            elif nm.startswith("onsager_"):
                _, a, b = nm.split("_"); a, b = int(a), int(b)
                E = torch.zeros(M, M, dtype=torch.float64, device=self.dev)
                E[a, b] = 1.0
                Lam = Lam - Lam * E + leaf * E
            elif nm.startswith("N_"):
                i = int(nm.split("_")[1])
                e = torch.zeros(M + 1, dtype=torch.float64, device=self.dev)
                e[i] = 1.0
                N_t = N_t - N_t * e + leaf * e
            elif nm.startswith("kappa_"):
                kap[int(nm.split("_")[1])] = leaf
            elif nm.startswith("dsig_"):
                dsig_d[int(nm.split("_")[1])] = leaf
            elif nm.startswith("dh_"):
                dh_d[int(nm.split("_")[1])] = leaf
            elif nm.startswith("Tm_"):
                Tm_d[int(nm.split("_")[1])] = leaf
            elif nm.startswith("eps2_"):
                eps2_d[int(nm.split("_")[1])] = leaf
            elif nm.startswith("L_"):
                L_d[int(nm.split("_")[1])] = leaf
        # ---- fill in constant coupling coefficients from neural_energy ------
        # (always apply full correction when neural_energy is provided; only
        # the requested cpl_* names get gradients — non-requested are constants)
        if neural_energy is not None:
            for (k, b), v in neural_energy.c.items():
                if (k, b) not in c_leaves:
                    c_leaves[(k, b)] = tt(float(v))
            c_meta = dict(deg_psi=neural_energy.deg_psi)
        elif c_leaves:
            # neural_energy not supplied but cpl_* names requested — infer meta
            deg_psi = tuple(sorted({b for (_, b) in c_leaves}))
            c_meta = dict(deg_psi=deg_psi)
        # ---- fill in constant basis coefficients from neural_energy.base ----
        # (always apply full phi-correction; only requested basis_* get grads)
        if neural_energy is not None:
            base = neural_energy.base
            b_meta = dict(mid=tt(base.mid), half=tt(base.half),
                          degrees=base.degrees)
            for (i, k), v in base.gamma.items():
                if (i, k) not in b_leaves:
                    b_leaves[(i, k)] = tt(float(v))
        # ---- per-crystallizable-species lists (in cryst order) ----
        Tt = tt(T)
        dsig = [dsig_d[k] for k in cryst]
        drive = [dh_d[k] * (Tt / Tm_d[k] - 1.0) for k in cryst]
        eps2 = [eps2_d[k] for k in cryst]
        L = [L_d[k] for k in cryst]
        # ---- initial conditions ----
        phis = [tt(phi0_list[i]) for i in range(M)]
        psis = [tt(psi0_list[j]) for j in range(K)]
        # ---- march + loss ----
        out = self.march(phis, psis, chi_t, N_t, Lam, kap, eps2, L,
                         dsig, drive, n_steps,
                         cpl_leaves=c_leaves if c_leaves else None,
                         cpl_meta=c_meta,
                         basis_leaves=b_leaves if b_leaves else None,
                         basis_meta=b_meta)
        xN = out[-1]
        blk = self.blk
        tgt = torch.tensor(float(target), dtype=torch.float64,
                           device=self.dev)
        loss = 0.5 * sum(((xN[2 * i::blk] - tgt) ** 2).sum() for i in range(M))
        loss = loss + 0.5 * sum(
            ((xN[2 * M + j::blk] - tgt) ** 2).sum() for j in range(K))
        loss.backward()
        return {nm: float(leaves[nm].grad) for nm in names}

    def loss_only(self, phi0_list, psi0_list, energy_params, engine_params,
                  n_steps, target, neural_energy=None):
        """Detached forward pass returning scalar phi+psi loss (no autograd).

        Uses torch.no_grad() so all operations are free of gradient tracking.
        Intended for FD gradient verification: call at slightly perturbed
        coefficients without building any leaf tensors.

        energy_params / engine_params : same dicts as grads().
        neural_energy : NeuralCrystalEnergy (or None).  When provided, its
          full coupling and full phi-basis correction are applied to the march.
        """
        M, K = self.M, self.K
        cryst = self.crystallizable
        chi0 = np.asarray(energy_params["chi"], np.float64)
        N0 = np.asarray(energy_params["N"], np.float64)
        ons0 = np.asarray(engine_params["onsager"], np.float64)
        kap0 = [float(k) for k in engine_params["kappa"]]
        dsig0 = {int(k): float(v) for k, v in energy_params["dsig"].items()}
        dh0 = {int(k): float(v) for k, v in energy_params["dh"].items()}
        Tm0 = {int(k): float(v) for k, v in energy_params["Tm"].items()}
        eps20 = {int(k): float(v) for k, v in engine_params["eps2"].items()}
        L0 = {int(k): float(v) for k, v in engine_params["L"].items()}
        T = float(energy_params["T"])
        tt = lambda a: torch.tensor(np.asarray(a), dtype=torch.float64,
                                    device=self.dev)
        with torch.no_grad():
            chi_t = tt(chi0)
            N_t = tt(N0)
            Lam = tt(ons0)
            kap = [tt(k) for k in kap0]
            Tt = tt(T)
            dsig = [tt(dsig0[k]) for k in cryst]
            drive = [tt(dh0[k]) * (Tt / tt(Tm0[k]) - 1.0) for k in cryst]
            eps2 = [tt(eps20[k]) for k in cryst]
            L = [tt(L0[k]) for k in cryst]
            phis = [tt(phi0_list[i]) for i in range(M)]
            psis = [tt(psi0_list[j]) for j in range(K)]
            c_leaves = None
            c_meta = None
            b_leaves = None
            b_meta = None
            if neural_energy is not None:
                c_leaves = {(k, b): tt(float(v))
                            for (k, b), v in neural_energy.c.items()}
                c_meta = dict(deg_psi=neural_energy.deg_psi)
                base = neural_energy.base
                b_leaves = {(i, k): tt(float(v))
                            for (i, k), v in base.gamma.items()}
                b_meta = dict(mid=tt(base.mid), half=tt(base.half),
                              degrees=base.degrees)
            out = self.march(phis, psis, chi_t, N_t, Lam, kap, eps2, L,
                             dsig, drive, n_steps,
                             cpl_leaves=c_leaves, cpl_meta=c_meta,
                             basis_leaves=b_leaves, basis_meta=b_meta)
            xN = out[-1]
            blk = self.blk
            tgt = torch.tensor(float(target), dtype=torch.float64,
                               device=self.dev)
            loss = 0.5 * sum(
                ((xN[2 * i::blk] - tgt) ** 2).sum() for i in range(M))
            loss = loss + 0.5 * sum(
                ((xN[2 * M + j::blk] - tgt) ** 2).sum() for j in range(K))
            return float(loss)


# --------------------------------------------------------------------------
# Coupled Cahn-Hilliard / Navier-Stokes (CHNS) twin (SP-0 Task 11)
# — autograd reference for adjoint/chns.CHNSAdjoint (three-way gate).
# --------------------------------------------------------------------------
class CHNSTwin:
    r"""Autograd twin of adjoint/chns.CHNSDiscrete (interface='ch', BDF1).

    Faithful torch reimplementation of the coupled (u, p, phi, mu) monolithic
    residual: variable rho(phi)/eta(phi) via mix_props, capillary mu*grad phi,
    AGG mass flux, gravity, symmetric-D viscous, SUPG/PSPG on the FULL momentum
    strong residual with a DIFFERENTIATED tau — every term mirrors the numpy
    mirror's residual rows.  Marched with torch.linalg.solve inside a plain
    Newton loop; autograd through the converged march returns the IFT gradient
    the hand adjoint (CHNSAdjoint) must reproduce.

    DOF layout: node-major blk = dim + 3, (u_0..u_{dim-1}, p, phi, mu).
    Boundary: no-slip u (strong), pressure pin at node 0 (strong).

    Differentiable scalar params (leaf tensors):
      rho_ratio, eta_ratio, We, mobility (== 1/Pe, the Pe*M knob), Fr.
    CPU float64 throughout.
    """

    def __init__(self, dm, case, dt, Cn, gravity=True, device="cpu"):
        self.dim = dm.dim
        self.nn = dm.n_nodes
        self.blk = self.dim + 3
        self.ndof = self.blk * self.nn
        self.dt = float(dt)
        self.Cn = float(Cn)
        self.Re = float(case.Re)
        self.gravity = bool(gravity)
        self.dev = device
        t = lambda a, d=torch.float64: torch.tensor(np.asarray(a), dtype=d,
                                                    device=device)
        self.bins = []
        blk = self.blk
        for pv, b in dm.bins.items():
            tb = dm.tables_by_p[pv]
            conn = dm.mesh.conn_of[pv].astype(np.int64)
            h = np.asarray(dm.mesh.tree.h()[dm.mesh.bins[pv]], np.float64)
            ne, nbf = conn.shape
            jac = (h / 2.0) ** self.dim
            dJxW = np.asarray(tb.w)[None, :] * jac[:, None]
            gdof = (conn[:, :, None] * blk
                    + np.arange(blk)[None, None, :]).reshape(ne, blk * nbf)
            self.bins.append(dict(
                conn=t(conn, torch.int64), N=t(tb.N), dN=t(tb.dN),
                dJxW=t(dJxW), dscale=t(2.0 / h), h=float(h[0]),
                ne=ne, nbf=nbf, nqp=np.asarray(tb.N).shape[0],
                gd=[t(gdof[:, f::blk].ravel(), torch.int64)
                    for f in range(blk)]))
        # boundary rows (no-slip velocity; pressure pin node 0)
        bnd = np.asarray(dm.mesh.boundary_nodes, bool)
        bc = []
        for a in np.where(bnd)[0]:
            for d in range(self.dim):
                bc.append(a * blk + d)
        bc.append(0 * blk + self.dim)
        self.bc_rows = t(np.asarray(bc, np.int64), torch.int64)
        self.coords = np.asarray(dm.mesh.node_coords, np.float64)
        self._dm = dm
        # attach a numpy CHNSDiscrete for the exact (detached) Newton Jacobian
        self._attach_mirror(case, gravity)

    def _ipv(self, field, B):
        # field [nn, dim] -> v[e,q,d], g[e,q,d,s]
        vals = field[B["conn"]]
        v = torch.einsum("qa,ead->eqd", B["N"], vals)
        g = torch.einsum("qas,ead,e->eqds", B["dN"], vals, B["dscale"])
        return v, g

    def _ip(self, field, B):
        vals = field[B["conn"]]
        v = torch.einsum("qa,ea->eq", B["N"], vals)
        g = torch.einsum("qad,ea,e->eqd", B["dN"], vals, B["dscale"])
        return v, g

    def unpack(self, x):
        blk, dim = self.blk, self.dim
        u = torch.stack([x[d::blk] for d in range(dim)], dim=1)
        p = x[dim::blk]
        phi = x[dim + 1::blk]
        mu = x[dim + 2::blk]
        return u, p, phi, mu

    def _residual(self, x, u_n, phi_n, P):
        dim, blk = self.dim, self.blk
        dt, Re, Cn = self.dt, self.Re, self.Cn
        rho_h, eta_h = 1.0, 1.0
        rho_l = 1.0 / P["rho_ratio"]
        eta_l = 1.0 / P["eta_ratio"]
        a_rho = 0.5 * (rho_h - rho_l)
        b_rho = 0.5 * (rho_h + rho_l)
        a_eta = 0.5 * (eta_h - eta_l)
        b_eta = 0.5 * (eta_h + eta_l)
        agg = -0.5 * (rho_h - rho_l) * P["mobility"]
        cw_inv = 1.0 / (Cn * P["We"])
        grav_scale = 1.0 / P["Fr"] ** 2
        mobility = P["mobility"]
        u, p, phi, mu = self.unpack(x)
        ghat = torch.zeros(dim, dtype=torch.float64, device=self.dev)
        if self.gravity:
            ghat[-1] = -1.0
        R = torch.zeros(self.ndof, dtype=torch.float64, device=self.dev)
        # soft clamp floor: mix_props clamps below 1e-3*lo; the twin uses
        # interior data so the raw values stay well above the floor (no clamp
        # engaged -> matches the mirror's unclamped branch).  We keep the raw
        # (unclamped) value so autograd sees the linear interp slope everywhere.
        for B in self.bins:
            dJxW, N, dN, ds = B["dJxW"], B["N"], B["dN"], B["dscale"]
            h_ = B["h"]
            u_gp, gu = self._ipv(u, B)
            un_gp, _ = self._ipv(u_n, B)
            p_gp, gp = self._ip(p, B)
            phi_gp, gphi = self._ip(phi, B)
            phin_gp, _ = self._ip(phi_n, B)
            mu_gp, gmu = self._ip(mu, B)
            rho_gp = a_rho * phi_gp + b_rho
            eta_gp = a_eta * phi_gp + b_eta
            c2CI = 36.0 * 16.0 * dim
            nu_loc = eta_gp / (rho_gp * Re)
            A_t = 4.0 / dt ** 2
            Bu_t = 4.0 * (u_gp ** 2).sum(-1) / h_ ** 2
            C_t = c2CI * nu_loc ** 2 / h_ ** 4
            denom = torch.sqrt(A_t + Bu_t + C_t)
            tau = 1.0 / (denom * rho_gp)
            J_gp = agg * gmu
            ugradu = torch.einsum("eqs,eqds->eqd", u_gp, gu)
            Jgradu = torch.einsum("eqs,eqds->eqd", J_gp, gu)
            ugradphi = torch.einsum("eqs,eqs->eq", u_gp, gphi)
            fcap = cw_inv * mu_gp[..., None] * gphi
            fgrav = (rho_gp[..., None] * grav_scale) * ghat[None, None, :]
            r_mom = (rho_gp[..., None] * (u_gp - un_gp) / dt
                     + rho_gp[..., None] * ugradu + Jgradu + gp - fcap - fgrav)
            mom_body = (rho_gp[..., None] * (u_gp - un_gp) / dt
                        + rho_gp[..., None] * ugradu + Jgradu - fcap - fgrav)
            Rmom = torch.einsum("eq,qa,eqd->ead", dJxW, N, mom_body)
            symgu = gu + gu.transpose(2, 3)
            dNp = torch.einsum("qas,e->eqas", dN, ds)
            Rvisc = (1.0 / Re) * torch.einsum(
                "eq,eq,eqds,eqas->ead", dJxW, eta_gp, symgu, dNp)
            Rpres = -torch.einsum("eq,eq,eqad->ead", dJxW, p_gp, dNp)
            ugw = torch.einsum("eqs,eqas->eqa", u_gp, dNp)
            Rsupg = torch.einsum("eq,eq,eqa,eqd->ead", dJxW, tau, ugw, r_mom)
            Ru = Rmom + Rvisc + Rpres + Rsupg
            for d in range(dim):
                R = R.index_add(0, B["gd"][d], Ru[:, :, d].reshape(-1))
            divu = torch.einsum("eqdd->eq", gu)
            Rcont = torch.einsum("eq,qa,eq->ea", dJxW, N, divu)
            Rpspg = torch.einsum("eq,eq,eqas,eqs->ea", dJxW, tau, dNp, r_mom)
            R = R.index_add(0, B["gd"][dim], (Rcont + Rpspg).reshape(-1))
            ch_body = (phi_gp - phin_gp) / dt + ugradphi + phi_gp * divu
            Rphi = (torch.einsum("eq,qa,eq->ea", dJxW, N, ch_body)
                    + mobility * torch.einsum("eq,eqas,eqs->ea", dJxW, dNp,
                                              gmu))
            R = R.index_add(0, B["gd"][dim + 1], Rphi.reshape(-1))
            fp = phi_gp ** 3 - phi_gp
            Rmu = (torch.einsum("eq,qa,eq->ea", dJxW, N, mu_gp - fp)
                   - Cn ** 2 * torch.einsum("eq,eqas,eqs->ea", dJxW, dNp,
                                            gphi))
            R = R.index_add(0, B["gd"][dim + 2], Rmu.reshape(-1))
        # boundary rows: residual = x_dof (no-slip, pin)
        R = R.index_copy(0, self.bc_rows, x[self.bc_rows])
        return R

    def _attach_mirror(self, case, gravity):
        """Lazily build a numpy CHNSDiscrete used ONLY to supply the exact
        analytic Newton Jacobian at the (detached) current iterate.  The
        gradient never flows through this — J is detached in the Newton update
        (standard IFT trick: at convergence the J-graph term vanishes because
        R=0), so using the FD-verified analytic mirror J is both exact and much
        cheaper than a per-iterate autograd Jacobian."""
        from .chns import CHNSDiscrete
        self._mirror = CHNSDiscrete(level=0, dim=self.dim, case=case,
                                    dt=self.dt, dm=self._dm, gravity=gravity,
                                    Cn_override=self.Cn, newton_tol=1e-12)

    def march(self, phi0, P, n_steps, u0=None, newton_max=40, newton_tol=1e-12):
        """March n_steps; returns the terminal packed state x_N (autograd).

        Newton Jacobian: if a numpy mirror was attached (_attach_mirror), the
        exact analytic J is assembled at the detached iterate with the current
        (detached) float param values; otherwise falls back to a per-iterate
        autograd Jacobian.  Either way the RESIDUAL is graph-connected, so
        autograd-through-convergence returns the IFT gradient."""
        dim, blk = self.dim, self.dim + 3
        t = lambda a: torch.tensor(np.asarray(a), dtype=torch.float64,
                                   device=self.dev)
        phi0 = t(phi0)
        u_n = (torch.zeros(self.nn, dim, dtype=torch.float64, device=self.dev)
               if u0 is None else t(u0))
        phi_n = phi0.clone()
        x = torch.zeros(self.ndof, dtype=torch.float64, device=self.dev)
        for d in range(dim):
            x[d::blk] = u_n[:, d]
        x[dim + 1::blk] = phi0
        mir = getattr(self, "_mirror", None)
        if mir is not None:
            # push current (detached) param values into the mirror
            pv = {k: float(v.detach()) if torch.is_tensor(v) else float(v)
                  for k, v in P.items()}
            mir.rho_ratio = pv["rho_ratio"]; mir.rho_l = 1.0 / mir.rho_ratio
            mir.eta_ratio = pv["eta_ratio"]; mir.eta_l = 1.0 / mir.eta_ratio
            mir.We = pv["We"]; mir.Fr = pv["Fr"]
            mir.Pe = 1.0 / pv["mobility"]
            mir.agg = -0.5 * (mir.rho_h - mir.rho_l) / mir.Pe
        for _ in range(n_steps):
            xk = x.clone()
            un_np = u_n.detach().cpu().numpy()
            phin_np = phi_n.detach().cpu().numpy()
            for it in range(newton_max):
                R = self._residual(xk, u_n, phi_n, P)
                if mir is not None:
                    mir.set_history(un_np, phin_np)
                    Js = mir.jacobian(xk.detach().cpu().numpy())
                    J = torch.tensor(Js.toarray(), dtype=torch.float64,
                                     device=self.dev)
                else:
                    J = _jacobian_dense(
                        lambda z: self._residual(z, u_n, phi_n, P),
                        xk, self.ndof)
                # J detached, R graph-connected: the corrective step carries
                # the IFT param-sensitivity (at convergence R~0 so the value is
                # unchanged, but its param-gradient is exactly -J^{-1} dR/dp).
                step = torch.linalg.solve(J, -R)
                xk = xk + step
                if float(step.detach().abs().max()) < newton_tol:
                    break
            x = xk
            u_n = torch.stack([x[d::blk] for d in range(dim)], dim=1)
            phi_n = x[dim + 1::blk]
        return x

    def grads(self, phi0, params0, names, n_steps, objective,
              phi_target=None, coords=None, u0=None):
        """Build leaf tensors for the requested scalar params, march, backprop
        the objective, return {name: leaf.grad}.  params0 is a dict with keys
        rho_ratio, eta_ratio, We, mobility, Fr (floats)."""
        leaves = {}
        P = {}
        for k, v in params0.items():
            if k in names:
                leaves[k] = torch.tensor(float(v), dtype=torch.float64,
                                         device=self.dev, requires_grad=True)
                P[k] = leaves[k]
            else:
                P[k] = torch.tensor(float(v), dtype=torch.float64,
                                    device=self.dev)
        xN = self.march(phi0, P, n_steps, u0=u0)
        J = self._objective(xN, objective, phi_target, coords)
        J.backward()
        return {nm: float(leaves[nm].grad) for nm in names}

    def loss(self, phi0, params0, n_steps, objective, phi_target=None,
             coords=None, u0=None):
        """Detached scalar objective (for FD checks)."""
        with torch.no_grad():
            P = {k: torch.tensor(float(v), dtype=torch.float64, device=self.dev)
                 for k, v in params0.items()}
            xN = self.march(phi0, P, n_steps, u0=u0)
            return float(self._objective(xN, objective, phi_target, coords))

    def _lumped(self):
        w = torch.zeros(self.nn, dtype=torch.float64, device=self.dev)
        for B in self.bins:
            Ma = torch.einsum("eq,qa->ea", B["dJxW"], B["N"])
            w = w.index_add(0, B["conn"].reshape(-1), Ma.reshape(-1))
        return w

    def _objective(self, xN, objective, phi_target, coords):
        blk, dim = self.blk, self.dim
        phi = xN[dim + 1::blk]
        w = self._lumped()
        if objective == "terminal_phi_mismatch":
            tgt = torch.tensor(np.asarray(phi_target), dtype=torch.float64,
                               device=self.dev)
            return 0.5 * (w * (phi - tgt) ** 2).sum()
        elif objective == "centroid_y":
            cc = self.coords if coords is None else np.asarray(coords)
            y = torch.tensor(cc[:, 1], dtype=torch.float64, device=self.dev)
            heavy = 0.5 * (1.0 + phi)
            num = (w * y * heavy).sum()
            den = (w * heavy).sum()
            return num / den
        raise ValueError(f"unknown objective {objective!r}")


def _jacobian_dense(fn, x, ndof):
    """Dense Jacobian d fn / d x via torch.autograd.functional.jacobian,
    detached from the outer graph (used inside the Newton loop where we only
    need the linear solve operator, not its derivative)."""
    xd = x.detach().clone().requires_grad_(True)
    Jf = torch.autograd.functional.jacobian(fn, xd, vectorize=True)
    return Jf.detach()
