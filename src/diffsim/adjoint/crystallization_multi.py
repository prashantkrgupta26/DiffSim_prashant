"""M-component + K-species crystallization adjoint: CrystalEnergy protocol,
the first parametric implementation AdditiveCrystalEnergy, and the discrete
2M+K block operator CrystalCHDiscrete.

f(phi, psi) = f_FH(phi; chi) + sum_k phi_k [ q(psi_k) dsig_k + p(psi_k) drive_k ]
drive_k = dh_k (T / Tm_k - 1)

``crystallizable`` is a tuple of M-species indices K ⊆ {0..M-1}.
``dsig``/``dh``/``Tm`` are dicts keyed by crystallizable index.

The ``CrystalEnergy`` protocol is the drop-in contract for the K>0 adjoint engine
(non-parametric / neural implementations replace this in sub-project ②).

COMPLEX-DTYPE PRESERVATION:  ``_q/_p`` etc. are pure polynomials in ψ; FHMultiEnergy
already preserves dtype; dsig/dh/Tm are plain scalars multiplied against arrays, so
complex-step on a param flows through.  No float casts anywhere on the phi/psi/param
path.

CrystalCHDiscrete DOF LAYOUT (node-major, blk = 2*M + K per node):
  global dof (node a, field f):  a * blk + f
  f = 2*i      -> phi_i  (i = 0 .. M-1)
  f = 2*i + 1  -> mu_i   (i = 0 .. M-1)
  f = 2*M + j  -> psi_k  where k = crystallizable[j], j = 0 .. K-1
"""
import numpy as np
import scipy.sparse as sp
from .multiphase import FHMultiEnergy
from .crystallization import _q, _qp, _qpp, _p, _pp, _ppp


class CrystalEnergy:
    """Protocol: coupled bulk free energy f(phi, psi).  Exposes the M exchange
    potentials df/dphi_i, the K Allen-Cahn driving forces df/dpsi_k, the Hessian
    blocks, and per-parameter cotangents.  Implementations: AdditiveCrystalEnergy
    (parametric); NeuralCrystalEnergy (sub-project 2, drop-in)."""

    M = 0
    crystallizable = ()
    param_names = ()

    def dfdphi(self, phis, psis):
        raise NotImplementedError

    def dfdpsi(self, phis, psis):
        raise NotImplementedError

    def d2fdphidphi(self, phis, psis):
        raise NotImplementedError

    def d2fdphidpsi(self, phis, psis):
        raise NotImplementedError

    def d2fdpsidpsi(self, phis, psis):
        raise NotImplementedError

    def dfdphi_dparam(self, phis, psis, name):
        raise NotImplementedError

    def dfdpsi_dparam(self, phis, psis, name):
        raise NotImplementedError


class AdditiveCrystalEnergy(CrystalEnergy):
    r"""f = f_FH(phi;chi) + sum_k phi_k[q(psi_k)dsig_k + p(psi_k)drive_k],
    drive_k = dh_k(T/Tm_k - 1).  chi/N via FHMultiEnergy; dsig/dh/Tm are the
    learnable crystal-bulk params.  K = crystallizable species subset."""

    def __init__(self, chi, N, crystallizable, dsig, dh, Tm, T=0.5, breg=0.0):
        self.fh = FHMultiEnergy(chi, N, breg=breg)
        self.M = self.fh.M
        self.crystallizable = tuple(int(k) for k in crystallizable)
        self.dsig = {int(k): dsig[k] for k in self.crystallizable}
        self.dh = {int(k): dh[k] for k in self.crystallizable}
        self.Tm = {int(k): Tm[k] for k in self.crystallizable}
        self.T = float(T)
        cp = tuple(f"{b}_{k}" for k in self.crystallizable
                   for b in ("dsig", "dh", "Tm"))
        self.param_names = self.fh.param_names + cp

    def _Tfactor(self, k):
        return self.T / self.Tm[k] - 1.0

    def _drive(self, k):
        return self.dh[k] * self._Tfactor(k)

    def _kpsi(self, psis, k):          # psi array for crystallizable species k
        return psis[self.crystallizable.index(k)]

    def dfdphi(self, phis, psis):
        mu = list(self.fh.mu(phis))
        for k in self.crystallizable:
            ps = self._kpsi(psis, k)
            mu[k] = mu[k] + _q(ps) * self.dsig[k] + _p(ps) * self._drive(k)
        return mu

    def dfdpsi(self, phis, psis):
        out = []
        for k in self.crystallizable:
            ps = self._kpsi(psis, k)
            out.append(phis[k] * (_qp(ps) * self.dsig[k] + _pp(ps) * self._drive(k)))
        return out

    def d2fdphidphi(self, phis, psis):
        return self.fh.dmu_dphi(phis)            # chi is psi-independent in v1

    def d2fdphidpsi(self, phis, psis):
        # [M][K]; only the diagonal crystallizable coupling is nonzero:
        # d(dfdphi_k)/dpsi_k = q'(psi_k)dsig_k + p'(psi_k)drive_k
        K = len(self.crystallizable)
        H = [[np.zeros_like(phis[0]) for _ in range(K)] for _ in range(self.M)]
        for j, k in enumerate(self.crystallizable):
            ps = self._kpsi(psis, k)
            H[k][j] = _qp(ps) * self.dsig[k] + _pp(ps) * self._drive(k)
        return H

    def d2fdpsidpsi(self, phis, psis):
        K = len(self.crystallizable)
        H = [[np.zeros_like(phis[0]) for _ in range(K)] for _ in range(K)]
        for j, k in enumerate(self.crystallizable):
            ps = self._kpsi(psis, k)
            H[j][j] = phis[k] * (_qpp(ps) * self.dsig[k] + _ppp(ps) * self._drive(k))
        return H

    def dfdphi_dparam(self, phis, psis, name):
        z = [np.zeros_like(phis[0]) for _ in range(self.M)]
        if name.startswith(("dsig_", "dh_", "Tm_")):
            b, k = name.rsplit("_", 1)
            k = int(k)
            ps = self._kpsi(psis, k)
            if b == "dsig":
                z[k] = _q(ps)
            elif b == "dh":
                z[k] = _p(ps) * self._Tfactor(k)
            else:   # Tm
                z[k] = _p(ps) * (-self.dh[k] * self.T / self.Tm[k] ** 2)
            return z
        return self.fh.dmu_dparam(phis, name)     # chi/N

    def dfdpsi_dparam(self, phis, psis, name):
        K = len(self.crystallizable)
        z = [np.zeros_like(phis[0]) for _ in range(K)]
        if name.startswith(("dsig_", "dh_", "Tm_")):
            b, k = name.rsplit("_", 1)
            k = int(k)
            j = self.crystallizable.index(k)
            ps = self._kpsi(psis, k)
            if b == "dsig":
                z[j] = phis[k] * _qp(ps)
            elif b == "dh":
                z[j] = phis[k] * _pp(ps) * self._Tfactor(k)
            else:   # Tm
                z[j] = phis[k] * _pp(ps) * (-self.dh[k] * self.T / self.Tm[k] ** 2)
        return z


# --------------------------------------------------------------------------
# discrete mesh operator: coupled multi-CH (2M) + multi-Allen-Cahn (K) block
# --------------------------------------------------------------------------
class CrystalCHDiscrete:
    r"""Vectorised numpy residual/Jacobian for the coupled M-species
    Cahn-Hilliard + K-species Allen-Cahn crystallization system, node-major
    ``blk = 2*M + K`` block.  The M-generic, K>0 sibling of
    ``multiphase.MultiCHDiscrete`` (K=0) and ``crystallization.CACHDiscrete``
    (M=1, K=1) — every bulk-energy quantity routes through a ``CrystalEnergy``
    object so a non-parametric / neural coupled energy is a drop-in.

    Weak residuals at Gauss points (per element), with sigma = c0/dt,
    dfdphi = energy.dfdphi(phi, psi) [M-vector], dfdpsi = energy.dfdpsi [K-vector]:

      R_phi_i = Int N (sigma phi_i - hist_phi_i) + Int gradN . (sum_j Lam_ij grad mu_j)
      R_mu_i  = Int N (mu_i - dfdphi_i) - kappa_i Int gradN . grad phi_i
      R_psi_k = Int N (sigma psi_k - hist_psi_k)
                + L_k ( Int N dfdpsi_k + eps2_k Int gradN . grad psi_k )

    Jacobian cross-blocks beyond the K=0 set (H2pc = d2f/dphi dpsi [M x K],
    H2cc = d2f/dpsi dpsi [K x K]):
      dR_mu_i / dpsi_k  = - WM(H2pc[i][j])
      dR_psi_k / dphi_i =  L_k WM(H2pc[i][j])
      dR_psi_k / dpsi_k' = L_k WM(H2cc[j][j'])  (+ sigma NN + L_k eps2_k LL on j==j')
    """

    def __init__(self, dm, M, crystallizable):
        T = dm.constraints.T
        assert T.shape[0] == T.shape[1] and (
            abs(T - sp.eye(T.shape[0])).nnz == 0), \
            "adjoint gate assumes constraints.T == identity (uniform mesh)"
        self.dm = dm
        self.M = int(M)
        self.crystallizable = tuple(int(k) for k in crystallizable)
        self.K = len(self.crystallizable)
        self.blk = 2 * self.M + self.K
        self.dim = dm.dim
        self.nn = dm.n_nodes
        self.ndof = self.blk * self.nn
        blk = self.blk
        self.bins = []
        for pv, b in dm.bins.items():
            tb = dm.tables_by_p[pv]
            conn = dm.mesh.conn_of[pv].astype(np.int64)
            h = np.asarray(dm.mesh.tree.h()[dm.mesh.bins[pv]], np.float64)
            N = np.asarray(tb.N, np.float64)
            dN = np.asarray(tb.dN, np.float64)
            w = np.asarray(tb.w, np.float64)
            ne, nbf = conn.shape
            dscale = 2.0 / h
            jac = (h / 2.0) ** self.dim
            dJxW = w[None, :] * jac[:, None]
            gdof = (conn[:, :, None] * blk
                    + np.arange(blk)[None, None, :]).reshape(ne, blk * nbf)
            self.bins.append(dict(conn=conn, N=N, dN=dN, ne=ne, nbf=nbf,
                                  nqp=N.shape[0], dscale=dscale, dJxW=dJxW,
                                  gdof=gdof))

    def interp(self, field):
        out = []
        for B in self.bins:
            vals = field[B["conn"]]
            v = np.einsum("qa,ea->eq", B["N"], vals)
            g = np.einsum("qad,ea,e->eqd", B["dN"], vals, B["dscale"])
            out.append((v, g))
        return out

    def _rowslices(self):
        """gdof column slices for phi_i / mu_i / psi_j rows."""
        M, blk = self.M, self.blk
        phi_c = [2 * i for i in range(M)]
        mu_c = [2 * i + 1 for i in range(M)]
        psi_c = [2 * M + j for j in range(self.K)]
        return phi_c, mu_c, psi_c

    def assemble(self, phis, mus, psis, hist_phi_gp, hist_psi_gp, params,
                 want_jac=True):
        """R (len ndof) and, if want_jac, J = dR/dx (csr ndof x ndof).
        phis/mus are length-M lists, psis is length-K list of length-nn arrays;
        hist_phi_gp[i][bi], hist_psi_gp[j][bi] are [ne, nqp] BDF history loads."""
        M, K, blk = self.M, self.K, self.blk
        cryst = self.crystallizable
        mobility = params.get("mobility")
        if mobility is None:
            from .neural_multiphase import MobilityClosure as _MC
            mobility = _MC("const", M=M, onsager=np.asarray(params["onsager"]))
        kap = params["kappa"]
        eps2, L = params["eps2"], params["L"]
        energy = params["energy"]
        sigma = params["sigma"]
        pi = [self.interp(phis[i]) for i in range(M)]
        mi = [self.interp(mus[i]) for i in range(M)]
        si = [self.interp(psis[j]) for j in range(K)]
        # dtype promotion so complex-step (phi/mu/psi, kappa/eps2/L, mobility,
        # and any energy param via chi/N/dsig/dh/Tm) flows through the scatter.
        dsrc = [np.asarray(x).dtype for x in (*phis, *mus, *psis)]
        dsrc.append(np.asarray(kap).dtype)
        for d in (eps2, L):
            if len(d):
                dsrc.append(np.asarray(list(d.values())).dtype)
        if mobility.onsager is not None:
            dsrc.append(np.asarray(mobility.onsager).dtype)
        for v in mobility.coeffs.values():
            dsrc.append(np.asarray(v).dtype)
        fh = getattr(energy, "fh", None)
        if fh is not None:
            dsrc.append(np.asarray(fh.chi).dtype)
            dsrc.append(np.asarray(fh.N).dtype)
        for dname in ("dsig", "dh", "Tm"):
            d = getattr(energy, dname, None)
            if d:
                dsrc.append(np.asarray(list(d.values())).dtype)
        R = np.zeros(self.ndof, dtype=np.result_type(*dsrc, np.float64))
        rows, cols, vals = [], [], []
        for bi, B in enumerate(self.bins):
            dJxW, N, dN, dscale = B["dJxW"], B["N"], B["dN"], B["dscale"]
            gdof, ne, nbf = B["gdof"], B["ne"], B["nbf"]
            phi_gp = [pi[i][bi][0] for i in range(M)]
            gphi = [pi[i][bi][1] for i in range(M)]
            mu_gp = [mi[i][bi][0] for i in range(M)]
            gmu = [mi[i][bi][1] for i in range(M)]
            psi_gp = [si[j][bi][0] for j in range(K)]
            gpsi = [si[j][bi][1] for j in range(K)]
            dfdphi = energy.dfdphi(phi_gp, psi_gp)
            dfdpsi = energy.dfdpsi(phi_gp, psi_gp)
            NN = np.einsum("eq,qa,qb->eab", dJxW, N, N)
            LL = np.einsum("eq,qad,qbd->eab",
                           dJxW * dscale[:, None] ** 2, dN, dN)
            WM = lambda wgt: np.einsum("eq,qa,qb->eab", dJxW * wgt, N, N)
            Lam = mobility.matrix(phi_gp)
            Ae = (np.zeros((ne, blk * nbf, blk * nbf))
                  if want_jac else None)
            H2pp = energy.d2fdphidphi(phi_gp, psi_gp) if want_jac else None
            H2pc = energy.d2fdphidpsi(phi_gp, psi_gp) if want_jac else None
            H2cc = energy.d2fdpsidpsi(phi_gp, psi_gp) if want_jac else None
            # --- phi/mu rows (the K=0 multi-CH block) ---
            for i in range(M):
                flux = np.zeros_like(gmu[0])
                for j in range(M):
                    flux = flux + Lam[i][j][:, :, None] * gmu[j]
                gN_flux = np.einsum("qad,e,eqd->eqa", dN, dscale, flux)
                Rphi = (np.einsum("eq,qa->ea",
                                  dJxW * (sigma * phi_gp[i] - hist_phi_gp[i][bi]),
                                  N)
                        + np.einsum("eq,eqa->ea", dJxW, gN_flux))
                np.add.at(R, gdof[:, 2 * i::blk].ravel(), Rphi.ravel())
                gN_gphi = np.einsum("qad,e,eqd->eqa", dN, dscale, gphi[i])
                Rmu = (np.einsum("eq,qa->ea",
                                 dJxW * (mu_gp[i] - dfdphi[i]), N)
                       - kap[i] * np.einsum("eq,eqa->ea", dJxW, gN_gphi))
                np.add.at(R, gdof[:, 2 * i + 1::blk].ravel(), Rmu.ravel())
                if not want_jac:
                    continue
                Ae[:, 2 * i::blk, 2 * i::blk] += sigma * NN
                for j in range(M):
                    Lij_LL = np.einsum("eq,e,qad,qbd->eab",
                                       dJxW * Lam[i][j], dscale ** 2, dN, dN)
                    Ae[:, 2 * i::blk, 2 * j + 1::blk] += Lij_LL
                    Ae[:, 2 * i + 1::blk, 2 * j::blk] += -WM(H2pp[i][j])
                dMii_dphii = mobility._dmatrix_dphi_diag(phi_gp, i)
                if dMii_dphii is not None:
                    dflux = dMii_dphii[:, :, None] * gmu[i]
                    dM_term = np.einsum("eq,eqd,qad,e,qb->eab",
                                        dJxW, dflux, dN, dscale, N)
                    Ae[:, 2 * i::blk, 2 * i::blk] += dM_term
                Ae[:, 2 * i + 1::blk, 2 * i::blk] += -kap[i] * LL
                Ae[:, 2 * i + 1::blk, 2 * i + 1::blk] += NN
                # dR_mu_i / dpsi_k  = - WM(H2pc[i][j])
                for j in range(K):
                    Ae[:, 2 * i + 1::blk, 2 * M + j::blk] += -WM(H2pc[i][j])
            # --- psi rows (the multi-Allen-Cahn block) ---
            for j, k in enumerate(cryst):
                prow = 2 * M + j
                gN_gpsi = np.einsum("qad,e,eqd->eqa", dN, dscale, gpsi[j])
                Rpsi = (np.einsum("eq,qa->ea",
                                  dJxW * (sigma * psi_gp[j] - hist_psi_gp[j][bi]),
                                  N)
                        + L[k] * (np.einsum("eq,qa->ea", dJxW * dfdpsi[j], N)
                                  + eps2[k] * np.einsum("eq,eqa->ea",
                                                        dJxW, gN_gpsi)))
                np.add.at(R, gdof[:, prow::blk].ravel(), Rpsi.ravel())
                if not want_jac:
                    continue
                # dR_psi_k / dphi_i = L_k WM(H2pc[i][j])
                for i in range(M):
                    Ae[:, prow::blk, 2 * i::blk] += L[k] * WM(H2pc[i][j])
                # dR_psi_k / dpsi_j'
                for jp in range(K):
                    block = L[k] * WM(H2cc[j][jp])
                    if jp == j:
                        block = block + sigma * NN + L[k] * eps2[k] * LL
                    Ae[:, prow::blk, 2 * M + jp::blk] += block
            if want_jac:
                rows.append(np.repeat(gdof, blk * nbf, axis=1).ravel())
                cols.append(np.tile(gdof, (1, blk * nbf)).ravel())
                vals.append(Ae.ravel())
        J = None
        if want_jac:
            J = sp.coo_matrix(
                (np.concatenate(vals),
                 (np.concatenate(rows), np.concatenate(cols))),
                shape=(self.ndof, self.ndof)).tocsr()
        return R, J

    def dR_dparam(self, phis, mus, psis, params, name):
        """Explicit dR/dp as a length-ndof vector at a committed state.  Engine
        params (eps2_k, L_k, kappa_i, mob_*) are handled directly; bulk-energy
        params route through energy.dfdphi_dparam (mu rows) and
        energy.dfdpsi_dparam (psi rows)."""
        M, K, blk = self.M, self.K, self.blk
        cryst = self.crystallizable
        energy = params["energy"]
        eps2, L, kap = params["eps2"], params["L"], params["kappa"]
        mobility = params.get("mobility")
        if mobility is None and "onsager" in params:
            from .neural_multiphase import MobilityClosure as _MC
            mobility = _MC("const", M=M, onsager=np.asarray(params["onsager"]))
        pi = [self.interp(phis[i]) for i in range(M)]
        mi = [self.interp(mus[i]) for i in range(M)]
        si = [self.interp(psis[j]) for j in range(K)]
        out = np.zeros(self.ndof)
        for bi, B in enumerate(self.bins):
            dJxW, N, dN, dscale = B["dJxW"], B["N"], B["dN"], B["dscale"]
            gdof = B["gdof"]
            phi_gp = [pi[i][bi][0] for i in range(M)]
            psi_gp = [si[j][bi][0] for j in range(K)]

            def addphi(i, arr):
                np.add.at(out, gdof[:, 2 * i::blk].ravel(), arr.ravel())

            def addmu(i, arr):
                np.add.at(out, gdof[:, 2 * i + 1::blk].ravel(), arr.ravel())

            def addpsi(j, arr):
                np.add.at(out, gdof[:, 2 * M + j::blk].ravel(), arr.ravel())

            if name.startswith("eps2_"):
                k = int(name.split("_")[1])
                j = cryst.index(k)
                gN_gpsi = np.einsum("qad,e,eqd->eqa", dN, dscale, si[j][bi][1])
                addpsi(j, L[k] * np.einsum("eq,eqa->ea", dJxW, gN_gpsi))
            elif name.startswith("L_"):
                k = int(name.split("_")[1])
                j = cryst.index(k)
                dfdpsi = energy.dfdpsi(phi_gp, psi_gp)
                gN_gpsi = np.einsum("qad,e,eqd->eqa", dN, dscale, si[j][bi][1])
                addpsi(j, np.einsum("eq,qa->ea", dJxW * dfdpsi[j], N)
                       + eps2[k] * np.einsum("eq,eqa->ea", dJxW, gN_gpsi))
            elif name.startswith("kappa_"):
                i = int(name.split("_")[1])
                gN_gphi = np.einsum("qad,e,eqd->eqa", dN, dscale, pi[i][bi][1])
                addmu(i, -np.einsum("eq,eqa->ea", dJxW, gN_gphi))
            elif name.startswith("mob_"):
                dLam = mobility.dmatrix_dparam(phi_gp, name)
                for i in range(M):
                    contrib = np.zeros_like(mi[0][bi][1])
                    for j in range(M):
                        contrib = contrib + dLam[i][j][:, :, None] * mi[j][bi][1]
                    gN = np.einsum("qad,e,eqd->eqa", dN, dscale, contrib)
                    addphi(i, np.einsum("eq,eqa->ea", dJxW, gN))
            else:  # bulk-energy coefficient via the energy object
                dphi = energy.dfdphi_dparam(phi_gp, psi_gp, name)
                for i in range(M):
                    addmu(i, -np.einsum("eq,qa->ea", dJxW * dphi[i], N))
                dpsi = energy.dfdpsi_dparam(phi_gp, psi_gp, name)
                for j, k in enumerate(cryst):
                    addpsi(j, L[k] * np.einsum("eq,qa->ea", dJxW * dpsi[j], N))
        return out
