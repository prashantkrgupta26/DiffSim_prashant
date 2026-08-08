"""M-component + K-species crystallization adjoint: CrystalEnergy protocol and
the first parametric implementation AdditiveCrystalEnergy.

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
"""
import numpy as np
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
