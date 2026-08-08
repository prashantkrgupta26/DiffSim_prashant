r"""Sub-project 2: learnable coupled crystallization free energy.

f(phi, psi) = f_base(phi)  +  sum_{k in crystallizable} phi_k * h_k(psi_k)
  f_base  = BasisMultiEnergy (FH + gauge-anchored Legendre phi-correction)
  h_k(psi) = sum_{b in deg_psi} c_{k,b} L_b(u),  u = 2*psi - 1   (b >= 1)

The b=0 constant mode is EXCLUDED: h_k -> h_k + const shifts mu_k by a constant,
a pure {phi_k} conservation gauge on the conserved phi_k -> unidentifiable.

Implements the CrystalEnergy protocol so CrystalCHDiscrete/Forward/Adjoint are a
drop-in (zero engine change).  Complex-dtype preserving (pure-polynomial basis,
coeffs multiply arrays) for complex-step verification.
"""
import numpy as np
from .neural_multiphase import BasisMultiEnergy, _legendre_np
from .crystallization_multi import CrystalEnergy


def _legendre2_np(u, k):
    """Second derivative d^2 P_k / du^2 on [-1, 1], complex-safe, k in 0..5."""
    if k == 0 or k == 1:
        return np.zeros_like(u)
    if k == 2:
        return 3.0 * np.ones_like(u)
    if k == 3:
        return 15.0 * u
    if k == 4:
        return (105.0 * u * u - 15.0) / 2.0
    if k == 5:
        return (315.0 * u ** 3 - 105.0 * u) / 2.0
    raise ValueError(f"basis degree {k} not in 0..5")


class NeuralCrystalEnergy(CrystalEnergy):
    def __init__(self, chi, N, crystallizable, deg_psi=(1, 2), coeffs=None,
                 dom_phi=(0.05, 0.95), breg=0.0, basis_degrees=(2, 3),
                 basis_coeffs=None):
        self.base = BasisMultiEnergy(chi, N, degrees=basis_degrees,
                                     coeffs=basis_coeffs, dom=dom_phi, breg=breg)
        self.M = self.base.M
        self.crystallizable = tuple(int(k) for k in crystallizable)
        self.deg_psi = tuple(int(b) for b in deg_psi)
        assert all(b >= 1 for b in self.deg_psi), \
            "deg_psi excludes b=0 (conserved-phi_k gauge, unidentifiable)"
        self.c = {(k, b): 0.0
                  for k in self.crystallizable for b in self.deg_psi}
        if coeffs:
            for nm, v in coeffs.items():
                _, sk, sb = nm.split("_")
                self.c[(int(sk), int(sb))] = v
        self.param_names = self.base.param_names + tuple(
            f"cpl_{k}_{b}" for k in self.crystallizable for b in self.deg_psi)

    # -- coupling helpers (h_k and derivatives in psi) --------------------
    def _kpsi(self, psis, k):
        return psis[self.crystallizable.index(k)]

    def _h(self, psi, k):
        u = 2.0 * psi - 1.0
        out = np.zeros_like(psi)
        for b in self.deg_psi:
            pk, _ = _legendre_np(u, b)
            out = out + self.c[(k, b)] * pk
        return out

    def _hp(self, psi, k):                       # dh/dpsi = 2 * sum c L_b'(u)
        u = 2.0 * psi - 1.0
        out = np.zeros_like(psi)
        for b in self.deg_psi:
            _, dpk = _legendre_np(u, b)
            out = out + self.c[(k, b)] * dpk * 2.0
        return out

    def _hpp(self, psi, k):                       # d2h/dpsi2 = 4 * sum c L_b''(u)
        u = 2.0 * psi - 1.0
        out = np.zeros_like(psi)
        for b in self.deg_psi:
            out = out + self.c[(k, b)] * _legendre2_np(u, b) * 4.0
        return out

    # -- CrystalEnergy protocol ------------------------------------------
    def dfdphi(self, phis, psis):
        mu = list(self.base.mu(phis))
        for k in self.crystallizable:
            mu[k] = mu[k] + self._h(self._kpsi(psis, k), k)
        return mu

    def dfdpsi(self, phis, psis):
        return [phis[k] * self._hp(self._kpsi(psis, k), k)
                for k in self.crystallizable]

    def d2fdphidphi(self, phis, psis):
        return self.base.dmu_dphi(phis)          # coupling is linear in phi

    def d2fdphidpsi(self, phis, psis):
        K = len(self.crystallizable)
        H = [[np.zeros_like(phis[0]) for _ in range(K)] for _ in range(self.M)]
        for j, k in enumerate(self.crystallizable):
            H[k][j] = self._hp(self._kpsi(psis, k), k)
        return H

    def d2fdpsidpsi(self, phis, psis):
        K = len(self.crystallizable)
        H = [[np.zeros_like(phis[0]) for _ in range(K)] for _ in range(K)]
        for j, k in enumerate(self.crystallizable):
            H[j][j] = phis[k] * self._hpp(self._kpsi(psis, k), k)
        return H

    def dfdphi_dparam(self, phis, psis, name):
        if name.startswith("cpl_"):
            _, sk, sb = name.split("_")
            k, b = int(sk), int(sb)
            z = [np.zeros_like(phis[0]) for _ in range(self.M)]
            pk, _ = _legendre_np(2.0 * self._kpsi(psis, k) - 1.0, b)
            z[k] = pk
            return z
        return self.base.dmu_dparam(phis, name)   # chi/N/basis_i_k

    def dfdpsi_dparam(self, phis, psis, name):
        K = len(self.crystallizable)
        z = [np.zeros_like(phis[0]) for _ in range(K)]
        if name.startswith("cpl_"):
            _, sk, sb = name.split("_")
            k, b = int(sk), int(sb)
            j = self.crystallizable.index(k)
            _, dpk = _legendre_np(2.0 * self._kpsi(psis, k) - 1.0, b)
            z[j] = phis[k] * dpk * 2.0
        return z

    # -- diagnostic -------------------------------------------------------
    def coupling_gauge_residual(self, k, n_quad=64):
        """<h_k, 1> over u in [-1, 1].  ~0 by construction: h_k is spanned by
        L_{b>=1}, all L2-orthogonal to the constant mode L_0."""
        x, w = np.polynomial.legendre.leggauss(n_quad)
        psi = 0.5 * (x + 1.0)          # u = 2psi-1 = x
        return float((w * self._h(psi, k)).sum())
