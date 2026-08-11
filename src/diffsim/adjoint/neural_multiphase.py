r"""Gauge-anchored learnable bulk free energy for the M-component Cahn-Hilliard
adjoint (M6: free-energy learning).

``BasisMultiEnergy`` = Flory-Huggins base (FHMultiEnergy) + a per-species
gauge-anchored polynomial correction to each exchange potential mu_i.  The
correction basis is shifted-Legendre degrees k >= 2 on the composition domain
[lo, hi].  P_k (k >= 2) are L2-orthogonal to {1, phi_i} on [lo, hi] by
construction — no explicit projection is needed and ``gauge_residual`` is ~1e-16.

The per-species, separable (v1) form keeps the correction tractable:

    mu_i_corr = sum_k gamma_{i,k} P_k(u_i),   u_i = (phi_i - mid) / half

    d mu_i_corr / d phi_j = delta_{ij} * sum_k gamma_{i,k} P'_k(u_i) / half

    d mu_i_corr / d gamma_{i,k} = P_k(u_i)    (all other species/degree: 0)

The implementation PRESERVES input dtype (complex arrays flow through for
complex-step gradient verification) — no float64 casts anywhere in the
correction path, mirroring the convention in FHMultiEnergy.

Implements the ``MultiEnergy`` protocol so it is a drop-in replacement for
FHMultiEnergy in MultiCHForward / MultiCHAdjoint.
"""
import numpy as np
from .multiphase import MultiEnergy, FHMultiEnergy


# ---------------------------------------------------------------------------
# MobilityClosure — named learnable mobility M(phi; a)
# ---------------------------------------------------------------------------
class MobilityClosure:
    """NAMED mobility M(phi;a).  'const' = the given Onsager matrix (existing
    behavior); 'phi_diag' = M_ii = m0*(1 + c*phi_i), off-diag 0 (learnable
    m0, c).

    ``.matrix(phi_gp)``      — M×M list-of-lists of [ne, nqp] arrays
    ``.dmatrix_dparam(phi_gp, name)`` — M×M list-of-lists of [ne, nqp] arrays
    ``.param_names``         — tuple of learnable parameter names

    Preserves complex dtype so complex-step verification flows through.
    """

    def __init__(self, name="const", M=2, onsager=None, coeffs=None):
        self.name = name
        self.M = int(M)
        self.onsager = None if onsager is None else np.asarray(onsager)
        self.coeffs = dict(coeffs or {})
        if name == "const":
            self.param_names = ()
        elif name == "phi_diag":
            self.coeffs.setdefault("mob_m0", 1.0)
            self.coeffs.setdefault("mob_c", 0.0)
            self.param_names = ("mob_m0", "mob_c")
        else:
            raise ValueError(f"unknown mobility closure {name!r}")

    def matrix(self, phi_gp):
        """Returns M×M list-of-lists of [ne, nqp] arrays at Gauss points."""
        M = self.M
        if self.name == "const":
            return [[self.onsager[i, j] * np.ones_like(phi_gp[0])
                     for j in range(M)] for i in range(M)]
        # phi_diag: M_ii = m0*(1 + c*phi_i), off-diagonal 0
        m0 = self.coeffs["mob_m0"]
        c = self.coeffs["mob_c"]
        out = [[np.zeros_like(phi_gp[0]) for _ in range(M)] for _ in range(M)]
        for i in range(M):
            out[i][i] = m0 * (1.0 + c * phi_gp[i])
        return out

    def dmatrix_dparam(self, phi_gp, name):
        """Returns M×M list-of-lists of [ne, nqp] arrays, dM/d(param name)."""
        M = self.M
        out = [[np.zeros_like(phi_gp[0]) for _ in range(M)] for _ in range(M)]
        if self.name == "phi_diag":
            if name == "mob_m0":
                c = self.coeffs["mob_c"]
                for i in range(M):
                    out[i][i] = (1.0 + c * phi_gp[i])
            elif name == "mob_c":
                m0 = self.coeffs["mob_m0"]
                for i in range(M):
                    out[i][i] = m0 * phi_gp[i]
        return out

    def _dmatrix_dphi_diag(self, phi_gp, i):
        """dM_ii/dphi_i at Gauss points: [ne, nqp] array, or None if zero.
        Used by assemble() to add the phi-dependent-mobility Jacobian term.
        For 'const': returns None (dM/dphi = 0, term absent).
        For 'phi_diag': dM_ii/dphi_i = m0 * c."""
        if self.name == "const":
            return None
        if self.name == "phi_diag":
            m0 = self.coeffs["mob_m0"]
            c = self.coeffs["mob_c"]
            # m0*c is a scalar; broadcast to [ne, nqp] matching phi_gp[i] shape
            return m0 * c * np.ones_like(phi_gp[i])
        return None


# ---------------------------------------------------------------------------
# Numpy port of neural_energy._legendre — preserves complex dtype
# ---------------------------------------------------------------------------
def _legendre_np(u, k):
    """Shifted-Legendre P_k(u) and derivative dP_k/du on u in [-1, 1].

    Closed-form expressions identical to ``neural_energy._legendre`` but using
    numpy ops so they accept numpy arrays of any dtype, including complex128
    (required by the complex-step derivative tests).  k must be in 0..5.

    k >= 2 are the gauge-anchored modes orthogonal to {1, u} on [-1, 1];
    k in {0, 1} are the unidentifiable T0/T1 gauge modes included for
    completeness / diagnostics.
    """
    if k == 0:
        return np.ones_like(u), np.zeros_like(u)
    if k == 1:
        return u + np.zeros_like(u), np.ones_like(u)
    if k == 2:
        return 0.5 * (3.0 * u * u - 1.0), 3.0 * u
    if k == 3:
        return 0.5 * (5.0 * u ** 3 - 3.0 * u), 0.5 * (15.0 * u * u - 3.0)
    if k == 4:
        return ((35.0 * u ** 4 - 30.0 * u * u + 3.0) / 8.0,
                (140.0 * u ** 3 - 60.0 * u) / 8.0)
    if k == 5:
        return ((63.0 * u ** 5 - 70.0 * u ** 3 + 15.0 * u) / 8.0,
                (315.0 * u ** 4 - 210.0 * u * u + 15.0) / 8.0)
    raise ValueError(f"basis degree {k} not in 0..5")


# ---------------------------------------------------------------------------
# BasisMultiEnergy
# ---------------------------------------------------------------------------
class BasisMultiEnergy(MultiEnergy):
    """FH base + a per-species gauge-anchored polynomial correction to each
    exchange potential mu_i.  Correction basis are shifted-Legendre degrees k>=2
    on the composition domain [lo, hi], L2-orthogonal to {1, phi_i} by
    construction (the M-component T0/T1 gauge; see neural_energy).

    Parameters
    ----------
    chi : array, shape (M+1, M+1)
        Symmetric Flory-Huggins interaction matrix (diagonal unused).
    N : array, shape (M+1,)
        Chain-length per species (including solvent at index M).
    degrees : tuple[int]
        Shifted-Legendre degrees included in the correction (each >= 2).
    coeffs : dict[str, float] or None
        Initial basis coefficients keyed by ``'basis_{i}_{k}'``.
        Missing keys default to 0.0.
    dom : tuple[float, float]
        Composition domain (lo, hi) over which the gauge orthogonality holds.
    breg : float
        Barrier regularisation passed through to FHMultiEnergy.
    """

    def __init__(self, chi, N, degrees=(2, 3), coeffs=None,
                 dom=(0.05, 0.95), breg=0.0):
        self.fh = FHMultiEnergy(chi, N, breg=breg)
        self.M = self.fh.M
        self.degrees = tuple(int(k) for k in degrees)
        self.lo, self.hi = float(dom[0]), float(dom[1])
        self.mid = 0.5 * (self.hi + self.lo)
        self.half = 0.5 * (self.hi - self.lo)
        # gamma[(i, k)] : learnable scalar coefficient; complex-safe on use
        self.gamma = {(i, k): 0.0
                      for i in range(self.M) for k in self.degrees}
        if coeffs:
            for nm, v in coeffs.items():
                _, si, sk = nm.split("_")
                self.gamma[(int(si), int(sk))] = v
        self.param_names = self.fh.param_names + tuple(
            f"basis_{i}_{k}" for i in range(self.M) for k in self.degrees)

    # -- correction helpers ------------------------------------------------

    def _corr_mu(self, phis, i):
        """Scalar correction to mu_i: sum_k gamma_{i,k} P_k(u_i)."""
        u = (phis[i] - self.mid) / self.half
        out = np.zeros_like(phis[i])
        for k in self.degrees:
            pk, _ = _legendre_np(u, k)
            out = out + self.gamma[(i, k)] * pk
        return out

    def _corr_dmu(self, phis, i):
        """d(corr mu_i)/d phi_i (diagonal contribution only)."""
        u = (phis[i] - self.mid) / self.half
        out = np.zeros_like(phis[i])
        for k in self.degrees:
            _, dpk = _legendre_np(u, k)
            out = out + self.gamma[(i, k)] * dpk / self.half
        return out

    # -- MultiEnergy protocol ----------------------------------------------

    def mu(self, phis):
        base = self.fh.mu(phis)
        return [base[i] + self._corr_mu(phis, i) for i in range(self.M)]

    def dmu_dphi(self, phis):
        H = self.fh.dmu_dphi(phis)
        for i in range(self.M):
            H[i][i] = H[i][i] + self._corr_dmu(phis, i)
        return H

    def dmu_dparam(self, phis, name):
        if name.startswith("basis_"):
            _, si, sk = name.split("_")
            i_sp, k_deg = int(si), int(sk)
            u = (phis[i_sp] - self.mid) / self.half
            pk, _ = _legendre_np(u, k_deg)
            z = [np.zeros_like(phis[j]) for j in range(self.M)]
            z[i_sp] = pk
            return z
        return self.fh.dmu_dparam(phis, name)

    # -- diagnostics -------------------------------------------------------

    def gauge_residual(self, species=0, n_quad=64):
        """<corr_mu_species, 1> and <corr_mu_species, phi_species> over [lo, hi].

        Both must be ~1e-16 by construction (P_k k>=2 are L2-orthogonal to
        {1, u} on [-1,1], hence to {1, phi} on [lo,hi]).
        """
        x, w = np.polynomial.legendre.leggauss(n_quad)
        cq = self.mid + self.half * x     # quadrature nodes on [lo, hi]
        wq = self.half * w                # weights (Jacobian absorbed)
        phis = [np.full_like(cq, self.mid) for _ in range(self.M)]
        phis[species] = cq
        r = self._corr_mu(phis, species)
        return float((wq * r).sum()), float((wq * cq * r).sum())
