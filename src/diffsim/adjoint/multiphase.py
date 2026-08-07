r"""M-component phase-separation adjoint (K=0): discrete IFT adjoint through the
multi-Cahn-Hilliard BDF march, the M-generic sibling of adjoint/phasefield.py.

Parity target: physics/multiphase.MultiPhaseStepper (K=0, bulk='p1', const mob).
The bulk free energy sits behind the MultiEnergy protocol so free-energy
learning (M6) reuses the identical cotangent path (a neural / extended-FH energy
is a drop-in replacement for FHMultiEnergy).

FIELDS AND LAYOUT.  n = M + 1 species with volume fractions phi (sum = 1; the
LAST species, index M, is the eliminated solvent phi_s = 1 - sum phi_i).  Each
retained species i = 0..M-1 carries a conserved (phi_i, mu_i) CH pair.
Node-major dof layout, block = 2M:  global dof of (node a, field f) is
a*2M + f, with f = 2i -> phi_i and f = 2i+1 -> mu_i.

THE ADJOINT IS THE IFT ADJOINT.  At each committed step the forward Newton has
driven R_n(x_n; x_hist, p) = 0.  For J = sum_n j(x_n),

    J_n^T lam_n = dj/dx_n - sum_{k>=1} (dR_{n+k}/dx_n)^T lam_{n+k}
    dJ/dp       = - sum_n lam_n^T (dR_n/dp)

where J_n = dR_n/dx_n is EXACTLY the converged forward Jacobian, and the BDF
history couples ALL M conserved phi-fields (each carries a -(ch_k/dt) Mass block
on its phi-rows).

Gate scope: uniform mesh (constraints.T == identity), natural no-flux BCs, BDF1
and variable-coefficient BDF2."""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu


# --------------------------------------------------------------------------
# bulk free energy: the MultiEnergy protocol + the parametric FH implementation
# --------------------------------------------------------------------------
class MultiEnergy:
    """Protocol: a bulk free energy exposing the M exchange potentials, their
    phi-Jacobian, and per-parameter cotangents.  Implementations: FHMultiEnergy
    (parametric FH, rung 1); a neural / extended-FH energy (M6) is a drop-in.

    All methods take ``phis`` = a length-M list of arrays (Gauss-point or nodal
    values) and broadcast elementwise."""
    M = 0
    param_names = ()

    def mu(self, phis):
        raise NotImplementedError

    def dmu_dphi(self, phis):
        raise NotImplementedError

    def dmu_dparam(self, phis, name):
        raise NotImplementedError


class FHMultiEnergy(MultiEnergy):
    r"""Flory-Huggins bulk exchange potentials (K=0 branch of the production p1
    form, physics/multiphase.np_potentials):

      mu_i = Ninv_i (ln phi_i + 1) - Ninv_M (ln ps + 1) + S(i) - S(M)
      S(m) = sum_{l != m} phi_hat_l chi[m, l]   (phi_hat_M = ps = 1 - sum phi)

    Parameters (``param_names``): the upper-triangle chi pairs including the
    solvent (``chi_a_b``, a < b) and per-species N (``N_i``, i = 0..M).  chi is a
    symmetric (M+1)x(M+1) array (diagonal unused); N is length M+1."""

    def __init__(self, chi, N, breg=0.0):
        self.chi = np.asarray(chi, np.float64)
        self.N = np.asarray(N, np.float64)
        self.M = self.chi.shape[0] - 1
        self.Ninv = 1.0 / self.N
        self.breg = float(breg)
        pairs = [f"chi_{a}_{b}" for a in range(self.M + 1)
                 for b in range(a + 1, self.M + 1)]
        self.param_names = tuple(pairs) + tuple(
            f"N_{i}" for i in range(self.M + 1))

    def _ps(self, phis):
        return 1.0 - sum(phis)

    def mu(self, phis):
        M, chi, Ninv = self.M, self.chi, self.Ninv
        ps = self._ps(phis)
        phi_of = lambda l: phis[l] if l < M else ps

        def S(m):
            return sum(phi_of(l) * chi[m, l]
                       for l in range(M + 1) if l != m)

        out = []
        for i in range(M):
            mu = (Ninv[i] * (np.log(phis[i]) + 1.0)
                  - Ninv[M] * (np.log(ps) + 1.0) + S(i) - S(M))
            if self.breg:
                b2 = lambda x: 1.0 / np.maximum(x, 1e-3) ** 2
                mu = mu + self.breg * (b2(ps) - b2(phis[i]))
            out.append(mu)
        return out

    def dmu_dphi(self, phis):
        M, chi, Ninv = self.M, self.chi, self.Ninv
        ps = self._ps(phis)
        ones = np.ones_like(phis[0])
        H = [[None] * M for _ in range(M)]
        for i in range(M):
            for j in range(M):
                term = (Ninv[M] / ps + (chi[i, j] if i != j else 0.0)
                        - chi[i, M] - chi[M, j])
                if i == j:
                    term = term + Ninv[i] / phis[i]
                if self.breg:
                    db2 = lambda x: -2.0 / np.maximum(x, 1e-3) ** 3
                    # d/dphi_j of breg (b2(ps) - b2(phi_i)); dps/dphi_j = -1
                    term = term + self.breg * (-db2(ps))
                    if i == j:
                        term = term - self.breg * db2(phis[i])
                H[i][j] = term * ones if np.isscalar(term) else term
        return H

    def dmu_dparam(self, phis, name):
        M = self.M
        ps = self._ps(phis)
        phi_of = lambda l: phis[l] if l < M else ps
        ones = np.ones_like(phis[0])
        z = [np.zeros_like(phis[0]) for _ in range(M)]
        if name.startswith("chi_"):
            _, a, b = name.split("_")
            a, b = int(a), int(b)
            for i in range(M):
                val = ((phi_of(b) if i == a else 0.0)
                       + (phi_of(a) if i == b else 0.0)
                       - (phi_of(b) if M == a else 0.0)
                       - (phi_of(a) if M == b else 0.0))
                z[i] = val * ones if np.isscalar(val) else val
            return z
        if name.startswith("N_"):
            j = int(name.split("_")[1])
            Nj = self.N[j]
            if j < M:
                z[j] = (-1.0 / Nj ** 2) * (np.log(phis[j]) + 1.0)
            else:  # solvent N_M enters every mu_i through -Ninv_M(ln ps + 1)
                for i in range(M):
                    z[i] = (1.0 / Nj ** 2) * (np.log(ps) + 1.0)
            return z
        raise KeyError(name)
