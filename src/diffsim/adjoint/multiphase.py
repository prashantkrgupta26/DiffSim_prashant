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
        # preserve input dtype (complex-step verification passes complex chi/N)
        self.chi = np.asarray(chi)
        self.N = np.asarray(N)
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


# --------------------------------------------------------------------------
# discrete mesh operator (numpy mirror of the production MultiPhaseStepper K=0)
# --------------------------------------------------------------------------
class MultiCHDiscrete:
    """Vectorised numpy multi-CH residual/Jacobian on a DeviceMesh, node-major
    2M block.  Bit-consistent with physics/multiphase.MultiPhaseStepper at K=0,
    bulk='p1', mob='const' (asserted by test_forward_parity_*)."""

    def __init__(self, dm, M):
        T = dm.constraints.T
        assert T.shape[0] == T.shape[1] and (
            abs(T - sp.eye(T.shape[0])).nnz == 0), \
            "adjoint gate assumes constraints.T == identity (uniform mesh)"
        self.dm = dm
        self.M = int(M)
        self.blk = 2 * self.M
        self.dim = dm.dim
        self.nn = dm.n_nodes
        self.ndof = self.blk * self.nn
        self.bins = []
        for pv, b in dm.bins.items():
            tb = dm.tables_by_p[pv]
            conn = dm.mesh.conn_of[pv].astype(np.int64)
            h = np.asarray(dm.mesh.tree.h()[dm.mesh.bins[pv]], np.float64)
            N = np.asarray(tb.N, np.float64)               # [nqp, nbf]
            dN = np.asarray(tb.dN, np.float64)             # [nqp, nbf, dim]
            w = np.asarray(tb.w, np.float64)               # [nqp]
            ne, nbf = conn.shape
            dscale = 2.0 / h
            jac = (h / 2.0) ** self.dim
            dJxW = w[None, :] * jac[:, None]               # [ne, nqp]
            gdof = (conn[:, :, None] * self.blk
                    + np.arange(self.blk)[None, None, :]).reshape(
                        ne, self.blk * nbf)
            self.bins.append(dict(conn=conn, N=N, dN=dN, ne=ne, nbf=nbf,
                                  nqp=N.shape[0], dscale=dscale, dJxW=dJxW,
                                  gdof=gdof))
        self._mass = None

    def interp(self, field):
        out = []
        for B in self.bins:
            vals = field[B["conn"]]
            v = np.einsum("qa,ea->eq", B["N"], vals)
            g = np.einsum("qad,ea,e->eqd", B["dN"], vals, B["dscale"])
            out.append((v, g))
        return out

    def mass_matrix(self):
        if self._mass is not None:
            return self._mass
        rows, cols, vals = [], [], []
        for B in self.bins:
            NN = np.einsum("eq,qa,qb->eab", B["dJxW"], B["N"], B["N"])
            conn, nbf = B["conn"], B["nbf"]
            rows.append(np.repeat(conn, nbf, axis=1).ravel())
            cols.append(np.tile(conn, (1, nbf)).ravel())
            vals.append(NN.ravel())
        self._mass = sp.coo_matrix(
            (np.concatenate(vals),
             (np.concatenate(rows), np.concatenate(cols))),
            shape=(self.nn, self.nn)).tocsr()
        return self._mass

    def assemble(self, phis, mus, hist_gp, params, want_jac=True):
        """R (len ndof) and, if want_jac, J = dR/dx (csr ndof x ndof).
        phis/mus are length-M lists of length-nn arrays; hist_gp[i][bi] is the
        [ne, nqp] BDF history load for species i (sum_k (ch_k/dt) interp)."""
        M, blk = self.M, self.blk
        Lam = np.asarray(params["onsager"])   # dtype preserved (complex-step)
        kap = params["kappa"]
        energy = params["energy"]
        sigma = params["sigma"]
        pi = [self.interp(phis[i]) for i in range(M)]
        mi = [self.interp(mus[i]) for i in range(M)]
        # promote R dtype so complex-step verification (imag perturbations in
        # phi/mu, chi/N, onsager/kappa) flows through the scatter unclipped.
        dsrc = [np.asarray(phis[i]).dtype for i in range(M)]
        dsrc += [np.asarray(mus[i]).dtype for i in range(M)]
        dsrc += [np.asarray(Lam).dtype, np.asarray(kap).dtype]
        for attr in ("chi", "N"):
            if hasattr(energy, attr):
                dsrc.append(np.asarray(getattr(energy, attr)).dtype)
        R = np.zeros(self.ndof, dtype=np.result_type(*dsrc, np.float64))
        rows, cols, vals = [], [], []
        for bi, B in enumerate(self.bins):
            dJxW, N, dN, dscale = B["dJxW"], B["N"], B["dN"], B["dscale"]
            phi_gp = [pi[i][bi][0] for i in range(M)]
            gphi = [pi[i][bi][1] for i in range(M)]
            mu_gp = [mi[i][bi][0] for i in range(M)]
            gmu = [mi[i][bi][1] for i in range(M)]
            muref = energy.mu(phi_gp)
            NN = np.einsum("eq,qa,qb->eab", dJxW, N, N)
            LL = np.einsum("eq,qad,qbd->eab",
                           dJxW * dscale[:, None] ** 2, dN, dN)
            ne, nbf = B["ne"], B["nbf"]
            Ae = np.zeros((ne, blk * nbf, blk * nbf)) if want_jac else None
            H = energy.dmu_dphi(phi_gp) if want_jac else None
            for i in range(M):
                # R_phi_i = Int N(sigma phi_i - hist_i) + Int gradN . flux_i
                flux = np.zeros_like(gmu[0])
                for j in range(M):
                    flux = flux + Lam[i, j] * gmu[j]
                gN_flux = np.einsum("qad,e,eqd->eqa", dN, dscale, flux)
                Rphi = (np.einsum("eq,qa->ea",
                                  dJxW * (sigma * phi_gp[i] - hist_gp[i][bi]),
                                  N)
                        + np.einsum("eq,eqa->ea", dJxW, gN_flux))
                np.add.at(R, B["gdof"][:, 2 * i::blk].ravel(), Rphi.ravel())
                # R_mu_i = Int N(mu_i - muref_i) - kap_i Int gradN . grad phi_i
                gN_gphi = np.einsum("qad,e,eqd->eqa", dN, dscale, gphi[i])
                Rmu = (np.einsum("eq,qa->ea", dJxW * (mu_gp[i] - muref[i]), N)
                       - kap[i] * np.einsum("eq,eqa->ea", dJxW, gN_gphi))
                np.add.at(R, B["gdof"][:, 2 * i + 1::blk].ravel(), Rmu.ravel())
                if not want_jac:
                    continue
                Ae[:, 2 * i::blk, 2 * i::blk] += sigma * NN         # dRphi/dphi
                for j in range(M):
                    Ae[:, 2 * i::blk, 2 * j + 1::blk] += Lam[i, j] * LL
                    FPP = np.einsum("eq,qa,qb->eab", dJxW * H[i][j], N, N)
                    Ae[:, 2 * i + 1::blk, 2 * j::blk] += -FPP       # dRmu/dphi
                Ae[:, 2 * i + 1::blk, 2 * i::blk] += -kap[i] * LL   # +kap term
                Ae[:, 2 * i + 1::blk, 2 * i + 1::blk] += NN         # dRmu/dmu
            if want_jac:
                gdof = B["gdof"]
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

    def dR_dparam(self, phis, mus, params, name):
        """Explicit dR/dp as a length-ndof vector at a committed state.  Bulk
        params route through the energy object; onsager/kappa are engine-level."""
        M, blk = self.M, self.blk
        energy = params["energy"]
        pi = [self.interp(phis[i]) for i in range(M)]
        mi = [self.interp(mus[i]) for i in range(M)]
        out = np.zeros(self.ndof)
        for bi, B in enumerate(self.bins):
            dJxW, N, dN, dscale = B["dJxW"], B["N"], B["dN"], B["dscale"]
            phi_gp = [pi[i][bi][0] for i in range(M)]
            if name.startswith("onsager_"):
                _, a, b = name.split("_")
                a, b = int(a), int(b)
                gmu_b = mi[b][bi][1]
                gN = np.einsum("qad,e,eqd->eqa", dN, dscale, gmu_b)
                Rphi = np.einsum("eq,eqa->ea", dJxW, gN)
                np.add.at(out, B["gdof"][:, 2 * a::blk].ravel(), Rphi.ravel())
            elif name.startswith("kappa_"):
                i = int(name.split("_")[1])
                gN = np.einsum("qad,e,eqd->eqa", dN, dscale, pi[i][bi][1])
                Rmu = -np.einsum("eq,eqa->ea", dJxW, gN)
                np.add.at(out, B["gdof"][:, 2 * i + 1::blk].ravel(),
                          Rmu.ravel())
            else:  # bulk-energy coefficient via the energy object
                dmu = energy.dmu_dparam(phi_gp, name)
                for i in range(M):
                    Rmu = -np.einsum("eq,qa->ea", dJxW * dmu[i], N)
                    np.add.at(out, B["gdof"][:, 2 * i + 1::blk].ravel(),
                              Rmu.ravel())
        return out
