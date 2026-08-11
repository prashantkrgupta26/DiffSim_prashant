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

    def stiffness_matrix(self):
        """Global scalar stiffness (Laplacian) K, K[a,b] = Int gradN_a.gradN_b.
        Used by objectives like the interfacial (gradient) energy
        J = sum_i (kappa_i/2) phi_i^T K phi_i, whose cotangent is kappa_i K phi_i."""
        rows, cols, vals = [], [], []
        for B in self.bins:
            LL = np.einsum("eq,qad,qbd->eab",
                           B["dJxW"] * B["dscale"][:, None] ** 2,
                           B["dN"], B["dN"])
            conn, nbf = B["conn"], B["nbf"]
            rows.append(np.repeat(conn, nbf, axis=1).ravel())
            cols.append(np.tile(conn, (1, nbf)).ravel())
            vals.append(LL.ravel())
        return sp.coo_matrix(
            (np.concatenate(vals),
             (np.concatenate(rows), np.concatenate(cols))),
            shape=(self.nn, self.nn)).tocsr()

    def assemble(self, phis, mus, hist_gp, params, want_jac=True):
        """R (len ndof) and, if want_jac, J = dR/dx (csr ndof x ndof).
        phis/mus are length-M lists of length-nn arrays; hist_gp[i][bi] is the
        [ne, nqp] BDF history load for species i (sum_k (ch_k/dt) interp).

        params["mobility"] must be a MobilityClosure (or legacy "onsager" key
        for back-compat).  The transport flux for species i is:
            flux_i = sum_j M_ij(phi) * grad(mu_j)
        For phi-dependent closures the Jacobian gains an extra term
            dRphi_i/dphi_i += int gradN . ((dM_ii/dphi_i) * grad(mu_i))
        (diagonal only for the phi_diag closure; off-diagonal M==0 there)."""
        M, blk = self.M, self.blk
        # --- mobility: prefer closure; fall back to legacy onsager key -------
        mobility = params.get("mobility")
        if mobility is None:
            # legacy path: params["onsager"] is a plain matrix
            from .neural_multiphase import MobilityClosure as _MC
            mobility = _MC("const", M=M,
                           onsager=np.asarray(params["onsager"]))
        kap = params["kappa"]
        energy = params["energy"]
        sigma = params["sigma"]
        pi = [self.interp(phis[i]) for i in range(M)]
        mi = [self.interp(mus[i]) for i in range(M)]
        # promote R dtype so complex-step verification (imag perturbations in
        # phi/mu, chi/N, mobility/kappa) flows through the scatter unclipped.
        dsrc = [np.asarray(phis[i]).dtype for i in range(M)]
        dsrc += [np.asarray(mus[i]).dtype for i in range(M)]
        dsrc += [np.asarray(kap).dtype]
        if mobility.name == "const" and mobility.onsager is not None:
            dsrc.append(np.asarray(mobility.onsager).dtype)
        for attr in ("chi", "N"):
            if hasattr(energy, attr):
                dsrc.append(np.asarray(getattr(energy, attr)).dtype)
        # also pick up complex dtype from closure coefficients (phi_diag)
        for v in mobility.coeffs.values():
            dsrc.append(np.asarray(v).dtype)
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
            # evaluate mobility matrix at Gauss points: M×M list-of-lists of
            # [ne, nqp] arrays
            Lam = mobility.matrix(phi_gp)
            for i in range(M):
                # R_phi_i = Int N(sigma phi_i - hist_i) + Int gradN . flux_i
                flux = np.zeros_like(gmu[0])
                for j in range(M):
                    flux = flux + Lam[i][j][:, :, None] * gmu[j]
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
                    # dRphi_i/dmu_j: Int gradN_a . (M_ij(phi) grad N_b_mu_j) dOmega
                    # M_ij is [ne,nqp]: must weight BEFORE integrating over q.
                    # Lij_LL[e,a,b] = Sum_q dJxW[e,q]*dscale[e]^2 * Lam_ij[e,q]
                    #                  * Sum_d dN[q,a,d]*dN[q,b,d]
                    Lij_LL = np.einsum("eq,e,qad,qbd->eab",
                                       dJxW * Lam[i][j], dscale ** 2, dN, dN)
                    Ae[:, 2 * i::blk, 2 * j + 1::blk] += Lij_LL
                    FPP = np.einsum("eq,qa,qb->eab", dJxW * H[i][j], N, N)
                    Ae[:, 2 * i + 1::blk, 2 * j::blk] += -FPP       # dRmu/dphi
                # phi-dependent mobility: extra Jacobian term dRphi_i/dphi_i
                # = Int gradN . ((dM_ii/dphi_i) * grad(mu_i))
                # For const closure dM/dphi = 0; for phi_diag dM_ii/dphi_i != 0
                # dM_ii/dphi_i (scalar field [ne, nqp])
                dMii_dphii = mobility._dmatrix_dphi_diag(phi_gp, i)
                if dMii_dphii is not None:
                    # flux contribution: dM_ii/dphi_i * grad(mu_i)
                    dflux = dMii_dphii[:, :, None] * gmu[i]
                    # integrate: Int gradN . dflux * N_a  (N_a is phi_i test fn)
                    # = einsum over (e, q, a=phi-row, b=phi-col)
                    # result shape (e, nbf_a, nbf_b) -> scatter to Ae[2i,2i]
                    dM_term = np.einsum(
                        "eq,eqd,qad,e,qb->eab",
                        dJxW, dflux, dN, dscale, N)
                    Ae[:, 2 * i::blk, 2 * i::blk] += dM_term
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
        params route through the energy object; mobility/kappa are engine-level.

        Handles:
          onsager_a_b   — legacy constant onsager entry (const closure)
          mob_*         — mobility closure parameter (phi_diag or other closures)
          kappa_i       — interface energy coefficient
          others        — delegated to energy.dmu_dparam
        """
        M, blk = self.M, self.blk
        energy = params["energy"]
        # retrieve mobility closure (prefer params["mobility"]; fall back to legacy)
        mobility = params.get("mobility")
        if mobility is None and "onsager" in params:
            from .neural_multiphase import MobilityClosure as _MC
            mobility = _MC("const", M=M,
                           onsager=np.asarray(params["onsager"]))
        pi = [self.interp(phis[i]) for i in range(M)]
        mi = [self.interp(mus[i]) for i in range(M)]
        out = np.zeros(self.ndof)
        for bi, B in enumerate(self.bins):
            dJxW, N, dN, dscale = B["dJxW"], B["N"], B["dN"], B["dscale"]
            phi_gp = [pi[i][bi][0] for i in range(M)]
            if name.startswith("onsager_"):
                # legacy: const closure, Lam[a,b] is a scalar that multiplies
                # grad(mu_b) in flux for species a -> dR_phi_a / d(onsager_a_b)
                _, a, b = name.split("_")
                a, b = int(a), int(b)
                gmu_b = mi[b][bi][1]
                gN = np.einsum("qad,e,eqd->eqa", dN, dscale, gmu_b)
                Rphi = np.einsum("eq,eqa->ea", dJxW, gN)
                np.add.at(out, B["gdof"][:, 2 * a::blk].ravel(), Rphi.ravel())
            elif name.startswith("mob_"):
                # mobility closure parameter: dR_phi_i/d(param) =
                # Int gradN . ((dM_ij/d(param)) * grad(mu_j)) for each i,j
                dLam = mobility.dmatrix_dparam(phi_gp, name)  # M×M list-of-lists
                for i in range(M):
                    contrib = np.zeros_like(mi[0][bi][1])  # [ne, nqp, dim]
                    for j in range(M):
                        dLij = dLam[i][j]  # [ne, nqp]
                        contrib = contrib + dLij[:, :, None] * mi[j][bi][1]
                    gN = np.einsum("qad,e,eqd->eqa", dN, dscale, contrib)
                    Rphi = np.einsum("eq,eqa->ea", dJxW, gN)
                    np.add.at(out, B["gdof"][:, 2 * i::blk].ravel(),
                              Rphi.ravel())
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


# --------------------------------------------------------------------------
# forward march (numpy) + per-step recorder for the adjoint
# --------------------------------------------------------------------------
class MultiCHForward:
    """Newton BDF march of the discrete multi-CH operator, recording per-step
    the converged state and BDF coefficients the adjoint replays.  M-generic
    sibling of phasefield.CHForward."""

    def __init__(self, dm, energy, onsager=None, kappa=None, dt=1e-2, order=1,
                 newton_tol=1e-12, newton_max=30, backend=None,
                 mobility=None):
        """Newton BDF march.  ``mobility`` may be a MobilityClosure or a plain
        matrix/array (auto-wrapped as const closure).  ``onsager`` is the legacy
        keyword; if both are provided, ``mobility`` takes precedence."""
        from .linsolve_backend import ScipyBackend
        from .neural_multiphase import MobilityClosure
        self.op = MultiCHDiscrete(dm, energy.M)
        self.M = energy.M
        self.energy = energy
        # --- resolve mobility ---
        # Precedence: explicit mobility= > onsager= keyword
        if mobility is not None:
            if isinstance(mobility, MobilityClosure):
                self.mobility = mobility
            else:
                # plain matrix / array
                self.mobility = MobilityClosure(
                    "const", M=self.M,
                    onsager=np.asarray(mobility, np.float64))
        elif onsager is not None:
            self.mobility = MobilityClosure(
                "const", M=self.M,
                onsager=np.asarray(onsager, np.float64))
        else:
            raise ValueError("MultiCHForward: supply mobility= or onsager=")
        # legacy attribute (backward-compat for callers that read fwd.onsager)
        if self.mobility.name == "const":
            self.onsager = self.mobility.onsager
        self.kappa = [float(k) for k in (kappa or [])]
        self.dt = float(dt)
        self.order = order
        self.newton_tol, self.newton_max = newton_tol, newton_max
        self.backend = backend or ScipyBackend()
        self.t = 0.0
        self.dt_prev = None
        self.steps = []

    def set_initial(self, phi0_list):
        self.phis = [np.asarray(p, np.float64).copy() for p in phi0_list]
        self.mus = [np.zeros_like(self.phis[0]) for _ in range(self.M)]
        snap = [p.copy() for p in self.phis]
        self.hist = [snap, [p.copy() for p in snap]]
        self.t = 0.0
        self.dt_prev = None
        self.steps = []

    def _bdf(self):
        if self.order == 1 or self.dt_prev is None or self.t < self.dt / 2:
            return 1.0, [1.0]
        rr = self.dt / self.dt_prev
        return ((1.0 + 2.0 * rr) / (1.0 + rr),
                [1.0 + rr, -rr * rr / (1.0 + rr)])

    def _hist_gp(self, ch):
        hist_gp = [None] * self.M
        for i in range(self.M):
            acc = None
            for k, cc in enumerate(ch):
                hi = self.op.interp(self.hist[k][i])
                contrib = [(cc / self.dt) * hi[bi][0]
                           for bi in range(len(hi))]
                if acc is None:
                    acc = contrib
                else:
                    acc = [acc[bi] + contrib[bi] for bi in range(len(hi))]
            hist_gp[i] = acc
        return hist_gp

    def _params(self, sigma):
        return dict(mobility=self.mobility, kappa=self.kappa,
                    energy=self.energy, sigma=sigma)

    def step(self, record=True):
        c0_, ch = self._bdf()
        sigma = c0_ / self.dt
        hist_gp = self._hist_gp(ch)
        params = self._params(sigma)
        blk = self.op.blk
        phis = [p.copy() for p in self.phis]
        mus = [m.copy() for m in self.mus]
        for it in range(self.newton_max):
            R, J = self.op.assemble(phis, mus, hist_gp, params, want_jac=True)
            dx = self.backend.solve(J, -R)
            for i in range(self.M):
                phis[i] = phis[i] + dx[2 * i::blk]
                mus[i] = mus[i] + dx[2 * i + 1::blk]
            if np.abs(dx).max() < self.newton_tol:
                break
        self.phis, self.mus = phis, mus
        if record:
            x = np.zeros(self.op.ndof)
            for i in range(self.M):
                x[2 * i::blk] = phis[i]
                x[2 * i + 1::blk] = mus[i]
            self.steps.append(dict(
                phis=[p.copy() for p in phis], mus=[m.copy() for m in mus],
                x=x, sigma=sigma, ch=list(ch), dt=self.dt,
                params=self._params(sigma)))
        self.hist = [[p.copy() for p in phis], self.hist[0]]
        self.t += self.dt
        self.dt_prev = self.dt
        return [p.copy() for p in phis], [m.copy() for m in mus]

    def run(self, n_steps):
        return [self.step() for _ in range(n_steps)]


# --------------------------------------------------------------------------
# IFT reverse-sweep adjoint
# --------------------------------------------------------------------------
class MultiCHAdjoint:
    """Reverse-sweep dJ/d{chi, N, onsager, kappa} for J = sum_n j(x_n).
    M-generic sibling of phasefield.CHAdjoint: the BDF history cotangent
    couples ALL M conserved phi-fields."""

    def __init__(self, fwd: MultiCHForward):
        self.fwd = fwd
        self.op = fwd.op

    def gradient(self, dJdx_list, param_names):
        """dJdx_list[n] = dj/dx_n as a length-ndof node-major vector.
        Returns {name: dJ/dname}.  Residual params (chi/N/onsager/kappa) use
        the reverse-sweep dR/dp; names of the form 'phi0_i' return the
        mean-composition (initial-condition) gradient dJ/dphi_mean_i, formed
        from the history cotangent that lands on the initial state x0."""
        op = self.op
        blk, M = op.blk, op.M
        steps = self.fwd.steps
        Ns = len(steps)
        Mass = op.mass_matrix()
        res_names = [nm for nm in param_names if not nm.startswith("phi0_")]
        phi0_names = [nm for nm in param_names if nm.startswith("phi0_")]
        grads = {nm: 0.0 for nm in param_names}
        pending = [np.zeros(op.ndof) for _ in range(Ns)]
        # cotangent that accumulates onto the initial condition x0 (the history
        # terms whose target step index kn < 0). Per retained phi-species.
        phi0_cot = [np.zeros(op.nn) for _ in range(M)] if phi0_names else None
        # J = dR/dx is independent of the BDF history load, so zero history
        # rebuilds the Jacobian at the converged iterate.
        zero_hist = [[np.zeros_like(B["dJxW"]) for B in op.bins]
                     for _ in range(M)]
        for n in range(Ns - 1, -1, -1):
            rec = steps[n]
            _, J = op.assemble(rec["phis"], rec["mus"], zero_hist,
                               rec["params"], want_jac=True)
            rhs = np.asarray(dJdx_list[n], np.float64) + pending[n]
            lam = self.fwd.backend.solve_T(J, rhs)
            for nm in res_names:
                dRdp = op.dR_dparam(rec["phis"], rec["mus"], rec["params"], nm)
                grads[nm] -= float(lam @ dRdp)
            # history cotangent: R_n depends on phi_i,{n-k} only through
            # -(ch_k/dt) Mass on the phi_i-block -> +(ch_k/dt) Mass lam_phi_i
            # onto the phi_i-columns of the earlier step (kn>=0), or onto the
            # initial-condition accumulator (kn<0 -> x0).
            ch, dt = rec["ch"], rec["dt"]
            for k, cc in enumerate(ch):
                kn = n - (k + 1)
                if kn < 0 and phi0_cot is None:
                    continue          # residual-only caller: exact old fast path
                for i in range(M):
                    hc = (cc / dt) * (Mass @ lam[2 * i::blk])
                    if kn < 0:
                        phi0_cot[i] += hc
                    else:
                        pending[kn][2 * i::blk] += hc
        # dJ/dphi_mean_i = sum_nodes (dJ/dx0)_{phi_i}, since a uniform mean
        # shift adds 1 to every nodal value of the initial phi_i field.
        for nm in phi0_names:
            i = int(nm.split("_")[1])
            grads[nm] = float(phi0_cot[i].sum())
        return grads
