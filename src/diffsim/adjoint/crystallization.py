r"""G2 track (a): differentiable coupled Cahn-Hilliard x Allen-Cahn
crystallization (M=1, K=1) — the crystallization material parameters
(dh, Tm, dsigma, eps2, L) made differentiable and three-way verified.

FIELDS (node-major 3-dof block: 3a+0 = phi, 3a+1 = mu, 3a+2 = psi):
  phi  conserved volume fraction (CH pair with chemical potential mu),
  psi  non-conserved crystallinity in [0, 1] (Allen-Cahn).
This is the retained-species core of physics/multiphase.py at (M, K) = (1, 1);
orientation theta (the 4th dof of the full 2M+2K block) is deferred (documented
frontier — it carries a KG-regularised |grad theta| that needs its own gate).

FREE ENERGY (representative crystallisation form, r14/Turnbull family
2310.11844 Eqs. 3+7 at K=M=1):
  f(phi, psi) = f_FH(phi) + phi [ q(psi) dsig + p(psi) drive ]
              + (kap/2)|grad phi|^2 + (eps2/2)|grad psi|^2
  q(psi) = psi^2 (1-psi)^2         (double-well barrier, height dsig)
  p(psi) = psi^2 (3 - 2 psi)       (monotone 0->1 interpolation)
  drive  = dh (T/Tm - 1)           (Turnbull driving; dh = latent heat > 0,
                                    drive < 0 below Tm so p(psi)->1 lowers f)

DYNAMICS (natural no-flux BCs):
  CH:  dphi/dt = div(M grad mu),  mu = df/dphi - kap lap phi
       df/dphi = f_FH'(phi) + q(psi) dsig + p(psi) drive
  AC:  dpsi/dt = -L [ df/dpsi - eps2 lap psi ]
       df/dpsi = phi [ q'(psi) dsig + p'(psi) drive ]

WEAK RESIDUAL (BDF: phi_t -> sigma phi - hist_phi, likewise psi):
  R_phi = Int N(sigma phi - hist_phi) + M Int gradN.grad mu
  R_mu  = Int N(mu - df/dphi)         - kap Int gradN.grad phi
  R_psi = Int N(sigma psi - hist_psi) + L[ Int N df/dpsi
                                           + eps2 Int gradN.grad psi ]

The IFT adjoint (CACHAdjoint) is identical in structure to phasefield.CHAdjoint
but the history cotangent now couples BOTH conserved-in-time fields (phi at
block-slot 0 AND psi at slot 2)."""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu

from .phasefield import FHEnergy, PolyEnergy


def _q(psi):
    return psi ** 2 * (1.0 - psi) ** 2


def _qp(psi):
    return 2.0 * psi - 6.0 * psi ** 2 + 4.0 * psi ** 3


def _qpp(psi):
    return 2.0 - 12.0 * psi + 12.0 * psi ** 2


def _p(psi):
    return 3.0 * psi ** 2 - 2.0 * psi ** 3


def _pp(psi):
    return 6.0 * psi - 6.0 * psi ** 2


def _ppp(psi):
    return 6.0 - 12.0 * psi


class CACHDiscrete:
    """Vectorised numpy residual/Jacobian for the coupled CH x AC system."""

    NF = 3   # dofs per node (phi, mu, psi)

    def __init__(self, dm):
        T = dm.constraints.T
        assert T.shape[0] == T.shape[1] and (abs(T - sp.eye(T.shape[0])).nnz
                                             == 0), "uniform mesh only"
        self.dm = dm
        self.dim = dm.dim
        self.nn = dm.n_nodes
        self.ndof = self.NF * self.nn
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
            gdof = (conn[:, :, None] * self.NF
                    + np.arange(self.NF)[None, None, :]).reshape(
                        ne, self.NF * nbf)
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

    @staticmethod
    def drive(params):
        return params["dh"] * (params["T"] / params["Tm"] - 1.0)

    def _fields(self, x):
        return self.interp(x[0::self.NF]), self.interp(x[1::self.NF]), \
            self.interp(x[2::self.NF])

    def assemble(self, x, hist_phi_gp, hist_psi_gp, params, want_jac=True):
        M, kap = params["M"], params["kappa"]
        eps2, L = params["eps2"], params["L"]
        dsig = params["dsig"]
        drv = self.drive(params)
        sigma = params["sigma"]
        energy = params["energy"]
        pi, mi, si = self._fields(x)
        R = np.zeros(self.ndof)
        rows, cols, vals = [], [], []
        NF = self.NF
        for bi, B in enumerate(self.bins):
            phi_, gphi = pi[bi]
            mu_, gmu = mi[bi]
            psi_, gpsi = si[bi]
            dJxW, N, dN, dscale = B["dJxW"], B["N"], B["dN"], B["dscale"]
            hphi = hist_phi_gp[bi]
            hpsi = hist_psi_gp[bi]
            q, qp, qpp = _q(psi_), _qp(psi_), _qpp(psi_)
            p, pp, ppp = _p(psi_), _pp(psi_), _ppp(psi_)
            fFHp = energy.fp(phi_)
            fFHpp = energy.fpp(phi_)
            dfdphi = fFHp + q * dsig + p * drv
            dfdpsi = phi_ * (qp * dsig + pp * drv)
            gN_gmu = np.einsum("qad,e,eqd->eqa", dN, dscale, gmu)
            gN_gphi = np.einsum("qad,e,eqd->eqa", dN, dscale, gphi)
            gN_gpsi = np.einsum("qad,e,eqd->eqa", dN, dscale, gpsi)
            Rphi = np.einsum("eq,qa->ea", dJxW * (sigma * phi_ - hphi), N) \
                + M * np.einsum("eq,eqa->ea", dJxW, gN_gmu)
            Rmu = np.einsum("eq,qa->ea", dJxW * (mu_ - dfdphi), N) \
                - kap * np.einsum("eq,eqa->ea", dJxW, gN_gphi)
            Rpsi = np.einsum("eq,qa->ea", dJxW * (sigma * psi_ - hpsi), N) \
                + L * (np.einsum("eq,qa->ea", dJxW * dfdpsi, N)
                       + eps2 * np.einsum("eq,eqa->ea", dJxW, gN_gpsi))
            gdof = B["gdof"]
            np.add.at(R, gdof[:, 0::NF].ravel(), Rphi.ravel())
            np.add.at(R, gdof[:, 1::NF].ravel(), Rmu.ravel())
            np.add.at(R, gdof[:, 2::NF].ravel(), Rpsi.ravel())
            if not want_jac:
                continue
            NN = np.einsum("eq,qa,qb->eab", dJxW, N, N)
            LL = np.einsum("eq,qad,qbd->eab", dJxW * dscale[:, None] ** 2,
                           dN, dN)
            WM = lambda wgt: np.einsum("eq,qa,qb->eab", dJxW * wgt, N, N)
            coup = qp * dsig + pp * drv          # d(df/dphi)/dpsi = d(df/dpsi)/dphi
            nbf = B["nbf"]
            Ae = np.zeros((B["ne"], NF * nbf, NF * nbf))
            # phi rows
            Ae[:, 0::NF, 0::NF] = sigma * NN            # Rphi/dphi
            Ae[:, 0::NF, 1::NF] = M * LL                # Rphi/dmu
            # mu rows
            Ae[:, 1::NF, 0::NF] = -WM(fFHpp) - kap * LL  # Rmu/dphi
            Ae[:, 1::NF, 1::NF] = NN                     # Rmu/dmu
            Ae[:, 1::NF, 2::NF] = -WM(coup)              # Rmu/dpsi
            # psi rows
            Ae[:, 2::NF, 0::NF] = L * WM(coup)           # Rpsi/dphi
            Ae[:, 2::NF, 2::NF] = (sigma * NN
                                   + L * (WM(phi_ * (qpp * dsig + ppp * drv))
                                          + eps2 * LL))  # Rpsi/dpsi
            rows.append(np.repeat(gdof, NF * nbf, axis=1).ravel())
            cols.append(np.tile(gdof, (1, NF * nbf)).ravel())
            vals.append(Ae.ravel())
        J = None
        if want_jac:
            J = sp.coo_matrix(
                (np.concatenate(vals),
                 (np.concatenate(rows), np.concatenate(cols))),
                shape=(self.ndof, self.ndof)).tocsr()
        return R, J

    def dR_dparam(self, x, params, name):
        """Analytic explicit dR/dp (length-ndof node-major vector)."""
        M, kap = params["M"], params["kappa"]
        eps2, L, dsig = params["eps2"], params["L"], params["dsig"]
        drv = self.drive(params)
        energy = params["energy"]
        pi, mi, si = self._fields(x)
        out = np.zeros(self.ndof)
        NF = self.NF
        for bi, B in enumerate(self.bins):
            phi_, gphi = pi[bi]
            mu_, gmu = mi[bi]
            psi_, gpsi = si[bi]
            dJxW, N, dN, dscale = B["dJxW"], B["N"], B["dN"], B["dscale"]
            gdof = B["gdof"]
            q, qp = _q(psi_), _qp(psi_)
            p, pp = _p(psi_), _pp(psi_)
            addmu = lambda arr: np.add.at(out, gdof[:, 1::NF].ravel(),
                                          arr.ravel())
            addpsi = lambda arr: np.add.at(out, gdof[:, 2::NF].ravel(),
                                           arr.ravel())
            addphi = lambda arr: np.add.at(out, gdof[:, 0::NF].ravel(),
                                           arr.ravel())
            if name == "M":
                gN_gmu = np.einsum("qad,e,eqd->eqa", dN, dscale, gmu)
                addphi(np.einsum("eq,eqa->ea", dJxW, gN_gmu))
            elif name == "kappa":
                gN_gphi = np.einsum("qad,e,eqd->eqa", dN, dscale, gphi)
                addmu(-np.einsum("eq,eqa->ea", dJxW, gN_gphi))
            elif name == "eps2":
                gN_gpsi = np.einsum("qad,e,eqd->eqa", dN, dscale, gpsi)
                addpsi(L * np.einsum("eq,eqa->ea", dJxW, gN_gpsi))
            elif name == "L":
                dfdpsi = phi_ * (qp * dsig + pp * drv)
                gN_gpsi = np.einsum("qad,e,eqd->eqa", dN, dscale, gpsi)
                addpsi(np.einsum("eq,qa->ea", dJxW * dfdpsi, N)
                       + eps2 * np.einsum("eq,eqa->ea", dJxW, gN_gpsi))
            elif name == "dsig":
                addmu(-np.einsum("eq,qa->ea", dJxW * q, N))     # -Int N q
                addpsi(L * np.einsum("eq,qa->ea", dJxW * (phi_ * qp), N))
            elif name in ("dh", "Tm", "T"):
                # drive derivative
                if name == "dh":
                    dd = params["T"] / params["Tm"] - 1.0
                elif name == "Tm":
                    dd = -params["dh"] * params["T"] / params["Tm"] ** 2
                else:  # T
                    dd = params["dh"] / params["Tm"]
                addmu(-np.einsum("eq,qa->ea", dJxW * (p * dd), N))
                addpsi(L * np.einsum("eq,qa->ea", dJxW * (phi_ * pp * dd), N))
            else:
                # FH bulk coeff via df/dphi = f_FH'(phi): Rmu has -Int N f'
                dfp = energy.dfp_dparam(phi_, name)
                addmu(-np.einsum("eq,qa->ea", dJxW * dfp, N))
        return out


class CACHForward:
    def __init__(self, dm, energy, M=1.0, kappa=1e-2, eps2=1e-2, L=1.0,
                 dsig=1.0, dh=1.0, Tm=1.0, T=0.5, dt=1e-2, order=1,
                 newton_tol=1e-12, newton_max=40):
        self.op = CACHDiscrete(dm)
        self.energy = energy
        self.M, self.kappa, self.eps2, self.L = map(
            float, (M, kappa, eps2, L))
        self.dsig, self.dh, self.Tm, self.T = map(float, (dsig, dh, Tm, T))
        self.dt, self.order = float(dt), order
        self.newton_tol, self.newton_max = newton_tol, newton_max
        self.t, self.dt_prev = 0.0, None
        self.steps = []
        # G3 quench schedule: per-step temperature control T_n entering
        # through BOTH the crystallisation drive dh(T/Tm-1) AND the Flory
        # chi B(T) = B0 + bT*(T - Tref).  bT=0 disables the chi channel.
        self.T_schedule = None
        self.bT, self.Tref = 0.0, 0.0
        self._B0 = getattr(energy, "B", None)
        self._A0 = getattr(energy, "A", None)
        self._istep = 0

    def set_schedule(self, T_list, bT=0.0, Tref=0.0):
        """Drive the march from a temperature time series T_list (len =
        n_steps); bT couples T into the Flory chi."""
        self.T_schedule = [float(v) for v in T_list]
        self.bT, self.Tref = float(bT), float(Tref)
        return self

    def _Tn(self):
        if self.T_schedule is None:
            return self.T
        return self.T_schedule[self._istep]

    def set_initial(self, phi0, psi0):
        NF = self.op.NF
        self.x = np.zeros(self.op.ndof)
        self.x[0::NF] = phi0
        self.x[2::NF] = psi0
        self.hist_phi = [phi0.copy(), phi0.copy()]
        self.hist_psi = [psi0.copy(), psi0.copy()]
        self.t, self.dt_prev, self.steps = 0.0, None, []
        self._istep = 0

    def _bdf(self):
        if self.order == 1 or self.dt_prev is None or self.t < self.dt / 2:
            return 1.0, [1.0]
        rr = self.dt / self.dt_prev
        return (1.0 + 2.0 * rr) / (1.0 + rr), [1.0 + rr, -rr * rr / (1.0 + rr)]

    def _hist_gp(self, hist_list, ch):
        hg = None
        for k, cc in enumerate(ch):
            hi = self.op.interp(hist_list[k])
            if hg is None:
                hg = [(cc / self.dt) * hi[bi][0] for bi in range(len(hi))]
            else:
                for bi in range(len(hi)):
                    hg[bi] = hg[bi] + (cc / self.dt) * hi[bi][0]
        return hg

    def _params(self, sigma):
        Tn = self._Tn()
        energy = self.energy
        if self.bT != 0.0 and self._B0 is not None:
            energy = FHEnergy(self._A0, self._B0 + self.bT * (Tn - self.Tref))
        return dict(M=self.M, kappa=self.kappa, eps2=self.eps2, L=self.L,
                    dsig=self.dsig, dh=self.dh, Tm=self.Tm, T=Tn,
                    energy=energy, sigma=sigma, bT=self.bT)

    def step(self, record=True):
        c0_, ch = self._bdf()
        sigma = c0_ / self.dt
        hpg = self._hist_gp(self.hist_phi, ch)
        hsg = self._hist_gp(self.hist_psi, ch)
        params = self._params(sigma)
        x = self.x.copy()
        NF = self.op.NF
        for it in range(self.newton_max):
            R, J = self.op.assemble(x, hpg, hsg, params, want_jac=True)
            dx = splu(J.tocsc()).solve(-R)
            x = x + dx
            if np.abs(dx).max() < self.newton_tol:
                break
        self.x = x
        if record:
            self.steps.append(dict(x=x.copy(), sigma=sigma, ch=list(ch),
                                   dt=self.dt, params=params, Tn=self._Tn()))
        self.hist_phi = [x[0::NF].copy(), self.hist_phi[0]]
        self.hist_psi = [x[2::NF].copy(), self.hist_psi[0]]
        self.t += self.dt
        self.dt_prev = self.dt
        self._istep += 1
        return x.copy()

    def run(self, n):
        return [self.step() for _ in range(n)]


class CACHAdjoint:
    """Reverse-sweep dJ/dp for the coupled crystallisation system."""

    def __init__(self, fwd: CACHForward):
        self.fwd = fwd
        self.op = fwd.op

    def gradient(self, dJdx_list, param_names):
        op = self.op
        NF = op.NF
        steps = self.fwd.steps
        N = len(steps)
        Mass = op.mass_matrix()
        grads = {nm: 0.0 for nm in param_names}
        pending = [np.zeros(op.ndof) for _ in range(N)]
        zhist = [np.zeros_like(B["dJxW"]) for B in op.bins]
        for n in range(N - 1, -1, -1):
            rec = steps[n]
            _, J = op.assemble(rec["x"], zhist, zhist, rec["params"],
                               want_jac=True)
            rhs = np.asarray(dJdx_list[n], np.float64) + pending[n]
            lam = splu(J.T.tocsc()).solve(rhs)
            for nm in param_names:
                dRdp = op.dR_dparam(rec["x"], rec["params"], nm)
                grads[nm] -= float(lam @ dRdp)
            # history coupling: phi (slot 0) and psi (slot 2) both stepped
            lam_phi = lam[0::NF]
            lam_psi = lam[2::NF]
            for k, cc in enumerate(rec["ch"]):
                kn = n - (k + 1)
                if kn < 0:
                    continue
                pending[kn][0::NF] += (cc / rec["dt"]) * (Mass @ lam_phi)
                pending[kn][2::NF] += (cc / rec["dt"]) * (Mass @ lam_psi)
        return grads

    def temperature_gradient(self, dJdx_list):
        """G3 quench-schedule gradient: dJ/dT_n at each step's control point
        (a TIME SERIES).  T enters R_n through the crystallisation drive
        dh(T/Tm-1) AND (if bT != 0) the Flory chi B(T); dJ/dT_n =
        -lam_n^T (dR_n/dT|drive + bT dR_n/dB).  Returns a length-N array.
        The reverse sweep is identical to gradient() — history couples phi
        and psi — but T_n is a per-step control so no cross-step chaining of
        the T-derivative is needed (each T_n enters only step n)."""
        op = self.op
        NF = op.NF
        steps = self.fwd.steps
        N = len(steps)
        Mass = op.mass_matrix()
        gT = np.zeros(N)
        pending = [np.zeros(op.ndof) for _ in range(N)]
        zhist = [np.zeros_like(B["dJxW"]) for B in op.bins]
        for n in range(N - 1, -1, -1):
            rec = steps[n]
            _, J = op.assemble(rec["x"], zhist, zhist, rec["params"],
                               want_jac=True)
            rhs = np.asarray(dJdx_list[n], np.float64) + pending[n]
            lam = splu(J.T.tocsc()).solve(rhs)
            dRdT = op.dR_dparam(rec["x"], rec["params"], "T")
            bT = rec["params"].get("bT", 0.0)
            if bT != 0.0:
                dRdT = dRdT + bT * op.dR_dparam(rec["x"], rec["params"], "B")
            gT[n] = -float(lam @ dRdT)
            lam_phi, lam_psi = lam[0::NF], lam[2::NF]
            for k, cc in enumerate(rec["ch"]):
                kn = n - (k + 1)
                if kn < 0:
                    continue
                pending[kn][0::NF] += (cc / rec["dt"]) * (Mass @ lam_phi)
                pending[kn][2::NF] += (cc / rec["dt"]) * (Mass @ lam_psi)
        return gT
