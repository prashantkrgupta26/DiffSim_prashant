r"""Differentiable phase-field: discrete IFT adjoint through a Cahn-Hilliard
BDF march (milestone G1/G2 track (a) material parameters).

WHY A SELF-CONTAINED DISCRETE FORWARD.  The production CahnHilliardStepper
(physics/cahn_hilliard.py) assembles its Jacobian in a Warp kernel compiled
enable_backward=False and solves it with splu/cuDSS — no cotangent flows.
This module reimplements the SAME discrete residual + Jacobian in vectorised
numpy, pulling the identical basis tables (N, dN, w, h) and dof layout
(node-major (c, mu), block 2a+0 = c, 2a+1 = mu) straight off the DeviceMesh,
so a converged forward step here matches the production stepper to Newton
tolerance (asserted in tests/test_phasefield_adjoint.py::test_forward_parity).

THE ADJOINT IS THE IFT ADJOINT (the clean one, per the milestone directive).
At each committed step the forward Newton has driven R_n(x_n; x_hist, p) = 0.
We do NOT tape the Newton iterations; we differentiate only the converged
implicit relation.  For J = sum_n j(x_n),

    J_n^T lam_n = dj/dx_n  -  sum_{k>=1} (dR_{n+k}/dx_n)^T lam_{n+k}
    dJ/dp       = - sum_n lam_n^T (dR_n/dp)

where J_n = dR_n/dx_n is EXACTLY the Jacobian the forward Newton assembled at
convergence, and dR_n/dp is analytic (the residual is an explicit function of
M, kappa and the bulk-energy coefficients at each Gauss point).  The BDF
history enters R_{n+k} only through the mass/time term, so the history
cotangent is (ch_k/dt) * Mass * lam[c-rows] mapped back to the c-columns of
step n-k (mirrors sbm/transient_adjoint.py's sigma/b1/b2 coupling).

Gate scope: uniform mesh (constraints.T == identity), natural no-flux BCs
(no Dirichlet rows), BDF1 (G1) and variable-coefficient BDF2 (G2)."""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu


# --------------------------------------------------------------------------
# bulk free energy models: f(c), f'(c), f''(c) + analytic parameter deriv of
# f'(c) w.r.t. each learnable coefficient (the adjoint needs dR_mu/dp).
# --------------------------------------------------------------------------
class PolyEnergy:
    """Double well f(c) = 1/4 (c^2 - 1)^2; f' = c^3 - c, f'' = 3c^2 - 1.
    No tunable bulk coefficient (mobility/kappa are the G1 params)."""
    param_names = ()

    def f(self, c):
        return 0.25 * (c * c - 1.0) ** 2

    def fp(self, c):
        return c ** 3 - c

    def fpp(self, c):
        return 3.0 * c * c - 1.0

    def dfp_dparam(self, c, name):
        raise KeyError(name)


class FHEnergy:
    """Flory-Huggins f(c) = A[c ln c + (1-c)ln(1-c)] + B c(1-c),
    f'  = A[ln c - ln(1-c)] + B(1 - 2c),  f'' = A[1/c + 1/(1-c)] - 2B.
    B is the Flory interaction chi (the G1 'chi' parameter); A the entropic
    scale.  Gates stay interior (c in ~[0.05, 0.95]) so the production log
    regularisation is inactive and the derivatives are the exact logs."""
    param_names = ("A", "B")

    def __init__(self, A=1.0, B=3.0):
        self.A = float(A)
        self.B = float(B)

    def f(self, c):
        return (self.A * (c * np.log(c) + (1.0 - c) * np.log(1.0 - c))
                + self.B * c * (1.0 - c))

    def fp(self, c):
        return (self.A * (np.log(c) - np.log(1.0 - c))
                + self.B * (1.0 - 2.0 * c))

    def fpp(self, c):
        return self.A * (1.0 / c + 1.0 / (1.0 - c)) - 2.0 * self.B

    def dfp_dparam(self, c, name):
        if name == "A":
            return np.log(c) - np.log(1.0 - c)
        if name == "B":
            return 1.0 - 2.0 * c
        raise KeyError(name)


# --------------------------------------------------------------------------
# discrete mesh operator (numpy mirror of the production CH kernel)
# --------------------------------------------------------------------------
class CHDiscrete:
    """Vectorised numpy CH residual/Jacobian on a DeviceMesh, bit-consistent
    with physics/cahn_hilliard.make_ch_newton."""

    def __init__(self, dm):
        T = dm.constraints.T
        assert T.shape[0] == T.shape[1] and (abs(T - sp.eye(T.shape[0])).nnz
                                             == 0), \
            "adjoint gate assumes constraints.T == identity (uniform mesh)"
        self.dm = dm
        self.dim = dm.dim
        self.nn = dm.n_nodes
        self.ndof = 2 * self.nn
        self.bins = []
        for pv, b in dm.bins.items():
            tb = dm.tables_by_p[pv]
            conn = dm.mesh.conn_of[pv].astype(np.int64)
            h = np.asarray(dm.mesh.tree.h()[dm.mesh.bins[pv]], np.float64)
            N = np.asarray(tb.N, np.float64)               # [nqp, nbf]
            dN = np.asarray(tb.dN, np.float64)             # [nqp, nbf, dim]
            w = np.asarray(tb.w, np.float64)               # [nqp]
            ne, nbf = conn.shape
            dscale = 2.0 / h                               # [ne]
            jac = (h / 2.0) ** self.dim                    # [ne]
            dJxW = w[None, :] * jac[:, None]               # [ne, nqp]
            # global (c,mu) dof indices per element: [ne, 2*nbf]
            gdof = (conn[:, :, None] * 2
                    + np.arange(2)[None, None, :]).reshape(ne, 2 * nbf)
            self.bins.append(dict(conn=conn, N=N, dN=dN, w=w, h=h, ne=ne,
                                  nbf=nbf, nqp=N.shape[0], dscale=dscale,
                                  dJxW=dJxW, gdof=gdof))
        self._mass = None

    # -- GP interpolation of a nodal scalar --------------------------------
    def interp(self, field):
        out = []
        for B in self.bins:
            vals = field[B["conn"]]                         # [ne, nbf]
            c_ = np.einsum("qa,ea->eq", B["N"], vals)
            gc = np.einsum("qad,ea,e->eqd", B["dN"], vals, B["dscale"])
            out.append((c_, gc))
        return out

    # -- lumped-consistent global mass matrix (history coupling) -----------
    def mass_matrix(self):
        if self._mass is not None:
            return self._mass
        rows, cols, vals = [], [], []
        for B in self.bins:
            NN = np.einsum("eq,qa,qb->eab", B["dJxW"], B["N"], B["N"])
            conn = B["conn"]
            nbf = B["nbf"]
            r = np.repeat(conn, nbf, axis=1)
            c = np.tile(conn, (1, nbf))
            rows.append(r.ravel())
            cols.append(c.ravel())
            vals.append(NN.ravel())
        self._mass = sp.coo_matrix(
            (np.concatenate(vals),
             (np.concatenate(rows), np.concatenate(cols))),
            shape=(self.nn, self.nn)).tocsr()
        return self._mass

    # -- residual + Jacobian at (c, mu) with GP history --------------------
    def assemble(self, c_free, mu_free, hist_gp, params, want_jac=True):
        """R (len 2*nn) and, if want_jac, J = dR/dx (csr 2nn x 2nn).
        hist_gp is a list (per bin) of [ne, nqp] BDF history at GPs
        (sum_k (ch_k/dt) * interp(c_{n-k}))."""
        M = params["M"]
        kap = params["kappa"]
        energy = params["energy"]
        sigma = params["sigma"]
        fc = params.get("fc")            # optional MMS sources (per bin, gp)
        fm = params.get("fm")
        ci = self.interp(c_free)
        mi = self.interp(mu_free)
        R = np.zeros(self.ndof)
        rows, cols, vals = [], [], []
        for bi, B in enumerate(self.bins):
            c_, gc = ci[bi]
            mu_, gmu = mi[bi]
            dJxW = B["dJxW"]
            N, dN, dscale = B["N"], B["dN"], B["dscale"]
            hg = hist_gp[bi]
            fp = energy.fp(c_)
            fpp = energy.fpp(c_)
            # gradN[e,q,a,d] = dN[q,a,d]*dscale[e]
            # R_c = Int N(sigma c - hist) + M Int gradN.grad mu - Int N fc
            gN_gmu = np.einsum("qad,e,eqd->eqa", dN, dscale, gmu)  # gradN.gmu
            fcq = (fc[bi] if fc is not None else 0.0)
            Rc = np.einsum("eq,qa->ea", dJxW * (sigma * c_ - hg), N) \
                + M * np.einsum("eq,eqa->ea", dJxW, gN_gmu)
            if fc is not None:
                Rc -= np.einsum("eq,qa->ea", dJxW * fcq, N)
            # R_mu = Int N(mu - f') - Int N fm - kap Int gradN.grad c
            gN_gc = np.einsum("qad,e,eqd->eqa", dN, dscale, gc)
            fmq = (fm[bi] if fm is not None else 0.0)
            Rm = np.einsum("eq,qa->ea", dJxW * (mu_ - fp), N) \
                - kap * np.einsum("eq,eqa->ea", dJxW, gN_gc)
            if fm is not None:
                Rm -= np.einsum("eq,qa->ea", dJxW * fmq, N)
            # scatter residual
            np.add.at(R, B["gdof"][:, 0::2].ravel(), Rc.ravel())
            np.add.at(R, B["gdof"][:, 1::2].ravel(), Rm.ravel())
            if not want_jac:
                continue
            NN = np.einsum("eq,qa,qb->eab", dJxW, N, N)
            LL = np.einsum("eq,qad,qbd->eab", dJxW * dscale[:, None] ** 2,
                           dN, dN)
            FPP = np.einsum("eq,qa,qb->eab", dJxW * fpp, N, N)
            nbf = B["nbf"]
            Ae = np.zeros((B["ne"], 2 * nbf, 2 * nbf))
            Ae[:, 0::2, 0::2] = sigma * NN            # dRc/dc
            Ae[:, 0::2, 1::2] = M * LL                # dRc/dmu
            Ae[:, 1::2, 0::2] = -FPP - kap * LL       # dRmu/dc
            Ae[:, 1::2, 1::2] = NN                    # dRmu/dmu
            gdof = B["gdof"]
            rows.append(np.repeat(gdof, 2 * nbf, axis=1).ravel())
            cols.append(np.tile(gdof, (1, 2 * nbf)).ravel())
            vals.append(Ae.ravel())
        J = None
        if want_jac:
            J = sp.coo_matrix(
                (np.concatenate(vals),
                 (np.concatenate(rows), np.concatenate(cols))),
                shape=(self.ndof, self.ndof)).tocsr()
        return R, J

    # -- analytic dR/dp vector at a committed state ------------------------
    def dR_dparam(self, c_free, mu_free, params, name):
        """Explicit derivative of the residual w.r.t. a scalar parameter,
        returned as a length-2nn vector (node-major (c, mu))."""
        energy = params["energy"]
        ci = self.interp(c_free)
        mi = self.interp(mu_free)
        out = np.zeros(self.ndof)
        for bi, B in enumerate(self.bins):
            c_, gc = ci[bi]
            mu_, gmu = mi[bi]
            dJxW = B["dJxW"]
            N, dN, dscale = B["N"], B["dN"], B["dscale"]
            if name == "M":
                # dR_c/dM = Int gradN . grad mu
                gN_gmu = np.einsum("qad,e,eqd->eqa", dN, dscale, gmu)
                Rc = np.einsum("eq,eqa->ea", dJxW, gN_gmu)
                np.add.at(out, B["gdof"][:, 0::2].ravel(), Rc.ravel())
            elif name == "kappa":
                # dR_mu/dkappa = -Int gradN . grad c
                gN_gc = np.einsum("qad,e,eqd->eqa", dN, dscale, gc)
                Rm = -np.einsum("eq,eqa->ea", dJxW, gN_gc)
                np.add.at(out, B["gdof"][:, 1::2].ravel(), Rm.ravel())
            else:
                # bulk-energy coefficient: R_mu has the -Int N f'(c) term,
                # so dR_mu/dp = -Int N (df'/dp)
                dfp = energy.dfp_dparam(c_, name)
                Rm = -np.einsum("eq,qa->ea", dJxW * dfp, N)
                np.add.at(out, B["gdof"][:, 1::2].ravel(), Rm.ravel())
        return out


# --------------------------------------------------------------------------
# forward march (numpy) + recorder
# --------------------------------------------------------------------------
class CHForward:
    """Newton BDF march of the discrete CH operator, recording per-step the
    converged state and coefficients the adjoint replays."""

    def __init__(self, dm, energy, M=1.0, kappa=1e-2, dt=1e-2, order=1,
                 newton_tol=1e-12, newton_max=30):
        self.op = CHDiscrete(dm)
        self.energy = energy
        self.M, self.kappa, self.dt = float(M), float(kappa), float(dt)
        self.order = order
        self.newton_tol, self.newton_max = newton_tol, newton_max
        self.t = 0.0
        self.dt_prev = None
        self.steps = []          # per-step record for the adjoint

    def set_initial(self, c0):
        c0 = np.asarray(c0, np.float64)
        self.c = c0.copy()
        self.mu = np.zeros_like(c0)
        self.hist = [c0.copy(), c0.copy()]
        self.t = 0.0
        self.dt_prev = None
        self.steps = []

    def _bdf(self):
        if self.order == 1 or self.dt_prev is None or self.t < self.dt / 2:
            return 1.0, [1.0]
        rr = self.dt / self.dt_prev
        return (1.0 + 2.0 * rr) / (1.0 + rr), [1.0 + rr, -rr * rr / (1.0 + rr)]

    def _hist_gp(self, ch):
        hist_gp = None
        for k, cc in enumerate(ch):
            hi = self.op.interp(self.hist[k])
            if hist_gp is None:
                hist_gp = [(cc / self.dt) * hi[bi][0]
                           for bi in range(len(hi))]
            else:
                for bi in range(len(hi)):
                    hist_gp[bi] = hist_gp[bi] + (cc / self.dt) * hi[bi][0]
        return hist_gp

    def _params(self, sigma):
        return dict(M=self.M, kappa=self.kappa, energy=self.energy,
                    sigma=sigma)

    def step(self, record=True):
        c0_, ch = self._bdf()
        sigma = c0_ / self.dt
        hist_gp = self._hist_gp(ch)
        params = self._params(sigma)
        c, mu = self.c.copy(), self.mu.copy()
        for it in range(self.newton_max):
            R, J = self.op.assemble(c, mu, hist_gp, params, want_jac=True)
            dx = splu(J.tocsc()).solve(-R)
            c = c + dx[0::2]
            mu = mu + dx[1::2]
            if np.abs(dx).max() < self.newton_tol:
                break
        self.c, self.mu = c, mu
        if record:
            self.steps.append(dict(
                c=c.copy(), mu=mu.copy(), sigma=sigma, ch=list(ch),
                dt=self.dt, params=self._params(sigma)))
        self.hist = [c.copy(), self.hist[0]]
        self.t += self.dt
        self.dt_prev = self.dt
        return c.copy(), mu.copy()

    def run(self, n_steps):
        return [self.step() for _ in range(n_steps)]


# --------------------------------------------------------------------------
# IFT reverse-sweep adjoint
# --------------------------------------------------------------------------
class CHAdjoint:
    """Reverse-sweep dJ/d{M, kappa, energy-coeffs} for J = sum_n j(x_n)."""

    def __init__(self, fwd: CHForward):
        self.fwd = fwd
        self.op = fwd.op

    def gradient(self, dJdx_list, param_names):
        """dJdx_list[n] = dj/dx_n as a length-2nn node-major vector.
        Returns {name: dJ/dname}."""
        op = self.op
        steps = self.fwd.steps
        N = len(steps)
        Mass = op.mass_matrix()
        grads = {nm: 0.0 for nm in param_names}
        # pending[n] = extra rhs cotangents from later steps (len-2nn)
        pending = [np.zeros(op.ndof) for _ in range(N)]
        # J = dR/dx is independent of the BDF history load, so a zero
        # history suffices to rebuild the Jacobian at the converged iterate.
        zero_hist = [np.zeros_like(B["dJxW"]) for B in op.bins]
        for n in range(N - 1, -1, -1):
            rec = steps[n]
            _, J = op.assemble(rec["c"], rec["mu"], zero_hist, rec["params"],
                               want_jac=True)
            rhs = np.asarray(dJdx_list[n], np.float64) + pending[n]
            lam = splu(J.T.tocsc()).solve(rhs)
            lam_c = lam[0::2]
            # parameter accumulation: dJ/dp -= lam^T dR/dp
            for nm in param_names:
                dRdp = op.dR_dparam(rec["c"], rec["mu"], rec["params"], nm)
                grads[nm] -= float(lam @ dRdp)
            # history cotangent to earlier steps: R_n depends on c_{n-k}
            # only through -(ch_k/dt) Mass on the c-block; contribution to
            # earlier rhs is +(ch_k/dt) Mass lam_c placed on the c-columns.
            ch = rec["ch"]
            dt = rec["dt"]
            for k, cc in enumerate(ch):
                kn = n - (k + 1)
                if kn < 0:
                    continue
                hc = (cc / dt) * (Mass @ lam_c)
                pending[kn][0::2] += hc
        return grads
