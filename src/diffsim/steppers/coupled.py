"""M2-A3: one-way coupled scalar transport over a flow history.

ScalarTransportStepper advances T (or a species C — A2: same brick,
different coefficients) with BDF1/BDF2 over a PROVIDED velocity
history: either a frozen steady field or the per-step fields of an NS
stepper run (one-way forced convection). Two-way (Boussinesq) is A4 —
the buoyancy source enters the NS step; this class stays the scalar
half.

Contract: advecting velocities arrive as GP fields per bin (the same
aq layout the NS steppers already compute — share, don't recompute).
Strong Dirichlet on marked nodes; SBM face blocks (SBMPoisson) add
linearly when a geometry is present.
"""
import numpy as np
from scipy.sparse.linalg import splu

from ..physics.scalar_transport import assemble_scalar_ad
from ..physics.poisson import gauss_points


class ScalarTransportStepper:
    def __init__(self, dm, kappa, dt, g_fn, dirichlet_nodes, order=2,
                 f_fn=None, supg=None):
        self.dm = dm
        self.kappa, self.dt = kappa, float(dt)
        self.order = order
        self.g_fn = g_fn                    # g(x, t) on dirichlet nodes
        self.f_fn = f_fn or (lambda x, t: np.zeros(len(x)))
        self.supg = supg
        self.dir_nodes = np.asarray(dirichlet_nodes)
        self.free_coords = dm.mesh.node_coords[dm.constraints.free_nodes]
        self.xq = gauss_points(dm.mesh, dm.tables_by_p)
        self.t = 0.0
        self.hist = []                       # [T_{n}, T_{n-1}] free vecs
        # GP interpolation of a free-vector to bins (scalar)
        self._Tcsr = dm.constraints.T.tocsr()

    def set_initial(self, T0_fn):
        T0 = T0_fn(self.free_coords)
        self.hist = [T0.copy(), T0.copy()]
        self.t = 0.0
        return T0

    def _bdf(self):
        if self.order == 1 or self.t < self.dt / 2:
            return 1.0, [1.0], 1               # c0, hist coeffs
        return 1.5, [2.0, -0.5], 2

    def gp_scalar(self, vec_free):
        """Free scalar vector -> GP values per bin."""
        full = np.asarray(self._Tcsr @ vec_free)
        out = {}
        for pv, b in self.dm.bins.items():
            conn = self.dm.mesh.conn_of[pv]
            tb = self.dm.tables_by_p[pv]
            out[pv] = np.einsum("qa,ea->eq", tb.N,
                                full[conn]).reshape(-1)
        return out

    def step(self, aq_by_bin):
        """One BDF step with the advecting GP field aq_by_bin."""
        t_new = self.t + self.dt
        c0, ch, _ = self._bdf()
        sigma = c0 / self.dt
        fq = {pv: self.f_fn(self.xq[pv], t_new) for pv in self.xq}
        # history contribution: sum ch_k T_{n-k} / dt enters the rhs via
        # the SAME weighted test functions (mass + SUPG) => assemble with
        # f_total = f + hist/dt evaluated at GPs
        hist_gp = None
        for k, c in enumerate(ch):
            g = self.gp_scalar(self.hist[k])
            if hist_gp is None:
                hist_gp = {pv: (c / self.dt) * g[pv] for pv in g}
            else:
                for pv in g:
                    hist_gp[pv] += (c / self.dt) * g[pv]
        fq = {pv: fq[pv] + hist_gp[pv] for pv in fq}
        A, b = assemble_scalar_ad(self.dm, aq_by_bin, fq, self.kappa,
                                  sigma=sigma, supg=self.supg)
        A = A.tolil()
        gvals = self.g_fn(self.free_coords[self.dir_nodes], t_new)
        for k, i in enumerate(self.dir_nodes):
            A.rows[i] = [int(i)]
            A.data[i] = [1.0]
            b[i] = gvals[k]
        Tn = splu(A.tocsr().tocsc()).solve(b)
        self.hist = [Tn.copy(), self.hist[0]]
        self.t = t_new
        return Tn
