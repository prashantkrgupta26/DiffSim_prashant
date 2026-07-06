import numpy as np
import warp as wp

# gp_interp is never taped (see operators.py note); skip backward codegen.
wp.set_module_options({"enable_backward": False})

from ..assembly.operators import ConstrainedOperator, _kernel_cache
from ..assembly.dirichlet import _BCOperator
from ..physics.poisson import make_load_kernel


def make_gp_interp_kernel(nbf: int, nqp: int):
    key = ("gp_interp", nbf, nqp)
    if key in _kernel_cache:
        return _kernel_cache[key]

    @wp.kernel(module="unique", enable_backward=False)
    def gp_interp(conn: wp.array2d(dtype=wp.int32), Ntab: wp.array2d(dtype=wp.float64),
                  u: wp.array(dtype=wp.float64), uq: wp.array(dtype=wp.float64)):
        e = wp.tid()
        for q in range(nqp):
            acc = wp.float64(0.0)
            for a in range(nbf):
                acc += Ntab[q, a] * u[conn[e, a]]
            uq[e * nqp + q] = acc

    _kernel_cache[key] = gp_interp
    return gp_interp


class BratuProblem:
    """F(u) = A_bc u - b_bc(lambda e^u), homogeneous Dirichlet on the unit cube.

    Uniform mesh only (single-bin DeviceMesh). Note: a mixed-p mesh constructs
    silently (the constructor only reads ``dm.tables``, which has no
    single-bin assert) and fails at the first ``residual``/``jac_action``
    call, where ``dm.conn`` asserts.
    """
    def __init__(self, dm, lam):
        self.dm, self.lam = dm, lam
        self.op = ConstrainedOperator(dm)
        c = dm.constraints
        self.dir_free = dm.mesh.boundary_nodes[c.free_nodes]
        self.bc_op = _BCOperator(self.op, self.dir_free)
        self.n_free = dm.n_free
        self.nqp, self.nbf = dm.tables.nqp, dm.tables.nbf
        self._interp = make_gp_interp_kernel(self.nbf, self.nqp)
        self._load = make_load_kernel(self.nbf, self.nqp, dm.dim)

    def _to_dev(self, v):
        return wp.array(np.asarray(v, np.float64), dtype=wp.float64, device=self.dm.device)

    def _source_reduced(self, u_free):
        dm, d = self.dm, self.dm.device
        u_full = dm.constraints.T @ u_free
        uq = wp.zeros(len(dm.mesh.tree) * self.nqp, dtype=wp.float64, device=d)
        wp.launch(self._interp, dim=len(dm.mesh.tree),
                  inputs=[dm.conn, dm.N, self._to_dev(u_full), uq], device=d)
        fq = self._to_dev(self.lam * np.exp(uq.numpy()))
        b_full = wp.zeros(dm.n_nodes, dtype=wp.float64, device=d)
        wp.launch(self._load, dim=len(dm.mesh.tree),
                  inputs=[dm.conn, dm.h, dm.N, dm.w, fq, b_full], device=d)
        b = np.asarray(dm.constraints.T.T @ b_full.numpy())
        b[self.dir_free] = 0.0
        return b, uq.numpy()

    def residual(self, u_free):
        y = wp.zeros(self.n_free, dtype=wp.float64, device=self.dm.device)
        self.bc_op.matvec(self._to_dev(u_free), y)
        b, _ = self._source_reduced(u_free)
        return y.numpy() - b

    def jac_action(self, u_free, v_free):
        # J v = A_bc v - M[lambda e^u] v  (mass matrix weighted by lambda e^u at Gauss points)
        y = wp.zeros(self.n_free, dtype=wp.float64, device=self.dm.device)
        self.bc_op.matvec(self._to_dev(v_free), y)
        dm, d = self.dm, self.dm.device
        _, uq = self._source_reduced(u_free)
        v_full = dm.constraints.T @ v_free
        vq = wp.zeros(len(dm.mesh.tree) * self.nqp, dtype=wp.float64, device=d)
        wp.launch(self._interp, dim=len(dm.mesh.tree),
                  inputs=[dm.conn, dm.N, self._to_dev(v_full), vq], device=d)
        gq = self._to_dev(self.lam * np.exp(uq) * vq.numpy())
        m_full = wp.zeros(dm.n_nodes, dtype=wp.float64, device=d)
        wp.launch(self._load, dim=len(dm.mesh.tree),
                  inputs=[dm.conn, dm.h, dm.N, dm.w, gq, m_full], device=d)
        mv = np.asarray(dm.constraints.T.T @ m_full.numpy())
        mv[self.dir_free] = 0.0
        return y.numpy() - mv

    def expand(self, u_free):
        return self.dm.constraints.T @ u_free
