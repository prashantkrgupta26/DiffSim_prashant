"""Vector-DOF matrix-free constrained matvec (M1b Task 1b).

DOF layout (spec S3.1, node-major blocks): global index = node*ndof + dof —
the TalyFEM `Ae(ndof*a + i, ndof*b + j)` convention. Constraints act
per-component: T_vec = kron(T, I_ndof), built once per epoch on host and
applied on device via csr_spmv (the hanging/p-transition machinery is
component-agnostic).

The concrete operator here is the vector Laplacian (per-component Poisson
stiffness) — the parity workhorse for the M1b solver stack; NS momentum
kernels plug into the same ConstrainedVectorOperator shape via the brick API
(Task 2+). Kernels: module="unique", enable_backward=False (never taped;
finding 4c patterns respected — no loop-reassigned locals feed adjoints)."""
import numpy as np
import scipy.sparse as sp
import warp as wp

from .femelm import FEMElm, fe_dN_s, fe_detJxW_s
from .operators import _kernel_cache, _csr_to_device, csr_spmv


def make_vector_poisson_matvec(nbf: int, nqp: int, dim: int, ndof: int):
    """y[node*ndof + c] += (K x)[node*ndof + c] per component c — the
    vector-Laplacian action with node-major blocks."""
    key = ("vec_poisson_mv", nbf, nqp, dim, ndof)
    if key in _kernel_cache:
        return _kernel_cache[key]

    dim_pow = float(dim)   # separate name: wp.float64(dim) on the bare name
                           # coerces `dim` kernel-wide and breaks range(dim)

    @wp.kernel(module="unique", enable_backward=False)
    def vec_poisson_mv(conn: wp.array2d(dtype=wp.int32),
                       h: wp.array(dtype=wp.float64),
                       dNtab: wp.array3d(dtype=wp.float64),
                       wtab: wp.array(dtype=wp.float64),
                       x: wp.array(dtype=wp.float64),
                       y: wp.array(dtype=wp.float64)):
        e = wp.tid()
        fe = FEMElm()
        fe.e = e
        fe.he = h[e]
        jac = wp.pow(fe.he * wp.float64(0.5), wp.float64(dim_pow))
        dscale = wp.float64(2.0) / fe.he
        for q in range(nqp):
            fe.q = q
            dJxW = fe_detJxW_s(wtab, fe, jac)
            for c in range(ndof):
                # g_d = sum_b dN_b,d * x_(b,c)
                for a in range(nbf):
                    val = wp.float64(0.0)
                    for d in range(dim):
                        g = wp.float64(0.0)
                        for b in range(nbf):
                            g += fe_dN_s(dNtab, fe, b, d, dscale) \
                                 * x[conn[e, b] * ndof + c]
                        val += fe_dN_s(dNtab, fe, a, d, dscale) * g
                    wp.atomic_add(y, conn[e, a] * ndof + c, val * dJxW)

    _kernel_cache[key] = vec_poisson_mv
    return vec_poisson_mv


class ConstrainedVectorOperator:
    """y_free = T_vec^T (A (T_vec x_free)) for ndof components, node-major.
    Mixed-p: one launch per bin accumulating into the same full vector.
    Operator protocol: .matvec/.matvec_numpy/.n_free/.device."""

    def __init__(self, dm, ndof: int):
        self.dm, self.ndof = dm, ndof
        d = dm.device
        T = dm.constraints.T.tocsr()
        T_vec = sp.kron(T, sp.identity(ndof, format="csr"), format="csr")
        self._Tv = _csr_to_device(T_vec, d)
        self._Tvt = _csr_to_device(T_vec.T.tocsr(), d)
        self.n_full = dm.n_nodes * ndof
        self.n_free = T.shape[1] * ndof
        self.device = d
        self._kernels = {
            pv: make_vector_poisson_matvec(b["nbf"], b["nqp"], dm.dim, ndof)
            for pv, b in dm.bins.items()}
        self._x_full = wp.zeros(self.n_full, dtype=wp.float64, device=d)
        self._y_full = wp.zeros(self.n_full, dtype=wp.float64, device=d)

    def matvec(self, x_free: wp.array, y_free: wp.array):
        dm, d = self.dm, self.device
        wp.launch(csr_spmv, dim=self.n_full,
                  inputs=[*self._Tv, x_free, self._x_full], device=d)
        self._y_full.zero_()
        for pv, b in dm.bins.items():
            wp.launch(self._kernels[pv], dim=len(b["eids"]),
                      inputs=[b["conn"], b["h"], b["dN"], b["w"],
                              self._x_full, self._y_full],
                      device=d)
        wp.launch(csr_spmv, dim=self.n_free,
                  inputs=[*self._Tvt, self._y_full, y_free], device=d)

    def matvec_numpy(self, x: np.ndarray) -> np.ndarray:
        xd = wp.array(np.ascontiguousarray(x, np.float64), dtype=wp.float64,
                      device=self.device)
        yd = wp.zeros(self.n_free, dtype=wp.float64, device=self.device)
        self.matvec(xd, yd)
        return yd.numpy()
