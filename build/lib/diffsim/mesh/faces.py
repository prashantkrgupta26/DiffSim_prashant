"""Face-restricted basis tables for SBM surrogate-face integration.

Spec: M1a plan Task 4; consumed by the SBM Dirichlet/Neumann face kernels
(spec S4.2 step 5) and the surrogate flux observable.

Face id f = 2*ax + side; side 0 fixes xi_ax = -1 (surrogate normal
n_tilde = -e_ax), side 1 fixes xi_ax = +1 — the face_offsets /
BoundaryTypes.WALL order. The remaining dim-1 tangent axes carry the 1D
Gauss lattice, x-fastest among the tangent axes (the reversed-itertools
idiom shared with basis_tables and nodes._local_offsets).

All quantities are REFERENCE-element: physical scaling is dN * (2/h),
d2N * (2/h)^2, and dS = w * (h/2)^(dim-1) — applied by the kernels.
Local node ordering a matches _local_offsets(p, dim) (load-bearing:
kernels index conn[e, a] against it).

Device note: warp arrays cap at 4 dims, so kernels receive d2N flattened to
[2*dim, nqf, nbf, dim*dim] (see femelm.fe_d2N_s); the host dataclass keeps
the unflattened [2*dim, nqf, nbf, dim, dim] layout for tests/reference use.
"""
from dataclasses import dataclass
from itertools import product as iproduct
import numpy as np

from .basis import gauss_1d, lagrange_1d, lagrange_1d_d2
from .nodes import _local_offsets


@dataclass(frozen=True)
class FaceTables:
    p: int
    dim: int
    nqf: int
    N: np.ndarray     # [2*dim, nqf, nbf]
    dN: np.ndarray    # [2*dim, nqf, nbf, dim]   (reference)
    d2N: np.ndarray   # [2*dim, nqf, nbf, dim, dim] (reference)
    w: np.ndarray     # [nqf]  (product of dim-1 1D Gauss weights)
    xi: np.ndarray    # [2*dim, nqf, dim] face-GP reference coords in [-1,1]^dim

    @property
    def nbf(self):
        return (self.p + 1) ** self.dim


def face_tables(p: int, dim: int, nq1: int = None) -> FaceTables:
    """nq1: 1D Gauss points per tangent axis (default p+1). Raising it is the
    differentiable-safe way to integrate surrogate-face terms more accurately
    — the shifted data g(x + d) is not polynomial, and the points stay fixed
    and geometry-independent (unlike cut-cell volume rules, spec S1.5)."""
    if nq1 is None:
        nq1 = p + 1
    if nq1 in (2, 3):
        pts, wts = gauss_1d(nq1 - 1)
    else:
        pts, wts = np.polynomial.legendre.leggauss(nq1)
    npe = p + 1
    nbf, nqf = npe ** dim, nq1 ** (dim - 1)
    aidx = _local_offsets(p, dim)                       # [nbf, dim]
    # tangent-lattice multi-indices, x-fastest among the tangent axes
    qidx_t = np.array(list(iproduct(*[range(nq1)] * (dim - 1))),
                      np.int64).reshape(nqf, dim - 1)[:, ::-1]
    w = np.array([np.prod(wts[qi]) for qi in qidx_t])

    N = np.zeros((2 * dim, nqf, nbf))
    dN = np.zeros((2 * dim, nqf, nbf, dim))
    d2N = np.zeros((2 * dim, nqf, nbf, dim, dim))
    xi_all = np.zeros((2 * dim, nqf, dim))
    for ax in range(dim):
        tang = [d for d in range(dim) if d != ax]
        for side, xi_fix in ((0, -1.0), (1, 1.0)):
            f = 2 * ax + side
            for q in range(nqf):
                xi = np.empty(dim)
                xi[ax] = xi_fix
                for j, d in enumerate(tang):
                    xi[d] = pts[qidx_t[q, j]]
                xi_all[f, q] = xi
                n1 = [lagrange_1d(p, xi[d])[0] for d in range(dim)]
                dn1 = [lagrange_1d(p, xi[d])[1] for d in range(dim)]
                d2n1 = [lagrange_1d_d2(p, xi[d]) for d in range(dim)]
                for a in range(nbf):
                    vals = np.array([n1[d][aidx[a, d]] for d in range(dim)])
                    N[f, q, a] = vals.prod()
                    for i in range(dim):
                        t = vals.copy()
                        t[i] = dn1[i][aidx[a, i]]
                        dN[f, q, a, i] = t.prod()
                        for j in range(i, dim):
                            t2 = vals.copy()
                            if i == j:
                                t2[i] = d2n1[i][aidx[a, i]]
                            else:
                                t2[i] = dn1[i][aidx[a, i]]
                                t2[j] = dn1[j][aidx[a, j]]
                            v = t2.prod()
                            d2N[f, q, a, i, j] = v
                            d2N[f, q, a, j, i] = v
    return FaceTables(p, dim, nqf, N, dN, d2N, w, xi_all)
