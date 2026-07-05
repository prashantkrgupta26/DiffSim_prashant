"""Point-evaluation weight matrix: sparse Lagrange interpolation weights of
arbitrary probe points against the mesh's nodal basis (probe QoIs, spec S14
observables). W [Npts, Nn] with u(x_i) = (W @ u_all)_i; dJ/du_all = W^T (...).
Host-side (scipy CSR); the containing element is located by integer anchor
arithmetic + LeafLookup (periodic-aware)."""
import numpy as np
import scipy.sparse as sp

from ..octree import morton
from ..octree.lookup import LeafLookup
from .basis import lagrange_1d
from .nodes import Mesh, _local_offsets


def point_eval_weights(mesh: Mesh, pts: np.ndarray) -> sp.csr_matrix:
    dim = mesh.dim
    tree = mesh.tree
    L = morton.lmax(dim)
    G = 1 << L
    pts = np.atleast_2d(np.asarray(pts, np.float64))
    lk = LeafLookup(tree)
    anchors = np.clip((pts * G).astype(np.int64), 0, G - 1)
    elems = lk.find(anchors)
    if (elems < 0).any():
        bad = np.where(elems < 0)[0][0]
        raise ValueError(f"probe point {pts[bad]} is outside the mesh")
    lo = tree.anchors() / G
    h = tree.h()
    p_elem = np.asarray(mesh.p_elem)
    row_of = {}
    for pv, eids in mesh.bins.items():
        for r, e in enumerate(eids):
            row_of[int(e)] = (pv, r)

    rows, cols, vals = [], [], []
    for i, (x, e) in enumerate(zip(pts, elems)):
        pv, r = row_of[int(e)]
        conn = mesh.conn_of[pv][r]
        xi = 2.0 * (x - lo[e]) / h[e] - 1.0
        offs = _local_offsets(pv, dim)
        N1 = [lagrange_1d(pv, xi[d])[0] for d in range(dim)]
        for a in range(len(offs)):
            w = 1.0
            for d in range(dim):
                w *= N1[d][offs[a, d]]
            if abs(w) > 1e-15:
                rows.append(i)
                cols.append(int(conn[a]))
                vals.append(w)
    return sp.csr_matrix((vals, (rows, cols)),
                         shape=(len(pts), len(mesh.node_coords)))
