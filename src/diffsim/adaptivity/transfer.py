"""M3 rung 1: mesh-to-mesh transfer operators (the differentiable-
adaptivity primitive).

P maps a FREE-space field on the OLD mesh to ALL-NODE values on the NEW
mesh: u_new_nodes = P @ u_old_free. Three node classes (same-octree
carve pairs — the epoch re-carve case):
  1. shared nodes (coordinate-matched): identity through the old
     constraint expansion (rows of cons_old.T);
  2. new nodes inside the old retained domain: FE interpolation
     (point_eval_weights on the old mesh);
  3. newly-exposed nodes OUTSIDE the old domain (the fresh strip near a
     moved boundary): nearest-old-node fallback — O(h) there, one cell
     wide; the SBM-consistent shifted evaluation is the recorded
     refinement (M3 ledger).
P is EXPLICIT sparse (the cuFEM adjoint-readiness memo's requirement 1)
so P^T is mechanical — the reverse-sweep contract.
"""
import numpy as np
import scipy.sparse as sp

from ..mesh.pointeval import point_eval_weights


def _coord_key(coords, h_key):
    return (np.round(coords / h_key).astype(np.int64)
            * np.array([1, 2_000_003, 4_000_037_007][:coords.shape[1]],
                       dtype=np.int64)[None, :]).sum(1)


def transfer_operator(mesh_old, cons_old, mesh_new):
    """P sparse [n_new_nodes, n_old_free]."""
    T_old = cons_old.T.tocsr()
    co, cn = mesh_old.node_coords, mesh_new.node_coords
    h_key = 0.5 ** 24
    ko = _coord_key(co, h_key)
    kn = _coord_key(cn, h_key)
    lut = {int(k): i for i, k in enumerate(ko)}
    rows_sh, old_sh, interp_idx = [], [], []
    for j, k in enumerate(kn):
        i = lut.get(int(k))
        if i is None:
            interp_idx.append(j)
        else:
            rows_sh.append(j)
            old_sh.append(i)
    # class 1: shared nodes = the old node's constraint row
    P = sp.lil_matrix((len(cn), T_old.shape[1]))
    Tcsr = T_old
    for j, i in zip(rows_sh, old_sh):
        s, e = Tcsr.indptr[i], Tcsr.indptr[i + 1]
        P.rows[j] = list(map(int, Tcsr.indices[s:e]))
        P.data[j] = list(map(float, Tcsr.data[s:e]))
    # classes 2+3: interpolate where inside, nearest fallback outside
    if interp_idx:
        pts = cn[np.asarray(interp_idx)]
        try:
            W = point_eval_weights(mesh_old, pts)          # [m, Nn_old]
            Wf = (W @ Tcsr).tolil()
            for r, j in enumerate(interp_idx):
                P.rows[j] = Wf.rows[r]
                P.data[j] = Wf.data[r]
        except ValueError:
            # mixed: per-point fallback
            for j in interp_idx:
                try:
                    W = point_eval_weights(mesh_old, cn[j:j + 1])
                    Wf = (W @ Tcsr).tocoo()
                    P.rows[j] = list(map(int, Wf.col))
                    P.data[j] = list(map(float, Wf.data))
                except ValueError:
                    i = int(np.argmin(((co - cn[j]) ** 2).sum(1)))
                    s, e = Tcsr.indptr[i], Tcsr.indptr[i + 1]
                    P.rows[j] = list(map(int, Tcsr.indices[s:e]))
                    P.data[j] = list(map(float, Tcsr.data[s:e]))
    return P.tocsr()
