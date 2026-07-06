"""Node generation and connectivity for k-D tensor-product Lagrange meshes.

Spec: S2.3, S11.1. Nodes live on the DOUBLED integer grid (2^(lmax+1) per
axis) so p2 midpoints are integers. Local ordering x fastest:
a = sum_i idx_i * (p+1)^i, axis 0 = x (load-bearing: basis.py and
constraints.py index against it). Periodic axes: icoord == G2 is identified
with 0 BEFORE dedup, so seam nodes are single DOFs (wrap-around identity);
periodic axes contribute no boundary flags.
"""
from dataclasses import dataclass
from itertools import product
import numpy as np
from ..octree import morton
from ..octree.build import Octree
from ..octree.lookup import face_neighbors

@dataclass(frozen=True)
class Mesh:
    p: int                      # max order present (back-compat: uniform order)
    tree: Octree
    node_coords: np.ndarray     # float64 [Nn, dim], physical unit cube
    node_icoords: np.ndarray    # int64  [Nn, dim], doubled grid
    conn: np.ndarray            # uniform meshes only; None when mixed
    boundary_nodes: np.ndarray  # bool   [Nn]
    p_elem: np.ndarray = None   # int8 [Ne]
    bins: dict = None           # p -> int64 element indices (ascending p)
    conn_of: dict = None        # p -> int32 [nb, (p+1)^dim]

    @property
    def dim(self):
        return self.tree.dim

def _local_offsets(p: int, dim: int) -> np.ndarray:
    """x-fastest lattice: row a = digits of a in base (p+1), axis 0 first."""
    npe = p + 1
    return np.array([tuple(idx) for idx in product(*[range(npe)] * dim)],
                    np.int64)[:, ::-1]  # product varies LAST axis fastest -> reverse

def _validate_one_knob(tree: Octree, p_elem: np.ndarray):
    """Spec S13.2: across any face, level XOR p may change - never both."""
    # face_neighbors is imported at top level — lookup.py has no dependency on
    # nodes.py so there is no circular import.
    lev = tree.levels.astype(np.int64)
    for nbr in face_neighbors(tree):
        ok = nbr >= 0
        i = np.where(ok)[0]
        j = nbr[i]
        bad = (lev[i] != lev[j]) & (p_elem[i] != p_elem[j])
        if bad.any():
            k = i[bad][0]
            raise ValueError(
                f"one-knob rule violated: elements {k} (level {lev[k]}, p {p_elem[k]}) "
                f"and {nbr[k]} (level {lev[nbr[k]]}, p {p_elem[nbr[k]]}) differ in both h and p")

def build_mesh(tree: Octree, p) -> Mesh:
    dim = tree.dim
    p_elem = (np.full(len(tree), p, np.int8) if np.isscalar(p)
              else np.asarray(p, np.int8))
    if not set(np.unique(p_elem)) <= {1, 2}:
        raise ValueError(
            f"p_elem must contain only 1 and 2, got {sorted(set(np.unique(p_elem)))}")
    uniform = len(np.unique(p_elem)) == 1
    if not uniform:
        _validate_one_knob(tree, p_elem)
    size = 1 << (morton.lmax(dim) - tree.levels.astype(np.int64))
    anchors2 = tree.anchors() * 2
    G2 = 2 * (1 << morton.lmax(dim))
    # emit per-bin lattices, dedup jointly (p1 corner nodes are a subset of
    # the p2 doubled-grid lattice, so shared interface nodes collapse to single DOFs)
    per_bin, all_ic, counts = {}, [], []
    for pv in sorted(np.unique(p_elem)):
        eids = np.where(p_elem == pv)[0]
        offs = _local_offsets(int(pv), dim)
        step2 = (2 * size[eids]) // int(pv)
        ic = (anchors2[eids, None, :] + offs[None, :, :] * step2[:, None, None]).reshape(-1, dim)
        per_bin[int(pv)] = eids
        all_ic.append(ic)
        counts.append((int(pv), len(eids), (int(pv) + 1) ** dim))
    all_ic = np.concatenate(all_ic, axis=0)
    for ax in range(dim):
        if tree.periodic[ax]:
            all_ic[:, ax] %= G2
    nodes, inverse = np.unique(all_ic, axis=0, return_inverse=True)
    conn_of, pos = {}, 0
    for pv, nb, npe_d in counts:
        conn_of[pv] = inverse[pos:pos + nb * npe_d].reshape(nb, npe_d).astype(np.int32)
        pos += nb * npe_d
    coords = nodes.astype(np.float64) / G2
    boundary = np.zeros(len(nodes), bool)
    for ax in range(dim):
        if not tree.periodic[ax]:
            boundary |= (nodes[:, ax] == 0) | (nodes[:, ax] == G2)
    pmax = int(p_elem.max())
    return Mesh(pmax, tree, coords, nodes,
                conn_of[pmax] if uniform else None, boundary,
                p_elem=p_elem, bins=per_bin, conn_of=conn_of)
