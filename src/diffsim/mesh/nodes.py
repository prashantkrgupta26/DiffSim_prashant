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

@dataclass(frozen=True)
class Mesh:
    p: int
    tree: Octree
    node_coords: np.ndarray    # float64 [Nn, dim], physical unit cube
    node_icoords: np.ndarray   # int64  [Nn, dim], doubled grid
    conn: np.ndarray           # int32  [Ne, (p+1)^dim]
    boundary_nodes: np.ndarray # bool   [Nn]

    @property
    def dim(self):
        return self.tree.dim

def _local_offsets(p: int, dim: int) -> np.ndarray:
    """x-fastest lattice: row a = digits of a in base (p+1), axis 0 first."""
    npe = p + 1
    return np.array([tuple(idx) for idx in product(*[range(npe)] * dim)],
                    np.int64)[:, ::-1]  # product varies LAST axis fastest -> reverse

def build_mesh(tree: Octree, p: int) -> Mesh:
    assert p in (1, 2)
    dim = tree.dim
    npe = p + 1
    offs = _local_offsets(p, dim)                       # [(p+1)^dim, dim]
    size = 1 << (morton.lmax(dim) - tree.levels.astype(np.int64))
    anchors2 = tree.anchors() * 2
    step2 = (2 * size) // p                             # h/p on the doubled grid
    all_ic = (anchors2[:, None, :] + offs[None, :, :] * step2[:, None, None]).reshape(-1, dim)
    G2 = 2 * (1 << morton.lmax(dim))
    for ax in range(dim):                               # periodic seam identity
        if tree.periodic[ax]:
            all_ic[:, ax] %= G2
    nodes, inverse = np.unique(all_ic, axis=0, return_inverse=True)
    conn = inverse.reshape(len(tree), npe**dim).astype(np.int32)
    coords = nodes.astype(np.float64) / G2
    boundary = np.zeros(len(nodes), bool)
    for ax in range(dim):
        if not tree.periodic[ax]:
            boundary |= (nodes[:, ax] == 0) | (nodes[:, ax] == G2)
    return Mesh(p, tree, coords, nodes, conn, boundary)
