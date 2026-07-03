from dataclasses import dataclass
import numpy as np
from ..octree import morton
from ..octree.build import Octree

@dataclass(frozen=True)
class Mesh:
    p: int
    tree: Octree
    node_coords: np.ndarray    # float64 [Nn,3], physical (unit cube)
    node_icoords: np.ndarray   # int64  [Nn,3], grid of size 2^(LMAX+1)+1
    conn: np.ndarray           # int32  [Ne, (p+1)^3]
    boundary_nodes: np.ndarray # bool   [Nn]

def build_mesh(tree: Octree, p: int) -> Mesh:
    assert p in (1, 2)
    npe = p + 1
    # local lattice offsets in units of h/p, x fastest
    # meshgrid 'ij' gives ax varying slowest; we need x fastest -> build explicitly:
    offs = np.array([[i, j, k] for k in range(npe) for j in range(npe) for i in range(npe)],
                    np.int64)                      # a = i + npe*j + npe^2*k
    # integer node coords on the doubled grid (so p2 midpoints are integers)
    anchors2 = tree.anchors() * 2                  # anchor on doubled grid
    # careful: for p=1 step2 = 2*size, for p=2 step2 = size
    size = 1 << (morton.LMAX - tree.levels.astype(np.int64))
    step2 = (2 * size) // p
    all_icoords = (anchors2[:, None, :] + offs[None, :, :] * step2[:, None, None]).reshape(-1, 3)
    nodes, inverse = np.unique(all_icoords, axis=0, return_inverse=True)
    conn = inverse.reshape(len(tree), npe**3).astype(np.int32)
    G2 = 2 * (1 << morton.LMAX)
    coords = nodes.astype(np.float64) / G2
    boundary = np.any((nodes == 0) | (nodes == G2), axis=1)
    return Mesh(p, tree, coords, nodes, conn, boundary)
