from dataclasses import dataclass
import numpy as np
import scipy.sparse as sp
from ..octree import morton
from ..octree.lookup import LeafLookup
from .nodes import Mesh
from .basis import lagrange_1d


@dataclass(frozen=True)
class Constraints:
    T: sp.csr_matrix        # Nn × Nfree; maps free-node values to all-node values
    free_nodes: np.ndarray  # int64[Nfree]
    hanging: np.ndarray     # bool[Nn]


def build_constraints(mesh: Mesh) -> Constraints:
    tree, p = mesh.tree, mesh.p
    lk = LeafLookup(tree)
    Nn = len(mesh.node_coords)
    lev = tree.levels.astype(np.int64)
    size2 = 2 * (1 << (morton.LMAX - lev))   # element size on doubled grid, per element
    anchors2 = tree.anchors() * 2             # element anchors on doubled grid

    # Build node -> list-of-elements incidence from conn
    node_elems = [[] for _ in range(Nn)]
    for e in range(len(tree)):
        for a in mesh.conn[e]:
            node_elems[a].append(e)

    # Doubled-grid domain size (exclusive upper bound)
    G2 = 2 * (1 << morton.LMAX)

    hanging = np.zeros(Nn, bool)
    owner = np.full(Nn, -1, np.int64)

    for n in range(Nn):
        ic = mesh.node_icoords[n]
        # Probe the up-to-8 octants that geometrically touch this node.
        # Each probe is offset by (dx, dy, dz) in {-1, 0}^3 on the doubled grid,
        # then halved to reach the LMAX grid where LeafLookup operates.
        touch = set()
        for dx in (-1, 0):
            for dy in (-1, 0):
                for dz in (-1, 0):
                    probe2 = ic + np.array([dx, dy, dz], np.int64)
                    if np.any(probe2 < 0) or np.any(probe2 >= G2):
                        continue
                    idx = lk.find((probe2 // 2)[None, :])[0]
                    if idx >= 0:
                        touch.add(int(idx))
        carriers = set(node_elems[n])
        non_carriers = touch - carriers
        if non_carriers:
            hanging[n] = True
            # Owner = coarsest touching non-carrier (lowest level = largest element)
            owner[n] = min(non_carriers, key=lambda e: lev[e])

    free_nodes = np.where(~hanging)[0]
    free_of = np.full(Nn, -1, np.int64)
    free_of[free_nodes] = np.arange(len(free_nodes), dtype=np.int64)

    rows, cols, vals = [], [], []

    # Identity rows for free nodes
    for n in free_nodes:
        rows.append(int(n))
        cols.append(int(free_of[n]))
        vals.append(1.0)

    # Interpolation rows for hanging nodes
    npe = p + 1
    for n in np.where(hanging)[0]:
        e = int(owner[n])
        # Reference coordinates of node n inside owner element e, mapped to [-1, 1]
        xi = 2.0 * (mesh.node_icoords[n] - anchors2[e]) / size2[e] - 1.0
        w1 = [lagrange_1d(p, xi[d])[0] for d in range(3)]
        for k in range(npe):
            for j in range(npe):
                for i in range(npe):
                    a = i + npe * j + npe * npe * k
                    w = w1[0][i] * w1[1][j] * w1[2][k]
                    if abs(w) < 1e-14:
                        continue
                    tgt = int(mesh.conn[e, a])
                    assert not hanging[tgt], (
                        "2:1 balance guarantees owner nodes are free"
                    )
                    rows.append(int(n))
                    cols.append(int(free_of[tgt]))
                    vals.append(float(w))

    T = sp.csr_matrix(
        (vals, (rows, cols)),
        shape=(Nn, len(free_nodes)),
    )
    return Constraints(T, free_nodes, hanging)
