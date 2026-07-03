from dataclasses import dataclass
from itertools import product as iproduct
import numpy as np
import scipy.sparse as sp
from ..octree import morton
from ..octree.lookup import LeafLookup
from .nodes import Mesh, _local_offsets
from .basis import lagrange_1d


@dataclass(frozen=True)
class Constraints:
    T: sp.csr_matrix        # Nn × Nfree; maps free-node values to all-node values
    free_nodes: np.ndarray  # int64[Nfree]
    hanging: np.ndarray     # bool[Nn]


def build_constraints(mesh: Mesh) -> Constraints:
    tree, dim = mesh.tree, mesh.dim
    lk = LeafLookup(tree)
    Nn = len(mesh.node_coords)
    lev = tree.levels.astype(np.int64)
    size2 = 2 * (1 << (morton.lmax(dim) - lev))   # element size on doubled grid, per element
    anchors2 = tree.anchors() * 2                   # element anchors on doubled grid

    # Build node -> list-of-elements incidence over all bins;
    # per-element (pv, local row) handle for weight computation.
    # Works for both uniform (single bin) and mixed meshes.
    node_elems = [[] for _ in range(Nn)]
    elem_conn_row = {}                       # global elem id -> (pv, local row)
    for pv, eids in mesh.bins.items():
        cn = mesh.conn_of[pv]
        for r, e in enumerate(eids):
            elem_conn_row[int(e)] = (pv, r)
            for a in cn[r]:
                node_elems[int(a)].append(int(e))

    p_elem = mesh.p_elem

    # Doubled-grid domain size (exclusive upper bound)
    G2 = 2 * (1 << morton.lmax(dim))

    hanging = np.zeros(Nn, bool)
    owner = np.full(Nn, -1, np.int64)

    probes = np.array(list(iproduct((-1, 0), repeat=dim)), np.int64)
    for n in range(Nn):
        ic = mesh.node_icoords[n]
        # Probe the up-to-2^dim octants that geometrically touch this node.
        # Each probe is offset by dp in {-1, 0}^dim on the doubled grid,
        # then halved to reach the lmax grid where LeafLookup operates.
        # LeafLookup.find handles periodic wrapping and returns -1 for
        # out-of-range probes on non-periodic axes.
        touch = set()
        for dp in probes:
            probe2 = ic + dp
            idx = lk.find((probe2 // 2)[None, :])[0]
            if idx >= 0:
                touch.add(int(idx))
        carriers = set(node_elems[n])
        non_carriers = touch - carriers
        if non_carriers:
            hanging[n] = True
            # Owner = min over non-carriers by (level, p_elem) lexicographic:
            # coarsest first (h-transition); at equal level, lower-order side
            # (p-transition, spec §13.2 minimum rule).
            owner[n] = min(non_carriers, key=lambda e: (lev[e], int(p_elem[e])))

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
    for n in np.where(hanging)[0]:
        e = int(owner[n])
        pv, r = elem_conn_row[e]
        offs = _local_offsets(pv, dim)
        # Reference coordinates of node n inside owner element e, mapped to [-1, 1].
        # For periodic axes, shift d_ic into the owner's box to handle seam nodes.
        d_ic = mesh.node_icoords[n] - anchors2[e]
        for ax in range(dim):
            if tree.periodic[ax]:
                if d_ic[ax] > size2[e]:
                    d_ic[ax] -= G2
                elif d_ic[ax] < 0:
                    d_ic[ax] += G2
        xi = 2.0 * d_ic / size2[e] - 1.0
        w1 = [lagrange_1d(pv, xi[d])[0] for d in range(dim)]
        for a in range(len(offs)):
            w = 1.0
            for d in range(dim):
                w *= w1[d][offs[a, d]]
            if abs(w) < 1e-14:
                continue
            tgt = int(mesh.conn_of[pv][r, a])
            assert not hanging[tgt], (
                "owner nodes must be free (one-knob + 2:1 balance)"
            )
            rows.append(int(n))
            cols.append(int(free_of[tgt]))
            vals.append(float(w))

    T = sp.csr_matrix(
        (vals, (rows, cols)),
        shape=(Nn, len(free_nodes)),
    )
    return Constraints(T, free_nodes, hanging)
