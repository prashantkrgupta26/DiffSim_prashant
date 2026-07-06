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
    """Vectorized constraint construction (campaign 2026-07-05; Dendro-KT
    exploration): the classification loop is replaced by ONE batched
    LeafLookup.find over all (node, probe) pairs plus sort/segment set
    algebra — no per-node Python. Measured against the reference
    implementation bit-for-bit (test_constraints_equivalence); the P1
    bottleneck (222 s at 66k DOFs) drops to the sort cost.

    Semantics identical to _build_constraints_reference (kept below as the
    equivalence oracle): hanging iff some geometrically-touching leaf does
    not carry the node; owner = min non-carrier by (level, p) — the h- then
    p-transition minimum rule; interpolation rows with transitive chain
    resolution."""
    tree, dim = mesh.tree, mesh.dim
    lk = LeafLookup(tree)
    Nn = len(mesh.node_coords)
    lev = tree.levels.astype(np.int64)
    size2 = 2 * (1 << (morton.lmax(dim) - lev))
    anchors2 = tree.anchors() * 2
    p_elem = np.asarray(mesh.p_elem)
    G2 = 2 * (1 << morton.lmax(dim))

    # ---- 1. ALL probes in ONE batched find -----------------------------
    probes = np.array(list(iproduct((-1, 0), repeat=dim)), np.int64)
    npr = len(probes)
    pts = (mesh.node_icoords[:, None, :] + probes[None, :, :]).reshape(
        -1, dim) // 2                                   # [Nn*2^dim, dim]
    touch_e = lk.find(pts).reshape(Nn, npr)             # -1 = outside

    # ---- 2. carrier pairs from connectivity (vectorized) ---------------
    NE = len(tree)
    car_keys = []
    for pv, eids in mesh.bins.items():
        cn = mesh.conn_of[pv]                            # [ne, nbf]
        e_col = np.asarray(eids, np.int64)[:, None]
        car_keys.append((cn.astype(np.int64) * NE
                         + e_col).ravel())               # key = n*NE + e
    car_keys = np.unique(np.concatenate(car_keys))

    # ---- 3. non-carrier touches -> hanging + owner ----------------------
    node_id = np.repeat(np.arange(Nn, dtype=np.int64), npr)
    ev = touch_e.ravel()
    valid = ev >= 0
    keys = node_id[valid] * NE + ev[valid]
    keys = np.unique(keys)                               # dedup (n, e)
    non_car = keys[~np.isin(keys, car_keys)]
    if len(non_car):
        n_nc = non_car // NE
        e_nc = non_car % NE
        # owner: lexicographic min by (level, p) within each node segment
        order = np.lexsort((p_elem[e_nc], lev[e_nc], n_nc))
        n_s, e_s = n_nc[order], e_nc[order]
        first = np.ones(len(n_s), bool)
        first[1:] = n_s[1:] != n_s[:-1]
        hang_nodes = n_s[first]
        owners = e_s[first]
    else:
        hang_nodes = np.empty(0, np.int64)
        owners = np.empty(0, np.int64)
    hanging = np.zeros(Nn, bool)
    hanging[hang_nodes] = True
    owner = np.full(Nn, -1, np.int64)
    owner[hang_nodes] = owners

    free_nodes = np.where(~hanging)[0]
    free_of = np.full(Nn, -1, np.int64)
    free_of[free_nodes] = np.arange(len(free_nodes), dtype=np.int64)

    # ---- 4. interpolation rows (per hanging node; boundary-sized) ------
    elem_conn_row = {}
    for pv, eids in mesh.bins.items():
        for r, e in enumerate(eids):
            elem_conn_row[int(e)] = (pv, r)

    rows = list(free_nodes)
    cols = list(free_of[free_nodes])
    vals = [1.0] * len(free_nodes)

    raw: dict[int, dict[int, float]] = {}
    for n in hang_nodes:
        e = int(owner[n])
        pv, r = elem_conn_row[e]
        offs = _local_offsets(pv, dim)
        d_ic = mesh.node_icoords[n] - anchors2[e]
        for ax in range(dim):
            if tree.periodic[ax]:
                if d_ic[ax] > size2[e]:
                    d_ic[ax] -= G2
                elif d_ic[ax] < 0:
                    d_ic[ax] += G2
        xi = 2.0 * d_ic / size2[e] - 1.0
        w1 = [lagrange_1d(pv, xi[d])[0] for d in range(dim)]
        row: dict[int, float] = {}
        for a in range(len(offs)):
            w = 1.0
            for d in range(dim):
                w *= w1[d][offs[a, d]]
            if abs(w) < 1e-14:
                continue
            tgt = int(mesh.conn_of[pv][r, a])
            row[tgt] = row.get(tgt, 0.0) + w
        raw[int(n)] = row

    MAX_ITERS = 10
    for _it in range(MAX_ITERS):
        changed = False
        new_raw: dict[int, dict[int, float]] = {}
        for n, row in raw.items():
            new_row: dict[int, float] = {}
            for tgt, w in row.items():
                if hanging[tgt]:
                    for tgt2, w2 in raw[tgt].items():
                        new_row[tgt2] = new_row.get(tgt2, 0.0) + w * w2
                    changed = True
                else:
                    new_row[tgt] = new_row.get(tgt, 0.0) + w
            new_raw[n] = new_row
        raw = new_raw
        if not changed:
            break
    else:
        raise RuntimeError("constraint chain resolution did not converge")

    for n, row in raw.items():
        for tgt, w in row.items():
            rows.append(int(n))
            cols.append(int(free_of[tgt]))
            vals.append(float(w))

    T = sp.csr_matrix((vals, (rows, cols)), shape=(Nn, len(free_nodes)))
    return Constraints(T, free_nodes, hanging)


def _build_constraints_reference(mesh: Mesh) -> Constraints:
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

    # --- Interpolation rows for hanging nodes with transitive chain resolution ---
    #
    # In mixed (p1+p2) meshes a p-hanging node's owner (the same-level p1
    # element) can itself have h-hanging corner nodes if a coarser diagonal
    # neighbour touches that corner (2:1 balance allows this off-face).  The
    # chain is always short (≤ 2 for p1/p2 with one-knob + 2:1 balance), but
    # rather than assert the chain is length-1 we resolve it to free nodes by
    # fixed-point substitution.  This is mathematically exact: substituting an
    # h-hanging corner's own interpolation row preserves partition-of-unity
    # (weight sums compose to 1) and polynomial reproduction (pointwise
    # exactness at each level).
    #
    # Step 1: build raw rows — targets may still be hanging.
    raw: dict[int, dict[int, float]] = {}   # hanging node -> {all_node: weight}

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
        row: dict[int, float] = {}
        for a in range(len(offs)):
            w = 1.0
            for d in range(dim):
                w *= w1[d][offs[a, d]]
            if abs(w) < 1e-14:
                continue
            tgt = int(mesh.conn_of[pv][r, a])
            row[tgt] = row.get(tgt, 0.0) + w
        raw[int(n)] = row

    # Step 2: resolve chains — substitute any hanging target with its own raw
    # row (weight-multiplied) until all targets are free nodes.  Terminates in
    # at most ceil(log2(max_chain_length)) ≈ 3 passes for real meshes.
    MAX_ITERS = 10
    for _it in range(MAX_ITERS):
        changed = False
        new_raw: dict[int, dict[int, float]] = {}
        for n, row in raw.items():
            new_row: dict[int, float] = {}
            for tgt, w in row.items():
                if hanging[tgt]:
                    # tgt is itself hanging — substitute its raw row
                    for tgt2, w2 in raw[tgt].items():
                        new_row[tgt2] = new_row.get(tgt2, 0.0) + w * w2
                    changed = True
                else:
                    new_row[tgt] = new_row.get(tgt, 0.0) + w
            new_raw[n] = new_row
        raw = new_raw
        if not changed:
            break
    else:
        raise RuntimeError(
            f"constraint chain resolution did not converge in {MAX_ITERS} iterations"
        )

    # Step 3: emit resolved rows into the sparse T matrix
    for n, row in raw.items():
        for tgt, w in row.items():
            assert not hanging[tgt], (
                f"node {tgt} still hanging after chain resolution (node {n})"
            )
            rows.append(int(n))
            cols.append(int(free_of[tgt]))
            vals.append(float(w))

    T = sp.csr_matrix(
        (vals, (rows, cols)),
        shape=(Nn, len(free_nodes)),
    )
    return Constraints(T, free_nodes, hanging)
