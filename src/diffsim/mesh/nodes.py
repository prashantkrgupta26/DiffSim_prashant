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

def _uniform_complete_level(tree: Octree):
    """Return the common level L if `tree` is a COMPLETE uniform grid
    (all leaves at one level L, and all (2^L)^dim cells present), else None.

    A complete uniform grid is exactly what ``build_uniform(L)`` emits: this is
    the 100M PPE hero case. On such a tree the P1/P2 node set is a structured
    lattice, so node numbering is closed-form (no O(N log N) dedup) and there
    are no hanging nodes (identity constraints). Detection is O(1) beyond a
    single level scan — cheap enough to run unconditionally.
    """
    lev = tree.levels
    if len(lev) == 0:
        return None
    L = int(lev[0])
    # all one level, and the leaf count is the full (2^L)^dim grid
    if not (lev == lev[0]).all():
        return None
    if len(tree) != (1 << (L * tree.dim)):
        return None
    return L

def _build_uniform_fast(tree: Octree, L: int, p: int) -> "Mesh":
    """Closed-form node numbering + connectivity for a complete uniform grid.

    Produces bit-for-bit the SAME Mesh as the general ``np.unique`` path:
    ``np.unique(all_ic, axis=0)`` orders nodes lexicographically with axis 0
    slowest. On the structured lattice that order is a pure arithmetic index
    (li_0 slowest), so we build ``nodes`` and ``conn`` by direct index math and
    skip the sort entirely. Periodic axes identify icoord == G2 with 0 exactly
    as the general path does (wrap BEFORE numbering).
    """
    dim = tree.dim
    size = 1 << (morton.lmax(dim) - L)          # scalar: common element size
    G2 = 2 * (1 << morton.lmax(dim))
    step2 = (2 * size) // p                      # scalar lattice step on doubled grid
    # lattice extent per axis (nodes 0..M along each axis), with periodic wrap
    M = p * (1 << L)                             # non-periodic: M+1 nodes/axis
    n_per = np.array([M if tree.periodic[ax] else M + 1 for ax in range(dim)],
                     np.int64)                   # nodes along each axis
    Nn = int(np.prod(n_per))

    # --- node icoords: structured lattice in lexicographic (axis 0 slowest) order
    grids = np.meshgrid(*[np.arange(n_per[ax], dtype=np.int64) * step2
                          for ax in range(dim)], indexing="ij")
    nodes = np.stack([g.ravel() for g in grids], axis=1)   # [Nn, dim], sorted

    # strides for the closed-form id (axis 0 slowest == np.unique axis=0 order)
    strides = np.ones(dim, np.int64)
    for ax in range(dim - 2, -1, -1):
        strides[ax] = strides[ax + 1] * n_per[ax + 1]

    # --- connectivity: id = sum_ax lidx_ax * strides[ax], lidx = ic // step2
    anchors2 = tree.anchors() * 2               # [Ne, dim]
    offs = _local_offsets(p, dim)               # [(p+1)^dim, dim]
    ic = (anchors2[:, None, :] + offs[None, :, :] * step2)  # [Ne, nbf, dim]
    for ax in range(dim):
        if tree.periodic[ax]:
            ic[..., ax] %= G2
    lidx = ic // step2                          # [Ne, nbf, dim]
    conn = (lidx * strides[None, None, :]).sum(axis=2).astype(np.int32)  # [Ne, nbf]

    coords = nodes.astype(np.float64) / G2
    boundary = np.zeros(Nn, bool)
    for ax in range(dim):
        if not tree.periodic[ax]:
            boundary |= (nodes[:, ax] == 0) | (nodes[:, ax] == G2)
    p_elem = np.full(len(tree), p, np.int8)
    per_bin = {p: np.arange(len(tree), dtype=np.int64)}
    conn_of = {p: conn}
    return Mesh(p, tree, coords, nodes, conn, boundary,
               p_elem=p_elem, bins=per_bin, conn_of=conn_of)

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
    # Fast path: complete uniform grid + single scalar order -> closed-form node
    # numbering (no O(N log N) dedup). Bit-for-bit identical to the general path
    # below (test_uniform_fast_equals_general); the 100M PPE hero case.
    if uniform and np.isscalar(p):
        L = _uniform_complete_level(tree)
        if L is not None:
            return _build_uniform_fast(tree, L, int(p))
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
