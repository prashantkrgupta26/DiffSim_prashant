from dataclasses import dataclass, field
import numpy as np
from . import morton

@dataclass(frozen=True)
class Octree:
    """Sorted leaf array of a (possibly incomplete) k-D tree, k = dim.

    keys: uint64 interleaved Morton keys at lmax(dim) anchor depth.
    levels: uint8 per-leaf depth. periodic: per-axis wrap flags — consumed
    by lookup/balance/nodes; the key encoding itself is periodicity-agnostic.
    Invariant: (keys, levels) lexicographically sorted and duplicate-free.
    """
    keys: np.ndarray
    levels: np.ndarray
    dim: int = 3
    periodic: tuple = None

    def __post_init__(self):
        if self.periodic is None:
            object.__setattr__(self, "periodic", (False,) * self.dim)
        assert len(self.periodic) == self.dim

    def __len__(self):
        return len(self.keys)

    def anchors(self):
        return morton.anchors(self.keys, dim=self.dim)

    def h(self):
        return 2.0 ** (-self.levels.astype(np.float64))

    def centers(self):
        scale = 2.0 ** (-morton.lmax(self.dim))
        half = 0.5 * self.h()
        return self.anchors() * scale + half[:, None]

def sort_unique(keys, levels):
    if len(keys) == 0:
        return keys, levels
    order = np.lexsort((levels, keys))
    k, l = keys[order], levels[order]
    keep = np.ones(len(k), bool)
    keep[1:] = (k[1:] != k[:-1]) | (l[1:] != l[:-1])
    return k[keep], l[keep]

def _make(keys, levels, dim, periodic) -> Octree:
    k, l = sort_unique(np.asarray(keys, np.uint64), np.asarray(levels, np.uint8))
    return Octree(k, l, dim=dim, periodic=periodic)

def build_uniform(level: int, dim: int = 3, periodic=None) -> Octree:
    n = 1 << level
    grids = np.meshgrid(*([np.arange(n, dtype=np.int64)] * dim), indexing="ij")
    xyz = np.stack([g.ravel() for g in grids], axis=1)
    keys = morton.encode(xyz, level, dim=dim)
    return _make(keys, np.full(len(keys), level, np.uint8), dim,
                 tuple(periodic) if periodic is not None else (False,) * dim)

def refine_elements(tree: Octree, mask: np.ndarray) -> Octree:
    keep_k, keep_l = tree.keys[~mask], tree.levels[~mask]
    ck, cl = morton.children(tree.keys[mask], tree.levels[mask], dim=tree.dim)
    keys = np.concatenate([keep_k, ck.ravel()])
    levels = np.concatenate([keep_l, np.repeat(cl, 1 << tree.dim)])
    return _make(keys, levels, tree.dim, tree.periodic)

def coarsen_elements(tree: Octree, mask: np.ndarray) -> Octree:
    """Inverse of :func:`refine_elements`: replace complete sibling groups by
    their parent. A parent is coarsened iff ALL 2**dim of its children are
    present as leaves AND every one is marked in ``mask``. Marked leaves whose
    sibling group is incomplete or partially marked are kept unchanged, as are
    level-0 leaves (which have no parent). One level of coarsening per call;
    loop for multi-level. Solution-adaptive AMR marks a region for coarsening
    only when its running-error indicator is low across the whole sibling
    group, so the all-siblings-marked rule is exactly the physical condition.
    """
    mask = np.asarray(mask, bool)
    dim = tree.dim
    nch = 1 << dim
    lev = tree.levels.astype(np.int64)
    can = mask & (lev > 0)
    idx = np.where(lev > 0)[0]
    if len(idx) == 0:
        return tree
    pk, pl = morton.parent(tree.keys[idx], tree.levels[idx], dim=dim)
    # group siblings by (parent key, parent level)
    order = np.lexsort((pl, pk))
    io, pko, plo = idx[order], pk[order], pl[order]
    grp_start = np.ones(len(io), bool)
    grp_start[1:] = (pko[1:] != pko[:-1]) | (plo[1:] != plo[:-1])
    gid = np.cumsum(grp_start) - 1
    ng = gid[-1] + 1
    size = np.bincount(gid, minlength=ng)
    all_marked = np.bincount(gid, weights=can[io].astype(np.int64),
                             minlength=ng)
    coarsen_grp = (size == nch) & (all_marked == nch)
    drop = np.zeros(len(tree), bool)
    drop[io] = coarsen_grp[gid]
    # emit surviving leaves + one parent per coarsened group
    par_first = grp_start & coarsen_grp[gid]
    new_pk = pko[par_first]
    new_pl = plo[par_first].astype(np.uint8)
    keys = np.concatenate([tree.keys[~drop], new_pk])
    levels = np.concatenate([tree.levels[~drop], new_pl])
    return _make(keys, levels, tree.dim, tree.periodic)


def build_adaptive(refine_fn, max_level: int, dim: int = 3, periodic=None) -> Octree:
    tree = build_uniform(0, dim=dim, periodic=periodic)
    while True:
        can = tree.levels < max_level
        want = refine_fn(tree.centers(), tree.h()[:, None])  # h broadcast-ready [N,1]
        mask = can & want
        if not mask.any():
            return tree
        tree = refine_elements(tree, mask)
