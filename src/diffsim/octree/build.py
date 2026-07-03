from dataclasses import dataclass
import numpy as np
from . import morton

@dataclass(frozen=True)
class Octree:
    keys: np.ndarray     # uint64, sorted
    levels: np.ndarray   # uint8

    def __len__(self):
        return len(self.keys)

    def anchors(self):
        return morton.anchors(self.keys)

    def h(self):
        return 2.0 ** (-self.levels.astype(np.float64))

    def centers(self):
        scale = 2.0 ** (-morton.LMAX)
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

def _make(keys, levels) -> Octree:
    k, l = sort_unique(np.asarray(keys, np.uint64), np.asarray(levels, np.uint8))
    return Octree(k, l)

def build_uniform(level: int) -> Octree:
    n = 1 << level
    g = np.arange(n, dtype=np.int64)
    X, Y, Z = np.meshgrid(g, g, g, indexing="ij")
    xyz = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    keys = morton.encode(xyz, level)
    return _make(keys, np.full(len(keys), level, np.uint8))

def refine_elements(tree: Octree, mask: np.ndarray) -> Octree:
    keep_k, keep_l = tree.keys[~mask], tree.levels[~mask]
    ck, cl = morton.children(tree.keys[mask], tree.levels[mask])
    keys = np.concatenate([keep_k, ck.ravel()])
    levels = np.concatenate([keep_l, np.repeat(cl, 8)])
    return _make(keys, levels)

def build_adaptive(refine_fn, max_level: int) -> Octree:
    tree = build_uniform(0)
    while True:
        can = tree.levels < max_level
        want = refine_fn(tree.centers(), tree.h()[:, None])
        mask = can & want
        if not mask.any():
            return tree
        tree = refine_elements(tree, mask)
