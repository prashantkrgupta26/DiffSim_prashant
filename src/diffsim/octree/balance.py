import numpy as np
from itertools import product
from . import morton
from .build import Octree, refine_elements
from .lookup import LeafLookup

def _nbr_offsets(dim: int) -> np.ndarray:
    return np.array([o for o in product((-1, 0, 1), repeat=dim) if any(o)], np.int64)

def _neighbor_levels(tree: Octree):
    lk = LeafLookup(tree)          # find() wraps periodic axes internally
    anchors = tree.anchors()
    size = (1 << (morton.lmax(tree.dim) - tree.levels.astype(np.int64)))[:, None]
    center = anchors + size // 2
    offs = _nbr_offsets(tree.dim)
    nbr_idx = np.empty((len(tree), len(offs)), np.int64)
    for j, off in enumerate(offs):
        nbr_idx[:, j] = lk.find(center + off * size)
    return nbr_idx

def check_balance(tree: Octree) -> bool:
    nbr = _neighbor_levels(tree)
    lev = tree.levels.astype(np.int64)
    for j in range(nbr.shape[1]):
        ok = nbr[:, j] >= 0
        if np.any(np.abs(lev[ok] - lev[nbr[ok, j]]) > 1):
            return False
    return True

def balance2to1(tree: Octree) -> Octree:
    while True:
        nbr = _neighbor_levels(tree)
        lev = tree.levels.astype(np.int64)
        to_refine = np.zeros(len(tree), bool)
        for j in range(nbr.shape[1]):
            ok = nbr[:, j] >= 0
            viol = ok.copy()
            viol[ok] = lev[ok] - lev[nbr[ok, j]] > 1
            to_refine[nbr[viol, j]] = True
        if not to_refine.any():
            return tree
        tree = refine_elements(tree, to_refine)
