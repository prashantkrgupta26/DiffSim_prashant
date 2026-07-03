import numpy as np
from itertools import product
from . import morton
from .build import Octree, refine_elements
from .lookup import LeafLookup

_NBR_OFFSETS = np.array([o for o in product((-1, 0, 1), repeat=3) if o != (0, 0, 0)],
                        np.int64)  # 26 neighbors

def _neighbor_levels(tree: Octree):
    """For each leaf, the leaf index of each of its 26 same-size neighbor probes."""
    lk = LeafLookup(tree)
    anchors = tree.anchors()
    size = (1 << (morton.LMAX - tree.levels.astype(np.int64)))[:, None]
    center = anchors + size // 2
    nbr_idx = np.empty((len(tree), 26), np.int64)
    for j, off in enumerate(_NBR_OFFSETS):
        probe = center + off * size          # lands inside face/edge/vertex neighbor
        nbr_idx[:, j] = lk.find(probe)
    return nbr_idx

def check_balance(tree: Octree) -> bool:
    nbr = _neighbor_levels(tree)
    lev = tree.levels.astype(np.int64)
    for j in range(26):
        ok = nbr[:, j] >= 0
        if np.any(np.abs(lev[ok] - lev[nbr[ok, j]]) > 1):
            return False
    return True

def balance2to1(tree: Octree) -> Octree:
    while True:
        nbr = _neighbor_levels(tree)
        lev = tree.levels.astype(np.int64)
        to_refine = np.zeros(len(tree), bool)
        for j in range(26):
            ok = nbr[:, j] >= 0
            # neighbor is more than one level coarser than me -> neighbor must refine
            viol = ok.copy()
            viol[ok] = lev[ok] - lev[nbr[ok, j]] > 1
            to_refine[nbr[viol, j]] = True
        if not to_refine.any():
            return tree
        tree = refine_elements(tree, to_refine)
