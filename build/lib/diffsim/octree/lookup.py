import numpy as np
from . import morton
from .build import Octree

def face_offsets(dim: int) -> np.ndarray:
    """Face order: axis0-, axis0+, axis1-, axis1+, ... (3D == BoundaryTypes.WALL)."""
    out = []
    for ax in range(dim):
        for sgn in (-1, 1):
            off = [0] * dim
            off[ax] = sgn
            out.append(off)
    return np.array(out, np.int64)

FACE_OFFSETS = face_offsets(3)   # back-compat constant (3D, matches M0's BoundaryTypes.WALL)

def _wrap(anchors: np.ndarray, tree: Octree) -> np.ndarray:
    """Wrap probe coords modulo the grid on periodic axes; leave others."""
    G = 1 << morton.lmax(tree.dim)
    a = anchors.copy()
    for ax in range(tree.dim):
        if tree.periodic[ax]:
            a[:, ax] %= G
    return a

class LeafLookup:
    """Containment lookup: lmax-grid anchor -> leaf index via truncated keys.
    Periodic axes wrap before the search (spec S11.1).

    VECTORIZED (2026-07-05 campaign; Dendro-KT exploration): per-level
    SORTED key arrays + np.searchsorted, sweeping levels finest-to-coarsest
    over the still-unresolved points — semantics identical to the original
    per-point dict walk (equivalence-gated), cost = (#levels-present) x
    (batched encode + searchsorted) instead of a Python loop per point.
    The original per-point walk was the MEASURED wall behind
    build_constraints (m0.5 finding / P1's 222 s): batching the callers
    alone gained nothing (1.0x) because the loop lived HERE."""

    def __init__(self, tree: Octree):
        self.tree = tree
        keys = tree.keys.astype(np.int64)
        levels = tree.levels.astype(np.int64)
        self._levels_present = np.unique(levels)[::-1]      # finest first
        self._by_level = {}
        for lvl in self._levels_present:
            m = levels == lvl
            k = keys[m]
            idx = np.where(m)[0].astype(np.int64)
            order = np.argsort(k, kind="stable")
            self._by_level[int(lvl)] = (k[order], idx[order])

    def find(self, anchors: np.ndarray) -> np.ndarray:
        dim = self.tree.dim
        L = morton.lmax(dim)
        anchors = _wrap(np.asarray(anchors, np.int64).reshape(-1, dim),
                        self.tree)
        n = len(anchors)
        out = np.full(n, -1, np.int64)
        G = 1 << L
        unresolved = np.all((anchors >= 0) & (anchors < G), axis=1)
        for lvl in self._levels_present:
            if not unresolved.any():
                break
            ii = np.where(unresolved)[0]
            trunc = anchors[ii] >> (L - int(lvl))
            k = morton.encode(trunc, int(lvl), dim=dim).astype(np.int64)
            sk, sidx = self._by_level[int(lvl)]
            pos = np.searchsorted(sk, k)
            pos_c = np.minimum(pos, len(sk) - 1)
            hit = (len(sk) > 0) & (sk[pos_c] == k)
            out[ii[hit]] = sidx[pos_c[hit]]
            unresolved[ii[hit]] = False
        return out

def face_neighbors(tree: Octree) -> list:
    lk = LeafLookup(tree)
    anchors = tree.anchors()
    size = (1 << (morton.lmax(tree.dim) - tree.levels.astype(np.int64)))[:, None]
    center_off = size // 2
    out = []
    for off in face_offsets(tree.dim):
        probe = anchors + center_off + off * (center_off + 1)
        idx = lk.find(probe)
        idx[idx == np.arange(len(tree))] = -1
        out.append(idx)
    return out
