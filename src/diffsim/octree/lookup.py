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
    Periodic axes wrap before the search (spec S11.1). O(N * lmax) host
    prototype loop — documented delta; cuFEM uses traversal."""
    def __init__(self, tree: Octree):
        self.tree = tree
        self._map = {}
        for i, (k, l) in enumerate(zip(tree.keys.tolist(), tree.levels.tolist())):
            self._map[(int(k), int(l))] = i

    def find(self, anchors: np.ndarray) -> np.ndarray:
        dim = self.tree.dim
        L = morton.lmax(dim)
        anchors = _wrap(np.asarray(anchors, np.int64).reshape(-1, dim), self.tree)
        out = np.full(len(anchors), -1, np.int64)
        G = 1 << L
        inside = np.all((anchors >= 0) & (anchors < G), axis=1)
        for i in np.where(inside)[0]:
            a = anchors[i]
            for lvl in range(L, -1, -1):
                key = int(morton.encode((a >> (L - lvl))[None, :], lvl, dim=dim)[0])
                j = self._map.get((key, lvl))
                if j is not None:
                    out[i] = j
                    break
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
