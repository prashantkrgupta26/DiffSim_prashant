import numpy as np
from . import morton
from .build import Octree

# Face order matches spec §3 BoundaryTypes.WALL: X_MINUS..Z_PLUS
FACE_OFFSETS = np.array(
    [[-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0], [0, 0, -1], [0, 0, 1]], np.int64
)

class LeafLookup:
    """Containment lookup: LMAX-grid anchor -> leaf index, via truncated keys."""
    def __init__(self, tree: Octree):
        self.tree = tree
        self._map = {}
        for i, (k, l) in enumerate(zip(tree.keys.tolist(), tree.levels.tolist())):
            self._map[(int(k), int(l))] = i

    def find(self, anchors: np.ndarray) -> np.ndarray:
        anchors = np.asarray(anchors, np.int64).reshape(-1, 3)
        out = np.full(len(anchors), -1, np.int64)
        G = 1 << morton.LMAX
        inside = np.all((anchors >= 0) & (anchors < G), axis=1)
        for i in np.where(inside)[0]:
            a = anchors[i]
            for lvl in range(morton.LMAX, -1, -1):
                xyz = a >> (morton.LMAX - lvl)
                key = int(morton.encode(xyz[None, :], lvl)[0])
                j = self._map.get((key, lvl))
                if j is not None:
                    out[i] = j
                    break
        return out

def face_neighbors(tree: Octree) -> list:
    lk = LeafLookup(tree)
    anchors = tree.anchors()
    size = (1 << (morton.LMAX - tree.levels.astype(np.int64)))[:, None]
    center_off = size // 2
    out = []
    for f in range(6):
        off = FACE_OFFSETS[f]
        # probe point: just outside the face, at the face center
        probe = anchors + center_off
        probe = probe + off * (center_off + 1)  # step past the face plane
        idx = lk.find(probe)
        idx[idx == np.arange(len(tree))] = -1   # safety: never self
        out.append(idx)
    return out
