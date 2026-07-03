import numpy as np
from . import morton
from .build import Octree


class SphereOracle:
    def __init__(self, center, radius):
        self.c = np.asarray(center, np.float64)
        self.r = float(radius)

    def classify(self, points: np.ndarray) -> np.ndarray:
        return np.linalg.norm(points - self.c, axis=1) - self.r


def carve(tree: Octree, oracle, samples_per_axis: int = 3):
    """Classify leaves against the oracle on an s^dim sample lattice and
    drop EXTERIOR leaves. Markers: 0 = INTERIOR, 1 = INTERCEPTED (spec S4.2;
    Gauss-point lambda-criterion classification arrives with SBM in M1)."""
    s, dim = samples_per_axis, tree.dim
    t = np.linspace(0.0, 1.0, s)
    grids = np.meshgrid(*([t] * dim), indexing="ij")
    offs = np.stack([g.ravel() for g in grids], axis=1)        # [s^dim, dim]
    scale = 2.0 ** -morton.lmax(dim)
    lo = tree.anchors() * scale
    h = tree.h()
    pts = (lo[:, None, :] + offs[None, :, :] * h[:, None, None]).reshape(-1, dim)
    sgn = oracle.classify(pts).reshape(len(tree), s**dim)
    n_in = (sgn < 0.0).sum(axis=1)
    keep = n_in > 0
    markers = np.where(n_in[keep] == s**dim, 0, 1).astype(np.int8)
    return Octree(tree.keys[keep], tree.levels[keep], dim=dim,
                  periodic=tree.periodic), markers
