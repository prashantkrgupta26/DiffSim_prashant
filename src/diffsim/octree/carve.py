import numpy as np
from . import morton
from .build import Octree, _make


class SphereOracle:
    def __init__(self, center, radius):
        self.c = np.asarray(center, np.float64)
        self.r = float(radius)

    def classify(self, points: np.ndarray) -> np.ndarray:
        return np.linalg.norm(points - self.c, axis=1) - self.r


def carve(tree: Octree, oracle, samples_per_axis: int = 3):
    s = samples_per_axis
    t = np.linspace(0.0, 1.0, s)
    ox, oy, oz = np.meshgrid(t, t, t, indexing="ij")
    offs = np.stack([ox.ravel(), oy.ravel(), oz.ravel()], axis=1)  # [s^3, 3]
    scale = 2.0 ** -morton.LMAX
    lo = tree.anchors() * scale
    h = tree.h()
    pts = (lo[:, None, :] + offs[None, :, :] * h[:, None, None]).reshape(-1, 3)
    sgn = oracle.classify(pts).reshape(len(tree), s**3)
    inside = sgn < 0.0
    n_in = inside.sum(axis=1)
    exterior = n_in == 0
    interior = n_in == s**3
    keep = ~exterior
    markers = np.where(interior[keep], 0, 1).astype(np.int8)
    return Octree(tree.keys[keep], tree.levels[keep]), markers
