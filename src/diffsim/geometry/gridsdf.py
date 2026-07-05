"""GridSDF geometry backend (spec S4.1 backend table, row 4): discrete SDF
values on a uniform corner grid over [lo, hi]^dim, multilinear interpolation.

The voxel VALUES are the differentiable parameter (= level-set shape
optimization). Interpolation is explicit gather + lerp (autograd-clean exact
FP64; not grid_sample). psi is C0 across cell faces and only approximately
eikonal => near_eikonal = False, Newton closest-point projection (the
admissibility report quantifies eps_inf ~ Dx^2).
"""
from itertools import product as iproduct

import numpy as np
import torch

from .oracle import SDFOracle


class GridSDF(SDFOracle):
    near_eikonal = False

    def __init__(self, values: torch.Tensor, lo: float = 0.0, hi: float = 1.0):
        values = values.to(torch.float64)
        self.values = values
        self.dim = values.ndim
        self.n = values.shape[0] - 1
        if any(s != self.n + 1 for s in values.shape):
            raise ValueError("values must be a cubic (n+1)^dim corner grid")
        self.lo, self.hi = float(lo), float(hi)

    @property
    def params(self):
        return [self.values]

    def psi(self, x: torch.Tensor) -> torch.Tensor:
        n = self.n
        t = (x - self.lo) / (self.hi - self.lo) * n
        t = torch.clamp(t, 0.0, float(n) - 1e-12)
        i0 = torch.clamp(t.floor().long(), 0, n - 1)         # [N, dim]
        frac = t - i0.to(torch.float64)                      # [N, dim]
        out = torch.zeros(x.shape[0], dtype=torch.float64)
        for corner in iproduct((0, 1), repeat=self.dim):
            idx = tuple(i0[:, d] + corner[d] for d in range(self.dim))
            w = torch.ones(x.shape[0], dtype=torch.float64)
            for d in range(self.dim):
                w = w * (frac[:, d] if corner[d] else 1.0 - frac[:, d])
            out = out + w * self.values[idx]
        return out

    @classmethod
    def from_oracle(cls, oracle, n: int, lo: float = 0.0, hi: float = 1.0):
        """Sample an existing oracle's psi on the (n+1)^dim corner grid."""
        axes = [np.linspace(lo, hi, n + 1)] * oracle.dim
        grids = np.meshgrid(*axes, indexing="ij")
        pts = np.stack([g.ravel() for g in grids], axis=1)
        vals = oracle.classify(pts).reshape(*(n + 1,) * oracle.dim)
        return cls(torch.tensor(vals, dtype=torch.float64), lo, hi)
