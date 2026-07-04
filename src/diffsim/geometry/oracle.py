"""GeometryOracle contract for SDF-family backends (spec S4.1).

The oracle owns a scalar field psi(x; theta) implemented in torch FP64:
psi < 0 inside the shape. Which side of psi = 0 is the computational domain
is chosen downstream by the `domain` flag (spec: M1a plan, Global
Constraints) — the oracle itself is domain-agnostic.

Contract (spec S4.1):
    classify(points[N])        -> psi[N]           (sign field, FP64 numpy)
    distance_vector(points[N]) -> d[N,dim], n_grad[N,dim], ok[N]
                                  (d = closest point - x; n_grad = grad psi /
                                   |grad psi| at the closest point, i.e. the
                                   direction of increasing psi — the caller
                                   orients it out of the domain)
    velocity(points[N], t)     -> zeros            (static geometry, M1)

Layout/precision: numpy-facing methods take/return C-contiguous FP64 arrays
[N, dim]; torch work happens in float64 always (FP64 spine, spec S2.2.4).
Differentiation: `psi` must be autograd-connected to every tensor in
`params`; the adjoint driver re-runs the torch chain in backward.
"""
import numpy as np
import torch


class SDFOracle:
    """Base class for SDF-family geometry backends (CSG, GridSDF, NeuralSDF)."""

    near_eikonal: bool = False   # True => |grad psi| ~ 1 verified; eikonal shortcut valid
    dim: int

    @property
    def params(self) -> list:
        """Leaf torch tensors psi depends on (the differentiable parameters)."""
        raise NotImplementedError

    def psi(self, x: torch.Tensor) -> torch.Tensor:
        """Signed field at x [N, dim] (torch float64) -> [N]."""
        raise NotImplementedError

    # ---- numpy bridge (forward pipeline; detached, FP64) -------------------
    def classify(self, pts: np.ndarray) -> np.ndarray:
        x = torch.as_tensor(np.ascontiguousarray(pts, np.float64))
        with torch.no_grad():
            return self.psi(x).numpy()

    def distance_vector(self, pts: np.ndarray):
        from .project import distance_numpy
        return distance_numpy(self, pts)

    def velocity(self, pts: np.ndarray, t: float = 0.0) -> np.ndarray:
        return np.zeros_like(np.asarray(pts, np.float64))
