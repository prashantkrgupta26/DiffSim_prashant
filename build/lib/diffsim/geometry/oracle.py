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


def admissibility(oracle: SDFOracle, band_pts: np.ndarray) -> dict:
    """Runtime admissibility diagnostics (spec S4.1), evaluated on a band of
    query points near the boundary:

      eps_inf            max |psi(y*)| over converged projections (residual
                         geometry error — the epsilon_inf estimate)
      c0                 min |grad psi| over the band (c0_hat)
      d_hausdorff_bound  eps_inf / min(1, c0)
      newton_ok_frac     fraction of points whose projection converged
    """
    from .project import _newton_iterate, _psi_grad

    x = torch.as_tensor(np.ascontiguousarray(band_pts, np.float64))
    y, _s, ok = _newton_iterate(oracle, x)
    psi_y, _ = _psi_grad(oracle, y)
    _, g_band = _psi_grad(oracle, x)
    c0 = float(g_band.norm(dim=1).min())
    eps = float(psi_y[ok].abs().max()) if bool(ok.any()) else float("inf")
    return {
        "eps_inf": eps,
        "c0": c0,
        "d_hausdorff_bound": eps / min(1.0, c0) if c0 > 0 else float("inf"),
        "newton_ok_frac": float(ok.to(torch.float64).mean()),
    }


def warn_refinement_plateau(diag: dict, h: float, p: int) -> bool:
    """Warn when geometry error bounds what refinement can buy: the
    eps_inf ~ h^(p+1) rule (spec S4.1). Returns True if the warning fired."""
    import warnings

    if diag["d_hausdorff_bound"] > h ** (p + 1):
        warnings.warn(
            f"geometry-limited refinement plateau: Hausdorff bound "
            f"{diag['d_hausdorff_bound']:.2e} exceeds h^(p+1) = {h ** (p + 1):.2e}; "
            "refining the mesh further cannot reduce the error")
        return True
    return False
