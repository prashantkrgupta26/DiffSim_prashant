"""AnalyticINRProxy — the NeuralSDF test double (spec N2, 2026-07-06).

Mirrors the GENIE structure of a linear-last-layer INR exactly:

    psi(x; alpha) = psi0(x) + sum_k alpha_k b_k(x)

with psi0 an analytic SDF (any existing oracle) and b_k fixed analytic
C-infinity features (Gaussian bumps centered on a shell around the base
surface — the "penultimate features" of the proxy). The geometry is
AFFINE in the design variables alpha, dpsi/dalpha_k = b_k(x) closed-form.
DiffSim never trains INRs (program boundary N0); real checkpoints
implement the same interface via ProvidedINROracle later.

Edits do not preserve eikonality — near_eikonal stays False and the
generic Newton+IFT projector handles the level-set field (the
admissibility audit is the safety contract)."""
import numpy as np
import torch

from .oracle import SDFOracle


class AnalyticINRProxy(SDFOracle):
    near_eikonal = False

    def __init__(self, base: SDFOracle, n_modes: int = 8,
                 bump_sigma_rel: float = 0.6, shell_radius: float = None,
                 center=None, seed: int = 0):
        """base: analytic oracle providing psi0 (e.g. Sphere).
        Features: n_modes Gaussian bumps with centers equispaced on the
        shell of radius shell_radius about `center` (defaults: the base
        Sphere's center/radius when available), width bump_sigma_rel *
        shell_radius."""
        self.base = base
        self.dim = base.dim
        if center is None:
            center = base.center.detach().numpy() if hasattr(
                base, "center") else np.zeros(self.dim) + 0.5
        if shell_radius is None:
            shell_radius = (float(base.radius) if hasattr(base, "radius")
                            else 0.25)
        self._centers = self._shell_points(np.asarray(center, np.float64),
                                           shell_radius, n_modes,
                                           self.dim, seed)
        self._sigma = bump_sigma_rel * shell_radius
        self.alpha = torch.zeros(n_modes, dtype=torch.float64)

    @staticmethod
    def _shell_points(c, r, n, dim, seed):
        if dim == 2:
            th = 2 * np.pi * np.arange(n) / n
            pts = c + r * np.stack([np.cos(th), np.sin(th)], axis=1)
        else:
            # Fibonacci sphere: near-uniform, deterministic
            i = np.arange(n) + 0.5
            phi = np.arccos(1 - 2 * i / n)
            th = np.pi * (1 + 5 ** 0.5) * i
            pts = c + r * np.stack([np.sin(phi) * np.cos(th),
                                    np.sin(phi) * np.sin(th),
                                    np.cos(phi)], axis=1)
        return torch.tensor(pts, dtype=torch.float64)

    @property
    def params(self):
        return [self.alpha]

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """b_k(x): [N, n_modes] Gaussian bumps (the proxy's 'penultimate
        features'; closed-form dpsi/dalpha)."""
        d2 = ((x[:, None, :] - self._centers[None, :, :]) ** 2).sum(-1)
        return torch.exp(-d2 / (2.0 * self._sigma ** 2))

    def psi(self, x: torch.Tensor) -> torch.Tensor:
        return self.base.psi(x) + self.features(x) @ self.alpha
