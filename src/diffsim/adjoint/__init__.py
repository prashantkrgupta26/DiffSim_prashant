"""Differentiable phase-field adjoint (milestone: genuine three-way-verified
adjoint through the Cahn-Hilliard / Allen-Cahn / multiphase stack)."""
from .phasefield import (CHDiscrete, CHForward, CHAdjoint, PolyEnergy,
                         FHEnergy, PolyBasisEnergy)
from .crystallization import CACHDiscrete, CACHForward, CACHAdjoint
from .multiphase import (MultiEnergy, FHMultiEnergy, MultiCHDiscrete,
                         MultiCHForward, MultiCHAdjoint)
from .linsolve_backend import (LinearBackend, ScipyBackend, CudssBackend,
                               scipy_to_torch_csr)

__all__ = ["CHDiscrete", "CHForward", "CHAdjoint", "PolyEnergy", "FHEnergy",
           "PolyBasisEnergy", "CACHDiscrete", "CACHForward", "CACHAdjoint",
           "MultiEnergy", "FHMultiEnergy", "MultiCHDiscrete",
           "MultiCHForward", "MultiCHAdjoint", "LinearBackend", "ScipyBackend",
           "CudssBackend", "scipy_to_torch_csr"]
