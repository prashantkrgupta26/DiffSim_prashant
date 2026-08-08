"""Differentiable phase-field adjoint (milestone: genuine three-way-verified
adjoint through the Cahn-Hilliard / Allen-Cahn / multiphase stack)."""
from .phasefield import (CHDiscrete, CHForward, CHAdjoint, PolyEnergy,
                         FHEnergy, PolyBasisEnergy)
from .crystallization import CACHDiscrete, CACHForward, CACHAdjoint
from .multiphase import (MultiEnergy, FHMultiEnergy, MultiCHDiscrete,
                         MultiCHForward, MultiCHAdjoint)
from .neural_multiphase import BasisMultiEnergy, MobilityClosure
from .linsolve_backend import (LinearBackend, ScipyBackend, CudssBackend,
                               scipy_to_torch_csr)
from .crystallization_multi import (CrystalEnergy, AdditiveCrystalEnergy,
                                    CrystalCHDiscrete, CrystalCHForward,
                                    CrystalCHAdjoint)

__all__ = ["CHDiscrete", "CHForward", "CHAdjoint", "PolyEnergy", "FHEnergy",
           "PolyBasisEnergy", "CACHDiscrete", "CACHForward", "CACHAdjoint",
           "MultiEnergy", "FHMultiEnergy", "MultiCHDiscrete",
           "MultiCHForward", "MultiCHAdjoint", "BasisMultiEnergy",
           "MobilityClosure",
           "LinearBackend", "ScipyBackend", "CudssBackend", "scipy_to_torch_csr",
           "CrystalEnergy", "AdditiveCrystalEnergy",
           "CrystalCHDiscrete", "CrystalCHForward", "CrystalCHAdjoint"]
