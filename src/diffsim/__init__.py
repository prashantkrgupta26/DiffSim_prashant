"""DiffSim — GPU-native, differentiable finite-element multiphysics on
adaptive octrees.

This module defines the **stable top-level public API** (critical-eval P1.1).
Names are resolved lazily (PEP 562 ``__getattr__``) so that ``import diffsim``
and ``diffsim.errors`` stay lightweight and do not force the heavy
Warp/CUDA-backed imports until a solver/assembly symbol is actually used.

Public surface (import-stable across minor versions):

    from diffsim import (
        solve_linear, NonlinearSolver,
        LinearSolveResult, NonlinearSolveResult,
        CEquation, PoissonBrick, SBMPoisson,
        DeviceMesh, build_mesh, build_uniform, refine_elements,
        build_constraints,
    )
    from diffsim.errors import (
        DiffSimError, ConfigError, BackendError,
        SolverError, ConvergenceError,
    )

Everything not listed in :data:`__all__` is an experimental / implementation
detail and may change without notice.
"""

from __future__ import annotations

__version__ = "0.0.1"

# name -> "submodule:attribute" (relative to the diffsim package)
_LAZY = {
    # solvers
    "solve_linear": "solvers.linsolve:solve_linear",
    "NonlinearSolver": "solvers.newton:NonlinearSolver",
    "LinearSolveResult": "solvers.result:LinearSolveResult",
    "NonlinearSolveResult": "solvers.result:NonlinearSolveResult",
    # equation / brick API
    "CEquation": "api.equation:CEquation",
    "assemble_brick_csr": "api.equation:assemble_brick_csr",
    "PoissonBrick": "api.example_bricks:PoissonBrick",
    # SBM
    "SBMPoisson": "sbm.poisson:SBMPoisson",
    # mesh / assembly
    "DeviceMesh": "assembly.operators:DeviceMesh",
    "build_mesh": "mesh.nodes:build_mesh",
    "build_constraints": "mesh.constraints:build_constraints",
    "build_uniform": "octree.build:build_uniform",
    "refine_elements": "octree.build:refine_elements",
}

# errors are cheap (no Warp/Torch) — expose eagerly
from .errors import (  # noqa: E402
    BackendError,
    ConfigError,
    ConvergenceError,
    DiffSimError,
    SolverError,
)

__all__ = [
    "__version__",
    # solvers
    "solve_linear",
    "NonlinearSolver",
    "LinearSolveResult",
    "NonlinearSolveResult",
    # equation / brick API
    "CEquation",
    "assemble_brick_csr",
    "PoissonBrick",
    # SBM
    "SBMPoisson",
    # mesh / assembly
    "DeviceMesh",
    "build_mesh",
    "build_constraints",
    "build_uniform",
    "refine_elements",
    # errors
    "DiffSimError",
    "ConfigError",
    "BackendError",
    "SolverError",
    "ConvergenceError",
]


def __getattr__(name):
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module 'diffsim' has no attribute {name!r}")
    import importlib

    mod_name, attr = target.split(":")
    mod = importlib.import_module(f".{mod_name}", __name__)
    value = getattr(mod, attr)
    globals()[name] = value  # cache for next access
    return value


def __dir__():
    return sorted(list(globals().keys()) + list(_LAZY.keys()))
