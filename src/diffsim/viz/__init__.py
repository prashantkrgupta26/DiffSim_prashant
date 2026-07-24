"""DiffSim visualization module — Phase 1.

Optional extra: pip install diffsim[viz]
Requires: pyvista>=0.44, meshio>=5.3

`import diffsim` never pulls pyvista/meshio in — every third-party viz import is
guarded inside the function body that needs it.
"""
from .export import export_vtu

__all__ = ["export_vtu"]
