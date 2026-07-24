"""DiffSim visualization module — Phase 1.

Optional extra: pip install diffsim[viz]
Requires: pyvista>=0.44, meshio>=5.3

`import diffsim` never pulls pyvista/meshio in — every third-party viz import is
guarded inside the function body that needs it.
"""
from .export import export_vtu, export_vtu_sbm, export_body_vtp
from .renders import Style, lic, mesh_slice, contour
from .plots import convergence, surface_profile, history
from .share import paraview_state

__all__ = [
    "export_vtu",
    "export_vtu_sbm",
    "export_body_vtp",
    "Style",
    "lic",
    "mesh_slice",
    "contour",
    "convergence",
    "surface_profile",
    "history",
    "paraview_state",
]
