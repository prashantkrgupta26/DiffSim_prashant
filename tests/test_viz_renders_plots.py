"""Render + plot smoke tests (spec §8): assert they run and produce non-empty
output of correct dimensions — pixel-exact gating is brittle and out of scope.

PyVista render tests need a working off-screen GL backend (EGL/OSMesa/xvfb).
On a display-less box without one they are skipped (skip-if-no-GL), NOT failed.
The matplotlib plot tests (Task 4) have no GL dependency and always run.
"""
import numpy as np
import pytest

pytest.importorskip("pyvista")

from diffsim.viz.renders import offscreen_gl_ok

_GL = offscreen_gl_ok()
gl_required = pytest.mark.skipif(
    not _GL, reason="no off-screen GL backend (EGL/OSMesa/xvfb) available")


def _write_test_vtu(tmp_path):
    """Build a tiny VTU with element_size cell data and phi_p point data."""
    from diffsim.octree.build import build_uniform
    from diffsim.mesh.nodes import build_mesh
    from diffsim.viz import export_vtu
    tree = build_uniform(2, dim=3)
    mesh = build_mesh(tree, p=1)
    Nn = len(mesh.node_coords)
    vtu = tmp_path / "tiny.vtu"
    export_vtu(mesh, vtu, fields={
        "phi_p": np.linspace(0.0, 1.0, Nn),
        "phi_f": np.linspace(0.1, 0.9, Nn),
    })
    return vtu


# ---------------------------------------------------------------------------
# Task 3 — PyVista renders (skip-if-no-GL)
# ---------------------------------------------------------------------------

@gl_required
def test_mesh_slice_runs(tmp_path):
    vtu = _write_test_vtu(tmp_path)
    from diffsim.viz.renders import mesh_slice, Style
    out = tmp_path / "slice.png"
    pl = mesh_slice(vtu, color_by="element_size", output=out,
                    style=Style(window_size=(400, 300)))
    assert out.exists()
    assert out.stat().st_size > 0
    img = np.array(pl.screenshot(return_img=True))
    assert img.shape[0] == 300 and img.shape[1] == 400
    assert img.shape[2] == 3


@gl_required
def test_contour_runs(tmp_path):
    vtu = _write_test_vtu(tmp_path)
    from diffsim.viz.renders import contour, Style
    out = tmp_path / "contour.png"
    contour(vtu, field="phi_p", output=out,
            style=Style(window_size=(400, 300)))
    assert out.exists()
    assert out.stat().st_size > 0


@gl_required
def test_lic_runs(tmp_path):
    """LIC: build a VTU with a vector field; assert PNG produced."""
    from diffsim.octree.build import build_uniform
    from diffsim.mesh.nodes import build_mesh
    from diffsim.viz import export_vtu
    from diffsim.viz.renders import lic, Style
    tree = build_uniform(2, dim=3)
    mesh = build_mesh(tree, p=1)
    Nn = len(mesh.node_coords)
    velocity = np.column_stack([
        np.sin(mesh.node_coords[:, 0] * np.pi),
        np.cos(mesh.node_coords[:, 1] * np.pi),
        np.zeros(Nn),
    ])
    vtu = tmp_path / "vel.vtu"
    export_vtu(mesh, vtu, fields={
        "velocity": velocity,
        "velocity_magnitude": np.linalg.norm(velocity, axis=1),
    })
    out = tmp_path / "lic.png"
    lic(vtu, color_by="velocity_magnitude", output=out,
        style=Style(window_size=(400, 300)))
    assert out.exists()
    assert out.stat().st_size > 0


def test_renders_importable_without_gl():
    """The render module imports and Style/offscreen_gl_ok work regardless of
    GL availability (guards the non-GL import path)."""
    from diffsim.viz.renders import Style, lic, mesh_slice, contour
    s = Style()
    assert s.cmap == "RdBu_r"
    assert callable(lic) and callable(mesh_slice) and callable(contour)
    # offscreen_gl_ok never raises, returns a bool
    assert isinstance(offscreen_gl_ok(), bool)
