"""Gate: round-trip a tiny level-2 uniform mesh to .vtu; check fields."""
import numpy as np
import pytest
import pathlib

pytest.importorskip("pyvista")
pytest.importorskip("meshio")

from diffsim.octree.build import build_uniform
from diffsim.mesh.nodes import build_mesh
from diffsim.viz import export_vtu


def _tiny_mesh():
    tree = build_uniform(2, dim=3)
    mesh = build_mesh(tree, p=1)
    return mesh


def test_export_vtu_creates_file(tmp_path):
    mesh = _tiny_mesh()
    Nn = len(mesh.node_coords)
    rng = np.random.default_rng(0)
    fields = {
        "phi_p": rng.random(Nn),
        "phi_f": rng.random(Nn),
    }
    out = export_vtu(mesh, tmp_path / "test.vtu", fields=fields)
    assert out.exists()
    assert out.stat().st_size > 0


def test_export_vtu_fields_present(tmp_path):
    """Independent-reference check: read the .vtu back with meshio and
    verify the arrays match the inputs exactly (NOT via the exporter's
    own arrays -- this is the house gate-hygiene rule)."""
    import meshio

    mesh = _tiny_mesh()
    Nn = len(mesh.node_coords)
    phi_p_ref = np.linspace(0.0, 1.0, Nn)
    phi_f_ref = np.linspace(0.1, 0.9, Nn)

    export_vtu(mesh, tmp_path / "ref.vtu",
               fields={"phi_p": phi_p_ref, "phi_f": phi_f_ref})

    m = meshio.read(str(tmp_path / "ref.vtu"))
    # point data
    np.testing.assert_allclose(m.point_data["phi_p"], phi_p_ref, atol=1e-12)
    np.testing.assert_allclose(m.point_data["phi_f"], phi_f_ref, atol=1e-12)
    np.testing.assert_allclose(
        m.point_data["phi_s"], 1.0 - phi_p_ref - phi_f_ref, atol=1e-12)
    # cell data must be present
    assert "element_size" in m.cell_data
    assert "level" in m.cell_data
    # coordinate round-trip: meshio returns point coordinates
    np.testing.assert_allclose(
        m.points, mesh.node_coords, atol=1e-12)


def test_export_vtu_cell_count(tmp_path):
    """Number of cells in the written file equals number of octree elements."""
    import meshio
    mesh = _tiny_mesh()
    Nn = len(mesh.node_coords)
    export_vtu(mesh, tmp_path / "cells.vtu",
               fields={"phi_p": np.zeros(Nn), "phi_f": np.zeros(Nn)})
    m = meshio.read(str(tmp_path / "cells.vtu"))
    total_cells = sum(len(cb.data) for cb in m.cells)
    assert total_cells == len(mesh.tree)


def test_export_npz_roundtrip(tmp_path):
    """Pass a .npz path as source; verify fields are written."""
    import meshio
    mesh = _tiny_mesh()
    Nn = len(mesh.node_coords)
    phi_p = np.linspace(0.0, 1.0, Nn)
    phi_f = np.linspace(0.1, 0.9, Nn)
    npz_path = tmp_path / "snap.npz"
    np.savez(npz_path, phi_p=phi_p, phi_f=phi_f,
             coords=mesh.node_coords, h=0.9, t=1.0)
    out = export_vtu(npz_path, tmp_path / "snap.vtu")
    m = meshio.read(str(out))
    np.testing.assert_allclose(m.point_data["phi_p"], phi_p, atol=1e-12)
