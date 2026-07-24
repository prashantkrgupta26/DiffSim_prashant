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


# ---------------------------------------------------------------------------
# Task 2 — SBM cell data + embedded-body .vtp
# ---------------------------------------------------------------------------

def _sbm_fixture(level=3, lam=1.0):
    """Build a tiny 3-D SBM setup around a sphere and return the pieces the
    exporter needs: (mesh, retained_tree, frac, sf, geom).

    level=3 (136 retained elements) is the smallest resolution that yields
    BOTH fully-interior elements (frac==1) and surrogate-face elements around
    a radius-0.3 sphere, which the element_type tests need."""
    from diffsim.geometry.csg import Sphere
    from diffsim.sbm.surrogate import (
        classify_lambda, extract_surrogate, GeometryData)
    from diffsim.mesh.faces import face_tables

    oracle = Sphere((0.5, 0.5, 0.5), 0.3)
    tree = build_uniform(level, dim=3)
    ret, frac = classify_lambda(tree, oracle, lam, domain="inside")
    sf = extract_surrogate(ret)
    mesh = build_mesh(ret, p=1)
    geo = GeometryData.evaluate(oracle, ret, sf, face_tables(1, 3),
                                domain="inside")
    return mesh, ret, frac, sf, geo


def test_export_vtu_sbm_cell_arrays(tmp_path):
    import meshio
    from diffsim.viz import export_vtu_sbm

    mesh, ret, frac, sf, geo = _sbm_fixture()
    Nn = len(mesh.node_coords)
    out = export_vtu_sbm(
        mesh, tmp_path / "sbm.vtu",
        fields={"phi_p": np.zeros(Nn), "phi_f": np.zeros(Nn)},
        retained_tree=ret, frac=frac, geom=geo, sf=sf)
    assert out.exists()

    m = meshio.read(str(out))
    # all three SBM cell arrays present
    assert "element_type" in m.cell_data
    assert "frac_in" in m.cell_data
    assert "d_mean" in m.cell_data
    ne = len(ret)
    # concatenated over cell blocks, they cover every element exactly once
    et = np.concatenate(m.cell_data["element_type"])
    fr = np.concatenate(m.cell_data["frac_in"])
    dm = np.concatenate(m.cell_data["d_mean"])
    assert et.shape == (ne,)
    assert fr.shape == (ne,)
    assert dm.shape == (ne,)
    # independent reference: elements with frac==1 that are NOT surrogate have
    # element_type 0
    surrogate = np.zeros(ne, bool)
    surrogate[np.unique(sf.elem)] = True
    interior = (frac == 1.0) & ~surrogate
    # order in the concatenated arrays follows mesh.bins ordering — for a
    # uniform p1 mesh that is the natural element order, so compare directly
    assert (et[interior] == 0).all()
    assert (et[surrogate] == 2).all()


def test_export_body_vtp_roundtrip(tmp_path):
    import pyvista as pv
    from diffsim.viz import export_body_vtp

    verts = np.array([[0.0, 0.0, 0.0],
                      [1.0, 0.0, 0.0],
                      [1.0, 1.0, 0.0],
                      [0.0, 1.0, 0.0]], dtype=np.float64)
    tris = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    out = export_body_vtp(verts, tris, tmp_path / "body.vtp")
    assert out.exists()

    surf = pv.read(str(out))
    np.testing.assert_allclose(np.asarray(surf.points), verts, atol=1e-12)
    # PyVista faces: flat [3, i0, i1, i2, 3, ...]; reshape to (Nt, 4), drop count
    faces = surf.faces.reshape(-1, 4)[:, 1:]
    np.testing.assert_array_equal(np.sort(faces, axis=0),
                                  np.sort(tris, axis=0))


def test_element_type_values(tmp_path):
    """A surrogate-face element has element_type==2; a fully-interior element
    has element_type==0."""
    from diffsim.viz import export_vtu_sbm
    import meshio

    mesh, ret, frac, sf, geo = _sbm_fixture()
    Nn = len(mesh.node_coords)
    out = export_vtu_sbm(
        mesh, tmp_path / "vals.vtu",
        fields={"phi_p": np.zeros(Nn), "phi_f": np.zeros(Nn)},
        retained_tree=ret, frac=frac, geom=geo, sf=sf)
    m = meshio.read(str(out))
    et = np.concatenate(m.cell_data["element_type"])

    # there IS at least one surrogate element and at least one interior element
    assert len(sf.elem) > 0
    known_surrogate = int(np.unique(sf.elem)[0])
    assert et[known_surrogate] == 2
    interior_ids = np.where((frac == 1.0))[0]
    interior_ids = [i for i in interior_ids
                    if i not in set(np.unique(sf.elem).tolist())]
    assert len(interior_ids) > 0
    assert et[interior_ids[0]] == 0
