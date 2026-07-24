"""ParaView .pvsm template gates (spec §7, §8): the state XML must parse, name
the correct VTU filename, and reference the correct color array."""
import xml.etree.ElementTree as ET

import pytest

pytest.importorskip("meshio")

from diffsim.viz import paraview_state


def _write_vtu(tmp_path):
    import numpy as np
    from diffsim.octree.build import build_uniform
    from diffsim.mesh.nodes import build_mesh
    from diffsim.viz import export_vtu
    tree = build_uniform(2, dim=3)
    mesh = build_mesh(tree, p=1)
    Nn = len(mesh.node_coords)
    vtu = tmp_path / "test.vtu"
    export_vtu(mesh, vtu, fields={
        "phi_p": np.zeros(Nn), "phi_f": np.zeros(Nn)})
    return vtu


def test_pvsm_is_valid_xml(tmp_path):
    vtu = _write_vtu(tmp_path)
    pvsm_str = paraview_state(vtu, physics="film")
    root = ET.fromstring(pvsm_str)
    assert root.tag is not None


def test_pvsm_references_vtu_filename(tmp_path):
    vtu = _write_vtu(tmp_path)
    pvsm_str = paraview_state(vtu, physics="film")
    assert "test.vtu" in pvsm_str


def test_pvsm_color_by_field(tmp_path):
    vtu = _write_vtu(tmp_path)
    pvsm_str = paraview_state(vtu, physics="film", color_by="phi_p")
    assert "phi_p" in pvsm_str


def test_pvsm_writes_file(tmp_path):
    vtu = _write_vtu(tmp_path)
    out = tmp_path / "scene.pvsm"
    paraview_state(vtu, physics="film", output=out)
    assert out.exists()
    assert out.stat().st_size > 0
    # parse it
    ET.parse(str(out))


def test_pvsm_ns_physics(tmp_path):
    vtu = _write_vtu(tmp_path)
    pvsm_str = paraview_state(vtu, physics="ns")
    assert "velocity_magnitude" in pvsm_str or "ns" in pvsm_str.lower()


def test_pvsm_camera_section(tmp_path):
    """Camera dict adds a RenderView proxy with the given position."""
    vtu = _write_vtu(tmp_path)
    pvsm_str = paraview_state(
        vtu, physics="ns",
        camera={"position": [1.0, 2.0, 3.0], "focal_point": [0.0, 0.0, 0.0]})
    root = ET.fromstring(pvsm_str)  # still valid XML with the camera block
    assert "CameraPosition" in pvsm_str
    assert root.tag is not None
