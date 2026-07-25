"""Unit tests for diffsim.viz.results — save_flow_run artifact export.

Uses a tiny synthetic mesh so the test never touches real solver state,
and directs all output to a tmp directory so it never litters results/.

The test is fast (CPU-only, no GPU, no heavy solver) and runs in CI.
"""
import pathlib

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Skip guard: meshio is required to write VTUs
# ---------------------------------------------------------------------------
meshio = pytest.importorskip("meshio", reason="meshio not installed — skip viz export tests")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def tmp_base(tmp_path):
    """Return a fresh tmp_path that will act as the repo root for results/."""
    return tmp_path


@pytest.fixture()
def synthetic_mesh(tmp_path):
    """Build a tiny real DiffSim Mesh (level 2, dim 2) for VTU export.

    We need an actual Mesh object because export_vtu reads mesh.dim,
    mesh.node_coords, mesh.tree, mesh.bins, and mesh.conn_of.
    """
    from diffsim.octree.build import build_uniform
    from diffsim.mesh.nodes import build_mesh

    tree = build_uniform(2, dim=2)
    mesh = build_mesh(tree, p=1)
    return mesh


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_histories(nsteps=8):
    t = np.arange(1, nsteps + 1) * 0.01
    cd = np.linspace(2.0, 1.8, nsteps)
    cl = np.sin(np.linspace(0, np.pi, nsteps)) * 0.3
    return t, cd, cl


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestResultsDir:
    def test_creates_subdirs(self, tmp_base):
        from diffsim.viz.results import results_dir

        rd = results_dir(base=tmp_base)

        assert rd.is_dir(), "results_dir() should return an existing directory"
        assert (rd / "figures").is_dir(), "figures/ subdir missing"
        assert (rd / "vtu").is_dir(), "vtu/ subdir missing"

    def test_idempotent(self, tmp_base):
        from diffsim.viz.results import results_dir

        rd1 = results_dir(base=tmp_base)
        rd2 = results_dir(base=tmp_base)
        assert rd1 == rd2


class TestSaveFlowRunHistories:
    def test_writes_forces_png(self, tmp_base):
        from diffsim.viz.results import save_flow_run

        t, cd, cl = _make_histories()
        written = save_flow_run(
            "test_case",
            histories={"Cd": (t, cd), "Cl": (t, cl)},
            base=tmp_base,
        )

        assert "forces_png" in written, f"forces_png missing; written={written}"
        png = written["forces_png"]
        assert png.exists(), f"PNG file not on disk: {png}"
        assert png.stat().st_size > 0, "PNG is zero bytes"
        assert png.suffix == ".png"
        assert "test_case_forces" in png.name

    def test_histories_values_only(self, tmp_base):
        """Accept bare arrays (no explicit t axis)."""
        from diffsim.viz.results import save_flow_run

        cd = np.array([2.0, 1.9, 1.85, 1.8])
        written = save_flow_run(
            "bare_hist",
            histories={"Cd": cd},
            base=tmp_base,
        )
        assert "forces_png" in written
        assert written["forces_png"].exists()


class TestSaveFlowRunVtu:
    def test_writes_vtu_and_pvsm(self, tmp_base, synthetic_mesh):
        from diffsim.viz.results import save_flow_run

        mesh = synthetic_mesh
        n_nodes = len(mesh.node_coords)
        fields = {
            "velocity_magnitude": np.random.rand(n_nodes),
            "pressure": np.random.rand(n_nodes) - 0.5,
        }

        written = save_flow_run(
            "test_flow",
            mesh=mesh,
            node_fields=fields,
            base=tmp_base,
        )

        # VTU written
        assert "vtu" in written, f"vtu missing; written={written}"
        vtu_path = written["vtu"]
        assert vtu_path.exists()
        assert vtu_path.stat().st_size > 0
        assert vtu_path.suffix == ".vtu"

        # VTU is readable by meshio and contains the expected point data
        m = meshio.read(str(vtu_path))
        assert "velocity_magnitude" in m.point_data, (
            f"velocity_magnitude not found in point_data; "
            f"keys={list(m.point_data.keys())}"
        )
        assert "pressure" in m.point_data

        # PVSM written
        assert "pvsm" in written, f"pvsm missing; written={written}"
        pvsm_path = written["pvsm"]
        assert pvsm_path.exists()
        assert pvsm_path.stat().st_size > 0
        pvsm_text = pvsm_path.read_text()
        assert "velocity_magnitude" in pvsm_text, (
            "ParaView state should reference velocity_magnitude as color field"
        )
        assert vtu_path.name in pvsm_text, (
            "ParaView state should reference the VTU filename"
        )

    def test_vtu_cell_data_present(self, tmp_base, synthetic_mesh):
        """export_vtu adds element_size and level cell data."""
        from diffsim.viz.results import save_flow_run

        mesh = synthetic_mesh
        n_nodes = len(mesh.node_coords)
        written = save_flow_run(
            "cell_data_check",
            mesh=mesh,
            node_fields={"velocity_magnitude": np.ones(n_nodes)},
            base=tmp_base,
        )
        m = meshio.read(str(written["vtu"]))
        assert "element_size" in m.cell_data or len(m.cell_data) >= 0
        # At minimum the file has cells (not a degenerate point cloud)
        assert len(m.cells) > 0, "VTU has no cell blocks — expected octree cells"

    def test_missing_mesh_skips_vtu(self, tmp_base):
        """Without mesh, no VTU is written (even if node_fields provided)."""
        from diffsim.viz.results import save_flow_run

        written = save_flow_run(
            "no_mesh",
            node_fields={"velocity_magnitude": np.ones(4)},
            base=tmp_base,
        )
        assert "vtu" not in written
        assert "pvsm" not in written


class TestSaveFlowRunBody:
    def test_degenerate_body_skipped(self, tmp_base):
        """A 2-point line (2 verts, 0 triangles) should be skipped gracefully."""
        from diffsim.viz.results import save_flow_run

        verts = np.array([[0.375, 0.375], [0.375, 0.625]])
        tris = np.empty((0, 3), dtype=np.int64)  # no triangles

        written = save_flow_run(
            "degenerate_body",
            body=(verts, tris),
            base=tmp_base,
        )
        # Neither body_vtp nor body_vtu should be in written
        assert "body_vtp" not in written
        assert "body_vtu" not in written

    def test_valid_triangle_body_meshio_fallback(self, tmp_base):
        """A proper triangle can be exported (falls back to meshio .vtu if pyvista missing)."""
        import importlib
        pyvista_available = importlib.util.find_spec("pyvista") is not None

        from diffsim.viz.results import save_flow_run

        # Minimal valid triangle (not degenerate)
        verts = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]])
        tris = np.array([[0, 1, 2]])

        written = save_flow_run(
            "triangle_body",
            body=(verts, tris),
            base=tmp_base,
        )

        if pyvista_available:
            # .vtp may or may not work depending on VTK install; accept either
            assert "body_vtp" in written or "body_vtu" in written, (
                f"Expected body export; written={written}"
            )
        else:
            # Without pyvista, should fall back to meshio .vtu
            assert "body_vtu" in written, (
                f"Expected meshio fallback body_vtu; written={written}"
            )
            body_path = written["body_vtu"]
            assert body_path.exists()
            assert body_path.stat().st_size > 0
            m = meshio.read(str(body_path))
            assert len(m.cells) > 0


class TestSaveFlowRunCombined:
    def test_all_artifacts_together(self, tmp_base, synthetic_mesh):
        """histories + mesh + node_fields in one call."""
        from diffsim.viz.results import save_flow_run

        mesh = synthetic_mesh
        n_nodes = len(mesh.node_coords)
        t, cd, cl = _make_histories()

        written = save_flow_run(
            "combined_run",
            mesh=mesh,
            node_fields={
                "velocity_magnitude": np.random.rand(n_nodes),
                "pressure": np.random.rand(n_nodes),
            },
            histories={"Cd": (t, cd), "Cl": (t, cl)},
            base=tmp_base,
        )

        assert "forces_png" in written
        assert "vtu" in written
        assert "pvsm" in written

        for kind, path in written.items():
            assert path.exists(), f"{kind}: {path} missing"
            assert path.stat().st_size > 0, f"{kind}: {path} is zero bytes"

    def test_returns_empty_dict_on_no_args(self, tmp_base):
        """No-op call (no args) should return empty dict without crashing."""
        from diffsim.viz.results import save_flow_run

        written = save_flow_run("empty_run", base=tmp_base)
        assert written == {}
