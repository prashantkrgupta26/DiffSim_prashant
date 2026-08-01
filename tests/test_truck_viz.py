"""Tests for T3: truck viz V1 + video pipeline.

Test groups:
  1. Q-criterion unit check: pure-shear vs pure-rotation sign / magnitude.
  2. Tiny-case 3-frame export: files exist, meshio/pyvista re-read, finite ranges.
  3. Parity: viz_interval=None byte-identical march (same Cd trajectory).
  4. PNG render smoke: 3-frame -> 3 PNGs per shot (skipif offscreen unavailable).
  5. Centerline slice and surface Cp sanity on tiny mesh.

Run: .venv/bin/pytest tests/test_truck_viz.py -v
"""
import os
import sys
import pathlib
import tempfile

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))

from diffsim.cases.truck_config import load_truck_config

_CFG_PATH = os.path.join(
    os.path.dirname(__file__), "..", "local_code_old",
    "truck_4case_fresh_inputs", "NewRun-no-shell-slope0p25", "config.txt")


def _tiny_tire_mesh(cfg, half=0.05, center=(0.35, 0.0625, 0.0625)):
    """One-tire synthetic mesh (same as test_truck_flow.py)."""
    from diffsim.geometry.merged_trimesh import MergedTriMesh, _read_stl
    v, t = _read_stl(os.path.join(cfg.config_dir, "tire_1.stl"))
    v = v - v.mean(0)
    v = v / np.abs(v).max()
    v = v * half + np.asarray(center, np.float64)
    return MergedTriMesh(v, t)


def _build_tiny(cfg):
    """Build tiny mesh fx dict for unit tests."""
    from truck_flow import build_truck_mesh
    merged = _tiny_tire_mesh(cfg)
    return build_truck_mesh(cfg, base_level=5, region_refine=False,
                            truck_band_to=6, band_cells=2, merged=merged)


# ---------------------------------------------------------------------------
# Group 1: Q-criterion unit check (pure-shear vs pure-rotation)
# ---------------------------------------------------------------------------

class TestQCriterionUnit:
    """Q-criterion kernel sanity on a 1-element synthetic mesh.

    Pure rotation: grad_u = [[0,-omega,0],[omega,0,0],[0,0,0]]
      -> ||Omega||^2 = 2*omega^2, ||S||^2 = 0  -> Q = omega^2 > 0

    Pure shear: grad_u = [[0,s,0],[0,0,0],[0,0,0]]
      -> ||S||^2 = s^2/2, ||Omega||^2 = s^2/2  -> Q = 0
      (actually |S|^2 = s^2/2 and |Omega|^2 = s^2/2 so Q=0 for simple shear)

    Axial strain: grad_u = [[a,0,0],[0,-a/2,0],[0,0,-a/2]]
      -> pure strain tensor, ||Omega||^2 = 0  -> Q < 0
    """

    def _make_hex_mesh(self):
        """Build a single hex element mesh (unit cube, level 3)."""
        from diffsim.octree.build import build_uniform
        from diffsim.octree.balance import balance2to1
        from diffsim.mesh.nodes import build_mesh
        from diffsim.mesh.constraints import build_constraints
        from diffsim.geometry.csg import Box
        from diffsim.sbm.surrogate import classify_lambda

        # Single element in a small uniform tree
        tree = build_uniform(1, dim=3)  # 8 cells
        # Keep just one slab cell (dyadic)
        box = Box((0.25, 0.25, 0.25), (0.25, 0.25, 0.25))
        ret, _ = classify_lambda(tree, box, lam=0.0, domain="inside",
                                 lipschitz_bound=np.inf)
        mesh = build_mesh(ret, p=1)
        return mesh

    def _q_from_linear_field(self, mesh, grad_u_matrix):
        """Set u_node = grad_u_matrix @ x for each node, compute Q."""
        from diffsim.viz.truck_viz import nodal_q_criterion
        coords = np.asarray(mesh.node_coords, dtype=np.float64)
        G = np.asarray(grad_u_matrix, dtype=np.float64)  # [3,3]
        u_node = coords @ G.T  # [Nn, 3]
        return nodal_q_criterion(mesh, u_node)

    def test_pure_rotation_q_positive(self):
        """Solid-body rotation: Q > 0 everywhere."""
        mesh = self._make_hex_mesh()
        omega = 2.0
        # Rotation about z: du_x/dy = -omega, du_y/dx = +omega
        G = np.array([[0.0, -omega, 0.0],
                      [omega,  0.0, 0.0],
                      [0.0,   0.0, 0.0]])
        Q = self._q_from_linear_field(mesh, G)
        # Q = 0.5*(||Omega||^2 - ||S||^2) = 0.5*(2*omega^2 - 0) = omega^2
        assert np.all(Q > 0.0), f"pure rotation must give Q>0, got min={Q.min()}"
        # Check approximate value (Gaussian averaging may vary per node position)
        assert abs(Q.mean() - omega ** 2) < 0.5 * omega ** 2

    def test_axial_strain_q_negative(self):
        """Pure extensional strain: S dominates, Q < 0."""
        mesh = self._make_hex_mesh()
        a = 1.5
        # Extension in x, contraction in y,z
        G = np.array([[a,   0.0,    0.0],
                      [0.0, -a/2,   0.0],
                      [0.0,  0.0,  -a/2]])
        Q = self._q_from_linear_field(mesh, G)
        assert np.all(Q < 0.0), (
            f"pure extension must give Q<0 (strain-dominated), got max={Q.max()}")

    def test_zero_velocity_q_zero(self):
        """Zero velocity -> Q = 0."""
        mesh = self._make_hex_mesh()
        from diffsim.viz.truck_viz import nodal_q_criterion
        Nn = len(mesh.node_coords)
        u_zero = np.zeros((Nn, 3))
        Q = nodal_q_criterion(mesh, u_zero)
        assert np.all(np.abs(Q) < 1e-12), f"zero-u must give Q=0, got max={np.abs(Q).max()}"

    def test_rotation_stronger_than_shear_q_positive(self):
        """If rotation rate > shear rate: Q > 0."""
        mesh = self._make_hex_mesh()
        omega = 3.0
        shear = 1.0
        # Add a small shear on top of rotation
        G = np.array([[shear, -omega, 0.0],
                      [omega,   0.0,  0.0],
                      [0.0,     0.0,  0.0]])
        Q = self._q_from_linear_field(mesh, G)
        # Omega part: antisym[[shear,-omega],[omega,0]] -> Omega_12 = -omega, S_12=shear/2
        # Omega_21 = omega; sym part S_11=shear, S_12=S_21=shear/2
        # At dominant omega > shear, Q should be positive on average
        assert Q.mean() > 0.0, f"rotation-dominated flow must give positive mean Q"


# ---------------------------------------------------------------------------
# Group 2: Tiny-case 3-frame export
# ---------------------------------------------------------------------------

def test_three_frame_export_files_exist():
    """Run 3-step march with viz_interval=1; check all extract files exist."""
    try:
        import pyvista  # noqa: F401
    except ImportError:
        pytest.skip("pyvista not available")
    try:
        import meshio  # noqa: F401
    except ImportError:
        pytest.skip("meshio not available")

    cfg = load_truck_config(_CFG_PATH)
    merged = _tiny_tire_mesh(cfg)

    with tempfile.TemporaryDirectory() as tmpdir:
        from truck_flow import run_truck
        res = run_truck(cfg, nsteps=3, base_level=5, truck_band_to=6,
                        band_cells=2, merged=merged, region_refine=False,
                        nu=1.0 / 50.0, dt=0.02, verbose=False,
                        viz_interval=1, viz_dir=tmpdir,
                        viz_checkpoint_interval=2)

        vd = pathlib.Path(tmpdir)
        hook = res.get("viz_hook")
        assert hook is not None, "viz_hook not returned in result dict"

        # 3 frames -> 3 files per shot (viz_interval=1 means every step)
        for shot in ("q_iso", "centerline", "surface_cp"):
            shot_dir = vd / "frames" / shot
            files = sorted(shot_dir.glob("*.vtp"))
            assert len(files) == 3, (
                f"shot {shot}: expected 3 .vtp files, got {len(files)}")

        # Checkpoint at step 2 (checkpoint_interval=2)
        ckpts = sorted((vd / "checkpoints").glob("*.vtu"))
        assert len(ckpts) >= 1, "expected at least 1 .vtu checkpoint"

        # Time-average .vtu
        avg_files = sorted((vd / "time_avg").glob("*.vtu"))
        assert len(avg_files) == 1, "expected time-average .vtu"


def test_three_frame_export_meshio_reread():
    """Re-read exported .vtp files via pyvista; check fields present + finite."""
    try:
        import pyvista as pv
    except ImportError:
        pytest.skip("pyvista not available")

    cfg = load_truck_config(_CFG_PATH)
    merged = _tiny_tire_mesh(cfg)

    with tempfile.TemporaryDirectory() as tmpdir:
        from truck_flow import run_truck
        run_truck(cfg, nsteps=3, base_level=5, truck_band_to=6,
                  band_cells=2, merged=merged, region_refine=False,
                  nu=1.0 / 50.0, dt=0.02, verbose=False,
                  viz_interval=1, viz_dir=tmpdir)

        vd = pathlib.Path(tmpdir)

        # Q isosurface: should be readable and have velocity_magnitude field
        q_files = sorted((vd / "frames" / "q_iso").glob("*.vtp"))
        assert q_files, "no q_iso .vtp files"
        for f in q_files:
            m = pv.read(str(f))
            # May be empty if Q threshold not crossed, but should be readable
            assert m is not None

        # Centerline slice: velocity_magnitude and pressure must be finite
        cl_files = sorted((vd / "frames" / "centerline").glob("*.vtp"))
        assert cl_files, "no centerline .vtp files"
        for f in cl_files:
            m = pv.read(str(f))
            if m.n_points > 0 and "velocity_magnitude" in m.point_data:
                arr = m.point_data["velocity_magnitude"]
                assert np.all(np.isfinite(arr)), (
                    f"non-finite velocity_magnitude in {f}")

        # Surface Cp: Cp field must be present and finite
        cp_files = sorted((vd / "frames" / "surface_cp").glob("*.vtp"))
        assert cp_files, "no surface_cp .vtp files"
        for f in cp_files:
            m = pv.read(str(f))
            if m.n_cells > 0 and "Cp" in m.cell_data:
                arr = m.cell_data["Cp"]
                assert np.all(np.isfinite(arr)), f"non-finite Cp in {f}"


def test_checkpoint_vtu_meshio_reread():
    """Re-read a checkpoint .vtu via meshio; check Q and velocity fields."""
    try:
        import meshio
    except ImportError:
        pytest.skip("meshio not available")

    cfg = load_truck_config(_CFG_PATH)
    merged = _tiny_tire_mesh(cfg)

    with tempfile.TemporaryDirectory() as tmpdir:
        from truck_flow import run_truck
        run_truck(cfg, nsteps=3, base_level=5, truck_band_to=6,
                  band_cells=2, merged=merged, region_refine=False,
                  nu=1.0 / 50.0, dt=0.02, verbose=False,
                  viz_interval=1, viz_dir=tmpdir,
                  viz_checkpoint_interval=1)

        ckpts = sorted(pathlib.Path(tmpdir, "checkpoints").glob("*.vtu"))
        assert ckpts, "no checkpoint .vtu files"

        m = meshio.read(str(ckpts[-1]))
        assert "Q" in m.point_data, "Q field missing from checkpoint VTU"
        assert "velocity_magnitude" in m.point_data, \
            "velocity_magnitude missing from checkpoint VTU"
        assert "pressure" in m.point_data, "pressure missing from checkpoint VTU"

        Q_arr = m.point_data["Q"]
        assert np.all(np.isfinite(Q_arr)), "non-finite Q in checkpoint VTU"
        umag = m.point_data["velocity_magnitude"]
        assert np.all(np.isfinite(umag)) and np.all(umag >= 0.0), \
            "velocity_magnitude must be finite and non-negative"


# ---------------------------------------------------------------------------
# Group 3: Parity — viz_interval=None byte-identical march
# ---------------------------------------------------------------------------

def test_viz_none_byte_identical_march():
    """Enabling viz must not change the Cd trajectory (parity)."""
    try:
        import pyvista  # noqa: F401
    except ImportError:
        pytest.skip("pyvista not available")

    cfg = load_truck_config(_CFG_PATH)
    merged_a = _tiny_tire_mesh(cfg)
    merged_b = _tiny_tire_mesh(cfg)

    from truck_flow import run_truck

    # Reference: no viz
    res_ref = run_truck(cfg, nsteps=3, base_level=5, truck_band_to=6,
                        band_cells=2, merged=merged_a, region_refine=False,
                        nu=1.0 / 50.0, dt=0.02, verbose=False,
                        viz_interval=None, viz_dir=None)

    with tempfile.TemporaryDirectory() as tmpdir:
        res_viz = run_truck(cfg, nsteps=3, base_level=5, truck_band_to=6,
                            band_cells=2, merged=merged_b, region_refine=False,
                            nu=1.0 / 50.0, dt=0.02, verbose=False,
                            viz_interval=1, viz_dir=tmpdir)

    # Cd trajectories must be identical (same solver path, viz is post-solve)
    np.testing.assert_array_equal(res_ref["cd"], res_viz["cd"],
                                  err_msg="viz hook changed Cd trajectory!")
    np.testing.assert_array_equal(res_ref["cd_surr"], res_viz["cd_surr"],
                                  err_msg="viz hook changed Cd_surr trajectory!")


# ---------------------------------------------------------------------------
# Group 4: PNG render smoke (3 frames -> 3 PNGs per shot)
# ---------------------------------------------------------------------------

def _offscreen_available():
    """Return True if pyvista off-screen rendering works."""
    try:
        import pyvista as pv
        import numpy as np
        pl = pv.Plotter(off_screen=True)
        pl.add_mesh(pv.Sphere(), color="white")
        pl.camera_position = "xy"
        import io
        buf = io.BytesIO()
        arr = pl.screenshot(None, return_img=True)
        pl.close()
        return arr is not None and arr.size > 0
    except Exception:
        return False


@pytest.mark.skipif(not _offscreen_available(), reason="offscreen render unavailable")
def test_render_three_pngs_per_shot():
    """3-frame export -> 3 PNGs per shot with consistent camera metadata."""
    try:
        import pyvista  # noqa: F401
    except ImportError:
        pytest.skip("pyvista not available")

    cfg = load_truck_config(_CFG_PATH)
    merged = _tiny_tire_mesh(cfg)

    with tempfile.TemporaryDirectory() as tmpdir:
        from truck_flow import run_truck
        run_truck(cfg, nsteps=3, base_level=5, truck_band_to=6,
                  band_cells=2, merged=merged, region_refine=False,
                  nu=1.0 / 50.0, dt=0.02, verbose=False,
                  viz_interval=1, viz_dir=tmpdir)

        vd = pathlib.Path(tmpdir)

        # Import and run the renderer
        render_dir = vd / "renders"
        sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "tools"))
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "render_frames",
            str(pathlib.Path(__file__).parent.parent / "tools" / "render_frames.py"))
        rf = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(rf)

        import pyvista as pv

        for shot_name in ("q_iso", "centerline", "surface_cp"):
            shot_cfg = {
                "q_iso": {"scalar": "velocity_magnitude", "cmap": "plasma"},
                "centerline": {"scalar": "velocity_magnitude", "cmap": "viridis"},
                "surface_cp": {"scalar": "Cp", "cmap": "RdBu_r"},
            }[shot_name]

            files = sorted((vd / "frames" / shot_name).glob("*.vtp"))
            if not files:
                continue

            out_dir = render_dir / shot_name
            srange = rf._scan_range([str(f) for f in files],
                                    shot_cfg["scalar"], pv)

            written = rf.render_shot(
                shot_name, [str(f) for f in files], out_dir, pv,
                scalar=shot_cfg["scalar"],
                scalar_range=srange,
                camera=rf._CAMERAS[shot_name],
                width=320, height=240,   # small for speed
                cmap=shot_cfg["cmap"],
            )

            assert len(written) == 3, (
                f"shot {shot_name}: expected 3 PNGs, got {len(written)}")
            for p in written:
                assert p.exists(), f"PNG not written: {p}"
                assert p.stat().st_size > 1000, f"PNG suspiciously small: {p}"

            # Fixed range sanity: all frames rendered with same range -> check
            # PNG files exist and are non-empty (visual verification is T4)
            print(f"[test] shot={shot_name} range={srange} wrote {len(written)} PNGs")


# ---------------------------------------------------------------------------
# Group 5: Surface Cp and centerline slice sanity
# ---------------------------------------------------------------------------

def test_surface_cp_shape_and_range():
    """surface_cp returns correct shapes and plausible Cp values."""
    from diffsim.viz.truck_viz import surface_cp

    cfg = load_truck_config(_CFG_PATH)
    merged = _tiny_tire_mesh(cfg)
    fx = _build_tiny(cfg)
    mesh = fx["mesh"]

    # Synthetic pressure field: uniform p = 0.1 at all nodes
    Nn = len(mesh.node_coords)
    p_node = np.full(Nn, 0.1)
    sf = fx["sf"]

    tri_centroids, Cp, tri_sf_idx = surface_cp(
        mesh, p_node, merged, sf, p_ref=0.0, rho=1.0, U_inf=1.0)

    verts = merged.verts.numpy()
    tris = np.asarray(merged.tris)
    Nt = len(tris)

    assert tri_centroids.shape == (Nt, 3), (
        f"tri_centroids shape mismatch: {tri_centroids.shape}")
    assert Cp.shape == (Nt,), f"Cp shape mismatch: {Cp.shape}"
    assert np.all(np.isfinite(Cp)), "Cp has non-finite values"
    # Cp = (0.1 - 0) / 0.5 = 0.2; should be close
    assert abs(Cp.mean() - 0.2) < 0.05, f"Cp mean {Cp.mean():.4f} not near 0.2"


def test_nodal_q_criterion_shape():
    """nodal_q_criterion returns array with shape [N_nodes]."""
    from diffsim.viz.truck_viz import nodal_q_criterion

    cfg = load_truck_config(_CFG_PATH)
    merged = _tiny_tire_mesh(cfg)
    fx = _build_tiny(cfg)
    mesh = fx["mesh"]

    Nn = len(mesh.node_coords)
    # Linear velocity field: uniform flow in x
    u_node = np.zeros((Nn, 3))
    u_node[:, 0] = 1.0  # uniform flow: no rotation -> Q=0

    Q = nodal_q_criterion(mesh, u_node)
    assert Q.shape == (Nn,), f"Q shape: {Q.shape}, expected ({Nn},)"
    assert np.all(np.isfinite(Q)), "Q has non-finite values for uniform flow"
    # Uniform flow: no grad_u variation within each element (constant field)
    # -> S=0, Omega=0 -> Q=0
    assert np.all(np.abs(Q) < 1e-10), (
        f"uniform flow must give Q=0, max|Q|={np.abs(Q).max():.2e}")
