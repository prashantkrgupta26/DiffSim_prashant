"""Tier-4/5 geometry-backend tests (M1a Task 9): GridSDF + TriMesh(STL)
unit accuracy, admissibility-bounded approximate-geometry patch, and the
cross-backend oracle suite — the same SBM solve through each backend,
compared against the AnalyticCSG baseline (NeuralSDF joins in M1c)."""
import numpy as np
import pytest
import torch
from diffsim.geometry.csg import Sphere
from diffsim.geometry.gridsdf import GridSDF
from diffsim.geometry.trimesh import TriMeshOracle, icosphere
from diffsim.geometry.oracle import admissibility
from diffsim.geometry.project import distance_numpy

pytestmark = pytest.mark.tier4


def test_gridsdf_interpolation_exact_on_grid():
    # values reproduced exactly at grid corners; linear fields exactly inside
    target = Sphere((0.5, 0.5), 0.3)
    g = GridSDF.from_oracle(target, n=64)
    corners = np.array([[0.25, 0.5], [0.5, 0.75], [0.0, 0.0]])
    assert np.allclose(g.classify(corners), target.classify(corners), atol=1e-12)
    # a LINEAR field is reproduced exactly by multilinear interpolation
    lin = 1.0 + 2.0 * np.linspace(0, 1, 65)[:, None] \
        - 0.5 * np.linspace(0, 1, 65)[None, :]
    gl = GridSDF(torch.tensor(lin, dtype=torch.float64))
    rng = np.random.default_rng(2)
    pts = rng.uniform(0, 1, (100, 2))
    assert np.allclose(gl.classify(pts), 1.0 + 2.0 * pts[:, 0] - 0.5 * pts[:, 1],
                       atol=1e-12)


def test_gridsdf_projection_accuracy():
    target = Sphere((0.5, 0.5), 0.3)
    g = GridSDF.from_oracle(target, n=128)
    dx = 1.0 / 128
    rng = np.random.default_rng(3)
    x = 0.5 + 0.28 * rng.uniform(-1, 1, (100, 2))
    d, n, ok = distance_numpy(g, x)
    assert ok.all()
    r = np.linalg.norm(x + d - 0.5, axis=1)
    assert np.abs(r - 0.3).max() < 5 * dx ** 2 / 0.3 + 5e-4   # ~interp error
    diag = admissibility(g, x)
    assert diag["newton_ok_frac"] == 1.0
    assert diag["eps_inf"] < 1e-10        # projection residual on psi_h itself


def test_gridsdf_values_autograd():
    target = Sphere((0.5, 0.5), 0.3)
    g = GridSDF.from_oracle(target, n=32)
    g.values.requires_grad_(True)
    # generic point strictly inside a cell (on-gridline points zero out the
    # weights of the far corners — a measured care point)
    x = torch.tensor([[0.821, 0.503]], dtype=torch.float64)
    g.psi(x).sum().backward()
    nz = int((g.values.grad != 0).sum())
    assert nz == 4                        # the enclosing cell corners
    assert abs(float(g.values.grad.sum()) - 1.0) < 1e-12   # PoU of lerp weights


def test_trimesh_icosphere_accuracy():
    r = 0.32
    o = icosphere(3, (0.5, 0.5, 0.5), r)
    # faceting sagitta bound from the LARGEST edge (subdivided-icosahedron
    # triangle sizes vary ~20% — the first triangle underestimates)
    v = o.verts.detach().numpy()
    tri_v = v[o.tris]                                     # [Nt, 3, 3]
    edges = np.concatenate([
        np.linalg.norm(tri_v[:, 0] - tri_v[:, 1], axis=1),
        np.linalg.norm(tri_v[:, 1] - tri_v[:, 2], axis=1),
        np.linalg.norm(tri_v[:, 2] - tri_v[:, 0], axis=1)])
    sagitta = r * (1 - np.cos(np.arcsin(edges.max() / (2 * r))))
    rng = np.random.default_rng(5)
    x = 0.5 + 0.3 * rng.uniform(-1, 1, (200, 3))
    psi = o.classify(x)
    exact = np.linalg.norm(x - 0.5, axis=1) - r
    assert np.abs(psi - exact).max() < 1.5 * sagitta + 1e-12
    # signs correct well away from the surface
    inside = 0.5 + 0.1 * rng.uniform(-1, 1, (50, 3))
    outside = np.array([[0.02, 0.03, 0.05], [0.95, 0.9, 0.93]])
    assert (o.classify(inside) < 0).all()
    assert (o.classify(outside) > 0).all()


def test_trimesh_distance_vector_and_verts_grad():
    r = 0.32
    o = icosphere(2, (0.5, 0.5, 0.5), r)
    x = np.array([[0.9, 0.5, 0.5], [0.5, 0.55, 0.72]])
    d, n, ok = o.distance_vector(x)
    assert ok.all()
    # closest point lies on the faceted surface: |psi(y)| ~ 0
    assert np.abs(o.classify(x + d)).max() < 1e-9
    # gradient flows to vertex positions
    o.verts.requires_grad_(True)
    d_t, n_t, ok = o.distance_torch(x)
    d_t.sum().backward()
    assert o.verts.grad is not None and float(o.verts.grad.abs().sum()) > 0


# ---------------------------------------------------------------------------
# Cross-backend oracle suite (tier5): same SBM solve through each backend
# ---------------------------------------------------------------------------
_CROSS = {"R": 0.32, "level": 4, "cache": {}}


def _sbm_case_3d(backend, device):
    from diffsim.octree.build import build_uniform
    from diffsim.mesh.nodes import build_mesh
    from diffsim.mesh.constraints import build_constraints
    from diffsim.mesh.basis import basis_tables
    from diffsim.mesh.faces import face_tables
    from diffsim.assembly.operators import DeviceMesh
    from diffsim.sbm.surrogate import (classify_lambda, extract_surrogate,
                                       GeometryData)
    from diffsim.sbm.poisson import SBMPoisson
    from diffsim.physics.poisson import l2_error_masked

    r = _CROSS["R"]
    target = Sphere((0.5,) * 3, r)
    oracle = {
        "csg": lambda: target,
        "trimesh": lambda: icosphere(3, (0.5,) * 3, r, device=device),
        "grid": lambda: GridSDF.from_oracle(target, n=96),
    }[backend]()
    u = lambda x: (np.sin(np.pi * x[:, 0]) * np.sin(np.pi * x[:, 1])
                   * np.sin(np.pi * x[:, 2]))
    f = lambda x: 3 * np.pi ** 2 * u(x)
    tree = build_uniform(_CROSS["level"], dim=3)
    ret, _ = classify_lambda(tree, oracle, 0.0)
    sf = extract_surrogate(ret)
    mesh = build_mesh(ret, p=1)
    cons = build_constraints(mesh)
    dm = DeviceMesh.from_mesh(mesh, cons, basis_tables(1, dim=3), device)
    geo = GeometryData.evaluate(oracle, ret, sf, face_tables(1, 3))
    if backend == "trimesh":
        diag_eps = 0.0        # closest-point is exact on the faceted surface
    else:
        diag_eps = admissibility(oracle, geo.xq)["eps_inf"]
    prob = SBMPoisson(dm, geo, sf, g_fn=u)
    uh = prob.solve(f_fn=f)
    err = l2_error_masked(dm, uh, u, lambda x: target.classify(x) < 0)
    return err, diag_eps


@pytest.mark.tier5
@pytest.mark.parametrize("backend", ["csg", "trimesh", "grid"])
def test_cross_backend_sphere_mms(backend, device):
    err, diag_eps = _sbm_case_3d(backend, device)
    if backend == "csg":
        _CROSS["cache"]["baseline"] = err
        return
    baseline = _CROSS["cache"].get("baseline")
    if baseline is None:                   # robustness to test reordering
        baseline, _ = _sbm_case_3d("csg", device)
        _CROSS["cache"]["baseline"] = baseline
    # discretization error dominates once geometry error is small; for the
    # faceted trimesh the geometry error is the sagitta, folded into the 2x
    assert err < 2.0 * baseline + 10.0 * diag_eps, (err, baseline, diag_eps)
