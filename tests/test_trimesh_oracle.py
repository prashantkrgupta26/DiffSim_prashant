"""TriMeshOracle contract tests."""
import numpy as np
import pytest

from diffsim.geometry.trimesh import TriMeshOracle, icosphere

pytestmark = pytest.mark.geometry



def test_update_vertices_refits_bvh(device):
    """Evaluation finding 4 (CONFIRMED, fixed): in-place vertex mutation
    left the fp32 BVH stale. Contract: update_vertices() refits; results
    must match a freshly constructed oracle."""
    base = icosphere(2, (0.5, 0.5, 0.5), 0.25, device=device)
    verts = base.verts.detach().numpy()
    tris = base.tris
    o1 = TriMeshOracle(verts, tris, device=device)
    pts = np.array([[0.6, 0.5, 0.5], [0.5, 0.72, 0.5], [0.3, 0.4, 0.55]])
    o1.distance_vector(pts)                      # warm query on ORIGINAL
    verts2 = verts * 1.35                        # large inflate: BVH must move
    o1.update_vertices(verts2)
    d1, n1, ok1 = o1.distance_vector(pts)
    o2 = TriMeshOracle(verts2, tris, device=device)
    d2, n2, ok2 = o2.distance_vector(pts)
    assert ok1.all() and ok2.all()
    assert np.abs(d1 - d2).max() < 1e-12
    assert np.abs(n1 - n2).max() < 1e-12
