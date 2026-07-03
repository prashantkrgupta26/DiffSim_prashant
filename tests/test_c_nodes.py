import numpy as np
import pytest
from diffsim.octree.build import build_uniform, refine_elements
from diffsim.octree.carve import SphereOracle, carve
from diffsim.mesh.nodes import build_mesh

pytestmark = pytest.mark.tier2

def test_C1_unique_node_count_uniform():
    for lvl in range(1, 4):
        m1 = build_mesh(build_uniform(lvl), p=1)
        assert len(m1.node_coords) == (2**lvl + 1) ** 3
        m2 = build_mesh(build_uniform(lvl), p=2)
        assert len(m2.node_coords) == (2 * 2**lvl + 1) ** 3

def test_C2_connectivity_shape_and_sharing():
    m = build_mesh(build_uniform(2), p=1)
    assert m.conn.shape == (64, 8)
    for e in range(64):
        assert len(set(m.conn[e])) == 8      # no duplicate nodes within an element
    # shared face: elements 0's X_PLUS corner coords appear in the neighbor
    c0 = set(map(tuple, m.node_icoords[m.conn[0]]))
    shared = [len(c0 & set(map(tuple, m.node_icoords[m.conn[e]]))) for e in range(1, 64)]
    assert max(shared) == 4                  # face neighbors share exactly 4 corners (p1)

def test_C4_C6_incomplete_octree_no_orphans():
    tree, _ = carve(build_uniform(4), SphereOracle((0.5, 0.5, 0.5), 0.4))
    m = build_mesh(tree, p=1)
    referenced = np.unique(m.conn.ravel())
    assert len(referenced) == len(m.node_coords)          # C6: no orphan nodes
    assert np.all(referenced == np.arange(len(m.node_coords)))
