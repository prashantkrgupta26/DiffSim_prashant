import numpy as np
import pytest
from diffsim.octree.build import build_uniform, refine_elements
from diffsim.octree.balance import balance2to1
from diffsim.mesh.nodes import build_mesh

pytestmark = pytest.mark.tier2

def _mixed_tree_and_p(dim=3):
    """Uniform level-2 tree; p2 band = elements touching x=0 plane, p1 elsewhere.
    Same level everywhere -> only the p knob changes across faces."""
    t = build_uniform(2, dim=dim)
    p = np.ones(len(t), np.int8)
    p[t.anchors()[:, 0] == 0] = 2
    return t, p

def test_mixed_bins_and_backcompat():
    t, p = _mixed_tree_and_p()
    m = build_mesh(t, p)
    assert m.conn is None and set(m.bins) == {1, 2}
    assert len(m.bins[1]) + len(m.bins[2]) == len(t)
    assert m.conn_of[1].shape[1] == 8 and m.conn_of[2].shape[1] == 27
    mu = build_mesh(t, 1)
    assert mu.conn is not None and np.all(mu.p_elem == 1)     # uniform back-compat

def test_one_knob_violation_raises():
    t = build_uniform(1, dim=3)
    mask = np.zeros(len(t), bool); mask[0] = True
    t = balance2to1(refine_elements(t, mask))                  # levels 1 and 2 coexist
    p = np.ones(len(t), np.int8)
    p[t.levels == 2] = 2                                       # p change ACROSS the h-face
    with pytest.raises(ValueError, match="one-knob"):
        build_mesh(t, p)

def test_p1_nodes_subset_of_p2_lattice():
    t, p = _mixed_tree_and_p()
    m = build_mesh(t, p)
    # every node icoord is an integer (on the doubled-grid lattice)
    assert np.all(m.node_icoords % 1 == 0)
    # global dedup produced no duplicate icoords
    assert len(np.unique(m.node_icoords, axis=0)) == len(m.node_icoords)
