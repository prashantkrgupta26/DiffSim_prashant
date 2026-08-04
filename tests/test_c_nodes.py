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

def test_hanging_constraints_partition_of_unity():
    from diffsim.mesh.constraints import build_constraints
    t = build_uniform(1)
    mask = np.zeros(8, bool); mask[0] = True
    t = refine_elements(t, mask)                        # one coarse-fine interface set
    for p in (1, 2):
        m = build_mesh(t, p=p)
        c = build_constraints(m)
        assert c.hanging.sum() > 0
        # rows sum to 1 (constant field reproduced through constraints)
        rowsum = np.asarray(c.T.sum(axis=1)).ravel()
        assert np.allclose(rowsum, 1.0, atol=1e-13)
        # a linear field is reproduced exactly by constrained interpolation
        f = lambda x: 1.0 + 2.0 * x[:, 0] - 3.0 * x[:, 1] + 0.5 * x[:, 2]
        u_free = f(m.node_coords[c.free_nodes])
        u_all = c.T @ u_free
        assert np.allclose(u_all, f(m.node_coords), atol=1e-12)

def test_uniform_mesh_has_no_hanging():
    from diffsim.mesh.constraints import build_constraints
    m = build_mesh(build_uniform(2), p=1)
    c = build_constraints(m)
    assert c.hanging.sum() == 0 and c.T.shape == (len(m.node_coords),) * 2


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("p", [1, 2])
@pytest.mark.parametrize("periodic", [None, "all", "one"])
@pytest.mark.parametrize("lvl", [2, 3, 4])
def test_uniform_fast_equals_general(dim, p, periodic, lvl):
    """Closed-form uniform node/constraint build is bit-for-bit identical to the
    general np.unique path (mesh) and the constraint reference (identity)."""
    import diffsim.mesh.nodes as _N
    from diffsim.mesh.constraints import (build_constraints,
                                          _build_constraints_reference)
    per = (None if periodic is None
           else (True,) * dim if periodic == "all"
           else (True,) + (False,) * (dim - 1))
    tree = build_uniform(lvl, dim=dim, periodic=per)

    saved = _N._uniform_complete_level
    try:
        _N._uniform_complete_level = lambda t: None
        m_gen = _N.build_mesh(tree, p=p)          # general np.unique path
    finally:
        _N._uniform_complete_level = saved
    m_fast = build_mesh(tree, p=p)                # closed-form fast path

    assert np.array_equal(m_fast.node_icoords, m_gen.node_icoords)
    assert np.allclose(m_fast.node_coords, m_gen.node_coords)
    assert np.array_equal(m_fast.conn, m_gen.conn)
    assert np.array_equal(m_fast.boundary_nodes, m_gen.boundary_nodes)
    for pv in m_gen.conn_of:
        assert np.array_equal(m_fast.conn_of[pv], m_gen.conn_of[pv])

    c_fast = build_constraints(m_fast)
    c_ref = _build_constraints_reference(m_fast)
    assert np.array_equal(c_fast.free_nodes, c_ref.free_nodes)
    assert np.array_equal(c_fast.hanging, c_ref.hanging)
    d = c_fast.T - c_ref.T
    assert d.nnz == 0 or abs(d).max() == 0.0


@pytest.mark.parametrize("p", [1, 2])
@pytest.mark.parametrize("periodic", [None, "all", "one"])
def test_uniform_fast_2d_L5(p, periodic):
    """Parity gate: 2-D L5 (32x32 elements) fast == general, all periodicity variants."""
    import diffsim.mesh.nodes as _N
    from diffsim.mesh.constraints import (build_constraints,
                                          _build_constraints_reference)
    per = (None if periodic is None
           else (True, True) if periodic == "all"
           else (True, False))
    tree = build_uniform(5, dim=2, periodic=per)
    saved = _N._uniform_complete_level
    try:
        _N._uniform_complete_level = lambda t: None
        m_gen = _N.build_mesh(tree, p=p)
    finally:
        _N._uniform_complete_level = saved
    m_fast = build_mesh(tree, p=p)
    assert np.array_equal(m_fast.node_icoords, m_gen.node_icoords)
    assert np.array_equal(m_fast.conn, m_gen.conn)
    assert np.array_equal(m_fast.boundary_nodes, m_gen.boundary_nodes)
    c_fast = build_constraints(m_fast)
    c_ref = _build_constraints_reference(m_fast)
    assert np.array_equal(c_fast.free_nodes, c_ref.free_nodes)
    d = c_fast.T - c_ref.T
    assert d.nnz == 0 or abs(d).max() == 0.0


@pytest.mark.parametrize("p", [1, 2])
@pytest.mark.parametrize("periodic", [None, "all", "one"])
def test_uniform_fast_3d_L4(p, periodic):
    """Parity gate: 3-D L4 (16x16x16 elements) fast == general, all periodicity variants."""
    import diffsim.mesh.nodes as _N
    from diffsim.mesh.constraints import (build_constraints,
                                          _build_constraints_reference)
    per = (None if periodic is None
           else (True, True, True) if periodic == "all"
           else (True, False, False))
    tree = build_uniform(4, dim=3, periodic=per)
    saved = _N._uniform_complete_level
    try:
        _N._uniform_complete_level = lambda t: None
        m_gen = _N.build_mesh(tree, p=p)
    finally:
        _N._uniform_complete_level = saved
    m_fast = build_mesh(tree, p=p)
    assert np.array_equal(m_fast.node_icoords, m_gen.node_icoords)
    assert np.array_equal(m_fast.conn, m_gen.conn)
    assert np.array_equal(m_fast.boundary_nodes, m_gen.boundary_nodes)
    c_fast = build_constraints(m_fast)
    c_ref = _build_constraints_reference(m_fast)
    assert np.array_equal(c_fast.free_nodes, c_ref.free_nodes)
    d = c_fast.T - c_ref.T
    assert d.nnz == 0 or abs(d).max() == 0.0


def test_uniform_fast_falls_through_for_adaptive():
    """Fast path must NOT trigger on non-uniform/adaptive trees (carved or refined).
    The general path must handle those correctly — fast path must be transparent."""
    from diffsim.mesh.constraints import build_constraints
    from diffsim.octree.carve import SphereOracle, carve
    # carved tree: _uniform_complete_level must return None (fewer elements)
    import diffsim.mesh.nodes as _N
    tree_carved, _ = carve(build_uniform(3), SphereOracle((0.5, 0.5, 0.5), 0.4))
    assert _N._uniform_complete_level(tree_carved) is None, \
        "Fast path incorrectly detected uniform level on carved tree"
    m = build_mesh(tree_carved, p=1)
    c = build_constraints(m)
    # carved trees have no hanging nodes only if all at same level; partial carve
    # may or may not — just verify constraints are shape-correct
    assert c.T.shape == (len(m.node_coords),) * 2
