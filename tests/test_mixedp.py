import numpy as np
import pytest
from diffsim.octree.build import build_uniform, refine_elements
from diffsim.octree.balance import balance2to1
from diffsim.mesh.nodes import build_mesh
from diffsim.mesh.constraints import build_constraints

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


def test_p_transition_weights_minimum_rule():
    t, p = _mixed_tree_and_p()
    m = build_mesh(t, p)
    c = build_constraints(m)
    assert c.hanging.sum() > 0                     # p2 interface midpoints are hanging
    rowsum = np.asarray(c.T.sum(axis=1)).ravel()
    assert np.allclose(rowsum, 1.0, atol=1e-13)
    # every hanging row's nonzero weights must be from {0.25, 0.5} patterns
    # (linear trace at midpoints) or general only if h-hanging exists (none here)
    Tc = c.T.tocoo()
    hang_rows = np.where(c.hanging)[0]
    w = Tc.data[np.isin(Tc.row, hang_rows)]
    assert np.all(np.isin(np.round(w, 12), [0.25, 0.5]))


def test_mixed_linear_reproduction_and_free_corners():
    t, p = _mixed_tree_and_p()
    m = build_mesh(t, p)
    c = build_constraints(m)
    f = lambda x: 1.0 + 2 * x[:, 0] - 3 * x[:, 1] + 0.5 * x[:, 2]
    u_all = c.T @ f(m.node_coords[c.free_nodes])
    assert np.allclose(u_all, f(m.node_coords), atol=1e-12)
    # nodes of a p1 element are all free (corners are never hanging at equal level)
    assert not c.hanging[m.conn_of[1][0]].any()


def test_mixed_with_h_transitions_combined():
    # p2 band at x=0 on a tree that ALSO has h-refinement away from the band:
    # both knobs in one mesh, never on one face
    t = build_uniform(2, dim=3)
    far = t.anchors()[:, 0] >= (1 << 21) // 2      # refine far-half elements
    mask = np.zeros(len(t), bool); mask[np.where(far)[0][:4]] = True
    t = balance2to1(refine_elements(t, mask))
    p = np.ones(len(t), np.int8)
    p[t.anchors()[:, 0] == 0] = 2                  # p2 band untouched by refinement
    m = build_mesh(t, p)                            # must pass one-knob validation
    c = build_constraints(m)
    f = lambda x: 1.0 - x[:, 0] + 4 * x[:, 1] + 2 * x[:, 2]
    u_all = c.T @ f(m.node_coords[c.free_nodes])
    assert np.allclose(u_all, f(m.node_coords), atol=1e-12)


# ---------------------------------------------------------------------------
# Task 11: mixed-p assembly — per-bin kernel dispatch
# ---------------------------------------------------------------------------

import warp as wp  # noqa: E402 (already imported at session init in conftest)
from diffsim.mesh.basis import basis_tables
from diffsim.assembly.operators import DeviceMesh, ConstrainedOperator, assemble_csr
from diffsim.assembly.dirichlet import DirichletPoisson
from diffsim.physics.poisson import l2_error


def _mixed_dm(device, dim=3):
    t, p = _mixed_tree_and_p(dim)
    m = build_mesh(t, p)
    c = build_constraints(m)
    tb = {1: basis_tables(1, dim=dim), 2: basis_tables(2, dim=dim)}
    return m, c, DeviceMesh.from_mesh(m, c, tb, device)


@pytest.mark.tier3
def test_mixed_matvec_vs_assembled(device):
    m, c, dm = _mixed_dm(device)
    A = assemble_csr(dm)
    op = ConstrainedOperator(dm)
    rng = np.random.default_rng(9)
    for _ in range(10):
        x = rng.standard_normal(dm.n_free)
        diff = np.abs(A @ x - op.matvec_numpy(x)).max()
        assert diff < 1e-10 * max(1.0, np.abs(A @ x).max()), diff


@pytest.mark.tier3
def test_mixed_linear_patch_machine_precision(device):
    m, c, dm = _mixed_dm(device)
    LIN = lambda x: 1.0 + 2 * x[:, 0] - 3 * x[:, 1] + 0.5 * x[:, 2]
    u = DirichletPoisson(dm).solve(g_fn=LIN, f_fn=lambda x: np.zeros(len(x)), tol=1e-13)
    assert l2_error(dm, u, LIN) < 1e-11

def test_p_elem_out_of_range_raises():
    t = build_uniform(1, dim=3)
    with pytest.raises(ValueError, match="p_elem"):
        build_mesh(t, np.full(len(t), 3, np.int8))
