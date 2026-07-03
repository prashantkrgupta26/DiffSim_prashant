import numpy as np
import pytest
import warp as wp
from diffsim.octree.build import build_uniform, refine_elements
from diffsim.mesh.nodes import build_mesh
from diffsim.mesh.constraints import build_constraints
from diffsim.mesh.basis import basis_tables
from diffsim.assembly.operators import DeviceMesh, ConstrainedOperator, integrate_volume

pytestmark = pytest.mark.tier3

def _setup(p, adaptive=False, device="cpu"):
    t = build_uniform(2)
    if adaptive:
        mask = np.zeros(len(t), bool); mask[0] = True
        t = refine_elements(t, mask)
    m = build_mesh(t, p=p)
    c = build_constraints(m)
    dm = DeviceMesh.from_mesh(m, c, basis_tables(p), device)
    return m, c, dm

def test_Q2_volume_conservation(device):
    for p in (1, 2):
        _, _, dm = _setup(p, adaptive=True, device=device)
        vol = integrate_volume(dm)
        assert abs(vol - 1.0) < 1e-13

def test_G1_operator_symmetry(device):
    for p in (1, 2):
        for adaptive in (False, True):
            m, c, dm = _setup(p, adaptive, device)
            op = ConstrainedOperator(dm)
            nf = len(c.free_nodes)
            rng = np.random.default_rng(3)
            for _ in range(5):
                v = rng.standard_normal(nf); w_ = rng.standard_normal(nf)
                Av = op.matvec_numpy(v)
                Aw = op.matvec_numpy(w_)
                assert abs(v @ Aw - w_ @ Av) < 1e-10 * (abs(v @ Av) + 1)

def test_G3_positive_semidefinite(device):
    m, c, dm = _setup(1, device=device)
    op = ConstrainedOperator(dm)
    rng = np.random.default_rng(4)
    for _ in range(20):
        v = rng.standard_normal(len(c.free_nodes))
        assert v @ op.matvec_numpy(v) >= -1e-10
    one = np.ones(len(c.free_nodes))
    assert np.abs(op.matvec_numpy(one)).max() < 1e-11   # constants in nullspace (pure Neumann)

def test_S3_matrix_free_vs_assembled(device):
    from diffsim.assembly.operators import assemble_csr, operator_diagonal
    for p in (1, 2):
        m, c, dm = _setup(p, adaptive=True, device=device)
        A = assemble_csr(dm)
        op = ConstrainedOperator(dm)
        rng = np.random.default_rng(5)
        for _ in range(20):
            x = rng.standard_normal(dm.n_free)
            diff = np.abs(A @ x - op.matvec_numpy(x)).max()
            assert diff < 1e-10 * max(1.0, np.abs(A @ x).max()), (p, diff)
        d = operator_diagonal(dm)
        assert np.allclose(d, A.diagonal(), atol=1e-12)
