import numpy as np
import pytest
from diffsim.octree.build import build_uniform, refine_elements
from diffsim.octree.balance import balance2to1
from diffsim.mesh.nodes import build_mesh
from diffsim.mesh.constraints import build_constraints
from diffsim.mesh.basis import basis_tables
from diffsim.assembly.operators import DeviceMesh, ConstrainedOperator, integrate_volume
from diffsim.assembly.dirichlet import DirichletPoisson
from diffsim.physics.poisson import l2_error

pytestmark = pytest.mark.tier3

def _dm(tree, p, device):
    m = build_mesh(tree, p=p)
    c = build_constraints(m)
    return m, c, DeviceMesh.from_mesh(m, c, basis_tables(p, dim=tree.dim), device)

@pytest.mark.parametrize("dim", [2, 4])
def test_kd_volume_and_symmetry(dim, device):
    t = build_uniform(2, dim=dim)
    mask = np.zeros(len(t), bool); mask[0] = True
    t = balance2to1(refine_elements(t, mask))
    m, c, dm = _dm(t, 1, device)
    assert abs(integrate_volume(dm) - 1.0) < 1e-13
    op = ConstrainedOperator(dm)
    rng = np.random.default_rng(5)
    v, w_ = rng.standard_normal(dm.n_free), rng.standard_normal(dm.n_free)
    assert abs(v @ op.matvec_numpy(w_) - w_ @ op.matvec_numpy(v)) < 1e-10

@pytest.mark.parametrize("dim", [2, 4])
def test_kd_linear_patch(dim, device):
    coef = np.arange(1, dim + 1, dtype=np.float64)
    LIN = lambda x: 1.0 + x @ coef
    ZERO = lambda x: np.zeros(len(x))
    t = build_uniform(2, dim=dim)
    mask = np.zeros(len(t), bool); mask[0] = True
    t = balance2to1(refine_elements(t, mask))
    for p in (1, 2):
        m, c, dm = _dm(t, p, device)
        u = DirichletPoisson(dm).solve(g_fn=LIN, f_fn=ZERO, tol=1e-13)
        assert l2_error(dm, u, LIN) < 1e-11

@pytest.mark.parametrize("dim,levels,expected", [(2, [3, 4, 5], 2.0), (4, [1, 2, 3], 2.0)])
def test_kd_mms_order(dim, levels, expected, device):
    U = lambda x: np.prod(np.sin(np.pi * x), axis=1)
    F = lambda x: dim * np.pi**2 * U(x)
    errs = []
    for lvl in levels:
        m, c, dm = _dm(build_uniform(lvl, dim=dim), 1, device)
        u = DirichletPoisson(dm).solve(g_fn=U, f_fn=F, tol=1e-13)
        errs.append(l2_error(dm, u, U))
    order = np.log2(errs[-2] / errs[-1])
    assert abs(order - expected) < 0.15, (dim, errs)

@pytest.mark.parametrize("device", [None], indirect=True)
def test_periodic_poisson_3d(device):
    # u periodic in x, Dirichlet in y,z: u = sin(2 pi x) sin(pi y) sin(pi z)
    U = lambda x: np.sin(2 * np.pi * x[:, 0]) * np.sin(np.pi * x[:, 1]) * np.sin(np.pi * x[:, 2])
    F = lambda x: (4 + 1 + 1) * np.pi**2 * U(x)
    errs = []
    for lvl in [3, 4]:
        t = build_uniform(lvl, dim=3, periodic=(True, False, False))
        m, c, dm = _dm(t, 1, device)
        u = DirichletPoisson(dm).solve(g_fn=U, f_fn=F, tol=1e-13)
        errs.append(l2_error(dm, u, U))
    assert abs(np.log2(errs[0] / errs[1]) - 2.0) < 0.2, errs
    # seam continuity: wrap nodes are single DOFs, so this holds by construction;
    # assert it anyway via icoords uniqueness
    assert len(np.unique(m.node_icoords, axis=0)) == len(m.node_icoords)
