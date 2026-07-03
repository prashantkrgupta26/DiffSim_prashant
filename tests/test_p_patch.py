import numpy as np
import pytest
from diffsim.octree.build import build_uniform, refine_elements
from diffsim.mesh.nodes import build_mesh
from diffsim.mesh.constraints import build_constraints
from diffsim.mesh.basis import basis_tables
from diffsim.assembly.operators import DeviceMesh
from diffsim.assembly.dirichlet import DirichletPoisson
from diffsim.physics.poisson import l2_error

pytestmark = pytest.mark.tier2

LIN = lambda x: 1.0 + 2.0 * x[:, 0] - 3.0 * x[:, 1] + 0.5 * x[:, 2]
ZERO = lambda x: np.zeros(len(x))

def _solve(tree, p, device):
    m = build_mesh(tree, p=p)
    c = build_constraints(m)
    dm = DeviceMesh.from_mesh(m, c, basis_tables(p), device)
    solver = DirichletPoisson(dm)
    u = solver.solve(g_fn=LIN, f_fn=ZERO, tol=1e-13)
    return m, dm, u

def test_P1_P2_linear_patch_uniform(device):
    for p in (1, 2):
        m, dm, u = _solve(build_uniform(2), p, device)
        err = l2_error(dm, u, LIN)
        assert err < 1e-11, (p, err)       # machine-precision patch (cuFEM criterion)

def test_P3_linear_patch_adaptive(device):
    t = build_uniform(2)
    mask = np.zeros(len(t), bool); mask[[0, 9, 27]] = True
    t = refine_elements(t, mask)
    from diffsim.octree.balance import balance2to1
    t = balance2to1(t)
    for p in (1, 2):
        m, dm, u = _solve(t, p, device)
        err = l2_error(dm, u, LIN)
        assert err < 1e-11, (p, err)       # hanging constraints must be exact for linears
