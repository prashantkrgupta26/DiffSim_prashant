"""M2-A1 gates: scalar advection-diffusion brick MMS (2D/3D, p1/p2),
Poisson limit, and advection-dominated sanity."""
import numpy as np
import pytest
from scipy.sparse.linalg import splu

from diffsim.octree.build import build_uniform
from diffsim.mesh.nodes import build_mesh
from diffsim.mesh.constraints import build_constraints
from diffsim.mesh.basis import basis_tables
from diffsim.assembly.operators import DeviceMesh
from diffsim.physics.scalar_transport import assemble_scalar_ad
from diffsim.physics.poisson import gauss_points, l2_error

pytestmark = pytest.mark.tier2


def _solve(dim, level, p, a_fn, kappa, u_star, f_fn, sigma=0.0,
           device="cuda:0"):
    tree = build_uniform(level, dim=dim)
    mesh = build_mesh(tree, p=p)
    cons = build_constraints(mesh)
    dm = DeviceMesh.from_mesh(mesh, cons, basis_tables(p, dim=dim),
                              device)
    xq = gauss_points(mesh, dm.tables_by_p)
    aq = {pv: a_fn(xq[pv]) for pv in xq}
    fq = {pv: f_fn(xq[pv]) for pv in xq}
    A, b = assemble_scalar_ad(dm, aq, fq, kappa, sigma=sigma)
    coords = mesh.node_coords[cons.free_nodes]
    bdry = np.zeros(len(coords), bool)
    for c in range(dim):
        bdry |= (np.abs(coords[:, c]) < 1e-12) | \
                (np.abs(coords[:, c] - 1.0) < 1e-12)
    A = A.tolil()
    for i in np.where(bdry)[0]:
        A.rows[i] = [int(i)]
        A.data[i] = [1.0]
        b[i] = u_star(coords[i:i + 1])[0]
    x = splu(A.tocsr().tocsc()).solve(b)
    u_all = np.asarray(cons.T @ x)
    return l2_error(dm, u_all, u_star)


def _case(dim):
    if dim == 2:
        u = lambda x: np.sin(np.pi * x[:, 0]) * np.cos(np.pi * x[:, 1])
        gu = lambda x: np.stack(
            [np.pi * np.cos(np.pi * x[:, 0]) * np.cos(np.pi * x[:, 1]),
             -np.pi * np.sin(np.pi * x[:, 0]) * np.sin(np.pi * x[:, 1])],
            axis=1)
        lap = lambda x: -2 * np.pi ** 2 * u(x)
    else:
        u = lambda x: (np.sin(np.pi * x[:, 0]) * np.cos(np.pi * x[:, 1])
                       * np.sin(np.pi * x[:, 2]))
        gu = lambda x: np.stack(
            [np.pi * np.cos(np.pi * x[:, 0]) * np.cos(np.pi * x[:, 1])
             * np.sin(np.pi * x[:, 2]),
             -np.pi * np.sin(np.pi * x[:, 0]) * np.sin(np.pi * x[:, 1])
             * np.sin(np.pi * x[:, 2]),
             np.pi * np.sin(np.pi * x[:, 0]) * np.cos(np.pi * x[:, 1])
             * np.cos(np.pi * x[:, 2])], axis=1)
        lap = lambda x: -3 * np.pi ** 2 * u(x)
    return u, gu, lap


def _a_rot(x):
    a = np.zeros_like(x)
    a[:, 0] = 1.0 + 0.5 * x[:, 1]
    a[:, 1] = -0.5 + 0.3 * x[:, 0]
    return a


@pytest.mark.parametrize("dim,levels,p,order_lo", [
    (2, (4, 5, 6), 1, 1.85),
    (2, (3, 4, 5), 2, 2.7),
    (3, (3, 4), 1, 1.7),
])
def test_scalar_ad_mms_orders(dim, levels, p, order_lo, device):
    """A1 gate: advection-diffusion MMS orders (kappa=0.7, sigma=0.4,
    rotating advecting field)."""
    u, gu, lap = _case(dim)
    kappa, sigma = 0.7, 0.4

    def f_fn(x):
        a = _a_rot(x[:, :dim])
        adv = (a[:, :dim] * gu(x)[:, :dim]).sum(1) if dim == 2 else \
            (np.pad(a, ((0, 0), (0, 1)))[:, :dim] * gu(x)).sum(1)
        return sigma * u(x) + adv - kappa * lap(x)

    def a_fn(x):
        a = np.zeros((len(x), dim))
        a[:, :2] = _a_rot(x)[:, :2]
        return a

    errs = [
        _solve(dim, lv, p, a_fn, kappa, u, f_fn, sigma, device)
        for lv in levels]
    orders = [np.log2(errs[i] / errs[i + 1]) for i in range(len(errs) - 1)]
    print(f"{dim}D p{p}: errs {[f'{e:.2e}' for e in errs]} "
          f"orders {[f'{o:.2f}' for o in orders]}")
    assert orders[-1] > order_lo, (errs, orders)


def test_poisson_limit(device):
    """a=0, sigma=0 must reproduce the Poisson brick's accuracy class."""
    u, gu, lap = _case(2)
    err = _solve(2, 5, 1, lambda x: np.zeros((len(x), 2)), 1.0, u,
                 lambda x: -lap(x), 0.0, device)
    assert err < 2e-3, err


def test_advection_dominated_stability(device):
    """Pe_h >> 1 sanity: SUPG keeps the solution bounded (no blowup) on
    an advection-dominated case (kappa=1e-4, |a|~1)."""
    u, gu, lap = _case(2)
    kappa = 1e-4

    def f_fn(x):
        a = _a_rot(x)
        return (a * gu(x)).sum(1) - kappa * lap(x)

    err = _solve(2, 5, 1, _a_rot, kappa, u, f_fn, 0.0, device)
    assert np.isfinite(err) and err < 0.05, err
