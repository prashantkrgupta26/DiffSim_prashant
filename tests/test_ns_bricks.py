"""M1b Task 5 gates, first battery: the linearized monolithic (u,p) brick on
steady problems — Stokes and Oseen MMS with the classic solenoidal
vortex (u = 0 on the box boundary; equal-order PSPG; pressure pinned).
Transient Taylor-Green joins with the stepper."""
import numpy as np
import pytest
from scipy.sparse.linalg import splu
from diffsim.octree.build import build_uniform
from diffsim.mesh.nodes import build_mesh
from diffsim.mesh.constraints import build_constraints
from diffsim.mesh.basis import basis_tables
from diffsim.assembly.operators import DeviceMesh
from diffsim.api.ns_bricks import assemble_linear_ns
from diffsim.physics.poisson import gauss_points

pytestmark = pytest.mark.tier5

PI = np.pi


def u_star(x):
    u1 = np.sin(PI * x[:, 0]) ** 2 * np.sin(2 * PI * x[:, 1])
    u2 = -np.sin(2 * PI * x[:, 0]) * np.sin(PI * x[:, 1]) ** 2
    return np.stack([u1, u2], axis=1)


def p_star(x):
    return np.sin(PI * x[:, 0]) * np.cos(PI * x[:, 1])


def f_star(x, nu, sigma, oseen):
    sx, sy = np.sin(PI * x[:, 0]), np.sin(PI * x[:, 1])
    s2x, s2y = np.sin(2 * PI * x[:, 0]), np.sin(2 * PI * x[:, 1])
    c2x, c2y = np.cos(2 * PI * x[:, 0]), np.cos(2 * PI * x[:, 1])
    u1 = sx ** 2 * s2y
    u2 = -s2x * sy ** 2
    du1x = PI * s2x * s2y
    du1y = 2 * PI * sx ** 2 * c2y
    du2x = -2 * PI * c2x * sy ** 2
    du2y = -PI * s2x * s2y
    lap1 = 2 * PI ** 2 * c2x * s2y - 4 * PI ** 2 * sx ** 2 * s2y
    lap2 = 4 * PI ** 2 * s2x * sy ** 2 - 2 * PI ** 2 * s2x * c2y
    dpx = PI * np.cos(PI * x[:, 0]) * np.cos(PI * x[:, 1])
    dpy = -PI * np.sin(PI * x[:, 0]) * np.sin(PI * x[:, 1])
    conv1 = (u1 * du1x + u2 * du1y) if oseen else 0.0
    conv2 = (u1 * du2x + u2 * du2y) if oseen else 0.0
    f1 = sigma * u1 + conv1 + dpx - nu * lap1
    f2 = sigma * u2 + conv2 + dpy - nu * lap2
    return np.stack([f1, f2], axis=1)


def _solve(level, nu, oseen, device, sigma=0.0):
    tree = build_uniform(level, dim=2)
    mesh = build_mesh(tree, p=1)
    cons = build_constraints(mesh)
    dm = DeviceMesh.from_mesh(mesh, cons, basis_tables(1, dim=2), device)
    ndof = 3
    xq = gauss_points(mesh, dm.tables_by_p)
    aq = {pv: (u_star(xq[pv]) if oseen else np.zeros((len(xq[pv]), 2)))
          for pv in xq}
    dq = {pv: np.zeros(len(xq[pv])) for pv in xq}   # div u* = 0 analytically
    fq = {pv: f_star(xq[pv], nu, sigma, oseen) for pv in xq}
    A, b = assemble_linear_ns(dm, aq, dq, fq, nu, sigma=sigma)
    # strong u = 0 on the box boundary; pin p at free node 0
    nfree = cons.T.shape[1]
    bdry = mesh.boundary_nodes[cons.free_nodes]
    rows = []
    for i in np.where(bdry)[0]:
        rows += [i * ndof, i * ndof + 1]
    rows.append(0 * ndof + 2)                       # pressure pin
    A = A.tolil()
    for r in rows:
        A.rows[r] = [r]
        A.data[r] = [1.0]
    A = A.tocsr()
    b[rows] = 0.0
    coords0 = mesh.node_coords[cons.free_nodes][0:1]
    b[2] = p_star(coords0)[0]
    x = splu(A.tocsc()).solve(b)
    x_all = x.reshape(nfree, ndof)
    coords = mesh.node_coords[cons.free_nodes]
    eu = x_all[:, :2] - u_star(coords)
    ep = x_all[:, 2] - p_star(coords)
    hu = np.sqrt((eu ** 2).sum(1).mean())
    hp = np.sqrt((ep ** 2).mean())
    return hu, hp


@pytest.mark.parametrize("oseen", [False, True])
def test_steady_mms_orders(oseen, device):
    nu = 0.1
    errs = [_solve(lv, nu, oseen, device) for lv in (3, 4, 5)]
    eu = [e[0] for e in errs]
    ep = [e[1] for e in errs]
    ou = [np.log2(eu[i] / eu[i + 1]) for i in range(2)]
    op_ = [np.log2(ep[i] / ep[i + 1]) for i in range(2)]
    assert ou[-1] > 1.8, (eu, ou)                   # velocity order 2
    assert op_[-1] > 1.2, (ep, op_)                 # pressure >= ~1.5 typ.
    assert eu[-1] < 2e-2, eu


def test_low_viscosity_stability(device):
    # advection-dominated: nu = 1e-3 at level 4 — SUPG keeps it clean
    hu, hp = _solve(4, 1e-3, True, device)
    assert np.isfinite(hu) and hu < 0.5, (hu, hp)
