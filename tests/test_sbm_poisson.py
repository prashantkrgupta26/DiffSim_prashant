"""Tier-5 SBM Poisson tests (M1a Tasks 6-7): the P4 keystone — SBM linear
patch on rotated geometries at all lambda, machine precision (spec S9.1) —
plus MMS convergence orders (Task 7 appends).

The SBM Dirichlet weak form is exact for linear u at ANY alpha, lambda, and
geometry: Su - g(x+d) vanishes identically and the consistency term equals
the true flux. A P4 failure at 1e-11 means the weak form or the geometry
cache is wrong — never loosen the tolerance (spec S9.2)."""
import numpy as np
import pytest
import torch
from diffsim.octree.build import build_uniform
from diffsim.mesh.nodes import build_mesh
from diffsim.mesh.constraints import build_constraints
from diffsim.mesh.basis import basis_tables
from diffsim.mesh.faces import face_tables
from diffsim.assembly.operators import DeviceMesh
from diffsim.geometry.csg import Sphere, Box
from diffsim.sbm.surrogate import classify_lambda, extract_surrogate, GeometryData
from diffsim.sbm.poisson import SBMPoisson

pytestmark = pytest.mark.tier5


def sbm_setup(oracle, level, p, lam, dim, device, domain="inside"):
    tree = build_uniform(level, dim=dim)
    ret, _ = classify_lambda(tree, oracle, lam, domain=domain)
    sf = extract_surrogate(ret)
    mesh = build_mesh(ret, p=p)
    cons = build_constraints(mesh)
    dm = DeviceMesh.from_mesh(mesh, cons, basis_tables(p, dim=dim), device)
    geo = GeometryData.evaluate(oracle, ret, sf, face_tables(p, dim), domain=domain)
    return dm, geo, sf


LIN = {
    2: (0.7, np.array([1.3, -0.4])),
    3: (0.7, np.array([1.3, -0.4, 0.9])),
    4: (0.7, np.array([1.3, -0.4, 0.9, -0.6])),
}
ZERO = lambda x: np.zeros(len(x))


def _patch(oracle, level, p, lam, dim, device, atol=1e-11, domain="inside",
           alpha=10.0):
    c0, cv = LIN[dim]
    u_lin = lambda x: c0 + x @ cv
    dm, geo, sf = sbm_setup(oracle, level, p, lam, dim, device, domain=domain)
    prob = SBMPoisson(dm, geo, sf, g_fn=u_lin, kappa=1.0, alpha=alpha)
    u = prob.solve(f_fn=ZERO, g_outer_fn=u_lin, tol=1e-14)
    err = np.abs(u - u_lin(dm.mesh.node_coords)).max()
    assert err < atol, err
    return dm, geo, sf


@pytest.mark.parametrize("lam", [0.5, 1.0])
@pytest.mark.parametrize("p", [1, 2])
def test_P4_patch_circle_2d(lam, p, device):
    _patch(Sphere((0.5, 0.5), 0.3), 5, p, lam, 2, device)


@pytest.mark.parametrize("lam", [0.5, 1.0])
def test_P4_patch_rotated_box_2d(lam, device):
    b = Box((0.5, 0.5), (0.23, 0.17), rotation=torch.tensor(0.4, dtype=torch.float64))
    _patch(b, 5, 1, lam, 2, device)


@pytest.mark.parametrize("lam", [0.5, 1.0])
def test_P4_patch_sphere_3d(lam, device):
    _patch(Sphere((0.5, 0.5, 0.5), 0.32), 4, 1, lam, 3, device)


def test_P4_patch_sphere_3d_p2(device):
    _patch(Sphere((0.5, 0.5, 0.5), 0.32), 3, 2, 1.0, 3, device)


def test_P4_patch_rotated_box_3d(device):
    b = Box((0.5, 0.5, 0.5), (0.25, 0.18, 0.22),
            rotation=torch.tensor([0.3, 0.5, 0.2], dtype=torch.float64))
    _patch(b, 4, 1, 1.0, 3, device)


def test_P4_patch_sphere_4d(device):
    # dim-coverage policy: SBM machinery is k-generic; the patch is the cheap
    # machine-precision structural check at k=4 (coarse level, p1, lam=1).
    _patch(Sphere((0.5,) * 4, 0.35), 3, 1, 1.0, 4, device)


def test_P4_patch_exterior_with_outer_dirichlet(device):
    # square minus disk (the M1b configuration): SBM on the disk + strong
    # outer Dirichlet, domain = {psi > 0} via the flag
    _patch(Sphere((0.5, 0.5), 0.25), 5, 1, 1.0, 2, device, domain="outside")


def test_alpha_insensitivity_of_patch(device):
    # keystone property: exactness does not depend on the penalty
    for alpha in (2.0, 10.0, 100.0):
        _patch(Sphere((0.5, 0.5), 0.3), 5, 1, 0.5, 2, device,
               atol=1e-10, alpha=alpha)


def test_operator_nonsymmetric_documented(device):
    # the adjoint-consistency shift term breaks symmetry: assert it, so a
    # future accidental "cg" swap fails loudly
    dm, geo, sf = sbm_setup(Sphere((0.5, 0.5), 0.3), 4, 1, 0.5, 2, device)
    prob = SBMPoisson(dm, geo, sf, g_fn=lambda x: x[:, 0])
    A, b, meta = prob.assemble(f_fn=ZERO)
    asym = abs(A - A.T).max()
    assert asym > 1e-12


def test_kappa_scaling_exact(device):
    # A(kappa) = kappa A(1); b = b_f + kappa b_g1 (meta carries the pieces
    # for the Task-10 kappa gradient)
    dm, geo, sf = sbm_setup(Sphere((0.5, 0.5), 0.3), 4, 1, 1.0, 2, device)
    g = lambda x: x[:, 0] + 0.2 * x[:, 1]
    f = lambda x: np.ones(len(x))
    p1 = SBMPoisson(dm, geo, sf, g_fn=g, kappa=1.0)
    p25 = SBMPoisson(dm, geo, sf, g_fn=g, kappa=2.5)
    A1, b1, m1 = p1.assemble(f_fn=f)
    A25, b25, m25 = p25.assemble(f_fn=f)
    assert abs(A25 - 2.5 * A1).max() < 1e-13
    assert np.allclose(b25 - b1, 1.5 * m1["bg1"], atol=1e-13)


def test_P4_patch_bicgstab_path(device):
    # keep the in-framework Krylov path honest (it becomes the matrix-free
    # M1b path): same machine-precision patch through bicgstab+Jacobi
    c0, cv = LIN[2]
    u_lin = lambda x: c0 + x @ cv
    dm, geo, sf = sbm_setup(Sphere((0.5, 0.5), 0.3), 4, 1, 1.0, 2, device)
    prob = SBMPoisson(dm, geo, sf, g_fn=u_lin)
    u = prob.solve(f_fn=ZERO, tol=1e-14, solver="bicgstab")
    assert np.abs(u - u_lin(dm.mesh.node_coords)).max() < 1e-11
