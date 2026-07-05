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


@pytest.mark.parametrize("lam", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("p", [1, 2])
def test_P4_patch_circle_2d(lam, p, device):
    _patch(Sphere((0.5, 0.5), 0.3), 5, p, lam, 2, device)


@pytest.mark.parametrize("lam", [0.5, 0.0])
def test_P4_patch_rotated_box_2d(lam, device):
    b = Box((0.5, 0.5), (0.23, 0.17), rotation=torch.tensor(0.4, dtype=torch.float64))
    _patch(b, 5, 1, lam, 2, device)


@pytest.mark.parametrize("lam", [0.5, 0.0])
def test_P4_patch_sphere_3d(lam, device):
    _patch(Sphere((0.5, 0.5, 0.5), 0.32), 4, 1, lam, 3, device)


def test_P4_patch_sphere_3d_p2(device):
    _patch(Sphere((0.5, 0.5, 0.5), 0.32), 3, 2, 0.0, 3, device)


def test_P4_patch_rotated_box_3d(device):
    b = Box((0.5, 0.5, 0.5), (0.25, 0.18, 0.22),
            rotation=torch.tensor([0.3, 0.5, 0.2], dtype=torch.float64))
    _patch(b, 4, 1, 0.0, 3, device)


def test_P4_patch_sphere_4d(device):
    # dim-coverage policy: SBM machinery is k-generic; the patch is the cheap
    # machine-precision structural check at k=4 (coarse level, p1, lam=1).
    _patch(Sphere((0.5,) * 4, 0.35), 3, 1, 0.0, 4, device)


def test_P4_patch_exterior_with_outer_dirichlet(device):
    # square minus disk (the M1b configuration): SBM on the disk + strong
    # outer Dirichlet, domain = {psi > 0} via the flag
    _patch(Sphere((0.5, 0.5), 0.25), 5, 1, 0.0, 2, device, domain="outside")


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
    dm, geo, sf = sbm_setup(Sphere((0.5, 0.5), 0.3), 4, 1, 0.0, 2, device)
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
    # M1b path). TINY problem (level 3) at tol 1e-12: WSL2 host-sync dot()
    # latency makes larger bicgstab solves minutes-long (recorded M1a
    # finding) — this test checks correctness of the path, not endurance.
    c0, cv = LIN[2]
    u_lin = lambda x: c0 + x @ cv
    dm, geo, sf = sbm_setup(Sphere((0.5, 0.5), 0.35), 3, 1, 0.0, 2, device)
    prob = SBMPoisson(dm, geo, sf, g_fn=u_lin)
    u = prob.solve(f_fn=ZERO, tol=1e-12, maxiter=3000, solver="bicgstab")
    assert np.abs(u - u_lin(dm.mesh.node_coords)).max() < 1e-9


# ---------------------------------------------------------------------------
# Task 7: MMS convergence orders (error on Omega, spec S13.3.3 warning)
# ---------------------------------------------------------------------------
from diffsim.physics.poisson import l2_error_masked

U2 = lambda x: np.sin(np.pi * x[:, 0]) * np.sin(np.pi * x[:, 1])
F2 = lambda x: 2 * np.pi ** 2 * U2(x)
U3 = lambda x: (np.sin(np.pi * x[:, 0]) * np.sin(np.pi * x[:, 1])
                * np.sin(np.pi * x[:, 2]))
F3 = lambda x: 3 * np.pi ** 2 * U3(x)

MMS_ERRS = {}   # captured ladders, locked into m1a_baselines.json in Task 11


def _mms_ladder(oracle, levels, p, lam, dim, device, u_fn, f_fn, kappa=1.0,
                domain="inside"):
    sgn = -1.0 if domain == "inside" else 1.0
    errs = []
    for lv in levels:
        dm, geo, sf = sbm_setup(oracle, lv, p, lam, dim, device, domain=domain)
        prob = SBMPoisson(dm, geo, sf, g_fn=u_fn, kappa=kappa)
        fk = (lambda x: kappa * f_fn(x))
        u = prob.solve(f_fn=fk, g_outer_fn=u_fn)
        errs.append(l2_error_masked(dm, u, u_fn,
                                    lambda x: sgn * oracle.classify(x) > 0))
    return np.array(errs)


def _slope(errs, levels):
    h = 2.0 ** -np.asarray(levels, float)
    k = min(3, len(errs))
    return np.polyfit(np.log(h[-k:]), np.log(errs[-k:]), 1)[0]


def test_mms_disk_p1_order2(device):
    levels = [4, 5, 6, 7]
    errs = _mms_ladder(Sphere((0.5, 0.5), 0.3), levels, 1, 0.0, 2, device, U2, F2)
    MMS_ERRS["disk_p1"] = errs
    assert abs(_slope(errs, levels) - 2.0) < 0.10, errs


def test_mms_disk_p1_lam05(device):
    levels = [4, 5, 6]
    errs = _mms_ladder(Sphere((0.5, 0.5), 0.3), levels, 1, 0.5, 2, device, U2, F2)
    MMS_ERRS["disk_p1_lam05"] = errs
    # lam=0.5 retained sets reshuffle between levels: prefactor noise on top
    # of order >= 2 (measured pairwise orders 2.06/2.47) — assert one-sided
    assert _slope(errs, levels) > 1.85, errs


def test_mms_disk_p2_order3(device):
    levels = [4, 5, 6]
    errs = _mms_ladder(Sphere((0.5, 0.5), 0.3), levels, 2, 0.0, 2, device, U2, F2)
    MMS_ERRS["disk_p2"] = errs
    # second-order shift restores order 3 with early SUPERconvergence
    # (measured pairwise 3.44 -> 2.98): resolved pair at 3 +- 0.15, plus a
    # one-sided guard that the overall slope stays >= cubic-ish
    order = float(np.log2(errs[-2] / errs[-1]))
    assert abs(order - 3.0) < 0.15, errs
    assert _slope(errs, levels) > 2.9, errs


def test_mms_exterior_p1_order2(device):
    # square minus disk, domain="outside" (M1b configuration), strong outer
    # Dirichlet + SBM disk
    levels = [4, 5, 6, 7]
    errs = _mms_ladder(Sphere((0.5, 0.5), 0.25), levels, 1, 0.0, 2, device,
                       U2, F2, domain="outside")
    MMS_ERRS["exterior_p1"] = errs
    assert abs(_slope(errs, levels) - 2.0) < 0.10, errs


def test_mms_sphere_3d_p1_order2(device):
    levels = [3, 4, 5]
    errs = _mms_ladder(Sphere((0.5, 0.5, 0.5), 0.32), levels, 1, 0.0, 3,
                       device, U3, F3)
    MMS_ERRS["sphere3d_p1"] = errs
    # level 3 is preasymptotic (h/r = 0.39): fit the resolved pair
    order = float(np.log2(errs[-2] / errs[-1]))
    assert abs(order - 2.0) < 0.15, errs


def test_mms_kappa_scaling_identical_errors(device):
    # kappa is a pure scaling of the operator and f: identical solutions,
    # hence identical errors (direct solver: near machine-identical)
    e1 = _mms_ladder(Sphere((0.5, 0.5), 0.3), [4, 5], 1, 0.0, 2, device, U2, F2,
                     kappa=1.0)
    e25 = _mms_ladder(Sphere((0.5, 0.5), 0.3), [4, 5], 1, 0.0, 2, device, U2, F2,
                      kappa=2.5)
    assert np.allclose(e1, e25, rtol=1e-12)


# ---------------------------------------------------------------------------
# Task 7b: spatially-varying kappa (closure-hook pathway, spec S6.3)
# ---------------------------------------------------------------------------
def _kap_lin(x):
    # linear, positive on the unit box in any dim: 1 + x + 2y (+0*z...)
    return 1.0 + x[:, 0] + 2.0 * x[:, 1]


@pytest.mark.parametrize("dim,level", [(2, 5), (3, 3), (4, 3)])
def test_varkappa_patch_exact(dim, level, device):
    # linear kappa + linear u: -div(kappa grad u) = -grad(kappa).b constant,
    # every integrand polynomial within quadrature exactness -> machine zero.
    # Runs at k = 2, 3, 4 per the dim-coverage policy.
    c0, cv = LIN[dim]
    u_lin = lambda x: c0 + x @ cv
    f = lambda x: np.full(len(x), -(1.0 * cv[0] + 2.0 * cv[1]))
    r = {2: 0.3, 3: 0.32, 4: 0.35}[dim]
    dm, geo, sf = sbm_setup(Sphere((0.5,) * dim, r), level, 1, 0.0, dim, device)
    prob = SBMPoisson(dm, geo, sf, g_fn=u_lin, kappa=_kap_lin)
    u = prob.solve(f_fn=f)
    assert np.abs(u - u_lin(dm.mesh.node_coords)).max() < 1e-10


def test_varkappa_scalar_consistency(device):
    # a constant callable must reproduce the scalar path to near machine
    dm, geo, sf = sbm_setup(Sphere((0.5, 0.5), 0.3), 4, 1, 0.0, 2, device)
    g = lambda x: x[:, 0] + 0.2 * x[:, 1]
    f = lambda x: np.ones(len(x))
    u_s = SBMPoisson(dm, geo, sf, g_fn=g, kappa=2.5).solve(f_fn=f)
    u_f = SBMPoisson(dm, geo, sf, g_fn=g,
                     kappa=lambda x: np.full(len(x), 2.5)).solve(f_fn=f)
    assert np.abs(u_s - u_f).max() < 1e-12


def test_varkappa_positivity_validated(device):
    dm, geo, sf = sbm_setup(Sphere((0.5, 0.5), 0.3), 4, 1, 0.0, 2, device)
    bad = lambda x: x[:, 0] - 0.5                     # crosses zero
    with pytest.raises(ValueError, match="positive"):
        SBMPoisson(dm, geo, sf, g_fn=lambda x: x[:, 0],
                   kappa=bad).assemble(f_fn=ZERO)


def test_mms_varkappa_order2(device):
    # u = sin(pi x) sin(pi y), kappa = 1 + x + 2y:
    # f = -grad(kappa).grad(u) + 2 pi^2 kappa u
    def f(x):
        s = np.sin(np.pi * x[:, 0]) * np.sin(np.pi * x[:, 1])
        ux = np.pi * np.cos(np.pi * x[:, 0]) * np.sin(np.pi * x[:, 1])
        uy = np.pi * np.sin(np.pi * x[:, 0]) * np.cos(np.pi * x[:, 1])
        return -(1.0 * ux + 2.0 * uy) + 2.0 * np.pi ** 2 * _kap_lin(x) * s
    levels = [4, 5, 6]
    errs = []
    for lv in levels:
        dm, geo, sf = sbm_setup(Sphere((0.5, 0.5), 0.3), lv, 1, 0.0, 2, device)
        prob = SBMPoisson(dm, geo, sf, g_fn=U2, kappa=_kap_lin)
        u = prob.solve(f_fn=f)
        errs.append(l2_error_masked(dm, u, U2,
                                    lambda x: Sphere((0.5, 0.5), 0.3).classify(x) < 0))
    MMS_ERRS["disk_varkappa_p1"] = np.array(errs)
    # superconvergent first gap (measured pairwise 2.57 -> 2.02): assert the
    # resolved pair, plus a one-sided slope guard
    order = float(np.log2(errs[-2] / errs[-1]))
    assert abs(order - 2.0) < 0.10, errs
    assert _slope(np.array(errs), levels) > 1.9, errs
