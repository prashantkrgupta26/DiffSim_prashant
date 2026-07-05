"""Tier-5 SBM Neumann tests (M1a Task 8): the verified Eq.-21 form (local-p-
refinement draft; production HTEquation.h:1928-37) — flux enforced along the
TRUE normal with first-order + (p2-gated) Hessian shift, area-corrected by
a = n.n_tilde, minus the surrogate-normal flux. Plus the S13.3.3 acceptance:
p1-everywhere degrades, a p2 band restores order 2, layer-sweep insensitive.

q_fn contract (production default): returns the kappa-NORMALIZED flux
grad(u).n at the MAPPED point M(x) = x + d; kernels multiply by kappa.
"""
import numpy as np
import pytest
from diffsim.octree.build import build_uniform
from diffsim.mesh.nodes import build_mesh
from diffsim.mesh.constraints import build_constraints
from diffsim.mesh.basis import basis_tables
from diffsim.mesh.faces import face_tables
from diffsim.assembly.operators import DeviceMesh
from diffsim.geometry.csg import Sphere
from diffsim.sbm.surrogate import (classify_lambda, extract_surrogate,
                                   GeometryData, p2_band)
from diffsim.sbm.poisson import SBMPoisson, surrogate_flux
from diffsim.physics.poisson import l2_error_masked

pytestmark = pytest.mark.tier5

R = 0.25
CTR2 = (0.5, 0.5)

U2 = lambda x: np.sin(np.pi * x[:, 0]) * np.cos(np.pi * x[:, 1])
F2 = lambda x: 2 * np.pi ** 2 * U2(x)


def _grad_u2(x):
    gx = np.pi * np.cos(np.pi * x[:, 0]) * np.cos(np.pi * x[:, 1])
    gy = -np.pi * np.sin(np.pi * x[:, 0]) * np.sin(np.pi * x[:, 1])
    return np.stack([gx, gy], axis=1)


def _q_of(oracle, domain="outside"):
    """kappa-normalized true flux grad(u*).n on Gamma, evaluated at mapped
    points; n = domain-outward normal (into the disk for exterior domains)."""
    sgn = 1.0 if domain == "inside" else -1.0

    def q_fn(y):                      # y = mapped points on Gamma
        n = sgn * (y - np.asarray(CTR2)) / np.linalg.norm(
            y - np.asarray(CTR2), axis=1, keepdims=True)
        return np.einsum("id,id->i", _grad_u2(y), n)
    return q_fn


def _neumann_setup(level, p_elem_or_p, device, n1_geo=None):
    """Exterior square-minus-disk: strong outer Dirichlet from u*, SBM
    NEUMANN on the disk. p_elem_or_p: int (uniform) or per-element array."""
    oracle = Sphere(CTR2, R)
    tree = build_uniform(level, dim=2)
    ret, _ = classify_lambda(tree, oracle, 1.0, domain="outside")
    sf = extract_surrogate(ret)
    if isinstance(p_elem_or_p, int):
        p_elem = p_elem_or_p
        p_face = p_elem_or_p
    else:
        p_elem = p_elem_or_p(ret, sf)
        p_face = int(np.asarray(p_elem)[sf.elem].max())
    mesh = build_mesh(ret, p=p_elem)
    cons = build_constraints(mesh)
    tables = {pv: basis_tables(pv, dim=2) for pv in mesh.bins}
    dm = DeviceMesh.from_mesh(mesh, cons, tables, device)
    ftab = face_tables(p_face, 2)
    geo = GeometryData.evaluate(oracle, ret, sf, ftab, domain="outside")
    return oracle, dm, geo, sf


def _solve_neumann(level, p_spec, device, corr_override=None):
    oracle, dm, geo, sf = _neumann_setup(level, p_spec, device)
    if corr_override is not None:
        geo = GeometryData(geo.xq, geo.d, geo.n,
                           np.full_like(geo.corr, corr_override), geo.ok,
                           geo.domain)
    prob = SBMPoisson(dm, geo=None, sf=None,
                      neumann=(sf, geo, _q_of(oracle)))
    u = prob.solve(f_fn=F2, g_outer_fn=U2)
    err = l2_error_masked(dm, u, U2, lambda x: oracle.classify(x) > 0)
    flux = surrogate_flux(dm, sf, geo, u)
    return u, err, flux, dm, geo, sf


def _true_total_flux():
    # F* = int_Gamma grad(u*).n dGamma, n into the disk (domain-outward)
    th = np.linspace(0, 2 * np.pi, 4096, endpoint=False)
    y = np.stack([CTR2[0] + R * np.cos(th), CTR2[1] + R * np.sin(th)], axis=1)
    n = -(y - np.asarray(CTR2)) / R
    q = np.einsum("id,id->i", _grad_u2(y), n)
    return q.sum() * (2 * np.pi * R / len(th))


def _band(n_layers):
    def marker(ret, sf):
        return p2_band(ret, sf, n_layers=n_layers)
    return marker


def test_neumann_patch_2d(device):
    # linear u: Hessian zero, shift exact -> machine-precision patch through
    # the Neumann path (outer Dirichlet pins the solution)
    oracle, dm, geo, sf = _neumann_setup(4, 1, device)
    c0, cv = 0.7, np.array([1.3, -0.4])
    u_lin = lambda x: c0 + x @ cv

    def q_lin(y):
        n = -(y - np.asarray(CTR2)) / np.linalg.norm(
            y - np.asarray(CTR2), axis=1, keepdims=True)
        return n @ cv
    prob = SBMPoisson(dm, geo=None, sf=None, neumann=(sf, geo, q_lin))
    u = prob.solve(f_fn=lambda x: np.zeros(len(x)), g_outer_fn=u_lin)
    assert np.abs(u - u_lin(dm.mesh.node_coords)).max() < 1e-10


@pytest.mark.parametrize("dim,level,r", [(3, 3, 0.3), (4, 3, 0.35)])
def test_neumann_patch_highdim(dim, level, r, device):
    # dim-coverage policy: Neumann structural patch at k=3 and k=4
    ctr = (0.5,) * dim
    oracle = Sphere(ctr, r)
    tree = build_uniform(level, dim=dim)
    ret, _ = classify_lambda(tree, oracle, 1.0, domain="outside")
    sf = extract_surrogate(ret)
    mesh = build_mesh(ret, p=1)
    cons = build_constraints(mesh)
    dm = DeviceMesh.from_mesh(mesh, cons, basis_tables(1, dim=dim), device)
    geo = GeometryData.evaluate(oracle, ret, sf, face_tables(1, dim),
                                domain="outside")
    cv = np.arange(1, dim + 1, dtype=np.float64)
    u_lin = lambda x: 0.7 + x @ cv

    def q_lin(y):
        n = -(y - np.asarray(ctr)) / np.linalg.norm(
            y - np.asarray(ctr), axis=1, keepdims=True)
        return n @ cv
    prob = SBMPoisson(dm, geo=None, sf=None, neumann=(sf, geo, q_lin))
    u = prob.solve(f_fn=lambda x: np.zeros(len(x)), g_outer_fn=u_lin)
    assert np.abs(u - u_lin(dm.mesh.node_coords)).max() < 1e-10


def test_area_correction_pi4_solve_level(device):
    """LOCKED: without the a = n.n_tilde factor the imposed flux DATA is
    inflated by the staircase ratio (4/pi in 2D). Care point (measured): the
    sin*cos MMS has ~zero NET flux, making the inflation invisible — the
    lock needs a flux-carrying solution. u* = |x-c|^2: q = -2R constant,
    F* = -4 pi R^2; second-order Taylor is exact on quadratics, so the
    corrected scheme is near-exact while corr-off shows the O(1) pathology
    in both L2 and the conservation identity (flux off by ~(4/pi - 1)F*)."""
    uq = lambda x: (x[:, 0] - CTR2[0]) ** 2 + (x[:, 1] - CTR2[1]) ** 2
    fq = lambda x: np.full(len(x), -4.0)
    q_const = lambda y: np.full(len(y), -2.0 * R)
    Fstar = -4.0 * np.pi * R ** 2

    def run(uncorrected_data, level=5):
        oracle, dm, geo, sf = _neumann_setup(level, _band(3), device)
        if uncorrected_data:
            # Isolate the DATA-term area correction (the pi/4 mechanism):
            # impose qbar/corr so the data term integrates the UNCORRECTED
            # flux density over the staircase — total imposed flux becomes
            # ~(4/pi) F* — while the LHS stays consistent. (Overriding corr
            # everywhere also perturbs the LHS a-term and partially cancels:
            # measured 11.4% vs 7.6% — not a clean lock.)
            corr = geo.corr.copy()
            q_fn = lambda y: q_const(y) / corr
        else:
            q_fn = q_const
        prob = SBMPoisson(dm, geo=None, sf=None, neumann=(sf, geo, q_fn))
        u = prob.solve(f_fn=fq, g_outer_fn=uq)
        err = l2_error_masked(dm, u, uq,
                              lambda x: Sphere(CTR2, R).classify(x) > 0)
        flux = surrogate_flux(dm, sf, geo, u)
        return err, flux

    # The correct statement of the pi/4 lock is CONVERGENCE vs STALL: the
    # corrected flux error is discretization error (measured 7.6% at level 5,
    # shrinking with level), while the uncorrected-data error is pinned at an
    # O(1) floor (measured ~13.7% — the (4/pi-1) inflation partially absorbed
    # by the outer-Dirichlet-constrained system). Single-level ratios are not
    # robust; the level-to-level behavior is.
    fe = {}
    for lv in (5, 6):
        _, flux_on = run(False, lv)
        _, flux_off = run(True, lv)
        fe[("on", lv)] = abs(flux_on - Fstar) / abs(Fstar)
        fe[("off", lv)] = abs(flux_off - Fstar) / abs(Fstar)
    assert fe[("on", 6)] < 0.6 * fe[("on", 5)], fe          # corrected converges
    assert fe[("off", 6)] > 0.8 * fe[("off", 5)] - 0.02, fe # uncorrected stalls
    assert fe[("off", 6)] > 3.0 * fe[("on", 6)], fe         # clear separation


def test_neumann_p1_degrades_band_restores(device):
    """The S13.3.3 acceptance core (MEASURED 2026-07-05, node-adjacent band):
    p1-everywhere caps at L2 order ~1.0 (the representability failure); a
    3-layer p2 band restores clean order 2 (2.01/2.04). (Flux asserts live in
    the area-correction lock: with a flux-carrying MMS the discrete flux
    matches F* at discretization order — the earlier 'identity' reading was
    a zero-net-flux MMS artifact.)"""
    levels = [5, 6, 7]
    e_p1, e_band = [], []
    Fstar = _true_total_flux()
    for lv in levels:
        _, e1, fl1, _, _, _ = _solve_neumann(lv, 1, device)
        _, e2, fl2, _, _, _ = _solve_neumann(lv, _band(3), device)
        e_p1.append(e1); e_band.append(e2)
    e_p1, e_band = np.array(e_p1), np.array(e_band)
    order_p1 = np.log2(e_p1[-2] / e_p1[-1])
    order_band = np.log2(e_band[-2] / e_band[-1])
    assert order_band > 1.85, (e_band, order_band)
    assert order_p1 < 1.3, (e_p1, order_p1)
    assert e_band[-1] < 0.3 * e_p1[-1], (e_band[-1], e_p1[-1])


def test_band_layer_sweep_insensitive(device):
    # restored rate insensitive to band thickness ONCE THICK ENOUGH for the
    # face-element Hessians to decouple from the minimum-rule p1 traces:
    # measured clean regime = node-adjacent layers >= 3 (band2 is marginal:
    # 1.78/1.37 — recorded in m1a findings for the draft's authors).
    levels = [5, 6, 7]
    orders = []
    for n_layers in (3, 4, 5):
        errs = []
        for lv in levels:
            _, e, _, dm, geo, sf = _solve_neumann(lv, _band(n_layers), device)
            # the hard rule: every shifted-Neumann GP inside p2 cells
            assert (np.asarray(dm.mesh.p_elem)[sf.elem] == 2).all()
            errs.append(e)
        orders.append(np.log2(errs[-2] / errs[-1]))
    assert max(orders) - min(orders) < 0.30, orders
    assert min(orders) > 1.85, orders
