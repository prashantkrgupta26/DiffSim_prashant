"""Tier-4 SBM surrogate tests (M1a Task 5): lambda-criterion classification,
surrogate-face extraction (whole-face invariant), per-epoch geometry cache,
and the pi/4 area-correction geometric lock (spec S4.2)."""
import numpy as np
import pytest
from diffsim.octree import morton
from diffsim.octree.build import Octree, build_uniform, refine_elements
from diffsim.geometry.csg import Sphere
from diffsim.mesh.faces import face_tables
from diffsim.sbm.surrogate import (classify_lambda, extract_surrogate,
                                   face_gauss_points, GeometryData)

pytestmark = pytest.mark.tier4


def _circle_case(level, lam):
    tree = build_uniform(level, dim=2)
    oracle = Sphere((0.5, 0.5), 0.3)              # interior problem: domain = disk
    ret, frac = classify_lambda(tree, oracle, lam)
    sf = extract_surrogate(ret)
    return oracle, ret, sf


def test_lambda_retention_ordering():
    tree = build_uniform(5, dim=2)
    oracle = Sphere((0.5, 0.5), 0.3)
    n_all = [len(classify_lambda(tree, oracle, lam)[0]) for lam in (0.0, 0.5, 1.0)]
    assert 0 < n_all[0] <= n_all[1] <= n_all[2]   # retention GROWS with lambda (production RatioGPSBM)
    ret1, frac1 = classify_lambda(tree, oracle, 0.0)
    assert (frac1 == 1.0).all()                    # lam=0: fully-interior only


def test_surrogate_faces_enclose_domain():
    oracle, ret, sf = _circle_case(5, 0.5)
    assert len(sf.elem) > 0
    ftab = face_tables(1, 2)
    xq = face_gauss_points(ret, sf, ftab)
    # face GPs sit near the true boundary: |psi| < 2 h_max
    assert np.abs(oracle.classify(xq)).max() < 2 * ret.h().max()


def test_geometry_data_circle():
    oracle, ret, sf = _circle_case(6, 0.0)
    ftab = face_tables(1, 2)
    geo = GeometryData.evaluate(oracle, ret, sf, ftab)
    # d points from surrogate GP to the circle: x + d lies on it
    r = np.linalg.norm(geo.xq + geo.d - 0.5, axis=1)
    assert np.abs(r - 0.3).max() < 1e-11
    assert (geo.corr > 0.0).all()                  # outward alignment
    assert np.allclose(np.linalg.norm(geo.n, axis=1), 1.0, atol=1e-12)
    # domain="inside": n points OUT of the disk (increasing psi)
    outward = (geo.xq - 0.5) / np.linalg.norm(geo.xq - 0.5, axis=1, keepdims=True)
    assert (np.einsum("id,id->i", geo.n, outward) > 0.9).all()


@pytest.mark.parametrize("level", [6, 7])
def test_pi_over_4_area_correction_lock(level):
    """LOCKED (spec S4.2 step 5): staircase perimeter of a circle -> 8r; the
    (n_bar . n) corrected perimeter -> 2 pi r. Ratio -> pi/4. Without the
    correction, any surrogate-boundary flux integral inherits the 4/pi error."""
    oracle, ret, sf = _circle_case(level, 0.0)
    ftab = face_tables(1, 2)
    geo = GeometryData.evaluate(oracle, ret, sf, ftab)
    h = ret.h()[sf.elem]
    dS = np.repeat(h, ftab.nqf) * np.tile(ftab.w, len(sf.elem)) / 2.0  # w*(h/2)^(dim-1)
    P_stair = dS.sum()
    P_corr = (geo.corr * dS).sum()
    P_true = 2 * np.pi * 0.3
    assert abs(P_corr - P_true) < 0.05 * P_true
    assert abs(P_corr / P_stair - np.pi / 4) < 0.03    # the pi/4 signature


def test_partially_exposed_face_raises():
    # Handcraft: level-1 cell A with a refined right neighbor missing ONE of
    # the two fine cells across A's +x face -> partially exposed -> error.
    t = build_uniform(1, dim=2)
    a = t.anchors()
    half = 1 << (morton.lmax(2) - 1)
    target = int(np.where((a[:, 0] == half) & (a[:, 1] == 0))[0][0])
    mask = np.zeros(len(t), bool)
    mask[target] = True
    t2 = refine_elements(t, mask)
    a2, lev2 = t2.anchors(), t2.levels
    drop = np.where((lev2 == 2) & (a2[:, 0] == half) & (a2[:, 1] == 0))[0]
    keep = np.ones(len(t2), bool)
    keep[drop[0]] = False
    t3 = Octree(t2.keys[keep], t2.levels[keep], dim=2, periodic=t2.periodic)
    with pytest.raises(ValueError, match="partially exposed"):
        extract_surrogate(t3)


def test_fully_exposed_fine_faces_ok():
    # The reverse configuration is legal: retained FINE elements against a
    # dropped coarse region give whole (fine) surrogate faces.
    t = build_uniform(1, dim=2)
    a = t.anchors()
    half = 1 << (morton.lmax(2) - 1)
    target = int(np.where((a[:, 0] == half) & (a[:, 1] == 0))[0][0])
    mask = np.zeros(len(t), bool)
    mask[target] = True
    t2 = refine_elements(t, mask)
    # drop the whole level-1 cell to the LEFT of the refined one
    a2, lev2 = t2.anchors(), t2.levels
    drop = np.where((lev2 == 1) & (a2[:, 0] == 0) & (a2[:, 1] == 0))[0]
    keep = np.ones(len(t2), bool)
    keep[drop[0]] = False
    t3 = Octree(t2.keys[keep], t2.levels[keep], dim=2, periodic=t2.periodic)
    sf = extract_surrogate(t3)
    assert len(sf.elem) > 0
    # the two fine cells at x = half emit their -x faces
    fine_minus_x = (t3.levels[sf.elem] == 2) & (sf.face == 0)
    assert fine_minus_x.sum() == 2


def test_exterior_configuration():
    # domain = square minus disk (the M1b configuration), via the domain flag
    tree = build_uniform(5, dim=2)
    oracle = Sphere((0.5, 0.5), 0.25)
    ret, _ = classify_lambda(tree, oracle, 0.0, domain="outside")
    sf = extract_surrogate(ret)
    ftab = face_tables(1, 2)
    geo = GeometryData.evaluate(oracle, ret, sf, ftab, domain="outside")
    # normals point INTO the disk (outward from the domain)
    to_center = (0.5 - geo.xq) / np.linalg.norm(0.5 - geo.xq, axis=1, keepdims=True)
    assert (np.einsum("id,id->i", geo.n, to_center) > 0.9).all()
    assert (geo.corr > 0.0).all()


def test_domain_flag_validation():
    tree = build_uniform(3, dim=2)
    with pytest.raises(ValueError, match="domain"):
        classify_lambda(tree, Sphere((0.5, 0.5), 0.3), 1.0, domain="both")


def test_lambda_volume_fraction_accuracy():
    # frac is a weighted quadrature estimate of the volume fraction: summing
    # frac * h^2 over ALL elements (lam -> 0+) reproduces the disk area.
    tree = build_uniform(5, dim=2)
    oracle = Sphere((0.5, 0.5), 0.3)
    ret, frac = classify_lambda(tree, oracle, lam=1.0, n1=5)
    area = (frac * ret.h() ** 2).sum()
    exact = np.pi * 0.3 ** 2
    assert abs(area - exact) < 5e-3 * exact
    # denser rule is at least as accurate
    _, frac9 = classify_lambda(tree, oracle, lam=1.0, n1=9)
    area9 = (frac9 * ret.h() ** 2).sum()
    assert abs(area9 - exact) <= abs(area - exact) + 1e-12


def test_lambda_narrowband_fastpath_exact():
    # The Lipschitz center-skip must not change the outcome: identical
    # retained sets and fracs vs dense-everywhere (lipschitz_bound=inf).
    tree = build_uniform(5, dim=2)
    oracle = Sphere((0.5, 0.5), 0.3)
    for lam in (0.5, 0.0):
        r1, f1 = classify_lambda(tree, oracle, lam)
        r2, f2 = classify_lambda(tree, oracle, lam, lipschitz_bound=np.inf)
        assert np.array_equal(r1.keys, r2.keys)
        assert np.allclose(f1, f2, atol=1e-15)


def test_lambda_and_surrogate_4d():
    # k=4 (space-time dimension): the whole pipeline is dim-generic.
    # 4-ball volume: pi^2 r^4 / 2.
    oracle = Sphere((0.5,) * 4, 0.35)
    tree = build_uniform(3, dim=4)
    ret, frac = classify_lambda(tree, oracle, lam=1.0, n1=5)
    vol = (frac * ret.h() ** 4).sum()
    exact = np.pi ** 2 * 0.35 ** 4 / 2.0
    assert abs(vol - exact) < 0.02 * exact
    # narrowband fast path exact in 4D too
    r2, f2 = classify_lambda(tree, oracle, 0.0, lipschitz_bound=np.inf)
    r1, f1 = classify_lambda(tree, oracle, 0.0)
    assert np.array_equal(r1.keys, r2.keys)
    # surrogate extraction + geometry cache on the lam=1 retained set
    sf = extract_surrogate(r1)
    assert len(sf.elem) > 0
    ftab = face_tables(1, 4)
    geo = GeometryData.evaluate(oracle, r1, sf, ftab)
    assert (geo.corr > 0.0).all()
    # Closed-staircase closure identity (exact at ANY level): the vector sum
    # of surrogate face areas vanishes, sum_f n_tilde_f h^3 = 0.
    from diffsim.octree.lookup import face_offsets
    h = r1.h()[sf.elem]
    ntilde = face_offsets(4).astype(np.float64)[sf.face]
    closure = (ntilde * (h ** 3)[:, None]).sum(axis=0)
    assert np.abs(closure).max() < 1e-12
    # Corrected area: at coarse h the lam=1 surrogate is an INSCRIBED
    # staircase (its faces sit up to ~h + h*sqrt(d)/2 inside Gamma), so
    # int corr dS approximates the area of that smaller 3-sphere and
    # converges to |Gamma| only as h -> 0 (the 2D pi/4 lock is the
    # quantitative version at fine h). Assert the honest O(h) bracket.
    dS = np.repeat(h ** 3, ftab.nqf) * np.tile(ftab.w, len(sf.elem)) / 8.0
    A_corr = (geo.corr * dS).sum()
    A_true = 2.0 * np.pi ** 2 * 0.35 ** 3
    h0 = r1.h().max()
    A_inner = 2.0 * np.pi ** 2 * (0.35 - 1.5 * h0) ** 3
    assert A_inner < A_corr < 1.05 * A_true, (A_inner, A_corr, A_true)
