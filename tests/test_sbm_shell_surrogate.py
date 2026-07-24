"""P2-R1 — two-sided co-dim-1 THIN SHELL surrogate tests (ThinShell §2.1,
Alg 3+5). The shell is a zero-thickness rigid surface with fluid on BOTH
sides: classify_shell_intercepted EXCLUDES the shell-cut band; the two-sided
extractor splits the exposed faces into Gamma~+ / Gamma~- by the sign of the
face integral of n.n_tilde; GeometryData shell-mode orients n per-face so
corr>0 on both sides. The VOLUMETRIC SBM path must stay bit-for-bit."""
import numpy as np
import pytest

from diffsim.octree.build import build_uniform
from diffsim.mesh.faces import face_tables
from diffsim.geometry.csg import Plane, Segment, Sphere
from diffsim.sbm.surrogate import (
    classify_lambda, classify_shell_intercepted, extract_surrogate,
    extract_two_sided_surrogate, GeometryData, face_gauss_points)

pytestmark = pytest.mark.tier4


# ---- the thin-shell oracles -------------------------------------------------

def test_plane_zero_set_and_eikonal():
    pl = Plane((0.5, 0.0), (1.0, 0.0))          # vertical line x = 0.5
    x = np.array([[0.5, 0.2], [0.7, 0.9], [0.3, 0.1]])
    psi = pl.classify(x)
    assert abs(psi[0]) < 1e-14                    # on the plane
    assert psi[1] > 0 and psi[2] < 0              # signed by side
    assert pl.near_eikonal
    # |grad psi| == 1 (eikonal): distance_vector shortcut applies
    d, n, ok = pl.distance_vector(x)
    assert ok.all()
    assert np.allclose(np.linalg.norm(n, axis=1), 1.0, atol=1e-12)
    # foot lands on the plane
    assert np.abs((x + d)[:, 0] - 0.5).max() < 1e-12


def test_segment_unsigned_distance():
    seg = Segment((0.375, 0.25), (0.375, 0.75))   # vertical finite plate
    x = np.array([[0.5, 0.5], [0.25, 0.5], [0.375, 1.0]])
    psi = seg.classify(x)
    assert psi.min() >= 0.0                        # unsigned distance
    assert abs(psi[0] - 0.125) < 1e-12             # perpendicular distance
    assert abs(psi[1] - 0.125) < 1e-12
    assert abs(psi[2] - 0.25) < 1e-12              # beyond the endpoint


# ---- classify_shell_intercepted (exclusion of the shell band) --------------

def test_shell_intercepted_excludes_the_cut_band():
    tree = build_uniform(5, dim=2)
    pl = Plane((0.5, 0.0), (1.0, 0.0))
    ret, intercepted = classify_shell_intercepted(tree, pl)
    assert len(ret) == len(tree) - intercepted.sum()
    assert intercepted.sum() > 0
    # a grid-aligned plane at x=0.5 touches BOTH adjacent columns' closed cells
    # => both are excluded (the plate ends up centered in the 2-column band).
    a = tree.anchors() * 2.0 ** -31
    h = tree.h()
    xlo, xhi = a[:, 0], a[:, 0] + h
    tol = 1e-12
    cut = (xlo <= 0.5 + tol) & (xhi >= 0.5 - tol)
    assert np.array_equal(intercepted, cut)


def test_shell_intercepted_narrowband_fastpath_exact():
    tree = build_uniform(5, dim=2)
    pl = Plane((0.5, 0.0), (1.0, 0.0))
    _, i1 = classify_shell_intercepted(tree, pl)
    _, i2 = classify_shell_intercepted(tree, pl, lipschitz_bound=np.inf)
    assert np.array_equal(i1, i2)


def test_shell_intercepted_segment():
    tree = build_uniform(5, dim=2)
    seg = Segment((0.5, 0.28125), (0.5, 0.71875))   # spans rows on the grid
    ret, intercepted = classify_shell_intercepted(tree, seg)
    assert intercepted.sum() > 0
    a = tree.anchors() * 2.0 ** -31
    h = tree.h()
    # intercepted cells straddle x=0.5 AND overlap the segment's y-extent
    xlo, xhi = a[:, 0], a[:, 0] + h
    ylo, yhi = a[:, 1], a[:, 1] + h
    straddle_x = (xlo <= 0.5) & (xhi >= 0.5)
    overlap_y = (yhi >= 0.28125) & (ylo <= 0.71875)
    assert (intercepted <= (straddle_x & overlap_y)).all()   # subset (+reach)
    assert (intercepted[straddle_x & (ylo >= 0.30) & (yhi <= 0.70)]).all()


# ---- two-sided extraction (Gamma~+ / Gamma~-) ------------------------------

def test_two_sided_split_plane():
    tree = build_uniform(5, dim=2)
    pl = Plane((0.5, 0.0), (1.0, 0.0))
    ret, _ = classify_shell_intercepted(tree, pl)
    ftab = face_tables(1, 2)
    (sfp, gp), (sfm, gm) = extract_two_sided_surrogate(ret, pl, ftab)
    # both sides load-bearing
    assert len(sfp.elem) > 0 and len(sfm.elem) > 0
    # corr > 0 on BOTH sides (per-face oriented n; no cancellation)
    assert (gp.corr > 0).all() and (gm.corr > 0).all()
    # The + / - split is by I_s = int n.n_tilde: the two sides carry OPPOSITE
    # true normals (distinct outward normals across Gamma, ThinShell §2.1).
    # + side: n aligned with +grad psi (+x); - side: n aligned with -x.
    assert np.allclose(gp.n[:, 0], 1.0, atol=1e-9)
    assert np.allclose(gm.n[:, 0], -1.0, atol=1e-9)
    # the two sides sit on opposite sides of the plate line x = 0.5
    assert (gp.xq[:, 0] < 0.5).all() and (gm.xq[:, 0] > 0.5).all()
    # feet of both sides project onto the plate x = 0.5
    assert np.abs((gp.xq + gp.d)[:, 0] - 0.5).max() < 1e-9
    assert np.abs((gm.xq + gm.d)[:, 0] - 0.5).max() < 1e-9


def test_two_sided_faces_partition_the_band_faces():
    # the union of +/- surrogate faces equals extract_surrogate on the
    # retained tree (the two sides partition all exposed band faces).
    tree = build_uniform(5, dim=2)
    pl = Plane((0.5, 0.0), (1.0, 0.0))
    ret, _ = classify_shell_intercepted(tree, pl)
    ftab = face_tables(1, 2)
    sf_all = extract_surrogate(ret)
    (sfp, _), (sfm, _) = extract_two_sided_surrogate(ret, pl, ftab)
    n_all = len(sf_all.elem)
    assert len(sfp.elem) + len(sfm.elem) == n_all


def test_two_sided_pressure_jump_geometry():
    # Gamma~+ and Gamma~- face pairs project to the SAME points on Gamma
    # (the two traces of one physical interface point => a pressure jump can
    # be read off, ThinShell "two-sided traces").
    tree = build_uniform(5, dim=2)
    pl = Plane((0.5, 0.0), (1.0, 0.0))
    ret, _ = classify_shell_intercepted(tree, pl)
    ftab = face_tables(1, 2)
    (sfp, gp), (sfm, gm) = extract_two_sided_surrogate(ret, pl, ftab)
    yp = np.unique(np.round((gp.xq + gp.d)[:, 1], 6))
    ym = np.unique(np.round((gm.xq + gm.d)[:, 1], 6))
    # both sides cover the same y-locations on the plate
    assert np.allclose(yp, ym)


# ---- BIT-FOR-BIT: volumetric SBM path unchanged ----------------------------

def test_volumetric_geometry_bitforbit_default():
    # GeometryData.evaluate default (mode="volumetric") == the historical path.
    tree = build_uniform(6, dim=2)
    oracle = Sphere((0.5, 0.5), 0.3)
    ret, _ = classify_lambda(tree, oracle, 0.0)
    sf = extract_surrogate(ret)
    ftab = face_tables(1, 2)
    g_def = GeometryData.evaluate(oracle, ret, sf, ftab)
    g_vol = GeometryData.evaluate(oracle, ret, sf, ftab, mode="volumetric")
    for a in ("xq", "d", "n", "corr"):
        assert np.array_equal(getattr(g_def, a), getattr(g_vol, a))
    assert (g_def.corr > 0).all()                 # one-sided invariant holds


def test_volumetric_mode_validation():
    tree = build_uniform(4, dim=2)
    oracle = Sphere((0.5, 0.5), 0.3)
    ret, _ = classify_lambda(tree, oracle, 0.0)
    sf = extract_surrogate(ret)
    with pytest.raises(ValueError, match="mode"):
        GeometryData.evaluate(oracle, ret, sf, face_tables(1, 2), mode="bogus")
