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


# ============================================================================
# 3-D validation tests — all mirror the 2-D counterparts in dim=3
# These establish that the shell surrogate is dim-generic and ready for the
# P2-R1 3-D slender-object hero rung.  Meshes kept small (level 3-4, up to
# 4096 cells) so each test completes in < 1 s on CPU.
# ============================================================================

def test_3d_shell_intercepted_excludes_the_cut_band():
    """3-D analogue of test_shell_intercepted_excludes_the_cut_band.

    An x=0.5 plane cuts the 4^3 = 4096-cell uniform octree.  The intercepted
    band is exactly the two columns of 512 cells straddling x=0.5 (the two
    columns whose closed [xlo, xhi] brackets contain 0.5), giving 512 excluded
    cells (2 * 16 * 16).  The geometric band criterion is identical to the 2-D
    one but generalises cleanly: the plane is grid-aligned, so every cell in the
    two columns adjacent to x=0.5 is included and no other cell is.
    """
    from diffsim.octree import morton
    tree = build_uniform(4, dim=3)
    pl = Plane((0.5, 0.0, 0.0), (1.0, 0.0, 0.0))          # plane x = 0.5
    ret, intercepted = classify_shell_intercepted(tree, pl)
    assert len(ret) == len(tree) - intercepted.sum()
    assert intercepted.sum() > 0
    # Geometric check: exactly the cells whose [xlo, xhi] straddle 0.5
    scale = 2.0 ** -morton.lmax(3)
    a = tree.anchors() * scale
    h = tree.h()
    xlo, xhi = a[:, 0], a[:, 0] + h
    tol = 1e-12
    cut = (xlo <= 0.5 + tol) & (xhi >= 0.5 - tol)
    assert np.array_equal(intercepted, cut)
    # Both columns (x < 0.5 and x > 0.5 sides) are excluded — symmetric band.
    a_band = a[intercepted]
    h_band = h[intercepted]
    x_centers = a_band[:, 0] + h_band / 2
    assert (x_centers < 0.5).any() and (x_centers > 0.5).any()


def test_3d_shell_intercepted_narrowband_fastpath_exact():
    """3-D: narrow-band fast-path gives the same intercepted mask as the
    exhaustive dense rule (lipschitz_bound=inf).  Mirrors the 2-D version."""
    tree = build_uniform(4, dim=3)
    pl = Plane((0.5, 0.0, 0.0), (1.0, 0.0, 0.0))
    _, i1 = classify_shell_intercepted(tree, pl)
    _, i2 = classify_shell_intercepted(tree, pl, lipschitz_bound=np.inf)
    assert np.array_equal(i1, i2)


def test_3d_two_sided_split_plane():
    """3-D analogue of test_two_sided_split_plane.

    The plane at x=0.5 with normal (1,0,0) must produce:
      - non-empty Gamma~+ and Gamma~- (both sides load-bearing);
      - corr > 0 on BOTH sides (per-face oriented shell-mode n);
      - Gamma~+ sits to the LEFT (x < 0.5) with n[:,0] == +1;
      - Gamma~- sits to the RIGHT (x > 0.5) with n[:,0] == -1;
      - every Gauss point's foot projects to x = 0.5 on the plane.
    """
    tree = build_uniform(4, dim=3)
    pl = Plane((0.5, 0.0, 0.0), (1.0, 0.0, 0.0))
    ret, _ = classify_shell_intercepted(tree, pl)
    ftab = face_tables(1, 3)
    (sfp, gp), (sfm, gm) = extract_two_sided_surrogate(ret, pl, ftab)
    # both sides non-empty
    assert len(sfp.elem) > 0 and len(sfm.elem) > 0
    # corr > 0 on both sides (per-face oriented shell mode)
    assert (gp.corr > 0).all() and (gm.corr > 0).all()
    # + side: xq left of the plate, n pointing right (+x)
    assert (gp.xq[:, 0] < 0.5).all()
    assert np.allclose(gp.n[:, 0], 1.0, atol=1e-9)
    # - side: xq right of the plate, n pointing left (-x)
    assert (gm.xq[:, 0] > 0.5).all()
    assert np.allclose(gm.n[:, 0], -1.0, atol=1e-9)
    # feet of both sides project onto the plate x = 0.5
    assert np.abs((gp.xq + gp.d)[:, 0] - 0.5).max() < 1e-9
    assert np.abs((gm.xq + gm.d)[:, 0] - 0.5).max() < 1e-9


def test_3d_two_sided_faces_partition_band_faces():
    """3-D: the union of +/- surrogate faces equals extract_surrogate on the
    retained tree — the two-sided split is a clean partition (no face appears
    on both sides; together they cover all surrogate faces)."""
    tree = build_uniform(4, dim=3)
    pl = Plane((0.5, 0.0, 0.0), (1.0, 0.0, 0.0))
    ret, _ = classify_shell_intercepted(tree, pl)
    ftab = face_tables(1, 3)
    sf_all = extract_surrogate(ret)
    (sfp, _), (sfm, _) = extract_two_sided_surrogate(ret, pl, ftab)
    assert len(sfp.elem) + len(sfm.elem) == len(sf_all.elem)


def test_3d_two_sided_pressure_jump_geometry():
    """3-D: both sides of the plate project to the SAME (y, z) points on Gamma,
    confirming that the two-sided traces share common foot locations and a
    pressure-jump can be formed (ThinShell §2.1 two-sided traces)."""
    tree = build_uniform(4, dim=3)
    pl = Plane((0.5, 0.0, 0.0), (1.0, 0.0, 0.0))
    ret, _ = classify_shell_intercepted(tree, pl)
    ftab = face_tables(1, 3)
    (sfp, gp), (sfm, gm) = extract_two_sided_surrogate(ret, pl, ftab)
    # feet must both lie on x = 0.5
    assert np.abs((gp.xq + gp.d)[:, 0] - 0.5).max() < 1e-9
    assert np.abs((gm.xq + gm.d)[:, 0] - 0.5).max() < 1e-9
    # y and z coverage of feet is the same on both sides
    yp = np.unique(np.round((gp.xq + gp.d)[:, 1], 6))
    ym = np.unique(np.round((gm.xq + gm.d)[:, 1], 6))
    assert np.allclose(yp, ym), "y-coverage of feet mismatch between sides"
    zp = np.unique(np.round((gp.xq + gp.d)[:, 2], 6))
    zm = np.unique(np.round((gm.xq + gm.d)[:, 2], 6))
    assert np.allclose(zp, zm), "z-coverage of feet mismatch between sides"


def test_3d_two_sided_opposing_normals_and_corr():
    """3-D: the two sides carry OPPOSING surrogate normals (n+ = -n-) at
    matched face positions, and both have corr (|n_tilde . n|) identically 1
    for a planar shell (the exact-SDF case with n_grad parallel to n_tilde).
    This is the geometric precondition for the two-sided Nitsche sum to BLOCK
    the flow through the shell rather than cancel."""
    tree = build_uniform(3, dim=3)
    pl = Plane((0.5, 0.0, 0.0), (1.0, 0.0, 0.0))
    ret, _ = classify_shell_intercepted(tree, pl)
    ftab = face_tables(1, 3)
    (sfp, gp), (sfm, gm) = extract_two_sided_surrogate(ret, pl, ftab)
    # exact SDF: corr == 1 everywhere
    assert np.allclose(gp.corr, 1.0, atol=1e-12)
    assert np.allclose(gm.corr, 1.0, atol=1e-12)
    # n on the + side is +e_x; n on the - side is -e_x
    assert np.allclose(gp.n, np.array([1.0, 0.0, 0.0]), atol=1e-12)
    assert np.allclose(gm.n, np.array([-1.0, 0.0, 0.0]), atol=1e-12)
    # the two sides have the same number of Gauss points (symmetric by
    # construction for a grid-aligned plane)
    assert len(sfp.elem) == len(sfm.elem)


def test_3d_twosided_vector_coupling_geometry():
    """3-D: geometry-level check that sbm_vector_dirichlet_twosided would
    receive non-trivial two-sided contributions (both sides load-bearing without
    a full NS solve). We confirm: (1) distinct, non-overlapping face index sets
    for + and - sides; (2) opposing normals (precondition that the two Nitsche
    terms ADD rather than cancel, blocking the shell); (3) per-face corr > 0 on
    both sides (well-posed assembly weight); (4) the number of DOF-supporting
    Gauss points equals nqf * (n_faces_per_side) on each side."""
    tree = build_uniform(3, dim=3)
    pl = Plane((0.5, 0.0, 0.0), (1.0, 0.0, 0.0))
    ret, _ = classify_shell_intercepted(tree, pl)
    ftab = face_tables(1, 3)
    (sfp, gp), (sfm, gm) = extract_two_sided_surrogate(ret, pl, ftab)
    nqf = ftab.nqf
    # distinct face index sets (no face counted twice)
    keys_p = set(zip(sfp.elem.tolist(), sfp.face.tolist()))
    keys_m = set(zip(sfm.elem.tolist(), sfm.face.tolist()))
    assert len(keys_p & keys_m) == 0, "some faces appear on BOTH sides"
    # both sides non-trivial
    assert len(sfp.elem) > 0 and len(sfm.elem) > 0
    # well-posed Nitsche weights
    assert (gp.corr > 0).all() and (gm.corr > 0).all()
    # opposing shell normals (blocks flow, not cancels)
    dot_pp = np.einsum("gd,gd->g", gp.n, gm.n[:len(gp.n)])
    # the two sides have the same count so we can zip; verify n+ . n- < 0
    # (they point in opposite x-directions)
    assert np.all(gp.n[:, 0] * gm.n[:, 0] < 0)
    # GP count consistent with face count
    assert len(gp.xq) == len(sfp.elem) * nqf
    assert len(gm.xq) == len(sfm.elem) * nqf
