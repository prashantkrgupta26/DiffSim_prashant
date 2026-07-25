"""3-D FiniteSheet CSG primitive tests.

Tests cover:
  1. Zero-set: points ON the patch return psi ~ 0; off-patch are correct Euclidean
  2. Normal/gradient along the patch normal
  3. Shell pipeline: classify_shell_intercepted gives a FINITE footprint (not the
     full x=0.5 plane), and extract_two_sided_surrogate returns two sides with
     opposing normals

The patch used throughout: center=(0.5, 0.5, 0.5), normal=(1,0,0), half=[0.25, 0.25]
=> covers y in [0.25, 0.75] and z in [0.25, 0.75] at x=0.5.
This is a FINITE patch -- the y/z extent does NOT cover the full [0, 1] domain,
so the shell footprint is strictly smaller than the full x=0.5 plane.
"""
import numpy as np
import pytest
import torch

from diffsim.geometry.csg import FiniteSheet, Plane
from diffsim.octree.build import build_uniform
from diffsim.octree import morton
from diffsim.mesh.faces import face_tables
from diffsim.sbm.surrogate import classify_shell_intercepted, extract_two_sided_surrogate

pytestmark = pytest.mark.tier4

# ---- shared fixture -----------------------------------------------------------

@pytest.fixture
def sheet():
    """Plate at x=0.5, covering y in [0.25, 0.75] and z in [0.25, 0.75]."""
    return FiniteSheet(
        center=[0.5, 0.5, 0.5],
        half=[0.25, 0.25],
        normal=[1.0, 0.0, 0.0],
    )


# ---- 1. Zero-set tests -------------------------------------------------------

def test_zero_set_on_patch(sheet):
    """Points ON the patch (x=0.5, y/z within half-extents) return psi ~ 0."""
    x = torch.tensor([
        [0.5, 0.5, 0.5],    # center
        [0.5, 0.3, 0.4],    # interior of patch
        [0.5, 0.25, 0.25],  # corner of patch
        [0.5, 0.75, 0.75],  # opposite corner
    ], dtype=torch.float64)
    psi = sheet.psi(x)
    assert psi.min() >= 0.0, "FiniteSheet psi must be unsigned (>= 0)"
    assert (psi < 1e-13).all(), f"Points on patch should have psi ~ 0, got {psi}"


def test_unsigned_distance_is_nonnegative(sheet):
    """psi >= 0 everywhere (unsigned distance)."""
    rng = np.random.default_rng(42)
    pts = rng.uniform(0, 1, (500, 3))
    psi = sheet.classify(pts)
    assert psi.min() >= 0.0, f"psi went negative: min={psi.min()}"


# ---- 2. Hand-computed distance cases -----------------------------------------

def test_distance_above_center(sheet):
    """Point directly above center (along normal) => psi = out-of-plane dist."""
    # x = [0.8, 0.5, 0.5]: normal=(1,0,0), center=(0.5,0.5,0.5)
    # d_perp = 0.8 - 0.5 = 0.3, point projects to (0.5, 0.5, 0.5) which is ON the patch
    # d_in = 0 (inside rectangle), psi = sqrt(0 + 0.3^2) = 0.3
    x = torch.tensor([[0.8, 0.5, 0.5]], dtype=torch.float64)
    psi = sheet.psi(x)
    assert abs(float(psi[0]) - 0.3) < 1e-12, f"Expected 0.3, got {float(psi[0])}"


def test_distance_beyond_edge(sheet):
    """Point beyond the y-edge but on the patch plane.

    x = [0.5, 0.9, 0.5]: on the patch plane (d_perp=0).
    In-plane: u = 0.9 - 0.5 = 0.4 (along t1=y), clamped to 0.25 => eu = 0.4 - 0.25 = 0.15
              v = 0.5 - 0.5 = 0.0 (along t2=z), clamped to 0.0 => ev = 0.0
    d_in = sqrt(0.15^2 + 0^2) = 0.15, d_perp = 0
    psi = sqrt(0.15^2 + 0^2) = 0.15
    """
    x = torch.tensor([[0.5, 0.9, 0.5]], dtype=torch.float64)
    psi = sheet.psi(x)
    assert abs(float(psi[0]) - 0.15) < 1e-12, f"Expected 0.15, got {float(psi[0])}"


def test_distance_beyond_corner(sheet):
    """Point beyond a corner, off the patch plane.

    x = [0.7, 0.9, 0.9]: normal=(1,0,0)
    d_perp = 0.7 - 0.5 = 0.2
    In-plane: u = 0.9-0.5=0.4, clamped to 0.25 => eu = 0.15
              v = 0.9-0.5=0.4, clamped to 0.25 => ev = 0.15
    d_in = sqrt(0.15^2 + 0.15^2) = 0.15*sqrt(2)
    psi = sqrt((0.15*sqrt(2))^2 + 0.2^2) = sqrt(0.0450 + 0.04) = sqrt(0.085)
    """
    x = torch.tensor([[0.7, 0.9, 0.9]], dtype=torch.float64)
    psi = sheet.psi(x)
    expected = np.sqrt(0.15**2 + 0.15**2 + 0.2**2)
    assert abs(float(psi[0]) - expected) < 1e-12, f"Expected {expected:.6f}, got {float(psi[0]):.6f}"


# ---- 3. Normal/gradient direction --------------------------------------------

def test_gradient_points_along_patch_normal(sheet):
    """gradient of psi at a point directly above the patch center points along normal."""
    # Use autograd to get the gradient
    x = torch.tensor([[0.8, 0.5, 0.5]], dtype=torch.float64, requires_grad=False)
    xv = x.clone().requires_grad_(True)
    psi = sheet.psi(xv)
    psi.sum().backward()
    grad = xv.grad[0]  # [3]
    # should point in +x direction (same as normal [1,0,0]) since x is to the right of center
    expected = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    assert torch.allclose(grad, expected, atol=1e-10), f"Expected gradient {expected}, got {grad}"


def test_distance_vector_foot_on_patch(sheet):
    """distance_vector returns foot on the patch and grad_psi along patch normal.

    n_grad convention: mirrors Plane.distance_vector which returns the CONSTANT
    plane normal (not flipped by side). For FiniteSheet, n_grad = normal always
    so that extract_two_sided_surrogate can split by sign(n_tilde . n_grad):
    cells on the normal's positive side get Is > 0 (Gamma~+) and cells on the
    negative side get Is < 0 (Gamma~-).
    """
    pts = np.array([
        [0.8, 0.5, 0.5],   # directly above center (d_perp = +0.3)
        [0.3, 0.5, 0.5],   # directly below center (d_perp = -0.2)
    ])
    d, n, ok = sheet.distance_vector(pts)
    assert ok.all()
    # feet should land on the patch center (x=0.5, y=0.5, z=0.5)
    feet = pts + d
    assert np.abs(feet[:, 0] - 0.5).max() < 1e-9, "Feet should be at x=0.5"
    assert np.abs(feet[:, 1] - 0.5).max() < 1e-9
    assert np.abs(feet[:, 2] - 0.5).max() < 1e-9
    # displacements should be correct
    assert abs(d[0, 0] - (-0.3)) < 1e-12, f"d[0] should be [-0.3,0,0], got {d[0]}"
    assert abs(d[1, 0] - 0.2) < 1e-12, f"d[1] should be [+0.2,0,0], got {d[1]}"
    # n_grad = patch normal (constant, like Plane.distance_vector convention):
    # mirrors Plane which returns normal=[1,0,0] regardless of which side of the
    # plane the query point is on. This enables the two-sided split via Is = n_tilde.n.
    assert np.allclose(np.linalg.norm(n, axis=1), 1.0, atol=1e-10)
    assert n[0, 0] > 0.9, f"n_grad should be +x (patch normal), got {n[0]}"
    assert n[1, 0] > 0.9, f"n_grad should be +x (constant patch normal), got {n[1]}"


# ---- 4. Shell pipeline: finite footprint ------------------------------------

def test_finite_footprint_vs_unbounded_plane():
    """KEY test: FiniteSheet excludes only the finite patch footprint.

    Setup: level-4 3-D uniform tree (16^3 = 4096 cells), cell size h=1/16.
    Plate: x=0.5, half=[0.25,0.25] => covers y in [0.25,0.75], z in [0.25,0.75].

    Expected:
    - The unbounded Plane at x=0.5 would intercept ALL 2*16*16=512 cells in
      the two x-columns straddling x=0.5 (y,z range over the full [0,1]).
    - The FiniteSheet should only intercept cells where the PATCH actually
      passes through -- those in the two x-columns straddling x=0.5 AND
      whose [ylo,yhi] x [zlo,zhi] overlaps the patch's [0.25,0.75]^2.
    - The FiniteSheet intercepted count must be strictly LESS THAN the
      Plane intercepted count (the finite plate vs unbounded plane difference).
    - No cell outside the patch footprint (y or z entirely outside [0.25,0.75])
      should be intercepted (up to one cell of reach due to the Lipschitz
      narrowband margin).
    """
    tree = build_uniform(4, dim=3)
    scale = 2.0 ** -morton.lmax(3)
    a = tree.anchors() * scale
    h = tree.h()

    sheet = FiniteSheet(
        center=[0.5, 0.5, 0.5],
        half=[0.25, 0.25],
        normal=[1.0, 0.0, 0.0],
    )
    plane = Plane([0.5, 0.0, 0.0], [1.0, 0.0, 0.0])

    _, intercepted_sheet = classify_shell_intercepted(tree, sheet, lipschitz_bound=np.inf)
    _, intercepted_plane = classify_shell_intercepted(tree, plane, lipschitz_bound=np.inf)

    # 1. The plate has a strictly smaller footprint than the full plane
    assert intercepted_sheet.sum() < intercepted_plane.sum(), (
        f"FiniteSheet intercepted {intercepted_sheet.sum()} cells, "
        f"Plane intercepted {intercepted_plane.sum()} -- sheet should be FEWER"
    )

    # 2. Every intercepted cell straddles x=0.5 (the patch x-position)
    xlo, xhi = a[:, 0], a[:, 0] + h
    straddle_x = (xlo <= 0.5 + 1e-12) & (xhi >= 0.5 - 1e-12)
    assert intercepted_sheet[~straddle_x].sum() == 0, (
        "FiniteSheet intercepted cells that don't straddle x=0.5"
    )

    # 3. Cells well OUTSIDE the patch extent (y or z center far from [0.25,0.75])
    # should not be intercepted. "Well outside" = cell center > 0.75 + h or < 0.25 - h
    y_center = a[:, 1] + h / 2
    z_center = a[:, 2] + h / 2
    clearly_outside_y = (y_center > 0.75 + h) | (y_center < 0.25 - h)
    clearly_outside_z = (z_center > 0.75 + h) | (z_center < 0.25 - h)
    clearly_outside = clearly_outside_y | clearly_outside_z
    assert intercepted_sheet[straddle_x & clearly_outside].sum() == 0, (
        "FiniteSheet intercepted cells clearly outside the patch y/z extent"
    )

    # 4. Cells at the patch center (y,z in [0.3, 0.7]) should be intercepted
    y_in_patch = (y_center >= 0.3) & (y_center <= 0.7)
    z_in_patch = (z_center >= 0.3) & (z_center <= 0.7)
    core_cells = straddle_x & y_in_patch & z_in_patch
    assert intercepted_sheet[core_cells].all(), (
        "FiniteSheet should intercept ALL cells in the patch core region"
    )


# ---- 5. Two-sided surrogate over the finite patch ---------------------------

def test_two_sided_split_finite_sheet():
    """extract_two_sided_surrogate returns two non-empty sides for the finite patch.

    Uses level-4 tree and the same plate as above (center=(0.5,0.5,0.5),
    half=[0.25,0.25], normal=(1,0,0)).

    IMPORTANT: a finite patch creates BOTH x-facing main faces (the two sides
    of the plate) AND y/z-facing EDGE faces at the boundary of the finite patch
    footprint. With n_grad = constant normal = [1,0,0]:
    - x-left faces: n_tilde=[1,0,0], Is > 0 => Gamma~+ (corr = 1)
    - x-right faces: n_tilde=[-1,0,0], Is < 0 => Gamma~- (corr = 1)
    - y/z-edge faces: n_tilde perpendicular to normal, Is = 0 => Gamma~+ (corr = 0)
    The MINUS side contains ONLY x-right faces (corr=1 everywhere), confirming
    the right side of the plate. The PLUS side contains x-left faces (corr=1)
    and edge faces (corr=0). We verify the main plate-blocking behavior via the
    MINUS side (pure, corr>0) and check the PLUS side has non-zero x-left faces.
    """
    tree = build_uniform(4, dim=3)
    sheet = FiniteSheet(
        center=[0.5, 0.5, 0.5],
        half=[0.25, 0.25],
        normal=[1.0, 0.0, 0.0],
    )
    ret, _ = classify_shell_intercepted(tree, sheet)
    ftab = face_tables(1, 3)
    (sfp, gp), (sfm, gm) = extract_two_sided_surrogate(ret, sheet, ftab)

    # both sides load-bearing
    assert len(sfp.elem) > 0, "FiniteSheet Gamma~+ is empty"
    assert len(sfm.elem) > 0, "FiniteSheet Gamma~- is empty"

    # MINUS side: all faces are x-right-facing (pure, no edge contamination)
    # => corr = |n_tilde . normal| = |[-1,0,0] . [1,0,0]| = 1 everywhere
    assert (gm.corr > 0).all(), "corr <= 0 on - side (x-right faces should all have corr=1)"

    # PLUS side: mix of x-left-facing (corr=1) and y/z-edge-facing (corr=0)
    # We only require no NEGATIVE corr (zero is OK for edge faces)
    assert (gp.corr >= 0).all(), "Negative corr found on + side"
    # At least some + faces have corr > 0 (the x-left-facing ones)
    assert (gp.corr > 0).any(), "All + side corr are 0 (no x-left faces found)"

    # MINUS side: xq to the RIGHT (x > 0.5), n pointing -x (away from plate)
    assert (gm.xq[:, 0] > 0.5 - 1e-9).all(), "Gamma~- xq should be right of the plate"
    # In shell mode: n = sign(n_tilde . n_grad) * n_grad = sign(-1)*[1,0,0] = [-1,0,0]
    assert np.allclose(gm.n[:, 0], -1.0, atol=1e-9), "Gamma~- should have n pointing -x"

    # PLUS side x-left faces: xq to the left (x < 0.5)
    xp_mask = sfp.face == 1  # face_id=1 is the +x face (pointing right, n_tilde=[1,0,0])
    if xp_mask.any():
        nqf = ftab.nqf
        # expand mask to GP level
        gp_xp_mask = np.repeat(xp_mask, nqf)
        assert (gp.xq[gp_xp_mask, 0] < 0.5 + 1e-9).all(), \
            "x-left Gamma~+ xq should be left of the plate"
        assert np.allclose(gp.n[gp_xp_mask, 0], 1.0, atol=1e-9), \
            "x-left Gamma~+ should have n pointing +x"

    # feet of minus-side project onto the patch (x=0.5)
    assert np.abs((gm.xq + gm.d)[:, 0] - 0.5).max() < 1e-8


def test_two_sided_faces_partition_band_faces_finite_sheet():
    """The union of +/- surrogate faces equals extract_surrogate on the retained tree."""
    from diffsim.sbm.surrogate import extract_surrogate
    tree = build_uniform(4, dim=3)
    sheet = FiniteSheet(
        center=[0.5, 0.5, 0.5],
        half=[0.25, 0.25],
        normal=[1.0, 0.0, 0.0],
    )
    ret, _ = classify_shell_intercepted(tree, sheet)
    ftab = face_tables(1, 3)
    sf_all = extract_surrogate(ret)
    (sfp, _), (sfm, _) = extract_two_sided_surrogate(ret, sheet, ftab)
    assert len(sfp.elem) + len(sfm.elem) == len(sf_all.elem), (
        f"+/- faces ({len(sfp.elem)}+{len(sfm.elem)}) don't sum to "
        f"all exposed faces ({len(sf_all.elem)})"
    )
