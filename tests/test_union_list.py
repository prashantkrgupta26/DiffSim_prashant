"""Tests for UnionList N-ary SDF union oracle (spec S4.1).

TDD: tests written before implementation; run with .venv/bin/pytest.

Test groups:
  Group 1 — two-sphere analytic: psi/classify/distance_vector vs analytic
             min-distance at points near/inside/far
  Group 2 — 3-body classify + distance_vector vs brute-force per-member min,
             including points equidistant between bodies
  Group 3 — 2-body TriMeshOracle union via icosphere (BVH path)

Tie-breaking contract: when |psi_i| == |psi_j| (equidistant), UnionList picks
the member with the lowest index (argmin semantics on the psi array). Tests
document the tolerance used (1e-12 for exact analytic; machine eps for
numerics).
"""
import numpy as np
import pytest
import torch

from diffsim.geometry.csg import Sphere, Box
from diffsim.geometry.union_list import UnionList

pytestmark = pytest.mark.geometry


# ---------------------------------------------------------------------------
# Group 1: two-sphere union vs analytic min-distance
# ---------------------------------------------------------------------------

class TestTwoSphereAnalytic:
    """UnionList of two Sphere oracles; reference = analytic SDF min."""

    def setup_method(self):
        # Sphere A: center (0,0,0) r=1 in 3D
        # Sphere B: center (3,0,0) r=0.5 in 3D
        self.sA = Sphere([0.0, 0.0, 0.0], 1.0)
        self.sB = Sphere([3.0, 0.0, 0.0], 0.5)
        self.union = UnionList([self.sA, self.sB])

    def _analytic_psi(self, pts):
        """Exact min(psi_A, psi_B)."""
        pA = np.linalg.norm(pts, axis=1) - 1.0
        pB = np.linalg.norm(pts - [3.0, 0.0, 0.0], axis=1) - 0.5
        return np.minimum(pA, pB)

    def test_psi_near(self):
        """Points near sphere A surface."""
        pts = np.array([
            [1.1, 0.0, 0.0],
            [0.0, 0.9, 0.0],
            [-1.05, 0.0, 0.0],
        ])
        x = torch.tensor(pts)
        got = self.union.psi(x).detach().numpy()
        ref = self._analytic_psi(pts)
        np.testing.assert_allclose(got, ref, atol=1e-12)

    def test_psi_inside(self):
        """Points inside each sphere (negative psi)."""
        pts = np.array([
            [0.5, 0.0, 0.0],   # inside A
            [3.0, 0.0, 0.0],   # inside B (center)
            [0.0, 0.0, 0.3],   # inside A
        ])
        x = torch.tensor(pts)
        got = self.union.psi(x).detach().numpy()
        ref = self._analytic_psi(pts)
        np.testing.assert_allclose(got, ref, atol=1e-12)

    def test_psi_far(self):
        """Points far from both spheres."""
        pts = np.array([
            [10.0, 0.0, 0.0],
            [0.0, 10.0, 0.0],
            [-5.0, -5.0, 5.0],
        ])
        x = torch.tensor(pts)
        got = self.union.psi(x).detach().numpy()
        ref = self._analytic_psi(pts)
        np.testing.assert_allclose(got, ref, atol=1e-12)

    def test_classify(self):
        """classify (numpy bridge) matches analytic psi."""
        pts = np.array([
            [0.5, 0.0, 0.0],   # inside A
            [3.0, 0.0, 0.0],   # inside B
            [1.5, 0.0, 0.0],   # between, positive
            [10.0, 0.0, 0.0],  # far positive
        ])
        got = self.union.classify(pts)
        ref = self._analytic_psi(pts)
        np.testing.assert_allclose(got, ref, atol=1e-12)

    def test_distance_vector_routes_to_closest_member(self):
        """distance_vector routes each point to the argmin-|psi| member
        and returns that member's (d, n_grad, ok).

        For Sphere oracles, |psi| = |dist_to_surface|.
        We compare d magnitude vs analytic closest-surface distance.
        """
        pts = np.array([
            [1.5, 0.0, 0.0],   # closer to sA (psi_A=0.5, psi_B=1.5) -> sA
            [3.3, 0.0, 0.0],   # closer to sB (psi_A=2.3, psi_B=-0.2->B inside)
            [2.0, 0.0, 0.0],   # psi_A=1.0, psi_B=0.5 -> sB closer
        ])
        d, n, ok = self.union.distance_vector(pts)
        assert ok.all()

        # check point [0]: routes to sA
        # sA closest on surface at (1,0,0), displacement = (1,0,0)-(1.5,0,0)=(-0.5,0,0)
        np.testing.assert_allclose(d[0], [-0.5, 0.0, 0.0], atol=1e-12)
        # n_grad should point outward from sA = (1,0,0) dir
        np.testing.assert_allclose(n[0], [1.0, 0.0, 0.0], atol=1e-6)

        # check point [1]: psi_A=1.0, psi_B=-0.2 -> |psi_B|=0.2 < |psi_A|=1.0 -> sB
        # inside B: closest point on surface in direction from center=(3,0,0) to (3.3,0,0)
        # foot = (3,0,0) + 0.5*(3.3-3,0,0)/0.3 = (3.5,0,0)
        # d = (3.5,0,0) - (3.3,0,0) = (0.2,0,0)
        np.testing.assert_allclose(d[1], [0.2, 0.0, 0.0], atol=1e-6)

        # check point [2]: psi_A=1.0, psi_B=0.5 -> routes to sB
        # sB closest: center=(3,0,0), r=0.5, pt=(2,0,0) -> foot=(2.5,0,0)
        # d = (2.5,0,0) - (2,0,0) = (0.5,0,0)
        np.testing.assert_allclose(d[2], [0.5, 0.0, 0.0], atol=1e-6)

    def test_dim(self):
        assert self.union.dim == 3

    def test_near_eikonal_false_for_two_spheres(self):
        """Exact min of two eikonal SDFs is NOT globally eikonal.

        The csg.py module docstring (lines 14-16) states: 'exact min/max are
        NOT globally eikonal => near_eikonal = False'.  The medial axis
        (equidistant surface between the two spheres) is a ridge where
        |grad psi| is discontinuous and the eikonal shortcut d = psi * n_grad
        gives wrong foot points.  UnionList follows this convention regardless
        of member eikonal status.
        """
        assert self.union.near_eikonal is False

    def test_params_concatenation(self):
        """params is the list of all member params concatenated."""
        p = self.union.params
        pA = self.sA.params   # [center, radius]
        pB = self.sB.params
        assert len(p) == len(pA) + len(pB)
        # each tensor is the same object
        assert p[0] is pA[0]
        assert p[1] is pA[1]
        assert p[2] is pB[0]
        assert p[3] is pB[1]

    def test_velocity_returns_zeros(self):
        """Static geometry: velocity must return zeros (base SDFOracle.velocity)."""
        pts = np.array([[0.5, 0.0, 0.0], [3.0, 0.0, 0.0]])
        v = self.union.velocity(pts, t=0.0)
        np.testing.assert_array_equal(v, np.zeros_like(pts))

    def test_dim_mismatch_raises(self):
        """Mixing oracles of different dims must raise ValueError."""
        s2d = Sphere([0.0, 0.0], 1.0)
        s3d = Sphere([0.0, 0.0, 0.0], 1.0)
        with pytest.raises(ValueError, match="dim"):
            UnionList([s2d, s3d])

    def test_empty_raises(self):
        """Empty list must raise ValueError."""
        with pytest.raises(ValueError):
            UnionList([])


# ---------------------------------------------------------------------------
# Group 2: 3-body classify + distance_vector vs brute-force
# ---------------------------------------------------------------------------

class TestThreeBodyBruteForce:
    """Three Box oracles — verify classify and distance_vector against
    brute-force per-member evaluation.

    Tie-breaking: equidistant points (|psi| equal across two members) use
    lowest-index member — documented tolerance 1e-10 for detecting
    'equidistant' in tests.
    """

    def setup_method(self):
        # Three boxes separated along x-axis
        self.bA = Box([0.0, 0.0, 0.0], [0.4, 0.4, 0.4])   # center 0, half 0.4
        self.bB = Box([2.0, 0.0, 0.0], [0.4, 0.4, 0.4])   # center 2
        self.bC = Box([4.0, 0.0, 0.0], [0.4, 0.4, 0.4])   # center 4
        self.members = [self.bA, self.bB, self.bC]
        self.union = UnionList(self.members)

    def _brute_classify(self, pts):
        """Reference: elementwise min over member psi values."""
        psis = np.stack([m.classify(pts) for m in self.members], axis=1)
        return psis.min(axis=1)

    def _brute_distance_vector(self, pts):
        """Reference: for each point, find argmin |psi|, call that member."""
        psis = np.stack([m.classify(pts) for m in self.members], axis=1)
        idx = np.argmin(np.abs(psis), axis=1)
        ds, ns, oks = [], [], []
        for i, pt in enumerate(pts):
            d, n, ok = self.members[idx[i]].distance_vector(pt[None])
            ds.append(d[0])
            ns.append(n[0])
            oks.append(ok[0])
        return np.stack(ds), np.stack(ns), np.array(oks)

    def test_classify_matches_brute(self):
        """classify matches elementwise min over 3 members."""
        rng = np.random.default_rng(42)
        pts = rng.uniform(-1.0, 5.0, (50, 3))
        got = self.union.classify(pts)
        ref = self._brute_classify(pts)
        np.testing.assert_allclose(got, ref, atol=1e-12)

    def test_distance_vector_matches_brute(self):
        """distance_vector routes to argmin-|psi| member, returns that d/n/ok.

        Note: points at the exact center of a box have a zero-gradient
        singularity in the Box SDF (degenerate Newton).  All test points here
        are kept away from the center (non-zero gradient guaranteed).
        """
        # Points chosen to avoid box-center degeneracy (zero grad at center).
        pts = np.array([
            [0.2, 0.0, 0.0],   # inside A (off-center, non-degenerate)
            [2.2, 0.0, 0.0],   # inside B (off-center)
            [4.2, 0.0, 0.0],   # inside C (off-center)
            [0.9, 0.0, 0.0],   # near surface of A (psi_A ~= 0.5, closest to A)
            [3.5, 0.0, 0.0],   # between B and C, closer to C
            [1.0, 0.0, 0.0],   # psi_A = 0.6, psi_B = 0.6 -> equidistant -> picks A (idx=0)
        ], dtype=np.float64)
        d_got, n_got, ok_got = self.union.distance_vector(pts)
        d_ref, n_ref, ok_ref = self._brute_distance_vector(pts)
        assert ok_got.all()
        assert ok_ref.all()
        np.testing.assert_allclose(d_got, d_ref, atol=1e-10)
        np.testing.assert_allclose(n_got, n_ref, atol=1e-10)

    def test_equidistant_picks_lowest_index(self):
        """At a point exactly equidistant (by |psi|) between A and B,
        UnionList picks the lowest-index member (A, index 0).

        Point at x=1.0 is exactly on the midpoint between box A surface (x=0.4)
        and box B surface (x=1.6): psi_A = 0.6, psi_B = 0.6 -> pick A.
        """
        # Verify the distances are actually equal to within numerical precision
        pt = np.array([[1.0, 0.0, 0.0]])
        psi_A = float(self.bA.classify(pt)[0])
        psi_B = float(self.bB.classify(pt)[0])
        # Both are exactly 0.6 for axis-aligned boxes
        assert abs(abs(psi_A) - abs(psi_B)) < 1e-12, (
            f"Expected equidistant: |psi_A|={abs(psi_A)}, |psi_B|={abs(psi_B)}")

        d_got, n_got, ok_got = self.union.distance_vector(pt)
        # Reference from member A (index 0, wins tie)
        d_ref, n_ref, ok_ref = self.bA.distance_vector(pt)
        np.testing.assert_allclose(d_got, d_ref, atol=1e-12)
        np.testing.assert_allclose(n_got, n_ref, atol=1e-12)

    def test_near_eikonal_false_for_any_union(self):
        """Exact min is never globally eikonal, regardless of member types.

        UnionList.near_eikonal is always False: the exact min of N SDFs has
        ridge sets at the medial axis where the gradient is discontinuous.
        This holds even when all members are individually eikonal (Sphere, Box).
        Consistent with csg.py: '_Blend' exact-min Union also sets
        near_eikonal = False unconditionally.
        """
        assert self.union.near_eikonal is False

    def test_near_eikonal_false_with_non_eikonal_member(self):
        """UnionList is False even with a mix of eikonal and non-eikonal members."""
        from diffsim.geometry.csg import Segment
        s = Segment([0.0, 0.0], [1.0, 0.0])
        bA2d = Box([0.0, 0.0], [0.4, 0.4])
        u = UnionList([bA2d, s])
        assert u.near_eikonal is False

    def test_params_all_members(self):
        """params concatenates all three members."""
        p = self.union.params
        expected_len = (len(self.bA.params) + len(self.bB.params)
                        + len(self.bC.params))
        assert len(p) == expected_len


# ---------------------------------------------------------------------------
# Group 3: TriMeshOracle union (BVH path) — two icospheres
# ---------------------------------------------------------------------------

class TestTriMeshUnion:
    """UnionList of two TriMeshOracle icospheres — exercises the BVH path.

    Both icospheres are on the cpu device (wp.Mesh works on cpu — verified in
    conftest: wp.init() succeeds; TriMeshOracle uses default_device() which is
    cpu when no GPU present in this env). Test is gated on Warp availability.
    """

    @pytest.fixture(autouse=True)
    def _require_warp(self):
        try:
            import warp as wp  # noqa: F401
        except ImportError:
            pytest.skip("Warp not installed")

    def setup_method(self):
        from diffsim.geometry.trimesh import icosphere
        # Use cpu device explicitly (conftest device fixture uses default_device)
        try:
            import warp as wp
            wp.init()
            from diffsim.device import default_device
            dev = default_device()
        except Exception:
            self._skip = True
            return
        self._skip = False

        # Two icospheres well-separated
        self.sA = icosphere(2, center=[0.0, 0.0, 0.0], radius=0.5, device=dev)
        self.sB = icosphere(2, center=[3.0, 0.0, 0.0], radius=0.5, device=dev)
        self.union = UnionList([self.sA, self.sB])

    def _require(self):
        if getattr(self, '_skip', False):
            pytest.skip("Warp init failed")

    def test_classify_inside_outside(self):
        """Points inside each sphere have psi < 0; far points have psi > 0."""
        self._require()
        pts = np.array([
            [0.0, 0.0, 0.0],   # inside A
            [3.0, 0.0, 0.0],   # inside B
            [10.0, 0.0, 0.0],  # far positive
        ], dtype=np.float64)
        psi = self.union.classify(pts)
        assert psi[0] < 0, f"Expected psi<0 inside sA, got {psi[0]}"
        assert psi[1] < 0, f"Expected psi<0 inside sB, got {psi[1]}"
        assert psi[2] > 0, f"Expected psi>0 far, got {psi[2]}"

    def test_psi_matches_brute_force(self):
        """psi of union matches elementwise min of member psi values."""
        self._require()
        pts = np.array([
            [0.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [1.5, 0.0, 0.0],
            [0.6, 0.0, 0.0],
            [2.4, 0.0, 0.0],
        ], dtype=np.float64)
        x = torch.tensor(pts)
        got = self.union.psi(x).detach().numpy()
        pA = self.sA.classify(pts)
        pB = self.sB.classify(pts)
        ref = np.minimum(pA, pB)
        # TriMeshOracle uses fp32 BVH for candidates: tolerance 1e-4
        np.testing.assert_allclose(got, ref, atol=1e-4)

    def test_distance_vector_ok(self):
        """distance_vector returns valid (d, n, ok) with ok=True for all points,
        and routes each point to the closer icosphere.

        Point [0.6, 0, 0] is 0.1 outside sphere A (radius 0.5 at origin):
          |d| should be ~0.1 and the foot should be on sphere A (x<1).
        Point [3.6, 0, 0] is 0.1 outside sphere B (radius 0.5 at (3,0,0)):
          |d| should be ~0.1 and the foot should be on sphere B (x>2.5).
        """
        self._require()
        pts = np.array([
            [0.6, 0.0, 0.0],   # outside A, routes to A
            [3.6, 0.0, 0.0],   # outside B, routes to B
            [1.5, 0.0, 0.0],   # midpoint, routes to whichever is closer
        ], dtype=np.float64)
        d, n, ok = self.union.distance_vector(pts)
        assert ok.all(), f"Expected all ok, got {ok}"
        # d shapes correct
        assert d.shape == (3, 3)
        assert n.shape == (3, 3)
        # n should be approximately unit vectors
        norms = np.linalg.norm(n, axis=1)
        np.testing.assert_allclose(norms, np.ones(3), atol=1e-6)
        # Routing correctness: |d| for pts[0] (outside A) should be ~0.1
        # and for pts[1] (outside B) should be ~0.1.
        # TriMeshOracle foot is on the mesh surface (fp32 BVH, so ~1e-3 tolerance)
        dist0 = np.linalg.norm(d[0])
        dist1 = np.linalg.norm(d[1])
        assert dist0 < 0.15, f"Expected |d|~0.1 for pt outside A, got {dist0:.4f}"
        assert dist1 < 0.15, f"Expected |d|~0.1 for pt outside B, got {dist1:.4f}"
        # foot for pt[0] should be in the A half-space (x < 1)
        foot0 = pts[0] + d[0]
        assert foot0[0] < 1.0, f"Foot for pt outside A should be x<1, got {foot0}"
        # foot for pt[1] should be in the B half-space (x > 2.5)
        foot1 = pts[1] + d[1]
        assert foot1[0] > 2.5, f"Foot for pt outside B should be x>2.5, got {foot1}"

    def test_near_eikonal_false(self):
        """TriMeshOracle has near_eikonal=False => union is False."""
        self._require()
        assert self.union.near_eikonal is False

    def test_velocity_zeros(self):
        """Static geometry: velocity returns zeros (inherited from SDFOracle)."""
        self._require()
        pts = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
        v = self.union.velocity(pts, t=1.0)
        np.testing.assert_array_equal(v, np.zeros_like(pts))
