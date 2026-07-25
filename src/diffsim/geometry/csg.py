"""AnalyticCSG geometry backend (spec S4.1 backend table, row 3).

Primitive SDFs + rigid transforms + min/max set operations with optional
polynomial smooth blends, all in torch FP64. Differentiable w.r.t. primitive
parameters (centers, radii, half-widths, rotation, offsets) — the `params`
property collects the leaf tensors; callers set requires_grad.

Symbols: psi = signed distance (< 0 inside the SHAPE; the computational
domain side is chosen downstream by the `domain` flag). q = local box
coordinates minus half-widths. R = rotation matrix (2D angle / 3D axis-angle
Rodrigues); x_local = R^T (x - c), implemented as (x - c) @ R for row-vector
batches.

Invariants: exact primitives (Sphere, Box) are eikonal (|grad psi| = 1 a.e.)
=> near_eikonal = True (eikonal distance shortcut valid). Smooth blends and
exact min/max are NOT globally eikonal => near_eikonal = False, Newton
closest-point projection required (geometry/project.py).
"""
import torch

from .oracle import SDFOracle


def _t(v) -> torch.Tensor:
    if torch.is_tensor(v):
        return v.to(torch.float64)
    return torch.as_tensor(v, dtype=torch.float64)


class Sphere(SDFOracle):
    """psi = |x - c| - r. Exact SDF in any dim (circle at dim = 2)."""

    near_eikonal = True

    def __init__(self, center, radius):
        self.center = _t(center)
        self.radius = _t(radius)
        self.dim = len(self.center)

    @property
    def params(self):
        return [self.center, self.radius]

    def psi(self, x):
        return torch.linalg.norm(x - self.center, dim=1) - self.radius


class Box(SDFOracle):
    """Exact box SDF: q = |x_local| - half; psi = |max(q,0)| + min(max_i q_i, 0).

    rotation: None, scalar angle (dim=2), or axis-angle vector (dim=3,
    Rodrigues). Rigid transforms preserve the SDF property.
    """

    near_eikonal = True

    def __init__(self, center, half, rotation=None):
        self.center = _t(center)
        self.half = _t(half)
        self.rotation = _t(rotation) if rotation is not None else None
        self.dim = len(self.center)

    @property
    def params(self):
        ps = [self.center, self.half]
        if self.rotation is not None:
            ps.append(self.rotation)
        return ps

    def _R(self):
        if self.dim == 2:
            c, s = torch.cos(self.rotation), torch.sin(self.rotation)
            return torch.stack([torch.stack([c, -s]), torch.stack([s, c])])
        if self.dim != 3:
            raise NotImplementedError(
                f"rotated Box only supports dim 2/3 (got dim={self.dim}); "
                "axis-angle Rodrigues below is 3-D-specific (evaluation "
                "geometry-review item — dim=4 rotations need a proper "
                "SO(4) parametrization)")
        # dim = 3: Rodrigues from axis-angle vector; safe at theta -> 0
        th = torch.linalg.norm(self.rotation)
        k = self.rotation / th.clamp_min(1e-300)
        K = torch.stack([
            torch.stack([torch.zeros((), dtype=torch.float64), -k[2], k[1]]),
            torch.stack([k[2], torch.zeros((), dtype=torch.float64), -k[0]]),
            torch.stack([-k[1], k[0], torch.zeros((), dtype=torch.float64)]),
        ])
        eye = torch.eye(3, dtype=torch.float64)
        return eye + torch.sin(th) * K + (1.0 - torch.cos(th)) * (K @ K)

    def psi(self, x):
        xl = x - self.center
        if self.rotation is not None:
            xl = xl @ self._R()          # row-vector form of R^T (x - c)
        q = torch.abs(xl) - self.half
        outside = torch.linalg.norm(torch.clamp(q, min=0.0), dim=1)
        inside = torch.clamp(q.max(dim=1).values, max=0.0)
        return outside + inside


class Plane(SDFOracle):
    """Signed distance to an (unbounded) hyperplane: psi = n_hat . (x - p0),
    with n_hat a unit normal and p0 a point on the plane. Its zero-set is a
    co-dim-1 surface — the THIN-SHELL primitive (ThinShell paper §2.4): a
    zero-thickness rigid surface Gamma with fluid on BOTH sides. Exact SDF
    (|grad psi| = 1) => near_eikonal. For a shell that spans the full extent
    of the computational box (e.g. the blocked-channel plate spanning the whole
    channel height, ThinShell Fig 5) the plane IS the plate exactly.

    Unlike Sphere/Box, NEITHER side of psi = 0 is an obstacle interior: both
    {psi<0} and {psi>0} are fluid. The shell surrogate excludes the band of
    elements the zero-set cuts (classify_shell_intercepted); the two-sided
    extractor then splits the exposed faces into Gamma~+ / Gamma~- by the sign
    of n . n_tilde (extract_two_sided_surrogate)."""

    near_eikonal = True

    def __init__(self, point, normal):
        n = _t(normal)
        self.point = _t(point)
        self.normal = n / torch.linalg.norm(n)
        self.dim = len(self.point)

    @property
    def params(self):
        return [self.point, self.normal]

    def psi(self, x):
        return (x - self.point) @ self.normal


class Segment(SDFOracle):
    """Unsigned distance to a line segment [a, b] in 2-D: psi = |x - proj|,
    the co-dim-1 thin-plate primitive for a FINITE plate (ThinShell §2.4.1,
    Fig 8: flow past a thin plate centered in the channel). psi >= 0 with the
    zero-set exactly the segment; grad psi flips across the plate (two-sided
    normals), so this is a genuine open-surface representation, not a carved
    body. NOT eikonal at the endpoints/segment (the distance-to-a-1-D-set has
    a ridge) => Newton projection.

    The two-sided surrogate never queries points ON the segment (surrogate GPs
    sit O(h) off it in the excluded band), so the endpoint/on-segment gradient
    degeneracy is not exercised by the shell pipeline."""

    near_eikonal = False

    def __init__(self, a, b):
        self.a = _t(a)
        self.b = _t(b)
        self.dim = len(self.a)

    @property
    def params(self):
        return [self.a, self.b]

    def psi(self, x):
        ab = self.b - self.a
        t = ((x - self.a) @ ab) / (ab @ ab)
        t = torch.clamp(t, 0.0, 1.0)
        proj = self.a + t.unsqueeze(1) * ab
        return torch.linalg.norm(x - proj, dim=1)


class FiniteSheet(SDFOracle):
    """Unsigned distance to a finite rectangular patch in 3-D: the co-dim-1
    thin-plate primitive for a FINITE plate in 3-D (dimensional lift of Segment
    to 3-D). The patch is defined by a center, two in-plane half-widths (half),
    and an outward normal (normalized on construction). psi >= 0 with the
    zero-set exactly the rectangular patch; grad psi flips across the plate,
    giving two-sided normals for the shell pipeline.

    Distance field decomposition:
      d_perp   = (x - center) . n_hat           (out-of-plane signed distance)
      p_in     = (x - center) - d_perp * n_hat  (in-plane projection vector)
      (u, v)   = (p_in . t1, p_in . t2)         (local in-plane coords)
      d_in     = ||(u - clamp(u,-h0,h0), v - clamp(v,-h1,h1))||  (0 inside rect)
      psi      = sqrt(d_in^2 + d_perp^2)

    t1, t2 are two orthonormal in-plane basis vectors computed from n_hat via
    the standard least-aligned-axis Gram-Schmidt (stable for any unit normal).

    NOT eikonal at the patch boundary (distance-to-a-2-D-set has a ridge) =>
    near_eikonal = False; Newton projection is used for distance_vector.

    The two-sided surrogate never queries points ON the patch (surrogate GPs
    sit O(h) off it in the excluded band), so the on-patch gradient degeneracy
    is not exercised by the shell pipeline.

    dim = 3 only.
    """

    near_eikonal = False
    dim = 3

    def __init__(self, center, half, normal):
        self.center = _t(center)                           # [3]
        self.half = _t(half)                               # [2] in-plane half-widths
        n = _t(normal)
        self.normal = n / torch.linalg.norm(n)             # [3] unit outward normal
        # build stable in-plane orthonormal basis t1, t2
        abs_n = torch.abs(self.normal)
        i = int(abs_n.argmin())
        e = torch.zeros(3, dtype=torch.float64)
        e[i] = 1.0
        t1 = torch.linalg.cross(self.normal, e)
        self._t1 = t1 / torch.linalg.norm(t1)
        self._t2 = torch.linalg.cross(self.normal, self._t1)

    @property
    def params(self):
        return [self.center, self.half, self.normal]

    def psi(self, x):
        xl = x - self.center                               # [N, 3]
        d_perp = xl @ self.normal                          # [N]   signed out-of-plane
        p_in = xl - d_perp.unsqueeze(1) * self.normal     # [N, 3] in-plane projection
        u = p_in @ self._t1                                # [N]
        v = p_in @ self._t2                                # [N]
        eu = u - torch.clamp(u, -self.half[0], self.half[0])   # [N] in-plane excess
        ev = v - torch.clamp(v, -self.half[1], self.half[1])   # [N]
        d_in_sq = eu ** 2 + ev ** 2                        # [N]
        return torch.sqrt(d_in_sq + d_perp ** 2)           # [N]

    def distance_vector(self, pts, y0=None):
        """Analytic closest-point projection onto the finite rectangular patch.

        Overrides the Newton-based default because psi = sqrt(...) has an
        undefined gradient at the zero-set (the patch itself), causing Newton
        to diverge. Instead we compute the foot directly:

          foot = center + clamp(u, -h0, h0) * t1 + clamp(v, -h1, h1) * t2

        where (u, v) are the in-plane coordinates of x - center.

        n_grad convention (mirroring Plane): we return the unit patch normal
        in the direction of INCREASING signed out-of-plane distance, i.e.
        sign(d_perp) * normal_hat for interior-of-patch feet, and the full
        normalized displacement direction (x - foot)/|x - foot| for
        edge/corner feet. This matches how ``extract_two_sided_surrogate``
        uses n_grad to split the band: the sign of (n_tilde . n_grad) must
        distinguish the two sides of the plate, exactly as it does for Plane.

        Returns (d, n_grad, ok) matching the SDFOracle contract:
          d       [N, 3]  displacement from x to the foot (foot - x)
          n_grad  [N, 3]  unit normal = grad psi / |grad psi| at foot
          ok      [N]     bool mask (always True for the analytic formula)
        """
        import numpy as np
        pts = np.ascontiguousarray(pts, np.float64)
        N = len(pts)
        c = self.center.detach().numpy()
        n = self.normal.detach().numpy()
        t1 = self._t1.detach().numpy()
        t2 = self._t2.detach().numpy()
        h0 = float(self.half[0])
        h1 = float(self.half[1])

        xl = pts - c                                       # [N, 3]
        d_perp = xl @ n                                    # [N]  signed out-of-plane
        u = xl @ t1                                        # [N]
        v = xl @ t2                                        # [N]
        u_proj = np.clip(u, -h0, h0)                       # [N]
        v_proj = np.clip(v, -h1, h1)                       # [N]
        # foot on the patch surface
        foot = c + u_proj[:, None] * t1 + v_proj[:, None] * t2   # [N, 3]
        d = foot - pts                                     # [N, 3]  foot - x

        # n_grad: unit normal used by the shell surrogate to split Gamma~+/~-.
        #
        # Convention (mirrors Plane.psi which returns the SIGNED field
        # psi = (x-p0).n, giving grad psi = n everywhere — constant, not
        # flipping across the zero-set).  The two-sided extractor splits by
        # the sign of I_s = integral(n_tilde . n_grad): for a constant n_grad
        # = normal, cells on the SAME side as normal get positive n_tilde.n
        # and cells on the opposite side get negative — exactly the left/right
        # split needed.
        #
        # For INTERIOR feet (u_proj == u, v_proj == v): x - foot = d_perp*n,
        # so (x-foot)/|x-foot| = sign(d_perp)*n — this FLIPS, so both sides
        # would map to the same I_s sign (both anti-parallel to their n_tilde).
        # For EDGE/CORNER feet the direction is a mixture.
        #
        # Solution: return n_grad = normal (the fixed patch outward normal)
        # for all GPs, regardless of side.  Shell mode (GeometryData.evaluate)
        # then orients n per-face via sign(n_tilde.n_grad)*n_grad so that
        # corr = |n_tilde.n_grad| > 0 on both sides.  The split is:
        #   n_tilde.n_grad > 0 (n_tilde parallel to normal) => Gamma~+
        #   n_tilde.n_grad < 0 (n_tilde anti-parallel to normal) => Gamma~-
        n_grad = np.broadcast_to(n[None, :], (N, 3)).copy()   # [N, 3] = normal
        ok = np.ones(N, bool)
        return d, n_grad, ok


class Complement(SDFOracle):
    """psi -> -psi: a set operation for composing shapes (e.g. a plate with a
    hole). NOT the mechanism for exterior domains — that is the `domain` flag."""

    def __init__(self, child):
        self.child = child
        self.dim = child.dim
        self.near_eikonal = child.near_eikonal

    @property
    def params(self):
        return self.child.params

    def psi(self, x):
        return -self.child.psi(x)


class Translate(SDFOracle):
    def __init__(self, child, offset):
        self.child = child
        self.offset = _t(offset)
        self.dim = child.dim
        self.near_eikonal = child.near_eikonal

    @property
    def params(self):
        return self.child.params + [self.offset]

    def psi(self, x):
        return self.child.psi(x - self.offset)


class _Blend(SDFOracle):
    """Shared smooth-min machinery: smin(a, b, k) with the quadratic
    polynomial blend; k = 0 gives the exact min. smax via -smin(-a, -b, k)."""

    near_eikonal = False   # blends (and exact min/max) are not globally eikonal

    def __init__(self, a, b, k=0.0):
        assert a.dim == b.dim
        self.a, self.b, self.k = a, b, float(k)
        self.dim = a.dim

    @property
    def params(self):
        return self.a.params + self.b.params

    def _smin(self, pa, pb):
        if self.k <= 0.0:
            return torch.minimum(pa, pb)
        h = torch.clamp(0.5 + 0.5 * (pb - pa) / self.k, 0.0, 1.0)
        return torch.lerp(pb, pa, h) - self.k * h * (1.0 - h)


class Union(_Blend):
    def psi(self, x):
        return self._smin(self.a.psi(x), self.b.psi(x))


class Intersection(_Blend):
    def psi(self, x):
        return -self._smin(-self.a.psi(x), -self.b.psi(x))
