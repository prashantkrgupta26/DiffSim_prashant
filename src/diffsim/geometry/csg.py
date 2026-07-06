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
