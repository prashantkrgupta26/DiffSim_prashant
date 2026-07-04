"""Tier-4 geometry oracle tests: AnalyticCSG backend (Task 2), Newton
closest-point projection + IFT adjoint (Task 3), TriMesh/GridSDF backends
(Task 9). Spec S4.1: oracle contract; sign convention: psi < 0 inside the
shape, domain side chosen downstream by the `domain` flag."""
import numpy as np
import pytest
import torch
from diffsim.geometry.csg import Sphere, Box, Complement, Union, Intersection, Translate

pytestmark = pytest.mark.tier4


def test_sphere_sdf_exact():
    for dim in (2, 3):
        s = Sphere((0.5,) * dim, 0.3)
        rng = np.random.default_rng(7)
        pts = rng.uniform(0, 1, (200, dim))
        psi = s.classify(pts)
        assert np.allclose(psi, np.linalg.norm(pts - 0.5, axis=1) - 0.3, atol=1e-14)


def test_rotated_box_sdf():
    th = 0.4
    b = Box((0.5, 0.5), (0.2, 0.1), rotation=torch.tensor(th, dtype=torch.float64))
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    corner = 0.5 + R @ np.array([0.2, 0.1])          # rotated corner lies on the boundary
    assert abs(b.classify(corner[None, :])[0]) < 1e-14
    assert b.classify(np.array([[0.5, 0.5]]))[0] < 0  # center inside


def test_rotated_box_3d_sdf():
    rot = torch.tensor([0.3, 0.5, 0.2], dtype=torch.float64)
    b = Box((0.5, 0.5, 0.5), (0.25, 0.18, 0.22), rotation=rot)
    # center inside; far corner of the unit cube outside
    assert b.classify(np.array([[0.5, 0.5, 0.5]]))[0] < 0
    assert b.classify(np.array([[0.0, 0.0, 0.0]]))[0] > 0
    # a point on the +x local face: rotate local (0.25, 0, 0) into world coords
    th = float(np.linalg.norm(rot.numpy()))
    k = rot.numpy() / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    R = np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)
    face_pt = 0.5 + R @ np.array([0.25, 0.0, 0.0])
    assert abs(b.classify(face_pt[None, :])[0]) < 1e-14


def test_complement_and_blend():
    dom = Complement(Sphere((0.5, 0.5), 0.25))        # exterior as a set operation
    assert dom.classify(np.array([[0.5, 0.5]]))[0] > 0     # inside sphere = outside complement
    assert dom.classify(np.array([[0.05, 0.05]]))[0] < 0
    a, b = Sphere((0.4, 0.5), 0.2), Sphere((0.6, 0.5), 0.2)
    uni = Union(a, b, k=0.05)
    x = np.array([[0.5, 0.5]])
    assert uni.classify(x)[0] <= min(a.classify(x)[0], b.classify(x)[0]) + 1e-12
    assert not uni.near_eikonal and Sphere((0, 0), 1.0).near_eikonal
    # intersection: point in a but not b is outside the intersection
    inter = Intersection(a, b)
    xa = np.array([[0.25, 0.5]])
    assert a.classify(xa)[0] < 0 and inter.classify(xa)[0] > 0


def test_translate():
    s = Translate(Sphere((0.0, 0.0), 0.3), (0.5, 0.5))
    assert abs(s.classify(np.array([[0.8, 0.5]]))[0]) < 1e-14


def test_params_graph():
    s = Sphere((0.5, 0.5), 0.3)
    for p in s.params:
        p.requires_grad_(True)
    x = torch.tensor([[0.9, 0.5]], dtype=torch.float64)
    s.psi(x).sum().backward()
    grads = [p.grad for p in s.params]
    assert all(g is not None for g in grads)
    # d psi / d r = -1 exactly
    assert abs(float(s.radius.grad) + 1.0) < 1e-14


def test_composed_params_graph():
    # params of a blend collect both children's leaves; gradients flow through smin
    a, b = Sphere((0.4, 0.5), 0.2), Sphere((0.6, 0.5), 0.2)
    uni = Union(a, b, k=0.05)
    for p in uni.params:
        p.requires_grad_(True)
    x = torch.tensor([[0.5, 0.62]], dtype=torch.float64)   # near both: blend region
    uni.psi(x).sum().backward()
    assert a.radius.grad is not None and b.radius.grad is not None
    assert float(a.radius.grad) < 0 and float(b.radius.grad) < 0


def test_velocity_static_zero():
    s = Sphere((0.5, 0.5), 0.3)
    v = s.velocity(np.zeros((4, 2)), t=1.0)
    assert v.shape == (4, 2) and not v.any()


# ---------------------------------------------------------------------------
# Task 3: Newton closest-point projection + IFT adjoint (Tier-2 VJP #3)
# ---------------------------------------------------------------------------
from diffsim.geometry.project import distance_numpy, distance_torch, closest_point
from diffsim.geometry.oracle import admissibility


def test_projection_lands_on_boundary():
    sphere3 = Sphere((0.5, 0.5, 0.5), 0.3)
    blend2 = Union(Sphere((0.4, 0.5), 0.2), Sphere((0.6, 0.5), 0.2), k=0.05)
    for oracle, dim in ((sphere3, 3), (blend2, 2)):
        rng = np.random.default_rng(3)
        x = 0.5 + 0.25 * rng.uniform(-1, 1, (100, dim))
        d, n, ok = distance_numpy(oracle, x)
        assert ok.all()
        assert np.abs(oracle.classify(x + d)).max() < 1e-11        # y on the zero set
        assert np.allclose(np.linalg.norm(n, axis=1), 1.0, atol=1e-12)


def test_projection_matches_analytic_sphere():
    s = Sphere((0.5, 0.5), 0.3)
    x = np.array([[0.9, 0.5], [0.5, 0.65], [0.2, 0.2]])
    d, n, ok = distance_numpy(s, x)
    r = np.linalg.norm(x - 0.5, axis=1)
    d_exact = (0.3 - r)[:, None] * (x - 0.5) / r[:, None]
    assert ok.all()
    assert np.allclose(d, d_exact, atol=1e-11)
    assert np.allclose(n, (x - 0.5) / r[:, None], atol=1e-11)


def test_newton_projection_on_blend_boundary():
    # Newton path (near_eikonal=False): projected point must satisfy the
    # closest-point stationarity: d parallel to grad psi at y.
    o = Union(Sphere((0.4, 0.5), 0.2), Sphere((0.6, 0.5), 0.2), k=0.05)
    x = np.array([[0.52, 0.71], [0.30, 0.44], [0.5, 0.35]])
    d, n, ok = distance_numpy(o, x)
    assert ok.all()
    cross = d[:, 0] * n[:, 1] - d[:, 1] * n[:, 0]      # 2D cross product
    assert np.abs(cross).max() < 1e-9


def _make_blend(requires):
    o = Union(Sphere((0.4, 0.5), 0.2), Sphere((0.6, 0.5), 0.2), k=0.05)
    for p in o.params:
        p.requires_grad_(requires)
    return o


def test_ift_vs_unrolled_gradient():
    """Tier-2 VJP #3 check: IFT backward == unrolled-autograd twin."""
    x = np.array([[0.52, 0.71], [0.30, 0.44]])
    W = torch.tensor([[1.0, 2.0], [3.0, -1.0]], dtype=torch.float64)

    o = _make_blend(True)
    d_t, n_t, ok = distance_torch(o, x)                     # IFT path
    assert ok.all()
    L = (d_t * W).sum() + 0.3 * n_t.sum()
    g_ift = torch.autograd.grad(L, o.params)

    # unrolled twin: the SAME algorithm the forward runs (gradient-projection
    # warm start, then full Newton on the augmented closest-point system),
    # with the autograd graph kept through every step including the solves
    o2 = _make_blend(True)
    xt = torch.as_tensor(x, dtype=torch.float64)
    N, dim = xt.shape
    y = xt.clone().requires_grad_(True)

    def psi_grad(yy):
        psi = o2.psi(yy)
        (g,) = torch.autograd.grad(psi.sum(), yy, create_graph=True)
        return psi, g

    for _ in range(15):                                     # warm start
        psi, g = psi_grad(y)
        y = y - (psi / (g * g).sum(dim=1)).unsqueeze(1) * g
    psi, g = psi_grad(y)
    s = ((xt - y) * g).sum(dim=1) / (g * g).sum(dim=1)
    eye = torch.eye(dim, dtype=torch.float64).expand(N, dim, dim)
    for _ in range(25):                                     # augmented Newton
        psi, g = psi_grad(y)
        H = torch.stack(
            [torch.autograd.grad(g[:, i].sum(), y, create_graph=True)[0]
             for i in range(dim)], dim=1)
        F = torch.cat([y - xt + s.unsqueeze(1) * g, psi.unsqueeze(1)], dim=1)
        top = torch.cat([eye + s.view(N, 1, 1) * H, g.unsqueeze(2)], dim=2)
        bot = torch.cat([g.unsqueeze(1),
                         torch.zeros(N, 1, 1, dtype=torch.float64)], dim=2)
        Jz = torch.cat([top, bot], dim=1)
        dz = torch.linalg.solve(Jz, F.unsqueeze(2)).squeeze(2)
        y = y - dz[:, :dim]
        s = s - dz[:, dim]
    d2 = y - xt
    psi_f, gf = psi_grad(y)
    n2 = gf / gf.norm(dim=1, keepdim=True)
    L2 = (d2 * W).sum() + 0.3 * n2.sum()
    g_unrolled = torch.autograd.grad(L2, o2.params)

    for a, b in zip(g_ift, g_unrolled):
        assert torch.allclose(a, b, atol=1e-9), (a, b)


def test_ift_gradient_vs_fd():
    # radius gradient of a scalar functional of d, via IFT vs central FD
    x = np.array([[0.52, 0.71], [0.30, 0.44]])
    W = torch.tensor([[1.0, 2.0], [3.0, -1.0]], dtype=torch.float64)

    def loss_at(r):
        o = Union(Sphere((0.4, 0.5), 0.2), Sphere((0.6, 0.5), r), k=0.05)
        d_t, n_t, ok = distance_torch(o, x)
        return float((d_t * W).sum())

    o = _make_blend(True)
    d_t, n_t, ok = distance_torch(o, x)
    (g_r,) = torch.autograd.grad((d_t * W).sum(), [o.b.radius])
    eps = 1e-6
    fd = (loss_at(0.2 + eps) - loss_at(0.2 - eps)) / (2 * eps)
    assert abs(float(g_r) - fd) < 1e-6 * max(abs(fd), 1.0), (float(g_r), fd)


def test_admissibility_sphere():
    s = Sphere((0.5, 0.5, 0.5), 0.3)
    band = 0.5 + 0.31 * np.random.default_rng(1).uniform(-1, 1, (200, 3))
    diag = admissibility(s, band)
    assert diag["newton_ok_frac"] == 1.0
    assert diag["eps_inf"] < 1e-11
    assert abs(diag["c0"] - 1.0) < 1e-12
    assert diag["d_hausdorff_bound"] < 1e-11


def test_projection_4d():
    # k=4 structural invariant (dim-coverage policy): projection and IFT
    # machinery are dim-generic. Eikonal shortcut + Newton path on a 4-sphere.
    s = Sphere((0.5,) * 4, 0.35)
    rng = np.random.default_rng(11)
    x = 0.5 + 0.3 * rng.uniform(-1, 1, (50, 4))
    d, n, ok = distance_numpy(s, x)
    assert ok.all()
    r = np.linalg.norm(x - 0.5, axis=1)
    assert np.allclose(np.linalg.norm(x + d - 0.5, axis=1), 0.35, atol=1e-11)
    assert np.allclose(n, (x - 0.5) / r[:, None], atol=1e-11)
    # Newton path (non-eikonal blend of two 4-spheres) + IFT gradient smoke
    a = Sphere((0.45,) * 4, 0.25)
    b = Sphere((0.55,) * 4, 0.25)
    o = Union(a, b, k=0.03)
    for pp in o.params:
        pp.requires_grad_(True)
    d_t, n_t, ok = distance_torch(o, x[:8])
    assert ok.numpy().all()
    (g_r,) = torch.autograd.grad(d_t.sum(), [a.radius])
    assert np.isfinite(float(g_r))
