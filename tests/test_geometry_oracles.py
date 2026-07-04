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
