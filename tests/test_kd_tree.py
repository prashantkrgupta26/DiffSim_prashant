import numpy as np
import pytest
from diffsim.octree.build import Octree, build_uniform, refine_elements, build_adaptive
from diffsim.octree.carve import SphereOracle, carve
from diffsim.octree import morton
from diffsim.octree.lookup import LeafLookup, face_neighbors, face_offsets
from diffsim.octree.balance import balance2to1, check_balance

pytestmark = pytest.mark.tier1

@pytest.mark.parametrize("dim", [2, 3, 4])
def test_kd_uniform_count_and_volume(dim):
    for lvl in range(0, 4 if dim < 4 else 3):
        t = build_uniform(lvl, dim=dim)
        assert len(t) == (2**dim) ** lvl
        assert t.dim == dim and t.periodic == (False,) * dim
        assert abs(np.sum(t.h() ** dim) - 1.0) < 1e-13   # leaves tile the unit k-cube

@pytest.mark.parametrize("dim", [2, 4])
def test_kd_refine_count(dim):
    t = build_uniform(1, dim=dim)
    mask = np.zeros(len(t), bool); mask[0] = True
    t2 = refine_elements(t, mask)
    assert len(t2) == len(t) - 1 + 2**dim
    assert t2.dim == dim                              # dim/periodic survive refinement

@pytest.mark.parametrize("dim", [2, 4])
def test_kd_face_neighbors_and_boundary(dim):
    t = build_uniform(2, dim=dim)
    nbrs = face_neighbors(t)
    assert len(nbrs) == 2 * dim
    a = t.anchors()
    xmin = a[:, 0] == 0
    assert np.all(nbrs[0][xmin] == -1) and np.all(nbrs[0][~xmin] >= 0)

def test_periodic_wrap_2d():
    t = build_uniform(2, dim=2, periodic=(True, False))
    nbrs = face_neighbors(t)
    assert np.all(nbrs[0] >= 0) and np.all(nbrs[1] >= 0)     # x wraps: no boundary
    ymin = t.anchors()[:, 1] == 0
    assert np.all(nbrs[2][ymin] == -1)                        # y does not wrap
    # wrap correctness: X_MINUS neighbor of an x=0 element is an x=max element
    e0 = np.where((t.anchors()[:, 0] == 0))[0][0]
    j = nbrs[0][e0]
    assert t.anchors()[j, 0] == (1 << morton.lmax(2)) - (1 << (morton.lmax(2) - 2))

@pytest.mark.parametrize("dim", [2, 4])
def test_kd_balance(dim):
    t = build_uniform(1, dim=dim)
    for _ in range(3):                       # deep nested + a second seed = imbalance
        mask = np.zeros(len(t), bool); mask[np.lexsort((t.keys,))[0]] = True
        t = refine_elements(t, mask)
    mask = np.zeros(len(t), bool); mask[np.lexsort((t.keys,))[1]] = True
    t = refine_elements(t, mask)
    tb = balance2to1(t)
    assert check_balance(tb)
    assert abs(np.sum(tb.h() ** dim) - 1.0) < 1e-13

def test_periodic_balance_wraps():
    # refine an x=0-adjacent element deeply; with x periodic the x=max column
    # must be forced to refine by the wrap-adjacency
    t = build_uniform(1, dim=2, periodic=(True, False))
    for _ in range(2):
        mask = np.zeros(len(t), bool); mask[np.lexsort((t.keys,))[0]] = True
        t = refine_elements(t, mask)
    tb = balance2to1(t)
    assert check_balance(tb)
    assert len(tb) > len(t)

@pytest.mark.parametrize("dim", [2, 3, 4])
def test_kd_carve_sphere(dim):
    lvl = {2: 5, 3: 4, 4: 5}[dim]  # 4D needs level 5 to achieve 0.35 error tolerance
    oracle = SphereOracle((0.5,) * dim, 0.45)
    tree, markers = carve(build_uniform(lvl, dim=dim), oracle)
    assert set(np.unique(markers)) <= {0, 1}
    vol = np.sum(tree.h() ** dim)
    # k-ball volume: pi r^2 (2D), 4/3 pi r^3 (3D), pi^2/2 r^4 (4D)
    vball = {2: np.pi * 0.45**2, 3: 4/3 * np.pi * 0.45**3, 4: np.pi**2 / 2 * 0.45**4}[dim]
    assert abs(vol - vball) / vball < 0.35            # coarse-level tolerance
