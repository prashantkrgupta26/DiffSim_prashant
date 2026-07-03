import numpy as np
import pytest
from diffsim.octree.build import Octree, build_uniform, refine_elements, build_adaptive
from diffsim.octree.carve import SphereOracle, carve

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
