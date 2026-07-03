import numpy as np
import pytest
from diffsim.octree import morton
from diffsim.octree.build import Octree, build_uniform, build_adaptive, refine_elements, sort_unique

pytestmark = pytest.mark.tier1

def test_O3_uniform_count():
    for lvl in range(0, 5):
        t = build_uniform(lvl)
        assert len(t) == 8**lvl
        assert np.all(np.diff(t.keys.astype(np.uint64)) > 0)  # sorted, unique

def test_O4_local_refine_count():
    t = build_uniform(2)          # 64 elements
    mask = np.zeros(len(t), bool); mask[10] = True
    t2 = refine_elements(t, mask)
    assert len(t2) == 64 - 1 + 8
    # reverse: coarsening those 8 back is Task-scope M4; count identity only here.

def test_O5_duplicate_removal():
    t = build_uniform(2)
    k = np.concatenate([t.keys, t.keys]); l = np.concatenate([t.levels, t.levels])
    k2, l2 = sort_unique(k, l)
    assert len(k2) == 64 and np.all(k2 == t.keys) and np.all(l2 == t.levels)
    k3, l3 = sort_unique(np.array([], np.uint64), np.array([], np.uint8))
    assert len(k3) == 0

def test_adaptive_refine_center():
    # refine only octants containing the domain center, to max_level 4
    def pred(centers, h):
        return np.all(np.abs(centers - 0.5) < h, axis=1)
    t = build_adaptive(pred, max_level=4)
    assert t.levels.max() == 4 and t.levels.min() < 4
    vol = np.sum(t.h() ** 3)
    assert abs(vol - 1.0) < 1e-14   # leaves tile the cube exactly
