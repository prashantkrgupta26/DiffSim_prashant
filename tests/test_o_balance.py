import numpy as np
import pytest
from diffsim.octree.build import build_uniform, refine_elements
from diffsim.octree.balance import balance2to1, check_balance

pytestmark = pytest.mark.tier1

def _refine_corner(tree, times):
    for _ in range(times):
        # refine the leaf containing the origin corner (deepest cascade seed)
        anchors = tree.anchors()
        target = np.argmin(anchors.sum(axis=1) + tree.levels.astype(np.int64) * 0)
        mask = np.zeros(len(tree), bool)
        mask[np.lexsort((tree.keys,))[0]] = True   # first leaf in SFC order = origin corner
        tree = refine_elements(tree, mask)
    return tree

def test_O6_lshape_cascade():
    # DEVIATION: The cascade structure created by refining the origin corner is always
    # balanced. Instead, we create an unbalanced tree by refining the first octant twice
    # and the second octant once, creating level differences > 1 at their boundary.
    t = build_uniform(1)
    # Refine the first octant [0,0,0] twice
    mask = np.zeros(len(t), bool)
    mask[0] = True
    t = refine_elements(t, mask)
    mask = np.zeros(len(t), bool)
    mask[0] = True
    t = refine_elements(t, mask)
    # Refine the second octant [0,0,1] once (creates level-4 adjacent to level-2)
    mask = np.zeros(len(t), bool)
    mask[np.lexsort((t.keys,))[1]] = True
    t = refine_elements(t, mask)
    # Now we have level-4 leaves next to level-2 leaves, violating 2:1 balance
    assert not check_balance(t)
    tb = balance2to1(t)
    assert check_balance(tb)
    assert len(tb) > len(t)            # cascading refinement happened
    assert abs(np.sum(tb.h() ** 3) - 1.0) < 1e-14

def test_O6_already_balanced_is_identity():
    t = build_uniform(3)
    tb = balance2to1(t)
    assert len(tb) == len(t) and np.all(tb.keys == t.keys)
