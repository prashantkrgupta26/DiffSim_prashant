import numpy as np
import pytest
from diffsim.octree import morton

pytestmark = pytest.mark.tier1

def test_O1_encode_decode_bijective():
    rng = np.random.default_rng(1)
    for level in [0, 1, 5, 12, morton.LMAX]:
        n = 1000
        xyz = rng.integers(0, 2**level, size=(n, 3), dtype=np.int64)
        keys = morton.encode(xyz, level)
        back = morton.decode(keys, np.full(n, level, dtype=np.uint8))
        assert np.array_equal(back, xyz)

def test_O1_overflow_rejected():
    with pytest.raises(ValueError):
        morton.encode(np.array([[0, 0, 0]]), morton.LMAX + 1)
    with pytest.raises(ValueError):
        morton.encode(np.array([[2, 0, 0]]), 1)  # coord out of range at level 1

def test_O2_parent_children_roundtrip_and_z_order():
    xyz = np.array([[3, 5, 7]], dtype=np.int64)
    keys = morton.encode(xyz, 4)
    levels = np.array([4], dtype=np.uint8)
    ck, cl = morton.children(keys, levels)
    assert ck.shape == (1, 8) and np.all(cl == 5)
    pk, pl = morton.parent(ck.reshape(-1), np.full(8, 5, dtype=np.uint8))
    assert np.all(pk == keys[0]) and np.all(pl == 4)
    assert np.all(np.diff(ck[0].astype(np.uint64)) > 0)  # emitted in ascending Morton order
