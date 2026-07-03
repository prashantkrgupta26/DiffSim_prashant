import numpy as np
import pytest
from diffsim.octree import morton

pytestmark = pytest.mark.tier1

@pytest.mark.parametrize("dim", [2, 3, 4])
def test_kd_encode_decode_bijective(dim):
    rng = np.random.default_rng(dim)
    L = morton.lmax(dim)
    for level in [0, 1, 5, L // 2, L]:
        xyz = rng.integers(0, 2**level, size=(500, dim), dtype=np.int64)
        keys = morton.encode(xyz, level, dim=dim)
        back = morton.decode(keys, np.full(500, level, np.uint8), dim=dim)
        assert np.array_equal(back, xyz)

@pytest.mark.parametrize("dim,L", [(2, 31), (3, 21), (4, 15)])
def test_kd_lmax_and_overflow(dim, L):
    assert morton.lmax(dim) == L
    with pytest.raises(ValueError):
        morton.encode(np.zeros((1, dim), np.int64), L + 1, dim=dim)
    with pytest.raises(ValueError):
        morton.encode(np.full((1, dim), 2, np.int64), 1, dim=dim)  # coord out of range

@pytest.mark.parametrize("dim", [2, 3, 4])
def test_kd_children_parent_roundtrip(dim):
    xyz = np.arange(1, dim + 1, dtype=np.int64).reshape(1, dim)
    keys = morton.encode(xyz, 4, dim=dim)
    ck, cl = morton.children(keys, np.array([4], np.uint8), dim=dim)
    assert ck.shape == (1, 2**dim) and np.all(cl == 5)
    assert np.all(np.diff(ck[0].astype(np.uint64)) > 0)   # ascending Morton order
    pk, pl = morton.parent(ck.reshape(-1), np.full(2**dim, 5, np.uint8), dim=dim)
    assert np.all(pk == keys[0]) and np.all(pl == 4)

def test_3d_backcompat_constant():
    assert morton.LMAX == 21 == morton.lmax(3)
