"""Morton (Z-order) keys for k-D trees, k in {2, 3, 4}. Host-side, NumPy-vectorized.

Spec: design doc S11.1. Key layout: per-axis coordinates at lmax(dim) depth,
bit-interleaved into one uint64 with axis 0 (x) MOST significant per bit
group; octant level carried in a separate uint8 array. Sorting by
(key, level) yields SFC pre-order. LMAX_FOR = {2: 31, 3: 21, 4: 15}:
dim*lmax bits used, level 2^lmax cells per axis maximum.

Symbols: xyz = integer octant coords at the octant's own level;
anchors = coords scaled to the lmax(dim) grid (level-independent).
Invariant: encode/decode are exact inverses for in-range input;
out-of-range level or coordinate raises ValueError (never truncates).
"""
import numpy as np

LMAX_FOR = {2: 31, 3: 21, 4: 15}
LMAX = 21  # 3D back-compat constant; new code uses lmax(dim)

def lmax(dim: int) -> int:
    return LMAX_FOR[dim]

# --- bit spread/compact: insert (dim-1) zero bits between input bits ------
# Standard magic-mask ladders; each mask keeps the bits at their target
# stride. Verified by the bijectivity tests over the full level range.

def _spread2(v):  # 31 bits -> every 2nd bit
    v = v.astype(np.uint64) & np.uint64(0x7FFFFFFF)
    v = (v | (v << np.uint64(16))) & np.uint64(0x0000FFFF0000FFFF)
    v = (v | (v << np.uint64(8)))  & np.uint64(0x00FF00FF00FF00FF)
    v = (v | (v << np.uint64(4)))  & np.uint64(0x0F0F0F0F0F0F0F0F)
    v = (v | (v << np.uint64(2)))  & np.uint64(0x3333333333333333)
    v = (v | (v << np.uint64(1)))  & np.uint64(0x5555555555555555)
    return v

def _compact2(v):
    v = v.astype(np.uint64) & np.uint64(0x5555555555555555)
    v = (v | (v >> np.uint64(1)))  & np.uint64(0x3333333333333333)
    v = (v | (v >> np.uint64(2)))  & np.uint64(0x0F0F0F0F0F0F0F0F)
    v = (v | (v >> np.uint64(4)))  & np.uint64(0x00FF00FF00FF00FF)
    v = (v | (v >> np.uint64(8)))  & np.uint64(0x0000FFFF0000FFFF)
    v = (v | (v >> np.uint64(16))) & np.uint64(0x7FFFFFFF)
    return v

def _spread3(v):  # 21 bits -> every 3rd bit (M0's part1by2, unchanged)
    v = v.astype(np.uint64) & np.uint64(0x1FFFFF)
    v = (v | (v << np.uint64(32))) & np.uint64(0x1F00000000FFFF)
    v = (v | (v << np.uint64(16))) & np.uint64(0x1F0000FF0000FF)
    v = (v | (v << np.uint64(8)))  & np.uint64(0x100F00F00F00F00F)
    v = (v | (v << np.uint64(4)))  & np.uint64(0x10C30C30C30C30C3)
    v = (v | (v << np.uint64(2)))  & np.uint64(0x1249249249249249)
    return v

def _compact3(v):
    v = v.astype(np.uint64) & np.uint64(0x1249249249249249)
    v = (v | (v >> np.uint64(2)))  & np.uint64(0x10C30C30C30C30C3)
    v = (v | (v >> np.uint64(4)))  & np.uint64(0x100F00F00F00F00F)
    v = (v | (v >> np.uint64(8)))  & np.uint64(0x1F0000FF0000FF)
    v = (v | (v >> np.uint64(16))) & np.uint64(0x1F00000000FFFF)
    v = (v | (v >> np.uint64(32))) & np.uint64(0x1FFFFF)
    return v

def _spread4(v):  # 15 bits -> every 4th bit
    v = v.astype(np.uint64) & np.uint64(0x7FFF)
    v = (v | (v << np.uint64(24))) & np.uint64(0x000000FF000000FF)
    v = (v | (v << np.uint64(12))) & np.uint64(0x000F000F000F000F)
    v = (v | (v << np.uint64(6)))  & np.uint64(0x0303030303030303)
    v = (v | (v << np.uint64(3)))  & np.uint64(0x1111111111111111)
    return v

def _compact4(v):
    v = v.astype(np.uint64) & np.uint64(0x1111111111111111)
    v = (v | (v >> np.uint64(3)))  & np.uint64(0x0303030303030303)
    v = (v | (v >> np.uint64(6)))  & np.uint64(0x000F000F000F000F)
    v = (v | (v >> np.uint64(12))) & np.uint64(0x000000FF000000FF)
    v = (v | (v >> np.uint64(24))) & np.uint64(0x7FFF)
    return v

_SPREAD = {2: _spread2, 3: _spread3, 4: _spread4}
_COMPACT = {2: _compact2, 3: _compact3, 4: _compact4}

def encode(xyz: np.ndarray, level, dim: int = 3) -> np.ndarray:
    xyz = np.asarray(xyz, np.int64).reshape(-1, dim)
    lev = np.asarray(level, np.int64)
    L = lmax(dim)
    if np.any(lev > L) or np.any(lev < 0):
        raise ValueError(f"level must be in [0, {L}] for dim={dim}")
    bound = np.int64(1) << (lev.reshape(-1, 1) if np.ndim(lev) else lev)
    if np.any(xyz < 0) or np.any(xyz >= bound):
        raise ValueError("coordinate out of range for level")
    shift = (L - lev).astype(np.uint64) if np.ndim(lev) else np.uint64(L - lev)
    spread = _SPREAD[dim]
    key = np.zeros(len(xyz), np.uint64)
    # axis 0 (x) gets the most-significant bit of each dim-bit group
    for i in range(dim):
        key |= spread(xyz[:, i].astype(np.uint64) << shift) << np.uint64(dim - 1 - i)
    return key

def decode(keys: np.ndarray, levels: np.ndarray, dim: int = 3) -> np.ndarray:
    keys = np.asarray(keys, np.uint64)
    L = lmax(dim)
    shift = (L - np.asarray(levels, np.int64)).astype(np.uint64)
    compact = _COMPACT[dim]
    out = np.empty((len(keys), dim), np.int64)
    for i in range(dim):
        out[:, i] = (compact(keys >> np.uint64(dim - 1 - i)) >> shift).astype(np.int64)
    return out

def anchors(keys: np.ndarray, dim: int = 3) -> np.ndarray:
    """Anchor coordinates at lmax(dim) depth (level-independent)."""
    keys = np.asarray(keys, np.uint64)
    compact = _COMPACT[dim]
    return np.stack([compact(keys >> np.uint64(dim - 1 - i)).astype(np.int64)
                     for i in range(dim)], axis=1)

def children(keys: np.ndarray, levels: np.ndarray, dim: int = 3):
    xyz = decode(keys, levels, dim=dim)
    lev = np.asarray(levels, np.int64)
    n_child = 1 << dim
    ck = np.empty((len(xyz), n_child), np.uint64)
    # child index m carries axis-i offset in bit (dim-1-i): matches the key
    # interleave (x most significant), so ascending m == ascending Morton.
    for m in range(n_child):
        off = np.array([(m >> (dim - 1 - i)) & 1 for i in range(dim)], np.int64)
        ck[:, m] = encode(xyz * 2 + off, lev + 1, dim=dim)
    return ck, (lev + 1).astype(np.uint8)

def parent(keys: np.ndarray, levels: np.ndarray, dim: int = 3):
    xyz = decode(keys, levels, dim=dim)
    lev = np.asarray(levels, np.int64)
    return encode(xyz // 2, lev - 1, dim=dim), (lev - 1).astype(np.uint8)
