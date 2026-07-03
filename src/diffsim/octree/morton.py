"""Morton (Z-order) keys for 3D octrees. Host-side, NumPy-vectorized.

Key layout: interleaved anchor coordinates at LMAX depth (63 bits used);
octant level is carried in a separate uint8 array. Sorting by (key, level)
yields SFC pre-order.
"""
import numpy as np

LMAX = 21
_M0 = np.uint64(0x1FFFFF)

def _part1by2(v: np.ndarray) -> np.ndarray:
    v = v.astype(np.uint64) & _M0
    v = (v | (v << np.uint64(32))) & np.uint64(0x1F00000000FFFF)
    v = (v | (v << np.uint64(16))) & np.uint64(0x1F0000FF0000FF)
    v = (v | (v << np.uint64(8)))  & np.uint64(0x100F00F00F00F00F)
    v = (v | (v << np.uint64(4)))  & np.uint64(0x10C30C30C30C30C3)
    v = (v | (v << np.uint64(2)))  & np.uint64(0x1249249249249249)
    return v

def _compact1by2(v: np.ndarray) -> np.ndarray:
    v = v.astype(np.uint64) & np.uint64(0x1249249249249249)
    v = (v | (v >> np.uint64(2)))  & np.uint64(0x10C30C30C30C30C3)
    v = (v | (v >> np.uint64(4)))  & np.uint64(0x100F00F00F00F00F)
    v = (v | (v >> np.uint64(8)))  & np.uint64(0x1F0000FF0000FF)
    v = (v | (v >> np.uint64(16))) & np.uint64(0x1F00000000FFFF)
    v = (v | (v >> np.uint64(32))) & _M0
    return v

def encode(xyz: np.ndarray, level) -> np.ndarray:
    xyz = np.asarray(xyz, dtype=np.int64).reshape(-1, 3)
    lev = np.asarray(level, dtype=np.int64)
    if np.any(lev > LMAX) or np.any(lev < 0):
        raise ValueError(f"level must be in [0, {LMAX}]")
    bound = np.int64(1) << (lev.reshape(-1, 1) if np.ndim(lev) else lev)
    if np.any(xyz < 0) or np.any(xyz >= bound):
        raise ValueError("coordinate out of range for level")
    shift = (LMAX - lev).astype(np.uint64) if np.ndim(lev) else np.uint64(LMAX - lev)
    ax = xyz[:, 0].astype(np.uint64) << shift
    ay = xyz[:, 1].astype(np.uint64) << shift
    az = xyz[:, 2].astype(np.uint64) << shift
    return (_part1by2(ax) << np.uint64(2)) | (_part1by2(ay) << np.uint64(1)) | _part1by2(az)

def decode(keys: np.ndarray, levels: np.ndarray) -> np.ndarray:
    keys = np.asarray(keys, dtype=np.uint64)
    shift = (LMAX - np.asarray(levels, dtype=np.int64)).astype(np.uint64)
    ax = _compact1by2(keys >> np.uint64(2)) >> shift
    ay = _compact1by2(keys >> np.uint64(1)) >> shift
    az = _compact1by2(keys) >> shift
    return np.stack([ax, ay, az], axis=1).astype(np.int64)

def anchors(keys: np.ndarray) -> np.ndarray:
    """Anchor coordinates at LMAX depth (level-independent)."""
    keys = np.asarray(keys, dtype=np.uint64)
    return np.stack([_compact1by2(keys >> np.uint64(2)),
                     _compact1by2(keys >> np.uint64(1)),
                     _compact1by2(keys)], axis=1).astype(np.int64)

def children(keys: np.ndarray, levels: np.ndarray):
    xyz = decode(keys, levels)
    lev = np.asarray(levels, dtype=np.int64)
    ck = np.empty((len(xyz), 8), dtype=np.uint64)
    for m in range(8):
        off = np.array([(m >> 2) & 1, (m >> 1) & 1, m & 1], dtype=np.int64)
        ck[:, m] = encode(xyz * 2 + off, lev + 1)
    return ck, (lev + 1).astype(np.uint8)

def parent(keys: np.ndarray, levels: np.ndarray):
    xyz = decode(keys, levels)
    lev = np.asarray(levels, dtype=np.int64)
    return encode(xyz // 2, lev - 1), (lev - 1).astype(np.uint8)
