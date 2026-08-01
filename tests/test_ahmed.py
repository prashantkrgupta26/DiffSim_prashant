"""Ahmed body: watertightness, placement asserts, tiny march gate."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))

from diffsim.cases.ahmed import (ahmed_profile, ahmed_verts_tris,
                                 ahmed_merged)


def _signed_volume(verts, tris):
    a, b, c = (verts[tris[:, i]] for i in range(3))
    return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)


def test_ahmed_watertight_and_oriented():
    verts, tris = ahmed_verts_tris(length=1.0)
    # every edge shared by exactly two triangles, opposite orientation
    edges = {}
    for t in tris:
        for i in range(3):
            e = (int(t[i]), int(t[(i + 1) % 3]))
            edges[e] = edges.get(e, 0) + 1
    for (u, v), cnt in edges.items():
        assert cnt == 1, f"duplicate directed edge {(u, v)}"
        assert edges.get((v, u), 0) == 1, f"unmatched edge {(u, v)}"
    # outward orientation => positive enclosed volume, plausible magnitude
    vol = _signed_volume(verts, tris)
    # H=0.2759, W=0.3726, minus front rounding + slant cut: ~0.09-0.103
    assert 0.080 < vol < 0.103


def test_ahmed_profile_convex_ccw():
    p = ahmed_profile()
    x, y = p[:, 0], p[:, 1]
    # closed CCW: shoelace area positive
    area = 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    assert area > 0
    # convex: all cross products of consecutive edge vectors >= 0 (tol)
    d = np.diff(np.vstack([p, p[:1]]), axis=0)
    cross = d[:-1, 0] * d[1:, 1] - d[:-1, 1] * d[1:, 0]
    assert np.all(cross > -1e-12)


def test_ahmed_clearance_assert():
    with pytest.raises(AssertionError):
        ahmed_merged(length=0.06, clearance=1e-4, band_level=11)


def test_ahmed_tiny_march():
    """Enlarged Ahmed through the body-agnostic pipeline: 3 steps finite."""
    from test_truck_viz import _CFG_PATH
    from diffsim.cases.truck_config import load_truck_config
    from diffsim.cases.truck import run_truck
    cfg = load_truck_config(_CFG_PATH)
    merged = ahmed_merged(length=0.2, clearance=0.02, x_front=0.28,
                          band_level=6)
    res = run_truck(cfg, nsteps=3, base_level=5, truck_band_to=6,
                    band_cells=2, merged=merged, region_refine=False,
                    nu=1.0 / 50.0, dt=0.02, verbose=False)
    assert res["n_excluded"] > 0 and res["sf_faces"] > 0
    for k in ("cd", "cd_surr"):
        assert res[k].shape == (3,)
        assert np.all(np.isfinite(res[k]))
