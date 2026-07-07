"""M1 audit gap fills (coverage doc, 2026-07-06): SBM4 (d_RMS vs lambda)
and SBM7b (pathological-geometry classification) — geometry-only, no
solves."""
import numpy as np
import pytest

from diffsim.octree.build import build_uniform
from diffsim.geometry.csg import Sphere
from diffsim.sbm.surrogate import (classify_lambda, extract_surrogate,
                                   GeometryData, face_gauss_points)
from diffsim.mesh.faces import face_tables

pytestmark = pytest.mark.geometry


def _d_rms(lam, level=5):
    oracle = Sphere((0.5, 0.5), 0.3)
    tree = build_uniform(level, dim=2)
    ret, _ = classify_lambda(tree, oracle, lam)
    sf = extract_surrogate(ret)
    geo = GeometryData.evaluate(oracle, ret, sf, face_tables(1, 2))
    return float(np.sqrt((geo.d ** 2).sum(1).mean()))


def test_sbm4_drms_vs_lambda(device):
    """SBM4: the surrogate's distance-vector RMS as a function of lambda.
    lambda=0.5 must be optimal-or-near (the draft's optimal-surrogate
    claim): d_RMS(0.5) <= d_RMS(0) and <= d_RMS(1)."""
    d0, d05, d1 = _d_rms(0.0), _d_rms(0.5), _d_rms(1.0)
    print(f"d_RMS: lam=0 {d0:.5f}, lam=0.5 {d05:.5f}, lam=1 {d1:.5f}")
    assert d05 <= d0 * 1.05, (d05, d0)
    assert d05 <= d1 * 1.05, (d05, d1)


@pytest.mark.parametrize("case", ["thin_gap", "small_hole"])
def test_sbm7b_pathological_classification(case, device):
    """SBM7(b): classification survives pathological geometry — a thin
    gap between two spheres and a small interior hole — producing a
    consistent surrogate (faces exist, geometry data evaluates, area
    correction bounded)."""
    from diffsim.geometry.csg import Union, Complement, Intersection
    if case == "thin_gap":
        # two spheres separated by ~1.5 cells at L5
        oracle = Union(Sphere((0.32, 0.5), 0.15),
                       Sphere((0.68, 0.5), 0.15))
    else:
        # annulus: sphere minus a small hole
        oracle = Intersection(Sphere((0.5, 0.5), 0.3),
                              Complement(Sphere((0.5, 0.5), 0.08)))
    tree = build_uniform(5, dim=2)
    ret, _ = classify_lambda(tree, oracle, 0.5)
    sf = extract_surrogate(ret)
    assert len(sf.elem) > 8, "surrogate collapsed on pathological case"
    geo = GeometryData.evaluate(oracle, ret, sf, face_tables(1, 2))
    assert np.isfinite(geo.d).all() and np.isfinite(geo.n).all()
    assert (np.abs(geo.corr) <= 1.0 + 1e-12).all()
    # distance vectors bounded by a couple of cells (no far-sheet feet)
    h = 1.0 / 2 ** 5
    assert np.linalg.norm(geo.d, axis=1).max() < 3 * h
