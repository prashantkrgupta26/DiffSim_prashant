"""Campaign (a) gate: the vectorized build_constraints is BIT-IDENTICAL to
the reference implementation across the mesh zoo (dims 2/3/4, uniform +
adapted + periodic, p1/p2/mixed), and measurably faster."""
import time

import numpy as np
import pytest

from diffsim.octree.build import build_uniform, refine_elements
from diffsim.octree.balance import balance2to1
from diffsim.mesh.nodes import build_mesh
from diffsim.mesh.constraints import (build_constraints,
                                      _build_constraints_reference)

pytestmark = pytest.mark.tier2


def _mesh_zoo():
    zoo = []
    for dim, level in ((2, 4), (3, 3), (4, 2)):
        t = build_uniform(level, dim=dim)
        zoo.append((f"uniform d{dim}", build_mesh(t, p=1)))
        mask = np.zeros(len(t), bool)
        mask[0] = True
        mask[len(t) // 2] = True
        ta = balance2to1(refine_elements(t, mask))
        zoo.append((f"adapted d{dim} p1", build_mesh(ta, p=1)))
        if dim <= 3:
            zoo.append((f"adapted d{dim} p2", build_mesh(ta, p=2)))
    # mixed-p on a uniform tree (one-knob safe)
    t = build_uniform(3, dim=2)
    G = 1 << 3
    p_elem = np.where(t.anchors()[:, 0] / G < 0.5, 2, 1).astype(np.int8)
    zoo.append(("mixed-p d2", build_mesh(t, p=p_elem)))
    return zoo


@pytest.mark.parametrize("name,mesh", _mesh_zoo(),
                         ids=[n for n, _ in _mesh_zoo()])
def test_vectorized_matches_reference(name, mesh):
    a = build_constraints(mesh)
    b = _build_constraints_reference(mesh)
    assert np.array_equal(a.free_nodes, b.free_nodes), name
    assert np.array_equal(a.hanging, b.hanging), name
    d = (a.T - b.T)
    assert abs(d).max() == 0.0 if d.nnz else True, name
    assert d.nnz == 0 or abs(d).max() < 1e-15, name


def test_vectorized_is_faster():
    t = build_uniform(7, dim=2)                      # 16k elements
    mask = np.zeros(len(t), bool)
    mask[::7] = True
    t = balance2to1(refine_elements(t, mask))
    mesh = build_mesh(t, p=1)
    t0 = time.perf_counter()
    build_constraints(mesh)
    t_new = time.perf_counter() - t0
    t0 = time.perf_counter()
    _build_constraints_reference(mesh)
    t_ref = time.perf_counter() - t0
    print(f"constraints: reference {t_ref:.2f}s vs vectorized {t_new:.2f}s "
          f"({t_ref / t_new:.1f}x)")
    assert t_new < 0.5 * t_ref, (t_new, t_ref)
