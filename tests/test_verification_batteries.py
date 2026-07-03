"""Spec S13.3 batteries: randomized fuzz over p1-only, p2-only, and mixed
meshes asserting the three machine-precision invariants: partition of unity,
polynomial reproduction (linear always; quadratic on p2-only), and trace
conformity across every shared face.

Import mechanism: tests/__init__.py + tests/helpers/__init__.py make the
helpers a proper sub-package.  pyproject.toml pythonpath=["."] ensures the
project root is on sys.path, so 'from tests.helpers...' resolves cleanly
without touching conftest.py or sys.path at runtime.
"""
import numpy as np
import pytest
from diffsim.octree.build import build_uniform, refine_elements
from diffsim.octree.balance import balance2to1
from diffsim.mesh.nodes import build_mesh
from diffsim.mesh.constraints import build_constraints
from tests.helpers.field_eval import trace_conformity_max_jump

pytestmark = pytest.mark.tier2


def _random_tree(rng, dim=3, base=1, rounds=3, max_level=4):
    t = build_uniform(base, dim=dim)
    for _ in range(rounds):
        mask = (rng.random(len(t)) < 0.25) & (t.levels < max_level)
        if mask.any():
            t = refine_elements(t, mask)
    return balance2to1(t)


def _checks(m, c, rng, reproduce_quadratic):
    dim = m.dim
    # Partition of unity: sum of each row of T must be 1
    assert np.allclose(np.asarray(c.T.sum(axis=1)).ravel(), 1.0, atol=1e-12)

    # Linear reproduction: c.T @ lin(free_coords) == lin(all_coords)
    coef = rng.standard_normal(dim)
    lin = lambda x: 3.0 + x @ coef
    u = c.T @ lin(m.node_coords[c.free_nodes])
    assert np.allclose(u, lin(m.node_coords), atol=1e-11)
    assert trace_conformity_max_jump(m, u) < 1e-11

    if reproduce_quadratic:
        # Quadratic reproduction: valid on p2-only meshes (owner trace is quadratic)
        A = rng.standard_normal((dim, dim)); A = A + A.T
        quad = lambda x: 1.0 + np.einsum("ni,ij,nj->n", x, A, x) + x @ coef
        uq = c.T @ quad(m.node_coords[c.free_nodes])
        assert np.allclose(uq, quad(m.node_coords), atol=1e-10)
        assert trace_conformity_max_jump(m, uq) < 1e-10

    # Random constrained field: conformity must hold for ANY free vector
    ur = c.T @ rng.standard_normal(len(c.free_nodes))
    assert trace_conformity_max_jump(m, ur) < 1e-11


@pytest.mark.parametrize("trial", range(5))
def test_battery_p1_only(trial):
    rng = np.random.default_rng(100 + trial)
    t = _random_tree(rng)
    m = build_mesh(t, 1)
    _checks(m, build_constraints(m), rng, reproduce_quadratic=False)


@pytest.mark.parametrize("trial", range(5))
def test_battery_p2_only(trial):
    rng = np.random.default_rng(200 + trial)
    t = _random_tree(rng)
    m = build_mesh(t, 2)
    _checks(m, build_constraints(m), rng, reproduce_quadratic=True)


@pytest.mark.parametrize("trial", range(5))
def test_battery_mixed(trial):
    rng = np.random.default_rng(300 + trial)
    t = _random_tree(rng)
    from diffsim.octree.lookup import face_neighbors
    lev = t.levels.astype(int)
    nbrs = face_neighbors(t)

    # Select elements whose ALL face neighbors are at the same level
    # (one-knob: only the p knob varies on those faces)
    same_level_ok = np.ones(len(t), bool)
    for nb in nbrs:
        ok = nb >= 0
        same_level_ok[ok] &= (lev[ok] == lev[nb[ok]])

    p = np.ones(len(t), np.int8)
    cand = np.where(same_level_ok)[0]
    # Promote a random subset of same-level candidates to p2.
    # Note: a p2 element's p1 same-level face-neighbor can have h-hanging
    # corner nodes (from diagonal coarser neighbours), creating a
    # p-hanging → h-hanging chain.  build_constraints handles these chains
    # via transitive closure, so no filtering is needed here.
    p[rng.choice(cand, size=max(1, len(cand) // 4), replace=False)] = 2

    # Demote any promoted element that has a level-differing face neighbor
    # (one-knob rule: level and p may not both differ across a face)
    for e in np.where(p == 2)[0]:
        for nb in nbrs:
            j = nb[e]
            if j >= 0 and lev[j] != lev[e]:
                p[e] = 1
                break

    if not (p == 2).any():
        pytest.skip("fuzz draw produced no valid p2 candidates")

    m = build_mesh(t, p)
    _checks(m, build_constraints(m), rng, reproduce_quadratic=False)
