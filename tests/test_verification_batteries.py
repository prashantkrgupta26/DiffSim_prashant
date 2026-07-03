"""Spec S13.3 batteries: randomized fuzz over p1-only, p2-only, and mixed
meshes asserting the three machine-precision invariants: partition of unity,
polynomial reproduction (linear always; quadratic on p2-only), and trace
conformity across every shared face.

Import mechanism: tests/__init__.py + tests/helpers/__init__.py make the
helpers a proper sub-package.  pyproject.toml pythonpath=["."] ensures the
project root is on sys.path, so 'from tests.helpers...' resolves cleanly
without touching conftest.py or sys.path at runtime.
"""
import json
import pathlib
import numpy as np
import pytest
from diffsim.octree.build import build_uniform, refine_elements
from diffsim.octree.balance import balance2to1
from diffsim.mesh.nodes import build_mesh
from diffsim.mesh.constraints import build_constraints
from diffsim.mesh.basis import basis_tables
from diffsim.assembly.operators import DeviceMesh
from diffsim.assembly.dirichlet import DirichletPoisson
from diffsim.physics.poisson import l2_error
from tests.helpers.field_eval import trace_conformity_max_jump

BASE = pathlib.Path(__file__).parent / "baselines" / "m05_baselines.json"

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
    assert np.allclose(u, lin(m.node_coords), atol=1e-12)
    assert trace_conformity_max_jump(m, u) < 1e-12

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


def test_chain_closure_deterministic():
    """Deterministic regression for the p→h→free transitive chain in mixed meshes.

    Mesh: build_uniform(1, dim=3) with element 0 (Morton-sorted anchor [0,0,0])
    refined once → 8 level-2 children + 7 level-1 elements; balance2to1 is a
    no-op since the maximum face-adjacent level difference is 2−1 = 1.

    Element 0 (level 2, anchor [0,0,0]) is the unique level-2 element whose
    ALL face-neighbors are also level-2, so it can be promoted to p2 without
    violating the one-knob rule.  Its p1 face-neighbors (elems 1/2/4) each
    border a level-1 element on their far face, so some of their corners are
    h-hanging.  The p-hanging midpoints on elem 0's p2 faces are owned by
    those p1 elements; the raw single-level interpolation rows for those
    p-hanging nodes therefore target h-hanging nodes — the p→h chain.

    Structural assertion: recompute raw single-level ownership rows (same probe
    logic as constraints.py, without chain resolution) and assert ≥1 raw row
    has a target that is itself hanging.  If this assertion fails, the mesh
    topology has changed and the test no longer exercises transitive closure.
    """
    from diffsim.octree.lookup import LeafLookup, face_neighbors as _face_nbrs
    from diffsim.mesh.nodes import _local_offsets
    from diffsim.octree import morton
    from itertools import product as iproduct

    # --- Build mesh ---
    t = build_uniform(1, dim=3)
    mask = np.zeros(len(t), bool)
    mask[0] = True   # refine element at Morton-sorted index 0 (anchor [0,0,0])
    t = balance2to1(refine_elements(t, mask))
    lev = t.levels.astype(int)
    assert len(t) == 15 and np.count_nonzero(lev == 2) == 8, (
        f"unexpected tree shape: {len(t)} elems, {np.count_nonzero(lev==2)} at level 2"
    )

    # Identify valid p2 candidates: level-2 elements with ALL face-neighbors
    # at the same level (one-knob compatible with p2 promotion).
    nbrs = _face_nbrs(t)
    same_level_face = np.ones(len(t), bool)
    for nb in nbrs:
        ok = nb >= 0
        same_level_face[ok] &= (lev[ok] == lev[nb[ok]])
    cand = np.where(same_level_face & (lev == 2))[0]
    assert len(cand) == 1 and cand[0] == 0, (
        f"expected exactly elem 0 as p2 candidate, got {cand}"
    )

    p = np.ones(len(t), np.int8)
    p[0] = 2
    m = build_mesh(t, p)
    c = build_constraints(m)

    # --- Structural chain-presence assertion ---
    # Recompute raw single-level ownership rows (owner element + Lagrange
    # weights) WITHOUT chain resolution.  Assert ≥1 raw target is hanging.
    # If this fails, the mesh no longer contains the chain and the test is
    # no longer a valid regression for transitive closure.
    lk = LeafLookup(t)
    levels64 = t.levels.astype(np.int64)
    dps = np.array(list(iproduct((-1, 0), repeat=3)), np.int64)
    node_elems: list = [[] for _ in range(len(m.node_coords))]
    conn_row: dict = {}
    for pv, eids in m.bins.items():
        cn = m.conn_of[pv]
        for r, e in enumerate(eids):
            conn_row[int(e)] = (pv, r)
            for a in cn[r]:
                node_elems[int(a)].append(int(e))
    a2 = t.anchors() * 2
    s2 = 2 * (1 << (morton.lmax(3) - levels64))
    ic2n = {tuple(ic.tolist()): i for i, ic in enumerate(m.node_icoords)}

    chain_found = False
    for n in np.where(c.hanging)[0]:
        ic = m.node_icoords[n]
        touch: set = set()
        for dp in dps:
            idx = lk.find(((ic + dp) // 2)[None, :])[0]
            if idx >= 0:
                touch.add(int(idx))
        non_c = touch - set(node_elems[n])
        if not non_c:
            continue
        oe = min(non_c, key=lambda e: (levels64[e], int(m.p_elem[e])))
        pv_o, _ = conn_row[oe]
        step = s2[oe] // pv_o
        for off in _local_offsets(pv_o, 3):
            corner_n = ic2n.get(tuple((a2[oe] + off * step).tolist()))
            if corner_n is not None and c.hanging[corner_n]:
                chain_found = True
                break
        if chain_found:
            break

    assert chain_found, (
        "no p→h chain in raw rows: mesh topology changed and no longer exercises "
        "transitive closure — update the mesh construction in this test"
    )

    # (a) build_constraints returned without error (reached this line)
    # (b) Partition of unity — row sums == 1 proves all T rows resolve to free nodes
    assert np.allclose(np.asarray(c.T.sum(axis=1)).ravel(), 1.0, atol=1e-12), (
        "partition of unity failed after chain resolution"
    )
    # (c) Linear reproduction exact to 1e-12
    rng = np.random.default_rng(42)
    coef = rng.standard_normal(3)
    lin = lambda x: 3.0 + x @ coef
    u = c.T @ lin(m.node_coords[c.free_nodes])
    assert np.allclose(u, lin(m.node_coords), atol=1e-12), (
        "linear reproduction failed after chain resolution"
    )


# ---------------------------------------------------------------------------
# Task 13: p2 quadratic patch + adaptive order-3 MMS + M0.5 baselines
# ---------------------------------------------------------------------------

@pytest.mark.tier3
def test_p2_quadratic_patch_adaptive(device):
    """-lap(u) = f with quadratic u must be exact for p2 through h-hanging faces."""
    t = build_uniform(2, dim=3)
    mask = np.zeros(len(t), bool)
    mask[[0, 9]] = True
    t = balance2to1(refine_elements(t, mask))
    m = build_mesh(t, 2)
    c = build_constraints(m)
    dm = DeviceMesh.from_mesh(m, c, basis_tables(2, dim=3), device)
    # U = x0^2 + x1^2 + x2^2  =>  lap(U) = 6  =>  -lap(U) = -6  =>  f = -6
    U = lambda x: x[:, 0]**2 + x[:, 1]**2 + x[:, 2]**2
    F = lambda x: np.full(len(x), -6.0)
    u = DirichletPoisson(dm).solve(g_fn=U, f_fn=F, tol=1e-13)
    assert l2_error(dm, u, U) < 1e-10


@pytest.mark.tier3
def test_p2_adaptive_mms_order3(device):
    """p2 on adaptive meshes (hanging faces) must achieve L2 order ≈ 3."""
    U_fn = lambda x: np.prod(np.sin(np.pi * x), axis=1)
    F_fn = lambda x: 3 * np.pi**2 * U_fn(x)
    errs = []
    for lvl in [1, 2, 3]:
        t = build_uniform(lvl, dim=3)
        mask = np.zeros(len(t), bool)
        mask[0] = True
        t = balance2to1(refine_elements(t, mask))
        m = build_mesh(t, 2)
        c = build_constraints(m)
        dm = DeviceMesh.from_mesh(m, c, basis_tables(2, dim=3), device)
        u = DirichletPoisson(dm).solve(g_fn=U_fn, f_fn=F_fn, tol=1e-13)
        errs.append(l2_error(dm, u, U_fn))
    assert abs(np.log2(errs[-2] / errs[-1]) - 3.0) < 0.25, errs
    # regression lock: create baseline on first run, compare thereafter
    results = {"p2_adaptive_mms": errs}
    if BASE.exists():
        ref = json.loads(BASE.read_text())
        assert np.allclose(errs, ref["p2_adaptive_mms"], rtol=1e-6), (errs, ref)
    else:
        BASE.parent.mkdir(parents=True, exist_ok=True)
        BASE.write_text(json.dumps(results, indent=2) + "\n")
        pytest.skip("baseline created; re-run to compare")
