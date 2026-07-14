"""A1 (C3 Phase-3 dynamic AMR): the conservative remesh/transfer core.

Host-side only (no GPU): octree coarsen primitive, common refinement, and the
L2-projection field/BDF-history transfer that conserves INT c dV to solver
tolerance under both refinement and coarsening, converges at 2nd order against
an analytic field, and round-trips losslessly on a representable field.
"""
import numpy as np
import pytest

from diffsim.octree.build import (build_uniform, refine_elements,
                                  coarsen_elements)
from diffsim.octree.balance import balance2to1
from diffsim.mesh.nodes import build_mesh
from diffsim.mesh.constraints import build_constraints
from diffsim.mesh.basis import basis_tables
from diffsim.adaptivity.remesh import (field_mass, conservative_transfer,
                                       consistent_mass_matrix,
                                       common_refinement)

TB = basis_tables(1, dim=2)


def _mk(tree):
    m = build_mesh(tree, p=1)
    return m, build_constraints(m)


def _analytic(coords):
    x, y = coords[:, 0], coords[:, 1]
    return 0.3 * np.sin(2 * np.pi * x) * np.sin(2 * np.pi * y) + 0.5


def _band(tree, r=0.3, band=1.5):
    rr = np.sqrt(((tree.centers() - 0.5) ** 2).sum(1))
    return np.abs(rr - r) < tree.h() * band


def _leafset(t):
    return set(zip(t.keys.tolist(), t.levels.tolist()))


def test_coarsen_inverts_refine():
    base = build_uniform(4, dim=2)
    ref = balance2to1(refine_elements(base, _band(base)))
    assert len(ref) > len(base)
    back = ref
    for _ in range(6):
        if not (back.levels > 4).any():
            break
        back = coarsen_elements(back, back.levels > 4)
    assert _leafset(back) == _leafset(base)


def test_coarsen_only_complete_sibling_groups():
    # marking only 3 of 4 siblings must NOT coarsen that group
    base = build_uniform(2, dim=2)
    ref = refine_elements(base, np.array([True] + [False] * (len(base) - 1)))
    fine = ref.levels == 3
    idx = np.where(fine)[0]
    mask = np.zeros(len(ref), bool)
    mask[idx[:3]] = True                       # only 3 of the 4 children
    out = coarsen_elements(ref, mask)
    assert _leafset(out) == _leafset(ref)      # unchanged


def test_common_refinement_is_finest():
    o = balance2to1(refine_elements(build_uniform(3, dim=2),
                                    _band(build_uniform(3, dim=2))))
    n = build_uniform(3, dim=2)
    cr = common_refinement(o, n)
    assert len(cr) >= max(len(o), len(n))
    # every common cell sits inside a single leaf of each parent tree
    from diffsim.octree.lookup import LeafLookup
    from diffsim.octree import morton
    for parent in (o, n):
        lk = LeafLookup(parent)
        idx = lk.find(cr.anchors())
        assert (idx >= 0).all()
        assert (parent.levels[idx] <= cr.levels).all()


@pytest.mark.parametrize("direction", ["refine", "coarsen"])
def test_mass_conserved_to_solver_tol(direction):
    coarse = build_uniform(4, dim=2)
    fine = balance2to1(refine_elements(coarse, _band(coarse)))
    if direction == "refine":
        ot, nt = coarse, fine
    else:
        ot, nt = fine, coarse
    om, oc = _mk(ot)
    nm, nc = _mk(nt)
    cf = _analytic(om.node_coords[oc.free_nodes])
    m0 = field_mass(om, oc, TB, cf)
    (cn,) = conservative_transfer(om, oc, TB, nm, nc, TB, [cf])
    m1 = field_mass(nm, nc, TB, cn)
    assert abs(m1 - m0) < 1e-12 * max(abs(m0), 1.0)


def test_bdf_history_all_levels_conserved():
    coarse = build_uniform(4, dim=2)
    fine = balance2to1(refine_elements(coarse, _band(coarse)))
    om, oc = _mk(fine)
    nm, nc = _mk(coarse)
    base = _analytic(om.node_coords[oc.free_nodes])
    fields = [base, 1.7 * base - 0.2]          # c^n, c^{n-1}
    outs = conservative_transfer(om, oc, TB, nm, nc, TB, fields)
    for fin, fout in zip(fields, outs):
        m0 = field_mass(om, oc, TB, fin)
        m1 = field_mass(nm, nc, TB, fout)
        assert abs(m1 - m0) < 1e-12 * max(abs(m0), 1.0)


def test_transfer_second_order_convergence():
    errs = []
    for lvl in (3, 4, 5):
        om, oc = _mk(build_uniform(lvl + 1, dim=2))
        nm, nc = _mk(build_uniform(lvl, dim=2))
        cf = _analytic(om.node_coords[oc.free_nodes])
        (cn,) = conservative_transfer(om, oc, TB, nm, nc, TB, [cf])
        exact = _analytic(nm.node_coords[nc.free_nodes])
        M = consistent_mass_matrix(nm, TB)
        e = np.asarray(nc.T @ (cn - exact))
        errs.append(float(np.sqrt(e @ (M @ e))))
    orders = [np.log2(errs[i] / errs[i + 1]) for i in range(len(errs) - 1)]
    assert min(orders) > 1.7            # 2nd-order projection, >2x over O(h)


def test_round_trip_lossless_on_representable_field():
    base = build_uniform(4, dim=2)
    fine = balance2to1(refine_elements(base, _band(base)))
    bm, bc = _mk(base)
    fm, fc = _mk(fine)
    c0 = _analytic(bm.node_coords[bc.free_nodes])
    (cf,) = conservative_transfer(bm, bc, TB, fm, fc, TB, [c0])   # refine
    (cb,) = conservative_transfer(fm, fc, TB, bm, bc, TB, [cf])   # coarsen back
    M = consistent_mass_matrix(bm, TB)
    e = np.asarray(bc.T @ (cb - c0))
    assert float(np.sqrt(e @ (M @ e))) < 1e-12
