"""Solution-adaptive remesh core: conservative field + BDF-history transfer
between two octree meshes (the missing half of dynamic AMR, C3 Phase-3).

The transfer must move a Cahn--Hilliard state (the conserved field ``c``, the
chemical potential ``mu``, and the BDF history levels ``c^n, c^{n-1}``) from an
old mesh to a new adapted mesh WITHOUT changing what the conserved field
represents: ``INT c dV`` must be preserved to solver tolerance regardless of
whether a region was refined or coarsened.

Two transfers are provided:

* :func:`nodal_transfer` -- inject the old field's nodal interpolant at the new
  nodes (``point_eval`` on the old mesh). EXACT under pure refinement (a P1
  interpolant of a P1 field on a nested finer mesh is the same function), but
  NOT conservative under coarsening: the coarse interpolant integrates to a
  different value. This is the failure mode C3 warns about.

* :func:`conservative_transfer` -- the L2 (Galerkin) projection with a
  consistent mass matrix, integrated on the COMMON REFINEMENT of the two meshes
  so both the new basis functions and the old field are polynomials on every
  quadrature cell. Because the FE space contains the constant and the basis is
  a partition of unity, this preserves ``INT c dV`` EXACTLY (to the linear
  solve tolerance): with ``A = T_new^T M_new T_new`` and
  ``rhs = T_new^T INT N_new (c_old) dV``,

      INT c_new dV = 1_free^T A c_new_free = 1_free^T rhs = INT c_old dV.

Everything is host-side (numpy/scipy); this is correctness work, not a hot loop.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu

from ..octree import morton
from ..octree.build import Octree
from ..octree.lookup import LeafLookup
from ..mesh.basis import gauss_1d
from ..mesh.pointeval import point_eval_weights
from .transfer import transfer_operator


def _as_tables_dict(mesh, tables_by_p):
    if not isinstance(tables_by_p, dict):
        return {mesh.p: tables_by_p}
    return tables_by_p


# --------------------------------------------------------------------------
# quadrature mass of a free-space field (matches assembly quadrature)
# --------------------------------------------------------------------------

def field_mass(mesh, cons, tables_by_p, c_free):
    """``INT c dV`` for a free-space nodal field ``c_free`` by Gauss-point
    quadrature -- the same integral the assembly computes (spec P3)."""
    tbp = _as_tables_dict(mesh, tables_by_p)
    full = np.asarray(cons.T @ np.asarray(c_free))
    dim = mesh.dim
    h_all = mesh.tree.h()
    mass = 0.0
    for pv, eids in mesh.bins.items():
        tb = tbp[pv]
        conn = mesh.conn_of[pv]
        vals = full[conn]                                  # [ne, nbf]
        c_gp = np.einsum("qa,ea->eq", tb.N, vals)          # [ne, nqp]
        h = h_all[eids]
        jac = (h / 2.0) ** dim
        wq = tb.w[None, :] * jac[:, None]                  # [ne, nqp]
        mass += float((c_gp * wq).sum())
    return mass


def consistent_mass_matrix(mesh, tables_by_p):
    """Global consistent mass matrix ``M_ab = INT N_a N_b dV`` (all nodes)."""
    tbp = _as_tables_dict(mesh, tables_by_p)
    dim = mesh.dim
    Nn = len(mesh.node_coords)
    h_all = mesh.tree.h()
    rows, cols, vals = [], [], []
    for pv, eids in mesh.bins.items():
        tb = tbp[pv]
        conn = mesh.conn_of[pv].astype(np.int64)
        ne, nbf = conn.shape
        Mref = np.einsum("qa,qb,q->ab", tb.N, tb.N, tb.w)  # [nbf, nbf]
        jac = (h_all[eids] / 2.0) ** dim                   # [ne]
        Me = jac[:, None, None] * Mref[None]               # [ne, nbf, nbf]
        r = np.repeat(conn, nbf, axis=1).ravel()
        c = np.tile(conn, (1, nbf)).ravel()
        rows.append(r); cols.append(c); vals.append(Me.ravel())
    return sp.csr_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(Nn, Nn))


# --------------------------------------------------------------------------
# common refinement of two octrees (finer cell wins everywhere)
# --------------------------------------------------------------------------

def common_refinement(old_tree: Octree, new_tree: Octree) -> Octree:
    """Octree whose leaves are, per region, the finer of ``old_tree`` and
    ``new_tree``. On every leaf of this tree BOTH a P1 field on ``old_tree``
    and a P1 basis function on ``new_tree`` restrict to a single element, hence
    are linear -- so a 2-point Gauss rule integrates their product exactly."""
    assert old_tree.dim == new_tree.dim
    dim = old_tree.dim
    lk_old = LeafLookup(old_tree)
    lk_new = LeafLookup(new_tree)
    L = morton.lmax(dim)

    def _keep(tree, lk_other, strict):
        # keep leaf if the OTHER tree is not strictly finer at its centre;
        # `strict` drops equal-level duplicates from the second side.
        if len(tree) == 0:
            return tree.keys, tree.levels
        anchors = tree.anchors()
        size = (1 << (L - tree.levels.astype(np.int64)))[:, None]
        centre = anchors + size // 2
        other = lk_other.find(centre)                      # leaf idx or -1
        other_all_lev = lk_other.tree.levels.astype(np.int64)
        other_lev = np.where(other >= 0,
                             other_all_lev[np.clip(other, 0, None)], -1)
        my_lev = tree.levels.astype(np.int64)
        if strict:
            keep = other_lev < my_lev
        else:
            keep = other_lev <= my_lev
        return tree.keys[keep], tree.levels[keep]

    ko, lo = _keep(old_tree, lk_new, strict=False)
    kn, ln = _keep(new_tree, lk_old, strict=True)
    keys = np.concatenate([ko, kn])
    levels = np.concatenate([lo, ln])
    from ..octree.build import _make
    return _make(keys, levels, dim, old_tree.periodic)


def _common_quadrature(common: Octree):
    """2-point-per-axis Gauss rule on every leaf of ``common``. Returns
    ``(xq [Nq, dim], wq [Nq])`` with ``wq`` carrying the physical Jacobian."""
    dim = common.dim
    from itertools import product as iproduct
    pts, wts = gauss_1d(1)                                  # 2-pt
    nq1 = len(pts)
    qidx = np.array(list(iproduct(*[range(nq1)] * dim)), np.int64)[:, ::-1]
    ref = pts[qidx]                                         # [nqp, dim]
    wref = np.prod(wts[qidx], axis=1)                       # [nqp]
    lo = common.anchors() * 2.0 ** (-morton.lmax(dim))      # [ne, dim]
    h = common.h()                                          # [ne]
    xq = lo[:, None, :] + (ref[None, :, :] + 1.0) * 0.5 * h[:, None, None]
    jac = (h / 2.0) ** dim                                  # [ne]
    wq = wref[None, :] * jac[:, None]                       # [ne, nqp]
    return xq.reshape(-1, dim), wq.reshape(-1)


# --------------------------------------------------------------------------
# the two transfers
# --------------------------------------------------------------------------

def nodal_transfer(mesh_old, cons_old, mesh_new, cons_new, fields):
    """Non-conservative reference: inject the old nodal interpolant at the new
    free nodes. Exact under refinement, LOSES MASS under coarsening."""
    P = transfer_operator(mesh_old, cons_old, mesh_new)     # [Nn_new, n_old_free]
    free = cons_new.free_nodes
    out = []
    for f in fields:
        new_all = np.asarray(P @ np.asarray(f))
        out.append(new_all[free])
    return out


def conservative_transfer(mesh_old, cons_old, tables_old,
                          mesh_new, cons_new, tables_new, fields):
    """L2/Galerkin projection with consistent mass, integrated on the common
    refinement. Conserves ``INT f dV`` exactly (to solver tol) for every field.

    ``fields`` is a sequence of free-space arrays on the OLD mesh; returns the
    corresponding free-space arrays on the NEW mesh."""
    tbp_old = _as_tables_dict(mesh_old, tables_old)
    common = common_refinement(mesh_old.tree, mesh_new.tree)
    xq, wq = _common_quadrature(common)

    T_old = cons_old.T.tocsr()
    T_new = cons_new.T.tocsr()
    W_old = point_eval_weights(mesh_old, xq)                # [Nq, Nn_old]
    W_new = point_eval_weights(mesh_new, xq)                # [Nq, Nn_new]
    Eo = (W_old @ T_old)                                    # [Nq, n_old_free]

    F = np.column_stack([np.asarray(f) for f in fields])    # [n_old_free, k]
    fq = Eo @ F                                             # old field at qp [Nq, k]
    # rhs_full[a, :] = INT N_a^new (c_old) dV  = W_new^T diag(wq) fq
    rhs_full = W_new.T @ (wq[:, None] * fq)                 # [Nn_new, k]
    rhs = np.asarray(T_new.T @ rhs_full)                    # [n_new_free, k]

    M_new = consistent_mass_matrix(mesh_new, tables_new)
    A = (T_new.T @ M_new @ T_new).tocsc()
    lu = splu(A)
    X = lu.solve(rhs)                                       # [n_new_free, k]
    return [X[:, j] for j in range(X.shape[1])]
