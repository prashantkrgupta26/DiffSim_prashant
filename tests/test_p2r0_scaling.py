"""P2-R0 Task 11 — scaling-pathway declaration gate.

Assertion: the SBM face block assembled by `sbm_vector_dirichlet` and
composed into the Leray-projection predictor introduces NO new nnz-pattern
class beyond the standard NS system assembled by `assemble_linear_ns` on the
same mesh and the same node set.

ARGUMENT: the SBM face block `Af_c` (constrained, free-node-major) scatters
ONLY into node-pairs ``(i, j)`` where both nodes share a common surrogate face
element — these are pre-existing NS element-connectivity pairs already present
in the `assemble_linear_ns` COO accumulation.  The SBM block adds new
*numeric values* (the Nitsche consistency/penalty contributions) but no new
*structural entries* (no node pair outside the NS sparsity graph).  The
constraint projection ``T_vec.T @ Af @ T_vec`` can only REDUCE the sparsity
(hanging-node masters absorb slave DOFs); it cannot add entries outside
``T_vec.T @ K_ns @ T_vec``.

This test builds both matrices on a small 2-D fixture (no GPU required for the
nnz structural check), then asserts:

1. ``nnz(Af_c)  <=  nnz(A_ns)``  — the face block is a structural sub-matrix.
2. ``nonzero pattern of Af_c`` is a strict subset of ``nonzero pattern of A_ns``
   (verified via the union sparsity graph: ``nnz(A_ns + |Af_c|) == nnz(A_ns)``).
3. The SBM block is non-trivially non-zero (non-vacuity guard: the test is not
   trivially satisfied by an all-zero face block).

SCALING-PATHWAY IMPLICATION (see the § P2-R0 scaling-pathway declaration in
the spec): no new nnz-space arrays are required beyond the standard NS system,
confirming the ChunkedCSR (#38) + fp32-IR (#36) 100M-DOF budget is not
inflated by the SBM composition.
"""
import os
import sys
import numpy as np
import pytest
import scipy.sparse as sp

sys.path.insert(0, os.path.dirname(__file__))

from diffsim.octree.build import build_uniform
from diffsim.mesh.nodes import build_mesh
from diffsim.mesh.constraints import build_constraints
from diffsim.mesh.basis import basis_tables
from diffsim.mesh.faces import face_tables
from diffsim.assembly.operators import DeviceMesh
from diffsim.geometry.csg import Sphere
from diffsim.sbm.surrogate import (classify_lambda, extract_surrogate,
                                    GeometryData)
from diffsim.sbm.vector import sbm_vector_dirichlet
from diffsim.api.ns_bricks import assemble_linear_ns
from diffsim.physics.poisson import gauss_points
from diffsim.solvers.timestepping import bdf_coeffs

pytestmark = pytest.mark.tier5

# ── small Re20-cylinder fixture (matches test_p2r0_projection_sbm.py) ──────
R = 0.07
CTR = (0.3, 0.5)
U_IN = 1.0
NU = 2 * U_IN * R / 20.0   # Re_diameter = 20


def _build_fixture(device, level=5):
    """2-D cylinder fixture at level 5 — the same mesh used across P2-R0 tests."""
    ndof, dim = 3, 2
    oracle = Sphere(CTR, R)
    tree = build_uniform(level, dim=2)
    ret, _ = classify_lambda(tree, oracle, 0.5, domain="outside")
    sf = extract_surrogate(ret)
    mesh = build_mesh(ret, p=1)
    cons = build_constraints(mesh)
    dm = DeviceMesh.from_mesh(mesh, cons, basis_tables(1, dim=2), device)
    geo = GeometryData.evaluate(oracle, ret, sf, face_tables(1, 2),
                                domain="outside")
    return dm, sf, geo, mesh, cons, ndof, dim


# ── test ───────────────────────────────────────────────────────────────────

def test_no_new_nnz_space(device):
    """SBM face block introduces no new nnz-pattern entries outside the
    standard NS system sparsity graph.

    This is the mandatory Task-11 scaling gate: the SBM composition is
    pattern-conservative — ChunkedCSR (#38) / fp32-IR (#36) 100M-DOF budgets
    are not inflated.
    """
    dm, sf, geo, mesh, cons, ndof, dim = _build_fixture(device)

    # ── 1. assemble the standard NS volume matrix on a zero advecting field ──
    T = dm.constraints.T.tocsr()
    a_node = np.zeros((dm.n_nodes, dim))
    xq = gauss_points(mesh, dm.tables_by_p)
    aq, dq = {}, {}
    for pv in dm.bins:
        tb = dm.tables_by_p[pv]
        conn = mesh.conn_of[pv]
        vals = a_node[conn]                              # [ne, nbf, dim]
        h = mesh.tree.h()[mesh.bins[pv]]
        aq[pv] = np.einsum("qa,ead->eqd", tb.N, vals).reshape(-1, dim)
        dq[pv] = (np.einsum("qad,ead->eq", tb.dN, vals)
                  * (2.0 / h)[:, None]).reshape(-1)
    dt = 0.05
    b0, _, _ = bdf_coeffs(1, dt)
    sigma = b0 / dt
    fq = {pv: np.zeros((aq[pv].shape[0], dim)) for pv in xq}
    A_ns, _ = assemble_linear_ns(dm, aq, dq, fq, NU, sigma=sigma,
                                 sig2tau=(2.0 * sigma) ** 2)

    # ── 2. assemble the SBM face block (constrained) ─────────────────────────
    T_vec = sp.kron(T, sp.identity(ndof, format="csr"), format="csr")
    g_body = lambda y: np.zeros((len(y), dim))
    Af_full, _ = sbm_vector_dirichlet(
        dm, sf, geo, g_body, NU, ndof, alpha=10.0, a_face=None)
    Af_c = (T_vec.T @ Af_full @ T_vec).tocsr()

    # ── 3. non-vacuity: the face block must have non-zero entries ─────────────
    assert Af_c.nnz > 0, (
        "SBM face block Af_c is all-zero — non-vacuity check failed; "
        "the surrogate face set may be empty or the assembly returned zeros.")

    # ── 4. nnz count: face block  ≤  NS block ────────────────────────────────
    assert Af_c.nnz <= A_ns.nnz, (
        f"SBM face block has MORE stored entries than the NS system: "
        f"Af_c.nnz={Af_c.nnz}, A_ns.nnz={A_ns.nnz}. "
        "This would violate the no-new-nnz-space scaling invariant.")

    # ── 5. pattern containment: every non-zero (row, col) of Af_c already
    #       exists in A_ns.  Union sparsity must equal A_ns sparsity. ─────────
    # Use absolute values to catch numerical cancellations producing stored
    # zeros in Af_c (sparse format keeps them; the pattern is what matters).
    A_ns_abs = A_ns.copy()
    A_ns_abs.data[:] = 1.0
    Af_c_abs = Af_c.copy()
    Af_c_abs.data[:] = 1.0
    union = (A_ns_abs + Af_c_abs).tocsr()
    union.eliminate_zeros()
    assert union.nnz == A_ns.nnz, (
        f"SBM face block introduces NEW sparsity entries: "
        f"nnz(A_ns)={A_ns.nnz}, nnz(union)={union.nnz}, "
        f"new entries={union.nnz - A_ns.nnz}. "
        "The SBM face block must scatter ONLY into existing NS node-pair slots.")
