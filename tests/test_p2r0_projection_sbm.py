"""P2-R0 projection+volumetric-SBM composition tests.

Task 2: the shifted-Nitsche vector Dirichlet block is threaded into the base
Leray projection stepper's PREDICTOR sub-solve (Step 1) without forking
`leray.py`. The immersed no-slip body is enforced WEAKLY (SBM), the box
inflow/walls STRONGLY (strong_mask + u_inf). This test asserts the SBM face
block measurably changes the predictor momentum block, that the surrogate face
set is non-empty, and that the geometry-only SBM block is assembled ONCE
(cached).
"""
import numpy as np
import pytest
import scipy.sparse as sp

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
from diffsim.steppers.leray_sbm import LeraySBMStepper

pytestmark = pytest.mark.tier5

R = 0.07
CTR = (0.3, 0.5)
U_IN = 1.0
NU = 2 * U_IN * R / 20.0                     # Re_diameter = 20


def _build_re20(device, level=5):
    """Re20 cylinder fixture mirroring tests/test_cylinder.py."""
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
    coords = mesh.node_coords[cons.free_nodes]
    on = lambda v, c: np.abs(coords[:, c] - v) < 1e-12
    strong = np.where(on(0.0, 0) | on(0.0, 1) | on(1.0, 1))[0]
    strong_mask = np.zeros(len(coords), dtype=bool)
    strong_mask[strong] = True
    u_inf = np.zeros((len(coords), dim))         # full free-node-major field
    inflow = strong[np.abs(coords[strong, 0]) < 1e-12]
    u_inf[inflow, 0] = U_IN                       # inflow x-velocity
    return oracle, dm, sf, geo, strong_mask, u_inf, mesh, cons


def test_predictor_sbm_block_composes(device):
    dim, ndof = 2, 3
    dt = 0.05
    oracle, dm, sf, geo, strong_mask, u_inf, mesh, cons = _build_re20(device)

    def f_fn(x, t):
        return np.zeros((len(x), dim))

    st = LeraySBMStepper(oracle, dm, NU, dt, f_fn,
                         u_inf=u_inf, strong_mask=strong_mask,
                         lam=0.5, domain="outside", order=1, picard_iters=1,
                         solver="splu", ppe_finescale=False)
    st.set_initial(lambda coords: np.zeros((len(coords), dim)))

    # (b) the surrogate face set is non-empty
    assert st.sf.elem.size > 0, "empty surrogate face set"

    # run ONE predictor-only step, capturing the assembled matrix
    A_with = st._predict(return_matrix=True)

    # (c) the SBM block is assembled ONCE (cached): call _predict again,
    # the constrained face block must be the SAME object (identity).
    af_ref = st.Af_c
    st._predict(return_matrix=False)
    assert st.Af_c is af_ref, "SBM face block re-assembled (not cached)"

    # (a) the assembled predictor matrix includes the SBM face entries:
    # build the BARE assemble_linear_ns predictor block on the SAME iterate
    # (no SBM face block, no strong rows overwrite) and compare the row-sum
    # at a surrogate-face node. They MUST differ by the SBM contribution.
    T = dm.constraints.T.tocsr()
    T_vec = sp.kron(T, sp.identity(ndof, format="csr"), format="csr")

    # reconstruct the bare block at the same (zero) advecting iterate
    from diffsim.solvers.timestepping import bdf_coeffs
    b0, b1, b2 = bdf_coeffs(1, dt)
    sigma = b0 / dt
    nfree = st.n_free
    a_node = np.zeros((nfree, dim))
    xq = gauss_points(mesh, dm.tables_by_p)

    def gp_field(node_vec):
        full = np.asarray(T @ node_vec)
        aq, dq = {}, {}
        for pv in dm.bins:
            tb = dm.tables_by_p[pv]
            conn = mesh.conn_of[pv]
            vals = full[conn]
            aq[pv] = np.einsum("qa,ead->eqd", tb.N, vals).reshape(-1, dim)
            h = mesh.tree.h()[mesh.bins[pv]]
            dq[pv] = (np.einsum("qad,ead->eq", tb.dN, vals)
                      * (2.0 / h)[:, None]).reshape(-1)
        return aq, dq

    aq, dq = gp_field(a_node)
    fq = {pv: np.zeros((aq[pv].shape[0], dim)) for pv in xq}
    # match the base stepper's timestab=True default: sig2tau=(2*sigma)**2
    A_bare, _ = assemble_linear_ns(dm, aq, dq, fq, NU, sigma=sigma,
                                   sig2tau=(2.0 * sigma) ** 2)

    # find a surrogate-face node (a retained-mesh node on a surrogate face)
    pv0 = sf.elem[0]
    face_node = int(mesh.conn_of[1][
        np.searchsorted(mesh.bins[1], sf.elem[0])][0])
    # map global -> free-node index (free_nodes is an array of node ids)
    free_of = np.full(dm.n_nodes, -1, dtype=np.int64)
    free_of[cons.free_nodes] = np.arange(len(cons.free_nodes))
    fn = free_of[face_node]
    assert fn >= 0, "surrogate face node not in the free set"

    row = fn * ndof + 0  # x-velocity dof of that node
    rs_with = np.abs(A_with.tocsr().getrow(row)).sum()
    rs_bare = np.abs(A_bare.tocsr().getrow(row)).sum()
    assert not np.isclose(rs_with, rs_bare), (
        f"SBM block did not change the momentum row: with={rs_with}, "
        f"bare={rs_bare}")

    # INDEPENDENT reference: the row difference (assembled-with minus bare)
    # must equal the hand-assembled SBM block's row EXACTLY, since this row
    # is a surrogate-face velocity dof that gets NO strong overwrite (weak
    # body). Rebuild Af_c independently from sbm_vector_dirichlet.
    Af_indep, _ = sbm_vector_dirichlet(
        dm, sf, geo, lambda y: np.zeros((len(y), dim)), NU, ndof)
    Af_c_indep = (T_vec.T @ Af_indep @ T_vec).tocsr()
    assert st.Af_c is st.Af_c  # (already checked cached identity above)
    delta = (A_with.tocsr().getrow(row) - A_bare.tocsr().getrow(row)).toarray()
    ref = Af_c_indep.getrow(row).toarray()
    # the surrogate row is weak (not overwritten), so delta == SBM row exactly
    assert np.allclose(delta, ref, atol=1e-10), (
        f"predictor row delta != independent SBM row; "
        f"max|delta-ref|={np.abs(delta - ref).max():.3e}")
    assert np.abs(ref).sum() > 0, "independent SBM row is all-zero"

    # MUTATION guard: the SBM block is load-bearing — zeroing it must make
    # the composed row collapse back to the bare row.
    st_zero_Af = st.Af_c.copy()
    st_zero_Af.data[:] = 0.0
    A_zeroed = st.base._predict(
        extra_block=(st_zero_Af, np.zeros_like(st.bf_c)),
        sbm_nodes=st._sbm_nodes, return_matrix=True)
    rs_zero = np.abs(A_zeroed.tocsr().getrow(row)).sum()
    assert np.isclose(rs_zero, rs_bare, atol=1e-8), (
        f"with SBM zeroed, row should match bare: zero={rs_zero}, "
        f"bare={rs_bare}")
