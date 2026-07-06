"""M1c flagship gate: d(drag)/d(cylinder-center) adjoint vs FD at Re=20
(steady, confined, level 5 — the test_cylinder smoke config)."""
import numpy as np
import pytest
import scipy.sparse as sp
from scipy.sparse.linalg import splu

from diffsim.octree.build import build_uniform
from diffsim.geometry.csg import Sphere
from diffsim.sbm.surrogate import classify_lambda, extract_surrogate, \
    GeometryData
from diffsim.mesh.nodes import build_mesh
from diffsim.mesh.constraints import build_constraints
from diffsim.mesh.basis import basis_tables
from diffsim.mesh.faces import face_tables
from diffsim.assembly.operators import DeviceMesh
from diffsim.api.ns_bricks import assemble_linear_ns
from diffsim.physics.poisson import gauss_points
from diffsim.sbm.vector import sbm_vector_dirichlet, surrogate_traction
from diffsim.sbm.ns_shape import drag_shape_gradient

pytestmark = [pytest.mark.tier5, pytest.mark.ad]

R = 0.07
U_IN = 1.0
NU = 2 * U_IN * R / 20.0
ALPHA = 10.0


def _steady(center, device, level=5):
    ndof, dim = 3, 2
    dt = 0.05
    oracle = Sphere(tuple(center), R)
    tree = build_uniform(level, dim=2)
    ret, _ = classify_lambda(tree, oracle, 0.5, domain="outside")
    sf = extract_surrogate(ret)
    mesh = build_mesh(ret, p=1)
    cons = build_constraints(mesh)
    dm = DeviceMesh.from_mesh(mesh, cons, basis_tables(1, dim=2), device)
    geo = GeometryData.evaluate(oracle, ret, sf, face_tables(1, 2),
                                domain="outside")
    T = cons.T.tocsr()
    T_vec = sp.kron(T, sp.identity(ndof, format="csr"), format="csr")
    nfree = T.shape[1]
    coords = mesh.node_coords[cons.free_nodes]
    xq = gauss_points(mesh, dm.tables_by_p)
    on = lambda v, c: np.abs(coords[:, c] - v) < 1e-12
    strong = np.where(on(0.0, 0) | on(0.0, 1) | on(1.0, 1))[0]
    g_strong = np.zeros((len(strong), 2))
    g_strong[np.abs(coords[strong, 0]) < 1e-12, 0] = U_IN

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

    x = np.zeros(nfree * ndof)
    prev_u = None
    sigma = 0.0        # TRUE steady Picard: b then has NO aq-dependence,
    #                    so dR/daq is fully covered by the taped volume
    #                    kernel (adjoint-Picard exactness)
    pin = int(np.argmax(coords[:, 0] + coords[:, 1]))
    strong_rows = np.concatenate(
        [np.array([i * ndof + c for i in strong for c in range(dim)],
                  np.int64), np.array([pin * ndof + dim], np.int64)])
    for step in range(160):
        u_node = x.reshape(nfree, ndof)[:, :dim]
        aq, dq = gp_field(u_node)
        fq = {pv: np.zeros_like(aq[pv]) for pv in xq}
        A, b = assemble_linear_ns(dm, aq, dq, fq, NU, sigma=sigma)
        Af, bf = sbm_vector_dirichlet(
            dm, sf, geo, lambda y: np.zeros((len(y), 2)), NU, ndof,
            alpha=ALPHA)
        A = (A + T_vec.T @ Af @ T_vec).tolil()
        b = b + np.asarray(T_vec.T @ bf)
        for k, i in enumerate(strong):
            for c in range(dim):
                r = i * ndof + c
                A.rows[r] = [int(r)]
                A.data[r] = [1.0]
                b[r] = g_strong[k, c]
        rp = pin * ndof + dim
        A.rows[rp] = [rp]
        A.data[rp] = [1.0]
        b[rp] = 0.0
        A = A.tocsr()
        x = splu(A.tocsc()).solve(b)
        u_new = x.reshape(nfree, ndof)[:, :dim]
        if prev_u is not None and np.abs(u_new - prev_u).max() < 1e-10 \
                and step > 3:
            break
        prev_u = u_new.copy()
    x_full = np.asarray(T_vec @ x)
    F = surrogate_traction(dm, sf, geo, x_full, NU, ndof)
    return dict(dm=dm, sf=sf, geo=geo, oracle=oracle, A=A, x_full=x_full,
                strong_rows=strong_rows, F=F, ret=ret)


def test_drag_gradient_adjoint_vs_fd(device):
    c0 = np.array([0.3, 0.5])
    eps = 1e-6
    base = _steady(c0, device)
    # frozen-classification trust region (m1a rule): the FD stencil must
    # not flip the retained set
    for dc in (+eps, -eps):
        r2, _ = classify_lambda(build_uniform(5, dim=2),
                                Sphere((c0[0] + dc, c0[1]), R), 0.5,
                                domain="outside")
        assert np.array_equal(r2.keys, base["ret"].keys), "eps flips cells"
    F0 = drag_shape_gradient(base["dm"], base["sf"], base["geo"],
                             base["oracle"], base["A"], base["x_full"],
                             NU, ALPHA, base["strong_rows"], ndof=3,
                             direction=0, fixed_point=True)
    assert abs(F0 - base["F"][0]) < 1e-10 * abs(F0)   # QoI self-consistency
    g_adj = float(base["oracle"].params[0].grad[0])   # d(drag)/d(c_x)
    Fp = _steady(c0 + np.array([eps, 0.0]), device)["F"][0]
    Fm = _steady(c0 - np.array([eps, 0.0]), device)["F"][0]
    fd = (Fp - Fm) / (2 * eps)
    print(f"drag gradient: adjoint = {g_adj:.6e}  FD = {fd:.6e}  "
          f"rel = {abs(g_adj - fd) / max(abs(fd), 1e-14):.2e}")
    assert abs(g_adj - fd) < 2e-4 * max(abs(fd), 1e-12), (g_adj, fd)
