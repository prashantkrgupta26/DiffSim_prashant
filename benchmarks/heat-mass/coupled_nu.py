"""M2-A3 EXIT: coupled Nu(Re=20, Pr=0.7) — NS flow (locked test_cylinder
config) advecting the scalar; hot cylinder; consistent-flux Nu_D vs
Hilpert (~2.5, loose gate)."""

import os as _bos, sys as _bsys  # noqa: E402  (benchmark import bootstrap)
_bsys.path.insert(0, _bos.path.dirname(_bos.path.dirname(_bos.path.abspath(__file__))))
import _bench_bootstrap  # noqa: E402,F401
import sys
import numpy as np
from scipy.sparse.linalg import splu

sys.path.insert(0, "tests")
sys.path.insert(0, "benchmarks")
import test_cylinder as tc
import heated_cylinder as hc
from diffsim.sbm.surrogate import face_gauss_points
from diffsim.mesh.faces import face_tables
from diffsim.mesh.pointeval import point_eval_weights
from diffsim.sbm.poisson import SBMPoisson
from diffsim.physics.scalar_transport import assemble_scalar_ad
from diffsim.physics.poisson import gauss_points

PR = 0.7


def main(level=6, steps=150):
    # 1) settled Re=20 flow (inlined from the locked test loop, BDF1
    #    pseudo-time to steady)
    import scipy.sparse as sp
    from diffsim.octree.build import build_uniform
    from diffsim.mesh.nodes import build_mesh
    from diffsim.mesh.constraints import build_constraints
    from diffsim.mesh.basis import basis_tables
    from diffsim.assembly.operators import DeviceMesh
    from diffsim.geometry.csg import Sphere
    from diffsim.sbm.surrogate import (classify_lambda, extract_surrogate,
                                       GeometryData)
    from diffsim.sbm.vector import sbm_vector_dirichlet
    from diffsim.api.ns_bricks import assemble_linear_ns
    ndof, dim, dt = 3, 2, 0.05
    oracle = Sphere(tc.CTR, tc.R)
    tree = build_uniform(level, dim=2)
    ret, _ = classify_lambda(tree, oracle, 0.5, domain="outside")
    sf = extract_surrogate(ret)
    mesh = build_mesh(ret, p=1)
    cons = build_constraints(mesh)
    dm = DeviceMesh.from_mesh(mesh, cons, basis_tables(1, dim=2),
                              "cuda:0")
    geo = GeometryData.evaluate(oracle, ret, sf, face_tables(1, 2),
                                domain="outside")
    Tc = cons.T.tocsr()
    T_vec = sp.kron(Tc, sp.identity(ndof, format="csr"), format="csr")
    nfree = Tc.shape[1]
    coords = mesh.node_coords[cons.free_nodes]
    xq0 = gauss_points(mesh, dm.tables_by_p)
    on = lambda v, c: np.abs(coords[:, c] - v) < 1e-12
    strong = np.where(on(0.0, 0) | on(0.0, 1) | on(1.0, 1))[0]
    g_strong = np.zeros((len(strong), 2))
    g_strong[np.abs(coords[strong, 0]) < 1e-12, 0] = tc.U_IN
    x = np.zeros(nfree * ndof)
    sigma = 1.0 / dt
    for step in range(steps):
        u_node = x.reshape(nfree, ndof)[:, :dim]
        fullu = np.asarray(Tc @ u_node.reshape(-1).reshape(nfree, dim)
                           .ravel().reshape(nfree, dim)[:, 0])
        aqf, dqf = {}, {}
        fullv = np.asarray(Tc @ u_node)
        for pv in dm.bins:
            tb = dm.tables_by_p[pv]
            connp = mesh.conn_of[pv]
            vals = fullv[connp]
            aqf[pv] = np.einsum("qa,ead->eqd", tb.N,
                                vals).reshape(-1, dim)
            h = mesh.tree.h()[mesh.bins[pv]]
            dqf[pv] = (np.einsum("qad,ead->eq", tb.dN, vals)
                       * (2.0 / h)[:, None]).reshape(-1)
        fqf = {pv: aqf[pv] / dt for pv in xq0}
        A, b = assemble_linear_ns(dm, aqf, dqf, fqf, tc.NU, sigma=sigma)
        Af, bf = sbm_vector_dirichlet(
            dm, sf, geo, lambda y: np.zeros((len(y), 2)), tc.NU, ndof)
        A = (A + T_vec.T @ Af @ T_vec).tolil()
        b = b + np.asarray(T_vec.T @ bf)
        for k, i in enumerate(strong):
            for c in range(dim):
                r = i * ndof + c
                A.rows[r] = [int(r)]; A.data[r] = [1.0]
                b[r] = g_strong[k, c]
        pin = int(np.argmax(coords[:, 0] + coords[:, 1]))
        rp = pin * ndof + dim
        A.rows[rp] = [rp]; A.data[rp] = [1.0]; b[rp] = 0.0
        x = splu(A.tocsr().tocsc()).solve(b)
    full = np.asarray(T_vec @ x).reshape(-1, ndof)
    kappa = tc.NU / PR

    # 2) u at scalar GPs
    xq = gauss_points(mesh, dm.tables_by_p)
    aq = {}
    for pv in dm.bins:
        conn = mesh.conn_of[pv]
        tb = dm.tables_by_p[pv]
        vals = full[:, :2][conn]                   # [ne, nbf, 2]
        aq[pv] = np.einsum("qa,ead->eqd", tb.N, vals).reshape(-1, 2)
    fq = {pv: np.zeros(len(xq[pv])) for pv in xq}

    # 3) u at face GPs via the face-owning element's basis (face GPs
    #    sit on the carve boundary — point-eval rejects them)
    ftab = face_tables(1, 2)
    connf = mesh.conn_of[1][np.searchsorted(mesh.bins[1], sf.elem)]
    nqf = ftab.nqf
    u_face = np.zeros((len(sf.elem) * nqf, 2))
    for fi in range(len(sf.elem)):
        f = int(sf.face[fi])
        for q in range(nqf):
            u_face[fi * nqf + q] = ftab.N[f][q] @ full[:, :2][connf[fi]]

    # 4) composed scalar system
    A_v, b_v = assemble_scalar_ad(dm, aq, fq, kappa, sigma=0.0)
    pe = 1.0 * 2 * tc.R / kappa
    prob = SBMPoisson(dm, geo=geo, sf=sf,
                      g_fn=lambda p: np.ones(len(p)), kappa=kappa,
                      alpha=20.0 * (1.0 + pe / 4.0))
    A_f, b_f = prob.face_system()
    A_a, b_a = hc.advective_faces(dm, sf, geo,
                                  a_fn=lambda gp: u_face[gp],
                                  g_fn=lambda gp: 1.0)
    A = (A_v + A_f + A_a).tolil()
    b = b_v + b_f + b_a
    coords = mesh.node_coords[cons.free_nodes]
    for i in np.where(np.abs(coords[:, 0]) < 1e-12)[0]:
        A.rows[i] = [int(i)]; A.data[i] = [1.0]; b[i] = 0.0
    T_free = splu(A.tocsr().tocsc()).solve(b)
    T_all = np.asarray(cons.T @ T_free)

    # 5) consistent-flux Nu
    fnodes = np.unique(mesh.conn_of[1][
        np.searchsorted(mesh.bins[1], sf.elem)].ravel())
    chi_full = np.zeros(dm.n_nodes); chi_full[fnodes] = 1.0
    chi = np.asarray(cons.T.T @ chi_full)
    Q = float(chi @ (A_v @ T_free - b_v))
    nu_d = Q / (np.pi * kappa * 1.0)
    print(f"L{level} Re=20 Pr={PR}: T [{T_all.min():.3f},"
          f"{T_all.max():.3f}] Q={Q:.4f} Nu_D={nu_d:.2f} "
          f"(Hilpert ~2.5)", flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 6,
         int(sys.argv[2]) if len(sys.argv) > 2 else 150)
