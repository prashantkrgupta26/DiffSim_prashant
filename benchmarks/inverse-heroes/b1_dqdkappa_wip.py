"""B1 job 1: full dQ/dkappa on the heated cylinder (L5) vs central FD.
Chain: Q = chi.(A_v(k) T(k) - b_v) with A(k) T(k) = b(k) (composed
system). dQ/dk = chi.(dA_v/dk T) + [chi.A_v - via T] ... implemented as:
lam = A^T \ (A_v^T chi) adjoint route:
dQ/dk = chi.(dA_v/dk T) - lam.(dA/dk T - db/dk)
where dA/dk = dA_v/dk + A_f/k (face block linear in k), db/dk = b_f/k.
Volume derivatives via the taped cotangents (kq per-GP, summed)."""

import os as _bos, sys as _bsys  # noqa: E402  (benchmark import bootstrap)
_bsys.path.insert(0, _bos.path.dirname(_bos.path.dirname(_bos.path.abspath(__file__))))
import _bench_bootstrap  # noqa: E402,F401
import sys
import numpy as np
from scipy.sparse.linalg import splu
sys.path.insert(0, "benchmarks")
import heated_cylinder as hc
from diffsim.sbm.scalar_adjoint import scalar_volume_cotangents

def solve_Q(kappa, ret_state=False):
    out = hc.run.__wrapped__ if hasattr(hc.run, "__wrapped__") else None
    # inline: reuse hc.run pieces at L5 with variable kappa
    from diffsim import default_device
    from diffsim.octree.build import build_uniform
    from diffsim.mesh.nodes import build_mesh
    from diffsim.mesh.constraints import build_constraints
    from diffsim.mesh.basis import basis_tables
    from diffsim.mesh.faces import face_tables
    from diffsim.assembly.operators import DeviceMesh
    from diffsim.geometry.csg import Sphere
    from diffsim.sbm.surrogate import (classify_lambda, extract_surrogate,
                                       GeometryData)
    from diffsim.sbm.poisson import SBMPoisson
    from diffsim.physics.scalar_transport import assemble_scalar_ad
    from diffsim.physics.poisson import gauss_points
    level = 5
    tree = build_uniform(level, dim=2)
    oracle = Sphere(hc.CTR, hc.R)
    ret, _ = classify_lambda(tree, oracle, 1.0, domain="outside")
    sf = extract_surrogate(ret)
    mesh = build_mesh(ret, p=1)
    cons = build_constraints(mesh)
    dm = DeviceMesh.from_mesh(mesh, cons, basis_tables(1, dim=2),
                              default_device())
    geo = GeometryData.evaluate(oracle, ret, sf, face_tables(1, 2),
                                domain="outside")
    xq = gauss_points(mesh, dm.tables_by_p)
    aq = {pv: np.tile([1.0, 0.0], (len(xq[pv]), 1)) for pv in xq}
    fq = {pv: np.zeros(len(xq[pv])) for pv in xq}
    A_v, b_v = assemble_scalar_ad(dm, aq, fq, kappa, sigma=0.0)
    # alpha FROZEN at the base-kappa value (solver parameter, not
    # physics: the kappa-gradient treats it as constant — standard)
    pe0 = 1.0 * 2 * hc.R / 0.05
    prob = SBMPoisson(dm, geo=geo, sf=sf,
                      g_fn=lambda p: np.ones(len(p)), kappa=kappa,
                      alpha=20.0 * (1.0 + pe0 / 4.0))
    A_f, b_f = prob.face_system()
    A_a, b_a = hc.advective_faces(dm, sf, geo,
                                  a_fn=lambda gp: np.array([1., 0.]),
                                  g_fn=lambda gp: 1.0)
    A = (A_v + A_f + A_a).tolil()
    b = b_v + b_f + b_a
    coords = mesh.node_coords[cons.free_nodes]
    dirn = np.where(np.abs(coords[:, 0]) < 1e-12)[0]
    for i in dirn:
        A.rows[i] = [int(i)]; A.data[i] = [1.0]; b[i] = 0.0
    A = A.tocsr()
    T_free = splu(A.tocsc()).solve(b)
    fnodes = np.unique(mesh.conn_of[1][
        np.searchsorted(mesh.bins[1], sf.elem)].ravel())
    chi_full = np.zeros(dm.n_nodes); chi_full[fnodes] = 1.0
    chi = np.asarray(cons.T.T @ chi_full)
    Q = float(chi @ (A_v @ T_free - b_v))
    if not ret_state:
        return Q
    return dict(Q=Q, A=A, A_v=A_v, A_f=A_f, b_f=b_f, chi=chi,
                T_free=T_free, dm=dm, cons=cons, aq=aq, xq=xq,
                dirn=dirn, kappa=kappa)

K0 = 0.05
st = solve_Q(K0, ret_state=True)
dm, cons = st["dm"], st["cons"]
# adjoint: dQ/dT = A_v^T chi (Q's explicit T-dependence); constraint:
# A^T lam = A_v^T chi with adjoint Dirichlet zeroing on dirn columns...
# row-replaced A already carries identity rows at dirn -> A^T has unit
# cols; solve directly:
rhs = st["A_v"].T @ st["chi"]
rhs[st["dirn"]] = 0.0                     # QoI has no BC-node dependence
lam = splu(st["A"].tocsc().T).solve(rhs)
lam[st["dirn"]] = 0.0  # replaced rows carry no dA/dk (identity rows)
# volume kappa-derivative via taped cotangents: T and lam in FULL space
T_full = np.asarray(cons.T @ st["T_free"])
lam_full = np.asarray(cons.T @ lam)
chi_fullv = np.asarray(cons.T @ st["chi"])
kq = {pv: np.full(len(st["xq"][pv]), K0) for pv in st["xq"]}
cot_lam = scalar_volume_cotangents(dm, st["aq"], kq, 0.0, T_full,
                                   lam_full)
cot_chi = scalar_volume_cotangents(dm, st["aq"], kq, 0.0, T_full,
                                   chi_fullv)
dAv_lam = -sum(c[0].sum() for c in cot_lam.values())   # lam.dAv/dk T
dAv_chi = -sum(c[0].sum() for c in cot_chi.values())   # chi.dAv/dk T
dAf_lam = float(lam @ (st["A_f"] @ st["T_free"] - st["b_f"])) / K0
dQdk = dAv_chi - dAv_lam - dAf_lam
for eps in (1e-4, 3e-5, 1e-5):
    fd = (solve_Q(K0 + eps) - solve_Q(K0 - eps)) / (2 * eps)
    print(f"eps={eps:.0e}: adjoint={dQdk:+.6e} FD={fd:+.6e} "
          f"rel={abs(dQdk-fd)/max(abs(fd),1e-30):.2e}", flush=True)
