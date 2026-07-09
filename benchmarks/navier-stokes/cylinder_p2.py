"""C item 1: cylinder Re=20 at p2 — SBM vector-face genericity + Cd."""

import os as _bos, sys as _bsys  # noqa: E402  (benchmark import bootstrap)
_bsys.path.insert(0, _bos.path.dirname(_bos.path.dirname(_bos.path.abspath(__file__))))
import _bench_bootstrap  # noqa: E402,F401
import sys, time
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu
sys.path.insert(0, "tests")
import test_cylinder as tc
from diffsim.octree.build import build_uniform
from diffsim.mesh.nodes import build_mesh
from diffsim.mesh.constraints import build_constraints
from diffsim.mesh.basis import basis_tables
from diffsim.mesh.faces import face_tables
from diffsim.assembly.operators import DeviceMesh
from diffsim.geometry.csg import Sphere
from diffsim.sbm.surrogate import (classify_lambda, extract_surrogate,
                                   GeometryData)
from diffsim.sbm.vector import sbm_vector_dirichlet, surrogate_traction
from diffsim.api.ns_bricks import assemble_linear_ns
from diffsim.physics.poisson import gauss_points

P, LEVEL, ndof, dim, dt = 2, 5, 3, 2, 0.05
oracle = Sphere(tc.CTR, tc.R)
tree = build_uniform(LEVEL, dim=2)
ret, _ = classify_lambda(tree, oracle, 0.5, domain="outside")
sf = extract_surrogate(ret)
mesh = build_mesh(ret, p=P)
cons = build_constraints(mesh)
dm = DeviceMesh.from_mesh(mesh, cons, basis_tables(P, dim=2), "cuda:0")
geo = GeometryData.evaluate(oracle, ret, sf, face_tables(P, 2),
                            domain="outside")
t0 = time.time()
import os
ALPHA = float(os.environ.get("SBM_ALPHA", "10"))
Af, bf = sbm_vector_dirichlet(dm, sf, geo,
                              lambda y: np.zeros((len(y), 2)), tc.NU,
                              ndof, alpha=ALPHA)
print(f"P2 VECTOR FACES: OK in {time.time()-t0:.1f}s nnz={Af.nnz}",
      flush=True)
Tc = cons.T.tocsr()
T_vec = sp.kron(Tc, sp.identity(ndof, format="csr"), format="csr")
nfree = Tc.shape[1]
coords = mesh.node_coords[cons.free_nodes]
xq = gauss_points(mesh, dm.tables_by_p)
on = lambda v, c: np.abs(coords[:, c] - v) < 1e-12
strong = np.where(on(0.0, 0) | on(0.0, 1) | on(1.0, 1))[0]
g_strong = np.zeros((len(strong), 2))
g_strong[np.abs(coords[strong, 0]) < 1e-12, 0] = tc.U_IN
x = np.zeros(nfree * ndof)
sigma = 1.0 / dt
pv = P
for step in range(120):
    u_node = x.reshape(nfree, ndof)[:, :dim]
    full = np.asarray(Tc @ u_node)
    tb = dm.tables_by_p[pv]
    vals = full[mesh.conn_of[pv]]
    aq = {pv: np.einsum("qa,ead->eqd", tb.N, vals).reshape(-1, dim)}
    h = mesh.tree.h()[mesh.bins[pv]]
    dq = {pv: (np.einsum("qad,ead->eq", tb.dN, vals)
               * (2.0 / h)[:, None]).reshape(-1)}
    fq = {pv: aq[pv] / dt}
    A, b = assemble_linear_ns(dm, aq, dq, fq, tc.NU, sigma=sigma)
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
    if step % 40 == 39:
        xf = np.asarray(T_vec @ x)
        F_ = surrogate_traction(dm, sf, geo, xf, tc.NU, ndof)
        print(f"  step {step+1}: Cd={F_[0]/(0.5*tc.U_IN**2*2*tc.R):.3f}",
              flush=True)
x_full = np.asarray(T_vec @ x)
F = surrogate_traction(dm, sf, geo, x_full, tc.NU, ndof)
Cd = F[0] / (0.5 * tc.U_IN ** 2 * 2 * tc.R)
print(f"P2 L{LEVEL} alpha={ALPHA:.0f} Re=20: Cd={Cd:.3f} (locked p1 1.352, lit 1.33)",
      flush=True)
