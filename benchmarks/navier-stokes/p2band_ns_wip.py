"""C item: p2-band SBM-NS composition — p2 ring at the cylinder, p1 far
field, Cd gate."""

import os as _bos, sys as _bsys  # noqa: E402  (benchmark import bootstrap)
_bsys.path.insert(0, _bos.path.dirname(_bos.path.dirname(_bos.path.abspath(__file__))))
import _bench_bootstrap  # noqa: E402,F401
import sys
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu
sys.path.insert(0, "tests")
import test_cylinder as tc
from diffsim import default_device
from diffsim.octree.build import build_uniform
from diffsim.mesh.nodes import build_mesh
from diffsim.mesh.constraints import build_constraints
from diffsim.mesh.basis import basis_tables
from diffsim.mesh.faces import face_tables
from diffsim.assembly.operators import DeviceMesh
from diffsim.geometry.csg import Sphere
from diffsim.sbm.surrogate import (classify_lambda, extract_surrogate,
                                   GeometryData, p2_band)
from diffsim.sbm.vector import sbm_vector_dirichlet, surrogate_traction
from diffsim.api.ns_bricks import assemble_linear_ns
from diffsim.physics.poisson import gauss_points

LEVEL, ndof, dim, dt = 5, 3, 2, 0.05
oracle = Sphere(tc.CTR, tc.R)
tree = build_uniform(LEVEL, dim=2)
ret, _ = classify_lambda(tree, oracle, 0.5, domain="outside")
sf = extract_surrogate(ret)
p_elem = p2_band(ret, sf, n_layers=3)
mesh = build_mesh(ret, p=p_elem)
cons = build_constraints(mesh)
tables = {1: basis_tables(1, dim=2), 2: basis_tables(2, dim=2)}
dm = DeviceMesh.from_mesh(mesh, cons, tables, default_device())
geo = GeometryData.evaluate(oracle, ret, sf, face_tables(2, 2),
                            domain="outside")
n2 = int((np.asarray(p_elem) == 2).sum())
print(f"MIXED MESH: {n2} p2 band elems / {len(p_elem)} total",
      flush=True)
Af, bf = sbm_vector_dirichlet(dm, sf, geo,
                              lambda y: np.zeros((len(y), 2)), tc.NU,
                              ndof, alpha=40.0)          # p^2 law
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
for step in range(120):
    u_node = x.reshape(nfree, ndof)[:, :dim]
    full = np.asarray(Tc @ u_node)
    aq, dq, fq = {}, {}, {}
    for pv in dm.bins:
        tb = dm.tables_by_p[pv]
        vals = full[mesh.conn_of[pv]]
        aq[pv] = np.einsum("qa,ead->eqd", tb.N, vals).reshape(-1, dim)
        h = mesh.tree.h()[mesh.bins[pv]]
        dq[pv] = (np.einsum("qad,ead->eq", tb.dN, vals)
                  * (2.0 / h)[:, None]).reshape(-1)
        fq[pv] = aq[pv] / dt
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
x_full = np.asarray(T_vec @ x)
F = surrogate_traction(dm, sf, geo, x_full, tc.NU, ndof)
Cd = F[0] / (0.5 * tc.U_IN ** 2 * 2 * tc.R)
print(f"P2-BAND NS L{LEVEL} Re=20: Cd={Cd:.3f} "
      f"(p2-full 1.334, p1 1.352, lit 1.33)", flush=True)
