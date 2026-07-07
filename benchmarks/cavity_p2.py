"""C gate 1: lid-driven cavity Re=100 at p2 L4/L5 — BDF1 pseudo-time to
steady, centerline u-velocity vs Ghia-class expectations (loose) and
p1-vs-p2 comparison."""
import numpy as np
from scipy.sparse.linalg import splu
from diffsim.octree.build import build_uniform
from diffsim.mesh.nodes import build_mesh
from diffsim.mesh.constraints import build_constraints
from diffsim.mesh.basis import basis_tables
from diffsim.assembly.operators import DeviceMesh
from diffsim.api.ns_bricks import assemble_linear_ns
from diffsim.physics.poisson import gauss_points

def cavity(p, level, nu=0.01, dt=0.05, steps=120):
    ndof, dim = 3, 2
    tree = build_uniform(level, dim=2)
    mesh = build_mesh(tree, p=p)
    cons = build_constraints(mesh)
    dm = DeviceMesh.from_mesh(mesh, cons, basis_tables(p, dim=2),
                              "cuda:0")
    Tc = cons.T.tocsr()
    nfree = Tc.shape[1]
    coords = mesh.node_coords[cons.free_nodes]
    xq = gauss_points(mesh, dm.tables_by_p)
    pv = p
    on = lambda v, c: np.abs(coords[:, c] - v) < 1e-12
    wall = np.where(on(0, 0) | on(1, 0) | on(0, 1) | on(1, 1))[0]
    lid = on(1, 1)
    x = np.zeros(nfree * ndof)
    sigma = 1.0 / dt
    for step in range(steps):
        u_node = x.reshape(nfree, ndof)[:, :dim]
        full = np.asarray(Tc @ u_node)
        tb = dm.tables_by_p[pv]
        vals = full[mesh.conn_of[pv]]
        aq = {pv: np.einsum("qa,ead->eqd", tb.N, vals).reshape(-1, dim)}
        h = mesh.tree.h()[mesh.bins[pv]]
        dq = {pv: (np.einsum("qad,ead->eq", tb.dN, vals)
                   * (2.0 / h)[:, None]).reshape(-1)}
        fq = {pv: aq[pv] / dt}
        A, b = assemble_linear_ns(dm, aq, dq, fq, nu, sigma=sigma)
        A = A.tolil()
        for i in wall:
            for c in range(dim):
                r = i * ndof + c
                A.rows[r] = [int(r)]; A.data[r] = [1.0]
                b[r] = 1.0 if (c == 0 and lid[i]) else 0.0
        A.rows[2] = [2]; A.data[2] = [1.0]; b[2] = 0.0
        x = splu(A.tocsr().tocsc()).solve(b)
    # centerline u_x at x=0.5
    sel = np.where(np.abs(coords[:, 0] - 0.5) < 1e-9)[0]
    ys = coords[sel, 1]
    ux = x.reshape(nfree, ndof)[sel, 0]
    o = np.argsort(ys)
    umin = float(ux.min())
    print(f"p{p} L{level}: u_min(centerline)={umin:+.4f} "
          f"(Ghia Re100 ~ -0.211)", flush=True)
    return ys[o], ux[o]

for p, lv in ((1, 5), (2, 4), (2, 5)):
    cavity(p, lv)
