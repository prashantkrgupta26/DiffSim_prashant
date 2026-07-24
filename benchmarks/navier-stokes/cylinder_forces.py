"""Immersed cylinder in channel flow — drag/lift via SBM shifted traction.

Usage:
    python benchmarks/cylinder_forces.py [--level 6] [--re 20] [--steps 400]

Re = 2 U r / nu (diameter-based). The Re=20 CI-locked configuration
(r = 0.07 at (0.3, 0.5) in [0,1]^2, 14% blockage, uniform inflow, no-slip
walls, free outflow) gives Cd = 2.847 at level 5 — top of the confined-
cylinder literature band. Re >= ~50 sheds a vortex street: raise --steps and
watch the lift history oscillate (Strouhal extraction lands with the M1b
benchmark task).

The force is the production SurfaceLoop pattern evaluated on the surrogate
boundary with the SBM machinery: F = oint (p n_hat - nu grad(u).n_hat) dS,
area-corrected staircase -> true boundary, n_hat = obstacle-outward
(= -geo.n; orientation contract in sbm/vector.py).
"""

import os as _bos, sys as _bsys  # noqa: E402  (benchmark import bootstrap)
_bsys.path.insert(0, _bos.path.dirname(_bos.path.dirname(_bos.path.abspath(__file__))))
import _bench_bootstrap  # noqa: E402,F401
import argparse
import sys
import os

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "tests"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=int, default=6)
    ap.add_argument("--re", type=float, default=20.0)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--dt", type=float, default=0.05)
    args = ap.parse_args()

    import test_cylinder as tc
    tc.NU = 2 * tc.U_IN * tc.R / args.re

    # reuse the test's solver loop but with driver knobs + force history
    import types
    import scipy.sparse as sp
    from scipy.sparse.linalg import splu
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
    from diffsim.sbm.vector import sbm_vector_dirichlet, surrogate_traction
    from diffsim.api.ns_bricks import assemble_linear_ns
    from diffsim.physics.poisson import gauss_points

    ndof, dim = 3, 2
    oracle = Sphere(tc.CTR, tc.R)
    tree = build_uniform(args.level, dim=2)
    ret, _ = classify_lambda(tree, oracle, 0.5, domain="outside")
    sf = extract_surrogate(ret)
    mesh = build_mesh(ret, p=1)
    cons = build_constraints(mesh)
    dm = DeviceMesh.from_mesh(mesh, cons, basis_tables(1, dim=2), default_device())
    geo = GeometryData.evaluate(oracle, ret, sf, face_tables(1, 2),
                                domain="outside")
    T = dm.constraints.T.tocsr()
    T_vec = sp.kron(T, sp.identity(ndof, format="csr"), format="csr")
    nfree = T.shape[1]
    coords = mesh.node_coords[cons.free_nodes]
    xq = gauss_points(mesh, dm.tables_by_p)
    on = lambda v, c: np.abs(coords[:, c] - v) < 1e-12
    strong = np.where(on(0.0, 0) | on(0.0, 1) | on(1.0, 1))[0]
    g_strong = np.zeros((len(strong), 2))
    g_strong[np.abs(coords[strong, 0]) < 1e-12, 0] = tc.U_IN
    x = np.zeros(nfree * ndof)
    sigma = 1.0 / args.dt
    qref = 0.5 * tc.U_IN ** 2 * 2 * tc.R

    def gp_field(node_vec):
        full = np.asarray(T @ node_vec)
        aq, dq = {}, {}
        for pv in dm.bins:
            tb = dm.tables_by_p[pv]
            vals = full[mesh.conn_of[pv]]
            aq[pv] = np.einsum("qa,ead->eqd", tb.N, vals).reshape(-1, dim)
            h = mesh.tree.h()[mesh.bins[pv]]
            dq[pv] = (np.einsum("qad,ead->eq", tb.dN, vals)
                      * (2.0 / h)[:, None]).reshape(-1)
        return aq, dq

    print(f"{'step':>5} {'Cd':>9} {'Cl':>9}")
    for step in range(1, args.steps + 1):
        u_node = x.reshape(nfree, ndof)[:, :dim]
        aq, dq = gp_field(u_node)
        fq = {pv: aq[pv] / args.dt for pv in xq}
        A, b = assemble_linear_ns(dm, aq, dq, fq, tc.NU, sigma=sigma)
        Af, bf = sbm_vector_dirichlet(
            dm, sf, geo, lambda y: np.zeros((len(y), 2)), tc.NU, ndof)
        A = (A + T_vec.T @ Af @ T_vec).tolil()
        b = b + np.asarray(T_vec.T @ bf)
        for k, i in enumerate(strong):
            for c in range(dim):
                r = i * ndof + c
                A.rows[r] = [int(r)]
                A.data[r] = [1.0]
                b[r] = g_strong[k, c]
        pin = int(np.argmax(coords[:, 0] + coords[:, 1])) * ndof + dim
        A.rows[pin] = [pin]
        A.data[pin] = [1.0]
        b[pin] = 0.0
        x = splu(A.tocsr().tocsc()).solve(b)
        if step % max(1, args.steps // 20) == 0:
            F = surrogate_traction(dm, sf, geo, np.asarray(T_vec @ x),
                                   tc.NU, ndof)
            print(f"{step:>5} {F[0] / qref:>9.4f} {F[1] / qref:>9.4f}")


if __name__ == "__main__":
    main()
