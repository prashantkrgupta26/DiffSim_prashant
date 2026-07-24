"""M2-A4: de Vahl Davis differentially heated cavity (Boussinesq).

Nondim form (Pr=0.71): u_t + u.grad u = -grad p + Pr lap u + Ra Pr T e_y;
T_t + u.grad T = lap T. No-slip walls; T=1 at x=0 (hot), T=0 at x=1;
natural top/bottom. One-way-lagged coupling per BDF step. Nu_avg at the
hot wall by consistent flux. Benchmarks: Ra 1e3 -> 1.118, 1e4 -> 2.243.

Run: python benchmarks/devahl_davis.py [Ra] [level] [steps]
"""

import os as _bos, sys as _bsys  # noqa: E402  (benchmark import bootstrap)
_bsys.path.insert(0, _bos.path.dirname(_bos.path.dirname(_bos.path.abspath(__file__))))
import _bench_bootstrap  # noqa: E402,F401
import sys

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu

from diffsim import default_device
from diffsim.octree.build import build_uniform
from diffsim.mesh.nodes import build_mesh
from diffsim.mesh.constraints import build_constraints
from diffsim.mesh.basis import basis_tables
from diffsim.assembly.operators import DeviceMesh
from diffsim.api.ns_bricks import assemble_linear_ns
from diffsim.physics.scalar_transport import assemble_scalar_ad
from diffsim.physics.poisson import gauss_points

PR = 0.71


def main(Ra=1e3, level=6, max_steps=600, dt=0.05):
    ndof, dim = 3, 2
    tree = build_uniform(level, dim=2)
    mesh = build_mesh(tree, p=1)
    cons = build_constraints(mesh)
    dm = DeviceMesh.from_mesh(mesh, cons, basis_tables(1, dim=2),
                              default_device())
    Tc = cons.T.tocsr()
    T_vec = sp.kron(Tc, sp.identity(ndof, format="csr"), format="csr")
    nfree = Tc.shape[1]
    coords = mesh.node_coords[cons.free_nodes]
    xq = gauss_points(mesh, dm.tables_by_p)
    on = lambda v, c: np.abs(coords[:, c] - v) < 1e-12
    wall = np.where(on(0, 0) | on(1, 0) | on(0, 1) | on(1, 1))[0]
    hot = np.where(on(0, 0))[0]
    cold = np.where(on(1, 0))[0]

    def gp_of(node_scalar_free):
        full = np.asarray(Tc @ node_scalar_free)
        out = {}
        for pv in dm.bins:
            tb = dm.tables_by_p[pv]
            out[pv] = np.einsum("qa,ea->eq", tb.N,
                                full[mesh.conn_of[pv]]).reshape(-1)
        return out

    def gp_vec(node_vec_free):                     # [nfree, 2]
        full = np.asarray(Tc @ node_vec_free)
        aq, dq = {}, {}
        for pv in dm.bins:
            tb = dm.tables_by_p[pv]
            vals = full[mesh.conn_of[pv]]
            aq[pv] = np.einsum("qa,ead->eqd", tb.N, vals).reshape(-1, dim)
            h = mesh.tree.h()[mesh.bins[pv]]
            dq[pv] = (np.einsum("qad,ead->eq", tb.dN, vals)
                      * (2.0 / h)[:, None]).reshape(-1)
        return aq, dq

    x = np.zeros(nfree * ndof)
    Tn = np.zeros(nfree)
    Tn[hot] = 1.0
    T_hist = [Tn.copy(), Tn.copy()]
    u_hist = [np.zeros((nfree, dim)), np.zeros((nfree, dim))]
    sigma = 1.5 / dt                               # BDF2
    pin = int(nfree // 2)

    for step in range(max_steps):
        c0, ch = (1.0, [1.0]) if step == 0 else (1.5, [2.0, -0.5])
        sig = c0 / dt
        # ---- NS step (advect by current u, buoyancy from current T)
        u_now = x.reshape(nfree, ndof)[:, :dim]
        aq, dq = gp_vec(u_now)
        Tgp = gp_of(T_hist[0])
        fq = {}
        for pv in xq:
            f = np.zeros((len(xq[pv]), dim))
            f[:, 1] = Ra * PR * Tgp[pv]
            hist = sum(c * gp_vec(u_hist[k])[0][pv]
                       for k, c in enumerate(ch)) / dt
            fq[pv] = f + hist
        A, b = assemble_linear_ns(dm, aq, dq, fq, PR, sigma=sig)
        A = A.tolil()
        for i in wall:
            for c in range(dim):
                r = i * ndof + c
                A.rows[r] = [int(r)]; A.data[r] = [1.0]; b[r] = 0.0
        rp = pin * ndof + dim
        A.rows[rp] = [rp]; A.data[rp] = [1.0]; b[rp] = 0.0
        x = splu(A.tocsr().tocsc()).solve(b)
        u_new = x.reshape(nfree, ndof)[:, :dim]
        # ---- scalar step (advect by u_new)
        aq2, _ = gp_vec(u_new)
        fq2 = {pv: sum(c * gp_of(T_hist[k])[pv]
                       for k, c in enumerate(ch)) / dt for pv in xq}
        As, bs = assemble_scalar_ad(dm, aq2, fq2, 1.0, sigma=sig)
        As_v = As.copy()
        bs_v = bs.copy()
        As = As.tolil()
        for i in hot:
            As.rows[i] = [int(i)]; As.data[i] = [1.0]; bs[i] = 1.0
        for i in cold:
            As.rows[i] = [int(i)]; As.data[i] = [1.0]; bs[i] = 0.0
        Tn = splu(As.tocsr().tocsc()).solve(bs)
        rate = max(np.abs(u_new - u_hist[0]).max(),
                   np.abs(Tn - T_hist[0]).max()) / dt
        T_hist = [Tn.copy(), T_hist[0]]
        u_hist = [u_new.copy(), u_hist[0]]
        if step % 50 == 0:
            print(f"  step {step}: rate {rate:.2e}", flush=True)
        if rate < 1e-6 and step > 10:
            break
    # ---- Nu_avg at the hot wall: consistent flux (steady operator,
    # sigma=0, no history) paired with the hot-wall indicator
    aqs, _ = gp_vec(u_hist[0])
    zero_f = {pv: np.zeros(len(xq[pv])) for pv in xq}
    As0, bs0 = assemble_scalar_ad(dm, aqs, zero_f, 1.0, sigma=0.0)
    chi = np.zeros(nfree)
    chi[hot] = 1.0
    Nu = abs(float(chi @ (As0 @ Tn - bs0)))
    tag = {1e3: 1.118, 1e4: 2.243, 1e5: 4.519}.get(Ra, None)
    print(f"Ra={Ra:.0e} L{level}: steps={step} Nu_avg={Nu:.4f} "
          f"(benchmark {tag})", flush=True)
    return Nu


if __name__ == "__main__":
    Ra = float(sys.argv[1]) if len(sys.argv) > 1 else 1e3
    lv = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    ns = int(sys.argv[3]) if len(sys.argv) > 3 else 600
    main(Ra, lv, ns)
