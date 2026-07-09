"""M2-B3: the S2 RETRAIN DEMO — recover a hidden conductivity closure on
the de Vahl Davis Ra=1e4 configuration by in-loop training.

APPROXIMATION (documented): the FLOW is frozen at the base-kappa steady
state (one-way — the closure perturbs only the scalar path). The fully
coupled retrain (closure -> T -> buoyancy -> u) is the recorded stretch.
"""

import os as _bos, sys as _bsys  # noqa: E402  (benchmark import bootstrap)
_bsys.path.insert(0, _bos.path.dirname(_bos.path.dirname(_bos.path.abspath(__file__))))
import _bench_bootstrap  # noqa: E402,F401
import numpy as np
import torch
from scipy.sparse.linalg import splu

import sys
sys.path.insert(0, "benchmarks")
import devahl_davis as dvd
from diffsim.physics.scalar_transport import assemble_scalar_ad
from diffsim.sbm.scalar_adjoint import scalar_volume_cotangents
from diffsim.mesh.pointeval import point_eval_weights

# ---- base coupled steady state (unit kappa) --------------------------
import io, contextlib
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    # run to steady; harvest module-scope state by re-executing main's
    # internals is heavy — instead call main and rebuild the scalar
    # operators here from a FROZEN velocity obtained by rerunning at
    # Ra=1e4 (fast: 48 steps)
    pass

# rebuild directly (compact re-implementation of the dvd loop, Ra=1e4)
from diffsim.octree.build import build_uniform
from diffsim.mesh.nodes import build_mesh
from diffsim.mesh.constraints import build_constraints
from diffsim.mesh.basis import basis_tables
from diffsim.assembly.operators import DeviceMesh
from diffsim.api.ns_bricks import assemble_linear_ns
from diffsim.physics.poisson import gauss_points
import scipy.sparse as sp

Ra, PR, level, dt = 1e4, 0.71, 5, 0.05
ndof, dim = 3, 2
tree = build_uniform(level, dim=2)
mesh = build_mesh(tree, p=1)
cons = build_constraints(mesh)
dm = DeviceMesh.from_mesh(mesh, cons, basis_tables(1, dim=2), "cuda:0")
Tc = cons.T.tocsr()
nfree = Tc.shape[1]
coords = mesh.node_coords[cons.free_nodes]
xq = gauss_points(mesh, dm.tables_by_p)
pv = 1
gpx = xq[pv]
ngp = len(gpx)
on = lambda v, c: np.abs(coords[:, c] - v) < 1e-12
wall = np.where(on(0, 0) | on(1, 0) | on(0, 1) | on(1, 1))[0]
hot = np.where(on(0, 0))[0]
cold = np.where(on(1, 0))[0]
dirT = np.concatenate([hot, cold])

def gp_vec(v):
    full = np.asarray(Tc @ v)
    tb = dm.tables_by_p[pv]
    vals = full[mesh.conn_of[pv]]
    a = np.einsum("qa,ead->eqd", tb.N, vals).reshape(-1, dim)
    h = mesh.tree.h()[mesh.bins[pv]]
    d = (np.einsum("qad,ead->eq", tb.dN, vals)
         * (2.0 / h)[:, None]).reshape(-1)
    return {pv: a}, {pv: d}

def gp_sc(s):
    full = np.asarray(Tc @ s)
    tb = dm.tables_by_p[pv]
    return {pv: np.einsum("qa,ea->eq", tb.N,
                          full[mesh.conn_of[pv]]).reshape(-1)}

# march the coupled system at unit kappa to steady (as in dvd, 60 steps)
x = np.zeros(nfree * ndof)
Tn = np.zeros(nfree); Tn[hot] = 1.0
Th = [Tn.copy(), Tn.copy()]
uh = [np.zeros((nfree, dim)), np.zeros((nfree, dim))]
for step in range(60):
    c0, ch = (1.0, [1.0]) if step == 0 else (1.5, [2.0, -0.5])
    sig = c0 / dt
    aq, dq = gp_vec(x.reshape(nfree, ndof)[:, :dim])
    Tgp = gp_sc(Th[0])
    fq = {pv: np.zeros((ngp, dim))}
    fq[pv][:, 1] = Ra * PR * Tgp[pv]
    fq[pv] += sum(c * gp_vec(uh[k])[0][pv] for k, c in enumerate(ch)) / dt
    A, b = assemble_linear_ns(dm, aq, dq, fq, PR, sigma=sig)
    A = A.tolil()
    for i in wall:
        for c in range(dim):
            r = i * ndof + c
            A.rows[r] = [int(r)]; A.data[r] = [1.0]; b[r] = 0.0
    rp = (nfree // 2) * ndof + dim
    A.rows[rp] = [rp]; A.data[rp] = [1.0]; b[rp] = 0.0
    x = splu(A.tocsr().tocsc()).solve(b)
    u_new = x.reshape(nfree, ndof)[:, :dim]
    aq2, _ = gp_vec(u_new)
    fq2 = {pv: sum(c * gp_sc(Th[k])[pv] for k, c in enumerate(ch)) / dt}
    As, bs = assemble_scalar_ad(dm, aq2, fq2, 1.0, sigma=sig)
    As = As.tolil()
    for i in hot:
        As.rows[i] = [int(i)]; As.data[i] = [1.0]; bs[i] = 1.0
    for i in cold:
        As.rows[i] = [int(i)]; As.data[i] = [1.0]; bs[i] = 0.0
    Tn = splu(As.tocsr().tocsc()).solve(bs)
    Th = [Tn.copy(), Th[0]]
    uh = [u_new.copy(), uh[0]]

AQ, _ = gp_vec(uh[0])                    # FROZEN flow
T_base = Th[0]
print(f"[base] coupled steady built (60 steps)", flush=True)

# ---- retrain problem: STEADY scalar with field kappa over frozen flow
fq0 = {pv: np.zeros(ngp)}
py, pxx = np.meshgrid(np.linspace(0.2, 0.8, 5), np.linspace(0.2, 0.8, 5))
W = point_eval_weights(mesh, np.column_stack([pxx.ravel(), py.ravel()]))

def solve_scalar(kq_np):
    A, b = assemble_scalar_ad(dm, AQ, fq0,
                              lambda x_: kq_np[:len(x_)], sigma=0.0)
    A = A.tolil()
    for i in hot:
        A.rows[i] = [int(i)]; A.data[i] = [1.0]; b[i] = 1.0
    for i in cold:
        A.rows[i] = [int(i)]; A.data[i] = [1.0]; b[i] = 0.0
    A = A.tocsr()
    return splu(A.tocsc()).solve(b), A

def kq_star(x):
    return 1.0 + 0.3 * np.tanh(6.0 * (x[:, 1] - 0.5))

target = W @ np.asarray(Tc @ solve_scalar(kq_star(gpx))[0])
Tgp_frozen = gp_sc(T_base)[pv]
feats = torch.tensor(np.column_stack([gpx, Tgp_frozen])).float()

torch.manual_seed(0)
net = torch.nn.Sequential(
    torch.nn.Linear(3, 16), torch.nn.Tanh(),
    torch.nn.Linear(16, 16), torch.nn.Tanh(),
    torch.nn.Linear(16, 1))
opt = torch.optim.Adam(net.parameters(), lr=5e-3)
for it in range(151):
    out = net(feats).double().ravel()
    kq_t = 1.0 + 0.5 * torch.tanh(out)
    kq_np = kq_t.detach().numpy()
    T, A = solve_scalar(kq_np)
    r = W @ np.asarray(Tc @ T) - target
    J = 0.5 * float(r @ r)
    if it == 0:
        print(f"[signal] J0 = {J:.3e}", flush=True)
        assert J > 1e-8
    rhs = np.asarray((W @ Tc).T @ r)
    rhs[dirT] = 0.0
    lam = splu(A.tocsc().T).solve(rhs)
    lam[dirT] = 0.0
    cl = scalar_volume_cotangents(dm, AQ, {pv: kq_np}, 0.0,
                                  np.asarray(Tc @ T),
                                  np.asarray(Tc @ lam))
    opt.zero_grad()
    kq_t.backward(torch.tensor(cl[pv][0]))
    opt.step()
    if it % 25 == 0:
        rmse = float(np.sqrt(((kq_np - kq_star(gpx)) ** 2).mean()))
        print(f"it {it:4d}: J={J:.4e} closure-RMSE={rmse:.4e}",
              flush=True)
rmse = float(np.sqrt(((kq_np - kq_star(gpx)) ** 2).mean()))
print(f"[result] J={J:.3e} closure RMSE={rmse:.4e} (field ~1.0+-0.3)",
      flush=True)
