"""M2-B2: closure recovery smoke — a torch MLP kappa(x,y; theta) trained
THROUGH the DiffSim adjoint chain to match probe data from a hidden
analytic closure. The S2 'in-the-loop' core.
"""
import numpy as np
import torch
from scipy.sparse.linalg import splu

from diffsim.octree.build import build_uniform
from diffsim.mesh.nodes import build_mesh
from diffsim.mesh.constraints import build_constraints
from diffsim.mesh.basis import basis_tables
from diffsim.assembly.operators import DeviceMesh
from diffsim.physics.scalar_transport import assemble_scalar_ad
from diffsim.physics.poisson import gauss_points
from diffsim.sbm.scalar_adjoint import scalar_volume_cotangents
from diffsim.mesh.pointeval import point_eval_weights

LEVEL = 5

tree = build_uniform(LEVEL, dim=2)
mesh = build_mesh(tree, p=1)
cons = build_constraints(mesh)
dm = DeviceMesh.from_mesh(mesh, cons, basis_tables(1, dim=2), "cuda:0")
xq = gauss_points(mesh, dm.tables_by_p)
pv = 1
gpx = xq[pv]                                    # [ngp, 2]
ngp = len(gpx)
aq = {pv: np.tile([1.0, 0.3], (ngp, 1))}
fq = {pv: np.zeros(ngp)}
coords = mesh.node_coords[cons.free_nodes]
bdry = np.zeros(len(coords), bool)
for c in range(2):
    bdry |= (np.abs(coords[:, c]) < 1e-12) | \
            (np.abs(coords[:, c] - 1) < 1e-12)
dirn = np.where(bdry)[0]
hot = set(np.where(np.abs(coords[:, 0]) < 1e-12)[0])
py, pxx = np.meshgrid(np.linspace(0.2, 0.8, 5), np.linspace(0.2, 0.8, 5))
W = point_eval_weights(mesh, np.column_stack([pxx.ravel(), py.ravel()]))


def solve(kq_np):
    A, b = assemble_scalar_ad(dm, aq, fq,
                              lambda x_: kq_np[:len(x_)], sigma=0.0)
    A = A.tolil()
    for i in dirn:
        A.rows[i] = [int(i)]; A.data[i] = [1.0]
        b[i] = 1.0 if i in hot else 0.0
    A = A.tocsr()
    T = splu(A.tocsc()).solve(b)
    return T, A


def probes(T):
    return W @ np.asarray(cons.T @ T)


def kq_star(x):                                  # hidden closure
    return 0.05 * (1.0 + 0.5 * np.tanh(8.0 * (x[:, 1] - 0.5)))


target = probes(solve(kq_star(gpx))[0])

net = torch.nn.Sequential(
    torch.nn.Linear(2, 16), torch.nn.Tanh(),
    torch.nn.Linear(16, 16), torch.nn.Tanh(),
    torch.nn.Linear(16, 1))
torch.manual_seed(0)
gpx_t = torch.tensor(gpx)
opt = torch.optim.Adam(net.parameters(), lr=3e-3)

# signal-scale check (H4 protocol)
kq0 = 0.05 * (1.0 + 0.1 * np.tanh(net(gpx_t.float()).double()
                                  .detach().numpy().ravel()))
J0 = 0.5 * float(((probes(solve(kq0)[0]) - target) ** 2).sum())
print(f"[signal] J0 = {J0:.3e} (floor ~1e-12 class)", flush=True)
assert J0 > 1e-8, "signal too weak to optimize"

for it in range(201):
    out = net(gpx_t.float()).double().ravel()
    kq_t = 0.05 * (1.0 + 0.1 * out)              # positive-ish scaling
    kq_np = kq_t.detach().numpy()
    T, A = solve(kq_np)
    r = probes(T) - target
    J = 0.5 * float(r @ r)
    dJdT_free = np.asarray((W @ cons.T).T @ r)
    rhs = dJdT_free.copy(); rhs[dirn] = 0.0
    lam = splu(A.tocsc().T).solve(rhs)
    lam[dirn] = 0.0
    cl = scalar_volume_cotangents(dm, aq, {pv: kq_np}, 0.0,
                                  np.asarray(cons.T @ T),
                                  np.asarray(cons.T @ lam))
    dJdkq = cl[pv][0]                            # -(-lam dR/dk) = +cot
    opt.zero_grad()
    kq_t.backward(torch.tensor(dJdkq))
    opt.step()
    if it % 25 == 0:
        rmse = float(np.sqrt(((kq_np - kq_star(gpx)) ** 2).mean()))
        print(f"it {it:4d}: J={J:.4e} kq-RMSE={rmse:.4e}", flush=True)

rmse = float(np.sqrt(((kq_np - kq_star(gpx)) ** 2).mean()))
print(f"[result] final J={J:.3e} closure RMSE={rmse:.4e} "
      f"(field scale 0.05)", flush=True)
