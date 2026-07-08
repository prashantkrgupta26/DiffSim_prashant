"""M4 adaptivity opener: spinodal CH with 2 re-mesh epochs; measure the
mass drift across the M3 P-transfer (interpolatory)."""
import numpy as np, time
from diffsim.octree.build import build_uniform, refine_elements
from diffsim.octree.balance import balance2to1
from diffsim.mesh.nodes import build_mesh
from diffsim.mesh.constraints import build_constraints
from diffsim.mesh.basis import basis_tables
from diffsim.assembly.operators import DeviceMesh
from diffsim.physics.cahn_hilliard import CahnHilliardStepper
from diffsim.adaptivity.transfer import transfer_operator

def make(tree, device="cuda:0"):
    mesh = build_mesh(tree, p=1)
    cons = build_constraints(mesh)
    dm = DeviceMesh.from_mesh(mesh, cons, basis_tables(1, dim=2), device)
    return mesh, cons, dm

def mass(st, dm, mesh, cf):
    v, _ = st._gp_scalar(cf)
    m = 0.0
    for pv, b in dm.bins.items():
        h = mesh.tree.h()[mesh.bins[pv]]
        ne = len(mesh.conn_of[pv])
        wq = np.tile(dm.tables_by_p[pv].w, ne) * np.repeat((h/2)**2, b["nqp"])
        m += float((wq * v[pv]).sum())
    return m

M, kap, dt = 1.0, 5e-4, 0.02
tree = build_uniform(5, dim=2)
mesh, cons, dm = make(tree)
st = CahnHilliardStepper(dm, M, kap, dt, order=1)
rng = np.random.default_rng(3)
st.set_initial(lambda x: 0.05 * rng.standard_normal(len(x)))
for _ in range(8):
    c, mu = st.step()
m0 = mass(st, dm, mesh, c)
for epoch in range(2):
    # marker: elements whose nodal |c| spread is large (interface band)
    conn = mesh.conn_of[1]
    cfull = np.asarray(cons.T @ c)
    spread = cfull[conn].max(1) - cfull[conn].min(1)
    mark = np.zeros(len(tree), bool)
    mark[mesh.bins[1][spread > 0.5]] = True
    t0 = time.time()
    tree2 = balance2to1(refine_elements(tree, mark))
    mesh2, cons2, dm2 = make(tree2)
    P = transfer_operator(mesh, cons, mesh2)
    t_epoch = time.time() - t0
    c2_nodes = P @ c
    mu2_nodes = P @ mu
    # free vectors on the new mesh (hanging constraints possible now):
    # least-squares restriction via T^T (T columns orthonormal-ish? no —
    # use lstsq-free approach: free nodes subset of nodes: pick rows)
    # build_mesh free nodes: cons2.free_nodes indexes nodes
    c2 = c2_nodes[cons2.free_nodes]
    mu2 = mu2_nodes[cons2.free_nodes]
    st2 = CahnHilliardStepper(dm2, M, kap, dt, order=1)
    st2.set_initial(lambda x: np.zeros(len(x)))
    st2.x[0::2] = c2; st2.x[1::2] = mu2
    st2.hist = [c2.copy(), c2.copy()]
    m_after = mass(st2, dm2, mesh2, c2)
    print(f"epoch {epoch}: +{mark.sum()} refined -> {len(tree2)} elems, "
          f"epoch-cost {t_epoch:.2f}s, mass drift {m_after - m0:+.3e}",
          flush=True)
    for _ in range(5):
        c, mu = st2.step()
    tree, mesh, cons, dm, st = tree2, mesh2, cons2, dm2, st2
    m0 = mass(st, dm, mesh, c)
print(f"final: {len(tree)} elems, c in [{c.min():.2f},{c.max():.2f}]",
      flush=True)
