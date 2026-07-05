"""M1b Task 8d: immersed-sphere flow smoke — the first 3-D SBM+NS
composition, enabled by the max_unroll=0 kernel mitigation (findings 6:
compile 79 min -> 1.3 s). Re = 100 (steady axisymmetric regime), coarse
level-4 CI variant: pipeline + traction correctness, not accuracy. The
Re=300 configuration (planar-symmetric shedding) is the documented nightly
run."""
import numpy as np
import pytest
import scipy.sparse as sp
from scipy.sparse.linalg import splu

from diffsim.octree.build import build_uniform
from diffsim.mesh.nodes import build_mesh
from diffsim.mesh.constraints import build_constraints
from diffsim.mesh.basis import basis_tables
from diffsim.mesh.faces import face_tables
from diffsim.assembly.operators import DeviceMesh
from diffsim.geometry.csg import Sphere
from diffsim.sbm.surrogate import classify_lambda, extract_surrogate, GeometryData
from diffsim.sbm.vector import sbm_vector_dirichlet, surrogate_traction
from diffsim.api.ns_bricks import assemble_linear_ns
from diffsim.physics.poisson import gauss_points

pytestmark = pytest.mark.tier5

R, CTR, U_IN = 0.12, (0.35, 0.5, 0.5), 1.0
NU = 2 * U_IN * R / 100.0                       # Re_D = 100


def test_sphere_re100_smoke(device):
    level, dim, dt = 4, 3, 0.05
    ndof = dim + 1
    oracle = Sphere(CTR, R)
    tree = build_uniform(level, dim=dim)
    ret, _ = classify_lambda(tree, oracle, 0.5, domain="outside")
    sf = extract_surrogate(ret)
    mesh = build_mesh(ret, p=1)
    cons = build_constraints(mesh)
    dm = DeviceMesh.from_mesh(mesh, cons, basis_tables(1, dim=dim), device)
    geo = GeometryData.evaluate(oracle, ret, sf, face_tables(1, dim),
                                domain="outside")
    T = cons.T.tocsr()
    T_vec = sp.kron(T, sp.identity(ndof, format="csr"), format="csr")
    nfree = T.shape[1]
    coords = mesh.node_coords[cons.free_nodes]
    xq = gauss_points(mesh, dm.tables_by_p)
    on = lambda v, c: np.abs(coords[:, c] - v) < 1e-12
    # inflow + lateral walls strong; outflow (x=1) free
    strong = np.where(on(0.0, 0) | on(0.0, 1) | on(1.0, 1)
                      | on(0.0, 2) | on(1.0, 2))[0]
    g_strong = np.zeros((len(strong), dim))
    g_strong[np.abs(coords[strong, 0]) < 1e-12, 0] = U_IN
    Af, bf = sbm_vector_dirichlet(
        dm, sf, geo, lambda y: np.zeros((len(y), dim)), NU, ndof)
    Af_c = (T_vec.T @ Af @ T_vec).tocsr()
    bf_c = np.asarray(T_vec.T @ bf)

    def gp_field(u_node):
        full = np.asarray(T @ u_node)
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
    sigma = 1.0 / dt
    qref = 0.5 * U_IN ** 2 * np.pi * R ** 2      # frontal area of a sphere
    for step in range(1, 81):
        u_node = x.reshape(nfree, ndof)[:, :dim]
        aq, dq = gp_field(u_node)
        fq = {pv: aq[pv] / dt for pv in xq}
        A, b = assemble_linear_ns(dm, aq, dq, fq, NU, sigma=sigma)
        A = (A + Af_c).tolil()
        b = b + bf_c
        for k, i in enumerate(strong):
            for c in range(dim):
                r = i * ndof + c
                A.rows[r] = [int(r)]
                A.data[r] = [1.0]
                b[r] = g_strong[k, c]
        pin = int(np.argmax(coords.sum(1))) * ndof + dim
        A.rows[pin] = [pin]
        A.data[pin] = [1.0]
        b[pin] = 0.0
        x_new = splu(A.tocsr().tocsc()).solve(b)
        rate = np.abs(x_new - x).max() / dt
        x = x_new
        if step > 10 and rate < 1e-2:
            break
    F = surrogate_traction(dm, sf, geo, np.asarray(T_vec @ x), NU, ndof)
    cd = F[0] / qref
    cl = np.sqrt(F[1] ** 2 + F[2] ** 2) / qref
    print(f"sphere Re=100 L{level}: Cd = {cd:.3f}  |C_lat| = {cl:.4f}  "
          f"steps = {step}")
    # smoke asserts (MEASURED: Cd = 0.381, |C_lat| = 3e-5, steady in 56):
    # the pipeline works — steady, drag downstream, axisymmetric to 1e-4 —
    # but at D/h = 3.8 the boundary layer is unresolvable and Cd is
    # under-predicted ~3x vs the unbounded literature (~1.1 at Re=100);
    # our own band-study rule (~15 cells across the feature) quantifies
    # exactly this preasymptotic regime. This lock is a PIPELINE regression
    # guard; the physics run is the level-6 nightly config.
    assert step < 80, "no steady state"
    assert F[0] > 0.0
    assert cl < 0.05 * cd, (cl, cd)                # axisymmetric wake
    assert 0.25 < cd < 1.5, cd                     # measured 0.381
