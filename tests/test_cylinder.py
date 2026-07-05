"""M1b Task 8b scaffolding: steady flow past an immersed cylinder at
Re = 2 U r/nu = 20 (symmetric wake regime) — the first SBM + NS composition:
strong inflow/walls, backflow-stabilized outflow, vector SBM no-slip on the
cylinder, surrogate_traction drag/lift. Smoke-level asserts + the measured
Cd recorded for the baseline lock (full confined-cylinder config vs
literature goes to the nightly benchmark)."""
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
from diffsim.solvers.timestepping import History

pytestmark = pytest.mark.tier5

R = 0.07
CTR = (0.3, 0.5)
U_IN = 1.0
NU = 2 * U_IN * R / 20.0                     # Re_diameter = 20


def test_cylinder_re20_steady_smoke(device):
    level, ndof, dim = 5, 3, 2
    dt = 0.05
    oracle = Sphere(CTR, R)
    tree = build_uniform(level, dim=2)
    ret, _ = classify_lambda(tree, oracle, 0.5, domain="outside")
    sf = extract_surrogate(ret)
    mesh = build_mesh(ret, p=1)
    cons = build_constraints(mesh)
    dm = DeviceMesh.from_mesh(mesh, cons, basis_tables(1, dim=2), device)
    geo = GeometryData.evaluate(oracle, ret, sf, face_tables(1, 2),
                                domain="outside")
    T = dm.constraints.T.tocsr()
    T_vec = sp.kron(T, sp.identity(ndof, format="csr"), format="csr")
    nfree = T.shape[1]
    coords = mesh.node_coords[cons.free_nodes]
    xq = gauss_points(mesh, dm.tables_by_p)

    # boundary sets: inflow (x=0) + walls (y=0,1) strong; outflow (x=1) free
    on = lambda v, c: np.abs(coords[:, c] - v) < 1e-12
    strong = np.where(on(0.0, 0) | on(0.0, 1) | on(1.0, 1))[0]
    g_strong = np.zeros((len(strong), 2))
    g_strong[np.abs(coords[strong, 0]) < 1e-12, 0] = U_IN   # inflow
    # walls: free-slip via u_y = 0 only? keep simple: no-slip walls =>
    # confined cylinder (blockage 2R/H = 0.14); record as config
    hist = History()
    hist.rotate(np.zeros(nfree * ndof))

    def gp_field(node_vec):
        full = np.asarray(T @ node_vec)
        aq, dq = {}, {}
        for pv in dm.bins:
            tb = dm.tables_by_p[pv]
            conn = mesh.conn_of[pv]
            vals = full[conn]
            aq[pv] = np.einsum("qa,ead->eqd", tb.N, vals).reshape(-1, dim)
            h = mesh.tree.h()[mesh.bins[pv]]
            dq[pv] = (np.einsum("qad,ead->eq", tb.dN, vals)
                      * (2.0 / h)[:, None]).reshape(-1)
        return aq, dq

    x = np.zeros(nfree * ndof)
    prev_u = None
    sigma = 1.0 / dt                               # BDF1 pseudo-time
    cd_hist = []
    for step in range(160):
        u_node = x.reshape(nfree, ndof)[:, :dim]
        aq, dq = gp_field(u_node)
        # face-GP advecting field for backflow (interp at face GPs ~ use
        # node values at mapped points is overkill; the outflow x=1 is a
        # STRONG box face here — backflow enters via the SBM face term only,
        # zero-field-safe): supply the domain-outward normal projection of
        # the current u at face GPs via nearest-element interp: skip (SBM
        # faces are at the cylinder where u ~ 0) — a_face=None
        # BDF1 pseudo-time RHS: f_eff = u^n/dt, and aq already holds the
        # GP values of u^n (the advecting field is the previous solution)
        fq = {pv: aq[pv] / dt for pv in xq}
        A, b = assemble_linear_ns(dm, aq, dq, fq, NU, sigma=sigma)
        Af, bf = sbm_vector_dirichlet(
            dm, sf, geo, lambda y: np.zeros((len(y), 2)), NU, ndof)
        A = (A + T_vec.T @ Af @ T_vec).tolil()
        b = b + np.asarray(T_vec.T @ bf)
        for k, i in enumerate(strong):
            for c in range(dim):
                r = i * ndof + c
                A.rows[r] = [int(r)]
                A.data[r] = [1.0]
                b[r] = g_strong[k, c]
        # pressure pin at an interior retained node far from the cylinder
        pin = int(np.argmax(coords[:, 0] + coords[:, 1]))
        rp = pin * ndof + dim
        A.rows[rp] = [rp]
        A.data[rp] = [1.0]
        b[rp] = 0.0
        x = splu(A.tocsr().tocsc()).solve(b)
        u_new = x.reshape(nfree, ndof)[:, :dim]
        if prev_u is not None:
            rate = np.abs(u_new - prev_u).max() / dt
            x_full = np.asarray(T_vec @ x)
            F = surrogate_traction(dm, sf, geo, x_full, NU, ndof)
            cd_hist.append(F[0] / (0.5 * U_IN ** 2 * 2 * R))
            if rate < 5e-3 and step > 10:
                break
        prev_u = u_new.copy()
    else:
        pytest.fail(f"no steady state; last rate {rate}")

    cd, cl = cd_hist[-1], None
    x_full = np.asarray(T_vec @ x)
    F = surrogate_traction(dm, sf, geo, x_full, NU, ndof)
    cd = F[0] / (0.5 * U_IN ** 2 * 2 * R)
    cl = F[1] / (0.5 * U_IN ** 2 * 2 * R)
    print(f"Re=20 confined cylinder: Cd = {cd:.3f}, Cl = {cl:.4f}, "
          f"steps = {step}")
    # smoke asserts (literature band for Re=20 with mild confinement is
    # Cd ~ 2.0-2.4; SBM-coarse + blockage tolerance is generous — the
    # MEASURED value gets locked in m1b_baselines at the benchmark task)
    assert F[0] > 0.0, F                            # drag pushes downstream
    assert 1.2 < cd < 4.0, cd
    assert abs(cl) < 0.3 * cd, (cl, cd)             # symmetric wake
