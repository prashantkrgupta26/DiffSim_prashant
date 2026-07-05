"""M1b Task 7 gates: vector SBM Dirichlet — the P4 keystone carried to
vector fields (linear velocity through the Stokes+SBM path is machine-
exact) at k=2,3; backflow term sign/zero checks."""
import os

import numpy as np
import pytest
from scipy.sparse.linalg import splu
from diffsim.octree.build import build_uniform
from diffsim.mesh.nodes import build_mesh
from diffsim.mesh.constraints import build_constraints
from diffsim.mesh.basis import basis_tables
from diffsim.mesh.faces import face_tables
from diffsim.assembly.operators import DeviceMesh
from diffsim.geometry.csg import Sphere
from diffsim.sbm.surrogate import classify_lambda, extract_surrogate, GeometryData
from diffsim.sbm.vector import sbm_vector_dirichlet
from diffsim.api.ns_bricks import assemble_linear_ns
from diffsim.physics.poisson import gauss_points
import scipy.sparse as sp

pytestmark = pytest.mark.tier5


@pytest.mark.parametrize("dim,level,r", [
    (2, 4, 0.25),
    pytest.param(3, 3, 0.3, marks=pytest.mark.skipif(
        not os.environ.get("DIFFSIM_RUN_3D_NS"),
        reason="one-time >1h 3D NS kernel compile (m1b findings 6) — "
               "set DIFFSIM_RUN_3D_NS=1 to run")),
])
def test_vector_p4_patch(dim, level, r, device):
    """Exterior domain, LINEAR velocity field, zero pressure: the SBM
    Dirichlet on the immersed sphere + strong outer Dirichlet reproduce the
    field to machine precision through the full vector NS assembly (Stokes,
    a=0: convection/SUPG vanish on the exact solution; grad-div exact since
    div u* = 0 for a traceless linear field)."""
    ndof = dim + 1
    ctr = (0.5,) * dim
    oracle = Sphere(ctr, r)
    tree = build_uniform(level, dim=dim)
    ret, _ = classify_lambda(tree, oracle, 1.0, domain="outside")
    sf = extract_surrogate(ret)
    mesh = build_mesh(ret, p=1)
    cons = build_constraints(mesh)
    dm = DeviceMesh.from_mesh(mesh, cons, basis_tables(1, dim=dim), device)
    geo = GeometryData.evaluate(oracle, ret, sf, face_tables(1, dim),
                                domain="outside")
    # traceless linear field: div u* = 0
    G = np.zeros((dim, dim))
    G[0, 1] = 0.7
    G[1, 0] = -0.3
    if dim == 3:
        G[0, 0], G[1, 1], G[2, 2] = 0.4, -0.15, -0.25
        G[2, 0] = 0.2
    else:
        G[0, 0], G[1, 1] = 0.5, -0.5
    u_lin = lambda x: x @ G.T
    nu = 0.7

    xq = gauss_points(mesh, dm.tables_by_p)
    aq = {pv: np.zeros((len(xq[pv]), dim)) for pv in xq}
    dq = {pv: np.zeros(len(xq[pv])) for pv in xq}
    fq = {pv: np.zeros((len(xq[pv]), dim)) for pv in xq}
    A, b = assemble_linear_ns(dm, aq, dq, fq, nu, sigma=0.0)
    Af, bf = sbm_vector_dirichlet(dm, sf, geo, u_lin, nu, ndof)
    T = dm.constraints.T.tocsr()
    T_vec = sp.kron(T, sp.identity(ndof, format="csr"), format="csr")
    A = (A + T_vec.T @ Af @ T_vec).tocsr()
    b = b + np.asarray(T_vec.T @ bf)
    # strong outer Dirichlet + pressure pin
    nfree = T.shape[1]
    coords = mesh.node_coords[cons.free_nodes]
    bdry = mesh.boundary_nodes[cons.free_nodes]
    A = A.tolil()
    gb = u_lin(coords)
    for i in np.where(bdry)[0]:
        for c in range(dim):
            rr = i * ndof + c
            A.rows[rr] = [rr]
            A.data[rr] = [1.0]
            b[rr] = gb[i, c]
    A.rows[dim] = [dim]
    A.data[dim] = [1.0]
    b[dim] = 0.0
    x = splu(A.tocsr().tocsc()).solve(b)
    u = x.reshape(nfree, ndof)[:, :dim]
    # machine-exact on the retained domain (patch: all terms consistent)
    mask = oracle.classify(coords) > -1e-12
    err = np.abs(u[mask] - gb[mask]).max()
    assert err < 1e-9, err
    p_err = np.abs(x.reshape(nfree, ndof)[:, dim][mask]).max()
    assert p_err < 1e-8, p_err


def test_backflow_term_properties(device):
    dim = 2
    oracle = Sphere((0.5, 0.5), 0.25)
    tree = build_uniform(4, dim=2)
    ret, _ = classify_lambda(tree, oracle, 1.0, domain="outside")
    sf = extract_surrogate(ret)
    mesh = build_mesh(ret, p=1)
    cons = build_constraints(mesh)
    dm = DeviceMesh.from_mesh(mesh, cons, basis_tables(1, dim=2), device)
    geo = GeometryData.evaluate(oracle, ret, sf, face_tables(1, 2),
                                domain="outside")
    g = lambda y: np.zeros((len(y), 2))
    ngp = len(geo.corr)
    # zero advecting field => backflow adds nothing
    A0, _ = sbm_vector_dirichlet(dm, sf, geo, g, 1.0, 3)
    Az, _ = sbm_vector_dirichlet(dm, sf, geo, g, 1.0, 3,
                                 a_face=np.zeros((ngp, 2)))
    assert abs(A0 - Az).max() < 1e-15
    # pure INFLOW everywhere (a = -n): term engaged, SPD-positive shift
    Ain, _ = sbm_vector_dirichlet(dm, sf, geo, g, 1.0, 3, a_face=-geo.n)
    D = (Ain - A0)
    assert abs(D).max() > 1e-3                       # engaged
    rng = np.random.default_rng(3)
    v = rng.standard_normal(D.shape[0])
    assert v @ (D @ v) >= -1e-12                     # positive semi-def
    # pure OUTFLOW (a = +n): the (a.n)_- clip kills it
    Aout, _ = sbm_vector_dirichlet(dm, sf, geo, g, 1.0, 3, a_face=geo.n)
    assert abs(Aout - A0).max() < 1e-15
