"""Tier-AD gradient checks (M1a Task 10; spec S9.1 Tier AD): three-way gate
— custom adjoint (Tier-2 VJP #1 + tape sweep) vs torch dense twin (unrolled
autograd through the IFT projection) vs central finite differences — plus
kappa gradient, frozen-classification trust region, and per-backend checks.

Gradients are exact for the DISCRETE system at the frozen epoch: FD re-runs
the full pipeline, so classification must not flip inside the FD stencil
(asserted). Tolerances per spec S9.2: rel < 1e-6 vs FD (FP64)."""
import numpy as np
import pytest
import torch
from diffsim.octree.build import build_uniform
from diffsim.mesh.nodes import build_mesh
from diffsim.mesh.constraints import build_constraints
from diffsim.mesh.basis import basis_tables
from diffsim.mesh.faces import face_tables
from diffsim.assembly.operators import DeviceMesh
from diffsim.geometry.csg import Sphere
from diffsim.sbm.surrogate import classify_lambda, extract_surrogate, GeometryData
from diffsim.sbm.poisson import SBMPoisson
from diffsim.sbm.adjoint import (solve_adjoint, shape_gradient, kappa_gradient,
                                 probe_qoi, volume_qoi)

pytestmark = pytest.mark.ad

U = lambda x: np.sin(np.pi * x[:, 0]) * np.sin(np.pi * x[:, 1])
F = lambda x: 2 * np.pi ** 2 * U(x)
U_t = lambda x: torch.sin(np.pi * x[:, 0]) * torch.sin(np.pi * x[:, 1])
KAPPA = 1.3
PROBES = 0.5 + 0.12 * np.array(
    [[np.cos(t), np.sin(t)] for t in np.linspace(0, 2 * np.pi, 9, endpoint=False)])


def _forward(theta, device):
    """Full pipeline: theta = (cx, cy, r) -> (J, parts...)."""
    cx, cy, r = [float(v) for v in theta]
    oracle = Sphere((cx, cy), r)
    tree = build_uniform(4, dim=2)
    ret, _ = classify_lambda(tree, oracle, 0.0)
    sf = extract_surrogate(ret)
    mesh = build_mesh(ret, p=1)
    cons = build_constraints(mesh)
    dm = DeviceMesh.from_mesh(mesh, cons, basis_tables(1, dim=2), device)
    geo = GeometryData.evaluate(oracle, ret, sf, face_tables(1, 2))
    prob = SBMPoisson(dm, geo, sf, g_fn=U, kappa=KAPPA)
    A, b, meta = prob.assemble(F)
    from scipy.sparse.linalg import splu
    u_free = splu(A.tocsc()).solve(b)
    u_all = np.asarray(dm.constraints.T @ u_free)
    evalJ, dJdu_fn = probe_qoi(dm, PROBES, np.zeros(len(PROBES)))
    return dict(J=evalJ(u_all), A=A, meta=meta, u_free=u_free, u_all=u_all,
                dJdu=dJdu_fn(u_all), prob=prob, dm=dm, oracle=oracle,
                ret=ret, sf=sf)


THETA0 = np.array([0.5, 0.5, 0.3])


def _adjoint_grad(device):
    fw = _forward(THETA0, device)
    lam = solve_adjoint(fw["A"], fw["dJdu"])
    shape_gradient(fw["prob"], fw["u_all"], lam, fw["oracle"], fw["meta"],
                   g_fn_torch=U_t)
    g = np.concatenate([fw["oracle"].center.grad.numpy(),
                        [float(fw["oracle"].radius.grad)]])
    return g, fw, lam


def test_dot_product_identity(device):
    fw = _forward(THETA0, device)
    A = fw["A"]
    rng = np.random.default_rng(11)
    v = rng.standard_normal(A.shape[0])
    w = rng.standard_normal(A.shape[0])
    assert abs(w @ (A @ v) - v @ (A.T @ w)) < 1e-12 * abs(w @ (A @ v) + 1e-30)


def test_frozen_classification_trust_region(device):
    # spec S1.5: below an element crossing, the retained set is constant
    eps = 1e-6
    keys0 = _forward(THETA0, device)["ret"].keys
    for i in range(3):
        for s in (+eps, -eps):
            th = THETA0.copy()
            th[i] += s
            assert np.array_equal(_forward(th, device)["ret"].keys, keys0)


def test_shape_gradient_three_way(device):
    g_adj, fw, lam = _adjoint_grad(device)

    # --- torch dense twin (unrolled autograd incl. IFT projection) -------
    from diffsim.sbm.reference import solve_dense_torch
    from diffsim.mesh.pointeval import point_eval_weights
    oracle2 = Sphere((0.5, 0.5), 0.3)
    for p in oracle2.params:
        p.requires_grad_(True)
    dm, sf = fw["dm"], fw["sf"]
    u_free_t, aux = solve_dense_torch(dm, sf, oracle2, U_t, F, kappa=KAPPA)
    W = point_eval_weights(dm.mesh, PROBES)
    M = torch.tensor((W @ dm.constraints.T).toarray(), dtype=torch.float64)
    J_t = ((M @ u_free_t) ** 2).sum()
    J_t.backward()
    g_twin = np.concatenate([oracle2.center.grad.numpy(),
                             [float(oracle2.radius.grad)]])
    # twin sanity: same J
    assert abs(float(J_t) - fw["J"]) < 1e-9 * max(abs(fw["J"]), 1e-12)
    scale = np.abs(g_adj).max()
    assert np.abs(g_adj - g_twin).max() < 1e-8 * scale, (g_adj, g_twin)

    # --- central FD -------------------------------------------------------
    eps = 1e-6
    for i in range(3):
        tp = THETA0.copy(); tp[i] += eps
        tm = THETA0.copy(); tm[i] -= eps
        fd = (_forward(tp, device)["J"] - _forward(tm, device)["J"]) / (2 * eps)
        assert abs(fd - g_adj[i]) < 1e-6 * max(abs(fd), scale), (i, fd, g_adj[i])


def test_kappa_gradient(device):
    fw = _forward(THETA0, device)
    lam = solve_adjoint(fw["A"], fw["dJdu"])
    g_k = kappa_gradient(fw["meta"], fw["u_free"], lam)
    eps = 1e-6
    global KAPPA
    k0 = KAPPA
    try:
        vals = {}
        for s in (+eps, -eps):
            KAPPA = k0 + s
            vals[s] = _forward(THETA0, device)["J"]
    finally:
        KAPPA = k0
    fd = (vals[eps] - vals[-eps]) / (2 * eps)
    assert abs(fd - g_k) < 1e-6 * max(abs(fd), 1e-9), (fd, g_k)


def test_gradient_gridsdf_voxels(device):
    from diffsim.geometry.gridsdf import GridSDF
    n = 48
    target = Sphere((0.5, 0.5), 0.3)

    def forward(values_np):
        oracle = GridSDF(torch.tensor(values_np, dtype=torch.float64))
        tree = build_uniform(4, dim=2)
        ret, _ = classify_lambda(tree, oracle, 0.0)
        sf = extract_surrogate(ret)
        mesh = build_mesh(ret, p=1)
        cons = build_constraints(mesh)
        dm = DeviceMesh.from_mesh(mesh, cons, basis_tables(1, dim=2), device)
        geo = GeometryData.evaluate(oracle, ret, sf, face_tables(1, 2))
        prob = SBMPoisson(dm, geo, sf, g_fn=U, kappa=KAPPA)
        A, b, meta = prob.assemble(F)
        from scipy.sparse.linalg import splu
        u_free = splu(A.tocsc()).solve(b)
        u_all = np.asarray(dm.constraints.T @ u_free)
        evalJ, dJdu_fn = probe_qoi(dm, PROBES, np.zeros(len(PROBES)))
        return evalJ(u_all), (oracle, prob, u_all, dJdu_fn, A, meta)

    base = GridSDF.from_oracle(target, n=n).values.numpy().copy()
    J0, (oracle, prob, u_all, dJdu_fn, A, meta) = forward(base)
    lam = solve_adjoint(A, dJdu_fn(u_all))
    shape_gradient(prob, u_all, lam, oracle, meta, g_fn_torch=U_t)
    gv = oracle.values.grad.numpy()
    # three voxels near the boundary with the largest sensitivities
    idx = np.dstack(np.unravel_index(np.argsort(-np.abs(gv).ravel())[:3],
                                     gv.shape))[0]
    eps = 1e-6
    for (i, j) in idx:
        vp = base.copy(); vp[i, j] += eps
        vm = base.copy(); vm[i, j] -= eps
        fd = (forward(vp)[0] - forward(vm)[0]) / (2 * eps)
        assert abs(fd - gv[i, j]) < 1e-5 * max(abs(fd), np.abs(gv).max()), (
            (i, j), fd, gv[i, j])


def test_gradient_trimesh_vertices(device):
    from diffsim.geometry.trimesh import icosphere

    def forward(verts_np, tris):
        from diffsim.geometry.trimesh import TriMeshOracle
        oracle = TriMeshOracle(verts_np, tris, device=device)
        tree = build_uniform(3, dim=3)
        ret, _ = classify_lambda(tree, oracle, 0.0)
        sf = extract_surrogate(ret)
        mesh = build_mesh(ret, p=1)
        cons = build_constraints(mesh)
        dm = DeviceMesh.from_mesh(mesh, cons, basis_tables(1, dim=3), device)
        geo = GeometryData.evaluate(oracle, ret, sf, face_tables(1, 3))
        # NONLINEAR data: a linear patch makes u == g on ANY domain (the P4
        # property) => dJ/d(geometry) is identically zero — noise vs noise
        # (measured gv ~ 5e-15; the night's third degenerate-MMS care point)
        U3 = lambda x: (np.sin(np.pi * x[:, 0]) * np.sin(np.pi * x[:, 1])
                        * np.sin(np.pi * x[:, 2]))
        prob = SBMPoisson(dm, geo, sf, g_fn=U3, kappa=1.0)
        A, b, meta = prob.assemble(lambda x: 3 * np.pi ** 2 * U3(x))
        from scipy.sparse.linalg import splu
        u_free = splu(A.tocsc()).solve(b)
        u_all = np.asarray(dm.constraints.T @ u_free)
        probes = 0.5 + 0.1 * np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1.0],
                                       [-1, 0, 0]])
        evalJ, dJdu_fn = probe_qoi(dm, probes, np.zeros(4))
        return evalJ(u_all), (oracle, prob, u_all, dJdu_fn, A, meta)

    base_o = icosphere(2, (0.5, 0.5, 0.5), 0.32, device=device)
    verts0 = base_o.verts.detach().numpy().copy()
    tris = base_o.tris
    J0, (oracle, prob, u_all, dJdu_fn, A, meta) = forward(verts0, tris)
    lam = solve_adjoint(A, dJdu_fn(u_all))
    U3_t = lambda x: (torch.sin(np.pi * x[:, 0]) * torch.sin(np.pi * x[:, 1])
                      * torch.sin(np.pi * x[:, 2]))
    shape_gradient(prob, u_all, lam, oracle, meta, g_fn_torch=U3_t)
    gv = oracle.verts.grad.numpy()
    # the vertex with the largest gradient, all 3 components
    vi = int(np.argmax(np.abs(gv).sum(axis=1)))
    eps = 1e-6
    for c in range(3):
        vp = verts0.copy(); vp[vi, c] += eps
        vm = verts0.copy(); vm[vi, c] -= eps
        fd = (forward(vp, tris)[0] - forward(vm, tris)[0]) / (2 * eps)
        assert abs(fd - gv[vi, c]) < 1e-4 * max(abs(fd), np.abs(gv).max()), (
            c, fd, gv[vi, c])


def test_shape_inverse_recovers_circle(device):
    """M1a capstone (plan Task 11): recover a circle's center+radius from
    probe observations via the full epoch loop — carve -> classify -> solve
    -> adjoint -> theta step -> re-carve (spec S4.3). Same-level synthetic
    data (inverse crime is fine for the machinery demo)."""
    truth = np.array([0.52, 0.47, 0.31])
    fw_truth = None

    def forward(theta):
        return _forward(theta, device)

    fw_truth = forward(truth)
    from diffsim.mesh.pointeval import point_eval_weights
    W = point_eval_weights(fw_truth["dm"].mesh, PROBES)
    targets = np.asarray(W @ fw_truth["u_all"])

    theta = torch.tensor([0.50, 0.50, 0.25], dtype=torch.float64)
    opt = torch.optim.Adam([theta], lr=2.0e-2)
    theta.requires_grad_(True)
    J_hist = []
    for it in range(60):
        th = theta.detach().numpy()
        cx, cy, r = [float(v) for v in th]
        oracle = Sphere((cx, cy), r)
        tree = build_uniform(4, dim=2)
        ret, _ = classify_lambda(tree, oracle, 0.0)
        sf = extract_surrogate(ret)
        mesh = build_mesh(ret, p=1)
        cons = build_constraints(mesh)
        dm = DeviceMesh.from_mesh(mesh, cons, basis_tables(1, dim=2), device)
        geo = GeometryData.evaluate(oracle, ret, sf, face_tables(1, 2))
        prob = SBMPoisson(dm, geo, sf, g_fn=U, kappa=KAPPA)
        A, b, meta = prob.assemble(F)
        from scipy.sparse.linalg import splu
        u_free = splu(A.tocsc()).solve(b)
        u_all = np.asarray(dm.constraints.T @ u_free)
        evalJ, dJdu_fn = probe_qoi(dm, PROBES, targets)
        J = evalJ(u_all)
        J_hist.append(J)
        if J < 1e-14:
            break
        lam = solve_adjoint(A, dJdu_fn(u_all))
        shape_gradient(prob, u_all, lam, oracle, meta, g_fn_torch=U_t)
        g = np.concatenate([oracle.center.grad.numpy(),
                            [float(oracle.radius.grad)]])
        opt.zero_grad()
        theta.grad = torch.tensor(g, dtype=torch.float64)
        opt.step()
        with torch.no_grad():
            theta[2].clamp_(0.15, 0.45)      # keep the disk inside the box

    th = theta.detach().numpy()
    assert abs(th[2] - truth[2]) < 5e-3, (th, truth, J_hist[-1])
    assert np.linalg.norm(th[:2] - truth[:2]) < 5e-3, (th, truth)
    assert J_hist[-1] < 1e-3 * max(J_hist[0], 1e-12), J_hist[::10]
