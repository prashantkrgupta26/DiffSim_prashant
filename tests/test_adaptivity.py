"""M3 rung 1 gates: transfer operator + adjoint across a scheduled
re-carve (spec 2026-07-07, task #20)."""
import numpy as np
import pytest
import scipy.sparse as sp
from scipy.sparse.linalg import splu

from diffsim.octree.build import build_uniform
from diffsim.mesh.nodes import build_mesh
from diffsim.mesh.constraints import build_constraints
from diffsim.mesh.basis import basis_tables
from diffsim.assembly.operators import DeviceMesh
from diffsim.geometry.csg import Sphere
from diffsim.sbm.surrogate import classify_lambda
from diffsim.adaptivity.transfer import transfer_operator
from diffsim.physics.scalar_transport import assemble_scalar_ad
from diffsim.physics.poisson import gauss_points

pytestmark = pytest.mark.tier2


def _carve(r, level=5, device="cuda:0"):
    tree = build_uniform(level, dim=2)
    ret, _ = classify_lambda(tree, Sphere((0.5, 0.5), r), 1.0,
                             domain="outside")
    mesh = build_mesh(ret, p=1)
    cons = build_constraints(mesh)
    dm = DeviceMesh.from_mesh(mesh, cons, basis_tables(1, dim=2), device)
    return mesh, cons, dm


def test_transfer_dot_identity_and_accuracy(device):
    mA, cA, _ = _carve(0.27)
    mB, cB, _ = _carve(0.253)
    P = transfer_operator(mA, cA, mB)
    rng = np.random.default_rng(0)
    v = rng.standard_normal(P.shape[1])
    w = rng.standard_normal(P.shape[0])
    lhs, rhs = float(w @ (P @ v)), float(v @ (P.T @ w))
    assert abs(lhs - rhs) < 1e-12 * max(abs(lhs), 1.0)          # (a)
    u = lambda x: np.sin(np.pi * x[:, 0]) * np.cos(np.pi * x[:, 1])
    uB = P @ u(mA.node_coords)          # no-hanging: free == nodes
    err = np.abs(uB - u(mB.node_coords))
    kA = set(map(tuple, np.round(mA.node_coords * 2**20).astype(int)))
    shared = np.array([tuple(k) in kA for k in
                       np.round(mB.node_coords * 2**20).astype(int)])
    assert err[shared].max() < 1e-12                            # (b) int
    assert err.max() < 3.5 * (1.0 / 32)                         # (b) strip


def _bdf1_ops(dm, mesh, cons, kappa, dt, device):
    xq = gauss_points(mesh, dm.tables_by_p)
    pv = 1
    ngp = len(xq[pv])
    aq = {pv: np.tile([1.0, 0.3], (ngp, 1))}
    z = {pv: np.zeros(ngp)}
    A, _ = assemble_scalar_ad(dm, aq, z, kappa, sigma=1.0 / dt, supg=0.0)
    M, _ = assemble_scalar_ad(dm, {pv: np.zeros((ngp, 2))}, z,
                              1e-30, sigma=1.0, supg=0.0)
    coords = mesh.node_coords[cons.free_nodes]
    bdry = np.zeros(len(coords), bool)
    for c in range(2):
        bdry |= (np.abs(coords[:, c]) < 1e-12) | \
                (np.abs(coords[:, c] - 1) < 1e-12)
    dirn = np.where(bdry)[0]
    A = A.tolil()
    for i in dirn:
        A.rows[i] = [int(i)]; A.data[i] = [1.0]
    return A.tocsr(), M.tocsr(), dirn


def test_rung1_exit_adjoint_across_recarve(device):
    """RUNG 1 EXIT: dJ/d(T0) through 3 BDF1 steps on carve A, transfer P,
    3 steps on carve B — adjoint (with P^T) vs FD at 1e-8-class."""
    kappa, dt = 0.05, 0.05
    mA, cA, dmA = _carve(0.27, device=device)
    mB, cB, dmB = _carve(0.253, device=device)
    P = transfer_operator(mA, cA, mB)
    A_A, M_A, dirA = _bdf1_ops(dmA, mA, cA, kappa, dt, device)
    A_B, M_B, dirB = _bdf1_ops(dmB, mB, cB, kappa, dt, device)
    luA, luB = splu(A_A.tocsc()), splu(A_B.tocsc())
    luAT, luBT = splu(A_A.tocsc().T), splu(A_B.tocsc().T)
    rng = np.random.default_rng(1)
    T0 = rng.standard_normal(A_A.shape[0]) * 0.1
    T0[dirA] = 0.0

    def forward(T0v):
        T = T0v.copy()
        for _ in range(3):
            r = M_A @ T / dt
            r[dirA] = 0.0
            T = luA.solve(r)
        T = P @ T
        T[dirB] = 0.0
        for _ in range(3):
            r = M_B @ T / dt
            r[dirB] = 0.0
            T = luB.solve(r)
        return 0.5 * float(T @ T), T

    J0, Tf = forward(T0)
    # reverse
    lam = Tf.copy()                       # dJ/dT_final
    for _ in range(3):
        z = luBT.solve(lam)
        z[dirB] = 0.0
        lam = M_B.T @ z / dt
    lam[dirB] = 0.0
    lam = P.T @ lam                       # ACROSS the re-carve
    for _ in range(3):
        z = luAT.solve(lam)
        z[dirA] = 0.0
        lam = M_A.T @ z / dt
    g = lam                               # dJ/dT0
    eps = 1e-6
    inner = np.setdiff1d(np.arange(len(T0)), dirA)
    for i in rng.choice(inner, 3, replace=False):
        Tp = T0.copy(); Tp[i] += eps
        Tm = T0.copy(); Tm[i] -= eps
        fd = (forward(Tp)[0] - forward(Tm)[0]) / (2 * eps)
        rel = abs(g[i] - fd) / max(abs(fd), 1e-12)
        assert rel < 1e-7, (int(i), rel, g[i], fd)


def test_rung2_beyond_trust_region(device):
    """RUNG 2 GATE: recover alpha* at ~2.3x the epoch trust region on
    the analytic-proxy sphere Poisson config — plain single-epoch GN
    must stall/fail; continuation must converge (<20% |alpha*|)."""
    from diffsim.geometry.inr_proxy import AnalyticINRProxy
    from diffsim.sbm.surrogate import (classify_lambda, extract_surrogate,
                                       GeometryData)
    from diffsim.sbm.poisson import SBMPoisson
    from diffsim.mesh.faces import face_tables
    from diffsim.mesh.pointeval import point_eval_weights
    from diffsim.adaptivity.continuation import continuation_recover

    LEVEL, K = 4, 2
    H = 1.0 / 2 ** LEVEL
    TRUST = 0.35 * H                                 # 0.0219

    import torch

    def epoch_fn(alpha):
        o = AnalyticINRProxy(Sphere((0.5, 0.5), 0.27), n_modes=K)
        o.alpha = torch.tensor(np.asarray(alpha, float))
        tree = build_uniform(LEVEL, dim=2)
        ret, _ = classify_lambda(tree, o, 1.0, domain="outside")
        sf = extract_surrogate(ret)
        mesh = build_mesh(ret, p=1)
        cons = build_constraints(mesh)
        dm = DeviceMesh.from_mesh(mesh, cons, basis_tables(1, dim=2),
                                  device)
        py, px = np.meshgrid(np.linspace(0.15, 0.85, 4),
                             np.linspace(0.15, 0.85, 4))
        pts = np.column_stack([px.ravel(), py.ravel()])
        keep = np.linalg.norm(pts - 0.5, axis=1) > 0.42
        W = point_eval_weights(mesh, pts[keep])
        return dict(o=o, ret=ret, sf=sf, mesh=mesh, cons=cons, dm=dm,
                    W=W)

    def solve_probes(E, alpha):
        import torch as _t
        E["o"].alpha = _t.tensor(np.asarray(alpha, float))
        geo = GeometryData.evaluate(E["o"], E["ret"], E["sf"],
                                    face_tables(1, 2), domain="outside")
        prob = SBMPoisson(E["dm"], geo=geo, sf=E["sf"],
                          g_fn=lambda x: np.ones(len(x)), kappa=1.0,
                          alpha=20.0)
        u = prob.solve(f_fn=lambda x: np.zeros(len(x)),
                       g_outer_fn=lambda x: np.zeros(len(x)))
        return E["W"] @ u

    # fixed target DATA from alpha* (its own carve — data, not epochs)
    astar = np.array([0.05, -0.03])                  # |a*| = 2.7x TRUST
    tgt_E = epoch_fn(astar)
    target = solve_probes(tgt_E, astar)

    def resid_fn(E, alpha):
        return solve_probes(E, alpha) - target

    def jac_fn(E, alpha, r0):
        eps = 1e-6
        Jc = np.zeros((len(r0), K))
        for k in range(K):
            a2 = np.asarray(alpha, float).copy(); a2[k] += eps
            Jc[:, k] = (solve_probes(E, a2) - target - r0) / eps
        return Jc

    # PLAIN single-epoch GN (the M2-D failure mode): epoch fixed at 0
    E0 = epoch_fn(np.zeros(K))
    a = np.zeros(K)
    for _ in range(6):
        r = resid_fn(E0, a)
        Jc = jac_fn(E0, a, r)
        lamb = 1e-3 * np.trace(Jc.T @ Jc) / K
        a = a + np.linalg.solve(Jc.T @ Jc + lamb * np.eye(K),
                                -(Jc.T @ r))
    err_plain = np.linalg.norm(a - astar)

    # CONTINUATION
    a_c, hist = continuation_recover(epoch_fn, resid_fn, jac_fn,
                                     np.zeros(K), TRUST, n_epochs=8,
                                     inner_iters=4, verbose=False)
    err_cont = np.linalg.norm(a_c - astar)
    print(f"beyond-trust: plain err={err_plain:.4f} "
          f"continuation err={err_cont:.4f} (|a*|={np.linalg.norm(astar):.4f})")
    assert err_cont < 0.2 * np.linalg.norm(astar), (err_cont, err_plain)
    assert err_cont < err_plain * 0.5                # continuation wins
