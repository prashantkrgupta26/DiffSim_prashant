"""NeuralSDF spec N4 gates 1-2 (against the AnalyticINRProxy)."""
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
from diffsim.geometry.oracle import admissibility
from diffsim.geometry.inr_proxy import AnalyticINRProxy
from diffsim.sbm.surrogate import (classify_lambda, extract_surrogate,
                                   GeometryData)
from diffsim.sbm.poisson import SBMPoisson

pytestmark = pytest.mark.geometry

CTR, R = (0.5, 0.5), 0.3
KAPPA = 1.0
U = lambda x: np.sin(np.pi * x[:, 0]) * np.cos(np.pi * x[:, 1])
F = lambda x: 2 * np.pi ** 2 * U(x)


def _band_pts(n=400, seed=3):
    rng = np.random.default_rng(seed)
    th = rng.uniform(0, 2 * np.pi, n)
    rr = R + rng.uniform(-0.08, 0.08, n)
    return np.stack([CTR[0] + rr * np.cos(th),
                     CTR[1] + rr * np.sin(th)], axis=1)


def test_gate1_admissibility(device):
    """Spec N4 gate 1: c0_hat and Newton masks at alpha=0 and at a random
    small alpha (the edit does NOT preserve eikonality — the audit is the
    contract that keeps the projector honest)."""
    proxy = AnalyticINRProxy(Sphere(CTR, R), n_modes=8)
    band = _band_pts()
    rep0 = admissibility(proxy, band)
    assert rep0["c0"] > 0.5, rep0            # base SDF: |grad| ~ 1
    d, n, ok = proxy.distance_vector(band)
    assert ok.all()
    rng = np.random.default_rng(7)
    with torch.no_grad():
        proxy.alpha += torch.tensor(rng.standard_normal(8) * 0.02)
    rep1 = admissibility(proxy, band)
    assert rep1["c0"] > 0.3, rep1            # edited: degraded but healthy
    d1, n1, ok1 = proxy.distance_vector(band)
    assert ok1.all()
    # the edit actually moved the boundary (not a no-op test)
    assert np.abs(d1 - d).max() > 1e-4


def test_gate2_forward_equivalence(device):
    """Spec N4 gate 2: at alpha=0 the proxy IS the base sphere — the SBM
    Poisson solution must match the analytic-oracle solution."""
    from scipy.sparse.linalg import splu

    def solve(oracle):
        tree = build_uniform(4, dim=2)
        ret, _ = classify_lambda(tree, oracle, 0.0)
        sf = extract_surrogate(ret)
        mesh = build_mesh(ret, p=1)
        cons = build_constraints(mesh)
        dm = DeviceMesh.from_mesh(mesh, cons, basis_tables(1, dim=2),
                                  device)
        geo = GeometryData.evaluate(oracle, ret, sf, face_tables(1, 2))
        prob = SBMPoisson(dm, geo, sf, g_fn=U, kappa=KAPPA)
        A, b, meta = prob.assemble(F)
        u = splu(A.tocsc()).solve(b)
        return u, ret

    u_ref, ret_ref = solve(Sphere(CTR, R))
    u_pxy, ret_pxy = solve(AnalyticINRProxy(Sphere(CTR, R), n_modes=8))
    assert np.array_equal(ret_ref.keys, ret_pxy.keys)   # same carve
    scale = np.abs(u_ref).max()
    err = np.abs(u_pxy - u_ref).max()
    # same zero level set; d/n via Newton (proxy) vs analytic path (base):
    # agreement to projector tolerance, far below discretization error
    assert err < 1e-8 * scale, err


def test_gate2b_perturbed_alpha_smoke(device):
    """Perturbed-alpha forward runs and responds smoothly (O(alpha))."""
    from scipy.sparse.linalg import splu

    def solve(alpha_np):
        oracle = AnalyticINRProxy(Sphere(CTR, R), n_modes=8)
        with torch.no_grad():
            oracle.alpha += torch.tensor(alpha_np)
        tree = build_uniform(4, dim=2)
        ret, _ = classify_lambda(tree, oracle, 0.0)
        sf = extract_surrogate(ret)
        mesh = build_mesh(ret, p=1)
        cons = build_constraints(mesh)
        dm = DeviceMesh.from_mesh(mesh, cons, basis_tables(1, dim=2),
                                  device)
        geo = GeometryData.evaluate(oracle, ret, sf, face_tables(1, 2))
        prob = SBMPoisson(dm, geo, sf, g_fn=U, kappa=KAPPA)
        A, b, meta = prob.assemble(F)
        return splu(A.tocsc()).solve(b), ret

    a = np.zeros(8)
    u0, ret0 = solve(a)
    a2 = a.copy(); a2[0] = 5e-3                     # inside trust region
    u1, ret1 = solve(a2)
    assert np.array_equal(ret0.keys, ret1.keys)      # frozen classification
    du = np.abs(u1 - u0).max()
    assert 1e-12 < du < 0.05 * np.abs(u0).max(), du  # responds, smoothly


@pytest.mark.ad
def test_gate3_alpha_gradient_three_way(device):
    """Spec N4 gate 3 (the 4c ritual in alpha-space): d(probe-QoI)/dalpha
    adjoint vs central FD for every mode; torch dense twin where the
    generic path supports the proxy."""
    from scipy.sparse.linalg import splu
    from diffsim.sbm.adjoint import solve_adjoint, shape_gradient, probe_qoi

    PROBES = 0.5 + 0.12 * np.array(
        [[1.0, 0.3], [-0.7, 0.8], [0.2, -1.0]])   # inside the disk

    def forward(alpha_np):
        oracle = AnalyticINRProxy(Sphere(CTR, R), n_modes=8)
        with torch.no_grad():
            oracle.alpha += torch.tensor(alpha_np)
        tree = build_uniform(4, dim=2)
        ret, _ = classify_lambda(tree, oracle, 0.0)
        sf = extract_surrogate(ret)
        mesh = build_mesh(ret, p=1)
        cons = build_constraints(mesh)
        dm = DeviceMesh.from_mesh(mesh, cons, basis_tables(1, dim=2),
                                  device)
        geo = GeometryData.evaluate(oracle, ret, sf, face_tables(1, 2))
        prob = SBMPoisson(dm, geo, sf, g_fn=U, kappa=KAPPA)
        A, b, meta = prob.assemble(F)
        u_free = splu(A.tocsc()).solve(b)
        u_all = np.asarray(cons.T @ u_free)
        evalJ, dJdu_fn = probe_qoi(dm, PROBES, np.zeros(len(PROBES)))
        return dict(J=evalJ(u_all), A=A, meta=meta, u_all=u_all,
                    dJdu=dJdu_fn(u_all), prob=prob, oracle=oracle,
                    ret=ret)

    a0 = np.zeros(8)
    fw = forward(a0)
    lam = solve_adjoint(fw["A"], fw["dJdu"])
    shape_gradient(fw["prob"], fw["u_all"], lam, fw["oracle"], fw["meta"])
    g_adj = fw["oracle"].alpha.grad.numpy().copy()

    eps = 1e-6
    # frozen-classification check on the worst mode direction
    for k in range(8):
        for s in (+eps, -eps):
            a = a0.copy(); a[k] += s
            assert np.array_equal(forward(a)["ret"].keys, fw["ret"].keys)
    scale = max(np.abs(g_adj).max(), 1e-12)
    for k in range(8):
        ap = a0.copy(); ap[k] += eps
        am = a0.copy(); am[k] -= eps
        fd = (forward(ap)["J"] - forward(am)["J"]) / (2 * eps)
        assert abs(fd - g_adj[k]) < 1e-5 * max(abs(fd), scale), (
            k, fd, g_adj[k])


@pytest.mark.tier5
@pytest.mark.ad
def test_gates45_ns_composition_and_drag_gradient(device):
    """Spec N4 gates 4+5: steady Re=20 flow around the PROXY cylinder
    (psi0 = the test_cylinder Sphere) — Cd within the locked baseline
    band, then d(Cd)/dalpha adjoint vs FD on 3 modes."""
    import sys as _sys, os as _os
    _sys.path.insert(0, _os.path.dirname(__file__))
    import test_ns_shape_gradient as tns
    from diffsim.sbm.ns_shape import drag_shape_gradient
    from diffsim.sbm.vector import surrogate_traction

    R_CYL = 0.07
    C_CYL = (0.3, 0.5)

    def steady(alpha_np):
        # mirror tns._steady but with the proxy oracle
        import scipy.sparse as sp
        from scipy.sparse.linalg import splu
        oracle = AnalyticINRProxy(Sphere(C_CYL, R_CYL), n_modes=8,
                                  bump_sigma_rel=0.8)
        with torch.no_grad():
            oracle.alpha += torch.tensor(alpha_np)
        st = tns._steady.__wrapped__ if hasattr(tns._steady, "__wrapped__") \
            else None
        # inline (tns._steady hardcodes Sphere): reuse its body via a
        # local copy driven by our oracle
        ndof, dim = 3, 2
        tree = build_uniform(5, dim=2)
        ret, _ = classify_lambda(tree, oracle, 0.5, domain="outside")
        sf = extract_surrogate(ret)
        mesh = build_mesh(ret, p=1)
        cons = build_constraints(mesh)
        dm = DeviceMesh.from_mesh(mesh, cons, basis_tables(1, dim=2),
                                  device)
        geo = GeometryData.evaluate(oracle, ret, sf, face_tables(1, 2),
                                    domain="outside")
        from diffsim.api.ns_bricks import assemble_linear_ns
        from diffsim.sbm.vector import sbm_vector_dirichlet
        from diffsim.physics.poisson import gauss_points
        T = cons.T.tocsr()
        T_vec = sp.kron(T, sp.identity(ndof, format="csr"), format="csr")
        nfree = T.shape[1]
        coords = mesh.node_coords[cons.free_nodes]
        xq = gauss_points(mesh, dm.tables_by_p)
        on = lambda v, c: np.abs(coords[:, c] - v) < 1e-12
        strong = np.where(on(0.0, 0) | on(0.0, 1) | on(1.0, 1))[0]
        g_strong = np.zeros((len(strong), 2))
        g_strong[np.abs(coords[strong, 0]) < 1e-12, 0] = 1.0

        def gp_field(node_vec):
            full = np.asarray(T @ node_vec)
            aq, dq = {}, {}
            for pv in dm.bins:
                tb = dm.tables_by_p[pv]
                conn = mesh.conn_of[pv]
                vals = full[conn]
                aq[pv] = np.einsum("qa,ead->eqd", tb.N,
                                   vals).reshape(-1, dim)
                h = mesh.tree.h()[mesh.bins[pv]]
                dq[pv] = (np.einsum("qad,ead->eq", tb.dN, vals)
                          * (2.0 / h)[:, None]).reshape(-1)
            return aq, dq

        x = np.zeros(nfree * ndof)
        prev = None
        pin = int(np.argmax(coords[:, 0] + coords[:, 1]))
        strong_rows = np.concatenate(
            [np.array([i * ndof + c for i in strong for c in range(dim)],
                      np.int64), np.array([pin * ndof + dim], np.int64)])
        for step in range(160):
            u_node = x.reshape(nfree, ndof)[:, :dim]
            aq, dq = gp_field(u_node)
            fq = {pv: np.zeros_like(aq[pv]) for pv in xq}
            A, b = assemble_linear_ns(dm, aq, dq, fq, tns.NU, sigma=0.0)
            Af, bf = sbm_vector_dirichlet(
                dm, sf, geo, lambda y: np.zeros((len(y), 2)), tns.NU,
                ndof, alpha=tns.ALPHA)
            A = (A + T_vec.T @ Af @ T_vec).tolil()
            b = b + np.asarray(T_vec.T @ bf)
            for k, i in enumerate(strong):
                for c in range(dim):
                    r = i * ndof + c
                    A.rows[r] = [int(r)]
                    A.data[r] = [1.0]
                    b[r] = g_strong[k, c]
            rp = pin * ndof + dim
            A.rows[rp] = [rp]
            A.data[rp] = [1.0]
            b[rp] = 0.0
            A = A.tocsr()
            x = splu(A.tocsc()).solve(b)
            u_new = x.reshape(nfree, ndof)[:, :dim]
            if prev is not None and np.abs(u_new - prev).max() < 1e-10 \
                    and step > 3:
                break
            prev = u_new.copy()
        x_full = np.asarray(T_vec @ x)
        F_ = surrogate_traction(dm, sf, geo, x_full, tns.NU, ndof)
        return dict(dm=dm, sf=sf, geo=geo, oracle=oracle, A=A,
                    x_full=x_full, strong_rows=strong_rows, F=F_, ret=ret)

    a0 = np.zeros(8)
    base = steady(a0)
    cd = base["F"][0] / (0.5 * 1.0 ** 2 * 2 * R_CYL)
    # gate 4: composition sane (the smoke test's band, sigma=0 steady)
    assert base["F"][0] > 0 and 1.2 < cd < 4.0, cd

    # gate 5: d(drag)/dalpha vs FD, 3 modes
    F0 = drag_shape_gradient(base["dm"], base["sf"], base["geo"],
                             base["oracle"], base["A"], base["x_full"],
                             tns.NU, tns.ALPHA, base["strong_rows"],
                             ndof=3, direction=0, fixed_point=True)
    g_adj = base["oracle"].alpha.grad.numpy().copy()
    eps = 1e-6
    scale = max(np.abs(g_adj).max(), 1e-12)
    for k in (0, 3, 6):
        ap = a0.copy(); ap[k] += eps
        am = a0.copy(); am[k] -= eps
        sp_, sm_ = steady(ap), steady(am)
        assert np.array_equal(sp_["ret"].keys, base["ret"].keys)
        fd = (sp_["F"][0] - sm_["F"][0]) / (2 * eps)
        assert abs(fd - g_adj[k]) < 5e-4 * max(abs(fd), scale), (
            k, fd, g_adj[k])
