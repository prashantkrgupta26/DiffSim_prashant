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
