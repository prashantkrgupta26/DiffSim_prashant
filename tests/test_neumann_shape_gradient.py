"""Gate for the NEUMANN branch of shape_gradient — previously ungated
anywhere (external evaluation finding 1: the taped Neumann residual
omitted the beta_neumann penalty terms, so for beta != 0 the reported
gradient belonged to a different functional than the forward solve;
CONFIRMED and fixed). Parametrized over beta = 0 AND beta != 0 so the
omission class can never return silently."""
import numpy as np
import pytest

from diffsim.octree.build import build_uniform
from diffsim.mesh.nodes import build_mesh
from diffsim.mesh.constraints import build_constraints
from diffsim.mesh.basis import basis_tables
from diffsim.mesh.faces import face_tables
from diffsim.assembly.operators import DeviceMesh
from diffsim.geometry.csg import Sphere
from diffsim.sbm.surrogate import (classify_lambda, extract_surrogate,
                                   GeometryData)
from diffsim.sbm.poisson import SBMPoisson
from diffsim.sbm.adjoint import solve_adjoint, shape_gradient, probe_qoi

pytestmark = pytest.mark.ad

R0 = 0.3
CTR0 = (0.5, 0.5)
PROBES = np.array([[0.15, 0.2], [0.85, 0.75], [0.2, 0.85]])


def _q(y):
    # smooth flux data, evaluated at mapped points y = x + d: its
    # center-dependence flows through the d chain
    return np.sin(2.0 * y[:, 0]) + 0.5 * y[:, 1]


def _forward(theta, beta, device):
    cx, cy, r = [float(v) for v in theta]
    oracle = Sphere((cx, cy), r)
    tree = build_uniform(4, dim=2)
    ret, _ = classify_lambda(tree, oracle, 1.0, domain="outside")
    sf = extract_surrogate(ret)
    mesh = build_mesh(ret, p=1)
    cons = build_constraints(mesh)
    dm = DeviceMesh.from_mesh(mesh, cons, basis_tables(1, dim=2), device)
    geo = GeometryData.evaluate(oracle, ret, sf, face_tables(1, 2),
                                domain="outside")
    prob = SBMPoisson(dm, geo=None, sf=None, neumann=(sf, geo, _q),
                      beta_neumann=beta)
    from scipy.sparse.linalg import splu
    A, b, meta = prob.assemble(lambda x: np.ones(len(x)),
                               g_outer_fn=lambda x: np.zeros(len(x)))
    u_free = splu(A.tocsc()).solve(b)
    u_all = np.asarray(dm.constraints.T @ u_free)
    evalJ, dJdu_fn = probe_qoi(dm, PROBES, np.zeros(len(PROBES)))
    return dict(J=evalJ(u_all), A=A, meta=meta, u_all=u_all,
                dJdu=dJdu_fn(u_all), prob=prob, oracle=oracle, ret=ret)


THETA0 = np.array([CTR0[0], CTR0[1], R0])


@pytest.mark.parametrize("beta", [0.0, 0.5])
def test_neumann_shape_gradient_adjoint_vs_fd(beta, device):
    fw = _forward(THETA0, beta, device)
    lam = solve_adjoint(fw["A"], fw["dJdu"])
    shape_gradient(fw["prob"], fw["u_all"], lam, fw["oracle"], fw["meta"])
    g_adj = np.concatenate([fw["oracle"].center.grad.numpy(),
                            [float(fw["oracle"].radius.grad)]])
    eps = 1e-6
    # frozen-classification trust region
    for i in range(3):
        for s in (+eps, -eps):
            th = THETA0.copy(); th[i] += s
            assert np.array_equal(_forward(th, beta, device)["ret"].keys,
                                  fw["ret"].keys)
    scale = max(np.abs(g_adj).max(), 1e-12)
    for i in range(3):
        tp = THETA0.copy(); tp[i] += eps
        tm = THETA0.copy(); tm[i] -= eps
        fd = (_forward(tp, beta, device)["J"]
              - _forward(tm, beta, device)["J"]) / (2 * eps)
        assert abs(fd - g_adj[i]) < 1e-5 * max(abs(fd), scale), (
            beta, i, fd, g_adj[i])
