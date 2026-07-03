import numpy as np
import pytest
from diffsim.octree.build import build_uniform
from diffsim.mesh.nodes import build_mesh
from diffsim.mesh.constraints import build_constraints
from diffsim.mesh.basis import basis_tables
from diffsim.assembly.operators import DeviceMesh
from diffsim.solvers.newton import NonlinearSolver
from diffsim.physics.bratu import BratuProblem

pytestmark = pytest.mark.tier3

def _bratu(device, lam=1.0, level=3, p=1):
    m = build_mesh(build_uniform(level), p=p)
    c = build_constraints(m)
    dm = DeviceMesh.from_mesh(m, c, basis_tables(p), device)
    return BratuProblem(dm, lam)

def test_newton_analytic_jacobian_converges(device):
    prob = _bratu(device)
    ns = NonlinearSolver(prob.residual, jac_action_fn=prob.jac_action,
                         snes_rtol=1e-10, snes_max_it=20)
    u, info = ns.solve(np.zeros(prob.n_free))
    assert info["converged"] and info["iters"] <= 6
    # superlinear tail: some interior contraction ratio much smaller than the first.
    # The last ratio may stagnate near krylov atol floor (h ~ 1e-12), so we check
    # the minimum over all ratios after step 0, not just the final ratio.
    h = info["fnorm_history"]
    ratios = [h[k + 1] / h[k] for k in range(len(h) - 1)]
    assert min(ratios[1:]) < 0.5 * ratios[0]
    # physics sanity: positive interior solution, max locked as regression value
    umax = prob.expand(u).max()
    assert 0.05 < umax < 0.30                      # lambda=1 cube Bratu lower branch

def test_jfnk_matches_analytic(device):
    prob = _bratu(device)
    ns_a = NonlinearSolver(prob.residual, jac_action_fn=prob.jac_action, snes_rtol=1e-10)
    ns_j = NonlinearSolver(prob.residual, jac_action_fn=None, snes_rtol=1e-10)
    ua, _ = ns_a.solve(np.zeros(prob.n_free))
    uj, _ = ns_j.solve(np.zeros(prob.n_free))
    assert np.abs(ua - uj).max() < 1e-6

def test_linesearch_rescues_bad_step(device):
    prob = _bratu(device, lam=5.0)                 # stiffer; full steps can overshoot
    ns = NonlinearSolver(prob.residual, jac_action_fn=prob.jac_action,
                         snes_rtol=1e-9, snes_max_it=40, linesearch="bt")
    u, info = ns.solve(np.zeros(prob.n_free))
    assert info["converged"]
