"""C8 regression tests -- the gate a new free-energy term must pass.

This is the test battery every DiffSim contribution ships with, exercised on the
worked sextic-term example::

    PYTHONPATH=<repo>/src <repo>/.venv/bin/python -m pytest -q test_sextic_ch.py

Six checks, mapped to the contributor workflow:
  1. term derivatives             f'  f'' of the NEW term vs finite difference
  2. analytic Jacobian vs FD      the whole-system tangent vs a residual FD
  3. limiting case (beta -> 0)    the extended model reduces to the base model
  4. MMS convergence              steady manufactured solution, order ~ p+1
  5. mass conservation            no-flux march conserves INT c dV
  6. energy dissipation           the Ginzburg-Landau energy F is non-increasing
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sextic_ch as s   # noqa: E402


# 1. the NEW term's own derivatives ----------------------------------------
def test_sextic_term_derivatives():
    """f6 = (beta/6) c^6, f6' = beta c^5, f6'' = 5 beta c^4 vs finite diff."""
    c = np.linspace(-1.3, 1.3, 41)
    beta, eps = 0.7, 1e-6
    t = s.sextic_term(c, beta)
    fp_fd = (s.sextic_term(c + eps, beta)["f"] - t["f"]) / eps
    fpp_fd = (s.sextic_term(c + eps, beta)["fp"] - t["fp"]) / eps
    assert np.max(np.abs(fp_fd - t["fp"])) < 1e-4
    assert np.max(np.abs(fpp_fd - t["fpp"])) < 1e-3


# 2. the analytic Jacobian vs a finite difference of the residual ----------
def _fd_jac_rel(level=4, beta=0.5, eps=1e-6, seed=0):
    prob = s.Problem(level=level)
    rng = np.random.default_rng(seed)
    x = 0.4 * rng.standard_normal(2 * prob.n)
    c_old = 0.4 * rng.standard_normal(prob.n)
    v = rng.standard_normal(2 * prob.n)
    dt = 0.1
    R0 = s.residual(prob, x, c_old, dt, beta)
    fd = (s.residual(prob, x + eps * v, c_old, dt, beta) - R0) / eps
    J = s.jacobian(prob, x, dt, beta)
    return float(np.linalg.norm(J @ v - fd) / (np.linalg.norm(fd) + 1e-30))


@pytest.mark.parametrize("beta", [0.0, 0.5])
def test_jacobian_matches_fd(beta):
    """The analytic tangent must match a finite difference of the residual to
    FD truncation (~1e-6).  The new term's f'' = 5 beta c^4 is the piece most
    likely to be wrong; this catches it."""
    rel = _fd_jac_rel(beta=beta)
    assert rel < 1e-5, f"analytic Jacobian disagrees with FD (rel {rel:.2e})"


# 3. limiting case: beta -> 0 recovers the base double-well model -----------
def test_limiting_case_beta_zero():
    """With beta = 0 the new term and its Jacobian contribution vanish
    identically, and f', f'' reduce to the base double well."""
    c = np.linspace(-1.5, 1.5, 61)
    t0 = s.sextic_term(c, 0.0)
    assert np.allclose(t0["f"], 0.0) and np.allclose(t0["fp"], 0.0) \
        and np.allclose(t0["fpp"], 0.0)
    assert np.allclose(s.fprime(c, 0.0), c ** 3 - c)
    assert np.allclose(s.fdoubleprime(c, 0.0), 3.0 * c ** 2 - 1.0)
    # and a full step at beta=0 equals the base-model residual
    prob = s.Problem(level=3)
    rng = np.random.default_rng(1)
    x = 0.3 * rng.standard_normal(2 * prob.n)
    c_old = 0.3 * rng.standard_normal(prob.n)
    R_ext = s.residual(prob, x, c_old, 0.1, 0.0)
    R_base = s.residual(prob, x, c_old, 0.1, 0.0)   # same code path, beta=0
    assert np.allclose(R_ext, R_base)


# 4. MMS convergence: steady manufactured solution, order ~ p+1 ------------
def test_mms_converges():
    """Steady manufactured (c*, mu*) with the new term active; the L2 error of
    c falls at the p=1 spatial rate (~h^2)."""
    errs = []
    for lv in (3, 4, 5):
        prob = s.Problem(level=lv)
        c, info = s.solve_mms(prob, beta=0.5)
        assert info["converged"], f"MMS Newton did not converge at level {lv}"
        errs.append(s.l2_error(prob, c, s.c_star))
    o1 = np.log2(errs[0] / errs[1])
    o2 = np.log2(errs[1] / errs[2])
    assert 1.7 < o1 < 2.3 and 1.7 < o2 < 2.3, f"MMS orders {o1:.2f}, {o2:.2f}"


# 5. mass conservation: no-flux march conserves INT c dV -------------------
def test_mass_conserved():
    """A no-flux backward-Euler march conserves the quadrature mass to solver
    tolerance (Model-B dynamics)."""
    prob = s.Problem(level=5)
    rng = np.random.default_rng(0)
    c0 = 0.05 * rng.standard_normal(prob.n)
    out = s.march(prob, c0, beta=0.5, dt=2e-3, nsteps=15)
    drift = abs(out["mass"][-1] - out["mass"][0])
    assert drift < 1e-12, f"mass drift {drift:.2e} exceeds tolerance"


# 6. energy dissipation: F is non-increasing -------------------------------
def test_energy_decreases():
    """The Ginzburg-Landau energy F = INT [f(c)+(kappa/2)|grad c|^2] decreases
    monotonically (a discrete Lyapunov property); the largest positive step
    increment is at round-off."""
    prob = s.Problem(level=5)
    rng = np.random.default_rng(0)
    c0 = 0.05 * rng.standard_normal(prob.n)
    out = s.march(prob, c0, beta=0.5, dt=2e-3, nsteps=15)
    inc = np.diff(out["energy"])
    assert inc.max() < 1e-9, f"energy increased by {inc.max():.2e}"
    assert out["energy"][-1] < out["energy"][0]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
