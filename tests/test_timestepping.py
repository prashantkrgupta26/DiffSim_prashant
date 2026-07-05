"""M1b Task 4 gates: BDF tables vs sympy-free closed forms, scalar-ODE
convergence orders (the acceptance the plan requires: 1, 2, 2), bootstrap
gate, variable-step table, history rotation."""
import numpy as np
import pytest
from diffsim.solvers.timestepping import (bdf_coeffs, bdf_order_now,
                                          extrapolate_velocity,
                                          extrapolate_pressure, History)

pytestmark = pytest.mark.tier2


def test_tables_exact():
    assert bdf_coeffs(1, 0.1) == (1.0, -1.0, 0.0)
    assert bdf_coeffs(2, 0.1) == (1.5, -2.0, 0.5)
    # variable step: consistency (sums to 0) + exactness on u=t
    dt, dtp = 0.05, 0.08
    b0, b1, b2 = bdf_coeffs(2, dt, dtp)
    assert abs(b0 + b1 + b2) < 1e-14                       # constants
    # exact derivative of u = t at t_{n+1}: (b0 t1 + b1 t0 + b2 tm1)/dt = 1
    t1, t0, tm1 = 0.0, -dt, -dt - dtp
    assert abs((b0 * t1 + b1 * t0 + b2 * tm1) / dt - 1.0) < 1e-13
    # and exact on u = t^2 (BDF2 is second order)
    d = (b0 * t1 ** 2 + b1 * t0 ** 2 + b2 * tm1 ** 2) / dt
    assert abs(d - 2 * t1) < 1e-13


def test_bootstrap_gate():
    dt = 0.1
    assert bdf_order_now(0.05, dt, 2) == 1          # t < 1.5 dt
    assert bdf_order_now(0.1, dt, 2) == 1
    # (0.15 is the exact float boundary — 1.5*0.1 rounds above it; the
    # production gate has the same semantics, so test strictly past it)
    assert bdf_order_now(0.16, dt, 2) == 2
    assert bdf_order_now(0.5, dt, 2, have_history=False) == 1


def _solve_ode(order, nsteps, T=1.0, lam=-2.0):
    """u' = lam u + f with manufactured u* = cos(3t): full stepping loop
    with the production bootstrap."""
    dt = T / nsteps
    ustar = lambda t: np.cos(3 * t)
    f = lambda t: -3 * np.sin(3 * t) - lam * ustar(t)
    hist = History()
    hist.rotate(np.array([ustar(0.0)]))              # exact IC
    t = 0.0
    for n in range(nsteps):
        t_new = t + dt
        o = bdf_order_now(t_new, dt, order, have_history=hist.have(2))
        b0, b1, b2 = bdf_coeffs(o, dt)
        rhs_h = hist.rhs_history(b1, b2, dt)
        # (b0/dt - lam) u = f - rhs_h
        u_new = (f(t_new) - rhs_h) / (b0 / dt - lam)
        hist.rotate(u_new, dt=dt)
        t = t_new
    return abs(float(hist.pre1[0]) - ustar(T))


@pytest.mark.parametrize("order,expected", [(1, 1.0), (2, 2.0)])
def test_scalar_ode_convergence(order, expected):
    errs = [_solve_ode(order, n) for n in (40, 80, 160)]
    rates = [np.log2(errs[i] / errs[i + 1]) for i in range(2)]
    for r in rates:
        assert abs(r - expected) < 0.15, (order, errs, rates)


def test_variable_step_second_order():
    # alternating steps: the variable-dt table must hold order 2
    def solve(nsteps):
        T, lam = 1.0, -2.0
        ustar = lambda t: np.cos(3 * t)
        f = lambda t: -3 * np.sin(3 * t) - lam * ustar(t)
        dts = np.tile([1.4, 0.6], nsteps // 2) * (T / nsteps)
        dts *= T / dts.sum()
        hist = History()
        hist.rotate(np.array([ustar(0.0)]))
        t = 0.0
        for i, dt in enumerate(dts):
            t_new = t + dt
            o = bdf_order_now(t_new, dt, 2, have_history=hist.have(2))
            b0, b1, b2 = bdf_coeffs(o, dt, hist.dt_prev)
            rhs_h = hist.rhs_history(b1, b2, dt)
            u_new = (f(t_new) - rhs_h) / (b0 / dt - lam)
            hist.rotate(u_new, dt=dt)
            t = t_new
        return abs(float(hist.pre1[0]) - ustar(t))

    errs = [solve(n) for n in (40, 80, 160)]
    rates = [np.log2(errs[i] / errs[i + 1]) for i in range(2)]
    for r in rates:
        assert abs(r - 2.0) < 0.2, (errs, rates)


def test_extrapolations_and_rotation():
    u1, u2 = np.array([3.0]), np.array([1.0])
    assert extrapolate_velocity(1, u1)[0] == 3.0
    assert extrapolate_velocity(2, u1, u2)[0] == 5.0     # 2u^n - u^{n-1}
    assert extrapolate_pressure(0, u1) is None
    assert extrapolate_pressure(2, u1, u2)[0] == 5.0
    h = History()
    for v in (1.0, 2.0, 3.0):
        h.rotate(np.array([v]))
    assert (h.pre1[0], h.pre2[0], h.pre3[0]) == (3.0, 2.0, 1.0)
    h.rotate(np.array([4.0]))
    assert (h.pre1[0], h.pre2[0], h.pre3[0]) == (4.0, 3.0, 2.0)
