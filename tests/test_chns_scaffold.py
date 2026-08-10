# tests/test_chns_scaffold.py
import numpy as np
from diffsim.physics.chns import mix_props, capillary_gp, tau_m_gp


def test_mix_props_pure_phases():
    rho, eta, nc = mix_props(np.array([1.0, -1.0]), 1000.0, 1.0, 100.0, 0.1)
    assert np.allclose(rho, [1000.0, 1.0]) and np.allclose(eta, [100.0, 0.1])
    assert nc == 0


def test_mix_props_clamps_overshoot():
    # phi = -1.2 overshoots: raw rho = -0.1*999.5+500.5 < 1.0 -> clamped to floor
    rho, eta, nc = mix_props(np.array([-1.2]), 1000.0, 1.0, 100.0, 0.1)
    assert rho[0] >= 1e-3 * 1.0 and nc == 1


def test_capillary_zero_for_uniform_phi():
    f = capillary_gp(np.array([0.7]), np.zeros((1, 2)), Cn=0.01, We=10.0)
    assert np.allclose(f, 0.0)


def test_tau_m_local_viscosity_matters():
    t_lo = tau_m_gp(np.zeros((1, 2)), np.ones(1), np.array([0.1]), 0.05, 1e-2, 35.0)
    t_hi = tau_m_gp(np.zeros((1, 2)), np.ones(1), np.array([100.0]), 0.05, 1e-2, 35.0)
    assert t_hi[0] < t_lo[0]  # stiffer viscosity -> smaller tau
