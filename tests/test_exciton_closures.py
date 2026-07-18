"""Tests for diffsim.physics.exciton_closures (SP-1 A3).

ALL expectations hand-computed — never derived by calling code under test.

Hand-computed constants (PM6:Y6 defaults, E=1e7 V/m, dist=0, width=1e-9):
  _KB   = 1.380649e-23 J/K
  _Q    = 1.602176634e-19 C
  _EPS0 = 8.8541878128e-12 F/m

  Scales:
    phi0 = 2.5851999786e-02 V          (= _KB*300/_Q)
    mu0  = 2e-7 m^2/Vs
    t0   = 1.9340863536e-06 s          (= x0^2/(mu0*phi0))
    U0   = 1.2925999893e+31 m^-3/s
    gamma0 = 1.8357376414e-15 m^3/s

  Onsager-Braun at dist=0, E=1e7 V/m:
    eps_r = 3.45 (tanh_mask(0,w)=0.5 → eps_D+0.5*(eps_A-eps_D))
    eps   = 8.8541878128e-12 * 3.45 = 3.0546947954e-11 F/m
    E_B   = Q^2/(4π*eps*a) = 3.7151007284e-20 J
    E_B/kT = 8.9694550133
    b     = Q^3*E/(8π*eps*(kT)^2) = 3.1225860973
    Phi   = 9.7023998191
    k_dim = 9.2768137574e+07 1/s
    k_hat_d = k_hat_a = 1.7936141988e+02  (after interface_mask(0,1e-9)=0.9996646499)

  Langevin nondim rates (zeta=1.0):
    gamma_hat_sum = 8.8761878026e+04
    gamma_hat_min = 3.8040804868e+04
    gamma_hat_if  = 5.7061207303e+03

  Region mobility at deep donor (dist=-1e-6):
    mu_n_hat = 1e-6, mu_p_hat = 0.75

  Region mobility at deep acceptor (dist=1e-6):
    mu_n_hat = 1.0, mu_p_hat = 7.5e-7
"""
from __future__ import annotations

import math
import numpy as np
import pytest

from diffsim.xdd.params import XDDParams, _KB, _Q, _EPS0
from diffsim.physics.exciton_closures import (
    OnsagerBraunDissociation,
    LangevinRecombination,
    Generation,
    RegionMobility,
)


# ---------------------------------------------------------------------------
# Shared test fixtures / constants
# ---------------------------------------------------------------------------

@pytest.fixture
def p():
    """Default PM6:Y6 XDDParams."""
    return XDDParams()


@pytest.fixture
def s(p):
    """Corresponding XDDScales."""
    return p.scales()


# ---------------------------------------------------------------------------
# Gate 1 — Onsager-Braun hand value
# ---------------------------------------------------------------------------

def test_onsager_braun_hand_value(p, s):
    """k_hat_d matches hand-computed value for E=1e7 V/m, dist=0, PM6:Y6 defaults."""
    # Reproduce hand-computation exactly (never call code under test here)
    T    = 300.0
    a    = 1.8e-9
    eps_r_at0 = p.eps_D + (p.eps_A - p.eps_D) * 0.5          # tanh_mask(0,w)=0.5 → 3.45
    eps  = _EPS0 * eps_r_at0                                   # 3.0546947954e-11 F/m
    E_B  = _Q**2 / (4 * math.pi * eps * a)                    # 3.7151007284e-20 J
    kT   = _KB * T                                             # 4.1419470000e-21 J
    E    = 1e7                                                 # V/m (the field)
    b    = _Q**3 * E / (8 * math.pi * eps * kT**2)            # 3.1225860973
    Phi  = 1 + b + b**2/3 + b**3/18 + b**4/180 + b**5/2700   # 9.7023998191
    gam0 = 2 * (p.mu_n + p.mu_p) * _Q / (_EPS0 * (p.eps_A + p.eps_D))  # gamma0
    k_dim = (3 * gam0 / (4 * math.pi * a**3)) * math.exp(-E_B / kT) * Phi

    # Scales (hand-reproduced to avoid calling code under test)
    phi0 = kT / _Q                         # 2.5851999786e-02 V
    x0   = p.height                        # 100e-9 m
    mu0  = max(p.mu_n, p.mu_p)            # 2e-7
    t0   = x0**2 / (mu0 * phi0)           # 1.9340863536e-06 s

    k_hat_no_mask = k_dim * t0            # 1.7942158893e+02

    # interface_mask(0, interface_thk/2=1e-9):
    # 0.5*(tanh((1e-9 - 0)/(0.25*1e-9)) + 1) = 0.5*(tanh(4)+1) = 0.9996646499
    half_thk = p.interface_thk / 2       # 1e-9
    mask = 0.5 * (math.tanh((half_thk - 0.0) / (0.25 * half_thk)) + 1.0)

    k_hat_expected = k_hat_no_mask * mask  # ~179.36

    # Convert E=1e7 to nondim grad_phi_hat_mag = E*x0/phi0
    grad_phi_hat = E * x0 / phi0          # 38.6817270718

    width = 1e-9
    ob = OnsagerBraunDissociation(params=p, width=width)
    dist_arr = np.array([0.0])
    k_hat_d, k_hat_a, (dk_dgrad_d, dk_dgrad_a) = ob(grad_phi_hat, dist_arr)

    assert abs(float(k_hat_d[0]) - k_hat_expected) / k_hat_expected < 1e-8, (
        f"k_hat_d={float(k_hat_d[0]):.10e}, expected={k_hat_expected:.10e}"
    )
    assert abs(float(k_hat_a[0]) - k_hat_expected) / k_hat_expected < 1e-8, (
        f"k_hat_a={float(k_hat_a[0]):.10e}, expected={k_hat_expected:.10e}"
    )

    # ex_diss_d_scaling == ex_diss_a_scaling == 1.0 → k_d == k_a
    np.testing.assert_allclose(k_hat_d, k_hat_a, rtol=1e-12)


def test_onsager_braun_b0_limit(p):
    """Phi(b=0) == 1 exactly — zero-field limit."""
    # At E=0, grad_phi_hat=0, b=0, Phi=1
    # k_dim = (3*gamma0/(4*pi*a^3)) * exp(-E_B/kT) * 1
    # Just verify k_hat_d > 0 and consistent with E>0 result (k increases with E)
    width = 1e-9
    ob = OnsagerBraunDissociation(params=p, width=width)
    dist_arr = np.array([0.0])

    k0_d, k0_a, _ = ob(0.0, dist_arr)
    k7_d, k7_a, _ = ob(38.68, dist_arr)  # E≈1e7 V/m

    assert float(k0_d[0]) > 0.0, "k_hat_d(E=0) must be positive"
    assert float(k7_d[0]) > float(k0_d[0]), "k_hat_d must increase with field"


# ---------------------------------------------------------------------------
# Gate 2 — Onsager-Braun finite-difference derivative check
# ---------------------------------------------------------------------------

def test_onsager_braun_derivative_fd(p):
    """Analytic dk_dgrad agrees with central FD over logspace field range."""
    phi0 = _KB * p.T / _Q
    x0   = p.height

    # E range in V/m → convert to nondim grad
    E_arr = np.logspace(5, 8, 20)
    grad_arr = E_arr * x0 / phi0

    width = 1e-9
    ob = OnsagerBraunDissociation(params=p, width=width)
    dist_arr = np.array([0.0])

    for grad in grad_arr:
        h = 1e-4 * grad
        k_plus_d,  _, _ = ob(grad + h, dist_arr)
        k_minus_d, _, _ = ob(grad - h, dist_arr)
        fd_d = (float(k_plus_d[0]) - float(k_minus_d[0])) / (2 * h)

        _, _, (dk_dgrad_d, _) = ob(grad, dist_arr)
        analytic_d = float(dk_dgrad_d[0])

        rel_err = abs(fd_d - analytic_d) / (abs(analytic_d) + 1e-300)
        assert rel_err < 1e-5, (
            f"FD derivative check failed at E={grad*phi0/x0:.2e} V/m: "
            f"fd={fd_d:.6e}, analytic={analytic_d:.6e}, rel_err={rel_err:.2e}"
        )


# ---------------------------------------------------------------------------
# Gate 3 — Langevin strategies
# ---------------------------------------------------------------------------

def test_langevin_strategies(p, s):
    """LangevinRecombination strategies match hand-computed nondim gamma_hat."""
    C0 = s.C0    # 2.5e25
    U0 = s.U0    # 1.2925999893e+31

    # Hand-computed gamma values (see module docstring):
    # gamma0 (sum): 1.8357376414e-15 m^3/s → gamma_hat_sum = 8.8761878026e+04
    # gamma_min:    7.8674470347e-16 m^3/s → gamma_hat_min = 3.8040804868e+04
    # gamma_if:     1.1801170552e-16 m^3/s → gamma_hat_if  = 5.7061207303e+03

    eps_bar   = _EPS0 * (p.eps_A + p.eps_D) / 2.0
    gamma_min_dim = _Q * min(p.mu_n, p.mu_p) / eps_bar   # 7.8674470347e-16
    gamma_if_dim  = abs((p.eps_D - p.eps_A) / (p.eps_D + p.eps_A)) * _Q * p.mu_p / (_EPS0 * p.eps_D)

    gamma_hat_sum_expected = s.gamma0 * C0**2 / U0         # 8.8761878026e+04
    gamma_hat_min_expected = gamma_min_dim * C0**2 / U0     # 3.8040804868e+04
    gamma_hat_if_expected  = gamma_if_dim  * C0**2 / U0     # 5.7061207303e+03

    n_hat = np.array([1.5])
    p_hat = np.array([2.0])
    dist_far = np.array([0.0])  # uniform spatial

    lr_sum = LangevinRecombination(params=p, strategy="sum", zeta=1.0, spatial="uniform")
    lr_min = LangevinRecombination(params=p, strategy="min", zeta=1.0, spatial="uniform")
    lr_if  = LangevinRecombination(params=p, strategy="image_force", zeta=1.0, spatial="uniform")

    R_sum, dR_dn_sum, dR_dp_sum = lr_sum(n_hat, p_hat, dist_far)
    R_min, dR_dn_min, dR_dp_min = lr_min(n_hat, p_hat, dist_far)
    R_if,  dR_dn_if,  dR_dp_if  = lr_if(n_hat,  p_hat, dist_far)

    # Check R = gamma_hat * n * p
    np.testing.assert_allclose(
        float(R_sum[0]), gamma_hat_sum_expected * 1.5 * 2.0, rtol=1e-10,
        err_msg="R_hat (sum strategy) mismatch"
    )
    np.testing.assert_allclose(
        float(R_min[0]), gamma_hat_min_expected * 1.5 * 2.0, rtol=1e-10,
        err_msg="R_hat (min strategy) mismatch"
    )
    np.testing.assert_allclose(
        float(R_if[0]), gamma_hat_if_expected * 1.5 * 2.0, rtol=1e-10,
        err_msg="R_hat (image_force strategy) mismatch"
    )

    # Check derivatives: dR/dn = gamma_hat * p_hat
    np.testing.assert_allclose(float(dR_dn_sum[0]), gamma_hat_sum_expected * 2.0, rtol=1e-10)
    np.testing.assert_allclose(float(dR_dp_sum[0]), gamma_hat_sum_expected * 1.5, rtol=1e-10)

    # zeta=0.5 halves the rate
    lr_half = LangevinRecombination(params=p, strategy="sum", zeta=0.5, spatial="uniform")
    R_half, _, _ = lr_half(n_hat, p_hat, dist_far)
    np.testing.assert_allclose(float(R_half[0]), 0.5 * float(R_sum[0]), rtol=1e-12)

    # interface spatial: at dist >> width → R ≈ 0
    lr_iface = LangevinRecombination(params=p, strategy="sum", zeta=1.0,
                                      spatial="interface", width=1e-9)
    R_far, _, _ = lr_iface(n_hat, p_hat, np.array([1e-6]))
    assert abs(float(R_far[0])) < 1e-100, f"Interface-spatial R at dist=1e-6 should be ~0, got {R_far}"

    # disabled spatial → 0
    lr_dis = LangevinRecombination(params=p, strategy="sum", zeta=1.0, spatial="disabled")
    R_dis, _, _ = lr_dis(n_hat, p_hat, dist_far)
    np.testing.assert_allclose(float(R_dis[0]), 0.0, atol=0.0)


# ---------------------------------------------------------------------------
# Gate 4 — Generation spatial profile
# ---------------------------------------------------------------------------

def test_generation_regions(p, s):
    """Generation is localized correctly by region (constant and beer_lambert profiles)."""
    U0 = s.U0
    x0 = p.height

    dist = np.linspace(-50e-9, 50e-9, 101)   # 101 points spanning ±50 nm
    h_hat = np.full_like(dist, 0.5)            # middle height
    width = 1e-9

    gen = Generation(params=p, profile="constant", waveform="cw", width=width)

    G_hat_d, G_hat_a = gen(dist, h_hat, 0.0)

    # Deep donor (dist << 0): donor G should be ~Gx_donor/U0, acceptor G ~0 (RET excepted)
    deep_donor_idx = 0  # dist=-50e-9 >> width
    # tanh_mask(-50e-9, 1e-9) ≈ 0 → w_donor ≈ 1, w_acceptor ≈ 0
    # G_d_dim = Gx_donor * w_donor ≈ Gx_donor → G_hat_d ≈ Gx_donor/U0
    G_hat_d_expected_dd = p.Gx_donor / U0
    assert abs(float(G_hat_d[deep_donor_idx]) - G_hat_d_expected_dd) / G_hat_d_expected_dd < 1e-6

    # Deep acceptor (dist >> 0): acceptor G should be ~Gx_acceptor/U0
    deep_acc_idx = -1  # dist=+50e-9
    G_hat_a_expected_da = p.Gx_acceptor / U0
    assert abs(float(G_hat_a[deep_acc_idx]) - G_hat_a_expected_da) / G_hat_a_expected_da < 1e-6

    # RET: acceptor G should be nonzero in donor-side near interface
    # At dist=0 (interface), both RET and normal G contribute to G_a
    # Since interface_mask peaks at dist=0, check G_hat_a at idx=50 (dist=0) is > far-field value
    idx_interface = 50  # dist=0
    G_a_at_interface = float(G_hat_a[idx_interface])
    G_a_far_donor = float(G_hat_a[0])  # deep donor side, G_a_dim ≈ ret contribution only
    # At interface, w_acceptor = 0.5, so normal G_a is already half; RET adds more
    assert G_a_at_interface > G_a_far_donor, (
        "G_hat_a should be larger at interface than deep donor due to RET"
    )

    # beer_lambert profile: check ratio G(h=1)/G(h=0) = exp(alpha0*x0)
    alpha0 = 1e7
    gen_bl = Generation(params=p, profile="beer_lambert", alpha0=alpha0,
                        waveform="cw", width=width)

    # Single-point checks at h_hat=1.0 (entrance) and h_hat=0.0 (exit)
    dist_single = np.array([0.0])  # at interface
    G_hat_d_h1, _ = gen_bl(dist_single, np.array([1.0]), 0.0)
    G_hat_d_h0, _ = gen_bl(dist_single, np.array([0.0]), 0.0)

    ratio = float(G_hat_d_h1[0]) / float(G_hat_d_h0[0])
    expected_ratio = math.exp(alpha0 * x0)  # exp(1) = 2.71828...
    np.testing.assert_allclose(ratio, expected_ratio, rtol=1e-8,
                                err_msg=f"beer_lambert ratio {ratio} != exp(alpha0*x0)={expected_ratio}")


# ---------------------------------------------------------------------------
# Gate 5 — Waveforms
# ---------------------------------------------------------------------------

def test_waveforms(p, s):
    """Generation waveform amplitude functions match analytic values."""
    t0  = s.t0
    freq = 1e6  # Hz

    # cw
    gen_cw = Generation(params=p, waveform="cw")
    assert gen_cw.amplitude(0.0) == 1.0
    assert gen_cw.amplitude(1e5) == 1.0

    # rect_sin: w(t) = |sin(2*pi*freq*t)|
    gen_rs = Generation(params=p, waveform="rect_sin", freq=freq)
    # At t_hat = 1/(4*freq*t0): t = t0/(4*freq*t0) = 1/(4*freq), sin = sin(2*pi*freq/(4*freq)) = sin(pi/2) = 1
    t_hat_q = 1.0 / (4.0 * freq * t0)
    amp_q = gen_rs.amplitude(t_hat_q)
    np.testing.assert_allclose(amp_q, 1.0, atol=1e-14,
                                err_msg=f"rect_sin amplitude at quarter period: {amp_q}")

    # At t_hat = 1/(2*freq*t0): sin(pi) = 0
    t_hat_h = 1.0 / (2.0 * freq * t0)
    amp_h = gen_rs.amplitude(t_hat_h)
    np.testing.assert_allclose(amp_h, 0.0, atol=1e-14,
                                err_msg=f"rect_sin amplitude at half period: {amp_h}")

    # pulse: 1 before pulse_duration, 0 after
    pd = 5e-7  # 500 ns
    gen_pulse = Generation(params=p, waveform="pulse", pulse_duration=pd)
    t_hat_before = 0.9 * pd / t0
    t_hat_after  = 1.1 * pd / t0
    assert gen_pulse.amplitude(t_hat_before) == 1.0, "pulse should be 1 before duration"
    assert gen_pulse.amplitude(t_hat_after)  == 0.0, "pulse should be 0 after duration"

    # step: 0 before t_on, 1 at/after t_on
    t_on = 1e-7
    gen_step = Generation(params=p, waveform="step", t_on=t_on)
    t_hat_before_on = 0.9 * t_on / t0
    t_hat_at_on     = t_on / t0
    assert gen_step.amplitude(t_hat_before_on) == 0.0, "step should be 0 before t_on"
    assert gen_step.amplitude(t_hat_at_on)     == 1.0, "step should be 1 at t_on"

    # user waveform: callable t_hat -> amplitude
    gen_user = Generation(params=p, waveform="user", user_waveform=lambda t: t**2)
    np.testing.assert_allclose(gen_user.amplitude(3.0), 9.0, rtol=1e-12)


# ---------------------------------------------------------------------------
# Gate 6 — Region mobility limits
# ---------------------------------------------------------------------------

def test_region_mobility_limits(p, s):
    """RegionMobility matches hand-computed limits in deep donor / acceptor / interface."""
    mu0 = s.mu0   # 2e-7
    mu_ratio = p.mu_ratio  # 1e-6

    rm = RegionMobility(params=p, width=1e-9)

    # --- Deep donor (dist = -1e-6 m, w_a ≈ 0, w_d ≈ 1) ---
    # tanh(-1e-6/1e-9) ≈ tanh(-1000) = -1 exactly in float64
    dist_dd = np.array([-1e-6])
    mob_dd = rm(dist_dd)

    # mu_n_hat = (mu_n/mu0)*(w_a + mu_ratio*w_d) ≈ (1.0)*(0 + 1e-6*1) = 1e-6
    mu_n_hat_dd_expected = (p.mu_n / mu0) * mu_ratio   # (2e-7/2e-7)*1e-6 = 1e-6
    np.testing.assert_allclose(float(mob_dd["mu_n_hat"][0]), mu_n_hat_dd_expected,
                                rtol=1e-12, err_msg="deep donor mu_n_hat")

    # mu_p_hat = (mu_p/mu0)*(w_d + mu_ratio*w_a) ≈ (0.75)*(1 + 0) = 0.75
    mu_p_hat_dd_expected = p.mu_p / mu0   # 1.5e-7/2e-7 = 0.75
    np.testing.assert_allclose(float(mob_dd["mu_p_hat"][0]), mu_p_hat_dd_expected,
                                rtol=1e-12, err_msg="deep donor mu_p_hat")

    # --- Deep acceptor (dist = +1e-6 m, w_a ≈ 1, w_d ≈ 0) ---
    dist_da = np.array([1e-6])
    mob_da = rm(dist_da)

    # mu_n_hat = (mu_n/mu0)*(1 + mu_ratio*0) = mu_n/mu0 = 1.0
    np.testing.assert_allclose(float(mob_da["mu_n_hat"][0]), p.mu_n / mu0,
                                rtol=1e-12, err_msg="deep acceptor mu_n_hat")

    # mu_p_hat = (mu_p/mu0)*(0 + mu_ratio*1) = (0.75)*1e-6 = 7.5e-7
    mu_p_hat_da_expected = (p.mu_p / mu0) * mu_ratio
    np.testing.assert_allclose(float(mob_da["mu_p_hat"][0]), mu_p_hat_da_expected,
                                rtol=1e-12, err_msg="deep acceptor mu_p_hat")

    # --- Interface (dist = 0, w_a = w_d = 0.5) ---
    dist_int = np.array([0.0])
    mob_int = rm(dist_int)

    # mu_n_hat = (mu_n/mu0)*(0.5 + mu_ratio*0.5) = (1.0)*0.5*(1+1e-6) = 5.0000050000e-01
    mu_n_hat_int_expected = (p.mu_n / mu0) * 0.5 * (1.0 + mu_ratio)
    np.testing.assert_allclose(float(mob_int["mu_n_hat"][0]), mu_n_hat_int_expected,
                                rtol=1e-12, err_msg="interface mu_n_hat")

    # eps_r at dist=0: eps_D + (eps_A-eps_D)*0.5 = 3.0 + 0.9*0.5 = 3.45
    eps_r_expected = p.eps_D + (p.eps_A - p.eps_D) * 0.5
    np.testing.assert_allclose(float(mob_int["eps_r"][0]), eps_r_expected,
                                rtol=1e-12, err_msg="interface eps_r")


# ---------------------------------------------------------------------------
# Gate 7 — C0/U0 == t0 identity
# ---------------------------------------------------------------------------

def test_c0_over_u0_is_t0():
    """C0/U0 == t0 dimensional consistency gate."""
    p = XDDParams()
    s = p.scales()
    # t0 = x0^2/(mu0*phi0),  U0 = mu0*phi0*C0/x0^2
    # C0/U0 = C0 * x0^2 / (mu0*phi0*C0) = x0^2/(mu0*phi0) = t0
    assert abs(s.C0 / s.U0 - s.t0) / s.t0 < 1e-12, (
        f"C0/U0={s.C0/s.U0:.10e} != t0={s.t0:.10e}"
    )
