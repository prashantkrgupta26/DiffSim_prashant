"""Tests for XDDParams + CPU-config/spec readers (Task A1, SP-1 R0).

TDD: tests written FIRST, verified RED, then implementation written to GREEN.
All gates from .superpowers/sdd/sp1-a1-brief.md.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

# ──────────────────────────────────────────────────────────────────────────────
# Paths to CPU reference files (read-only; never modified)
# ──────────────────────────────────────────────────────────────────────────────
_CPU_TEST = (
    "/Users/baskarg/Dropbox/work/Projects/ClaudeCode/old/DiffSim_old"
    "/mypapers/OSC/StructureProperty/"
    "baskargroup-excitonic_drift_diffusion-0f21069f7995"
    "/drift_diffusion/test"
)
_CONFIG  = f"{_CPU_TEST}/config.txt"
_SOLAR   = f"{_CPU_TEST}/solar.spec"
_ACC_Y6  = f"{_CPU_TEST}/acceptor_y6.spec"
_DON_PM6 = f"{_CPU_TEST}/donor_pm6.spec"
_PL_PM6  = f"{_CPU_TEST}/donor_pl_pm6.spec"
_ACC_PCBM  = f"{_CPU_TEST}/acceptor_pcbm.spec"
_PL_P3HT   = f"{_CPU_TEST}/donor_pl_p3ht.spec"
_BLEND_PM6Y6 = f"{_CPU_TEST}/blend_pm6-y6.spec"


# ──────────────────────────────────────────────────────────────────────────────
# Constants (same as implementation — written here for independent check)
# ──────────────────────────────────────────────────────────────────────────────
_KB = 1.380649e-23       # J/K
_Q  = 1.602176634e-19    # C
_EPS0 = 8.8541878128e-12 # F/m
_H    = 6.62607015e-34   # J·s
_C    = 2.99792458e8     # m/s


# ──────────────────────────────────────────────────────────────────────────────
# Gate 1: scales from default params — hand-computed values
# ──────────────────────────────────────────────────────────────────────────────
class TestScalesCanonical:
    """Gate 1: scales() from default XDDParams match hand-computed SI values."""

    def _hand_scales(self):
        """Independent hand computation — must NOT call XDDParams.scales()."""
        mu_n = 2e-7
        mu_p = 1.5e-7
        eps_A = 3.9
        eps_D = 3.0
        T = 300.0
        height = 100e-9
        N_C = 2.5e25

        x0   = height
        phi0 = _KB * T / _Q
        C0   = N_C
        mu0  = max(mu_n, mu_p)
        t0   = x0 * x0 / (mu0 * phi0)
        eps_m = _EPS0 * max(eps_A, eps_D)
        U0   = mu0 * phi0 * C0 / (x0 * x0)
        J0   = U0 * x0 * _Q
        lam2 = phi0 * eps_m / (x0 * x0 * C0 * _Q)
        gam0 = 2 * (mu_n + mu_p) * _Q / (_EPS0 * (eps_A + eps_D))
        return dict(x0=x0, phi0=phi0, C0=C0, mu0=mu0, t0=t0,
                    eps_m=eps_m, U0=U0, J0=J0, lambda2=lam2, gamma0=gam0)

    def test_phi0_approx(self):
        from diffsim.xdd.params import XDDParams
        p = XDDParams()
        s = p.scales()
        ref = self._hand_scales()
        assert abs(s.phi0 - ref["phi0"]) / ref["phi0"] < 1e-10

    def test_phi0_value(self):
        """phi0 at 300 K should be ~0.025852 V."""
        from diffsim.xdd.params import XDDParams
        s = XDDParams().scales()
        assert abs(s.phi0 - 0.025851999) < 1e-8

    def test_x0(self):
        from diffsim.xdd.params import XDDParams
        s = XDDParams().scales()
        assert s.x0 == pytest.approx(1e-7, rel=1e-12)

    def test_mu0(self):
        from diffsim.xdd.params import XDDParams
        s = XDDParams().scales()
        assert s.mu0 == pytest.approx(2e-7, rel=1e-12)

    def test_t0_dimensionally_correct(self):
        """t0 must equal x0^2/(mu0*phi0) — not the inverted form."""
        from diffsim.xdd.params import XDDParams
        s = XDDParams().scales()
        ref = self._hand_scales()
        assert abs(s.t0 - ref["t0"]) / ref["t0"] < 1e-10

    def test_t0_numerical_value(self):
        """At default params: t0 ≈ 1.934e-6 s."""
        from diffsim.xdd.params import XDDParams
        s = XDDParams().scales()
        ref = self._hand_scales()
        # hand: (1e-7)^2 / (2e-7 * 0.025852) ≈ 1.934e-6
        assert abs(s.t0 - ref["t0"]) / ref["t0"] < 1e-6

    def test_lambda2_to_6sigfig(self):
        from diffsim.xdd.params import XDDParams
        s = XDDParams().scales()
        ref = self._hand_scales()
        assert abs(s.lambda2 - ref["lambda2"]) / ref["lambda2"] < 1e-6

    def test_gamma0_to_6sigfig(self):
        from diffsim.xdd.params import XDDParams
        s = XDDParams().scales()
        ref = self._hand_scales()
        assert abs(s.gamma0 - ref["gamma0"]) / ref["gamma0"] < 1e-6

    def test_J0_has_correct_units_structure(self):
        """J0 = U0 * x0 * q — sanity check via independent formula."""
        from diffsim.xdd.params import XDDParams
        s = XDDParams().scales()
        ref = self._hand_scales()
        assert abs(s.J0 - ref["J0"]) / ref["J0"] < 1e-10

    def test_U0_consistent(self):
        from diffsim.xdd.params import XDDParams
        s = XDDParams().scales()
        ref = self._hand_scales()
        assert abs(s.U0 - ref["U0"]) / ref["U0"] < 1e-10

    def test_eps_m_consistent(self):
        from diffsim.xdd.params import XDDParams
        s = XDDParams().scales()
        ref = self._hand_scales()
        assert abs(s.eps_m - ref["eps_m"]) / ref["eps_m"] < 1e-10


# ──────────────────────────────────────────────────────────────────────────────
# Gate 2: tau_split
# ──────────────────────────────────────────────────────────────────────────────
class TestTauSplit:
    """Gate 2: radiative/non-radiative lifetime splits from q_r."""

    def test_q_r_eq_1_tau_r_equals_tau_x(self):
        from diffsim.xdd.params import XDDParams
        p = XDDParams(q_r_donor=1.0, tau_x_donor=1e-9)
        assert p.tau_r_donor == pytest.approx(1e-9, rel=1e-12)

    def test_q_r_eq_1_tau_nr_is_inf(self):
        from diffsim.xdd.params import XDDParams
        p = XDDParams(q_r_donor=1.0, tau_x_donor=1e-9)
        assert math.isinf(p.tau_nr_donor)

    def test_q_r_eq_1_tau_nr_acceptor_is_inf(self):
        from diffsim.xdd.params import XDDParams
        p = XDDParams(q_r_acceptor=1.0, tau_x_acceptor=1e-9)
        assert math.isinf(p.tau_nr_acceptor)

    def test_q_r_0p25_harmonic_identity_donor(self):
        """1/tau_r + 1/tau_nr == 1/tau_x to rel 1e-12."""
        from diffsim.xdd.params import XDDParams
        tau_x = 2e-9
        p = XDDParams(q_r_donor=0.25, tau_x_donor=tau_x)
        lhs = 1.0 / p.tau_r_donor + 1.0 / p.tau_nr_donor
        rhs = 1.0 / tau_x
        assert abs(lhs - rhs) / rhs < 1e-12

    def test_q_r_0p25_harmonic_identity_acceptor(self):
        from diffsim.xdd.params import XDDParams
        tau_x = 3e-9
        p = XDDParams(q_r_acceptor=0.25, tau_x_acceptor=tau_x)
        lhs = 1.0 / p.tau_r_acceptor + 1.0 / p.tau_nr_acceptor
        rhs = 1.0 / tau_x
        assert abs(lhs - rhs) / rhs < 1e-12

    def test_q_r_0p5_tau_r_double_tau_x(self):
        """q_r=0.5 -> tau_r = tau_x/0.5 = 2*tau_x."""
        from diffsim.xdd.params import XDDParams
        tau_x = 1e-9
        p = XDDParams(q_r_donor=0.5, tau_x_donor=tau_x)
        assert p.tau_r_donor == pytest.approx(2 * tau_x, rel=1e-12)


# ──────────────────────────────────────────────────────────────────────────────
# Gate 3: from_cpu_config
# ──────────────────────────────────────────────────────────────────────────────
class TestFromCpuConfig:
    """Gate 3: parse real config.txt; ≥10 fields matched; unknowns in unrecognized."""

    @pytest.fixture(scope="class")
    @classmethod
    def params(cls):
        from diffsim.xdd.params import XDDParams
        return XDDParams.from_cpu_config(_CONFIG)

    def test_mu_n(self, params):
        assert params.mu_n == pytest.approx(2e-7, rel=1e-9)

    def test_eps_A(self, params):
        assert params.eps_A == pytest.approx(3.9, rel=1e-9)

    def test_E_g(self, params):
        assert params.E_g == pytest.approx(1.1, rel=1e-9)

    def test_a(self, params):
        assert params.a == pytest.approx(1.8e-9, rel=1e-9)

    def test_interface_thk(self, params):
        assert params.interface_thk == pytest.approx(2e-9, rel=1e-9)

    def test_height(self, params):
        assert params.height == pytest.approx(100e-9, rel=1e-9)

    def test_Voc(self, params):
        assert params.Voc == pytest.approx(0.0, abs=1e-12)

    def test_solver_strategy(self, params):
        assert params.solver_strategy == 3

    def test_time_adaptivity_strategy(self, params):
        assert params.time_adaptivity_strategy == 4

    def test_dt(self, params):
        assert params.dt == pytest.approx(1e-11, rel=1e-9)

    def test_recombination_strategy(self, params):
        assert params.recombination_strategy == 1

    def test_refine_lvl(self, params):
        assert params.refine_lvl == 8

    def test_boundary_refine_lvl(self, params):
        assert params.boundary_refine_lvl == 8

    def test_dt_max(self, params):
        assert params.dt_max == pytest.approx(5e-6, rel=1e-9)

    def test_no_crash_on_solver_keys(self):
        """Parsing must not raise even though config has ksp_atol etc."""
        from diffsim.xdd.params import XDDParams
        p = XDDParams.from_cpu_config(_CONFIG)
        # ksp_* keys are inside solver_options blocks — they go to unrecognized
        assert hasattr(p, "unrecognized") and isinstance(p.unrecognized, dict)

    def test_if_acceptor_excitons_true(self, params):
        assert params.if_acceptor_excitons is True

    def test_morphology_file_parsed(self, params):
        assert "morph_bilayer_2D" in params.morphology_file


# ──────────────────────────────────────────────────────────────────────────────
# Gate 4: read_spec + ret_factor
# ──────────────────────────────────────────────────────────────────────────────
class TestReadSpec:
    """Gate 4a: read_spec returns valid numpy arrays with monotone grids."""

    def test_solar_shape(self):
        from diffsim.xdd.params import read_spec
        wl, val = read_spec(_SOLAR)
        assert len(wl) == 2002
        assert len(val) == 2002

    def test_solar_wavelengths_monotone(self):
        from diffsim.xdd.params import read_spec
        wl, _ = read_spec(_SOLAR)
        assert np.all(np.diff(wl) > 0)

    def test_acceptor_y6_shape(self):
        from diffsim.xdd.params import read_spec
        wl, val = read_spec(_ACC_Y6)
        assert len(wl) == 901

    def test_donor_pl_pm6_shape(self):
        from diffsim.xdd.params import read_spec
        wl, val = read_spec(_PL_PM6)
        assert len(wl) == 7420

    def test_returns_numpy_arrays(self):
        from diffsim.xdd.params import read_spec
        wl, val = read_spec(_SOLAR)
        assert isinstance(wl, np.ndarray)
        assert isinstance(val, np.ndarray)

    def test_first_wavelength_solar(self):
        from diffsim.xdd.params import read_spec
        wl, _ = read_spec(_SOLAR)
        assert wl[0] == pytest.approx(280.0, abs=0.1)


class TestRetFactor:
    """Gate 4b: ret_factor_from_spectra bands for PM6:Y6 and P3HT:PCBM."""

    def test_pm6_y6_ret_above_0p9(self):
        """PM6:Y6 PL/acceptor overlap should be > 0.9 (strong RET)."""
        from diffsim.xdd.params import read_spec, ret_factor_from_spectra
        wl_pl, pl = read_spec(_PL_PM6)
        wl_acc, acc = read_spec(_ACC_Y6)
        ret = ret_factor_from_spectra((wl_pl, pl), (wl_acc, acc))
        assert ret > 0.9, f"PM6:Y6 RET expected > 0.9, got {ret:.4f}"

    def test_p3ht_pcbm_ret_below_0p2(self):
        """P3HT:PCBM PL/acceptor overlap should be < 0.2 (weak RET)."""
        from diffsim.xdd.params import read_spec, ret_factor_from_spectra
        wl_pl, pl = read_spec(_PL_P3HT)
        wl_acc, acc = read_spec(_ACC_PCBM)
        ret = ret_factor_from_spectra((wl_pl, pl), (wl_acc, acc))
        assert ret < 0.2, f"P3HT:PCBM RET expected < 0.2, got {ret:.4f}"

    def test_ret_factor_in_0_1(self):
        from diffsim.xdd.params import read_spec, ret_factor_from_spectra
        wl_pl, pl = read_spec(_PL_PM6)
        wl_acc, acc = read_spec(_ACC_Y6)
        ret = ret_factor_from_spectra((wl_pl, pl), (wl_acc, acc))
        assert 0.0 <= ret <= 1.0


# ──────────────────────────────────────────────────────────────────────────────
# Gate 5: generation_from_spectra
# ──────────────────────────────────────────────────────────────────────────────
class TestGenerationFromSpectra:
    """Gate 5: G_total in [1e27, 1e29] m^-3 s^-1; split fractions sum to 1."""

    @pytest.fixture(scope="class")
    @classmethod
    def gen_result(cls):
        from diffsim.xdd.params import read_spec, generation_from_spectra
        solar   = read_spec(_SOLAR)
        blend   = read_spec(_BLEND_PM6Y6)
        donor   = read_spec(_DON_PM6)
        acceptor = read_spec(_ACC_Y6)
        height = 100e-9
        return generation_from_spectra(solar, blend, donor, acceptor, height)

    def test_g_total_in_physical_band(self, gen_result):
        G_total, G_donor, G_acceptor = gen_result
        assert 1e27 <= G_total <= 1e29, (
            f"G_total={G_total:.3e} not in [1e27, 1e29] m^-3 s^-1")

    def test_split_fractions_sum_to_1(self, gen_result):
        G_total, G_donor, G_acceptor = gen_result
        assert abs((G_donor + G_acceptor) - G_total) / G_total < 1e-10

    def test_g_donor_positive(self, gen_result):
        _, G_donor, _ = gen_result
        assert G_donor > 0

    def test_g_acceptor_positive(self, gen_result):
        _, _, G_acceptor = gen_result
        assert G_acceptor > 0


# ──────────────────────────────────────────────────────────────────────────────
# Structural tests: replace() / frozen idiom
# ──────────────────────────────────────────────────────────────────────────────
class TestReplace:
    def test_replace_returns_new_instance(self):
        from diffsim.xdd.params import XDDParams
        p = XDDParams()
        p2 = p.replace(mu_n=1e-7)
        assert p2 is not p
        assert p2.mu_n == pytest.approx(1e-7)
        assert p.mu_n == pytest.approx(2e-7)   # original unchanged

    def test_replace_does_not_mutate_original(self):
        from diffsim.xdd.params import XDDParams
        p = XDDParams()
        _ = p.replace(T=350.0)
        assert p.T == 300.0
