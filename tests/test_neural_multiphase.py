"""Tests for BasisMultiEnergy (Task 1, M6: gauge-anchored learnable bulk energy).

TDD structure:
  - test_basis_reduces_to_fh_at_zero_coeffs  (brief Step 1 / parity)
  - test_basis_correction_is_gauge_orthogonal (brief Step 1 / gauge)
  - test_basis_derivs_complex_step           (added per task description)
"""
import numpy as np
import pytest

from diffsim.adjoint import BasisMultiEnergy, FHMultiEnergy


def _chiN(M=2):
    chi = np.zeros((M + 1, M + 1))
    chi[0, 1] = chi[1, 0] = 2.5
    chi[0, 2] = chi[2, 0] = 1.0
    chi[1, 2] = chi[2, 1] = 0.8
    return chi, np.ones(M + 1)


# --------------------------------------------------------------------------
# Test 1: FH parity when all basis coefficients are zero
# --------------------------------------------------------------------------
def test_basis_reduces_to_fh_at_zero_coeffs():
    chi, N = _chiN()
    be = BasisMultiEnergy(chi, N, degrees=(2, 3))          # coeffs default 0
    fh = FHMultiEnergy(chi, N)
    phis = [np.full(5, 0.30), np.full(5, 0.32)]
    for a, b in zip(be.mu(phis), fh.mu(phis)):
        assert np.allclose(a, b)
    assert any(nm.startswith("basis_") for nm in be.param_names)


# --------------------------------------------------------------------------
# Test 2: gauge orthogonality — correction is L2-orthogonal to {1, phi_i}
# --------------------------------------------------------------------------
def test_basis_correction_is_gauge_orthogonal():
    chi, N = _chiN()
    be = BasisMultiEnergy(chi, N, degrees=(2, 3),
                          coeffs={"basis_0_2": 0.7, "basis_0_3": -0.4})
    # <corr_mu_0, 1> and <corr_mu_0, phi_0> over the domain must be ~0
    r0, r1 = be.gauge_residual(species=0)
    assert abs(r0) < 1e-10 and abs(r1) < 1e-10


# --------------------------------------------------------------------------
# Test 3: complex-step verification of dmu_dphi and dmu_dparam
# --------------------------------------------------------------------------
def _cstep_mu(be, phis, j):
    """Complex-step dmu/d(phis[j]) — returns list of M arrays."""
    h = 1e-30
    p = [x.astype(complex) for x in phis]
    p[j] = p[j] + 1j * h
    return [mu_i.imag / h for mu_i in be.mu(p)]


def test_basis_derivs_complex_step():
    """dmu_dphi diagonal correction and dmu_dparam('basis_i_k') vs complex step.

    Non-trivial coefficients so the correction path is exercised.
    Uses 1e-30 imaginary perturbation (same idiom as test_energy_derivs_complex_step).
    """
    chi, N = _chiN(M=2)
    coeffs = {"basis_0_2": 0.5, "basis_0_3": -0.3,
              "basis_1_2": 0.2, "basis_1_3": 0.1}
    be = BasisMultiEnergy(chi, N, degrees=(2, 3), coeffs=coeffs)
    rng = np.random.default_rng(42)
    # compositions well inside dom=(0.05, 0.95)
    phis = [0.20 + 0.15 * rng.random(8), 0.25 + 0.15 * rng.random(8)]

    M = 2
    H = be.dmu_dphi(phis)

    # --- dmu_dphi: diagonal ---
    for j in range(M):
        cs_col = _cstep_mu(be, phis, j)
        for i in range(M):
            assert np.allclose(H[i][j], cs_col[i], atol=1e-9, rtol=1e-7), \
                f"dmu_dphi[{i}][{j}] mismatch"

    # --- dmu_dparam for basis names ---
    h = 1e-30
    for name in [n for n in be.param_names if n.startswith("basis_")]:
        _, si, sk = name.split("_")
        i_sp, k_deg = int(si), int(sk)
        # complex-step w.r.t. this gamma: perturb gamma[(i_sp, k_deg)] by 1j*h
        old = be.gamma[(i_sp, k_deg)]
        be.gamma[(i_sp, k_deg)] = old + 1j * h
        p_c = [x.astype(complex) for x in phis]
        mu_c = be.mu(p_c)
        cs_dparam = [mu_c[i].imag / h for i in range(M)]
        be.gamma[(i_sp, k_deg)] = old

        an_dparam = be.dmu_dparam(phis, name)
        for i in range(M):
            assert np.allclose(an_dparam[i], cs_dparam[i], atol=1e-9, rtol=1e-7), \
                f"dmu_dparam({name})[{i}] mismatch"
